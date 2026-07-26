//! Process-wide admission for authenticated Bolt message working sets.
//!
//! A framing limit bounds one connection, but without a shared budget many
//! authenticated connections can each buffer and decode a maximum-sized RUN at
//! the same time. This budget charges a conservative multiple of the wire body
//! while the message is buffered, decoded, converted to runtime parameters, or
//! retained in the RUN prefetch queue.

use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::{BoltError, Result};
use crate::message::DEFAULT_POST_AUTH_MESSAGE_BYTES;

/// Fixed decoded/container allowance charged to every data message.
pub const MESSAGE_MEMORY_BASE_BYTES: usize = 64 * 1024;

/// Estimated peak resident bytes charged per byte of authenticated Bolt body.
///
/// PackStream float vectors decode into enum slots, are converted into runtime
/// values, and can be cloned once by a materialising query operator. Sixteen is
/// deliberately more conservative than the codec's eight-times decoded-heap
/// ceiling and also covers the retained wire body.
pub const MESSAGE_MEMORY_BYTES_PER_WIRE_BYTE: usize = 16;

/// Framing retains the contiguous wire body with geometric `Vec` capacity and
/// a current chunk. Decode amplification is charged atomically only after the
/// terminating chunk has arrived.
const FRAMING_MEMORY_BYTES_PER_WIRE_BYTE: usize = 2;

/// Default shared budget when the server's total-RSS governor is disabled.
///
/// This admits one maximum-sized default message or several normal vector
/// batches, instead of multiplying the per-connection 64 MiB ceiling by the
/// connection limit.
pub const DEFAULT_MESSAGE_MEMORY_BUDGET_BYTES: usize = MESSAGE_MEMORY_BASE_BYTES
    + DEFAULT_POST_AUTH_MESSAGE_BYTES * MESSAGE_MEMORY_BYTES_PER_WIRE_BYTE;

/// Semaphore accounting granularity. Page-sized units keep the permit count
/// well below Tokio's `u32` acquire-many limit even for multi-GiB budgets.
const BUDGET_UNIT_BYTES: usize = 4 * 1024;

/// Smallest useful configured budget: one authenticated data byte upgraded to
/// its decoded working-set charge, rounded to semaphore granularity.
pub const MIN_MESSAGE_MEMORY_BUDGET_BYTES: usize =
    (MESSAGE_MEMORY_BASE_BYTES + MESSAGE_MEMORY_BYTES_PER_WIRE_BYTE).div_ceil(BUDGET_UNIT_BYTES)
        * BUDGET_UNIT_BYTES;

/// Maximum body retained without a data-budget lease while its request tag is
/// still unknown. Complete pressure-relief/control frames below this bound
/// bypass the data budget so COMMIT/ROLLBACK/RESET/PULL can always make
/// progress. Non-control messages acquire a lease before leaving the framer.
pub(crate) const CONTROL_FRAME_MAX_BYTES: usize = 4 * 1024;

/// Shared authenticated-message memory budget.
///
/// Clone the `Arc` into every [`crate::Session`] accepted by one server. Idle
/// authenticated sessions hold no permits; a lease exists only from the first
/// data-frame bytes until that message has been consumed by the session.
#[derive(Debug)]
pub struct MessageMemoryBudget {
    permits: Arc<Semaphore>,
    capacity_units: u32,
    capacity_bytes: usize,
}

impl MessageMemoryBudget {
    /// Construct a budget in estimated resident bytes.
    ///
    /// The value is rounded up to at least one accounting unit. Values above
    /// Tokio's acquire-many range are rejected rather than silently truncated.
    pub fn try_new(capacity_bytes: usize) -> Result<Self> {
        if capacity_bytes < MIN_MESSAGE_MEMORY_BUDGET_BYTES {
            return Err(BoltError::Protocol(format!(
                "Bolt message memory budget must be at least \
                     {MIN_MESSAGE_MEMORY_BUDGET_BYTES} bytes"
            )));
        }
        let units = capacity_bytes
            .saturating_add(BUDGET_UNIT_BYTES - 1)
            .checked_div(BUDGET_UNIT_BYTES)
            .unwrap_or(usize::MAX);
        let capacity_units = u32::try_from(units).map_err(|_| {
            BoltError::Protocol(format!(
                "Bolt message memory budget {capacity_bytes} exceeds the supported maximum"
            ))
        })?;
        Ok(Self {
            permits: Arc::new(Semaphore::new(capacity_units as usize)),
            capacity_units,
            capacity_bytes: (capacity_units as usize).saturating_mul(BUDGET_UNIT_BYTES),
        })
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Estimated resident charge for a body of `wire_bytes`.
    pub fn estimated_bytes_for_wire(wire_bytes: usize) -> usize {
        MESSAGE_MEMORY_BASE_BYTES
            .saturating_add(wire_bytes.saturating_mul(MESSAGE_MEMORY_BYTES_PER_WIRE_BYTE))
    }

    fn framing_bytes_for_wire(wire_bytes: usize) -> usize {
        MESSAGE_MEMORY_BASE_BYTES
            .saturating_add(wire_bytes.saturating_mul(FRAMING_MEMORY_BYTES_PER_WIRE_BYTE))
    }

    pub(crate) fn framing_units_for_wire(&self, wire_bytes: usize) -> Result<u32> {
        self.units_for_estimate(
            Self::framing_bytes_for_wire(wire_bytes),
            "Bolt framed message",
        )
    }

    pub(crate) fn decoded_units_for_wire(&self, wire_bytes: usize) -> Result<u32> {
        self.units_for_estimate(
            Self::estimated_bytes_for_wire(wire_bytes),
            "Bolt in-flight message memory",
        )
    }

    fn units_for_estimate(&self, estimated: usize, what: &'static str) -> Result<u32> {
        let units = estimated
            .saturating_add(BUDGET_UNIT_BYTES - 1)
            .checked_div(BUDGET_UNIT_BYTES)
            .unwrap_or(usize::MAX);
        let units = u32::try_from(units).unwrap_or(u32::MAX);
        if units > self.capacity_units {
            return Err(BoltError::TooLarge {
                what,
                len: estimated,
                max: self.capacity_bytes,
            });
        }
        Ok(units)
    }

    pub(crate) fn try_acquire_units(self: &Arc<Self>, units: u32) -> Result<OwnedSemaphorePermit> {
        if units > self.capacity_units {
            return Err(BoltError::TooLarge {
                what: "Bolt in-flight message memory",
                len: (units as usize).saturating_mul(BUDGET_UNIT_BYTES),
                max: self.capacity_bytes,
            });
        }
        match Arc::clone(&self.permits).try_acquire_many_owned(units) {
            Ok(permit) => Ok(permit),
            Err(tokio::sync::TryAcquireError::NoPermits) => Err(BoltError::MemoryBudgetExhausted {
                what: "Bolt in-flight message memory",
                requested: (units as usize).saturating_mul(BUDGET_UNIT_BYTES),
                available: self
                    .permits
                    .available_permits()
                    .saturating_mul(BUDGET_UNIT_BYTES),
                capacity: self.capacity_bytes,
            }),
            Err(tokio::sync::TryAcquireError::Closed) => Err(BoltError::Protocol(
                "Bolt message memory budget closed".into(),
            )),
        }
    }

    #[cfg(test)]
    pub(crate) fn available_bytes(&self) -> usize {
        self.permits
            .available_permits()
            .saturating_mul(BUDGET_UNIT_BYTES)
    }
}

/// Exact lease retained with one buffered/decoded/prefetched message.
#[derive(Debug)]
pub(crate) struct MessageMemoryLease {
    permit: OwnedSemaphorePermit,
}

impl MessageMemoryLease {
    pub(crate) fn new(permit: OwnedSemaphorePermit) -> Self {
        Self { permit }
    }

    pub(crate) fn units(&self) -> u32 {
        self.permit.num_permits() as u32
    }

    pub(crate) fn merge(&mut self, permit: OwnedSemaphorePermit) {
        self.permit.merge(permit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_requires_one_decodable_data_byte() {
        assert!(MessageMemoryBudget::try_new(MIN_MESSAGE_MEMORY_BUDGET_BYTES - 1).is_err());
        assert!(MessageMemoryBudget::try_new(MIN_MESSAGE_MEMORY_BUDGET_BYTES).is_ok());
    }

    #[test]
    fn default_budget_admits_exactly_one_maximum_default_message() {
        let budget =
            Arc::new(MessageMemoryBudget::try_new(DEFAULT_MESSAGE_MEMORY_BUDGET_BYTES).unwrap());
        assert_eq!(budget.capacity_bytes(), DEFAULT_MESSAGE_MEMORY_BUDGET_BYTES);
        let units = budget
            .decoded_units_for_wire(DEFAULT_POST_AUTH_MESSAGE_BYTES)
            .unwrap();
        let permit = budget.try_acquire_units(units).unwrap();
        assert_eq!(budget.available_bytes(), 0);
        drop(permit);
        assert!(matches!(
            budget.decoded_units_for_wire(DEFAULT_POST_AUTH_MESSAGE_BYTES + 1),
            Err(BoltError::TooLarge {
                what: "Bolt in-flight message memory",
                ..
            })
        ));
    }
}
