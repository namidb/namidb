//! Cooperative cancellation for read queries (the query timeout).
//!
//! A read query's wall-clock deadline rides a tokio task-local scoped over
//! the whole execution. Because the deadline lives in this crate, the
//! CPU-bound SST decode and merge loops can probe it directly and abort a
//! single long-running operator mid-flight, not only at the query operator
//! boundaries above them (a giant single-SST decode used to run to
//! completion regardless of the deadline). The query layer scopes the
//! deadline through [`with_deadline`] and reads it through the same
//! task-local.
//!
//! When no deadline is in scope (writes, tests, the no-timeout server
//! config) every probe is a cheap task-local miss and the read path keeps
//! its baseline cost.
//!
//! The same task-local also carries an optional [`MemoryHook`]: the query
//! layer's per-query memory account. Storage reads report the rows they
//! materialise for the query through a [`RetainMeter`], and every
//! cooperative probe gives the hook a chance to stop the query (for
//! example because the process reached its memory ceiling). A hook that
//! trips surfaces as [`Error::MemoryLimit`] / [`Interrupt::MemoryLimit`].

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::error::{Error, Result};

/// What interrupted a cooperative probe: the wall clock, an operator (or
/// a client that went away), or the query's memory account.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interrupt {
    Timeout,
    Cancelled,
    /// The scoped [`MemoryHook`] refused more memory for this query.
    MemoryLimit,
}

/// A query's memory account, as seen from storage. The query layer
/// implements it and scopes it with [`with_memory_hook`]; storage never
/// needs to know the limits behind it.
pub trait MemoryHook: Send + Sync {
    /// The read is about to keep `bytes` more of materialised output for
    /// this query (rows it will hand back). Returns `false` when the query
    /// may not grow by that much; the read then stops with
    /// [`Error::MemoryLimit`].
    fn retain(&self, bytes: usize) -> bool;

    /// Called from every cooperative probe. Returns `false` when the query
    /// must stop for memory reasons that are not tied to one allocation (the
    /// process reached its ceiling). Must be cheap: rate-limit any real
    /// measurement internally.
    fn probe(&self) -> bool;
}

/// The guards a task runs under: an optional wall-clock deadline, an
/// optional operator-flippable cancel flag and an optional memory hook.
/// All are probed by the same ~hundred cooperative check sites, so an
/// admin cancel or a memory stop aborts exactly where a timeout would.
#[derive(Clone, Default)]
struct CancelCtx {
    deadline: Option<Instant>,
    cancel: Option<Arc<AtomicBool>>,
    memory: Option<Arc<dyn MemoryHook>>,
}

impl CancelCtx {
    /// The context in scope on this task, or the empty one.
    fn current() -> Self {
        CTX.try_with(Clone::clone).unwrap_or_default()
    }
}

tokio::task_local! {
    static CTX: CancelCtx;
}

/// Run `fut` with `deadline` scoped on the current task, so any read this
/// task performs can probe it. `None` runs `fut` unguarded (an
/// already-scoped cancel flag from an enclosing [`with_cancel_flag`] stays
/// visible). A `Some` deadline inherits any enclosing cancel flag rather
/// than masking it.
pub async fn with_deadline<F: Future>(deadline: Option<Instant>, fut: F) -> F::Output {
    match deadline {
        Some(at) => {
            let ctx = CancelCtx {
                deadline: Some(at),
                ..CancelCtx::current()
            };
            CTX.scope(ctx, fut).await
        }
        None => fut.await,
    }
}

/// Run `fut` with an operator cancel flag scoped on the current task. The
/// server registers one per in-flight query; flipping it makes every
/// cooperative probe under this scope return [`Error::Cancelled`]. Inherits
/// any enclosing deadline; an inner [`with_deadline`] inherits this flag.
pub async fn with_cancel_flag<F: Future>(flag: Arc<AtomicBool>, fut: F) -> F::Output {
    let ctx = CancelCtx {
        cancel: Some(flag),
        ..CancelCtx::current()
    };
    CTX.scope(ctx, fut).await
}

/// Run `fut` with `hook` as the query's memory account (see
/// [`MemoryHook`]). Inherits any enclosing deadline and cancel flag.
/// `None` runs `fut` with whatever is already in scope.
pub async fn with_memory_hook<F: Future>(hook: Option<Arc<dyn MemoryHook>>, fut: F) -> F::Output {
    match hook {
        Some(hook) => {
            let ctx = CancelCtx {
                memory: Some(hook),
                ..CancelCtx::current()
            };
            CTX.scope(ctx, fut).await
        }
        None => fut.await,
    }
}

/// The memory hook in scope, if any — for re-scoping onto a spawned task.
pub fn current_memory_hook() -> Option<Arc<dyn MemoryHook>> {
    CTX.try_with(|ctx| ctx.memory.clone()).ok().flatten()
}

/// The operator cancel flag in scope, if any — for re-scoping onto a
/// spawned task (task-locals do not cross `tokio::spawn`).
pub fn current_cancel_flag() -> Option<Arc<AtomicBool>> {
    CTX.try_with(|ctx| ctx.cancel.clone()).ok().flatten()
}

/// The active interrupt, if any. Operator cancel wins over the clock so the
/// surfaced error names the actual cause; the memory hook is asked last.
#[inline]
pub fn interrupted() -> Option<Interrupt> {
    CTX.try_with(|ctx| {
        if ctx
            .cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            Some(Interrupt::Cancelled)
        } else if ctx.deadline.is_some_and(|at| Instant::now() >= at) {
            Some(Interrupt::Timeout)
        } else if ctx.memory.as_ref().is_some_and(|hook| !hook.probe()) {
            Some(Interrupt::MemoryLimit)
        } else {
            None
        }
    })
    .ok()
    .flatten()
}

/// `true` when a deadline, an operator cancel flag or a memory hook is
/// scoped on this task, whether or not it has fired. Lets a caller pick an
/// interruptible (slightly costlier) algorithm only when something can
/// interrupt it.
#[inline]
pub fn guarded() -> bool {
    CTX.try_with(|ctx| ctx.deadline.is_some() || ctx.cancel.is_some() || ctx.memory.is_some())
        .unwrap_or(false)
}

/// `true` when a guard is in scope and has fired (deadline passed OR the
/// query was cancelled). The historical name predates operator cancel.
#[inline]
pub fn deadline_exceeded() -> bool {
    interrupted().is_some()
}

/// `Err(Error::Timeout)` / `Err(Error::Cancelled)` when a guard in scope has
/// fired, else `Ok(())`. Call it periodically inside a long CPU-bound loop
/// so the work aborts cooperatively instead of pinning a worker until it
/// returns.
#[inline]
pub fn check() -> Result<()> {
    match interrupted() {
        None => Ok(()),
        Some(Interrupt::Timeout) => Err(Error::Timeout),
        Some(Interrupt::Cancelled) => Err(Error::Cancelled),
        Some(Interrupt::MemoryLimit) => Err(Error::MemoryLimit),
    }
}

/// Bytes a [`RetainMeter`] accumulates before it reports them to the hook.
const RETAIN_FLUSH_BYTES: usize = 256 * 1024;

/// Amortised reporting of materialised read output to the scoped
/// [`MemoryHook`]. Create one per read, call [`Self::active`] before
/// building an estimate (it is `false` without a hook, so unmetered reads
/// pay nothing), and [`Self::add`] the estimate of every row kept. The
/// hook hears about the bytes every [`RETAIN_FLUSH_BYTES`], so a read
/// overshoots a limit by at most that much.
pub struct RetainMeter {
    hook: Option<Arc<dyn MemoryHook>>,
    pending: usize,
}

impl RetainMeter {
    /// A meter bound to the hook in scope on this task, if any.
    pub fn new() -> Self {
        Self {
            hook: current_memory_hook(),
            pending: 0,
        }
    }

    /// A meter that never reports, for reads whose rows the caller does not
    /// keep (a visitor consumes them one by one).
    pub fn disabled() -> Self {
        Self {
            hook: None,
            pending: 0,
        }
    }

    /// Whether a hook is listening.
    #[inline]
    pub fn active(&self) -> bool {
        self.hook.is_some()
    }

    /// Account `bytes` more of retained output.
    #[inline]
    pub fn add(&mut self, bytes: usize) -> Result<()> {
        let Some(hook) = &self.hook else {
            return Ok(());
        };
        self.pending = self.pending.saturating_add(bytes);
        if self.pending >= RETAIN_FLUSH_BYTES {
            let pending = std::mem::take(&mut self.pending);
            if !hook.retain(pending) {
                return Err(Error::MemoryLimit);
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for RetainMeter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainMeter")
            .field("active", &self.hook.is_some())
            .field("pending", &self.pending)
            .finish()
    }
}

impl Default for RetainMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RetainMeter {
    /// Report what is still pending. The result cannot be returned from
    /// here; a hook that refuses it remembers, and its next
    /// [`MemoryHook::probe`] stops the query.
    fn drop(&mut self) {
        if let Some(hook) = &self.hook {
            if self.pending > 0 {
                let _ = hook.retain(std::mem::take(&mut self.pending));
            }
        }
    }
}

/// How many rows a decode/merge loop processes between deadline probes.
/// Probing every row would put an `Instant::now()` on the hot path; a power
/// of two lets the compiler turn the modulus into a mask.
pub const CHECK_STRIDE: usize = 1024;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn guarded_reports_any_scoped_guard() {
        assert!(!guarded(), "no guard outside a scope");
        with_deadline(None, async { assert!(!guarded()) }).await;
        let later = Instant::now() + Duration::from_secs(60);
        with_deadline(Some(later), async {
            assert!(guarded());
            assert!(check().is_ok(), "an unexpired deadline has not fired");
        })
        .await;
        let flag = Arc::new(AtomicBool::new(false));
        with_cancel_flag(flag.clone(), async {
            assert!(guarded());
            assert!(check().is_ok());
            flag.store(true, Ordering::Relaxed);
            assert!(matches!(check(), Err(Error::Cancelled)));
        })
        .await;
    }
}
