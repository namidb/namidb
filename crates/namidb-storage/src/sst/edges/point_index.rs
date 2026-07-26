//! Exact-edge point-sidecar value encoding.
//!
//! The surrounding B+tree maps the direction-specific composite
//! `(key_id, partner_id)` to one value. Keeping the winner LSN, tombstone bit
//! and complete property map here lets bound-endpoint MERGE avoid opening the
//! graph-sized CSR and its Arrow property streams.

use std::collections::BTreeMap;

use bytes::{BufMut, Bytes, BytesMut};
use namidb_core::Value;

use crate::error::{Error, Result};

const MAGIC: &[u8; 4] = b"NEP1";
const HEADER_SIZE: usize = 4 + 8 + 1 + 4 + 4;
const FLAG_TOMBSTONE: u8 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct EdgePointValue {
    pub lsn: u64,
    pub tombstone: bool,
    pub properties: BTreeMap<String, Value>,
}

/// Encode one exact-edge value. `properties_payload` is the canonical JSON
/// encoding of `BTreeMap<String, Value>` produced by the edge writer.
pub fn encode(lsn: u64, tombstone: bool, properties_payload: &[u8]) -> Result<Bytes> {
    if tombstone && !properties_payload.is_empty() {
        return Err(Error::invariant(
            "edge-point tombstone cannot carry properties",
        ));
    }
    let payload_len = u32::try_from(properties_payload.len())
        .map_err(|_| Error::invariant("edge-point property payload exceeds 4 GiB"))?;
    let flags = u8::from(tombstone) * FLAG_TOMBSTONE;
    let checksum = value_checksum(lsn, flags, payload_len, properties_payload);
    let mut out = BytesMut::with_capacity(HEADER_SIZE + properties_payload.len());
    out.extend_from_slice(MAGIC);
    out.put_u64_le(lsn);
    out.put_u8(flags);
    out.put_u32_le(payload_len);
    out.put_u32_le(checksum);
    out.extend_from_slice(properties_payload);
    Ok(out.freeze())
}

/// Decode and validate one exact-edge value. Existence-only callers pass
/// `materialize_properties = false`; length and checksum are still verified,
/// but the JSON map is not allocated or parsed.
pub fn decode(bytes: &[u8], materialize_properties: bool) -> Result<EdgePointValue> {
    if bytes.len() < HEADER_SIZE || &bytes[..4] != MAGIC {
        return Err(Error::invariant("invalid edge-point value header"));
    }
    let lsn = u64::from_le_bytes(
        bytes[4..12]
            .try_into()
            .expect("edge-point LSN slice length"),
    );
    let flags = bytes[12];
    if flags & !FLAG_TOMBSTONE != 0 {
        return Err(Error::invariant(format!(
            "edge-point value has unknown flags 0x{flags:02x}"
        )));
    }
    let tombstone = flags & FLAG_TOMBSTONE != 0;
    let payload_len = u32::from_le_bytes(
        bytes[13..17]
            .try_into()
            .expect("edge-point payload length slice"),
    ) as usize;
    let expected_crc =
        u32::from_le_bytes(bytes[17..21].try_into().expect("edge-point checksum slice"));
    let payload = bytes
        .get(HEADER_SIZE..)
        .filter(|payload| payload.len() == payload_len)
        .ok_or_else(|| Error::invariant("edge-point property payload length mismatch"))?;
    if value_checksum(lsn, flags, payload_len as u32, payload) != expected_crc {
        return Err(Error::invariant("edge-point value checksum mismatch"));
    }
    if tombstone && !payload.is_empty() {
        return Err(Error::invariant(
            "edge-point tombstone carries a property payload",
        ));
    }
    let properties = if materialize_properties && !tombstone {
        serde_json::from_slice(payload)
            .map_err(|error| Error::invariant(format!("edge-point properties decode: {error}")))?
    } else {
        BTreeMap::new()
    };
    Ok(EdgePointValue {
        lsn,
        tombstone,
        properties,
    })
}

fn value_checksum(lsn: u64, flags: u8, payload_len: u32, payload: &[u8]) -> u32 {
    let mut checksum = crc32fast::Hasher::new();
    checksum.update(&lsn.to_le_bytes());
    checksum.update(&[flags]);
    checksum.update(&payload_len.to_le_bytes());
    checksum.update(payload);
    checksum.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_value_round_trips_and_existence_skips_json_materialization() {
        let properties = BTreeMap::from([
            ("code".into(), Value::Str("A-1".into())),
            ("weight".into(), Value::I64(7)),
        ]);
        let payload = serde_json::to_vec(&properties).unwrap();
        let encoded = encode(42, false, &payload).unwrap();
        assert_eq!(
            decode(&encoded, true).unwrap(),
            EdgePointValue {
                lsn: 42,
                tombstone: false,
                properties,
            }
        );
        assert!(decode(&encoded, false).unwrap().properties.is_empty());
    }

    #[test]
    fn point_value_rejects_corruption_and_tombstone_payloads() {
        assert!(encode(1, true, b"{}").is_err());
        let mut encoded = encode(1, false, b"{}").unwrap().to_vec();
        *encoded.last_mut().unwrap() ^= 0xFF;
        assert!(decode(&encoded, false).is_err());

        let mut lsn_corrupt = encode(1, false, b"{}").unwrap().to_vec();
        lsn_corrupt[4] ^= 0x01;
        assert!(decode(&lsn_corrupt, false).is_err());

        let mut flag_corrupt = encode(1, true, b"").unwrap().to_vec();
        flag_corrupt[12] ^= FLAG_TOMBSTONE;
        assert!(decode(&flag_corrupt, false).is_err());
    }
}
