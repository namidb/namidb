//! Helper: build the inverse partner SST input from a forward-sorted list of
//! edges.
//!
//! The frozen memtable produces edges sorted by `(edge_type, src, dst)`. The
//! forward partner SST consumes that order directly. The inverse partner SST
//! needs the same edges resorted by `(dst, src)`. This module owns the
//! transposition so the writer code stays simple.

use crate::sst::edges::writer::EdgeRecord;

/// Transpose forward-direction records into inverse-direction records.
///
/// Input records are expected to be sorted by `(key_id, partner_id)` —
/// i.e. the way the writer for the **forward** partner needs them. Output
/// records are sorted by `(partner_of_input, key_of_input)`, which is the
/// order the writer for the **inverse** partner needs.
///
/// The transposition only swaps `key_id` and `partner_id` per record; LSN,
/// tombstone and overflow are copied through.
pub fn transpose_forward_to_inverse(forward: &[EdgeRecord]) -> Vec<EdgeRecord> {
    let mut out: Vec<EdgeRecord> = forward
        .iter()
        .map(|r| EdgeRecord {
            key_id: r.partner_id,
            partner_id: r.key_id,
            lsn: r.lsn,
            tombstone: r.tombstone,
            declared_properties: r.declared_properties.clone(),
            overflow_json: r.overflow_json.clone(),
        })
        .collect();
    out.sort_by(|a, b| {
        a.key_id
            .cmp(&b.key_id)
            .then(a.partner_id.cmp(&b.partner_id))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sst::edges::writer::{EdgeSstWriter, EdgeSstWriterOptions};
    use crate::sst::edges::{reader::EdgeSstReader, EdgeDirection};

    fn key(top: u64, bot: u64) -> [u8; 16] {
        let mut k = [0u8; 16];
        k[..8].copy_from_slice(&top.to_le_bytes());
        k[8..].copy_from_slice(&bot.to_le_bytes());
        k
    }

    fn record(k: [u8; 16], p: [u8; 16], lsn: u64) -> EdgeRecord {
        EdgeRecord {
            key_id: k,
            partner_id: p,
            lsn,
            tombstone: false,
            declared_properties: vec![],
            overflow_json: None,
        }
    }

    #[test]
    fn transpose_swaps_and_resorts() {
        // Forward: src=1 has out-edges to [10, 20]; src=2 has out-edge to 10.
        let fwd = vec![
            record(key(1, 0), key(10, 0), 100),
            record(key(1, 0), key(20, 0), 101),
            record(key(2, 0), key(10, 0), 102),
        ];
        let inv = transpose_forward_to_inverse(&fwd);
        // Inverse: dst=10 has in-edges from [1, 2]; dst=20 from [1].
        assert_eq!(inv.len(), 3);
        // Sorted by key_id (= dst), then partner_id (= src).
        assert_eq!(inv[0].key_id, key(10, 0));
        assert_eq!(inv[0].partner_id, key(1, 0));
        assert_eq!(inv[0].lsn, 100);
        assert_eq!(inv[1].key_id, key(10, 0));
        assert_eq!(inv[1].partner_id, key(2, 0));
        assert_eq!(inv[1].lsn, 102);
        assert_eq!(inv[2].key_id, key(20, 0));
        assert_eq!(inv[2].partner_id, key(1, 0));
        assert_eq!(inv[2].lsn, 101);
    }

    #[test]
    fn full_round_trip_forward_and_inverse_partners() {
        // Build a graph; write forward + inverse SSTs from the same input;
        // assert every original edge can be reached from both directions.
        let fwd_edges = vec![
            record(key(1, 0), key(10, 0), 1),
            record(key(1, 0), key(20, 0), 2),
            record(key(2, 0), key(10, 0), 3),
            record(key(3, 0), key(30, 0), 4),
        ];

        let mut fwd_w = EdgeSstWriter::new(EdgeSstWriterOptions::new(
            EdgeDirection::Forward,
            "KNOWS",
            "P",
            "P",
        ));
        fwd_w.extend(fwd_edges.clone()).unwrap();
        let fwd = EdgeSstReader::open(fwd_w.finish().unwrap().body).unwrap();

        let inv_records = transpose_forward_to_inverse(&fwd_edges);
        let mut inv_w = EdgeSstWriter::new(EdgeSstWriterOptions::new(
            EdgeDirection::Inverse,
            "KNOWS",
            "P",
            "P",
        ));
        inv_w.extend(inv_records).unwrap();
        let inv = EdgeSstReader::open(inv_w.finish().unwrap().body).unwrap();

        // Direction is recorded.
        assert_eq!(fwd.direction(), EdgeDirection::Forward);
        assert_eq!(inv.direction(), EdgeDirection::Inverse);

        // For every original edge (src, dst): src in fwd → dst, dst in inv → src.
        for e in &fwd_edges {
            let f = fwd.lookup(&e.key_id).unwrap().unwrap();
            assert!(f.partners.contains(&e.partner_id));
            let i = inv.lookup(&e.partner_id).unwrap().unwrap();
            assert!(i.partners.contains(&e.key_id));
        }
    }
}
