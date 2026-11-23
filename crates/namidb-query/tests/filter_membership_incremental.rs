//! The membership set of a search filter is kept across flushes and
//! compactions: a flush patches it with the rows of the new SSTs, a
//! compaction of rows it already reflects carries it, anything it cannot
//! show to be equivalent recomputes it. Every answer must equal a full read
//! of the filter's columns, whatever the order of creates, value flips,
//! relabels, deletes, flushes, L0 merges, deeper merges (which drop
//! tombstones) and restarts, and wherever the reads fall between them.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_query::{
    filter_membership_ids_for_tests, filter_membership_paths, filter_membership_sst_key_reads,
    RuntimeValue,
};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

const LABEL: &str = "Doc";
const OTHER: &str = "Other";
const TIERS: usize = 5;

/// The path and key-read counters are process-wide: the tests that read
/// them run one at a time.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn schema() -> Schema {
    let props = || {
        vec![
            PropertyDef::new("vigente", DataType::Bool, true).unwrap(),
            PropertyDef::new("tier", DataType::Utf8, true).unwrap(),
            PropertyDef::new("n", DataType::Int64, true).unwrap(),
        ]
    };
    SchemaBuilder::new()
        .label(LabelDef {
            name: LABEL.into(),
            properties: props(),
        })
        .unwrap()
        .label(LabelDef {
            name: OTHER.into(),
            properties: props(),
        })
        .unwrap()
        .build()
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

/// Deterministic pseudo-random choices.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

#[derive(Clone)]
struct Doc {
    label: &'static str,
    vigente: bool,
    tier: usize,
    n: i64,
}

impl Doc {
    fn record(&self) -> NodeWriteRecord {
        let mut properties = BTreeMap::new();
        properties.insert("vigente".into(), CoreValue::Bool(self.vigente));
        properties.insert("tier".into(), CoreValue::Str(format!("t{}", self.tier)));
        properties.insert("n".into(), CoreValue::I64(self.n));
        NodeWriteRecord {
            properties,
            schema_version: 1,
            ..Default::default()
        }
    }
}

fn filter(entries: &[(&str, RuntimeValue)]) -> RuntimeValue {
    RuntimeValue::Map(
        entries
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect(),
    )
}

fn filters() -> Vec<RuntimeValue> {
    vec![
        filter(&[("vigente", RuntimeValue::Bool(true))]),
        filter(&[("tier", RuntimeValue::String("t2".into()))]),
        filter(&[
            ("vigente", RuntimeValue::Bool(false)),
            ("tier", RuntimeValue::String("t0".into())),
        ]),
    ]
}

/// The incremental answer of every filter equals the full read.
async fn assert_exact(writer: &WriterSession, docs: &BTreeMap<NodeId, Doc>, context: &str) {
    let snapshot = writer.snapshot();
    for filter in filters() {
        let served = filter_membership_ids_for_tests(&snapshot, LABEL, &filter, false)
            .await
            .unwrap();
        let exact = filter_membership_ids_for_tests(&snapshot, LABEL, &filter, true)
            .await
            .unwrap();
        assert_eq!(served, exact, "{context}: {filter:?}");
        // And the model agrees with the full read.
        let RuntimeValue::Map(map) = &filter else {
            unreachable!()
        };
        let model = docs
            .iter()
            .filter(|(_, doc)| doc.label == LABEL)
            .filter(|(_, doc)| {
                map.iter().all(|(key, value)| match (key.as_str(), value) {
                    ("vigente", RuntimeValue::Bool(flag)) => doc.vigente == *flag,
                    ("tier", RuntimeValue::String(tier)) => format!("t{}", doc.tier) == *tier,
                    _ => unreachable!(),
                })
            })
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        assert_eq!(exact.as_deref(), Some(model.as_slice()), "{context}: model");
    }
}

async fn randomized(seed: u64) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let name = format!("membership-{seed}");
    let mut writer = WriterSession::open(store.clone(), paths(&name))
        .await
        .unwrap();
    let mut rng = Rng(seed);
    let mut docs: BTreeMap<NodeId, Doc> = BTreeMap::new();
    let mut ordinal = 0i64;
    let mut new_doc = |rng: &mut Rng| {
        ordinal += 1;
        Doc {
            label: LABEL,
            vigente: rng.below(2) == 0,
            tier: rng.below(TIERS),
            n: ordinal,
        }
    };
    for _ in 0..60 {
        let id = NodeId::new();
        let doc = new_doc(&mut rng);
        writer.upsert_node(doc.label, id, &doc.record()).unwrap();
        docs.insert(id, doc);
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    assert_eq!(docs.len(), 60);
    let mut pending = false;
    for step in 0..160 {
        let context = format!("seed {seed} step {step}");
        match rng.below(12) {
            0 | 1 => {
                for _ in 0..1 + rng.below(4) {
                    let id = NodeId::new();
                    let doc = new_doc(&mut rng);
                    writer.upsert_node(doc.label, id, &doc.record()).unwrap();
                    docs.insert(id, doc);
                }
                pending = true;
            }
            2 | 3 => {
                // Flip a filter value of an existing node (flushed or not).
                let ids = docs.keys().copied().collect::<Vec<_>>();
                let id = ids[rng.below(ids.len())];
                let doc = docs.get_mut(&id).unwrap();
                if rng.below(2) == 0 {
                    doc.vigente = !doc.vigente;
                } else {
                    doc.tier = (doc.tier + 1 + rng.below(TIERS - 1)) % TIERS;
                }
                writer.upsert_node(doc.label, id, &doc.record()).unwrap();
                pending = true;
            }
            4 => {
                // Relabel, either way.
                let ids = docs.keys().copied().collect::<Vec<_>>();
                let id = ids[rng.below(ids.len())];
                let doc = docs.get_mut(&id).unwrap();
                doc.label = if doc.label == LABEL { OTHER } else { LABEL };
                writer.upsert_node(doc.label, id, &doc.record()).unwrap();
                pending = true;
            }
            5 => {
                let ids = docs.keys().copied().collect::<Vec<_>>();
                if ids.len() > 20 {
                    let id = ids[rng.below(ids.len())];
                    let doc = docs.remove(&id).unwrap();
                    writer.tombstone_node(doc.label, id).unwrap();
                    pending = true;
                }
            }
            6 => {
                writer.commit_batch().await.unwrap();
                pending = false;
                writer.flush(schema()).await.unwrap();
            }
            7 => {
                if pending {
                    writer.commit_batch().await.unwrap();
                    pending = false;
                }
                writer.compact_l0(&schema()).await.unwrap();
            }
            8 => {
                // A flush and a compaction with no read in between.
                writer.commit_batch().await.unwrap();
                pending = false;
                writer.flush(schema()).await.unwrap();
                writer.compact_l0(&schema()).await.unwrap();
            }
            9 => {
                // Restart: a new writer session over the same store.
                writer.commit_batch().await.unwrap();
                pending = false;
                writer.flush(schema()).await.unwrap();
                drop(writer);
                writer = WriterSession::open(store.clone(), paths(&name))
                    .await
                    .unwrap();
            }
            _ => {
                if pending {
                    writer.commit_batch().await.unwrap();
                    pending = false;
                }
                assert_exact(&writer, &docs, &context).await;
            }
        }
    }
    writer.commit_batch().await.unwrap();
    assert_exact(&writer, &docs, &format!("seed {seed} end")).await;
}

#[tokio::test]
async fn randomized_writes_flushes_and_compactions_keep_the_set_exact() {
    let _serial = SERIAL.lock().await;
    let (_, _, patched_before, recomputed_before) = filter_membership_paths();
    for seed in [1, 7, 42, 1234, 99991] {
        randomized(seed).await;
    }
    let (_, _, patched, recomputed) = filter_membership_paths();
    assert!(patched > patched_before, "no read patched a set");
    assert!(recomputed > recomputed_before, "no read recomputed a set");
}

/// Deterministic sequences for each path, so each one is exercised even if
/// the random walk misses it.
#[tokio::test]
async fn each_flush_and_compaction_takes_the_expected_path() {
    let _serial = SERIAL.lock().await;
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let mut writer = WriterSession::open(store.clone(), paths("membership-paths"))
        .await
        .unwrap();
    let mut docs = BTreeMap::new();
    for n in 0..40 {
        let id = NodeId::new();
        let doc = Doc {
            label: LABEL,
            vigente: n % 2 == 0,
            tier: n as usize % TIERS,
            n,
        };
        writer.upsert_node(LABEL, id, &doc.record()).unwrap();
        docs.insert(id, doc);
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    assert_exact(&writer, &docs, "seeded").await;

    // A flush of a flip and a delete: patched.
    let ids = docs.keys().copied().collect::<Vec<_>>();
    let flipped = ids[3];
    let doc = docs.get_mut(&flipped).unwrap();
    doc.vigente = !doc.vigente;
    writer.upsert_node(LABEL, flipped, &doc.record()).unwrap();
    let deleted = ids[4];
    docs.remove(&deleted);
    writer.tombstone_node(LABEL, deleted).unwrap();
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    let before = filter_membership_paths();
    assert_exact(&writer, &docs, "after a flush").await;
    let after = filter_membership_paths();
    assert!(after.2 > before.2, "a flush of new rows patches");

    // A compaction of rows the set already reflects: carried, without
    // reading the rewritten SSTs' keys.
    writer.compact_l0(&schema()).await.unwrap();
    let before = filter_membership_paths();
    let reads_before = filter_membership_sst_key_reads();
    assert_exact(&writer, &docs, "after a seen compaction").await;
    let after = filter_membership_paths();
    assert!(after.1 > before.1, "a compaction of seen rows carries");
    assert_eq!(
        filter_membership_sst_key_reads(),
        reads_before,
        "a rewrite of seen rows is carried without reading keys"
    );

    // Two flushes (deletes) and a compaction with no read in between: the
    // merge may drop a member's tombstone with its row, so members inside
    // the merged SSTs' key ranges that no new SST holds are re-evaluated.
    // Two members of the `vigente: true` set.
    let members = docs
        .iter()
        .filter(|(_, doc)| doc.vigente)
        .map(|(id, _)| *id)
        .take(2)
        .collect::<Vec<_>>();
    for deleted in members {
        docs.remove(&deleted);
        writer.tombstone_node(LABEL, deleted).unwrap();
        writer.commit_batch().await.unwrap();
        writer.flush(schema()).await.unwrap();
    }
    writer.compact_l0(&schema()).await.unwrap();
    let before = filter_membership_paths();
    assert_exact(&writer, &docs, "after an unseen compaction").await;
    let after = filter_membership_paths();
    assert!(after.2 > before.2, "an unseen compaction patches");

    // A restart is a new writer session: recomputed.
    writer.commit_batch().await.unwrap();
    drop(writer);
    let writer = WriterSession::open(store, paths("membership-paths"))
        .await
        .unwrap();
    let before = filter_membership_paths();
    assert_exact(&writer, &docs, "after a restart").await;
    let after = filter_membership_paths();
    assert!(after.3 > before.3, "a new writer session recomputes");
}

/// A member deleted after its set was built, then a merge that drops the
/// tombstone together with the row it shadows, with no read in between:
/// the new corpus holds neither, so nothing in the merge's output names the
/// node. The set must not be carried as is; the answer excludes the node.
#[tokio::test]
async fn a_merge_that_drops_a_newer_tombstone_is_not_carried_blindly() {
    let _serial = SERIAL.lock().await;
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let mut writer = WriterSession::open(store, paths("membership-tombstone-drop"))
        .await
        .unwrap();
    let mut docs = BTreeMap::new();
    for n in 0..60 {
        let id = NodeId::new();
        let doc = Doc {
            label: LABEL,
            vigente: n % 2 == 0,
            tier: n as usize % TIERS,
            n,
        };
        writer.upsert_node(LABEL, id, &doc.record()).unwrap();
        docs.insert(id, doc);
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();
    // The set is built at this watermark.
    assert_exact(&writer, &docs, "built").await;

    let member = docs
        .iter()
        .find(|(_, doc)| doc.vigente)
        .map(|(id, _)| *id)
        .unwrap();
    docs.remove(&member);
    writer.tombstone_node(LABEL, member).unwrap();
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    // The only level below: the merge drops the tombstone and the row.
    writer.compact_l0(&schema()).await.unwrap();

    let reads_before = filter_membership_sst_key_reads();
    let snapshot = writer.snapshot();
    let served = filter_membership_ids_for_tests(
        &snapshot,
        LABEL,
        &filter(&[("vigente", RuntimeValue::Bool(true))]),
        false,
    )
    .await
    .unwrap()
    .unwrap();
    assert!(!served.contains(&member), "a dropped member is not served");
    assert!(
        filter_membership_sst_key_reads() > reads_before,
        "a merge that consumed a newer row reads the new SSTs' keys"
    );
    assert_exact(&writer, &docs, "after the tombstone was dropped").await;
}
