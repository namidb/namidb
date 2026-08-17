//! Item 40 (25 TB readiness), storage half: a flush that fails on LOCAL
//! persistence (unwritable/full spool disk) must be a clean, recoverable
//! error — writer not poisoned, memtable restored so reads keep serving
//! every acknowledged row, later commits still accepted — and a retry after
//! the disk clears must succeed losslessly.
//!
//! This file must stay a single-test integration binary: it mutates the
//! process-global `NAMIDB_SPOOL_DIR`, which every concurrent flush in the
//! same process would observe. As its own test binary it owns the process.

use std::collections::BTreeMap;

use namidb_core::id::NodeId;
use namidb_core::schema::{DataType, LabelDef, PropertyDef, SchemaBuilder};
use namidb_core::value::Value;
use namidb_storage::{NodeWriteRecord, WriterSession};

fn schema() -> namidb_core::schema::Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build()
}

fn upsert(writer: &mut WriterSession, name: &str) {
    let mut props: BTreeMap<String, Value> = BTreeMap::new();
    props.insert("name".into(), Value::Str(name.into()));
    writer
        .upsert_node(
            "Person",
            NodeId::new(),
            &NodeWriteRecord {
                properties: props,
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
}

async fn person_found(writer: &WriterSession, name: &str) -> bool {
    writer
        .snapshot()
        .lookup_node_by_property("Person", "name", name)
        .await
        .unwrap()
        .is_some()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spool_failure_is_recoverable_and_lossless() {
    let (store, paths) = namidb_storage::parse_uri("memory://spool-fail").unwrap();
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    for i in 0..200 {
        upsert(&mut writer, &format!("persona-{i:03}"));
    }
    writer.commit_batch().await.unwrap();
    assert!(person_found(&writer, "persona-007").await);
    let staged_bytes = writer.memtable_bytes();
    assert!(staged_bytes > 0);

    // Break the spool: every flush build now fails opening scratch files.
    std::env::set_var("NAMIDB_SPOOL_DIR", "/nonexistent/namidb-spool-fail-test");
    let err = writer
        .flush(schema())
        .await
        .expect_err("flush must fail with the spool unwritable");
    assert!(
        err.is_local_persistence(),
        "spool failure must classify as local persistence, got: {err}"
    );
    assert!(
        !writer.is_poisoned(),
        "a local-persistence flush failure must not poison the session"
    );
    assert_eq!(
        writer.memtable_bytes(),
        staged_bytes,
        "the frozen memtable must be restored intact after the failed flush"
    );
    assert!(
        person_found(&writer, "persona-007").await,
        "reads must keep serving acknowledged rows after the failed flush"
    );

    // The writer still accepts and commits new work while unable to flush.
    upsert(&mut writer, "persona-tardia");
    writer.commit_batch().await.unwrap();
    assert!(person_found(&writer, "persona-tardia").await);

    // Clear the disk condition: the retry must succeed and lose nothing.
    let spool = tempfile::tempdir().unwrap();
    std::env::set_var("NAMIDB_SPOOL_DIR", spool.path());
    let outcome = writer.flush(schema()).await.expect("retry flush succeeds");
    assert!(outcome.ssts_written >= 1, "the retry must publish SSTs");
    assert_eq!(writer.memtable_bytes(), 0, "the memtable must drain");
    for name in ["persona-007", "persona-199", "persona-tardia"] {
        assert!(
            person_found(&writer, name).await,
            "{name} must survive into the SSTs"
        );
    }
}
