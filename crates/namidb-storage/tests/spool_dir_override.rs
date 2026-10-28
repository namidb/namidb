//! `set_default_spool_dir` reaches every spool: the flush/compaction spool
//! files and the text and vector index build configs.
//!
//! This file must stay a single-test integration binary: it sets the
//! process-global spool override and `NAMIDB_SPOOL_DIR`, which every
//! concurrent flush in the same process would observe.

use std::collections::BTreeMap;

use namidb_core::id::NodeId;
use namidb_core::schema::SchemaBuilder;
use namidb_core::value::Value;
use namidb_storage::{set_default_spool_dir, NodeWriteRecord, WriterSession};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn override_wins_over_env_in_every_spool() {
    // An env dir that does not exist: any spool still reading only the env
    // fails, so a passing flush proves it used the override.
    std::env::set_var("NAMIDB_SPOOL_DIR", "/nonexistent/namidb-spool-override");
    let spool = tempfile::tempdir().unwrap();
    set_default_spool_dir(Some(spool.path().to_path_buf()));

    #[cfg(feature = "text-index")]
    {
        use namidb_storage::sst::text::v4::{TextV4BuildOptions, TextV4ExternalBuildConfig};
        let text = namidb_storage::sst::text::ExternalTextIndexBuildOptions::from_env().unwrap();
        assert_eq!(text.spool_directory.as_deref(), Some(spool.path()));
        let text_v4 = TextV4ExternalBuildConfig::from_env(TextV4BuildOptions::default()).unwrap();
        assert_eq!(text_v4.spool_directory.as_deref(), Some(spool.path()));
    }
    #[cfg(feature = "vector-index")]
    {
        let vector =
            namidb_storage::sst::vector::v5::external::VectorV5ExternalBuildConfig::from_env()
                .unwrap();
        assert_eq!(vector.scratch_dir, spool.path());
    }

    let (store, paths) = namidb_storage::parse_uri("memory://spool-override").unwrap();
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let mut props: BTreeMap<String, Value> = BTreeMap::new();
    props.insert("name".into(), Value::Str("a".into()));
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
    writer.commit_batch().await.unwrap();
    let outcome = writer
        .flush(SchemaBuilder::new().build())
        .await
        .expect("flush spools under the override, not the missing env dir");
    assert!(outcome.ssts_written >= 1);
}
