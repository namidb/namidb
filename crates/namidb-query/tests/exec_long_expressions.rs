//! Long generated expressions (`a OR b OR ...` with thousands of terms) go
//! through every recursive pass: lowering, the optimiser, evaluation and
//! drop. Up to the parser's tree-depth cap they must answer on the 16 MiB
//! query threads the server and the Python client run statements on; past
//! it they fail with a parse error instead of overflowing the stack, which
//! aborts the whole process.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;

use namidb_query::{execute, lower, optimize, parse, Params, RuntimeValue, StatsCatalog};

/// The query thread stack of `namidb-server` and the Python client.
const QUERY_THREAD_STACK: usize = 16 * 1024 * 1024;

async fn run(writer: &WriterSession, cypher: &str) -> Result<Vec<namidb_query::Row>, String> {
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
    let query = parse(cypher).map_err(|errors| format!("{errors:?}"))?;
    let plan = optimize(lower(&query).map_err(|e| e.to_string())?, &catalog);
    execute(&plan, &snap, &Params::new())
        .await
        .map_err(|e| e.to_string())
}

fn on_query_thread(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(QUERY_THREAD_STACK)
        .spawn(f)
        .unwrap()
        .join()
        .expect("query thread panicked");
}

#[test]
fn long_or_and_plus_chains_answer_or_fail_cleanly() {
    on_query_thread(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let paths = NamespacePaths::new("t", NamespaceId::new("long").unwrap());
            let mut writer = WriterSession::open(Arc::new(InMemory::new()), paths)
                .await
                .unwrap();
            for i in 0..50_i64 {
                let mut p = BTreeMap::new();
                p.insert("k".into(), CoreValue::I64(i));
                writer
                    .upsert_node(
                        "Item",
                        NodeId::new(),
                        &NodeWriteRecord {
                            properties: p,
                            schema_version: 1,
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
            writer.commit_batch().await.unwrap();

            let or_chain = |n: usize| {
                format!(
                    "MATCH (n:Item) WHERE {} RETURN count(n) AS c",
                    (0..n)
                        .map(|i| format!("n.k = {i}"))
                        .collect::<Vec<_>>()
                        .join(" OR ")
                )
            };
            // Close to the cap: answers.
            let rows = run(&writer, &or_chain(2_000)).await.unwrap();
            assert_eq!(rows[0].get("c"), Some(&RuntimeValue::Integer(50)));
            let plus = format!("RETURN {} AS x", vec!["1"; 2_000].join(" + "));
            let rows = run(&writer, &plus).await.unwrap();
            assert_eq!(rows[0].get("x"), Some(&RuntimeValue::Integer(2_000)));
            let and = format!(
                "MATCH (n:Item) WHERE {} RETURN count(n) AS c",
                (0..2_000)
                    .map(|i| format!("n.k <> {}", 1_000 + i))
                    .collect::<Vec<_>>()
                    .join(" AND ")
            );
            let rows = run(&writer, &and).await.unwrap();
            assert_eq!(rows[0].get("c"), Some(&RuntimeValue::Integer(50)));

            // Far past it: a parse error, and the process is still here.
            let err = run(&writer, &or_chain(20_000)).await.unwrap_err();
            assert!(err.contains("too deep"), "{err}");
            let plus = format!("RETURN {} AS x", vec!["1"; 50_000].join(" + "));
            let err = run(&writer, &plus).await.unwrap_err();
            assert!(err.contains("too deep"), "{err}");
        });
    });
}
