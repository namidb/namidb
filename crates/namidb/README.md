# namidb

The public façade crate for [NamiDB](https://github.com/namidb/namidb),
the graph database that lives in your bucket.

This is the **stable umbrella API**. It re-exports the types you reach
for most often from
[`namidb-core`](../namidb-core/),
[`namidb-storage`](../namidb-storage/),
[`namidb-graph`](../namidb-graph/) and
[`namidb-query`](../namidb-query/), so downstream code can depend on a
single `namidb = "0.1"` line in `Cargo.toml`.

If you're embedding NamiDB into a Rust application, **start here**.

## Example

```rust
use std::sync::Arc;

use namidb::core::id::NamespaceId;
use namidb::query::{execute, lower, parse, Params};
use namidb::storage::{NamespacePaths, WriterSession};
use object_store::{memory::InMemory, ObjectStore};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths   = NamespacePaths::new("tenants", NamespaceId::new("demo")?);
    let mut writer = WriterSession::open(store, paths).await?;

    // ... upsert nodes / edges, then commit_batch + flush ...

    let snap = writer.snapshot();
    let q    = parse("MATCH (a:Person) RETURN count(*) AS n")?;
    let plan = lower(&q)?;
    let rows = execute(&plan, &snap, &Params::new()).await?;

    println!("{rows:?}");
    Ok(())
}
```

For Python, see [`namidb-py`](../namidb-py/) (`pip install namidb`).

## License

[Business Source License 1.1](../../LICENSE), © LESAI, Corp.
