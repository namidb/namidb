# namidb

The workspace façade crate for [NamiDB](https://github.com/namidb/namidb),
the graph database that lives in your bucket.

This crate is not currently published to crates.io and its low-level Rust
surface is not covered by NamiDB's released-artifact compatibility guarantee.
It is the intended future umbrella API for curated types from
[`namidb-core`](../namidb-core/),
[`namidb-storage`](../namidb-storage/),
[`namidb-graph`](../namidb-graph/) and
[`namidb-query`](../namidb-query/). Until that façade is published, Rust
embedders should use an explicit git/path revision and expect implementation
types to evolve between patch releases.

## Example

```rust
use namidb::{DataType, LabelDef, PropertyDef, Schema};

fn main() -> namidb::Result<()> {
    let schema = Schema::builder()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false)?],
        })?
        .build();

    assert!(schema.label("Person").is_some());
    Ok(())
}
```

For Python, see [`namidb-py`](../namidb-py/) (`pip install namidb`).

## License

[Business Source License 1.1](../../LICENSE), © NamiDB, Inc.
