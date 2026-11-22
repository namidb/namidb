//! Hub-shaped synthetic graph for planner reproduction (`graph-shape`).
//!
//! Reproduces a shape that stresses anchor choice and label predicates:
//!
//! - a few `:Device` hub nodes, one of them holding most of the
//!   `ON_DEVICE` edges (default 1.7M of 2.4M nodes point at it);
//! - a small `:Call` label (default 2,958 nodes);
//! - the bulk of the graph in `:Message`, `:Contact` and `:Location`;
//! - every non-device node has exactly one `(x)-[:ON_DEVICE]->(:Device)`.
//!
//! `:Message` and `:Call` nodes carry an `embedding` list of floats (the
//! shape a Python client writes), so a store written by an older engine keeps
//! those inside the node's overflow JSON and one written by the current
//! engine keeps them in the binary property column.
//!
//! Everything is a pure function of `(config, seed)`. The generator can write
//! straight into a namespace (`--uri`) or emit NDJSON files (`--ndjson-dir`)
//! for `bench/load_ndjson.py`, which loads them through any released Python
//! client, so the same graph can be written by an older engine release to
//! compare on-disk layouts.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use namidb_core::id::NodeId;
use namidb_core::Value;
use namidb_query::{RuntimeValue, StatsCatalog};
use namidb_storage::{EdgeWriteRecord, NodeWriteRecord, ReaderSession, SstCache, WriterSession};
use serde_json::json;

/// Generator knobs. Defaults match the reported production shape.
#[derive(Debug, Clone)]
pub struct ShapeConfig {
    /// Total node count, devices included.
    pub nodes: u64,
    /// Number of `:Device` hubs.
    pub hubs: u64,
    /// `ON_DEVICE` in-degree of the largest hub (device 0). The other hubs
    /// share the remaining non-device nodes evenly.
    pub hub_max_degree: u64,
    /// Number of `:Call` nodes.
    pub calls: u64,
    /// Length of the `embedding` list on `:Message` / `:Call` (0 = none).
    pub embedding_dim: usize,
    /// Rows staged between two memtable flushes in direct-write mode.
    pub flush_every: u64,
    pub seed: u64,
}

impl Default for ShapeConfig {
    fn default() -> Self {
        Self {
            nodes: 2_400_000,
            hubs: 4,
            hub_max_degree: 1_700_000,
            calls: 2_958,
            embedding_dim: 64,
            flush_every: 250_000,
            seed: 42,
        }
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Kind of the node at global index `i`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Device,
    Call,
    Message,
    Contact,
    Location,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Device => "Device",
            Kind::Call => "Call",
            Kind::Message => "Message",
            Kind::Contact => "Contact",
            Kind::Location => "Location",
        }
    }
}

impl ShapeConfig {
    pub fn kind(&self, i: u64) -> Kind {
        if i < self.hubs {
            Kind::Device
        } else if i < self.hubs + self.calls {
            Kind::Call
        } else {
            match (i - self.hubs - self.calls) % 10 {
                0..=5 => Kind::Message,
                6 | 7 => Kind::Contact,
                _ => Kind::Location,
            }
        }
    }

    /// Stable UUID for the node at global index `i`.
    pub fn uuid(&self, i: u64) -> uuid::Uuid {
        let hi = splitmix(self.seed ^ i.wrapping_mul(0xA24B_AED4_963E_E407));
        let lo = splitmix(hi ^ i);
        // Version-4 / RFC-4122 variant bits so every client accepts it.
        let mut bytes = ((u128::from(hi) << 64) | u128::from(lo)).to_be_bytes();
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        uuid::Uuid::from_bytes(bytes)
    }

    /// The device (hub index) node `i` points at. Device 0 receives
    /// `hub_max_degree` of the non-device nodes, the rest share evenly.
    pub fn device_of(&self, i: u64) -> u64 {
        let non_devices = self.nodes.saturating_sub(self.hubs).max(1);
        let h = splitmix(self.seed.wrapping_add(i).wrapping_mul(31)) % non_devices;
        if self.hubs <= 1 || h < self.hub_max_degree.min(non_devices) {
            0
        } else {
            1 + h % (self.hubs - 1)
        }
    }

    fn embedding(&self, i: u64) -> Vec<f64> {
        let mut state = splitmix(self.seed ^ 0xE3B0_C442 ^ i);
        (0..self.embedding_dim)
            .map(|_| {
                state = splitmix(state);
                // An f32 widened to f64, as a Python float list from a model
                // would be: ~18 significant digits once printed as JSON.
                ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) as f64
            })
            .collect()
    }

    /// Property map for node `i`.
    pub fn properties(&self, i: u64) -> BTreeMap<String, Value> {
        let mut p = BTreeMap::new();
        let h = splitmix(self.seed ^ (i << 1));
        let ts = 1_700_000_000_000_i64 + (h % 31_536_000_000) as i64;
        match self.kind(i) {
            Kind::Device => {
                p.insert("name".into(), Value::Str(format!("device-{i}")));
                p.insert(
                    "serial".into(),
                    Value::Str(format!("SN{:012}", h % 1_000_000_000_000)),
                );
            }
            Kind::Call => {
                p.insert("ts".into(), Value::I64(ts));
                p.insert("duration".into(), Value::I64((h % 3_600) as i64));
                let dir = if h.is_multiple_of(2) { "in" } else { "out" };
                p.insert("direction".into(), Value::Str(dir.into()));
                p.insert(
                    "number".into(),
                    Value::Str(format!("+1555{:07}", h % 10_000_000)),
                );
            }
            Kind::Message => {
                p.insert("ts".into(), Value::I64(ts));
                p.insert(
                    "body".into(),
                    Value::Str(format!(
                        "message {i} lorem ipsum dolor sit amet consectetur {}",
                        h % 100_000
                    )),
                );
            }
            Kind::Contact => {
                p.insert(
                    "name".into(),
                    Value::Str(format!("contact-{}", h % 1_000_000)),
                );
                p.insert(
                    "number".into(),
                    Value::Str(format!("+1555{:07}", h % 10_000_000)),
                );
            }
            Kind::Location => {
                p.insert("ts".into(), Value::I64(ts));
                p.insert(
                    "lat".into(),
                    Value::F64((h % 180_000) as f64 / 1000.0 - 90.0),
                );
                p.insert(
                    "lon".into(),
                    Value::F64((h % 360_000) as f64 / 1000.0 - 180.0),
                );
            }
        }
        if self.embedding_dim > 0 && matches!(self.kind(i), Kind::Message | Kind::Call) {
            let list = self.embedding(i).into_iter().map(Value::F64).collect();
            p.insert("embedding".into(), Value::List(list));
        }
        p
    }
}

fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => json!(b),
        Value::I64(i) => json!(i),
        Value::F64(f) => json!(f),
        Value::Str(s) => json!(s),
        Value::List(items) => serde_json::Value::Array(items.iter().map(value_to_json).collect()),
        other => json!(format!("{other:?}")),
    }
}

/// Write the graph as NDJSON: one `nodes_<Label>.ndjson` per label (each
/// line `{"id": uuid, ...properties}`) and `edges_ON_DEVICE.ndjson`
/// (`{"src": uuid, "dst": uuid}`), for `bench/load_ndjson.py`.
pub fn write_ndjson(cfg: &ShapeConfig, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut files: BTreeMap<&'static str, BufWriter<File>> = BTreeMap::new();
    for kind in [
        Kind::Device,
        Kind::Call,
        Kind::Message,
        Kind::Contact,
        Kind::Location,
    ] {
        let f = File::create(dir.join(format!("nodes_{}.ndjson", kind.label())))?;
        files.insert(kind.label(), BufWriter::new(f));
    }
    let mut edges = BufWriter::new(File::create(dir.join("edges_ON_DEVICE.ndjson"))?);
    for i in 0..cfg.nodes {
        let kind = cfg.kind(i);
        let mut obj = serde_json::Map::new();
        obj.insert("id".into(), json!(cfg.uuid(i).to_string()));
        for (k, v) in cfg.properties(i) {
            obj.insert(k, value_to_json(&v));
        }
        let w = files.get_mut(kind.label()).expect("every kind has a file");
        serde_json::to_writer(&mut *w, &serde_json::Value::Object(obj))?;
        w.write_all(b"\n")?;
        if kind != Kind::Device {
            let dst = cfg.uuid(cfg.device_of(i));
            serde_json::to_writer(
                &mut edges,
                &json!({"src": cfg.uuid(i).to_string(), "dst": dst.to_string()}),
            )?;
            edges.write_all(b"\n")?;
        }
    }
    for (_, mut f) in files {
        f.flush()?;
    }
    edges.flush()?;
    Ok(())
}

/// Write the graph straight into the namespace at `uri` with the current
/// engine, flushing every `flush_every` rows and compacting at the end, the
/// way an embedded client's auto-compaction leaves a bulk-loaded store.
pub async fn write_direct(cfg: &ShapeConfig, uri: &str) -> Result<()> {
    let (store, paths) = namidb_storage::parse_uri(uri).context("parse uri")?;
    let mut writer = WriterSession::open(store, paths)
        .await
        .context("open writer")?;
    let started = Instant::now();
    let mut staged = 0u64;
    for i in 0..cfg.nodes {
        let kind = cfg.kind(i);
        let record = NodeWriteRecord {
            properties: cfg.properties(i),
            schema_version: 0,
            ..Default::default()
        };
        writer.upsert_node(kind.label(), NodeId::from_uuid(cfg.uuid(i)), &record)?;
        if kind != Kind::Device {
            writer.upsert_edge(
                "ON_DEVICE",
                NodeId::from_uuid(cfg.uuid(i)),
                NodeId::from_uuid(cfg.uuid(cfg.device_of(i))),
                &EdgeWriteRecord::default(),
            )?;
        }
        staged += 1;
        if staged >= cfg.flush_every {
            flush(&mut writer).await?;
            staged = 0;
            eprintln!(
                "graph-shape: {} / {} nodes written ({:.0}s)",
                i + 1,
                cfg.nodes,
                started.elapsed().as_secs_f64()
            );
        }
    }
    flush(&mut writer).await?;
    loop {
        let basis = writer.compaction_basis();
        if !basis.needs_compaction() {
            break;
        }
        let schema = basis.schema().clone();
        let outcome = writer.compact_l0(&schema).await.context("compact")?;
        eprintln!(
            "graph-shape: compaction removed {} SSTs, wrote {} ({:.0}s)",
            outcome.source_ssts_removed,
            outcome.new_ssts_written,
            started.elapsed().as_secs_f64()
        );
        if outcome.new_ssts_written == 0 && outcome.source_ssts_removed == 0 {
            break;
        }
    }
    Ok(())
}

async fn flush(writer: &mut WriterSession) -> Result<()> {
    writer.commit_batch().await.context("commit")?;
    let schema = writer.snapshot().manifest().manifest.schema.clone();
    writer.flush(schema).await.context("flush")?;
    Ok(())
}

/// Peak resident set size of this process, from `/proc/self/status`.
pub fn vm_hwm_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let rest = line.strip_prefix("VmHWM:")?;
        let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kb * 1024)
    })
}

fn json_to_runtime(v: &serde_json::Value) -> RuntimeValue {
    match v {
        serde_json::Value::Null => RuntimeValue::Null,
        serde_json::Value::Bool(b) => RuntimeValue::Bool(*b),
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) => RuntimeValue::Integer(i),
            None => RuntimeValue::Float(n.as_f64().unwrap_or(f64::NAN)),
        },
        serde_json::Value::String(s) => RuntimeValue::String(s.clone()),
        serde_json::Value::Array(a) => RuntimeValue::List(a.iter().map(json_to_runtime).collect()),
        serde_json::Value::Object(o) => RuntimeValue::Map(
            o.iter()
                .map(|(k, v)| (k.clone(), json_to_runtime(v)))
                .collect(),
        ),
    }
}

/// Run one read query read-only against `uri` and print a JSON report:
/// the plan, row count, per-run wall time and the process peak RSS.
/// `$hubs` is bound to the ids of every `:Device` node and `$hub` to the
/// largest one, so the anchored queries need no hand-written ids.
pub async fn run_query(
    uri: &str,
    query: &str,
    params_json: Option<&str>,
    repeat: usize,
    cache_bytes: usize,
    timeout_ms: Option<u64>,
) -> Result<serde_json::Value> {
    let (store, paths) = namidb_storage::parse_uri(uri).context("parse uri")?;
    let reader = ReaderSession::open(store, paths, Some(SstCache::new(cache_bytes)))
        .await
        .context("open reader")?;
    let owned = reader.snapshot();
    let snap = owned.borrow();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let mut params = namidb_query::Params::new();
    if let Some(text) = params_json {
        let v: serde_json::Value = serde_json::from_str(text).context("params JSON")?;
        if let serde_json::Value::Object(o) = v {
            for (k, v) in o {
                params.insert(k, json_to_runtime(&v));
            }
        }
    }
    if query.contains("$hub") {
        let parsed = namidb_query::parse("MATCH (d:Device) RETURN id(d) AS id ORDER BY d.name")
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let plan = namidb_query::plan(&parsed, &catalog).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let rows = namidb_query::execute(&plan, &snap, &params).await?;
        let ids: Vec<RuntimeValue> = rows
            .iter()
            .filter_map(|r| r.bindings.get("id").cloned())
            .collect();
        if let Some(first) = ids.first() {
            params.insert("hub".into(), first.clone());
        }
        params.insert("hubs".into(), RuntimeValue::List(ids));
    }

    let parsed = namidb_query::parse(query).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let plan = namidb_query::plan(&parsed, &catalog).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let explain = namidb_query::explain_plan_lines(&parsed, &plan, &snap, &catalog);
    let rss_before = vm_hwm_bytes();
    let mut runs_ms = Vec::new();
    let mut row_count = 0usize;
    let mut first_row = serde_json::Value::Null;
    let mut error = None;
    for _ in 0..repeat.max(1) {
        let deadline = timeout_ms.map(|ms| Instant::now() + std::time::Duration::from_millis(ms));
        let started = Instant::now();
        match namidb_query::execute_with_limits(&plan, &snap, &params, deadline, None).await {
            Ok(rows) => {
                runs_ms.push(started.elapsed().as_secs_f64() * 1e3);
                row_count = rows.len();
                if let Some(r) = rows.first() {
                    first_row = json!(r
                        .bindings
                        .iter()
                        .map(|(k, v)| (k.clone(), format!("{v:?}")))
                        .collect::<BTreeMap<_, _>>());
                }
            }
            Err(e) => {
                runs_ms.push(started.elapsed().as_secs_f64() * 1e3);
                error = Some(e.to_string());
                break;
            }
        }
    }
    if namidb_core::profile::enabled() {
        eprintln!("{}", namidb_core::profile::dump_table());
    }
    Ok(json!({
        "query": query,
        "plan": explain,
        "rows": row_count,
        "first_row": first_row,
        "runs_ms": runs_ms,
        "error": error,
        "vm_hwm_bytes_before": rss_before,
        "vm_hwm_bytes": vm_hwm_bytes(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_is_deterministic_and_skewed() {
        let cfg = ShapeConfig {
            nodes: 20_000,
            hub_max_degree: 15_000,
            calls: 100,
            ..ShapeConfig::default()
        };
        assert_eq!(cfg.uuid(7), cfg.uuid(7));
        assert_ne!(cfg.uuid(7), cfg.uuid(8));
        assert_eq!(cfg.kind(0), Kind::Device);
        assert_eq!(cfg.kind(4), Kind::Call);
        assert_eq!(cfg.kind(104), Kind::Message);
        let mut degree = [0u64; 4];
        for i in cfg.hubs..cfg.nodes {
            degree[cfg.device_of(i) as usize] += 1;
        }
        // Device 0 is the hub; the others share the rest.
        assert!(degree[0] > 14_000 && degree[0] < 16_000, "{degree:?}");
        assert!(degree[1..].iter().all(|d| *d > 1_000), "{degree:?}");
    }
}
