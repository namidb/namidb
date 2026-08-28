//! Rollback compatibility for manifests written by 2.0.5.
//!
//! Keep the local DTOs below frozen to the public 2.0.4 wire shape. This is
//! deliberately not a second copy of the current manifest type: additions to
//! the current type must remain unknown fields to this decoder.

use chrono::{DateTime, TimeZone, Utc};
use namidb_core::{LabelDictionary, Schema};
use namidb_storage::manifest::{
    EqualityIndexDescriptor, EqualityKeyEncoding, KindSpecificStats, LabelIndexDescriptor,
    NodeLocatorDescriptor, PagedPropertyIndexDescriptor, PerLabelPropertyStat, PropertyIndexFormat,
    SearchIndexBuildState, TextIndexDescriptor, UniquePropertyIndexDescriptor,
    VectorIndexDescriptor, VectorMetric, VectorQuantization,
};
use namidb_storage::{
    BloomDescriptor, DegreeHistogram, Epoch, Manifest, PropertyColumnStats, SstDescriptor, SstKind,
    SstLevel, WalSegmentDescriptor,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Exact top-level manifest field list published by 2.0.4.
#[derive(Debug, Serialize, Deserialize)]
struct Manifest204 {
    version: u64,
    epoch: Epoch,
    writer_id: Uuid,
    created_at: DateTime<Utc>,
    schema: Schema,
    #[serde(default)]
    ssts: Vec<SstDescriptor204>,
    #[serde(default)]
    wal_segments: Vec<WalSegmentDescriptor>,
    #[serde(default)]
    label_dict: LabelDictionary,
    #[serde(default)]
    vector_indexes: Vec<VectorIndexDescriptor>,
    #[serde(default)]
    text_indexes: Vec<TextIndexDescriptor>,
}

/// Exact SST descriptor field list published by 2.0.4.
#[derive(Debug, Serialize, Deserialize)]
struct SstDescriptor204 {
    id: Uuid,
    kind: SstKind,
    scope: String,
    level: SstLevel,
    path: String,
    size_bytes: u64,
    row_count: u64,
    created_at: DateTime<Utc>,
    #[serde(with = "serde_key16")]
    min_key: [u8; 16],
    #[serde(with = "serde_key16")]
    max_key: [u8; 16],
    min_lsn: u64,
    max_lsn: u64,
    schema_version_min: u64,
    schema_version_max: u64,
    #[serde(default)]
    property_stats: Vec<PropertyColumnStats>,
    kind_specific: KindSpecificStats,
    #[serde(default)]
    bloom: Option<BloomDescriptor>,
    #[serde(default)]
    unique_property_indices: Vec<UniquePropertyIndexDescriptor204>,
    #[serde(default)]
    equality_property_indices: Vec<EqualityIndexDescriptor204>,
    #[serde(default)]
    label_index: Option<LabelIndexDescriptor>,
    #[serde(default)]
    per_label_property_stats: Vec<PerLabelPropertyStat>,
}

#[derive(Debug, Serialize, Deserialize)]
struct UniquePropertyIndexDescriptor204 {
    property: String,
    path: String,
    size_bytes: u64,
    entry_count: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct EqualityIndexDescriptor204 {
    property: String,
    path: String,
    size_bytes: u64,
    distinct_values: u64,
}

fn descriptor(
    ordinal: u128,
    kind: SstKind,
    scope: &str,
    path: &str,
    kind_specific: KindSpecificStats,
) -> SstDescriptor {
    SstDescriptor {
        id: Uuid::from_u128(ordinal),
        kind,
        scope: scope.into(),
        level: SstLevel(1),
        path: path.into(),
        size_bytes: 4096 + ordinal as u64,
        row_count: 17,
        created_at: Utc.timestamp_opt(1_700_000_000, 0).single().unwrap(),
        min_key: [ordinal as u8; 16],
        max_key: [0xf0 | ordinal as u8; 16],
        min_lsn: 100,
        max_lsn: 200,
        schema_version_min: 7,
        schema_version_max: 7,
        property_stats: Vec::new(),
        kind_specific,
        bloom: None,
        unique_property_indices: Vec::new(),
        equality_property_indices: Vec::new(),
        composite_equality_indices: Vec::new(),
        label_index: None,
        node_locator: None,
        per_label_property_stats: Vec::new(),
    }
}

#[test]
fn manifest_205_round_trips_through_the_204_wire_prefix() {
    let node_body = "sst/level1/nodes-articles.parquet";
    let edge_body = "sst/level1/cita.ep.csr";
    let inverse_edge_body = "sst/level1/cita-inverse.csr";
    let vector_body = "sst/level1/articulo-embedding.vg";
    let text_body = "sst/level1/articulo-texto.ft";
    let legacy_unique = "sst/level1/nodes-articles-key.uidx.bin";
    let legacy_equality = "sst/level1/nodes-articles-vigente.eqidx.bin";

    let mut node = descriptor(
        1,
        SstKind::Nodes,
        "",
        node_body,
        KindSpecificStats::Nodes { tombstone_count: 2 },
    );
    node.unique_property_indices = vec![UniquePropertyIndexDescriptor {
        property: "key".into(),
        path: legacy_unique.into(),
        size_bytes: 800,
        entry_count: 17,
        format: PropertyIndexFormat::BincodeV0,
        paged: Some(PagedPropertyIndexDescriptor {
            path: "sst/level1/nodes-articles-key.pidx".into(),
            size_bytes: 512,
        }),
        paged_build_unsupported: false,
    }];
    node.equality_property_indices = vec![EqualityIndexDescriptor {
        property: "vigente".into(),
        path: legacy_equality.into(),
        size_bytes: 700,
        distinct_values: 2,
        key_encoding: EqualityKeyEncoding::ScalarV1,
        mixed_type_complete: true,
        format: PropertyIndexFormat::BincodeV0,
        paged: Some(PagedPropertyIndexDescriptor {
            path: "sst/level1/nodes-articles-vigente.pidx".into(),
            size_bytes: 384,
        }),
        paged_build_unsupported: false,
    }];
    node.label_index = Some(LabelIndexDescriptor {
        path: "sst/level1/nodes-articles.labels.bin".into(),
        size_bytes: 300,
        label_count: 1,
        posting_count: 17,
        format: PropertyIndexFormat::BincodeV0,
        per_label_counts: vec![(1, 17)],
    });
    node.node_locator = Some(NodeLocatorDescriptor {
        path: "sst/level1/nodes-articles.nloc".into(),
        size_bytes: 256,
        entry_count: 17,
        property_pages: None,
    });

    let edge_stats = || KindSpecificStats::Edges {
        key_count: 11,
        tombstone_count: 1,
        degree_histogram: Box::new(DegreeHistogram::empty()),
    };
    let edge = descriptor(2, SstKind::EdgesFwd, "CITA", edge_body, edge_stats());
    let inverse_edge = descriptor(
        3,
        SstKind::EdgesInv,
        "CITA",
        inverse_edge_body,
        edge_stats(),
    );
    let vector = descriptor(
        4,
        SstKind::VectorGraph,
        "articulo_embedding",
        vector_body,
        KindSpecificStats::VectorGraph {
            dim: 3,
            metric: "cosine".into(),
            point_count: 17,
            r: 8,
            l_build: 16,
            alpha: 1.2,
            entry_medoid: 4,
        },
    );
    let text = descriptor(
        5,
        SstKind::TextIndex,
        "articulo_texto",
        text_body,
        KindSpecificStats::TextIndex {
            doc_count: 17,
            term_count: 91,
            total_len: 240,
        },
    );

    let mut current = Manifest::empty(Epoch::ZERO, Uuid::from_u128(99));
    current.version = 345;
    current.created_at = Utc.timestamp_opt(1_700_000_001, 0).single().unwrap();
    current.ssts = vec![node, edge, inverse_edge, vector, text];
    current.wal_segments = vec![WalSegmentDescriptor {
        seq: 9,
        path: "wal/00000000000000000009.wal".into(),
        last_lsn: 211,
        xxh3: Some(0xfeed_beef),
    }];
    current.vector_indexes = vec![VectorIndexDescriptor {
        name: "articulo_embedding".into(),
        label: "Articulo".into(),
        property: "embedding".into(),
        dim: 3,
        metric: VectorMetric::Cosine,
        r: 8,
        l_build: 16,
        alpha: 1.2,
        quantization: VectorQuantization::None,
    }];
    current.text_indexes = vec![TextIndexDescriptor::new(
        "articulo_texto".into(),
        "Articulo".into(),
        vec!["titulo".into(), "texto".into()],
    )];
    current.search_index_builds = vec![
        SearchIndexBuildState {
            kind: SstKind::VectorGraph,
            name: "articulo_embedding".into(),
            catalog_signature: r#"{"dim":3,"metric":"cosine"}"#.into(),
            max_node_lsn: 200,
        },
        SearchIndexBuildState {
            kind: SstKind::TextIndex,
            name: "articulo_texto".into(),
            catalog_signature: r#"{"properties":["texto","titulo"]}"#.into(),
            max_node_lsn: 200,
        },
    ];

    let current_json = serde_json::to_value(&current).unwrap();
    let legacy: Manifest204 = serde_json::from_value(current_json.clone())
        .expect("a 2.0.4 manifest decoder must accept a complete 2.0.5 manifest");

    assert_eq!(legacy.version, current.version);
    assert_eq!(legacy.epoch, current.epoch);
    assert_eq!(legacy.writer_id, current.writer_id);
    assert_eq!(legacy.created_at, current.created_at);
    assert_eq!(legacy.schema, current.schema);
    assert_eq!(legacy.label_dict, current.label_dict);
    assert_eq!(legacy.wal_segments, current.wal_segments);
    assert_eq!(legacy.vector_indexes, current.vector_indexes);
    assert_eq!(legacy.text_indexes, current.text_indexes);

    let body_paths: Vec<_> = legacy
        .ssts
        .iter()
        .map(|descriptor| descriptor.path.as_str())
        .collect();
    assert_eq!(
        body_paths,
        [
            node_body,
            edge_body,
            inverse_edge_body,
            vector_body,
            text_body
        ],
        "rollback must retain every authoritative SST body"
    );
    assert_eq!(
        legacy.ssts[0].unique_property_indices[0].path, legacy_unique,
        "2.0.4 must retain the bincode unique index, not the paged mirror"
    );
    assert_eq!(
        legacy.ssts[0].equality_property_indices[0].path, legacy_equality,
        "2.0.4 must retain the bincode equality index, not the paged mirror"
    );

    let current_node = &current_json["ssts"][0];
    assert_eq!(
        current_node["unique_property_indices"][0]["paged"]["path"],
        "sst/level1/nodes-articles-key.pidx"
    );
    assert_eq!(
        current_node["node_locator"]["path"],
        "sst/level1/nodes-articles.nloc"
    );
    assert!(
        current_json["search_index_builds"]
            .as_array()
            .is_some_and(|states| states.len() == 2),
        "the fixture must exercise 2.0.5-only top-level state"
    );
    assert!(
        edge_body.ends_with(".ep.csr"),
        "the forward CSR name must imply the optional 2.0.5 .epidx mirror"
    );

    let legacy_json = serde_json::to_value(&legacy).unwrap();
    assert!(legacy_json.get("search_index_builds").is_none());
    assert!(legacy_json["ssts"][0].get("node_locator").is_none());
    assert!(legacy_json["ssts"][0]["unique_property_indices"][0]
        .get("paged")
        .is_none());
    assert!(legacy_json["ssts"][0]["equality_property_indices"][0]
        .get("key_encoding")
        .is_none());
}

/// Frozen copy of the 2.0.4 `[u8; 16]` JSON representation.
mod serde_key16 {
    use base64::Engine as _;
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &[u8; 16], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&base64::engine::general_purpose::STANDARD.encode(value))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 16], D::Error> {
        let raw = String::deserialize(deserializer)?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(raw)
            .map_err(D::Error::custom)?;
        let bytes: [u8; 16] = bytes.try_into().map_err(|bytes: Vec<u8>| {
            D::Error::custom(format!("expected 16 bytes, got {}", bytes.len()))
        })?;
        Ok(bytes)
    }
}
