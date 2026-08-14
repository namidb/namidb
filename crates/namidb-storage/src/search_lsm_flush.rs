//! Atomic incremental search-index work attached to one node flush.
//!
//! This coordinator reads before-images from the pinned persisted manifest,
//! compares them with the final rows in the frozen memtable, and prepares at
//! most one VG6/FT4 delta per registered index. It performs no object PUT and
//! no manifest CAS itself: [`crate::flush::flush`] uploads these immutable
//! artifacts beside the new Nodes SST, then applies the manifest update in the
//! same CAS. Any preparation or upload error therefore happens before the
//! Nodes descriptor can become visible.
//!
//! Without `vector-index` or `text-index` there is no registered index this
//! coordinator can build for, so `IndexWork` has no constructible variants and
//! the whole builder half of the module is inert. Its helpers are still
//! compiled to keep the two configurations honest, so that build alone waives
//! the resulting dead-code diagnostics.
#![cfg_attr(
    not(any(feature = "vector-index", feature = "text-index")),
    allow(dead_code, unused_imports, unused_variables, unused_mut)
)]

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

#[cfg(test)]
use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use uuid::Uuid;
use xxhash_rust::xxh3::Xxh3;

use namidb_core::{NodeId, Schema, Value};

use crate::error::{Error, Result};
use crate::flush::{NodeRow, NodeWriteRecord, SidecarPayload};
use crate::manifest::{
    KindSpecificStats, LoadedManifest, Manifest, ManifestStore, SstDescriptor, SstKind, SstLevel,
};
use crate::memtable::{MemOp, MemtableSnapshot};
use crate::read::{NodeView, Snapshot};
use crate::search_lsm::{
    catalog_signature, encode_search_barrier, is_canonical_search_barrier_descriptor,
    search_barrier_descriptor, validate_search_lsm, wrap_legacy_search_base, CoverageDisposition,
    SearchCoverage, SearchEventRange, SearchLsmKind, SearchLsmState, SearchLsmStatus,
    MAX_ACTIVE_SEARCH_SEGMENTS,
};
use crate::spooled_object::SpooledObject;
use crate::sst::search_delta::SearchFilterValue;

#[cfg(feature = "text-index")]
use crate::manifest::TextIndexDescriptor;
#[cfg(feature = "vector-index")]
use crate::manifest::VectorIndexDescriptor;
#[cfg(feature = "text-index")]
use crate::sst::text::v4::{
    text_v4_payload_fingerprint, TextV4BuildContext, TextV4ExternalBuildConfig,
    TextV4ExternalBuilder, TextV4Mutation, TextV4Payload,
};
#[cfg(feature = "vector-index")]
use crate::sst::vector::v6::{
    vector_v6_payload_fingerprint, VectorV6BuildContext, VectorV6ExternalBuildConfig,
    VectorV6ExternalBuilder, VectorV6Mutation, VectorV6Payload,
};

const SEARCH_LSM_MAX_SEGMENTS_ENV: &str = "NAMIDB_SEARCH_LSM_MAX_SEGMENTS";
const INDEX_BUILD_MEMORY_ENV: &str = "NAMIDB_INDEX_BUILD_MEMORY_BYTES";
const DEFAULT_SEARCH_LSM_MAX_SEGMENTS: usize = 8;
const MIN_SEARCH_LSM_SEGMENTS: usize = 2;
const DEFAULT_INDEX_BUILD_MEMORY_BYTES: usize = 64 * 1024 * 1024;
const CLASSIFIER_VERSION: u16 = 1;
const LOOKUP_CHUNK_MAX_ROWS: usize = 512;
const MIN_PER_INDEX_BUILD_MEMORY_BYTES: usize = 512 * 1024;

/// One immutable search object prepared for the flush upload phase.
#[derive(Debug)]
pub(crate) struct SearchFlushUpload {
    pub(crate) path: Path,
    pub(crate) body: SidecarPayload,
}

/// Manifest-only half of a prepared search flush.
#[derive(Debug, Default)]
pub(crate) struct SearchFlushManifestUpdate {
    descriptors: Vec<SstDescriptor>,
    states: Vec<SearchLsmState>,
    retired_descriptor_ids: HashSet<Uuid>,
}

impl SearchFlushManifestUpdate {
    pub(crate) fn apply(self, manifest: &mut Manifest) {
        manifest
            .ssts
            .retain(|descriptor| !self.retired_descriptor_ids.contains(&descriptor.id));
        for state in &self.states {
            manifest.search_lsm.retain(|existing| {
                existing.kind != state.kind || existing.index_name != state.index_name
            });
        }
        manifest.ssts.extend(self.descriptors);
        manifest.search_lsm.extend(self.states);
    }

    pub(crate) fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }
}

/// Complete local work product. Consuming it cannot perform external I/O.
#[derive(Debug, Default)]
pub(crate) struct SearchFlushPlan {
    uploads: Vec<SearchFlushUpload>,
    manifest: SearchFlushManifestUpdate,
}

impl SearchFlushPlan {
    pub(crate) fn into_parts(self) -> (Vec<SearchFlushUpload>, SearchFlushManifestUpdate) {
        (self.uploads, self.manifest)
    }
}

#[derive(Debug, Clone, Copy)]
struct SearchFlushConfig {
    max_segments: usize,
    build_memory_bytes: usize,
}

impl SearchFlushConfig {
    fn from_env() -> Result<Self> {
        let max_segments = match std::env::var(SEARCH_LSM_MAX_SEGMENTS_ENV) {
            Ok(value) => parse_segment_cap(&value)?,
            Err(std::env::VarError::NotPresent) => DEFAULT_SEARCH_LSM_MAX_SEGMENTS,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(Error::precondition(format!(
                    "{SEARCH_LSM_MAX_SEGMENTS_ENV} is not valid UTF-8"
                )));
            }
        };
        let build_memory_bytes = match std::env::var(INDEX_BUILD_MEMORY_ENV) {
            Ok(value) => value.parse::<usize>().map_err(|error| {
                Error::precondition(format!(
                    "{INDEX_BUILD_MEMORY_ENV} must be an exact byte count: {error}"
                ))
            })?,
            Err(std::env::VarError::NotPresent) => DEFAULT_INDEX_BUILD_MEMORY_BYTES,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(Error::precondition(format!(
                    "{INDEX_BUILD_MEMORY_ENV} is not valid UTF-8"
                )));
            }
        };
        if build_memory_bytes == 0 {
            return Err(Error::precondition(format!(
                "{INDEX_BUILD_MEMORY_ENV} must be positive"
            )));
        }
        Ok(Self {
            max_segments,
            build_memory_bytes,
        })
    }
}

fn parse_segment_cap(value: &str) -> Result<usize> {
    let parsed = value.trim().parse::<usize>().map_err(|error| {
        Error::precondition(format!(
            "{SEARCH_LSM_MAX_SEGMENTS_ENV} must be an exact integer: {error}"
        ))
    })?;
    if parsed < MIN_SEARCH_LSM_SEGMENTS {
        return Err(Error::precondition(format!(
            "{SEARCH_LSM_MAX_SEGMENTS_ENV} must be at least {MIN_SEARCH_LSM_SEGMENTS}"
        )));
    }
    Ok(parsed.min(MAX_ACTIVE_SEARCH_SEGMENTS))
}

/// Prepare search deltas and their exact manifest coverage for one new Nodes
/// SST. No object is uploaded by this function.
pub(crate) async fn prepare_search_flush(
    manifest_store: &ManifestStore,
    base: &LoadedManifest,
    final_schema: &Schema,
    node_sst: &SstDescriptor,
    node_rows: &[NodeRow],
) -> Result<SearchFlushPlan> {
    if node_sst.kind != SstKind::Nodes || node_rows.is_empty() {
        return Err(Error::invariant(
            "search flush requires one non-empty Nodes SST",
        ));
    }
    if node_sst.row_count != node_rows.len() as u64 {
        return Err(Error::invariant(
            "search flush Nodes descriptor/row count diverged",
        ));
    }

    #[cfg(not(feature = "vector-index"))]
    if !base.manifest.vector_indexes.is_empty() {
        return Err(Error::precondition(
            "registered vector indexes require the vector-index build feature before node flush",
        ));
    }
    #[cfg(not(feature = "text-index"))]
    if !base.manifest.text_indexes.is_empty() {
        return Err(Error::precondition(
            "registered text indexes require the text-index build feature before node flush",
        ));
    }

    let index_count = base
        .manifest
        .vector_indexes
        .len()
        .checked_add(base.manifest.text_indexes.len())
        .ok_or_else(|| Error::invariant("search index count overflows"))?;
    if index_count == 0 {
        return Ok(SearchFlushPlan::default());
    }
    let config = SearchFlushConfig::from_env()?;
    let build_pool = config.build_memory_bytes / 2;
    let per_index_memory = build_pool / index_count;
    if per_index_memory < MIN_PER_INDEX_BUILD_MEMORY_BYTES {
        return Err(Error::precondition(format!(
            "{INDEX_BUILD_MEMORY_ENV}={} cannot fund {index_count} concurrent search builders; \
             each requires at least {MIN_PER_INDEX_BUILD_MEMORY_BYTES} bytes",
            config.build_memory_bytes
        )));
    }

    // A malformed committed generation is corruption, not an invitation to
    // silently replace it with a fallback state. Only a legitimate projected
    // catalog/schema change may invalidate the old generation and start a new
    // Building generation below.
    validate_search_lsm(&base.manifest).map_err(|error| {
        Error::invariant(format!(
            "committed Search-LSM state is invalid before node flush: {error}"
        ))
    })?;
    let mut projected = base.manifest.clone();
    projected.schema = final_schema.clone();
    let projected_lsm_valid = validate_search_lsm(&projected).is_ok();
    let store = manifest_store.store().clone();
    let paths = manifest_store.paths().clone();
    let mut retired = HashSet::new();
    let mut works: Vec<IndexWork> = Vec::with_capacity(index_count);

    #[cfg(feature = "vector-index")]
    for descriptor in projected.vector_indexes.clone() {
        let state = select_or_initialize_state(
            store.clone(),
            &paths,
            &projected,
            SearchLsmKind::Vector,
            &descriptor.name,
            projected_lsm_valid,
            &mut retired,
        )
        .await?;
        let event = next_event(&state)?;
        let filters = crate::search_lsm::vector_native_filter_properties(&projected, &descriptor);
        let build_config = VectorV6ExternalBuildConfig {
            memory_budget_bytes: per_index_memory,
            ..Default::default()
        };
        let builder = VectorV6ExternalBuilder::with_config(
            &state,
            &descriptor,
            VectorV6BuildContext {
                sst_id: Uuid::now_v7(),
                event_ranges: vec![event],
                complete_filter_properties: filters.clone(),
            },
            build_config,
        )?;
        works.push(IndexWork::Vector(VectorWork {
            state,
            event,
            label_id: projected.label_dict.id(&descriptor.label).map(|id| id.0),
            descriptor,
            filters,
            builder,
            digest: classifier_hasher(SearchLsmKind::Vector),
        }));
    }

    #[cfg(feature = "text-index")]
    for descriptor in projected.text_indexes.clone() {
        let state = select_or_initialize_state(
            store.clone(),
            &paths,
            &projected,
            SearchLsmKind::Text,
            &descriptor.name,
            projected_lsm_valid,
            &mut retired,
        )
        .await?;
        let event = next_event(&state)?;
        let filters = crate::search_lsm::text_native_filter_properties(&projected, &descriptor);
        let build_config = TextV4ExternalBuildConfig {
            memory_budget_bytes: per_index_memory,
            ..Default::default()
        };
        let builder = TextV4ExternalBuilder::with_config(
            &state,
            TextV4BuildContext {
                sst_id: Uuid::now_v7(),
                event_ranges: vec![event],
                complete_filter_properties: filters.clone(),
            },
            build_config,
        )?;
        works.push(IndexWork::Text(TextWork {
            state,
            event,
            label_id: projected.label_dict.id(&descriptor.label).map(|id| id.0),
            descriptor,
            filters,
            builder,
            digest: classifier_hasher(SearchLsmKind::Text),
            live_total_len: 0,
        }));
    }

    // One pinned persisted before-image source is shared by every index.
    // Chunks bound decoded NodeViews and after-image records independently of
    // corpus size; builders retain only their explicitly budgeted workspace.
    let empty_memtable = MemtableSnapshot::empty();
    let snapshot = Snapshot::new(base.clone(), &empty_memtable, store, paths.clone());
    let lookup_budget = (config.build_memory_bytes / 4).max(1);
    let mut start = 0usize;
    while start < node_rows.len() {
        let end = chunk_end(node_rows, start, lookup_budget)?;
        let ids = node_rows[start..end]
            .iter()
            .map(|row| NodeId::from_uuid(Uuid::from_bytes(row.id)))
            .collect::<Vec<_>>();
        let before = snapshot.batch_lookup_nodes("", &ids).await?;
        if before.len() != end - start {
            return Err(Error::invariant(
                "search before-image batch returned the wrong cardinality",
            ));
        }
        let chunk = node_rows[start..end]
            .iter()
            .zip(before)
            .map(|(row, before)| (row.id, row.lsn, row.op.clone(), before))
            .collect::<Vec<_>>();
        works = crate::flush::run_flush_build(move || {
            let mut works = works;
            for (node_id, lsn, op, before) in chunk {
                let after = match op {
                    MemOp::Tombstone => None,
                    MemOp::Upsert(payload) => Some(NodeWriteRecord::decode(&payload)?),
                };
                for work in &mut works {
                    work.observe(node_id, lsn, before.as_ref(), after.as_ref())?;
                }
            }
            Ok(works)
        })
        .await?;
        start = end;
    }

    let finish_paths = paths.clone();
    let finish_node_sst = node_sst.clone();
    let plan = crate::flush::run_flush_build(move || {
        let mut plan = SearchFlushPlan {
            manifest: SearchFlushManifestUpdate {
                retired_descriptor_ids: retired,
                ..Default::default()
            },
            ..Default::default()
        };
        for work in works {
            work.finish(
                config.max_segments,
                &finish_node_sst,
                &finish_paths,
                &mut plan,
            )?;
        }
        Ok(plan)
    })
    .await?;

    // Validate the exact manifest shape before a single PUT begins. Active
    // generations must cover every visible Nodes SST; Building generations
    // remain explicit fallback but still validate every physical reference.
    let mut validation = projected;
    validation.ssts.push(node_sst.clone());
    let SearchFlushManifestUpdate {
        descriptors,
        states,
        retired_descriptor_ids,
    } = &plan.manifest;
    validation
        .ssts
        .retain(|descriptor| !retired_descriptor_ids.contains(&descriptor.id));
    for state in states {
        validation.search_lsm.retain(|existing| {
            existing.kind != state.kind || existing.index_name != state.index_name
        });
    }
    validation.ssts.extend(descriptors.iter().cloned());
    validation.search_lsm.extend(states.iter().cloned());
    validate_search_lsm(&validation)
        .map_err(|error| Error::invariant(format!("prepared search flush is invalid: {error}")))?;
    Ok(plan)
}

fn next_event(state: &SearchLsmState) -> Result<SearchEventRange> {
    let end = state
        .next_event_seq
        .checked_add(1)
        .ok_or_else(|| Error::precondition("search event sequence exhausted u64"))?;
    Ok(SearchEventRange::new(state.next_event_seq, end))
}

fn chunk_end(rows: &[NodeRow], start: usize, byte_budget: usize) -> Result<usize> {
    let mut end = start;
    let mut bytes = 0usize;
    while end < rows.len() && end - start < LOOKUP_CHUNK_MAX_ROWS {
        let row_bytes = match &rows[end].op {
            MemOp::Tombstone => 64,
            MemOp::Upsert(payload) => payload.len().saturating_add(64),
        };
        if end > start && bytes.saturating_add(row_bytes) > byte_budget {
            break;
        }
        if end == start && row_bytes > byte_budget {
            return Err(Error::precondition(format!(
                "one node mutation requires {row_bytes} bytes, above the {byte_budget}-byte \
                 search before-image chunk budget"
            )));
        }
        bytes = bytes
            .checked_add(row_bytes)
            .ok_or_else(|| Error::precondition("search lookup chunk accounting overflows"))?;
        end += 1;
    }
    Ok(end)
}

fn classifier_hasher(kind: SearchLsmKind) -> Xxh3 {
    let mut hasher = Xxh3::new();
    hasher.update(b"NamiDB/SearchFlushClassifier/v1");
    hasher.update(&CLASSIFIER_VERSION.to_le_bytes());
    hasher.update(&[match kind {
        SearchLsmKind::Vector => 0,
        SearchLsmKind::Text => 1,
    }]);
    hasher
}

fn update_classifier_digest(
    hasher: &mut Xxh3,
    node_id: [u8; 16],
    lsn: u64,
    before: Option<u64>,
    after: Option<u64>,
) {
    hasher.update(&node_id);
    hasher.update(&lsn.to_le_bytes());
    for fingerprint in [before, after] {
        match fingerprint {
            Some(fingerprint) => {
                hasher.update(&[1]);
                hasher.update(&fingerprint.to_le_bytes());
            }
            None => hasher.update(&[0]),
        }
    }
}

fn filter_payload(
    properties: &BTreeMap<String, Value>,
    filters: &[String],
) -> BTreeMap<String, SearchFilterValue> {
    filters
        .iter()
        .filter_map(|property| {
            properties
                .get(property)
                .and_then(SearchFilterValue::from_value)
                .map(|value| (property.clone(), value))
        })
        .collect()
}

#[cfg(feature = "vector-index")]
fn vector_payload(
    descriptor: &VectorIndexDescriptor,
    filters: &[String],
    carries_label: bool,
    properties: &BTreeMap<String, Value>,
) -> Option<VectorV6Payload> {
    if !carries_label {
        return None;
    }
    let vector = match properties.get(&descriptor.property)? {
        Value::Vec(vector) => vector.clone(),
        Value::VecI8 { codes, scale } => {
            codes.iter().map(|code| f32::from(*code) * *scale).collect()
        }
        _ => return None,
    };
    if vector.len() != descriptor.dim as usize || vector.iter().any(|value| !value.is_finite()) {
        return None;
    }
    // A zero-norm vector is not cosine-rankable. The flat scan's
    // `vector_score(Cosine, …)` drops it and both V5 builders skip it, so it is
    // not a member here either: returning `None` records a suppress instead of
    // a live payload, which is what keeps the delta's corpus equal to the flat
    // scan's. Dot and L2 are well-defined on the zero vector, so keep it there.
    if descriptor.metric == crate::manifest::VectorMetric::Cosine
        && vector.iter().all(|value| *value == 0.0)
    {
        return None;
    }
    Some(VectorV6Payload {
        vector,
        filters: filter_payload(properties, filters),
    })
}

#[cfg(feature = "text-index")]
fn text_payload(
    descriptor: &TextIndexDescriptor,
    filters: &[String],
    carries_label: bool,
    properties: &BTreeMap<String, Value>,
) -> Option<TextV4Payload> {
    if !carries_label {
        return None;
    }
    let parts = descriptor
        .properties
        .iter()
        .filter_map(|property| match properties.get(property) {
            Some(Value::Str(value)) => Some(value.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return None;
    }
    Some(TextV4Payload {
        text: parts.join(" "),
        filters: filter_payload(properties, filters),
    })
}

enum IndexWork {
    #[cfg(feature = "vector-index")]
    Vector(VectorWork),
    #[cfg(feature = "text-index")]
    Text(TextWork),
}

impl IndexWork {
    fn observe(
        &mut self,
        node_id: [u8; 16],
        lsn: u64,
        before: Option<&NodeView>,
        after: Option<&NodeWriteRecord>,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "vector-index")]
            Self::Vector(work) => work.observe(node_id, lsn, before, after),
            #[cfg(feature = "text-index")]
            Self::Text(work) => work.observe(node_id, lsn, before, after),
            #[cfg(not(any(feature = "vector-index", feature = "text-index")))]
            _ => unreachable!("IndexWork has no constructible variants without search features"),
        }
    }

    fn finish(
        self,
        max_segments: usize,
        node_sst: &SstDescriptor,
        paths: &crate::paths::NamespacePaths,
        plan: &mut SearchFlushPlan,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "vector-index")]
            Self::Vector(work) => work.finish(max_segments, node_sst, paths, plan),
            #[cfg(feature = "text-index")]
            Self::Text(work) => work.finish(max_segments, node_sst, paths, plan),
        }
    }
}

#[cfg(feature = "vector-index")]
struct VectorWork {
    state: SearchLsmState,
    event: SearchEventRange,
    label_id: Option<u32>,
    descriptor: VectorIndexDescriptor,
    filters: Vec<String>,
    builder: VectorV6ExternalBuilder,
    digest: Xxh3,
}

#[cfg(feature = "vector-index")]
impl VectorWork {
    fn observe(
        &mut self,
        node_id: [u8; 16],
        lsn: u64,
        before: Option<&NodeView>,
        after: Option<&NodeWriteRecord>,
    ) -> Result<()> {
        let before_payload = before.and_then(|node| {
            vector_payload(
                &self.descriptor,
                &self.filters,
                node.labels.contains(&self.descriptor.label),
                &node.properties,
            )
        });
        let after_payload = after.and_then(|record| {
            vector_payload(
                &self.descriptor,
                &self.filters,
                self.label_id
                    .is_some_and(|label| record.labels.contains(&label)),
                &record.properties,
            )
        });
        let before_fingerprint = before_payload
            .as_ref()
            .map(vector_v6_payload_fingerprint)
            .transpose()?;
        let after_fingerprint = after_payload
            .as_ref()
            .map(vector_v6_payload_fingerprint)
            .transpose()?;
        update_classifier_digest(
            &mut self.digest,
            node_id,
            lsn,
            before_fingerprint,
            after_fingerprint,
        );
        self.builder.push(VectorV6Mutation {
            node_id,
            lsn,
            before: before_payload,
            after: after_payload,
        })
    }

    fn finish(
        self,
        max_segments: usize,
        node_sst: &SstDescriptor,
        paths: &crate::paths::NamespacePaths,
        plan: &mut SearchFlushPlan,
    ) -> Result<()> {
        let VectorWork {
            mut state,
            event,
            descriptor,
            builder,
            digest,
            ..
        } = self;
        let artifact = builder.finish()?;
        let disposition = if let Some(artifact) = artifact {
            ensure_segment_room(&state, max_segments)?;
            let segment = artifact.output.segment.clone();
            let data_descriptor = vector_segment_descriptor(
                &descriptor,
                &segment,
                &artifact.output.version_table,
                artifact.len,
                paths,
            );
            let path = absolute_descriptor_path(paths, &data_descriptor);
            plan.uploads.push(SearchFlushUpload {
                path,
                body: SidecarPayload::Spooled(SpooledObject::from_file(
                    artifact.file,
                    artifact.len,
                )),
            });
            plan.manifest.descriptors.push(data_descriptor);
            state.segments.push(segment);
            CoverageDisposition::Segment
        } else {
            let before_after_digest = digest.digest().max(1);
            state.proven_empty_event_ranges.push(event);
            CoverageDisposition::ProvenEmpty {
                classifier_version: CLASSIFIER_VERSION,
                before_after_digest,
            }
        };
        finish_state(state, event, disposition, node_sst, paths, plan)
    }
}

#[cfg(feature = "text-index")]
struct TextWork {
    state: SearchLsmState,
    event: SearchEventRange,
    label_id: Option<u32>,
    descriptor: TextIndexDescriptor,
    filters: Vec<String>,
    builder: TextV4ExternalBuilder,
    digest: Xxh3,
    live_total_len: u64,
}

#[cfg(feature = "text-index")]
impl TextWork {
    fn observe(
        &mut self,
        node_id: [u8; 16],
        lsn: u64,
        before: Option<&NodeView>,
        after: Option<&NodeWriteRecord>,
    ) -> Result<()> {
        let before_payload = before.and_then(|node| {
            text_payload(
                &self.descriptor,
                &self.filters,
                node.labels.contains(&self.descriptor.label),
                &node.properties,
            )
        });
        let after_payload = after.and_then(|record| {
            text_payload(
                &self.descriptor,
                &self.filters,
                self.label_id
                    .is_some_and(|label| record.labels.contains(&label)),
                &record.properties,
            )
        });
        let before_fingerprint = before_payload
            .as_ref()
            .map(text_v4_payload_fingerprint)
            .transpose()?;
        let after_fingerprint = after_payload
            .as_ref()
            .map(text_v4_payload_fingerprint)
            .transpose()?;
        if before_fingerprint != after_fingerprint {
            if let Some(payload) = &after_payload {
                self.live_total_len = self
                    .live_total_len
                    .checked_add(crate::text::tokenize(&payload.text).len() as u64)
                    .ok_or_else(|| Error::invariant("FT4 live token count overflows"))?;
            }
        }
        update_classifier_digest(
            &mut self.digest,
            node_id,
            lsn,
            before_fingerprint,
            after_fingerprint,
        );
        self.builder.push(TextV4Mutation {
            node_id,
            lsn,
            before: before_payload,
            after: after_payload,
        })
    }

    fn finish(
        self,
        max_segments: usize,
        node_sst: &SstDescriptor,
        paths: &crate::paths::NamespacePaths,
        plan: &mut SearchFlushPlan,
    ) -> Result<()> {
        let TextWork {
            mut state,
            event,
            descriptor,
            builder,
            digest,
            live_total_len,
            ..
        } = self;
        let artifact = builder.finish()?;
        let disposition = if let Some(artifact) = artifact {
            ensure_segment_room(&state, max_segments)?;
            let segment = artifact.output.segment.clone();
            let data_descriptor = text_segment_descriptor(
                &descriptor,
                &segment,
                &artifact.output.version_table,
                artifact.len,
                live_total_len,
                paths,
            );
            let path = absolute_descriptor_path(paths, &data_descriptor);
            plan.uploads.push(SearchFlushUpload {
                path,
                body: SidecarPayload::Spooled(SpooledObject::from_file(
                    artifact.file,
                    artifact.len,
                )),
            });
            plan.manifest.descriptors.push(data_descriptor);
            state.segments.push(segment);
            CoverageDisposition::Segment
        } else {
            let before_after_digest = digest.digest().max(1);
            state.proven_empty_event_ranges.push(event);
            CoverageDisposition::ProvenEmpty {
                classifier_version: CLASSIFIER_VERSION,
                before_after_digest,
            }
        };
        finish_state(state, event, disposition, node_sst, paths, plan)
    }
}

fn ensure_segment_room(state: &SearchLsmState, max_segments: usize) -> Result<()> {
    if state.segments.len() >= max_segments {
        return Err(Error::precondition(format!(
            "search index '{}' has {} active segments and \
             {SEARCH_LSM_MAX_SEGMENTS_ENV}={max_segments}; compact it before flushing another delta",
            state.index_name,
            state.segments.len()
        )));
    }
    Ok(())
}

fn finish_state(
    mut state: SearchLsmState,
    event: SearchEventRange,
    disposition: CoverageDisposition,
    node_sst: &SstDescriptor,
    paths: &crate::paths::NamespacePaths,
    plan: &mut SearchFlushPlan,
) -> Result<()> {
    state.next_event_seq = event.end;
    state.coverage.push(SearchCoverage {
        node_sst_id: node_sst.id,
        node_sst_max_lsn: node_sst.max_lsn,
        event_ranges: vec![event],
        disposition,
    });
    if let Some(old_barrier) = state.compat_barrier_sst_id {
        plan.manifest.retired_descriptor_ids.insert(old_barrier);
    }
    let barrier_id = Uuid::now_v7();
    state.compat_barrier_sst_id = Some(barrier_id);
    let barrier_body = encode_search_barrier(&state)
        .map_err(|error| Error::invariant(format!("search barrier encode failed: {error}")))?;
    let file_name = format!(
        "{}-{}-{}.slb",
        barrier_id.simple(),
        state.kind.sst_kind().path_tag(),
        state.index_name
    );
    let path = paths.sst_object(SstLevel::L0.as_u32(), &file_name);
    let relative = relative_sst_path(SstLevel::L0.as_u32(), &file_name);
    let barrier = search_barrier_descriptor(
        &state,
        barrier_id,
        SstLevel::L0,
        relative,
        barrier_body.len() as u64,
    );
    plan.uploads.push(SearchFlushUpload {
        path,
        body: SidecarPayload::InMemory(barrier_body.into()),
    });
    plan.manifest.descriptors.push(barrier);
    plan.manifest.states.push(state);
    Ok(())
}

#[cfg(feature = "vector-index")]
fn vector_segment_descriptor(
    descriptor: &VectorIndexDescriptor,
    segment: &crate::search_lsm::SearchSegmentRef,
    version: &crate::sst::search_delta::SearchVersionTableRef,
    size_bytes: u64,
    paths: &crate::paths::NamespacePaths,
) -> SstDescriptor {
    let file_name = format!(
        "{}-{}-{}.vg6",
        segment.sst_id.simple(),
        SstKind::VectorGraph.path_tag(),
        descriptor.name
    );
    search_segment_descriptor(
        segment,
        SstKind::VectorGraph,
        descriptor.name.clone(),
        relative_sst_path(SstLevel::L0.as_u32(), &file_name),
        size_bytes,
        version,
        KindSpecificStats::VectorGraph {
            dim: descriptor.dim,
            metric: match descriptor.metric {
                crate::manifest::VectorMetric::Cosine => "cosine",
                crate::manifest::VectorMetric::Dot => "dot",
                crate::manifest::VectorMetric::Euclidean => "euclidean",
            }
            .into(),
            point_count: segment.live_payload_count,
            r: descriptor.r,
            l_build: descriptor.l_build,
            alpha: descriptor.alpha,
            entry_medoid: 0,
        },
        paths,
    )
}

#[cfg(feature = "text-index")]
fn text_segment_descriptor(
    descriptor: &TextIndexDescriptor,
    segment: &crate::search_lsm::SearchSegmentRef,
    version: &crate::sst::search_delta::SearchVersionTableRef,
    size_bytes: u64,
    live_total_len: u64,
    paths: &crate::paths::NamespacePaths,
) -> SstDescriptor {
    let file_name = format!(
        "{}-{}-{}.ft4",
        segment.sst_id.simple(),
        SstKind::TextIndex.path_tag(),
        descriptor.name
    );
    search_segment_descriptor(
        segment,
        SstKind::TextIndex,
        descriptor.name.clone(),
        relative_sst_path(SstLevel::L0.as_u32(), &file_name),
        size_bytes,
        version,
        KindSpecificStats::TextIndex {
            doc_count: segment.live_payload_count,
            term_count: 0,
            total_len: live_total_len,
        },
        paths,
    )
}

#[allow(clippy::too_many_arguments)]
fn search_segment_descriptor(
    segment: &crate::search_lsm::SearchSegmentRef,
    kind: SstKind,
    scope: String,
    path: String,
    size_bytes: u64,
    version: &crate::sst::search_delta::SearchVersionTableRef,
    kind_specific: KindSpecificStats,
    _paths: &crate::paths::NamespacePaths,
) -> SstDescriptor {
    SstDescriptor {
        id: segment.sst_id,
        kind,
        scope,
        level: SstLevel::L0,
        path,
        size_bytes,
        row_count: segment.mutation_count,
        created_at: chrono::Utc::now(),
        min_key: version.min_node_id,
        max_key: version.max_node_id,
        min_lsn: segment.min_lsn,
        max_lsn: segment.max_lsn,
        schema_version_min: 0,
        schema_version_max: 0,
        property_stats: Vec::new(),
        kind_specific,
        bloom: None,
        unique_property_indices: Vec::new(),
        equality_property_indices: Vec::new(),
        label_index: None,
        node_locator: None,
        per_label_property_stats: Vec::new(),
    }
}

fn absolute_descriptor_path(
    paths: &crate::paths::NamespacePaths,
    descriptor: &SstDescriptor,
) -> Path {
    Path::from(format!(
        "{}/{}",
        paths.namespace_prefix().as_ref(),
        descriptor.path
    ))
}

fn relative_sst_path(level: u32, file_name: &str) -> String {
    format!("sst/level{level}/{file_name}")
}

async fn select_or_initialize_state(
    store: Arc<dyn ObjectStore>,
    paths: &crate::paths::NamespacePaths,
    manifest: &Manifest,
    kind: SearchLsmKind,
    index_name: &str,
    projected_lsm_valid: bool,
    retired: &mut HashSet<Uuid>,
) -> Result<SearchLsmState> {
    let signature = catalog_signature(manifest, kind, index_name).ok_or_else(|| {
        Error::precondition(format!(
            "search catalog identity for '{index_name}' is missing or ambiguous"
        ))
    })?;
    let matches = manifest
        .search_lsm
        .iter()
        .filter(|state| state.kind == kind && state.index_name == index_name)
        .collect::<Vec<_>>();
    if projected_lsm_valid && matches.len() == 1 && matches[0].catalog_signature == signature {
        return Ok(matches[0].clone());
    }
    for state in matches {
        if let Some(barrier) = state.compat_barrier_sst_id {
            retired.insert(barrier);
        }
    }
    for descriptor in manifest.ssts.iter().filter(|descriptor| {
        descriptor.kind == kind.sst_kind()
            && descriptor.scope == index_name
            && is_canonical_search_barrier_descriptor(descriptor)
    }) {
        retired.insert(descriptor.id);
    }

    if let Some(state) = try_adopt_legacy(store, paths, manifest, kind, index_name).await? {
        return Ok(state);
    }

    let old_nodes = manifest
        .ssts
        .iter()
        .any(|descriptor| descriptor.kind == SstKind::Nodes);
    let scoped_physical = manifest
        .ssts
        .iter()
        .any(|descriptor| descriptor.kind == kind.sst_kind() && descriptor.scope == index_name);
    Ok(SearchLsmState {
        index_name: index_name.into(),
        kind,
        catalog_signature: signature,
        generation_id: Uuid::now_v7(),
        status: if old_nodes || scoped_physical {
            SearchLsmStatus::Building
        } else {
            SearchLsmStatus::Active
        },
        ..Default::default()
    })
}

async fn try_adopt_legacy(
    store: Arc<dyn ObjectStore>,
    paths: &crate::paths::NamespacePaths,
    manifest: &Manifest,
    kind: SearchLsmKind,
    index_name: &str,
) -> Result<Option<SearchLsmState>> {
    let scoped = manifest
        .ssts
        .iter()
        .filter(|descriptor| {
            descriptor.kind == kind.sst_kind()
                && descriptor.scope == index_name
                && !is_canonical_search_barrier_descriptor(descriptor)
        })
        .collect::<Vec<_>>();
    let [base] = scoped.as_slice() else {
        return Ok(None);
    };
    let max_node_lsn = manifest
        .ssts
        .iter()
        .filter(|descriptor| descriptor.kind == SstKind::Nodes)
        .map(|descriptor| descriptor.max_lsn)
        .max()
        .unwrap_or(0);
    if max_node_lsn == 0 || base.max_lsn < max_node_lsn {
        return Ok(None);
    }
    let marker_matches = manifest.search_index_builds.iter().any(|marker| {
        marker.kind == kind.sst_kind()
            && marker.name == index_name
            && marker.max_node_lsn >= max_node_lsn
            && legacy_marker_signatures(manifest, kind, index_name)
                .iter()
                .any(|signature| signature == &marker.catalog_signature)
    });
    if !marker_matches {
        return Ok(None);
    }
    let absolute = absolute_descriptor_path(paths, base);
    let magic = store.get_range(&absolute, 0..8).await?;
    let supported = match kind {
        #[cfg(feature = "vector-index")]
        SearchLsmKind::Vector => magic.as_ref() == crate::sst::vector::v5::MAGIC_V5,
        #[cfg(not(feature = "vector-index"))]
        SearchLsmKind::Vector => false,
        #[cfg(feature = "text-index")]
        SearchLsmKind::Text => magic.as_ref() == crate::sst::text::RANGE_READABLE_MAGIC,
        #[cfg(not(feature = "text-index"))]
        SearchLsmKind::Text => false,
    };
    if !supported {
        return Ok(None);
    }
    let state = wrap_legacy_search_base(
        manifest,
        kind,
        index_name,
        base.id,
        Uuid::now_v7(),
        Uuid::now_v7(),
    )
    .map_err(|error| Error::precondition(format!("legacy search adoption failed: {error}")))?;
    Ok(Some(state))
}

fn legacy_marker_signatures(
    manifest: &Manifest,
    kind: SearchLsmKind,
    index_name: &str,
) -> Vec<String> {
    match kind {
        SearchLsmKind::Vector => manifest
            .vector_indexes
            .iter()
            .find(|descriptor| descriptor.name == index_name)
            .map(|descriptor| {
                vec![
                    crate::search_lsm::vector_catalog_signature(manifest, descriptor),
                    crate::search_lsm::legacy_vector_catalog_signature(manifest, descriptor),
                ]
            })
            .unwrap_or_default(),
        SearchLsmKind::Text => manifest
            .text_indexes
            .iter()
            .find(|descriptor| descriptor.name == index_name)
            .map(|descriptor| {
                vec![
                    crate::search_lsm::text_catalog_signature(descriptor),
                    crate::search_lsm::legacy_text_catalog_signature(descriptor),
                ]
            })
            .unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_cap_defaults_clamps_and_rejects_values_without_progress() {
        assert_eq!(DEFAULT_SEARCH_LSM_MAX_SEGMENTS, 8);
        assert_eq!(parse_segment_cap("2").unwrap(), 2);
        assert_eq!(parse_segment_cap(" 8 ").unwrap(), 8);
        assert_eq!(
            parse_segment_cap("999").unwrap(),
            MAX_ACTIVE_SEARCH_SEGMENTS
        );
        assert!(parse_segment_cap("0").is_err());
        assert!(parse_segment_cap("1").is_err());
        assert!(parse_segment_cap("eight").is_err());
    }

    #[test]
    fn chunks_respect_rows_and_bytes_without_splitting_one_row() {
        let rows = (0_u128..600)
            .map(|id| NodeRow {
                id: id.to_be_bytes(),
                lsn: id as u64 + 1,
                op: MemOp::Upsert(Bytes::from_static(b"12345678")),
            })
            .collect::<Vec<_>>();
        assert_eq!(chunk_end(&rows, 0, usize::MAX).unwrap(), 512);
        assert_eq!(chunk_end(&rows, 512, usize::MAX).unwrap(), 600);
        assert!(chunk_end(&rows, 0, 1).is_err());
    }

    #[test]
    fn classifier_digest_binds_before_after_and_lsn() {
        let mut first = classifier_hasher(SearchLsmKind::Vector);
        update_classifier_digest(&mut first, [1; 16], 7, Some(11), Some(12));
        let mut same = classifier_hasher(SearchLsmKind::Vector);
        update_classifier_digest(&mut same, [1; 16], 7, Some(11), Some(12));
        let mut changed = classifier_hasher(SearchLsmKind::Vector);
        update_classifier_digest(&mut changed, [1; 16], 8, Some(11), Some(12));
        assert_eq!(first.digest(), same.digest());
        assert_ne!(first.digest(), changed.digest());
    }

    #[cfg(all(feature = "vector-index", feature = "text-index"))]
    #[test]
    fn projection_classifies_relabel_property_removal_and_filter_only_changes() {
        use crate::manifest::{TextIndexDescriptor, VectorMetric, VectorQuantization};

        let vector = VectorIndexDescriptor {
            name: "emb".into(),
            label: "Doc".into(),
            property: "embedding".into(),
            dim: 2,
            metric: VectorMetric::Cosine,
            r: 8,
            l_build: 16,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        };
        let text = TextIndexDescriptor::new(
            "body".into(),
            "Doc".into(),
            vec!["title".into(), "body".into()],
        );
        let filters = vec!["active".to_string()];
        let properties = BTreeMap::from([
            ("embedding".into(), Value::Vec(vec![1.0, 0.0])),
            ("title".into(), Value::Str("Hello".into())),
            ("body".into(), Value::Str("world".into())),
            ("active".into(), Value::Bool(true)),
        ]);
        let vector_before = vector_payload(&vector, &filters, true, &properties).unwrap();
        let text_before = text_payload(&text, &filters, true, &properties).unwrap();
        assert!(vector_payload(&vector, &filters, false, &properties).is_none());
        assert!(text_payload(&text, &filters, false, &properties).is_none());

        let mut removed = properties.clone();
        removed.remove("embedding");
        assert!(vector_payload(&vector, &filters, true, &removed).is_none());
        removed.remove("title");
        removed.remove("body");
        assert!(text_payload(&text, &filters, true, &removed).is_none());

        let mut filter_only = properties;
        filter_only.insert("active".into(), Value::Bool(false));
        assert_ne!(
            vector_before,
            vector_payload(&vector, &filters, true, &filter_only).unwrap()
        );
        assert_ne!(
            text_before,
            text_payload(&text, &filters, true, &filter_only).unwrap()
        );
    }

    #[test]
    fn full_segment_cap_backpressures_but_proven_empty_needs_no_slot() {
        let mut state = SearchLsmState {
            index_name: "idx".into(),
            ..Default::default()
        };
        state.segments = vec![crate::search_lsm::SearchSegmentRef::default(); 8];
        assert!(ensure_segment_room(&state, 8).is_err());
        // ProvenEmpty goes directly through finish_state and deliberately does
        // not call this segment-room gate.
        assert_eq!(state.segments.len(), 8);
    }

    #[cfg(all(feature = "vector-index", feature = "text-index"))]
    #[tokio::test]
    async fn two_flushes_publish_exact_coverage_and_filter_only_deltas() {
        use std::sync::Arc;

        use namidb_core::{DataType, LabelDef, NamespaceId, PropertyDef, SchemaBuilder};
        use object_store::memory::InMemory;

        use crate::fence::WriterFence;
        use crate::manifest::{
            TextIndexDescriptor, VectorIndexDescriptor, VectorMetric, VectorQuantization,
        };
        use crate::memtable::{MemKey, Memtable};

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = crate::paths::NamespacePaths::new(
            "tests",
            NamespaceId::new("search-flush-two").unwrap(),
        );
        let manifests = ManifestStore::new(store, paths);
        let base = manifests.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![
                    PropertyDef::new("embedding", DataType::FloatVector { dim: 2 }, false).unwrap(),
                    PropertyDef::new("body", DataType::Utf8, false).unwrap(),
                    PropertyDef::new("active", DataType::Bool, true)
                        .unwrap()
                        .with_indexed(true),
                    PropertyDef::new("note", DataType::Utf8, false).unwrap(),
                ],
            })
            .unwrap()
            .build();
        let mut catalog = base.manifest.next_version(fence.writer_id);
        catalog.schema = schema.clone();
        let label_id = catalog.label_dict.intern("Doc").0;
        catalog.vector_indexes.push(VectorIndexDescriptor {
            name: "doc_embedding".into(),
            label: "Doc".into(),
            property: "embedding".into(),
            dim: 2,
            metric: VectorMetric::Cosine,
            r: 8,
            l_build: 16,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        });
        catalog.text_indexes.push(TextIndexDescriptor::new(
            "doc_body".into(),
            "Doc".into(),
            vec!["body".into()],
        ));
        let mut current = manifests.commit(&fence, &base, catalog).await.unwrap();
        let node = NodeId::from_uuid(Uuid::from_u128(7));

        let payload = |active: bool, note: &str| {
            NodeWriteRecord {
                properties: BTreeMap::from([
                    ("embedding".into(), Value::Vec(vec![1.0, 0.0])),
                    ("body".into(), Value::Str("hello legal world".into())),
                    ("active".into(), Value::Bool(active)),
                    ("note".into(), Value::Str(note.into())),
                ]),
                schema_version: 1,
                labels: vec![label_id],
            }
            .encode()
            .unwrap()
        };

        let mut first = Memtable::new();
        first.apply(
            MemKey::Node { id: node },
            10,
            MemOp::Upsert(payload(true, "one")),
        );
        let first_frozen = first.freeze();
        let first =
            crate::flush::flush(&manifests, &fence, &current, &first_frozen, schema.clone())
                .await
                .unwrap();
        current = first.committed;
        validate_search_lsm(&current.manifest).unwrap();
        assert_eq!(current.manifest.search_lsm.len(), 2);
        for state in &current.manifest.search_lsm {
            assert_eq!(state.status, SearchLsmStatus::Active);
            assert_eq!(state.segments.len(), 1);
            assert_eq!(state.coverage.len(), 1);
            assert_eq!(state.next_event_seq, 1);
        }

        // Unrelated-property-only update: exact before/after projection is
        // equal, so both indexes receive ProvenEmpty coverage and no data
        // segment.
        let mut second = Memtable::new();
        second.apply(
            MemKey::Node { id: node },
            20,
            MemOp::Upsert(payload(true, "two")),
        );
        let second_frozen = second.freeze();
        let second =
            crate::flush::flush(&manifests, &fence, &current, &second_frozen, schema.clone())
                .await
                .unwrap();
        current = second.committed;
        validate_search_lsm(&current.manifest).unwrap();
        for state in &current.manifest.search_lsm {
            assert_eq!(state.segments.len(), 1);
            assert_eq!(state.coverage.len(), 2);
            assert_eq!(state.proven_empty_event_ranges.len(), 1);
            let CoverageDisposition::ProvenEmpty {
                classifier_version,
                before_after_digest,
            } = state.coverage[1].disposition
            else {
                panic!("second coverage must be proven-empty");
            };
            assert_eq!(classifier_version, CLASSIFIER_VERSION);
            assert_ne!(before_after_digest, 0);
        }

        // Native-filter-only update must be a real live delta for both VG6 and
        // FT4 even though vector/text content itself did not change.
        let mut third = Memtable::new();
        third.apply(
            MemKey::Node { id: node },
            30,
            MemOp::Upsert(payload(false, "two")),
        );
        let third_frozen = third.freeze();
        let third = crate::flush::flush(&manifests, &fence, &current, &third_frozen, schema)
            .await
            .unwrap();
        let reloaded = manifests.load_current().await.unwrap();
        assert_eq!(reloaded.manifest.version, third.committed.manifest.version);
        validate_search_lsm(&reloaded.manifest).unwrap();
        for state in &reloaded.manifest.search_lsm {
            assert_eq!(state.segments.len(), 2);
            assert_eq!(state.coverage.len(), 3);
            assert_eq!(state.next_event_seq, 3);
            assert!(matches!(
                state.coverage[2].disposition,
                CoverageDisposition::Segment
            ));
            let barriers = reloaded
                .manifest
                .ssts
                .iter()
                .filter(|descriptor| {
                    descriptor.kind == state.kind.sst_kind()
                        && descriptor.scope == state.index_name
                        && is_canonical_search_barrier_descriptor(descriptor)
                })
                .count();
            assert_eq!(barriers, 1, "old barrier must retire atomically");
        }
    }
}
