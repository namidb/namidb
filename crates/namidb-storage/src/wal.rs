//! Write-ahead log on object storage.
//!
//! ## Design
//!
//! A WAL is a sequence of **immutable, append-once segment objects** in
//! `<namespace>/wal/<seq>.wal`. Each segment is sealed once written —
//! there is no "open" segment we mutate. The writer batches incoming
//! records in memory (group commit) and seals a segment when either the
//! time or size threshold is met, then PUTs the whole thing to object
//! storage with `PutMode::Create`. Two writers that pick the same `seq`
//! cannot both succeed.
//!
//! Acknowledgement to the client happens **after** the segment PUT returns
//! success — at that point the records are durable.
//!
//! ## Binary format (v0)
//!
//! ```text
//! Segment ::= Header Records+ Footer
//! Header ::= "TGWL" (4B) | version: u16 | reserved: u16 | record_count: u32 | first_lsn: u64
//! Record ::= length: u32 | crc32: u32 | lsn: u64 | payload: bytes[length]
//! Footer ::= "TGEL" (4B) | crc32_of_header_plus_records: u32
//! ```
//!
//! All integers are little-endian. CRC32 uses the IEEE polynomial via
//! [`crc32fast`]. `version = 0` for now; bumping it is a breaking change
//! and requires a new manifest schema field.

use std::sync::Arc;

use bytes::{Buf, BufMut, Bytes, BytesMut};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};
use tracing::{debug, instrument, trace};

use crate::error::{Error, Result};
use crate::paths::NamespacePaths;

const MAGIC_HEADER: &[u8; 4] = b"TGWL";
const MAGIC_FOOTER: &[u8; 4] = b"TGEL";
const FORMAT_VERSION: u16 = 0;

/// A single WAL record. `lsn` is assigned by the writer monotonically and
/// becomes the durable order of operations within a namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalRecord {
 pub lsn: u64,
 pub payload: Bytes,
}

/// A sealed WAL segment ready to be PUT to object storage, or just read
/// back from it.
#[derive(Debug, Clone)]
pub struct WalSegment {
 pub seq: u64,
 pub records: Vec<WalRecord>,
}

impl WalSegment {
 pub fn new(seq: u64) -> Self {
 Self {
 seq,
 records: Vec::new(),
 }
 }

 pub fn is_empty(&self) -> bool {
 self.records.is_empty()
 }
 pub fn len(&self) -> usize {
 self.records.len()
 }

 pub fn push(&mut self, rec: WalRecord) {
 self.records.push(rec);
 }

 pub fn first_lsn(&self) -> u64 {
 self.records.first().map(|r| r.lsn).unwrap_or(0)
 }
 pub fn last_lsn(&self) -> u64 {
 self.records.last().map(|r| r.lsn).unwrap_or(0)
 }

 /// Encode the segment into a single contiguous byte buffer ready for
 /// `put_opts`. Allocates exactly once.
 pub fn encode(&self) -> Bytes {
 let payload_bytes: usize = self.records.iter().map(|r| r.payload.len()).sum();
 let cap = 4 + 2 + 2 + 4 + 8 + self.records.len() * (4 + 4 + 8) + payload_bytes + 4 + 4;
 let mut buf = BytesMut::with_capacity(cap);

 buf.put_slice(MAGIC_HEADER);
 buf.put_u16_le(FORMAT_VERSION);
 buf.put_u16_le(0); // reserved
 buf.put_u32_le(self.records.len() as u32);
 buf.put_u64_le(self.first_lsn());

 for rec in &self.records {
 let mut hasher = crc32fast::Hasher::new();
 hasher.update(&rec.lsn.to_le_bytes());
 hasher.update(&rec.payload);
 let crc = hasher.finalize();
 buf.put_u32_le(rec.payload.len() as u32);
 buf.put_u32_le(crc);
 buf.put_u64_le(rec.lsn);
 buf.put_slice(&rec.payload);
 }

 let so_far_crc = crc32fast::hash(&buf[..]);
 buf.put_slice(MAGIC_FOOTER);
 buf.put_u32_le(so_far_crc);
 buf.freeze()
 }

 /// Parse a segment from its on-disk bytes. Validates magic numbers,
 /// per-record CRCs, and the segment-wide footer CRC.
 pub fn decode(seq: u64, mut bytes: Bytes) -> Result<Self> {
 let footer_overhead = 4 + 4;
 if bytes.len() < 4 + 2 + 2 + 4 + 8 + footer_overhead {
 return Err(Error::Corrupted {
 path: format!("wal#{seq}"),
 detail: format!("buffer too short ({} bytes)", bytes.len()),
 });
 }

 // Pull off the footer first so the rest of `bytes` is exactly what
 // the CRC was computed over.
 let footer_start = bytes.len() - footer_overhead;
 let body = bytes.slice(..footer_start);
 let mut footer = bytes.split_off(footer_start);
 let footer_magic = &footer[..4];
 if footer_magic != MAGIC_FOOTER {
 return Err(Error::Corrupted {
 path: format!("wal#{seq}"),
 detail: "footer magic mismatch".into(),
 });
 }
 footer.advance(4);
 let footer_crc = footer.get_u32_le();
 let actual_crc = crc32fast::hash(&body[..]);
 if footer_crc != actual_crc {
 return Err(Error::Corrupted {
 path: format!("wal#{seq}"),
 detail: format!(
 "footer crc mismatch: declared {footer_crc:#x}, actual {actual_crc:#x}"
 ),
 });
 }

 let mut cursor = body;
 let header_magic = cursor.split_to(4);
 if &header_magic[..] != MAGIC_HEADER {
 return Err(Error::Corrupted {
 path: format!("wal#{seq}"),
 detail: "header magic mismatch".into(),
 });
 }
 let version = cursor.get_u16_le();
 if version != FORMAT_VERSION {
 return Err(Error::Corrupted {
 path: format!("wal#{seq}"),
 detail: format!("unsupported WAL format version {version}"),
 });
 }
 let _reserved = cursor.get_u16_le();
 let record_count = cursor.get_u32_le();
 let _first_lsn_hint = cursor.get_u64_le();

 let mut records = Vec::with_capacity(record_count as usize);
 for i in 0..record_count {
 if cursor.remaining() < 4 + 4 + 8 {
 return Err(Error::Corrupted {
 path: format!("wal#{seq}"),
 detail: format!("truncated record at index {i}"),
 });
 }
 let length = cursor.get_u32_le() as usize;
 let crc = cursor.get_u32_le();
 let lsn = cursor.get_u64_le();
 if cursor.remaining() < length {
 return Err(Error::Corrupted {
 path: format!("wal#{seq}"),
 detail: format!(
 "payload short by {} bytes at record {i}",
 length - cursor.remaining()
 ),
 });
 }
 let payload = cursor.split_to(length);
 let mut hasher = crc32fast::Hasher::new();
 hasher.update(&lsn.to_le_bytes());
 hasher.update(&payload);
 let actual = hasher.finalize();
 if actual != crc {
 return Err(Error::Corrupted {
 path: format!("wal#{seq}"),
 detail: format!(
 "record {i} crc mismatch: declared {crc:#x}, actual {actual:#x}"
 ),
 });
 }
 records.push(WalRecord { lsn, payload });
 }

 if cursor.has_remaining() {
 return Err(Error::Corrupted {
 path: format!("wal#{seq}"),
 detail: format!("{} trailing bytes after records", cursor.remaining()),
 });
 }

 Ok(WalSegment { seq, records })
 }
}

/// I/O surface for the WAL. Tiny on purpose: append a sealed segment, list
/// segments, read a segment back.
///
/// Group-commit batching is the caller's responsibility (it lives in the
/// memtable / writer machinery), so we can keep this layer dependency-free.
#[derive(Clone)]
pub struct WalStore {
 store: Arc<dyn ObjectStore>,
 paths: NamespacePaths,
}

impl std::fmt::Debug for WalStore {
 fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
 f.debug_struct("WalStore")
 .field("paths", &self.paths)
 .finish()
 }
}

impl WalStore {
 pub fn new(store: Arc<dyn ObjectStore>, paths: NamespacePaths) -> Self {
 Self { store, paths }
 }

 pub fn paths(&self) -> &NamespacePaths {
 &self.paths
 }

 /// PUT a sealed segment with `PutMode::Create`. Returns
 /// [`Error::Precondition`] if two writers tried to claim the same seq —
 /// the loser is fenced.
 #[instrument(skip(self, segment), fields(
 namespace = %self.paths.namespace(),
 seq = segment.seq,
 records = segment.len(),
 ))]
 pub async fn append_segment(&self, segment: &WalSegment) -> Result<Path> {
 let path = self.paths.wal_segment(segment.seq);
 let body = segment.encode();
 trace!(bytes = body.len(), "PUT wal segment");
 let opts = PutOptions::from(PutMode::Create);
 match self
 .store
 .put_opts(&path, PutPayload::from(body), opts)
 .await
 {
 Ok(_) => Ok(path),
 Err(object_store::Error::AlreadyExists { .. }) => Err(Error::precondition(format!(
 "WAL segment {} already exists; another writer raced ahead",
 segment.seq
 ))),
 Err(other) => Err(Error::ObjectStore(other)),
 }
 }

 /// List every WAL segment under this namespace, in seq order.
 ///
 /// Cheap because object stores return prefix listings in lexical order
 /// and we pad seq numbers to a fixed width on encoding.
 #[instrument(skip(self), fields(namespace = %self.paths.namespace()))]
 pub async fn list_segments(&self) -> Result<Vec<WalSegmentRef>> {
 use futures::TryStreamExt;
 let wal_dir = self.paths.wal_dir();
 let mut stream = self.store.list(Some(&wal_dir));
 let mut out = Vec::new();
 while let Some(meta) = stream.try_next().await? {
 let name = meta
 .location
 .filename()
 .ok_or_else(|| {
 Error::invariant(format!("WAL object has no filename: {}", meta.location))
 })?
 .to_string();
 // Filenames are `<16-hex-digits>.wal`
 let Some(stem) = name.strip_suffix(".wal") else {
 debug!(?name, "skipping non-WAL file under wal dir");
 continue;
 };
 let seq = u64::from_str_radix(stem, 16).map_err(|_| Error::Corrupted {
 path: meta.location.as_ref().to_string(),
 detail: format!("WAL filename '{name}' is not a 16-hex-digit seq"),
 })?;
 out.push(WalSegmentRef {
 seq,
 path: meta.location,
 size_bytes: meta.size,
 });
 }
 out.sort_by_key(|r| r.seq);
 Ok(out)
 }

 /// Fetch and decode a segment.
 #[instrument(skip(self), fields(namespace = %self.paths.namespace(), seq))]
 pub async fn read_segment(&self, seq: u64) -> Result<WalSegment> {
 let path = self.paths.wal_segment(seq);
 let res = self.store.get(&path).await?;
 let body = res.bytes().await?;
 WalSegment::decode(seq, body)
 }

 /// Convenience: read all segments newer than or equal to `start_seq`.
 pub async fn read_segments_since(&self, start_seq: u64) -> Result<Vec<WalSegment>> {
 let refs = self.list_segments().await?;
 let mut out = Vec::new();
 for r in refs.into_iter().filter(|r| r.seq >= start_seq) {
 out.push(self.read_segment(r.seq).await?);
 }
 Ok(out)
 }
}

/// Lightweight reference to a segment as observed in a `list` call.
#[derive(Debug, Clone)]
pub struct WalSegmentRef {
 pub seq: u64,
 pub path: Path,
 pub size_bytes: u64,
}

#[cfg(test)]
mod tests {
 use std::sync::Arc;

 use namidb_core::NamespaceId;
 use object_store::memory::InMemory;

 use super::*;

 fn store() -> (Arc<dyn ObjectStore>, NamespacePaths) {
 let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
 let paths = NamespacePaths::new("", NamespaceId::new("acme").unwrap());
 (store, paths)
 }

 fn rec(lsn: u64, msg: &str) -> WalRecord {
 WalRecord {
 lsn,
 payload: Bytes::copy_from_slice(msg.as_bytes()),
 }
 }

 #[test]
 fn encode_decode_roundtrip() {
 let mut seg = WalSegment::new(7);
 seg.push(rec(1, "alpha"));
 seg.push(rec(2, "beta"));
 seg.push(rec(3, ""));
 seg.push(rec(4, &"x".repeat(4096)));
 let bytes = seg.encode();
 let back = WalSegment::decode(7, bytes).unwrap();
 assert_eq!(back.seq, 7);
 assert_eq!(back.records, seg.records);
 }

 #[test]
 fn decode_rejects_truncated_buffer() {
 let mut seg = WalSegment::new(1);
 seg.push(rec(1, "hello"));
 let bytes = seg.encode();
 let truncated = bytes.slice(..bytes.len() - 5);
 let err = WalSegment::decode(1, truncated).unwrap_err();
 match err {
 Error::Corrupted { .. } => {}
 other => panic!("expected Corrupted, got {other:?}"),
 }
 }

 #[test]
 fn decode_detects_record_crc_corruption() {
 let mut seg = WalSegment::new(2);
 seg.push(rec(1, "payload"));
 let mut bytes = seg.encode().to_vec();
 // Flip a byte deep inside the record payload region.
 let payload_offset = 4 + 2 + 2 + 4 + 8 + 4 + 4 + 8 + 2; // header + record-header + a few bytes in
 bytes[payload_offset] ^= 0xff;
 // Recompute footer CRC so we exercise the *per-record* CRC check
 // rather than the segment-wide one.
 let body = &bytes[..bytes.len() - 4 - 4];
 let new_crc = crc32fast::hash(body);
 let crc_bytes = new_crc.to_le_bytes();
 let len = bytes.len();
 bytes[len - 4..].copy_from_slice(&crc_bytes);

 let err = WalSegment::decode(2, Bytes::from(bytes)).unwrap_err();
 match err {
 Error::Corrupted { detail, .. } => assert!(detail.contains("crc")),
 other => panic!("expected Corrupted, got {other:?}"),
 }
 }

 #[tokio::test]
 async fn append_then_list_then_read() {
 let (store, paths) = store();
 let wal = WalStore::new(store, paths);

 for seq in 1..=3u64 {
 let mut seg = WalSegment::new(seq);
 seg.push(rec(seq * 10, &format!("seg{seq}")));
 wal.append_segment(&seg).await.unwrap();
 }

 let refs = wal.list_segments().await.unwrap();
 let seqs: Vec<u64> = refs.iter().map(|r| r.seq).collect();
 assert_eq!(seqs, vec![1, 2, 3]);

 let seg = wal.read_segment(2).await.unwrap();
 assert_eq!(seg.records.len(), 1);
 assert_eq!(seg.records[0].lsn, 20);
 assert_eq!(seg.records[0].payload.as_ref(), b"seg2");
 }

 #[tokio::test]
 async fn append_is_create_only() {
 let (store, paths) = store();
 let wal = WalStore::new(store, paths);

 let mut seg = WalSegment::new(42);
 seg.push(rec(1, "once"));
 wal.append_segment(&seg).await.unwrap();

 let mut dup = WalSegment::new(42);
 dup.push(rec(1, "again"));
 let err = wal.append_segment(&dup).await.unwrap_err();
 match err {
 Error::Precondition(msg) => assert!(msg.contains("42")),
 other => panic!("expected Precondition, got {other:?}"),
 }
 }

 #[tokio::test]
 async fn read_segments_since_returns_in_order() {
 let (store, paths) = store();
 let wal = WalStore::new(store, paths);

 for seq in [5u64, 1, 3, 2, 4] {
 let mut seg = WalSegment::new(seq);
 seg.push(rec(seq, "x"));
 wal.append_segment(&seg).await.unwrap();
 }

 let segs = wal.read_segments_since(3).await.unwrap();
 let seqs: Vec<u64> = segs.iter().map(|s| s.seq).collect();
 assert_eq!(seqs, vec![3, 4, 5]);
 }
}
