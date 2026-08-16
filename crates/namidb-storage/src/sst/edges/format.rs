//! Wire layout: file header + footer + section table.
//!
//! Defined by [RFC-002](../../../../../docs/rfc/002-sst-format.md) §3.2.1
//! and §3.2.8.

use bytes::Bytes;
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

use crate::error::{Error, Result};

// ─── magic + versions ───────────────────────────────────────────────────

pub const HEADER_MAGIC: [u8; 8] = *b"TGEDGE\0\0";
pub const FOOTER_MAGIC: [u8; 8] = *b"TGEDGE\xFE\xEF";
pub const HEADER_LEN: usize = 64;
pub const FORMAT_MAJOR: u8 = 1;
/// v1.1 adds the optional `SECTION_EDGE_ORDINALS` accelerator. Readers keep
/// accepting v1.0 bodies and fall back to the eager CSR path when it is absent.
///
/// v1.2 adds `SECTION_PAGE_CHECKSUMS`. It is the first version for which
/// arbitrary section ranges can be authenticated without downloading the
/// complete section.
pub const FORMAT_MINOR: u8 = 2;

// ─── flag bits ──────────────────────────────────────────────────────────

pub const FLAG_HAS_PROPERTIES: u32 = 1 << 0;
pub const FLAG_HAS_TOMBSTONES: u32 = 1 << 1;
pub const FLAG_SKEW_BUCKETS: u32 = 1 << 2;
pub const FLAG_INVERSE_PARTNER: u32 = 1 << 3;
pub const FLAGS_RESERVED_MASK: u32 = !0u32 << 4; // bits 4..31

// ─── section kinds ──────────────────────────────────────────────────────

pub const SECTION_KEY_IDS: u16 = 0x0001;
pub const SECTION_OFFSETS: u16 = 0x0002;
pub const SECTION_PARTNERS: u16 = 0x0003;
pub const SECTION_PER_EDGE_LSN: u16 = 0x0004;
pub const SECTION_PER_EDGE_TOMBSTONES: u16 = 0x0005;
pub const SECTION_FENCE_INDEX: u16 = 0x0006;
/// Fixed-width cumulative edge ordinals, one `u64` LE per key plus a final
/// sentinel. Entry `i` is the first per-edge row owned by `key_ids[i]`.
///
/// Byte offsets alone cannot locate `per_edge_lsn` or tombstone rows without
/// scanning every preceding partner block. This additive v1.1 section makes a
/// complete adjacency lookup independently range-readable.
pub const SECTION_EDGE_ORDINALS: u16 = 0x0007;
/// Rooted directory of fixed-size page checksums for every other section.
///
/// The directory section's complete xxhash is stored in the checksummed
/// footer. Directory entries then authenticate partial object-store reads
/// from topology, metadata, fence, and property sections.
pub const SECTION_PAGE_CHECKSUMS: u16 = 0x0008;
/// Optional v1.2 identity/root record emitted by current writers. It binds the
/// immutable SST UUID and fixed header to every preceding data-section
/// descriptor. Legacy v1.2 bodies without it remain readable.
pub const SECTION_SST_BINDING: u16 = 0x0009;
pub const SECTION_PROPERTY_STREAM: u16 = 0x0100;

pub const CODEC_NONE: u8 = 0;
pub const CODEC_ZSTD: u8 = 1;
/// Independently decodable property pages, raw or one Zstd frame per page.
pub const CODEC_PROPERTY_PAGED_NONE: u8 = 2;
pub const CODEC_PROPERTY_PAGED_ZSTD: u8 = 3;
const SECTION_ENTRY_FIXED_LEN: usize = 29;
const PAGE_CHECKSUM_DIRECTORY_MAGIC: [u8; 8] = *b"TGEPGC02";
const PAGE_CHECKSUM_DIRECTORY_HEADER_LEN: usize = 16;
const PAGE_CHECKSUM_SECTION_HEADER_LEN: usize = 20;
/// Writer page size for independently verifiable ranged reads.
pub const EDGE_CHECKSUM_PAGE_BYTES: u32 = 64 * 1024;
pub const MIN_EDGE_CHECKSUM_PAGE_BYTES: u32 = 4 * 1024;
pub const MAX_EDGE_CHECKSUM_PAGE_BYTES: u32 = 1024 * 1024;
/// Prevent a malformed directory from forcing an unbounded eager allocation.
pub const MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES: u64 = 64 * 1024 * 1024;

/// The overflow-property reserved name (mirrors the node SST convention).
pub const OVERFLOW_JSON_NAME: &str = "__overflow_json";
const SST_BINDING_MAGIC: [u8; 8] = *b"TGEBND02";
pub const SST_BINDING_LEN: usize = 40;

// ─── header ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeFileHeader {
    pub format_major: u8,
    pub format_minor: u8,
    pub flags: u32,
    pub edge_type_id: [u8; 16],
    pub src_label_id: [u8; 16],
    pub dst_label_id: [u8; 16],
}

impl EdgeFileHeader {
    pub fn new(edge_type: &str, src_label: &str, dst_label: &str, flags: u32) -> Self {
        Self {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            flags,
            edge_type_id: blake3_short(edge_type),
            src_label_id: blake3_short(src_label),
            dst_label_id: blake3_short(dst_label),
        }
    }

    pub fn encode(&self, buf: &mut Vec<u8>) {
        let start = buf.len();
        buf.extend_from_slice(&HEADER_MAGIC); // 0..8
        buf.push(self.format_major); // 8
        buf.push(self.format_minor); // 9
        buf.extend_from_slice(&(HEADER_LEN as u16).to_le_bytes()); // 10..12
        buf.extend_from_slice(&self.flags.to_le_bytes()); // 12..16
        buf.extend_from_slice(&self.edge_type_id); // 16..32
        buf.extend_from_slice(&self.src_label_id); // 32..48
        buf.extend_from_slice(&self.dst_label_id); // 48..64
        debug_assert_eq!(buf.len() - start, HEADER_LEN);
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_LEN {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!("edge SST header truncated ({} bytes)", buf.len()),
            });
        }
        if buf[..8] != HEADER_MAGIC {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: "edge SST header magic mismatch".into(),
            });
        }
        let format_major = buf[8];
        let format_minor = buf[9];
        if format_major != FORMAT_MAJOR {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!("unsupported edge SST format_major={format_major}"),
            });
        }
        if format_minor > FORMAT_MINOR {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!("unsupported edge SST format_minor={format_minor}"),
            });
        }
        let header_size = u16::from_le_bytes([buf[10], buf[11]]);
        if header_size as usize != HEADER_LEN {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!("edge SST header_size={header_size} != {HEADER_LEN}"),
            });
        }
        let flags = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
        if flags & FLAGS_RESERVED_MASK != 0 {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!("edge SST sets unknown reserved flag bits in 0x{flags:08x}"),
            });
        }
        let edge_type_id: [u8; 16] = buf[16..32].try_into().unwrap();
        let src_label_id: [u8; 16] = buf[32..48].try_into().unwrap();
        let dst_label_id: [u8; 16] = buf[48..64].try_into().unwrap();
        Ok(Self {
            format_major,
            format_minor,
            flags,
            edge_type_id,
            src_label_id,
            dst_label_id,
        })
    }
}

fn blake3_short(s: &str) -> [u8; 16] {
    let hash = blake3::hash(s.as_bytes());
    let bytes = hash.as_bytes();
    let mut out = [0u8; 16];
    out.copy_from_slice(&bytes[..16]);
    out
}

// ─── section table ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionEntry {
    pub kind: u16,
    pub offset: u64,
    pub length: u64,
    pub codec: u8,
    pub xxhash3_64: u64,
    pub name: String,
}

impl SectionEntry {
    /// Encode a single entry into `buf`.
    pub fn encode(&self, buf: &mut Vec<u8>) -> Result<()> {
        if self.name.len() > u8::MAX as usize {
            return Err(Error::invariant(format!(
                "section name '{}' exceeds 255 bytes",
                self.name
            )));
        }
        buf.extend_from_slice(&self.kind.to_le_bytes());
        buf.extend_from_slice(&self.offset.to_le_bytes());
        buf.extend_from_slice(&self.length.to_le_bytes());
        buf.push(self.codec);
        buf.push(0); // reserved
        buf.extend_from_slice(&self.xxhash3_64.to_le_bytes());
        buf.push(self.name.len() as u8);
        buf.extend_from_slice(self.name.as_bytes());
        Ok(())
    }

    pub fn decode(buf: &[u8], cursor: usize) -> Result<(Self, usize)> {
        // Fixed prefix includes `name_len`.
        let need = SECTION_ENTRY_FIXED_LEN;
        let fixed_end = cursor
            .checked_add(need)
            .ok_or_else(|| corrupted("section entry cursor overflows usize"))?;
        if fixed_end > buf.len() {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: "section entry header truncated".into(),
            });
        }
        let kind = u16::from_le_bytes([buf[cursor], buf[cursor + 1]]);
        let offset = u64::from_le_bytes(buf[cursor + 2..cursor + 10].try_into().unwrap());
        let length = u64::from_le_bytes(buf[cursor + 10..cursor + 18].try_into().unwrap());
        let codec = buf[cursor + 18];
        // reserved at cursor+19
        let xxhash3_64 = u64::from_le_bytes(buf[cursor + 20..cursor + 28].try_into().unwrap());
        let name_len = buf[cursor + 28] as usize;
        let name_start = cursor
            .checked_add(29)
            .ok_or_else(|| corrupted("section entry name start overflows usize"))?;
        let name_end = name_start
            .checked_add(name_len)
            .ok_or_else(|| corrupted("section entry name end overflows usize"))?;
        if name_end > buf.len() {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: "section entry name truncated".into(),
            });
        }
        let name = std::str::from_utf8(&buf[name_start..name_end])
            .map_err(|e| Error::Corrupted {
                path: "<edges>".into(),
                detail: format!("section entry name not utf-8: {e}"),
            })?
            .to_string();
        Ok((
            Self {
                kind,
                offset,
                length,
                codec,
                xxhash3_64,
                name,
            },
            (name_end) - cursor,
        ))
    }
}

/// Fixed-size additive identity/root record for current v1.2 bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeSstBinding {
    pub sst_id: [u8; 16],
    pub header_xxhash3_64: u64,
    pub sections_xxhash3_64: u64,
}

impl EdgeSstBinding {
    pub fn for_sections(
        sst_id: [u8; 16],
        header: &[u8],
        sections: &[SectionEntry],
    ) -> Result<Self> {
        let mut hasher = Xxh3::new();
        let mut encoded = Vec::new();
        for section in sections {
            if matches!(section.kind, SECTION_SST_BINDING | SECTION_PAGE_CHECKSUMS) {
                continue;
            }
            encoded.clear();
            section.encode(&mut encoded)?;
            hasher.update(&encoded);
        }
        Ok(Self {
            sst_id,
            header_xxhash3_64: xxh3_64(header),
            sections_xxhash3_64: hasher.digest(),
        })
    }

    pub fn encode(self) -> [u8; SST_BINDING_LEN] {
        let mut out = [0u8; SST_BINDING_LEN];
        out[..8].copy_from_slice(&SST_BINDING_MAGIC);
        out[8..24].copy_from_slice(&self.sst_id);
        out[24..32].copy_from_slice(&self.header_xxhash3_64.to_le_bytes());
        out[32..40].copy_from_slice(&self.sections_xxhash3_64.to_le_bytes());
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != SST_BINDING_LEN {
            return Err(corrupted(format!(
                "edge SST binding length {} != {SST_BINDING_LEN}",
                bytes.len()
            )));
        }
        if bytes[..8] != SST_BINDING_MAGIC {
            return Err(corrupted("edge SST binding magic mismatch"));
        }
        Ok(Self {
            sst_id: bytes[8..24].try_into().unwrap(),
            header_xxhash3_64: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            sections_xxhash3_64: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
        })
    }

    pub fn validate(
        self,
        header: &[u8],
        sections: &[SectionEntry],
        expected_sst_id: Option<[u8; 16]>,
    ) -> Result<()> {
        let expected = Self::for_sections(self.sst_id, header, sections)?;
        if self.header_xxhash3_64 != expected.header_xxhash3_64 {
            return Err(corrupted("edge SST binding header checksum mismatch"));
        }
        if self.sections_xxhash3_64 != expected.sections_xxhash3_64 {
            return Err(corrupted("edge SST binding section-root mismatch"));
        }
        if expected_sst_id.is_some_and(|id| id != self.sst_id) {
            return Err(corrupted(
                "edge SST binding UUID does not match immutable object path",
            ));
        }
        Ok(())
    }
}

// ─── independently verifiable section pages ────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionPageChecksums {
    pub offset: u64,
    pub length: u64,
    pub checksums: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgePageChecksumDirectory {
    pub page_size: u32,
    pub sections: Vec<SectionPageChecksums>,
}

impl EdgePageChecksumDirectory {
    /// Build checksums over every supplied data section. The checksum
    /// directory itself must not be present yet because it is the root's leaf,
    /// not a recursively checksummed target.
    pub fn build(file: &[u8], sections: &[SectionEntry], page_size: u32) -> Result<Self> {
        validate_checksum_page_size(page_size)?;
        if sections.len() > u32::MAX as usize {
            return Err(Error::invariant("page checksum section count exceeds u32"));
        }
        let page_size_u64 = u64::from(page_size);
        let mut directory_sections = Vec::with_capacity(sections.len());
        let mut directory_bytes = PAGE_CHECKSUM_DIRECTORY_HEADER_LEN;
        for entry in sections {
            if entry.kind == SECTION_PAGE_CHECKSUMS {
                return Err(Error::invariant(
                    "page checksum directory cannot checksum itself",
                ));
            }
            let end = entry
                .offset
                .checked_add(entry.length)
                .ok_or_else(|| Error::invariant("section offset+length overflows u64"))?;
            let start = usize::try_from(entry.offset)
                .map_err(|_| Error::invariant("section offset does not fit usize"))?;
            let end = usize::try_from(end)
                .map_err(|_| Error::invariant("section end does not fit usize"))?;
            let bytes = file.get(start..end).ok_or_else(|| {
                Error::invariant(format!(
                    "section kind 0x{:04x} is outside the encoded edge body",
                    entry.kind
                ))
            })?;
            let page_count = entry.length.div_ceil(page_size_u64);
            if page_count > u64::from(u32::MAX) {
                return Err(Error::invariant("section page count exceeds u32"));
            }
            let page_capacity = usize::try_from(page_count)
                .map_err(|_| Error::invariant("section page count does not fit usize"))?;
            directory_bytes = directory_bytes
                .checked_add(PAGE_CHECKSUM_SECTION_HEADER_LEN)
                .and_then(|len| {
                    page_capacity
                        .checked_mul(8)
                        .and_then(|hashes| len.checked_add(hashes))
                })
                .ok_or_else(|| {
                    Error::invariant("page checksum directory length overflows usize")
                })?;
            if directory_bytes as u64 > MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES {
                return Err(Error::invariant(format!(
                    "page checksum directory requires more than \
                     {MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES} bytes"
                )));
            }
            let mut checksums = Vec::with_capacity(page_capacity);
            for page in bytes.chunks(page_size as usize) {
                checksums.push(xxh3_64(page));
            }
            if checksums.len() != page_capacity {
                return Err(Error::invariant(
                    "encoded section page count disagrees with section length",
                ));
            }
            directory_sections.push(SectionPageChecksums {
                offset: entry.offset,
                length: entry.length,
                checksums,
            });
        }
        Ok(Self {
            page_size,
            sections: directory_sections,
        })
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        validate_checksum_page_size(self.page_size)?;
        let section_count = u32::try_from(self.sections.len())
            .map_err(|_| Error::invariant("page checksum section count exceeds u32"))?;
        let mut encoded_len = PAGE_CHECKSUM_DIRECTORY_HEADER_LEN;
        for section in &self.sections {
            validate_directory_section_shape(section, self.page_size)?;
            encoded_len = encoded_len
                .checked_add(PAGE_CHECKSUM_SECTION_HEADER_LEN)
                .and_then(|len| {
                    section
                        .checksums
                        .len()
                        .checked_mul(8)
                        .and_then(|hashes| len.checked_add(hashes))
                })
                .ok_or_else(|| {
                    Error::invariant("page checksum directory length overflows usize")
                })?;
        }
        if encoded_len as u64 > MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES {
            return Err(Error::invariant(format!(
                "page checksum directory requires {encoded_len} bytes, above limit \
                 {MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES}"
            )));
        }

        let mut out = Vec::with_capacity(encoded_len);
        out.extend_from_slice(&PAGE_CHECKSUM_DIRECTORY_MAGIC);
        out.extend_from_slice(&self.page_size.to_le_bytes());
        out.extend_from_slice(&section_count.to_le_bytes());
        for section in &self.sections {
            let page_count = u32::try_from(section.checksums.len())
                .map_err(|_| Error::invariant("section page count exceeds u32"))?;
            out.extend_from_slice(&section.offset.to_le_bytes());
            out.extend_from_slice(&section.length.to_le_bytes());
            out.extend_from_slice(&page_count.to_le_bytes());
            for checksum in &section.checksums {
                out.extend_from_slice(&checksum.to_le_bytes());
            }
        }
        debug_assert_eq!(out.len(), encoded_len);
        Ok(out)
    }

    /// Decode a directory only after its complete section checksum has been
    /// verified. Coverage must match the footer's non-directory sections
    /// exactly and in footer order.
    pub fn decode(bytes: &[u8], footer: &EdgeFileFooter) -> Result<Self> {
        if bytes.len() as u64 > MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES {
            return Err(corrupted(format!(
                "page checksum directory length {} exceeds limit \
                 {MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES}",
                bytes.len()
            )));
        }
        if bytes.len() < PAGE_CHECKSUM_DIRECTORY_HEADER_LEN {
            return Err(corrupted("page checksum directory header truncated"));
        }
        if bytes[..8] != PAGE_CHECKSUM_DIRECTORY_MAGIC {
            return Err(corrupted("page checksum directory magic mismatch"));
        }
        let page_size = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        validate_checksum_page_size(page_size).map_err(invariant_as_corrupted)?;
        let section_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let targets: Vec<_> = footer
            .sections
            .iter()
            .filter(|entry| entry.kind != SECTION_PAGE_CHECKSUMS)
            .collect();
        if section_count as usize != targets.len() {
            return Err(corrupted(format!(
                "page checksum directory covers {section_count} sections, footer requires {}",
                targets.len()
            )));
        }
        let minimum_len = PAGE_CHECKSUM_DIRECTORY_HEADER_LEN
            .checked_add(
                targets
                    .len()
                    .checked_mul(PAGE_CHECKSUM_SECTION_HEADER_LEN)
                    .ok_or_else(|| corrupted("page checksum directory minimum length overflows"))?,
            )
            .ok_or_else(|| corrupted("page checksum directory minimum length overflows"))?;
        if minimum_len > bytes.len() {
            return Err(corrupted(
                "page checksum directory cannot contain its declared section records",
            ));
        }

        let page_size_u64 = u64::from(page_size);
        let mut cursor = PAGE_CHECKSUM_DIRECTORY_HEADER_LEN;
        let mut sections = Vec::with_capacity(targets.len());
        for target in targets {
            let header_end = cursor
                .checked_add(PAGE_CHECKSUM_SECTION_HEADER_LEN)
                .ok_or_else(|| corrupted("page checksum record cursor overflows"))?;
            let header = bytes
                .get(cursor..header_end)
                .ok_or_else(|| corrupted("page checksum section record truncated"))?;
            let offset = u64::from_le_bytes(header[..8].try_into().unwrap());
            let length = u64::from_le_bytes(header[8..16].try_into().unwrap());
            let page_count = u32::from_le_bytes(header[16..20].try_into().unwrap());
            if offset != target.offset || length != target.length {
                return Err(corrupted(format!(
                    "page checksum target [{offset}, {}) does not match footer section \
                     kind 0x{:04x} [{}, {})",
                    offset.saturating_add(length),
                    target.kind,
                    target.offset,
                    target.offset.saturating_add(target.length)
                )));
            }
            let expected_pages = length.div_ceil(page_size_u64);
            if u64::from(page_count) != expected_pages {
                return Err(corrupted(format!(
                    "page checksum count {page_count} for section kind 0x{:04x} != \
                     ceil({length}/{page_size}) ({expected_pages})",
                    target.kind
                )));
            }
            let hash_bytes = usize::try_from(page_count)
                .ok()
                .and_then(|count| count.checked_mul(8))
                .ok_or_else(|| corrupted("page checksum byte count overflows usize"))?;
            let hashes_end = header_end
                .checked_add(hash_bytes)
                .ok_or_else(|| corrupted("page checksum record end overflows usize"))?;
            let hashes = bytes
                .get(header_end..hashes_end)
                .ok_or_else(|| corrupted("page checksum values truncated"))?;
            let mut checksums = Vec::with_capacity(page_count as usize);
            for raw in hashes.chunks_exact(8) {
                checksums.push(u64::from_le_bytes(raw.try_into().unwrap()));
            }
            sections.push(SectionPageChecksums {
                offset,
                length,
                checksums,
            });
            cursor = hashes_end;
        }
        if cursor != bytes.len() {
            return Err(corrupted(format!(
                "page checksum directory has {} trailing bytes",
                bytes.len() - cursor
            )));
        }
        Ok(Self {
            page_size,
            sections,
        })
    }

    pub fn for_section(&self, entry: &SectionEntry) -> Option<&SectionPageChecksums> {
        self.sections
            .iter()
            .find(|section| section.offset == entry.offset && section.length == entry.length)
    }
}

fn validate_checksum_page_size(page_size: u32) -> Result<()> {
    if !(MIN_EDGE_CHECKSUM_PAGE_BYTES..=MAX_EDGE_CHECKSUM_PAGE_BYTES).contains(&page_size)
        || !page_size.is_power_of_two()
    {
        return Err(Error::invariant(format!(
            "edge checksum page size {page_size} must be a power of two in \
             [{MIN_EDGE_CHECKSUM_PAGE_BYTES}, {MAX_EDGE_CHECKSUM_PAGE_BYTES}]"
        )));
    }
    Ok(())
}

fn validate_directory_section_shape(section: &SectionPageChecksums, page_size: u32) -> Result<()> {
    let expected_pages = section.length.div_ceil(u64::from(page_size));
    if section.checksums.len() as u64 != expected_pages {
        return Err(Error::invariant(format!(
            "section [{}, {}) has {} page checksums, expected {expected_pages}",
            section.offset,
            section.offset.saturating_add(section.length),
            section.checksums.len()
        )));
    }
    if section.checksums.len() > u32::MAX as usize {
        return Err(Error::invariant("section page count exceeds u32"));
    }
    Ok(())
}

fn corrupted(detail: impl Into<String>) -> Error {
    Error::Corrupted {
        path: "<edges>".into(),
        detail: detail.into(),
    }
}

fn invariant_as_corrupted(error: Error) -> Error {
    match error {
        Error::Invariant(detail) => corrupted(detail),
        other => other,
    }
}

// ─── footer body + trailer ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeFileFooter {
    pub sections: Vec<SectionEntry>,
    pub key_count: u64,
    pub edge_count: u64,
    pub offsets_bits: u8,
    pub min_key_id: [u8; 16],
    pub max_key_id: [u8; 16],
    pub min_lsn: u64,
    pub max_lsn: u64,
    pub schema_version_min: u64,
    pub schema_version_max: u64,
}

/// Trailer is fixed at 20 bytes: xxhash(8) + footer_len(4) + magic(8).
pub const FOOTER_TRAILER_LEN: usize = 20;

impl EdgeFileFooter {
    /// Encode the footer (body + trailer). Returns the total bytes appended
    /// to `buf`, which is also written into the trailer's `footer_len` field.
    pub fn encode(&self, buf: &mut Vec<u8>) -> Result<usize> {
        let section_count = u32::try_from(self.sections.len())
            .map_err(|_| Error::invariant("edge footer section count exceeds u32"))?;
        let start = buf.len();
        // Body
        for entry in &self.sections {
            entry.encode(buf)?;
        }
        buf.extend_from_slice(&section_count.to_le_bytes());
        buf.extend_from_slice(&self.key_count.to_le_bytes());
        buf.extend_from_slice(&self.edge_count.to_le_bytes());
        buf.push(self.offsets_bits);
        buf.extend_from_slice(&self.min_key_id);
        buf.extend_from_slice(&self.max_key_id);
        buf.extend_from_slice(&self.min_lsn.to_le_bytes());
        buf.extend_from_slice(&self.max_lsn.to_le_bytes());
        buf.extend_from_slice(&self.schema_version_min.to_le_bytes());
        buf.extend_from_slice(&self.schema_version_max.to_le_bytes());
        let body_end = buf.len();

        // Trailer
        let body_hash = xxh3_64(&buf[start..body_end]);
        buf.extend_from_slice(&body_hash.to_le_bytes());
        let footer_len_pos = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes()); // placeholder
        buf.extend_from_slice(&FOOTER_MAGIC);
        let total = buf.len() - start;
        let total_u32 = u32::try_from(total)
            .map_err(|_| Error::invariant("encoded edge footer length exceeds u32"))?;
        // Backpatch footer_len.
        let fl = total_u32.to_le_bytes();
        buf[footer_len_pos..footer_len_pos + 4].copy_from_slice(&fl);
        Ok(total)
    }

    /// Decode the footer from a fully-buffered file body. Returns the
    /// footer plus the byte offset at which the footer body starts.
    pub fn decode(body: &Bytes) -> Result<(Self, usize)> {
        let file_len = body.len();
        let footer_len = Self::encoded_len_from_trailer(body)? as usize;
        if footer_len > file_len {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!("footer_len {footer_len} exceeds file size {file_len}"),
            });
        }
        let footer_start = file_len - footer_len;
        let footer_bytes = &body[footer_start..];
        let (footer, ranged_footer_start) = Self::decode_from_tail(file_len as u64, footer_bytes)?;
        debug_assert_eq!(ranged_footer_start, footer_start as u64);
        Ok((footer, footer_start))
    }

    /// Read the encoded footer length from any byte slice ending at the
    /// object's final byte. This is intentionally independent of the complete
    /// body so an object-store reader can start with one suffix request.
    pub fn encoded_len_from_trailer(tail: &[u8]) -> Result<u32> {
        if tail.len() < FOOTER_TRAILER_LEN {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!("edge SST footer trailer truncated ({} bytes)", tail.len()),
            });
        }
        let trailer_start = tail.len() - FOOTER_TRAILER_LEN;
        if tail[tail.len() - 8..] != FOOTER_MAGIC {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: "edge SST footer magic mismatch".into(),
            });
        }
        let footer_len = u32::from_le_bytes(
            tail[trailer_start + 8..trailer_start + 12]
                .try_into()
                .unwrap(),
        );
        if footer_len < FOOTER_TRAILER_LEN as u32 {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!(
                    "footer_len {footer_len} is smaller than trailer length {FOOTER_TRAILER_LEN}"
                ),
            });
        }
        Ok(footer_len)
    }

    /// Decode an **exact footer suffix** fetched with a ranged GET.
    ///
    /// `tail` must contain precisely the `footer_len` bytes ending at
    /// `file_size`; section offsets remain absolute file offsets. Returning the
    /// logical footer start lets callers validate/cache the fetched range
    /// without allocating a zero-filled prefix as large as the SST.
    pub fn decode_from_tail(file_size: u64, tail: &[u8]) -> Result<(Self, u64)> {
        let footer_len = Self::encoded_len_from_trailer(tail)? as usize;
        if footer_len != tail.len() {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!(
                    "footer suffix length {} does not match encoded footer_len {footer_len}",
                    tail.len()
                ),
            });
        }
        let footer_len_u64 = footer_len as u64;
        let footer_start =
            file_size
                .checked_sub(footer_len_u64)
                .ok_or_else(|| Error::Corrupted {
                    path: "<edges>".into(),
                    detail: format!("footer_len {footer_len} exceeds file size {file_size}"),
                })?;
        if footer_start < HEADER_LEN as u64 {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!(
                    "footer starts at {footer_start}, before {HEADER_LEN}-byte edge header"
                ),
            });
        }

        let trailer_start = tail.len() - FOOTER_TRAILER_LEN;
        let body_end = trailer_start;
        let body_hash_stored =
            u64::from_le_bytes(tail[trailer_start..trailer_start + 8].try_into().unwrap());
        let body_hash_computed = xxh3_64(&tail[..body_end]);
        if body_hash_stored != body_hash_computed {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: "edge SST footer xxhash mismatch".into(),
            });
        }

        // Parse body. The section table comes first, followed by the
        // fixed-shape tail. We don't know the section count up front, so
        // we walk from byte zero of this exact suffix and bound by the
        // fixed-tail start.
        let tail_size: usize = 4 // section_count
 + 8 // key_count
 + 8 // edge_count
 + 1 // offsets_bits
 + 16 // min_key_id
 + 16 // max_key_id
 + 8 // min_lsn
 + 8 // max_lsn
 + 8 // schema_version_min
 + 8; // schema_version_max
        if tail_size > body_end {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: "footer body too small to contain the fixed tail".into(),
            });
        }
        let tail_start = body_end - tail_size;
        let section_count =
            u32::from_le_bytes(tail[tail_start..tail_start + 4].try_into().unwrap());
        let mut cursor = tail_start + 4;
        let key_count = u64::from_le_bytes(tail[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let edge_count = u64::from_le_bytes(tail[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let offsets_bits = tail[cursor];
        cursor += 1;
        let min_key_id: [u8; 16] = tail[cursor..cursor + 16].try_into().unwrap();
        cursor += 16;
        let max_key_id: [u8; 16] = tail[cursor..cursor + 16].try_into().unwrap();
        cursor += 16;
        let min_lsn = u64::from_le_bytes(tail[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let max_lsn = u64::from_le_bytes(tail[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let schema_version_min = u64::from_le_bytes(tail[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let schema_version_max = u64::from_le_bytes(tail[cursor..cursor + 8].try_into().unwrap());

        // Section table walk: from byte zero of the suffix up to tail_start.
        let table_end = tail_start;
        let section_capacity = usize::try_from(section_count)
            .map_err(|_| Error::invariant("edge section count does not fit usize"))?;
        if section_capacity > table_end / SECTION_ENTRY_FIXED_LEN {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!(
                    "section_count {section_count} cannot fit in {table_end}-byte footer table"
                ),
            });
        }
        let mut sections = Vec::with_capacity(section_capacity);
        let mut table_cursor = 0usize;
        for _ in 0..section_count {
            if table_cursor >= table_end {
                return Err(Error::Corrupted {
                    path: "<edges>".into(),
                    detail: "section table ended before declared count".into(),
                });
            }
            let (entry, consumed) = SectionEntry::decode(tail, table_cursor)?;
            table_cursor += consumed;
            sections.push(entry);
        }
        if table_cursor != table_end {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!("section table ended at {table_cursor}, expected {table_end}"),
            });
        }

        // Validate section ranges. RFC-002 §3.2.8 requires the section
        // table on disk to be sorted ascending by `offset`; rejecting an
        // out-of-order table here is the only way to keep readers and
        // writers from drifting on that invariant. Once sortedness is
        // guaranteed, overlap detection is a single linear pass.
        let mut prev_end: u64 = HEADER_LEN as u64;
        for (i, entry) in sections.iter().enumerate() {
            if i > 0 && entry.offset < sections[i - 1].offset {
                return Err(Error::Corrupted {
                    path: "<edges>".into(),
                    detail: format!(
 "section table not sorted by offset: entry {} at {} precedes entry {} at {}",
 i,
 entry.offset,
 i - 1,
 sections[i - 1].offset
 ),
                });
            }
            let end = entry
                .offset
                .checked_add(entry.length)
                .ok_or_else(|| Error::Corrupted {
                    path: "<edges>".into(),
                    detail: format!("section '{}' offset+length overflow", entry.name),
                })?;
            if entry.offset < HEADER_LEN as u64 || end > footer_start {
                return Err(Error::Corrupted {
                    path: "<edges>".into(),
                    detail: format!(
                        "section '{}' [{}, {}) outside of file body [{}, {})",
                        entry.name, entry.offset, end, HEADER_LEN, footer_start
                    ),
                });
            }
            if entry.offset < prev_end {
                return Err(Error::Corrupted {
                    path: "<edges>".into(),
                    detail: format!(
 "section '{}' overlaps a prior section: starts at {} but body before it ends at {}",
 entry.name, entry.offset, prev_end
 ),
                });
            }
            prev_end = end;
        }

        Ok((
            Self {
                sections,
                key_count,
                edge_count,
                offsets_bits,
                min_key_id,
                max_key_id,
                min_lsn,
                max_lsn,
                schema_version_min,
                schema_version_max,
            },
            footer_start,
        ))
    }

    /// Look up the first section whose `(kind, name)` matches.
    pub fn find(&self, kind: u16, name: &str) -> Option<&SectionEntry> {
        self.sections
            .iter()
            .find(|e| e.kind == kind && e.name == name)
    }

    /// Look up the first section by `kind` ignoring the name.
    pub fn find_kind(&self, kind: u16) -> Option<&SectionEntry> {
        self.sections.iter().find(|e| e.kind == kind)
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    #[test]
    fn header_round_trip() {
        let h = EdgeFileHeader::new(
            "KNOWS",
            "Person",
            "Person",
            FLAG_HAS_PROPERTIES | FLAG_INVERSE_PARTNER,
        );
        let mut buf = Vec::new();
        h.encode(&mut buf);
        assert_eq!(buf.len(), HEADER_LEN);
        let back = EdgeFileHeader::decode(&buf).unwrap();
        assert_eq!(back, h);
    }

    #[test]
    fn header_rejects_bad_magic() {
        let h = EdgeFileHeader::new("X", "Y", "Z", 0);
        let mut buf = Vec::new();
        h.encode(&mut buf);
        buf[0] = b'!';
        let err = EdgeFileHeader::decode(&buf).unwrap_err();
        assert!(matches!(err, Error::Corrupted { .. }));
    }

    #[test]
    fn header_rejects_unknown_reserved_flag() {
        let h = EdgeFileHeader::new("X", "Y", "Z", 1 << 10); // reserved bit 10
        let mut buf = Vec::new();
        h.encode(&mut buf);
        let err = EdgeFileHeader::decode(&buf).unwrap_err();
        assert!(matches!(err, Error::Corrupted { .. }));
    }

    #[test]
    fn header_rejects_future_major() {
        let h = EdgeFileHeader::new("X", "Y", "Z", 0);
        let mut buf = Vec::new();
        h.encode(&mut buf);
        buf[8] = 2;
        let err = EdgeFileHeader::decode(&buf).unwrap_err();
        assert!(matches!(err, Error::Corrupted { .. }));
    }

    #[test]
    fn header_rejects_future_minor() {
        let h = EdgeFileHeader::new("X", "Y", "Z", 0);
        let mut buf = Vec::new();
        h.encode(&mut buf);
        buf[9] = FORMAT_MINOR + 1;
        let err = EdgeFileHeader::decode(&buf).unwrap_err();
        assert!(matches!(err, Error::Corrupted { .. }));
    }

    #[test]
    fn page_checksum_directory_round_trips_all_data_sections() {
        let mut file = Vec::new();
        EdgeFileHeader::new("X", "Y", "Z", 0).encode(&mut file);
        let first_offset = file.len() as u64;
        file.extend_from_slice(&vec![0x5a; MIN_EDGE_CHECKSUM_PAGE_BYTES as usize + 17]);
        let second_offset = file.len() as u64;
        let sections = vec![
            SectionEntry {
                kind: SECTION_KEY_IDS,
                offset: first_offset,
                length: u64::from(MIN_EDGE_CHECKSUM_PAGE_BYTES) + 17,
                codec: CODEC_NONE,
                xxhash3_64: 0,
                name: String::new(),
            },
            SectionEntry {
                kind: SECTION_OFFSETS,
                offset: second_offset,
                length: 0,
                codec: CODEC_NONE,
                xxhash3_64: xxh3_64(&[]),
                name: String::new(),
            },
        ];
        let directory =
            EdgePageChecksumDirectory::build(&file, &sections, MIN_EDGE_CHECKSUM_PAGE_BYTES)
                .unwrap();
        assert_eq!(directory.sections[0].checksums.len(), 2);
        assert!(directory.sections[1].checksums.is_empty());
        let encoded = directory.encode().unwrap();
        let checksum_entry = SectionEntry {
            kind: SECTION_PAGE_CHECKSUMS,
            offset: second_offset,
            length: encoded.len() as u64,
            codec: CODEC_NONE,
            xxhash3_64: xxh3_64(&encoded),
            name: String::new(),
        };
        let footer = EdgeFileFooter {
            sections: sections
                .iter()
                .cloned()
                .chain(std::iter::once(checksum_entry))
                .collect(),
            key_count: 0,
            edge_count: 0,
            offsets_bits: 24,
            min_key_id: [0; 16],
            max_key_id: [0; 16],
            min_lsn: 0,
            max_lsn: 0,
            schema_version_min: 0,
            schema_version_max: 0,
        };
        assert_eq!(
            EdgePageChecksumDirectory::decode(&encoded, &footer).unwrap(),
            directory
        );
    }

    #[test]
    fn page_checksum_directory_rejects_truncation_and_malicious_page_count() {
        let section = SectionEntry {
            kind: SECTION_KEY_IDS,
            offset: HEADER_LEN as u64,
            length: 16,
            codec: CODEC_NONE,
            xxhash3_64: 0,
            name: String::new(),
        };
        let checksum_entry = SectionEntry {
            kind: SECTION_PAGE_CHECKSUMS,
            offset: HEADER_LEN as u64 + 16,
            length: 0,
            codec: CODEC_NONE,
            xxhash3_64: 0,
            name: String::new(),
        };
        let footer = EdgeFileFooter {
            sections: vec![section.clone(), checksum_entry],
            key_count: 1,
            edge_count: 1,
            offsets_bits: 24,
            min_key_id: [0; 16],
            max_key_id: [0; 16],
            min_lsn: 0,
            max_lsn: 0,
            schema_version_min: 0,
            schema_version_max: 0,
        };
        let file = vec![0u8; HEADER_LEN + 16];
        let mut encoded =
            EdgePageChecksumDirectory::build(&file, &[section], MIN_EDGE_CHECKSUM_PAGE_BYTES)
                .unwrap()
                .encode()
                .unwrap();
        assert!(EdgePageChecksumDirectory::decode(&encoded[..encoded.len() - 1], &footer).is_err());

        // Header(16) + target offset/length(16) = page_count position.
        encoded[32..36].copy_from_slice(&u32::MAX.to_le_bytes());
        let result =
            std::panic::catch_unwind(|| EdgePageChecksumDirectory::decode(&encoded, &footer));
        assert!(result.is_ok(), "malicious page count must never panic");
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn footer_round_trip_empty() {
        // Build a minimal "file": header + footer (no sections).
        let mut file = Vec::new();
        EdgeFileHeader::new("KNOWS", "P", "P", 0).encode(&mut file);
        let footer = EdgeFileFooter {
            sections: Vec::new(),
            key_count: 0,
            edge_count: 0,
            offsets_bits: 32,
            min_key_id: [0; 16],
            max_key_id: [0; 16],
            min_lsn: 0,
            max_lsn: 0,
            schema_version_min: 0,
            schema_version_max: 0,
        };
        footer.encode(&mut file).unwrap();
        let body = Bytes::from(file);
        let (back, footer_start) = EdgeFileFooter::decode(&body).unwrap();
        assert_eq!(back, footer);
        assert_eq!(footer_start, HEADER_LEN);
    }

    #[test]
    fn footer_round_trip_with_sections() {
        let mut file = Vec::new();
        EdgeFileHeader::new("KNOWS", "P", "P", 0).encode(&mut file);
        // Reserve some byte ranges for two fake sections.
        let s1_start = file.len() as u64;
        file.extend_from_slice(b"section-one-bytes");
        let s1_end = file.len() as u64;
        let s2_start = s1_end;
        file.extend_from_slice(b"another-section");
        let s2_end = file.len() as u64;

        let footer = EdgeFileFooter {
            sections: vec![
                SectionEntry {
                    kind: SECTION_KEY_IDS,
                    offset: s1_start,
                    length: s1_end - s1_start,
                    codec: CODEC_NONE,
                    xxhash3_64: 0,
                    name: String::new(),
                },
                SectionEntry {
                    kind: SECTION_PROPERTY_STREAM,
                    offset: s2_start,
                    length: s2_end - s2_start,
                    codec: CODEC_ZSTD,
                    xxhash3_64: 0,
                    name: "since".into(),
                },
            ],
            key_count: 10,
            edge_count: 25,
            offsets_bits: 24,
            min_key_id: [1; 16],
            max_key_id: [2; 16],
            min_lsn: 100,
            max_lsn: 200,
            schema_version_min: 3,
            schema_version_max: 3,
        };
        footer.encode(&mut file).unwrap();
        let body = Bytes::from(file);
        let (back, footer_start) = EdgeFileFooter::decode(&body).unwrap();
        assert_eq!(back, footer);
        assert!(footer_start > HEADER_LEN);
    }

    #[test]
    fn footer_decodes_from_exact_ranged_suffix() {
        let mut file = Vec::new();
        EdgeFileHeader::new("KNOWS", "P", "P", 0).encode(&mut file);
        file.extend_from_slice(&[0x5a; 128]);
        let footer = EdgeFileFooter {
            sections: vec![SectionEntry {
                kind: SECTION_KEY_IDS,
                offset: HEADER_LEN as u64,
                length: 128,
                codec: CODEC_NONE,
                xxhash3_64: xxh3_64(&[0x5a; 128]),
                name: String::new(),
            }],
            key_count: 8,
            edge_count: 8,
            offsets_bits: 24,
            min_key_id: [1; 16],
            max_key_id: [2; 16],
            min_lsn: 1,
            max_lsn: 8,
            schema_version_min: 0,
            schema_version_max: 0,
        };
        let footer_len = footer.encode(&mut file).unwrap();
        let suffix = &file[file.len() - footer_len..];
        let (decoded, start) = EdgeFileFooter::decode_from_tail(file.len() as u64, suffix).unwrap();
        assert_eq!(decoded, footer);
        assert_eq!(start, (HEADER_LEN + 128) as u64);
    }

    #[test]
    fn footer_ranged_decoder_requires_exact_suffix() {
        let mut file = Vec::new();
        EdgeFileHeader::new("X", "Y", "Z", 0).encode(&mut file);
        let footer = EdgeFileFooter {
            sections: Vec::new(),
            key_count: 0,
            edge_count: 0,
            offsets_bits: 24,
            min_key_id: [0; 16],
            max_key_id: [0; 16],
            min_lsn: 0,
            max_lsn: 0,
            schema_version_min: 0,
            schema_version_max: 0,
        };
        let footer_len = footer.encode(&mut file).unwrap();
        let suffix_with_extra = &file[file.len() - footer_len - 1..];
        assert!(EdgeFileFooter::decode_from_tail(file.len() as u64, suffix_with_extra).is_err());
    }

    #[test]
    fn footer_rejects_bad_xxhash() {
        let mut file = Vec::new();
        EdgeFileHeader::new("X", "Y", "Z", 0).encode(&mut file);
        let footer = EdgeFileFooter {
            sections: Vec::new(),
            key_count: 0,
            edge_count: 0,
            offsets_bits: 32,
            min_key_id: [0; 16],
            max_key_id: [0; 16],
            min_lsn: 0,
            max_lsn: 0,
            schema_version_min: 0,
            schema_version_max: 0,
        };
        footer.encode(&mut file).unwrap();
        let last = file.len() - FOOTER_TRAILER_LEN; // xxhash position
        file[last] ^= 0xff;
        let body = Bytes::from(file);
        let err = EdgeFileFooter::decode(&body).unwrap_err();
        assert!(matches!(err, Error::Corrupted { .. }));
    }

    #[test]
    fn footer_rejects_overlapping_sections() {
        let mut file = Vec::new();
        EdgeFileHeader::new("X", "Y", "Z", 0).encode(&mut file);
        // Pad some body bytes.
        file.extend_from_slice(&[0u8; 128]);
        let body_end = file.len() as u64;
        let footer = EdgeFileFooter {
            sections: vec![
                SectionEntry {
                    kind: SECTION_KEY_IDS,
                    offset: HEADER_LEN as u64,
                    length: 100,
                    codec: CODEC_NONE,
                    xxhash3_64: 0,
                    name: String::new(),
                },
                SectionEntry {
                    // overlaps the first section by 50 bytes
                    kind: SECTION_OFFSETS,
                    offset: HEADER_LEN as u64 + 50,
                    length: 20,
                    codec: CODEC_NONE,
                    xxhash3_64: 0,
                    name: String::new(),
                },
            ],
            key_count: 0,
            edge_count: 0,
            offsets_bits: 32,
            min_key_id: [0; 16],
            max_key_id: [0; 16],
            min_lsn: 0,
            max_lsn: 0,
            schema_version_min: 0,
            schema_version_max: 0,
        };
        // Sanity: body_end accommodates both sections individually.
        assert!(body_end >= HEADER_LEN as u64 + 100);
        footer.encode(&mut file).unwrap();
        let body = Bytes::from(file);
        let err = EdgeFileFooter::decode(&body).unwrap_err();
        match err {
            Error::Corrupted { detail, .. } => assert!(detail.contains("overlap")),
            other => panic!("expected Corrupted with overlap detail, got {other:?}"),
        }
    }

    #[test]
    fn footer_rejects_section_outside_file() {
        let mut file = Vec::new();
        EdgeFileHeader::new("X", "Y", "Z", 0).encode(&mut file);
        let footer = EdgeFileFooter {
            sections: vec![SectionEntry {
                kind: SECTION_KEY_IDS,
                offset: 1_000_000, // way past EOF
                length: 16,
                codec: CODEC_NONE,
                xxhash3_64: 0,
                name: String::new(),
            }],
            key_count: 0,
            edge_count: 0,
            offsets_bits: 32,
            min_key_id: [0; 16],
            max_key_id: [0; 16],
            min_lsn: 0,
            max_lsn: 0,
            schema_version_min: 0,
            schema_version_max: 0,
        };
        footer.encode(&mut file).unwrap();
        let body = Bytes::from(file);
        let err = EdgeFileFooter::decode(&body).unwrap_err();
        assert!(matches!(err, Error::Corrupted { .. }));
    }

    #[test]
    fn footer_rejects_unsorted_section_table() {
        // RFC-002 §3.2.8 requires the on-disk section table to be sorted
        // ascending by `offset`. A writer that emits it out of order must
        // be rejected even if the byte ranges are otherwise valid — the
        // reader's ranged-GET prefetch logic relies on the invariant.
        let mut file = Vec::new();
        EdgeFileHeader::new("X", "Y", "Z", 0).encode(&mut file);
        // Reserve two non-overlapping body ranges in canonical order.
        let s1_start = file.len() as u64;
        file.extend_from_slice(&[0u8; 32]);
        let s1_end = file.len() as u64;
        let s2_start = s1_end;
        file.extend_from_slice(&[0u8; 32]);
        let s2_end = file.len() as u64;

        // Build a footer that lists the *later* section first.
        let footer = EdgeFileFooter {
            sections: vec![
                SectionEntry {
                    kind: SECTION_OFFSETS,
                    offset: s2_start,
                    length: s2_end - s2_start,
                    codec: CODEC_NONE,
                    xxhash3_64: 0,
                    name: String::new(),
                },
                SectionEntry {
                    kind: SECTION_KEY_IDS,
                    offset: s1_start,
                    length: s1_end - s1_start,
                    codec: CODEC_NONE,
                    xxhash3_64: 0,
                    name: String::new(),
                },
            ],
            key_count: 0,
            edge_count: 0,
            offsets_bits: 32,
            min_key_id: [0; 16],
            max_key_id: [0; 16],
            min_lsn: 0,
            max_lsn: 0,
            schema_version_min: 0,
            schema_version_max: 0,
        };
        footer.encode(&mut file).unwrap();
        let body = Bytes::from(file);
        let err = EdgeFileFooter::decode(&body).unwrap_err();
        match err {
            Error::Corrupted { detail, .. } => assert!(detail.contains("not sorted")),
            other => panic!("expected Corrupted with not-sorted detail, got {other:?}"),
        }
    }
}
