//! Independently decodable property pages for edge SST v1.2.
//!
//! Legacy property sections use one Arrow IPC stream (`CODEC_NONE` /
//! `CODEC_ZSTD`). Current writers use this additive envelope and codecs 2/3:
//! a small directory maps edge-row ranges to independently compressed blocks.

use xxhash_rust::xxh3::xxh3_64;

use crate::error::{Error, Result};

pub(crate) const PROPERTY_PAGES_MAGIC: [u8; 8] = *b"TGEPRP02";
pub(crate) const PROPERTY_PAGES_VERSION: u16 = 1;
pub(crate) const PROPERTY_PAGES_HEADER_LEN: usize = 32;
pub(crate) const PROPERTY_PAGE_ENTRY_LEN: usize = 44;
pub(crate) const MAX_PROPERTY_PAGE_DIRECTORY_BYTES: u64 = 64 * 1024 * 1024;
pub(crate) const MAX_PROPERTY_PAGE_DECODE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PropertyPageEntry {
    pub first_row: u64,
    pub row_count: u32,
    pub offset: u64,
    pub encoded_len: u64,
    pub decoded_len: u64,
    pub checksum: u64,
}

impl PropertyPageEntry {
    pub(crate) fn encode(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.first_row.to_le_bytes());
        out.extend_from_slice(&self.row_count.to_le_bytes());
        out.extend_from_slice(&self.offset.to_le_bytes());
        out.extend_from_slice(&self.encoded_len.to_le_bytes());
        out.extend_from_slice(&self.decoded_len.to_le_bytes());
        out.extend_from_slice(&self.checksum.to_le_bytes());
    }

    fn decode(bytes: &[u8]) -> Self {
        Self {
            first_row: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            row_count: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            offset: u64::from_le_bytes(bytes[12..20].try_into().unwrap()),
            encoded_len: u64::from_le_bytes(bytes[20..28].try_into().unwrap()),
            decoded_len: u64::from_le_bytes(bytes[28..36].try_into().unwrap()),
            checksum: u64::from_le_bytes(bytes[36..44].try_into().unwrap()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PropertyPageIndex {
    pub row_count: u64,
    pub entries: Vec<PropertyPageEntry>,
}

impl PropertyPageIndex {
    pub(crate) fn parse_prefix(bytes: &[u8], section_len: u64) -> Result<Self> {
        let prefix_len = Self::required_prefix_len(bytes, section_len)?;
        let page_count = (prefix_len - PROPERTY_PAGES_HEADER_LEN) / PROPERTY_PAGE_ENTRY_LEN;
        let row_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        if bytes.len() < prefix_len {
            return Err(corrupted("edge property-page directory truncated"));
        }

        let mut entries = Vec::with_capacity(page_count);
        let mut expected_row = 0u64;
        let mut expected_offset = prefix_len as u64;
        for raw in
            bytes[PROPERTY_PAGES_HEADER_LEN..prefix_len].chunks_exact(PROPERTY_PAGE_ENTRY_LEN)
        {
            let entry = PropertyPageEntry::decode(raw);
            if entry.row_count == 0 {
                return Err(corrupted("edge property page has zero rows"));
            }
            if entry.first_row != expected_row {
                return Err(corrupted(
                    "edge property pages do not cover rows contiguously",
                ));
            }
            if entry.offset != expected_offset {
                return Err(corrupted("edge property page payloads are not contiguous"));
            }
            if entry.decoded_len > MAX_PROPERTY_PAGE_DECODE_BYTES as u64 {
                return Err(corrupted(format!(
                    "edge property page decoded length {} exceeds {}",
                    entry.decoded_len, MAX_PROPERTY_PAGE_DECODE_BYTES
                )));
            }
            expected_row = expected_row
                .checked_add(u64::from(entry.row_count))
                .ok_or_else(|| corrupted("edge property row count exceeds u64"))?;
            expected_offset = expected_offset
                .checked_add(entry.encoded_len)
                .ok_or_else(|| corrupted("edge property payload end exceeds u64"))?;
            if expected_offset > section_len {
                return Err(corrupted("edge property page exceeds its section"));
            }
            entries.push(entry);
        }
        if expected_row != row_count {
            return Err(corrupted(format!(
                "edge property pages cover {expected_row} rows, header declares {row_count}"
            )));
        }
        if expected_offset != section_len {
            return Err(corrupted(format!(
                "edge property pages end at {expected_offset}, section length is {section_len}"
            )));
        }
        Ok(Self { row_count, entries })
    }

    pub(crate) fn required_prefix_len(bytes: &[u8], section_len: u64) -> Result<usize> {
        if bytes.len() < PROPERTY_PAGES_HEADER_LEN {
            return Err(corrupted("edge property-page header truncated"));
        }
        if bytes[..8] != PROPERTY_PAGES_MAGIC {
            return Err(corrupted("edge property-page magic mismatch"));
        }
        let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
        if version != PROPERTY_PAGES_VERSION {
            return Err(corrupted(format!(
                "unsupported edge property-page version {version}"
            )));
        }
        if bytes[10..12] != [0, 0] {
            return Err(corrupted("edge property-page flags are non-zero"));
        }
        let page_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let entry_len = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
        if entry_len != PROPERTY_PAGE_ENTRY_LEN {
            return Err(corrupted(format!(
                "edge property-page entry length {entry_len} != {PROPERTY_PAGE_ENTRY_LEN}"
            )));
        }
        if bytes[28..32] != [0, 0, 0, 0] {
            return Err(corrupted("edge property-page reserved bytes are non-zero"));
        }
        let directory_len = page_count
            .checked_mul(PROPERTY_PAGE_ENTRY_LEN)
            .ok_or_else(|| corrupted("edge property-page directory length overflows usize"))?;
        if directory_len as u64 > MAX_PROPERTY_PAGE_DIRECTORY_BYTES {
            return Err(corrupted(format!(
                "edge property-page directory requires {directory_len} bytes, above \
                 {MAX_PROPERTY_PAGE_DIRECTORY_BYTES}"
            )));
        }
        let prefix_len = PROPERTY_PAGES_HEADER_LEN
            .checked_add(directory_len)
            .ok_or_else(|| corrupted("edge property-page prefix length overflows usize"))?;
        if prefix_len as u64 > section_len {
            return Err(corrupted(
                "edge property-page directory exceeds its section",
            ));
        }
        Ok(prefix_len)
    }
}

pub(crate) fn decode_property_page(
    encoded: &[u8],
    entry: PropertyPageEntry,
    compressed: bool,
) -> Result<Vec<Option<String>>> {
    if encoded.len() as u64 != entry.encoded_len {
        return Err(corrupted("edge property page encoded length mismatch"));
    }
    if xxh3_64(encoded) != entry.checksum {
        return Err(corrupted("edge property page checksum mismatch"));
    }
    let decoded_len = usize::try_from(entry.decoded_len)
        .map_err(|_| corrupted("edge property page decoded length does not fit usize"))?;
    let decoded = if compressed {
        zstd::bulk::decompress(encoded, decoded_len)
            .map_err(|error| corrupted(format!("edge property page zstd decode: {error}")))?
    } else {
        if encoded.len() != decoded_len {
            return Err(corrupted("raw edge property page length mismatch"));
        }
        encoded.to_vec()
    };
    if decoded.len() != decoded_len {
        return Err(corrupted("edge property page decoded length mismatch"));
    }
    decode_rows(&decoded, entry.row_count)
}

fn decode_rows(bytes: &[u8], expected_rows: u32) -> Result<Vec<Option<String>>> {
    if bytes.len() < 4 {
        return Err(corrupted("edge property page row header truncated"));
    }
    let row_count = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    if row_count != expected_rows {
        return Err(corrupted("edge property page row count mismatch"));
    }
    let mut cursor = 4usize;
    let mut rows = Vec::with_capacity(row_count as usize);
    for _ in 0..row_count {
        let tag = *bytes
            .get(cursor)
            .ok_or_else(|| corrupted("edge property row tag truncated"))?;
        cursor += 1;
        match tag {
            0 => rows.push(None),
            1 => {
                let len_end = cursor
                    .checked_add(4)
                    .ok_or_else(|| corrupted("edge property row length cursor overflows"))?;
                let len_bytes = bytes
                    .get(cursor..len_end)
                    .ok_or_else(|| corrupted("edge property row length truncated"))?;
                let len = u32::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
                cursor = len_end;
                let value_end = cursor
                    .checked_add(len)
                    .ok_or_else(|| corrupted("edge property row end overflows"))?;
                let value = bytes
                    .get(cursor..value_end)
                    .ok_or_else(|| corrupted("edge property row value truncated"))?;
                rows.push(Some(
                    std::str::from_utf8(value)
                        .map_err(|error| {
                            corrupted(format!("edge property row is not utf-8: {error}"))
                        })?
                        .to_string(),
                ));
                cursor = value_end;
            }
            other => {
                return Err(corrupted(format!(
                    "edge property row has unknown tag {other}"
                )))
            }
        }
    }
    if cursor != bytes.len() {
        return Err(corrupted("edge property page has trailing bytes"));
    }
    Ok(rows)
}

fn corrupted(detail: impl Into<String>) -> Error {
    Error::Corrupted {
        path: "<edges>".into(),
        detail: detail.into(),
    }
}
