//! Per-SST statistics embedded in the manifest's [`SstDescriptor`].
//!
//! Defined by [RFC-002](../../../../docs/rfc/002-sst-format.md) §4.1 and §4.3.
//!
//! These types are JSON-serialisable; their byte budget is small enough to
//! embed in the manifest directly (a few hundred bytes per SST).

use serde::{Deserialize, Serialize};

use namidb_core::DataType;

/// Statistics for one declared property column inside an SST.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyColumnStats {
    pub name: String,
    pub null_count: u64,
    pub min: Option<StatScalar>,
    pub max: Option<StatScalar>,
    /// HyperLogLog++ sketch bytes (1 KiB). `None` for vector / JSON columns
    /// and for columns where the writer chose to skip NDV estimation.
    pub ndv_estimate: Option<HllSketchBytes>,
}

impl PropertyColumnStats {
    /// Build empty stats for a column that has not yet observed any value.
    pub fn empty(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            null_count: 0,
            min: None,
            max: None,
            ndv_estimate: None,
        }
    }

    /// Best-effort `DataType` derived from the recorded min/max scalar.
    ///
    /// Returns `None` for an all-NULL column where the writer never
    /// saw a non-null value to record. Schema-introspection callers
    /// can fall back to `null` / unknown in that case.
    pub fn observed_data_type(&self) -> Option<DataType> {
        let scalar = self.min.as_ref().or(self.max.as_ref())?;
        Some(match scalar {
            StatScalar::Bool(_) => DataType::Bool,
            StatScalar::Int32(_) => DataType::Int32,
            StatScalar::Int64(_) => DataType::Int64,
            StatScalar::Float32(_) => DataType::Float32,
            StatScalar::Float64(_) => DataType::Float64,
            StatScalar::Utf8(_) => DataType::Utf8,
            StatScalar::LargeUtf8(_) => DataType::LargeUtf8,
            StatScalar::Binary(_) => DataType::Binary,
            StatScalar::Date32(_) => DataType::Date32,
            StatScalar::TimestampMicrosUtc(_) => DataType::TimestampMicrosUtc,
        })
    }
}

/// Per-type literal stat value, mirroring the cases of `DataType` we expose
/// in the schema. JSON-encoded with the natural representation of each variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum StatScalar {
    Bool(bool),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Utf8(String),
    LargeUtf8(String),
    Binary(Vec<u8>),
    Date32(i32),
    TimestampMicrosUtc(i64),
}

impl StatScalar {
    /// Cheap classifier — does this scalar match the supplied `DataType`?
    /// Useful in tests; not used in hot path.
    pub fn is_compatible_with(&self, dt: &DataType) -> bool {
        matches!(
            (self, dt),
            (StatScalar::Bool(_), DataType::Bool)
                | (StatScalar::Int32(_), DataType::Int32)
                | (StatScalar::Int64(_), DataType::Int64)
                | (StatScalar::Float32(_), DataType::Float32)
                | (StatScalar::Float64(_), DataType::Float64)
                | (StatScalar::Utf8(_), DataType::Utf8)
                | (StatScalar::LargeUtf8(_), DataType::LargeUtf8)
                | (StatScalar::Binary(_), DataType::Binary)
                | (StatScalar::Date32(_), DataType::Date32)
                | (
                    StatScalar::TimestampMicrosUtc(_),
                    DataType::TimestampMicrosUtc
                )
        )
    }
}

/// Opaque HyperLogLog++ sketch bytes. We keep the inner representation
/// behind a struct so we can swap the underlying algorithm without breaking
/// serde callers.
///
/// v1.0 of the SST writer **does not yet emit sketches** — every
/// `PropertyColumnStats::ndv_estimate` is `None`. The plumbing exists so
/// the cost-based optimizer can consume sketches as soon as the
/// writer side lands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HllSketchBytes(pub Vec<u8>);

impl HllSketchBytes {
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// Log2-spaced degree histogram for an edge SST.
///
/// `counts[i]` is the number of keys whose degree lies in `[2^i, 2^(i+1))`.
/// 64 buckets cover up to degree `2^64`, far beyond anything physically
/// representable. `max_degree` and `sum_degree` are tracked separately for
/// cost-model use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DegreeHistogram {
    #[serde(with = "serde_array64_u32")]
    pub counts: [u32; 64],
    pub max_degree: u64,
    pub sum_degree: u64,
}

impl DegreeHistogram {
    pub fn empty() -> Self {
        Self {
            counts: [0; 64],
            max_degree: 0,
            sum_degree: 0,
        }
    }

    /// Record one key's degree.
    pub fn observe(&mut self, degree: u64) {
        if degree > self.max_degree {
            self.max_degree = degree;
        }
        self.sum_degree = self.sum_degree.saturating_add(degree);
        let bucket = match degree {
            0 => 0,
            d => 63 - d.leading_zeros() as usize, // floor(log2(d))
        };
        self.counts[bucket] = self.counts[bucket].saturating_add(1);
    }

    /// Sum across all buckets.
    pub fn key_count(&self) -> u64 {
        self.counts.iter().map(|&c| c as u64).sum()
    }
}

/// Helper: serde for `[u32; 64]` as a JSON array of numbers.
mod serde_array64_u32 {
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &[u32; 64], s: S) -> Result<S::Ok, S::Error> {
        v.as_ref().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u32; 64], D::Error> {
        let v = Vec::<u32>::deserialize(d)?;
        if v.len() != 64 {
            return Err(D::Error::custom(format!(
                "DegreeHistogram counts must have 64 entries, got {}",
                v.len()
            )));
        }
        let mut out = [0u32; 64];
        out.copy_from_slice(&v);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree_histogram_buckets_powers_of_two() {
        let mut h = DegreeHistogram::empty();
        h.observe(1); // bucket 0
        h.observe(2); // bucket 1
        h.observe(3); // bucket 1 (in [2, 4))
        h.observe(4); // bucket 2
        h.observe(7); // bucket 2 (in [4, 8))
        h.observe(8); // bucket 3
        h.observe(1024); // bucket 10
        assert_eq!(h.counts[0], 1);
        assert_eq!(h.counts[1], 2);
        assert_eq!(h.counts[2], 2);
        assert_eq!(h.counts[3], 1);
        assert_eq!(h.counts[10], 1);
        assert_eq!(h.max_degree, 1024);
        assert_eq!(h.sum_degree, 1 + 2 + 3 + 4 + 7 + 8 + 1024);
        assert_eq!(h.key_count(), 7);
    }

    #[test]
    fn degree_histogram_zero_goes_to_bucket_zero() {
        let mut h = DegreeHistogram::empty();
        h.observe(0);
        assert_eq!(h.counts[0], 1);
        assert_eq!(h.max_degree, 0);
        assert_eq!(h.sum_degree, 0);
    }

    #[test]
    fn degree_histogram_round_trips_through_json() {
        let mut h = DegreeHistogram::empty();
        for d in [1u64, 5, 10, 1000, 1_000_000] {
            h.observe(d);
        }
        let s = serde_json::to_string(&h).unwrap();
        let r: DegreeHistogram = serde_json::from_str(&s).unwrap();
        assert_eq!(h, r);
    }

    #[test]
    fn stat_scalar_compat_with_datatype() {
        assert!(StatScalar::Int32(5).is_compatible_with(&DataType::Int32));
        assert!(!StatScalar::Int32(5).is_compatible_with(&DataType::Int64));
        assert!(StatScalar::Utf8("x".into()).is_compatible_with(&DataType::Utf8));
        assert!(StatScalar::TimestampMicrosUtc(0).is_compatible_with(&DataType::TimestampMicrosUtc));
    }

    #[test]
    fn property_column_stats_round_trips() {
        let s = PropertyColumnStats {
            name: "age".into(),
            null_count: 3,
            min: Some(StatScalar::Int32(0)),
            max: Some(StatScalar::Int32(120)),
            ndv_estimate: None,
        };
        let j = serde_json::to_string(&s).unwrap();
        let r: PropertyColumnStats = serde_json::from_str(&j).unwrap();
        assert_eq!(s, r);
    }
}
