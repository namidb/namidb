//! Identifier types: [`NodeId`], [`EdgeId`], [`NamespaceId`].
//!
//! Identifiers are 128-bit ULIDs (encoded as UUIDv7) — they sort by creation
//! time, which the storage layer exploits for LSM key ordering.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};

/// Identifier of a node in a graph.
///
/// Encoded as UUIDv7. Sorts lexicographically by creation time, which yields
/// time-clustered LSM keys and stable cross-partition ordering.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub Uuid);

impl NodeId {
 pub fn new() -> Self {
 NodeId(Uuid::now_v7())
 }
 pub fn from_uuid(u: Uuid) -> Self {
 NodeId(u)
 }
 pub fn as_bytes(&self) -> &[u8; 16] {
 self.0.as_bytes()
 }
}

impl Default for NodeId {
 fn default() -> Self {
 Self::new()
 }
}

impl fmt::Debug for NodeId {
 fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
 write!(f, "NodeId({})", self.0)
 }
}

impl fmt::Display for NodeId {
 fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
 fmt::Display::fmt(&self.0, f)
 }
}

impl FromStr for NodeId {
 type Err = Error;
 fn from_str(s: &str) -> Result<Self> {
 Uuid::parse_str(s)
 .map(NodeId)
 .map_err(|e| Error::InvalidId(format!("NodeId: {e}")))
 }
}

/// Identifier of a (source, edge_type, target) triple instance.
///
/// Encoded as UUIDv7. The triple itself is stored in the CSR adjacency
/// structures; the EdgeId is only required when edges carry their own
/// properties and we need to address a specific edge.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EdgeId(pub Uuid);

impl EdgeId {
 pub fn new() -> Self {
 EdgeId(Uuid::now_v7())
 }
 pub fn from_uuid(u: Uuid) -> Self {
 EdgeId(u)
 }
 pub fn as_bytes(&self) -> &[u8; 16] {
 self.0.as_bytes()
 }
}

impl Default for EdgeId {
 fn default() -> Self {
 Self::new()
 }
}

impl fmt::Debug for EdgeId {
 fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
 write!(f, "EdgeId({})", self.0)
 }
}

impl fmt::Display for EdgeId {
 fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
 fmt::Display::fmt(&self.0, f)
 }
}

impl FromStr for EdgeId {
 type Err = Error;
 fn from_str(s: &str) -> Result<Self> {
 Uuid::parse_str(s)
 .map(EdgeId)
 .map_err(|e| Error::InvalidId(format!("EdgeId: {e}")))
 }
}

/// Identifier of a tenant namespace.
///
/// A namespace is the unit of multi-tenancy: one namespace == one logical
/// graph == one root prefix in object storage. We use stringly-typed names
/// because tenants will pick human-readable identifiers (e.g. `acme-prod`).
#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NamespaceId(String);

impl NamespaceId {
 /// Construct a namespace, validating the name.
 ///
 /// Rules: `[a-z0-9][a-z0-9-]{0,62}` — DNS-label-ish so it can also
 /// appear in URLs and S3 prefixes without escaping.
 pub fn new(name: impl Into<String>) -> Result<Self> {
 let name = name.into();
 if name.is_empty() || name.len() > 63 {
 return Err(Error::InvalidId(format!(
 "namespace '{name}' has invalid length (1..=63)"
 )));
 }
 let first = name.chars().next().unwrap();
 if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
 return Err(Error::InvalidId(format!(
 "namespace '{name}' must start with [a-z0-9]"
 )));
 }
 for c in name.chars() {
 let ok = c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-';
 if !ok {
 return Err(Error::InvalidId(format!(
 "namespace '{name}' contains invalid char '{c}' (allowed: [a-z0-9-])"
 )));
 }
 }
 Ok(NamespaceId(name))
 }

 pub fn as_str(&self) -> &str {
 &self.0
 }
}

impl fmt::Debug for NamespaceId {
 fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
 write!(f, "NamespaceId({})", self.0)
 }
}

impl fmt::Display for NamespaceId {
 fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
 f.write_str(&self.0)
 }
}

impl FromStr for NamespaceId {
 type Err = Error;
 fn from_str(s: &str) -> Result<Self> {
 NamespaceId::new(s)
 }
}

#[cfg(test)]
mod tests {
 use super::*;

 #[test]
 fn node_id_roundtrip() {
 let id = NodeId::new();
 let s = id.to_string();
 let parsed: NodeId = s.parse().unwrap();
 assert_eq!(id, parsed);
 }

 #[test]
 fn node_id_v7_is_time_ordered() {
 let a = NodeId::new();
 std::thread::sleep(std::time::Duration::from_millis(2));
 let b = NodeId::new();
 // UUIDv7's first 48 bits are millisecond timestamp → strictly
 // ordered for samples >1ms apart.
 assert!(a < b, "{a} should sort before {b}");
 }

 #[test]
 fn namespace_validation() {
 assert!(NamespaceId::new("acme").is_ok());
 assert!(NamespaceId::new("acme-prod").is_ok());
 assert!(NamespaceId::new("acme-prod-1").is_ok());
 assert!(NamespaceId::new("1acme").is_ok());

 assert!(NamespaceId::new("").is_err());
 assert!(NamespaceId::new("ACME").is_err());
 assert!(NamespaceId::new("acme_prod").is_err());
 assert!(NamespaceId::new("-acme").is_err());
 assert!(NamespaceId::new("a".repeat(64)).is_err());
 }
}
