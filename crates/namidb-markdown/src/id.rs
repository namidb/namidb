//! Deterministic node identity for vault notes.
//!
//! Obsidian resolves `[[wikilinks]]` by note *name*, not by a stored id, and
//! a link can point at a note that is parsed later (or never). To make the
//! ingest order-independent and idempotent, a note's [`NodeId`] is derived
//! from its normalized key rather than minted from a clock.
//!
//! We hash the normalized key with BLAKE3 and use the first 16 bytes as the
//! UUID payload. Re-ingesting the same vault yields the same ids, so an
//! upsert overwrites in place instead of duplicating the graph.

use namidb_core::NodeId;
use uuid::Uuid;

/// Derive a stable [`NodeId`] from a note's normalized key.
///
/// Deterministic: the same key always maps to the same id, across runs and
/// across machines. Not a v7 UUID (those are time-ordered and random); the
/// version/variant nibbles are left as whatever BLAKE3 produced, which is
/// fine because the storage layer only ever treats a [`NodeId`] as an opaque
/// 128-bit key.
pub fn stable_node_id(normalized_key: &str) -> NodeId {
    let digest = blake3::hash(normalized_key.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest.as_bytes()[..16]);
    NodeId::from_uuid(Uuid::from_bytes(bytes))
}

/// Normalize a note name or wikilink target into a resolution key.
///
/// The goal is that every spelling that a human would consider "the same
/// note" collapses to one key, so `[[User Role]]`, `[[user-role]]` and a file
/// named `user_role.md` all resolve to the same node. Rules:
///
/// - lowercase (ASCII),
/// - any run of separators (`-`, `_`, whitespace) becomes a single `-`,
/// - leading/trailing `-` trimmed.
///
/// This deliberately ignores accents and unicode case folding; it is a v1
/// subset, not Obsidian's full resolver.
pub fn normalize_key(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_sep = false;
    for ch in name.trim().chars() {
        if ch == '-' || ch == '_' || ch.is_whitespace() {
            if !out.is_empty() && !prev_sep {
                out.push('-');
                prev_sep = true;
            }
        } else {
            out.extend(ch.to_lowercase());
            prev_sep = false;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_is_deterministic() {
        assert_eq!(stable_node_id("user-role"), stable_node_id("user-role"));
        assert_ne!(stable_node_id("user-role"), stable_node_id("project"));
    }

    #[test]
    fn kebab_snake_and_spaces_collapse() {
        assert_eq!(normalize_key("User Role"), "user-role");
        assert_eq!(normalize_key("user_role"), "user-role");
        assert_eq!(normalize_key("user-role"), "user-role");
        assert_eq!(normalize_key("  User   Role  "), "user-role");
        // The kebab/snake mix the engine's own memory/ dir exercises.
        assert_eq!(
            stable_node_id(&normalize_key("user_role")),
            stable_node_id(&normalize_key("user-role"))
        );
    }
}
