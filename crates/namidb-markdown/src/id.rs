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
/// - common Latin diacritics are folded to their base letter (`á`->`a`,
///   `ñ`->`n`, `ü`->`u`, ...), so `[[Matías]]` resolves to `matias.md`,
/// - lowercased (Unicode case folding),
/// - any run of separators (`-`, `_`, whitespace) becomes a single `-`,
/// - leading/trailing `-` trimmed.
///
/// Diacritic folding covers the Latin-1 accented letters (Western European,
/// including Spanish). Latin Extended (e.g. `ł`, `ő`) and non-Latin scripts
/// are left as-is; this is a v1 resolver, not Obsidian's full one.
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
            out.extend(fold_diacritic(ch).to_lowercase());
            prev_sep = false;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

/// Fold a Latin-1 accented letter to its base letter, leaving every other
/// character (Latin Extended, non-Latin scripts) untouched. Case is preserved
/// here; [`normalize_key`] lowercases afterwards.
fn fold_diacritic(c: char) -> char {
    match c {
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => 'A',
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => 'a',
        'Ç' => 'C',
        'ç' => 'c',
        'È' | 'É' | 'Ê' | 'Ë' => 'E',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'Ì' | 'Í' | 'Î' | 'Ï' => 'I',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        'Ñ' => 'N',
        'ñ' => 'n',
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' => 'O',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => 'o',
        'Ù' | 'Ú' | 'Û' | 'Ü' => 'U',
        'ù' | 'ú' | 'û' | 'ü' => 'u',
        'Ý' => 'Y',
        'ý' | 'ÿ' => 'y',
        other => other,
    }
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

    #[test]
    fn latin_diacritics_fold_to_base() {
        assert_eq!(normalize_key("Matías"), "matias");
        assert_eq!(normalize_key("café"), "cafe");
        assert_eq!(normalize_key("Año Nuevo"), "ano-nuevo");
        assert_eq!(normalize_key("Über"), "uber");
        assert_eq!(normalize_key("garçon"), "garcon");
        // The point of the fold: an accented wikilink resolves to the
        // unaccented filename and vice-versa.
        assert_eq!(
            stable_node_id(&normalize_key("Matías")),
            stable_node_id(&normalize_key("matias"))
        );
        // Latin Extended is left as-is: only the Latin-1 `ó` folds here, so
        // `ł` and `ź` survive (lowercased).
        assert_eq!(normalize_key("Łódź"), "łodź");
    }
}
