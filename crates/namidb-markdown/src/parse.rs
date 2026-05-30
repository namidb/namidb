//! Pure, storage-free parsing of a markdown vault into a graph shape.
//!
//! Nothing here touches the storage engine: [`parse_vault`] turns a directory
//! of `.md` files into a [`VaultGraph`] of [`ParsedNote`]s, and the rest of
//! the module is the per-file machinery it leans on. Keeping it pure makes the
//! gnarly bits (frontmatter typing, wikilink extraction, code exclusion) unit
//! testable without spinning up a `WriterSession`.
//!
//! ## v1 subset (deliberate)
//!
//! This is intentionally a reduced subset of Obsidian's behaviour, not a
//! faithful clone:
//!
//! - Wikilinks `[[Note]]`, `[[Note|alias]]`, `[[Note#heading]]`,
//!   `[[Note#^block]]`, `[[folder/Note]]` and embeds `![[Note]]` all resolve
//!   to the target note's basename and produce one `LINKS_TO` edge.
//! - Same-note refs (`[[#heading]]`) and links to non-`.md` targets are
//!   ignored for edge purposes (the latter still normalize like any name).
//! - Wikilinks inside fenced or inline code are excluded.
//! - Frontmatter is parsed as YAML; malformed frontmatter yields no
//!   properties rather than failing the note.
//! - Inline `#tags` are NOT collected (only a frontmatter `tags` list, as a
//!   plain property). Heading/block anchors are dropped, not modelled.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::Context;
use namidb_core::{NodeId, Value};
use regex::Regex;
use yaml_rust2::{Yaml, YamlLoader};

use crate::id::{normalize_key, stable_node_id};

/// Property names the storage layer manages itself; we never let frontmatter
/// write them (the flush path rejects them anyway).
const RESERVED_PROPS: [&str; 2] = ["tombstone", "lsn"];

/// One note, parsed and ready to ingest.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedNote {
    /// Stable id derived from [`ParsedNote::key`].
    pub id: NodeId,
    /// Normalized resolution key (see [`normalize_key`]).
    pub key: String,
    /// Human title (the file stem, original case).
    pub title: String,
    /// Path relative to the vault root, `/`-separated.
    pub rel_path: String,
    /// Properties to store on the node: frontmatter plus the engine-owned
    /// `title`, `path` and `body`.
    pub properties: BTreeMap<String, Value>,
    /// Normalized keys this note links to, deduplicated, in first-seen order.
    pub links: Vec<String>,
}

/// A parsed vault: every `.md` file under the root, in path order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VaultGraph {
    pub notes: Vec<ParsedNote>,
}

/// Parse every `.md` file under `dir` (recursively) into a [`VaultGraph`].
///
/// Directories whose name starts with `.` (e.g. `.obsidian`, `.git`) and a
/// top-level `_templates` directory are skipped. Files are visited in sorted
/// path order so the result is deterministic.
pub fn parse_vault(dir: &Path) -> anyhow::Result<VaultGraph> {
    let mut files: Vec<(String, String)> = Vec::new();
    walk_md(dir, dir, &mut files).with_context(|| format!("walking vault at {}", dir.display()))?;
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let notes = files
        .iter()
        .map(|(rel, raw)| parse_note(rel, raw))
        .collect();
    Ok(VaultGraph { notes })
}

fn walk_md(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            if name.starts_with('.') || name == "_templates" {
                continue;
            }
            walk_md(&path, root, out)?;
        } else if is_markdown(&path) {
            let raw =
                fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, raw));
        }
    }
    Ok(())
}

fn is_markdown(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("md") | Some("markdown")
    )
}

/// Parse one note's raw text given its vault-relative path.
pub fn parse_note(rel_path: &str, raw: &str) -> ParsedNote {
    let (frontmatter, body) = split_frontmatter(raw);

    let mut properties = match frontmatter {
        Some(yaml) => frontmatter_to_props(yaml),
        None => BTreeMap::new(),
    };

    let title = note_title(rel_path);
    let key = normalize_key(&title);

    // Engine-owned properties. `title` defers to a frontmatter title if the
    // author set one; `path` and `body` are authoritative.
    properties
        .entry("title".to_string())
        .or_insert_with(|| Value::Str(title.clone()));
    properties.insert("path".to_string(), Value::Str(rel_path.to_string()));
    properties.insert("body".to_string(), Value::Str(body.to_string()));

    let links = extract_wikilinks(body);

    ParsedNote {
        id: stable_node_id(&key),
        key,
        title,
        rel_path: rel_path.to_string(),
        properties,
        links,
    }
}

fn note_title(rel_path: &str) -> String {
    let base = rel_path.rsplit('/').next().unwrap_or(rel_path);
    base.strip_suffix(".md")
        .or_else(|| base.strip_suffix(".markdown"))
        .unwrap_or(base)
        .to_string()
}

/// Split a leading YAML frontmatter block from the body.
///
/// Returns `(Some(yaml), body)` when `raw` opens with a `---` line and has a
/// matching closing `---` line; otherwise `(None, raw)`.
pub fn split_frontmatter(raw: &str) -> (Option<&str>, &str) {
    let after_open = match raw
        .strip_prefix("---\n")
        .or_else(|| raw.strip_prefix("---\r\n"))
    {
        Some(rest) => rest,
        None => return (None, raw),
    };
    let open_len = raw.len() - after_open.len();

    let mut offset = 0usize;
    for line in after_open.split_inclusive('\n') {
        if line.trim_end() == "---" {
            let yaml = &after_open[..offset];
            let body_start = open_len + offset + line.len();
            return (Some(yaml), &raw[body_start..]);
        }
        offset += line.len();
    }
    // Opening fence with no close: not valid frontmatter, treat as body.
    (None, raw)
}

fn frontmatter_to_props(yaml: &str) -> BTreeMap<String, Value> {
    let docs = match YamlLoader::load_from_str(yaml) {
        Ok(docs) => docs,
        Err(_) => return BTreeMap::new(),
    };
    let mut props = BTreeMap::new();
    if let Some(Yaml::Hash(hash)) = docs.first() {
        for (k, v) in hash.iter() {
            let Some(key) = k.as_str() else { continue };
            if RESERVED_PROPS.contains(&key) {
                continue;
            }
            if let Some(value) = yaml_to_value(v) {
                props.insert(key.to_string(), value);
            }
        }
    }
    props
}

fn yaml_to_value(y: &Yaml) -> Option<Value> {
    match y {
        Yaml::Boolean(b) => Some(Value::Bool(*b)),
        Yaml::Integer(i) => Some(Value::I64(*i)),
        Yaml::Real(s) => s.parse::<f64>().ok().map(Value::F64),
        Yaml::String(s) => Some(Value::Str(s.clone())),
        Yaml::Array(items) => Some(Value::List(
            items.iter().filter_map(yaml_to_value).collect(),
        )),
        Yaml::Hash(hash) => {
            let mut map = BTreeMap::new();
            for (k, v) in hash.iter() {
                if let (Some(ks), Some(vv)) = (k.as_str(), yaml_to_value(v)) {
                    map.insert(ks.to_string(), vv);
                }
            }
            Some(Value::Map(map))
        }
        // Null, BadValue, Alias and any future variant carry no usable scalar.
        _ => None,
    }
}

/// Extract wikilink targets from a note body as normalized keys, deduplicated
/// in first-seen order. Links inside code are excluded.
pub fn extract_wikilinks(body: &str) -> Vec<String> {
    let masked = mask_code(body);
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for caps in wikilink_regex().captures_iter(&masked) {
        if let Some(key) = link_target_key(&caps[2]) {
            if seen.insert(key.clone()) {
                out.push(key);
            }
        }
    }
    out
}

fn wikilink_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Group 1: optional embed marker. Group 2: the inner target text.
    RE.get_or_init(|| Regex::new(r"(!?)\[\[([^\[\]\r\n]+?)\]\]").expect("valid wikilink regex"))
}

/// Reduce a wikilink's inner text to a normalized note key.
///
/// `Note|alias` -> `note`, `Note#heading` -> `note`, `folder/Note` -> `note`.
/// Returns `None` for same-note refs like `#heading` that have no note part.
fn link_target_key(inner: &str) -> Option<String> {
    let before_alias = inner.split('|').next().unwrap_or(inner);
    let before_anchor = before_alias.split('#').next().unwrap_or(before_alias);
    let trimmed = before_anchor.trim();
    if trimmed.is_empty() {
        return None;
    }
    let base = trimmed.rsplit('/').next().unwrap_or(trimmed);
    let key = normalize_key(base);
    if key.is_empty() {
        None
    } else {
        Some(key)
    }
}

/// Replace fenced and inline code spans with spaces so wikilink scanning never
/// matches inside an example. Byte length is preserved; pulldown-cmark gives
/// codepoint-aligned ranges so the result stays valid UTF-8.
fn mask_code(body: &str) -> String {
    use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

    let mut buf = body.as_bytes().to_vec();
    let mut code_block_start: Option<usize> = None;

    for (event, range) in Parser::new_ext(body, Options::empty()).into_offset_iter() {
        match event {
            Event::Start(Tag::CodeBlock(_)) => code_block_start = Some(range.start),
            Event::End(TagEnd::CodeBlock) => {
                if let Some(start) = code_block_start.take() {
                    mask(&mut buf, start, range.end);
                }
            }
            Event::Code(_) => mask(&mut buf, range.start, range.end),
            _ => {}
        }
    }

    String::from_utf8_lossy(&buf).into_owned()
}

fn mask(buf: &mut [u8], start: usize, end: usize) {
    let start = start.min(buf.len());
    let end = end.min(buf.len());
    for byte in &mut buf[start..end] {
        *byte = b' ';
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter_split_and_typing() {
        let raw = "---\ntitle: Hello\ncount: 3\nratio: 1.5\ndone: true\ntags:\n  - a\n  - b\n---\nbody text\n";
        let (fm, body) = split_frontmatter(raw);
        assert_eq!(body, "body text\n");
        let props = frontmatter_to_props(fm.unwrap());
        assert_eq!(props.get("title"), Some(&Value::Str("Hello".into())));
        assert_eq!(props.get("count"), Some(&Value::I64(3)));
        assert_eq!(props.get("ratio"), Some(&Value::F64(1.5)));
        assert_eq!(props.get("done"), Some(&Value::Bool(true)));
        assert_eq!(
            props.get("tags"),
            Some(&Value::List(vec![
                Value::Str("a".into()),
                Value::Str("b".into())
            ]))
        );
    }

    #[test]
    fn no_frontmatter_is_all_body() {
        let raw = "# Heading\n[[Other]]\n";
        let (fm, body) = split_frontmatter(raw);
        assert!(fm.is_none());
        assert_eq!(body, raw);
    }

    #[test]
    fn malformed_frontmatter_yields_no_props_but_keeps_body() {
        let raw = "---\n:\n::: not yaml\n---\nbody\n";
        let note = parse_note("Bad.md", raw);
        assert_eq!(
            note.properties.get("body"),
            Some(&Value::Str("body\n".into()))
        );
    }

    #[test]
    fn wikilink_variants() {
        let body = "Links: [[Alpha]], [[Beta|the beta]], [[Gamma#section]], [[notes/Delta]], ![[Epsilon]] and [[#self]].";
        let links = extract_wikilinks(body);
        assert_eq!(links, vec!["alpha", "beta", "gamma", "delta", "epsilon"]);
    }

    #[test]
    fn wikilinks_in_code_are_ignored() {
        let body = "Real [[Alpha]].\n\n```\nnot a link [[Fenced]]\n```\n\nInline `[[Inline]]` ignored, [[Beta]] kept.";
        let links = extract_wikilinks(body);
        assert_eq!(links, vec!["alpha", "beta"]);
    }

    #[test]
    fn duplicate_links_collapse_preserving_order() {
        let body = "[[B]] then [[A]] then [[b]] again";
        assert_eq!(extract_wikilinks(body), vec!["b", "a"]);
    }

    #[test]
    fn parse_note_sets_engine_props_and_stable_id() {
        let note = parse_note(
            "dir/User Role.md",
            "---\nrole: founder\n---\nsee [[Project]]\n",
        );
        assert_eq!(note.key, "user-role");
        assert_eq!(note.title, "User Role");
        assert_eq!(note.id, stable_node_id("user-role"));
        assert_eq!(
            note.properties.get("role"),
            Some(&Value::Str("founder".into()))
        );
        assert_eq!(
            note.properties.get("path"),
            Some(&Value::Str("dir/User Role.md".into()))
        );
        assert_eq!(note.links, vec!["project"]);
    }
}
