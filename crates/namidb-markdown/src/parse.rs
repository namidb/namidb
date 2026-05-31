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
//!   `[[Note#^block]]`, `[[folder/Note]]` resolve to the target note's basename
//!   and produce a `LINKS_TO` edge. Embeds `![[Note]]` resolve the same way but
//!   produce a distinct `EMBEDS` edge instead.
//! - Standard markdown links `[text](note.md)` to a local `.md`/`.markdown`
//!   file also produce a `LINKS_TO` edge (basename-resolved, percent-decoded).
//!   External URLs, mail/other schemes, bare anchors and non-markdown files
//!   are ignored. Markdown image embeds (`![]()`) are not treated as links.
//! - A frontmatter property whose value *is* a `[[Note]]` wikilink (or a list
//!   of them, e.g. `up: "[[Parent]]"`) also produces a `LINKS_TO` edge. A value
//!   that merely contains `[[...]]` inside prose or code does not.
//! - Same-note refs (`[[#heading]]`) carry no target.
//! - Links inside fenced or inline code are excluded.
//! - Frontmatter is parsed as YAML; malformed frontmatter yields no
//!   properties rather than failing the note. A duplicate top-level key (which
//!   YAML rejects outright) is recovered last-wins instead of dropping every
//!   key with it.
//! - Inline `#tags` and a frontmatter `tags` list are merged into one
//!   deduplicated `tags` property (nested tags `#area/topic` kept; `#123` with
//!   no letters is not a tag). Tags are a property, not yet `:Tag` nodes.
//! - Heading/block anchors are dropped, not modelled.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::Context;
use namidb_core::schema::is_reserved_property_name;
use namidb_core::{NodeId, Value};
use regex::Regex;
use yaml_rust2::{Yaml, YamlLoader};

use crate::id::{normalize_key, stable_node_id};

/// The loader's own engine-owned property names, dropped from frontmatter so
/// an author cannot override them (they are set below: `key` from the file
/// stem, `content_hash` from the raw bytes). The engine-reserved names
/// (`node_id`/`tombstone`/`lsn` and the `__`/`prop_` prefixes) are dropped
/// separately via [`is_reserved_property_name`] so this crate cannot drift
/// from the engine's reserved set. All are dropped here because nothing on the
/// write path re-validates property names, so an injected reserved key would
/// otherwise be stored verbatim and shadow an engine-managed value.
const RESERVED_KEYS: [&str; 2] = ["key", "content_hash"];

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
    /// `title`, `path`, `body` and `key`.
    pub properties: BTreeMap<String, Value>,
    /// Normalized keys this note links to (non-embed body wikilinks + markdown
    /// links + whole-value frontmatter wikilinks), deduplicated, body links
    /// first. Becomes `:LINKS_TO` edges.
    pub links: Vec<String>,
    /// Normalized keys this note embeds (`![[X]]`), deduplicated. Becomes
    /// `:EMBEDS` edges, kept separate from links.
    pub embeds: Vec<String>,
    /// String tags on this note (frontmatter `tags` strings + inline `#tags`),
    /// deduplicated. Each becomes a `:Tag` node linked by a `:TAGGED` edge.
    pub tags: Vec<String>,
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

    // Capture wikilinks living in frontmatter values now, while `properties`
    // holds only frontmatter keys, so the engine-owned `body` (added below) is
    // not re-scanned and double-counted.
    let frontmatter_links = extract_frontmatter_links(&properties);

    let title = note_title(rel_path);
    let key = normalize_key(&title);

    // Engine-owned properties. `title` defers to a frontmatter title only if
    // the author set one *as a string*; a non-string `title:` (e.g. a bare
    // `title: 2025` that YAML types as an integer) falls back to the file stem
    // so `title` is always a string and readers can `as_str()` it. `path`,
    // `body` and `key` are authoritative. `key` is the normalized resolution
    // key, stored so a query can resolve a note by name (kebab/snake/spaces)
    // without re-implementing the normalization.
    if !matches!(properties.get("title"), Some(Value::Str(_))) {
        properties.insert("title".to_string(), Value::Str(title.clone()));
    }
    properties.insert("path".to_string(), Value::Str(rel_path.to_string()));
    properties.insert("body".to_string(), Value::Str(body.to_string()));
    properties.insert("key".to_string(), Value::Str(key.clone()));
    // Content hash of the raw file bytes (frontmatter + body), so an
    // incremental sync can tell a changed note from an unchanged one by
    // comparing this against the stored value without re-parsing. Hash the raw
    // input rather than just `body` so a frontmatter-only edit (a tag or link
    // change) still registers as a change.
    properties.insert(
        "content_hash".to_string(),
        Value::Str(blake3::hash(raw.as_bytes()).to_hex().to_string()),
    );

    // Fold inline `#tags` into the `tags` property. Only acts when there are
    // inline tags, and never clobbers a frontmatter `tags` value that is not a
    // string or list (e.g. a map): an author's value is preserved verbatim.
    // Tags are kept as written (case-sensitive). Existing list items are kept
    // as-is (including non-string ones); inline tags are appended unless an
    // equal string is already present.
    let inline_tags = extract_tags(body);
    if !inline_tags.is_empty() {
        let merged: Option<Value> = match properties.get("tags") {
            None => Some(Value::List(
                inline_tags.iter().map(|t| Value::Str(t.clone())).collect(),
            )),
            Some(Value::List(items)) => {
                let present: HashSet<&str> = items
                    .iter()
                    .filter_map(|v| match v {
                        Value::Str(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .collect();
                let mut merged = items.clone();
                for tag in &inline_tags {
                    if !present.contains(tag.as_str()) {
                        merged.push(Value::Str(tag.clone()));
                    }
                }
                Some(Value::List(merged))
            }
            Some(Value::Str(existing)) => {
                let mut merged = vec![Value::Str(existing.clone())];
                for tag in &inline_tags {
                    if tag != existing {
                        merged.push(Value::Str(tag.clone()));
                    }
                }
                Some(Value::List(merged))
            }
            // Map / number / bool / etc: leave the author's value untouched.
            Some(_) => None,
        };
        if let Some(value) = merged {
            properties.insert("tags".to_string(), value);
        }
    }

    // Body links first (in document order), then any frontmatter links not
    // already present, so both kinds become `LINKS_TO` edges with the body's
    // ordering preserved.
    let mut links = extract_links(body);
    let mut seen: HashSet<String> = links.iter().cloned().collect();
    for key in frontmatter_links {
        if seen.insert(key.clone()) {
            links.push(key);
        }
    }
    let embeds = extract_embeds(body);

    // The note's string tags (for `:Tag` nodes), taken from the final `tags`
    // property so they stay consistent with what is stored/displayed, then
    // deduplicated (frontmatter may list the same tag twice) in first-seen
    // order so each note links to a tag at most once.
    let mut tags: Vec<String> = match properties.get("tags") {
        Some(Value::List(items)) => items
            .iter()
            .filter_map(|v| match v {
                Value::Str(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        Some(Value::Str(s)) => vec![s.clone()],
        _ => Vec::new(),
    };
    {
        let mut seen = HashSet::new();
        tags.retain(|t| seen.insert(t.clone()));
    }

    ParsedNote {
        id: stable_node_id(&key),
        key,
        title,
        rel_path: rel_path.to_string(),
        properties,
        links,
        embeds,
        tags,
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
        // A duplicate top-level key makes yaml_rust2 reject the WHOLE document,
        // which would silently drop every other (valid) key with it. Recover
        // the common authoring slip (e.g. two `tags:`) by keeping the last
        // value for each duplicated top-level key and re-parsing once, the way
        // editors like Obsidian resolve duplicate properties. Any other
        // malformed YAML still yields no properties, as documented.
        Err(e) if e.to_string().contains("duplicated key") => {
            match YamlLoader::load_from_str(&dedup_top_level_keys(yaml)) {
                Ok(docs) => docs,
                Err(_) => return BTreeMap::new(),
            }
        }
        Err(_) => return BTreeMap::new(),
    };
    let mut props = BTreeMap::new();
    if let Some(Yaml::Hash(hash)) = docs.first() {
        for (k, v) in hash.iter() {
            let Some(key) = k.as_str() else { continue };
            if RESERVED_KEYS.contains(&key) || is_reserved_property_name(key) {
                continue;
            }
            if let Some(value) = yaml_to_value(v) {
                props.insert(key.to_string(), value);
            }
        }
    }
    props
}

/// Best-effort recovery for frontmatter rejected because a top-level key is
/// duplicated. Splits the document into top-level entries (a column-0 `key:`
/// line plus its indented continuation lines) and keeps only the LAST entry
/// for each key, preserving order otherwise. Only the top level is considered;
/// nested duplicates and exotic (quoted/complex) keys are left to the strict
/// re-parse, which drops to no properties if it still fails, so this never
/// makes a note that already parsed worse.
fn dedup_top_level_keys(yaml: &str) -> String {
    // Group lines: a new group begins at each top-level `key:` line; every
    // other line (indented, blank, comment, list item) attaches to the
    // current group.
    let mut groups: Vec<(Option<String>, String)> = Vec::new();
    for line in yaml.split_inclusive('\n') {
        match top_level_key(line) {
            Some(key) => groups.push((Some(key), line.to_string())),
            None => match groups.last_mut() {
                Some(group) => group.1.push_str(line),
                None => groups.push((None, line.to_string())),
            },
        }
    }
    // Index of the last group carrying each key; earlier duplicates are dropped.
    let mut last: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, (key, _)) in groups.iter().enumerate() {
        if let Some(k) = key {
            last.insert(k.as_str(), i);
        }
    }
    let mut out = String::with_capacity(yaml.len());
    for (i, (key, text)) in groups.iter().enumerate() {
        let earlier_dup = key.as_deref().is_some_and(|k| last.get(k) != Some(&i));
        if !earlier_dup {
            out.push_str(text);
        }
    }
    out
}

/// The simple key of a top-level `key: ...` line (column 0, unquoted), or
/// `None` if the line is not a top-level mapping entry (indented, blank, a
/// comment, a list item, or with no `key:` separator). `:` only counts as a
/// separator when it ends the line or is followed by whitespace, so a scalar
/// like `https://x` is not mistaken for a key.
fn top_level_key(line: &str) -> Option<String> {
    let first = line.chars().next()?;
    if first.is_whitespace() || first == '#' || first == '-' {
        return None;
    }
    let content = line.trim_end_matches(['\n', '\r']);
    let colon = content.find(':')?;
    let after = &content[colon + 1..];
    if !after.is_empty() && !after.starts_with(char::is_whitespace) {
        return None;
    }
    let key = content[..colon].trim();
    (!key.is_empty()).then(|| key.to_string())
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

/// Split wikilink targets into `(links, embeds)` by the `!` embed marker, each
/// deduplicated in first-seen order. `[[X]]` is a link, `![[X]]` an embed; a
/// target both linked and embedded in the same note appears in both lists.
fn classify_wikilinks(body: &str) -> (Vec<String>, Vec<String>) {
    let masked = mask_code(body);
    let (mut links, mut embeds) = (Vec::new(), Vec::new());
    let (mut seen_links, mut seen_embeds) = (HashSet::new(), HashSet::new());
    for caps in wikilink_regex().captures_iter(&masked) {
        if let Some(key) = link_target_key(&caps[2]) {
            if caps[1].is_empty() {
                if seen_links.insert(key.clone()) {
                    links.push(key);
                }
            } else if seen_embeds.insert(key.clone()) {
                embeds.push(key);
            }
        }
    }
    (links, embeds)
}

/// Note targets a body links to (not embeds), as normalized keys, deduplicated
/// in first-seen order. Combines non-embed `[[wikilinks]]` with standard
/// markdown links `[text](note.md)` to a local `.md`/`.markdown` file. External
/// URLs, mail/other schemes, bare anchors and non-note files are ignored.
pub fn extract_links(body: &str) -> Vec<String> {
    let (links, _embeds) = classify_wikilinks(body);
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for key in links.into_iter().chain(extract_markdown_links(body)) {
        if seen.insert(key.clone()) {
            out.push(key);
        }
    }
    out
}

/// Note targets a body embeds (`![[X]]`), as normalized keys, deduplicated in
/// first-seen order.
pub fn extract_embeds(body: &str) -> Vec<String> {
    classify_wikilinks(body).1
}

/// A frontmatter property value that *is* a single `[[wikilink]]`, reduced to
/// its normalized key. The whole (trimmed) value must be one plain wikilink, so
/// a prose or code value that merely contains `[[...]]` (a description, a code
/// snippet like `arr[[i]]`) does not become a spurious edge. Embeds (`![[X]]`)
/// and aliases/anchors are handled by [`link_target_key`].
fn frontmatter_link_target(value: &str) -> Option<String> {
    let inner = value.trim().strip_prefix("[[")?.strip_suffix("]]")?;
    // Reject anything but a lone wikilink: nested/adjacent brackets or newlines
    // mean this is not a clean single link target.
    if inner.contains("[[") || inner.contains("]]") || inner.contains(['\r', '\n']) {
        return None;
    }
    link_target_key(inner)
}

/// Wikilink targets found in frontmatter property values, as normalized keys in
/// first-seen order. Obsidian treats a property whose value is a `[[note]]` as a
/// link, so these become `LINKS_TO` edges alongside body links. Only string and
/// string-list values are scanned, and only values that are wholly one wikilink
/// count (see [`frontmatter_link_target`]). The `tags` property is skipped: a
/// tag is never a link target.
fn extract_frontmatter_links(props: &BTreeMap<String, Value>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let add = |s: &str, seen: &mut HashSet<String>, out: &mut Vec<String>| {
        if let Some(key) = frontmatter_link_target(s) {
            if seen.insert(key.clone()) {
                out.push(key);
            }
        }
    };
    for (name, value) in props {
        // Skip the engine-owned/special properties: `tags` is its own concern,
        // and `title`/`path`/`body` are display/content fields the loader sets,
        // not link fields, so a `title: "[[X]]"` must not forge an edge. (`key`
        // and `content_hash` were already dropped by `frontmatter_to_props`.)
        if matches!(name.as_str(), "tags" | "title" | "path" | "body") {
            continue;
        }
        match value {
            Value::Str(s) => add(s, &mut seen, &mut out),
            Value::List(items) => {
                for item in items {
                    if let Value::Str(s) = item {
                        add(s, &mut seen, &mut out);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Markdown-link targets (`[text](note.md)`) that resolve to a local note, as
/// normalized keys in document order. Uses the CommonMark parser, so links
/// inside code are not emitted; images (`![]()`) are not links and are skipped.
fn extract_markdown_links(body: &str) -> Vec<String> {
    use pulldown_cmark::{Event, Options, Parser, Tag};
    let mut out = Vec::new();
    for event in Parser::new_ext(body, Options::empty()) {
        if let Event::Start(Tag::Link { dest_url, .. }) = event {
            if let Some(key) = md_link_target_key(&dest_url) {
                out.push(key);
            }
        }
    }
    out
}

/// Reduce a markdown link destination to a normalized note key, or `None` if it
/// is not a link to a local `.md`/`.markdown` note (external URL, mail/other
/// scheme, bare anchor, or non-markdown file).
fn md_link_target_key(dest: &str) -> Option<String> {
    let dest = dest.trim();
    // Empty, same-page anchor, or protocol-relative URL (`//host/...`).
    if dest.is_empty() || dest.starts_with('#') || dest.starts_with("//") {
        return None;
    }
    // Absolute URL (`http://`, `https://`, ...) — not a note.
    if dest.contains("://") {
        return None;
    }
    // A URI scheme before the first `:` (RFC-3986:
    // `ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )`) — `mailto:`, `tel:`, ...
    if let Some(colon) = dest.find(':') {
        let scheme = &dest[..colon];
        if scheme
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic())
            && scheme
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
        {
            return None;
        }
    }
    // Strip the literal fragment/query, take the basename, then percent-decode
    // it (decoding last so an encoded `%2F`/`%23` becomes data in the stem
    // rather than a structural separator we'd split on).
    let path = dest.split(['#', '?']).next().unwrap_or(dest);
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let decoded = percent_decode(base);
    let lower = decoded.to_ascii_lowercase();
    let stem = if lower.ends_with(".md") {
        &decoded[..decoded.len() - 3]
    } else if lower.ends_with(".markdown") {
        &decoded[..decoded.len() - 9]
    } else {
        return None; // not a markdown note link
    };
    // Any path/scheme separator surviving into the stem (e.g. `1:foo`, or a
    // decoded `%2F`/`%23`) means this is not a clean single-note name; a real
    // note key can never contain these, so it would only ever dangle.
    if stem.contains([':', '/', '\\', '#', '?']) {
        return None;
    }
    let key = normalize_key(stem);
    if key.is_empty() {
        None
    } else {
        Some(key)
    }
}

/// Minimal percent-decoder for link destinations (`%20` -> space, etc.).
/// Leaves malformed escapes untouched.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                out.push(h * 16 + l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn wikilink_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Group 1: optional embed marker. Group 2: the inner target text.
    RE.get_or_init(|| Regex::new(r"(!?)\[\[([^\[\]\r\n]+?)\]\]").expect("valid wikilink regex"))
}

/// Inline `#tags` in a note body, in first-seen order, deduplicated. Excludes
/// matches inside code (via [`mask_code`]), requires at least one non-digit
/// character (so `#123` is not a tag), and keeps nested tags (`#area/topic`).
fn extract_tags(body: &str) -> Vec<String> {
    let masked = mask_code(body);
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for caps in tag_regex().captures_iter(&masked) {
        let tag = caps[1].to_string();
        if seen.insert(tag.clone()) {
            out.push(tag);
        }
    }
    out
}

fn tag_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Preceded by start-of-text or whitespace (so `#` mid-word, in URLs, or a
    // heading's `# ` are excluded); then a tag of letters/digits/`_`/`-`/`/`
    // with at least one non-digit. `\s` matches newlines, covering line starts.
    RE.get_or_init(|| {
        Regex::new(r"(?:^|\s)#([\p{L}\p{N}_/-]*[\p{L}_/-][\p{L}\p{N}_/-]*)")
            .expect("valid tag regex")
    })
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
    fn reserved_property_names_are_dropped_from_frontmatter() {
        // Engine-reserved bare names and the `__`/`prop_` prefixes, plus the
        // loader's own `key`, must never be stored from frontmatter: nothing
        // re-validates names on the write path, so an injected reserved key
        // would otherwise shadow an engine-managed column.
        let note = parse_note(
            "N.md",
            "---\nnode_id: 999\ntombstone: true\nlsn: 7\nkey: hijack\n\
             __x: 1\nprop_y: 2\nrole: founder\n---\nbody\n",
        );
        for reserved in ["node_id", "tombstone", "lsn", "__x", "prop_y"] {
            assert!(
                !note.properties.contains_key(reserved),
                "reserved `{reserved}` must not be stored from frontmatter"
            );
        }
        // `key` is the engine-owned resolution key (the stem), not the
        // frontmatter's `hijack` value.
        assert_eq!(note.properties.get("key"), Some(&Value::Str("n".into())));
        // A normal property still passes through.
        assert_eq!(
            note.properties.get("role"),
            Some(&Value::Str("founder".into()))
        );
    }

    #[test]
    fn content_hash_tracks_raw_bytes_and_is_engine_owned() {
        // Identical bytes hash identically; the hash is stored for sync.
        let a = parse_note("N.md", "---\ntags: [x]\n---\nbody\n");
        let b = parse_note("N.md", "---\ntags: [x]\n---\nbody\n");
        assert!(
            matches!(a.properties.get("content_hash"), Some(Value::Str(_))),
            "content_hash is stored"
        );
        assert_eq!(
            a.properties.get("content_hash"),
            b.properties.get("content_hash")
        );

        // A frontmatter-only edit (tag change, body identical) changes the
        // hash, since we hash the raw bytes and not just the body.
        let c = parse_note("N.md", "---\ntags: [y]\n---\nbody\n");
        assert_ne!(
            a.properties.get("content_hash"),
            c.properties.get("content_hash"),
            "a frontmatter edit changes the content hash"
        );

        // An author cannot inject content_hash via frontmatter.
        let injected = parse_note("N.md", "---\ncontent_hash: deadbeef\n---\nbody\n");
        assert_ne!(
            injected.properties.get("content_hash"),
            Some(&Value::Str("deadbeef".into())),
            "frontmatter content_hash is dropped"
        );
    }

    #[test]
    fn non_string_frontmatter_title_falls_back_to_stem() {
        // A bare `title: 2025` types as an integer in YAML; storing it would
        // break readers that expect `title` to be a string, so fall back to
        // the file stem.
        let note = parse_note("My Note.md", "---\ntitle: 2025\n---\nbody\n");
        assert_eq!(
            note.properties.get("title"),
            Some(&Value::Str("My Note".into()))
        );
        // A string frontmatter title is still honored.
        let custom = parse_note("My Note.md", "---\ntitle: Custom\n---\nbody\n");
        assert_eq!(
            custom.properties.get("title"),
            Some(&Value::Str("Custom".into()))
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
        // Genuinely broken YAML (not a duplicate key) still drops to no
        // user properties, so the recovery path stays scoped.
        assert!(!note.properties.contains_key("role"));
    }

    #[test]
    fn duplicate_top_level_key_recovers_last_wins_not_total_loss() {
        // A doubled `tags:` makes yaml_rust2 reject the whole document; without
        // recovery `title` and `role` would vanish too. Recovery keeps the last
        // `tags` (flow style here) and preserves the sibling keys.
        let note = parse_note(
            "N.md",
            "---\ntitle: Kept\nrole: founder\ntags: [a]\ntags: [b, c]\n---\nbody\n",
        );
        assert_eq!(
            note.properties.get("title"),
            Some(&Value::Str("Kept".into()))
        );
        assert_eq!(
            note.properties.get("role"),
            Some(&Value::Str("founder".into()))
        );
        assert_eq!(
            note.properties.get("tags"),
            Some(&Value::List(vec![
                Value::Str("b".into()),
                Value::Str("c".into())
            ])),
            "last duplicate wins"
        );
    }

    #[test]
    fn duplicate_top_level_key_recovers_block_style_value() {
        // The block (indented list) form is the common Obsidian style: the
        // grouping must attach the indented items to their key and still drop
        // the earlier duplicate.
        let note = parse_note(
            "N.md",
            "---\ntags:\n  - a\nrole: x\ntags:\n  - b\n  - c\n---\nbody\n",
        );
        assert_eq!(note.properties.get("role"), Some(&Value::Str("x".into())));
        assert_eq!(
            note.properties.get("tags"),
            Some(&Value::List(vec![
                Value::Str("b".into()),
                Value::Str("c".into())
            ]))
        );
    }

    #[test]
    fn inline_tags_extracted_excluding_headings_and_code() {
        let body = "# Heading is not a tag\n\nText with #rust and #area/db, #123 is not.\n\n\
                    ```\n#fenced\n```\nInline `#code` ignored. URL https://x/#frag too.";
        assert_eq!(extract_tags(body), vec!["rust", "area/db"]);
    }

    #[test]
    fn parse_note_merges_frontmatter_and_inline_tags() {
        let note = parse_note(
            "N.md",
            "---\ntags: [project, rust]\n---\nbody #rust and #new\n",
        );
        let tags: Vec<&str> = match note.properties.get("tags") {
            Some(Value::List(items)) => items
                .iter()
                .filter_map(|v| match v {
                    Value::Str(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect(),
            other => panic!("expected a tags list, got {other:?}"),
        };
        // Frontmatter first, inline merged, `rust` deduplicated.
        assert_eq!(tags, vec!["project", "rust", "new"]);
    }

    #[test]
    fn no_tags_means_no_tags_property() {
        let note = parse_note("N.md", "plain body, no hashes\n");
        assert!(!note.properties.contains_key("tags"));
    }

    #[test]
    fn tag_merge_keeps_non_string_frontmatter_list_items() {
        let note = parse_note("N.md", "---\ntags: [2025, ok]\n---\nbody #rust\n");
        match note.properties.get("tags") {
            Some(Value::List(items)) => {
                assert!(items.contains(&Value::I64(2025)), "numeric item preserved");
                assert!(items.contains(&Value::Str("ok".into())));
                assert!(
                    items.contains(&Value::Str("rust".into())),
                    "inline appended"
                );
            }
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[test]
    fn tag_merge_does_not_clobber_unmergeable_frontmatter() {
        // A map-shaped `tags:` is unusual but must be preserved, not replaced.
        let note = parse_note("N.md", "---\ntags:\n  weird: 1\n---\nbody #rust\n");
        assert!(
            matches!(note.properties.get("tags"), Some(Value::Map(_))),
            "map-shaped tags preserved verbatim"
        );
    }

    #[test]
    fn wikilink_variants() {
        let body = "Links: [[Alpha]], [[Beta|the beta]], [[Gamma#section]], [[notes/Delta]], ![[Epsilon]] and [[#self]].";
        let links = extract_wikilinks(body);
        assert_eq!(links, vec!["alpha", "beta", "gamma", "delta", "epsilon"]);
    }

    #[test]
    fn embeds_are_separated_from_links() {
        let body = "link [[A]], embed ![[B]], md [C](C.md), embed-md image ![](D.md)";
        // Links: non-embed wikilink A + markdown link C. Image (`![]()`) is not
        // a note link. Embed B goes to embeds only.
        assert_eq!(extract_links(body), vec!["a", "c"]);
        assert_eq!(extract_embeds(body), vec!["b"]);
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
    fn markdown_links_to_notes_become_targets() {
        let body = "See [Alpha](Alpha.md), [Beta](notes/Beta.markdown), \
                    [spaced](User%20Role.md), [up](../Gamma.md#section).";
        assert_eq!(
            extract_markdown_links(body),
            vec!["alpha", "beta", "user-role", "gamma"]
        );
    }

    #[test]
    fn markdown_non_note_links_are_ignored() {
        let body = "[site](https://example.com), [mail](mailto:a@b.com), \
                    [img](pic.png), [anchor](#section), [doc](report.pdf), \
                    ![embed](Note.md).";
        // Only real note links count; the image (`![]()`) is not a link.
        assert!(extract_markdown_links(body).is_empty());
    }

    #[test]
    fn markdown_links_reject_scheme_and_structural_garbage() {
        // None of these reduce to a clean note name, so none must produce an
        // edge (each would otherwise leave a `:`/`/`/`#` in the key that can
        // never match a stored note).
        let body = "[a](//cdn.example.com/x.md) [b](1:foo.md) [c](a-b:thing.md) \
                    [d](x.y:thing.md) [e](a%2Fb.md) [f](note%23x.md) [g](tel:5550100)";
        assert!(extract_markdown_links(body).is_empty());
    }

    #[test]
    fn markdown_links_in_code_are_ignored() {
        let body = "Real [Alpha](Alpha.md).\n\n```\n[Fenced](Nope.md)\n```\n\n\
                    Inline `[Inline](Nope.md)` ignored.";
        assert_eq!(extract_markdown_links(body), vec!["alpha"]);
    }

    #[test]
    fn extract_links_merges_wikilinks_and_markdown_and_dedups() {
        let body = "[[Alpha]] and [Alpha](Alpha.md) and [Beta](Beta.md) and [[beta]]";
        // Wikilinks first (in order), then new markdown targets; duplicates drop.
        assert_eq!(extract_links(body), vec!["alpha", "beta"]);
    }

    #[test]
    fn frontmatter_wikilinks_become_links() {
        let note = parse_note(
            "N.md",
            "---\nup: \"[[Parent]]\"\nrelated:\n  - \"[[Alpha]]\"\n  - \"[[Beta]]\"\n\
             note: \"see ![[Embed]] here\"\n---\nbody links [[Alpha]] and [[Gamma]]\n",
        );
        // Body links keep their document order and come first.
        assert_eq!(note.links[0], "alpha");
        assert_eq!(note.links[1], "gamma");
        // Frontmatter links are added, deduped against the body (alpha once).
        let mut all = note.links.clone();
        all.sort();
        assert_eq!(all, vec!["alpha", "beta", "gamma", "parent"]);
        // An embed in a frontmatter value is neither a link nor an embed.
        assert!(!note.embeds.contains(&"embed".to_string()));
    }

    #[test]
    fn frontmatter_links_require_a_whole_wikilink_value() {
        // A value that only *contains* `[[...]]` inside prose or code is not a
        // link, so descriptions and code snippets do not grow spurious edges.
        let note = parse_note(
            "N.md",
            "---\ndescription: \"use [[foo]] syntax\"\nexample: \"arr[[0]][[1]]\"\n---\nbody\n",
        );
        assert!(note.links.is_empty(), "prose/code values are not links");
    }

    #[test]
    fn frontmatter_tags_are_never_link_targets() {
        // A wikilink-shaped tag stays a tag and does not also become a link.
        let note = parse_note("N.md", "---\ntags: \"[[meta]]\"\n---\nbody\n");
        assert!(note.links.is_empty(), "the tags property is not scanned");
    }

    #[test]
    fn frontmatter_engine_owned_keys_are_not_link_sources() {
        // A wikilink in `title`/`path`/`body` (display/content fields the loader
        // owns) must not forge a phantom edge; only real link fields count.
        let note = parse_note(
            "N.md",
            "---\ntitle: \"[[Foo]]\"\nbody: \"[[Bar]]\"\nup: \"[[Real]]\"\n---\nplain\n",
        );
        assert_eq!(note.links, vec!["real"], "only `up` is a link field");
    }

    #[test]
    fn frontmatter_without_wikilinks_adds_no_links() {
        let note = parse_note(
            "N.md",
            "---\ntitle: Plain\ntags: [a, b]\ncount: 3\n---\nno links here\n",
        );
        assert!(note.links.is_empty(), "no wikilinks anywhere");
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
        assert_eq!(
            note.properties.get("key"),
            Some(&Value::Str("user-role".into())),
            "normalized key is stored for name resolution"
        );
        assert_eq!(note.links, vec!["project"]);
    }
}
