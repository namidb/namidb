# namidb-markdown

Loads an Obsidian-style markdown vault into NamiDB as a graph.

Each `.md` file becomes a `Note` node. Each `[[wikilink]]` becomes a
`LINKS_TO` edge. YAML frontmatter becomes node properties, and the raw note
body is kept as a `body` property. The original files stay the source of
truth; the graph is a derived index you can rebuild at any time.

```rust
use std::path::Path;
use namidb_markdown::{load_vault, LoadOptions};

let outcome = load_vault(Path::new("./vault"), &mut writer, &LoadOptions::default()).await?;
writer.commit_batch().await?;
```

Once loaded, the queries that Obsidian draws but cannot run are plain Cypher:

```cypher
// backlinks of a note
MATCH (src:Note)-[:LINKS_TO]->(:Note {path: $path}) RETURN src

// notes two hops away
MATCH (:Note {path: $path})-[:LINKS_TO*2..2]-(n:Note) RETURN DISTINCT n

// orphan notes (no links in or out)
MATCH (n:Note) WHERE NOT EXISTS((n)-[:LINKS_TO]-()) RETURN n
```

## Scope

This is a deliberate v1 subset, not a faithful Obsidian clone. It covers
wikilink variants (`[[a]]`, `[[a|alias]]`, `[[a#heading]]`, `[[a^block]]`,
`[[dir/a]]`, `![[embed]]`), code-fence exclusion, and flat frontmatter typing.
It does not model heading/block anchors as separate targets, inline `#tags`,
placeholder nodes for dangling links, or write-back to `.md`. Wikilinks resolve
by normalized basename, so `[[User Role]]`, `[[user-role]]` and `user_role.md`
collapse to one note.
