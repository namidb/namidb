# namidb-markdown

Loads an Obsidian-style markdown vault into NamiDB as a graph.

Each `.md` file becomes a `Note` node. A `[[wikilink]]` or a markdown link
`[text](note.md)` becomes a `LINKS_TO` edge; an embed `![[note]]` becomes a
distinct `EMBEDS` edge. Inline `#tags` and a frontmatter `tags` list become
shared `:Tag` nodes linked by `:TAGGED` edges. Other frontmatter becomes node
properties, and the raw body is kept as a `body` property. The files stay the
source of truth; the graph is a derived index you can rebuild at any time.

```rust
use std::path::Path;
use namidb_markdown::{load_vault, LoadOptions};

let outcome = load_vault(Path::new("./vault"), &mut writer, &LoadOptions::default()).await?;
writer.commit_batch().await?;
```

Once loaded, the queries that Obsidian draws but cannot run are plain Cypher.
Traversals span both reference edges so embeds count like links:

```cypher
// backlinks of a note (links and embeds)
MATCH (src:Note)-[:LINKS_TO|:EMBEDS]->(:Note {path: $path}) RETURN DISTINCT src

// notes two hops away
MATCH (:Note {path: $path})-[:LINKS_TO|:EMBEDS*2..2]-(n:Note) RETURN DISTINCT n

// orphan notes (nothing references them, and they reference nothing)
MATCH (n:Note) WHERE NOT EXISTS((n)-[:LINKS_TO|:EMBEDS]-()) RETURN n

// notes that share a tag
MATCH (:Note {path: $path})-[:TAGGED]->(:Tag)<-[:TAGGED]-(o:Note) RETURN DISTINCT o
```

## Scope

This is a deliberate v1 subset, not a faithful Obsidian clone. It covers
wikilink variants (`[[a]]`, `[[a|alias]]`, `[[a#heading]]`, `[[a^block]]`,
`[[dir/a]]`), embeds (`![[a]]`), markdown links to local `.md`/`.markdown`
files, inline and frontmatter `#tags`, and code-fence exclusion. Dangling
references can optionally become placeholder stub nodes (`LoadOptions::placeholders`
/ `--placeholders` / `placeholders=True`), marked `placeholder: true`; querying
`MATCH (n:Note) WHERE n.placeholder = true` lists unresolved references. It does
not model heading/block anchors as separate targets or write-back to `.md`.
Names resolve by normalized basename, so `[[User Role]]`, `[[user-role]]` and
`user_role.md` collapse to one note.
