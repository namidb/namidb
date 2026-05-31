# namidb-mcp

A local MCP (Model Context Protocol) server that exposes a NamiDB graph
namespace to agents like Claude Code over stdio. Instead of grepping flat
markdown files, the agent gets real graph traversals.

## Install

Download a prebuilt archive for your platform from the
[Releases page](https://github.com/namidb/namidb/releases) (each `v*` release
ships `namidb` and `namidb-mcp` for Linux x86_64/aarch64, macOS arm64 and
Windows x86_64) and put the binaries on your `PATH`. With a Rust toolchain you
can instead build from source:

```bash
cargo install --git https://github.com/namidb/namidb namidb-mcp
```

## Use

Point it at a namespace where a vault was loaded (see `namidb-markdown` or
`namidb load-vault`), or let it load one on startup:

```bash
# Load a vault into an ephemeral namespace and serve it
namidb-mcp --vault ./my-vault

# Serve an already-loaded durable namespace
namidb-mcp --store "file:///var/lib/namidb?ns=vault"
```

Then register it with your MCP client. For Claude Code:

```json
{
  "mcpServers": {
    "namidb": { "command": "namidb-mcp", "args": ["--vault", "./my-vault"] }
  }
}
```

## Tools

All read-only.

- `list_notes` - every note (title, path)
- `get_note {note}` - a note's title, path and full body
- `backlinks {note}` - notes linking to a note
- `neighbors {note, hops?}` - notes within N hops (undirected, default 1, max 5)
- `orphans` - notes with no links in or out
- `search {text}` - notes whose title or body contains a substring
- `list_tags` - every tag in the graph
- `notes_by_tag {tag}` - notes carrying a tag (exact, case-sensitive)
- `tags_of {note}` - a note's tags
- `cypher {query}` - run an arbitrary read-only Cypher query

This is the single-user local server. Multi-tenant hosting belongs in the
cloud layer.
