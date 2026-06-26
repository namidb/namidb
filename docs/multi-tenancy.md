# Multi-tenancy: isolation modes and namespace-scoped tokens

**Status:** operator guide
**Author(s):** NamiDB team
**Created:** 2026-06-26
**Updated:** 2026-06-26
**Applies to:** v1.4 + the namespace-scoping change set
**Related:** RFC-032 (filtered ANN pre-filtering), RFC-030 (vector index),
RFC-015 (auth / external PDP)

## Summary

NamiDB supports two ways to serve many tenants from one deployment, and they
sit at opposite ends of the isolation/cost curve:

- **Property mode** — one namespace holds every tenant's data, distinguished by
  a `tenant_id` (or similar) property. Isolation is *logical only*: every query
  must carry a tenant predicate. All tenants share **one** vector index per
  `(label, property)`.
- **Namespace-per-tenant** — one namespace per tenant (`--multi-tenant`), each
  with its own manifest, SSTs, WAL, writer, and **its own** vector index.
  Isolation is *physical*, enforced by namespace-scoped tokens.

This guide explains the trade-off, then documents exactly how to provision a
namespace and a namespace-scoped token **today** — the real `--auth-tokens-file`
JSON shape and the JWT `namespaces` claim the server parses — with worked
examples. A final section sketches a proposed `namidb token mint` subcommand;
that command does not exist yet and is clearly marked *Proposed*.

Everything under **Implemented** is grounded in the v1.4 code plus the
namespace-scoping change set. Everything under **Proposed / Future** is design
only.

## Motivation

The choice of isolation mode is made once, early, and is expensive to reverse
(it changes where data physically lives and how tokens are issued). Operators
need a precise picture of what each mode actually enforces — and, critically,
what property mode does *not* enforce — before they hand a port to more than one
tenant. The shared-vector-index behaviour in particular is easy to get wrong:
an approximate-nearest-neighbour (ANN) search over a shared index returns
*global* neighbours, so a tenant filter applied around the search is a
correctness-and-leakage concern, not a free knob.

## The two isolation modes

### Property mode (one namespace, logical isolation)

All tenants live in a single namespace. A `tenant_id` property tags each node;
queries scope themselves with a predicate. There is exactly **one** vector index
per `(label, property)` for the whole namespace — a vector index is built into
that namespace's manifest and SSTs and is addressed through its
`NamespacePaths`, with no row-level partitioning by tenant. ANN/KNN therefore
returns the global nearest neighbours across *all* tenants, and any tenant
restriction is applied as a filter *around* the search.

Two ways to express the tenant restriction today (both **Implemented**):

- **A natural `WHERE` predicate** on a vector query, e.g.

  ```cypher
  MATCH (d:Doc)
  WHERE d.tenant_id = $t
  RETURN d
  ORDER BY cosine_similarity(d.embedding, $q) LIMIT $k
  ```

- **The `filter` map of `db.index.vector.queryNodes`**:

  ```cypher
  CALL db.index.vector.queryNodes('docEmb', $k, $q, {filter: {tenant_id: $t}})
  YIELD node, score
  RETURN node, score
  ```

  The optional 4th map argument carries `ef` (beam width) and/or `filter`; the
  `filter` map is compiled to a residual `post_filter` bound to the `node`
  binding (`crates/namidb-query/src/exec/walker.rs:2329`, `walker.rs:2340-2353`).

Both forms are served by **post-filtering**: the index over-fetches the global
top candidates, the executor materializes them, applies the predicate, and falls
back to an exact flat scan when the over-fetch under-fills (RFC-030's freshness
gate + exact fallback). RFC-032 (filtered ANN pre-filtering) proposes pushing a
low-cardinality equality predicate *into* the beam search so a selective
`tenant_id = $t` is served from the index instead of post-filtered — but that is
**Proposed**; v1.4 post-filters.

The load-bearing caveat: **property-mode isolation is logical, not a security
boundary.** A forgotten predicate, or the global ANN candidate set itself, can
surface another tenant's rows. Treat the predicate as an application invariant
you must enforce on every query, not as an access control the engine guarantees.

### Namespace-per-tenant (physical isolation)

Run the server with `--multi-tenant` (`crates/namidb-server/src/main.rs:229-233`).
Each tenant gets its own namespace, and a namespace is a full physical boundary:
its own manifest pointer family, SSTs, WAL, in-memory snapshot, single writer,
and background maintenance (flush / compaction / orphan sweep) are all spawned
per namespace (`crates/namidb-server/src/registry.rs:135-209`,
`registry.rs:215`). Each namespace has its **own** vector index, so ANN never
crosses a tenant boundary and the index only spans one tenant's corpus.

Access is enforced by namespace-scoped tokens (see *Provisioning*, below): a
token scoped to `acme` is rejected for any other namespace *before* the request
runs.

### Trade-off table

| Dimension | Property mode (one namespace) | Namespace-per-tenant (`--multi-tenant`) |
|---|---|---|
| **Isolation strength** | Logical only. Enforced by a `WHERE` / `filter` predicate the application adds to every query. A missing predicate or the global ANN leaks across tenants. **Not** a security boundary. | Physical. Separate `NamespacePaths`, manifest, SSTs, WAL, and writer per tenant. A scoped token cannot reach another namespace (`role_for_in` / `principal_for_in`). |
| **Vector index size** | One shared index per `(label, property)`; smallest aggregate footprint; one build, one writer. ANN returns global neighbours. | One index **per namespace**; aggregate index bytes scale with tenant count (no sharing), but each index spans only one tenant — better selectivity/recall. |
| **Noisy neighbour** | High. All tenants share one writer, one SST set, one ANN graph; one tenant's data volume and read/write load affect everyone. | Low. Per-namespace writer + maintenance isolate load. Bounded by `--max-namespaces` and idle eviction. |
| **Ops cost** | Lowest. One namespace, one token can cover everything; no per-tenant provisioning. | Higher. Per-tenant token provisioning (manual today), lazy namespace creation, Bolt needs one port per namespace, watch `--max-namespaces` / idle eviction. |

Rule of thumb: use **property mode** for many small, low-trust-boundary tenants
where a per-query predicate is acceptable and cost dominates; use
**namespace-per-tenant** when tenants need a real data boundary, independent
vector indexes, or noisy-neighbour isolation.

## How request-time namespace selection works (Implemented)

This matters because it is what makes a scoped token safe. In `--multi-tenant`
mode the HTTP router exposes both `/:namespace/v0/...` and unprefixed `/v0/...`
routes (`crates/namidb-server/src/lib.rs:366-393`). For each request the auth
middleware resolves the target namespace **before** authenticating:

- `resolve_request_namespace` (`lib.rs:418-429`) reads axum/matchit's captured
  `:namespace` path parameter first — the same value the handler will serve —
  and only falls back to the `X-NamiDB-Namespace` header
  (`namespace_from_header`, `lib.rs:397-404`, lowercased lookup) and then
  `--default-namespace` when no path param was captured. Reading the captured
  param (rather than re-parsing the URI) is deliberate: it closes the
  `/v0/v0/...` cross-tenant bypass class (`lib.rs:406-417`).
- `require_auth_multi` (`lib.rs:809-848`) then calls
  `auth.principal_for_in(token, &namespace)` (`lib.rs:830`). A token that is
  valid overall but scoped to a *different* namespace is rejected with `401` and
  the message `"missing or invalid bearer token, or token not scoped to this
  namespace"`.

**Bolt is single-namespace only.** The Bolt `LOGON` path uses the
namespace-agnostic `auth.principal_for` (`crates/namidb-server/src/bolt.rs:904`);
there is no per-request namespace over Bolt. To serve namespace-per-tenant over
Bolt you run **one server/port per namespace**.

## Provisioning a namespace and a scoped token (Implemented)

There is **no integrated "create namespace + token" command** today. Provisioning
is two independent steps:

1. **Mint a namespace-scoped token** by editing the `--auth-tokens-file` JSON (or
   issuing a scoped JWT from your IdP).
2. **Materialize the namespace** — namespaces are created *lazily* on first
   access; there is no create/delete DDL.

### Namespace names

A namespace name must match `[a-z0-9][a-z0-9-]{0,62}` (length 1..=63),
validated by `NamespaceId::new` (`crates/namidb-core/src/id.rs:153-177`). The
registry validates the same way on open (`registry.rs:168`), and the CLI exposes
the check directly:

```bash
namidb namespace-check acme        # ok: acme
namidb namespace-check 'Acme!'     # error: invalid namespace
```

### Static tokens: the real `--auth-tokens-file` format

The file is parsed into `TokenFile` / `TokenFileEntry`
(`crates/namidb-server/src/auth.rs:309-326`) and loaded once at boot by
`AuthConfig::load_file` (`auth.rs:142-176`). The exact accepted shape:

```json
{
  "tokens": [
    { "name": "acme-rw",  "token": "<secret>", "role": "read-write", "namespaces": ["acme"] },
    { "name": "acme-ro",  "token": "<secret>", "role": "read-only",  "namespaces": ["acme"] },
    { "name": "acme-staging", "token": "<secret>", "namespaces": ["acme", "acme-staging"] },
    { "name": "platform", "token": "<secret>", "role": "read-write" }
  ]
}
```

Field semantics, exactly as the code treats them:

- **`token`** (required, non-empty). The shared secret, compared in constant
  time. An empty secret is rejected (`auth.rs:158-163`).
- **`name`** (optional). A human label only; defaults to `token-<i>`
  (`auth.rs:165`). Surfaces as the `Principal.subject`.
- **`role`** (optional). `"read-write"` or `"read-only"`; defaults to
  `"read-write"` (`RoleSpec`, `auth.rs:328-335`). A `read-only` token has every
  write (and admin flush) rejected.
- **`namespaces`** (optional). This is the scoping field
  (`AuthToken.namespaces`, `auth.rs:84`):
  - **omitted or `null`** ⇒ *unscoped* — the token reaches **every** namespace
    (the back-compat default). This is what `"platform"` above is.
  - **a list** ⇒ scoped to exactly those namespaces.
  - **`[]` (empty array)** ⇒ denies **all** namespaces. This is almost never
    what you want; omit the key instead of writing `[]`.

A file with an empty `tokens` array is rejected outright — that would silently
disable auth; omit `--auth-tokens-file` to run open on purpose
(`auth.rs:147-152`).

Enforcement happens in `role_for_in` / `principal_for_in` →
`token_principal` (`auth.rs:242-260`, `auth.rs:274-301`): on the multi-tenant
path a scoped token grants its role only when the requested namespace is in its
set; an unscoped token grants always. (On the single-tenant / Bolt path the
namespace is the empty-string sentinel `""`, which bypasses the scope check —
`role_for` / `principal_for`, `auth.rs:205-215`, `auth.rs:277`.)

### Worked example (HTTP, namespace-per-tenant)

Start a multi-tenant server. Note that in `--multi-tenant` mode the `?ns=` part
of `--store` is ignored — the registry uses the bucket/root and routes each
namespace under a flat layout (`lib.rs:501-519`):

```bash
namidb-server \
  --multi-tenant \
  --store "s3://my-bucket?region=us-east-1" \
  --auth-tokens-file /etc/namidb/auth-tokens.json \
  --default-namespace acme \
  --max-namespaces 100
```

With the `auth-tokens.json` above, the `acme-rw` token reaches only `acme`.
Target the namespace by path:

```bash
curl -s https://host:8080/acme/v0/cypher \
  -H "authorization: Bearer $ACME_RW" \
  -H 'content-type: application/json' \
  -d '{"query":"MATCH (n) RETURN count(n) AS n"}'
```

…or unprefixed, with the header (resolves the same namespace):

```bash
curl -s https://host:8080/v0/cypher \
  -H "authorization: Bearer $ACME_RW" \
  -H 'x-namidb-namespace: acme' \
  -H 'content-type: application/json' \
  -d '{"query":"MATCH (n) RETURN count(n) AS n"}'
```

The same token aimed at another namespace is rejected before the query runs:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://host:8080/other/v0/cypher \
  -H "authorization: Bearer $ACME_RW" \
  -H 'content-type: application/json' \
  -d '{"query":"MATCH (n) RETURN count(n)"}'
# 401
```

### Materializing the namespace

Namespaces are lazy: the registry opens a `WriterSession` on first access and
there is no explicit DDL (`registry.rs:135-209`). The first authorized request
above is enough to materialize `acme`. If you want to pre-create it out of band,
issue one write through the CLI against the same store (this opens and commits a
`WriterSession` for that namespace — `crates/namidb-cli/src/main.rs:487-516`):

```bash
namidb run --store "s3://my-bucket?ns=acme&region=us-east-1" \
  'CREATE (:_Bootstrap {at: timestamp()})'
```

There is **no delete-namespace** command; reclaiming a namespace means deleting
its objects out of band.

### JWT scoping (Implemented, `jwt` feature)

NamiDB is a JWT **verifier**, not an issuer: it validates a bearer JWT against an
external IdP's JWKS and never mints one. With `--multi-tenant` plus
`--jwt-namespaces-claim`, a JWT is scoped the same way a static token's
`namespaces` list scopes it.

Configure it (`crates/namidb-server/src/main.rs:48-75`):

```bash
namidb-server \
  --multi-tenant \
  --store "s3://my-bucket?region=us-east-1" \
  --jwt-jwks-url "https://issuer/.well-known/jwks.json" \
  --jwt-issuer "https://issuer/" \
  --jwt-audience "namidb" \
  --jwt-groups-claim groups \
  --jwt-write-group namidb-writers \
  --jwt-read-group  namidb-readers \
  --jwt-namespaces-claim tenants
```

What the validator parses (`crates/namidb-server/src/jwt.rs`):

- **Signature**: only asymmetric JWKS algorithms `RS256/384/512`, `ES256/384`
  (`ACCEPTED_ALGS`, `jwt.rs:29-35`). Symmetric `HS*` and `none` are refused
  (`jwt.rs:165-167`).
- **`exp`** with 30s leeway (`jwt.rs:255`); optional `iss` / `aud` enforced only
  when configured (`jwt.rs:257-266`).
- **`sub`** → `Principal.subject` (`jwt.rs:240-245`).
- **groups claim** (`--jwt-groups-claim`, default `groups`) → role:
  `--jwt-write-group` wins over `--jwt-read-group` (`jwt.rs:218-238`).
- **namespaces claim** (`--jwt-namespaces-claim`): the scoping check at
  `jwt.rs:202-211`. When configured, the token must name the requested namespace
  in that claim; unconfigured ⇒ unscoped (any namespace). The claim value may be
  a **JSON array of strings or a single string** (`extract_group_strings`,
  `jwt.rs:273-283`).

A matching JWT payload for the config above:

```json
{
  "iss": "https://issuer/",
  "aud": "namidb",
  "sub": "alice@acme.example",
  "exp": 1750000000,
  "groups": ["namidb-writers"],
  "tenants": ["acme"]
}
```

This token authenticates as read-write against namespace `acme` and is rejected
for any namespace not in `tenants`. Mint and rotate it in your IdP — NamiDB only
verifies it, and refreshes the JWKS hourly so key rotation needs no restart
(`crates/namidb-server/src/lib.rs:461-465`).

### Adding or rotating a static token requires a restart

`AuthConfig::load_file` runs once at boot (`lib.rs:437-438`); there is no SIGHUP
reload for the static-token file. To add or rotate a static token: write the new
entry, restart the server, then remove the old entry on a later restart. (JWT
key rotation does *not* need a restart — the JWKS refreshes on a timer.)

## Proposed / Future

None of this exists yet; it is design intent.

### `namidb token mint` (proposed CLI)

The `namidb` CLI today has no token/tenant subcommand — its commands are
`version`, `namespace-check`, `parse`, `explain`, `run`, `load-vault`, `backup`,
`restore` (`crates/namidb-cli/src/main.rs:40-191`). A proposed subcommand would
remove the hand-edit-JSON step for static tokens:

```text
# PROPOSED — does not exist yet
namidb token mint \
  --file auth-tokens.json \        # created if absent, appended if present
  --name acme-rw \
  --role read-write \              # default read-write
  --namespace acme \               # repeatable; omit ⇒ unscoped
  --generate                       # mint a 256-bit URL-safe secret (or --secret <s>)
```

What it would do, reusing existing symbols:

- Validate every `--namespace` with `NamespaceId::new` so a token can never name
  a namespace the server would reject.
- With `--generate`, draw a CSPRNG secret, print it **once** to stdout, and never
  log it.
- Append a `{ "name", "token", "role", "namespaces" }` object to `tokens`,
  **omitting** `namespaces` for an unscoped token (never writing `[]`), so the
  emitted file deserializes through `AuthConfig::load_file` with identical
  enforcement.
- Write atomically with `0600` permissions.

This is provisioning sugar only; it would still require a server restart to take
effect (no hot-reload), and it would not create the namespace — namespaces remain
lazy.

### Other future work

- **JWT minting / NamiDB as a key authority.** Out of scope today: NamiDB holds
  no signing key. Minting JWTs would mean generating/rotating a private key and
  serving a JWKS — a new trust boundary, RFC-sized.
- **Explicit namespace lifecycle.** A real create/list/delete API instead of
  lazy `get_or_open`, so provisioning and de-provisioning are first-class.
- **Static-token hot-reload.** A SIGHUP / file-watch reload so a minted token
  takes effect without a restart.
- **PDP namespace input.** The external-PDP request document advertises an
  `input.namespace` field (`crates/namidb-server/src/pdp.rs:19-26`) but
  `plan_input` does not currently emit it (`pdp.rs:138-151`), so OPA policies
  cannot gate by namespace yet. Threading the resolved namespace into
  `AuthzHook::check` would close that gap.

## Drawbacks and operational notes

- **Property-mode isolation is best-effort.** The global vector ANN returns
  cross-tenant neighbours; a post-filter by `tenant_id` can degrade recall, and a
  missing predicate leaks data. Do not treat it as a security boundary.
- **`namespaces: []` is a foot-gun.** An empty array denies every namespace.
  Omit the key for an unscoped token.
- **Boot-only static-token load.** A newly minted static token needs a server
  restart; only JWT keys rotate live.
- **Bolt cannot be namespace-scoped per request.** Namespace-per-tenant over Bolt
  means one server/port per tenant.
- **Capacity and eviction.** In `--multi-tenant` mode, `--max-namespaces`
  (default 100) caps concurrently open namespaces and idle ones are evicted
  oldest-first (`registry.rs:128-165`); `--namespace-idle-timeout` (default 1h)
  controls idleness. Eviction drops the in-memory session, not the data.

## References

- `crates/namidb-server/src/auth.rs` — `AuthConfig`, `Principal`, `AuthToken`,
  `TokenFile` / `TokenFileEntry`, `role_for_in` / `principal_for_in`.
- `crates/namidb-server/src/lib.rs` — `build_multi_tenant_router`,
  `resolve_request_namespace`, `require_auth_multi`, multi-tenant registry wiring.
- `crates/namidb-server/src/jwt.rs` — `JwtValidator`, `namespaces_claim` scoping,
  `ACCEPTED_ALGS`.
- `crates/namidb-server/src/registry.rs` — lazy `get_or_open`, per-namespace
  maintenance, capacity/eviction.
- `crates/namidb-server/src/main.rs` — `--multi-tenant`, `--default-namespace`,
  `--auth-tokens-file`, `--jwt-*`, `--max-namespaces`,
  `--namespace-idle-timeout`.
- `crates/namidb-query/src/exec/walker.rs` — `db.index.vector.queryNodes` and its
  `filter` map.
- `crates/namidb-core/src/id.rs` — `NamespaceId` name rules.
- RFC-030 (vector index), RFC-032 (filtered ANN pre-filtering), RFC-015 (auth /
  external PDP).
