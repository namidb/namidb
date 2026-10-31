# Multi-tenancy: isolation modes and namespace-scoped tokens

**Status:** operator guide
**Author(s):** NamiDB team
**Created:** 2026-06-26
**Updated:** 2026-10-28
**Applies to:** v3.0
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

Everything under **Implemented** is grounded in the current code. Source
references name the file and the symbol, not line numbers, so they stay
valid as the code moves. Everything under **Proposed / Future** is design
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
  binding (`crates/namidb-query/src/exec/walker.rs`).

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

Run the server with `--multi-tenant` (`crates/namidb-server/src/main.rs`).
Each tenant gets its own namespace, and a namespace is a full physical boundary:
its own manifest pointer family, SSTs, WAL, in-memory snapshot, single writer,
and background maintenance (flush / compaction / orphan sweep) are all spawned
per namespace (`NamespaceRegistry::get_or_open` in
`crates/namidb-server/src/registry.rs`). Each namespace has its **own** vector index, so ANN never
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
| **Ops cost** | Lowest. One namespace, one token can cover everything; no per-tenant provisioning. | Higher. Per-tenant token provisioning (manual today), lazy namespace creation, Bolt clients must set the session database, watch `--max-namespaces` / idle eviction. |

Rule of thumb: use **property mode** for many small, low-trust-boundary tenants
where a per-query predicate is acceptable and cost dominates; use
**namespace-per-tenant** when tenants need a real data boundary, independent
vector indexes, or noisy-neighbour isolation.

## How request-time namespace selection works (Implemented)

This matters because it is what makes a scoped token safe. In `--multi-tenant`
mode the HTTP router exposes both `/:namespace/v0/...` and unprefixed `/v0/...`
routes (`crates/namidb-server/src/lib.rs`). For each request the auth
middleware resolves the target namespace **before** authenticating:

- `resolve_request_namespace` (`lib.rs`) reads axum/matchit's captured
  `:namespace` path parameter first — the same value the handler will serve —
  and only falls back to the `X-NamiDB-Namespace` header
  (`namespace_from_header`, `lib.rs`, lowercased lookup) and then
  `--default-namespace` when no path param was captured. Reading the captured
  param (rather than re-parsing the URI) is deliberate: it closes the
  `/v0/v0/...` cross-tenant bypass class (`lib.rs`).
- `require_auth_multi` (`lib.rs`) then calls
  `auth.principal_for_in(token, &namespace)` (`lib.rs`). A token that is
  valid overall but scoped to a *different* namespace is rejected with `401` and
  the message `"missing or invalid bearer token, or token not scoped to this
  namespace"`.
- The resolved namespace is passed on to the authorization hook
  (`AuthzHook::check` / `check_schema`), so an external PDP receives it as
  `input.namespace` and a policy can gate by tenant.

**Bolt routes by the `db` field (since 2.3.0).** In `--multi-tenant` mode the
Bolt listener serves every namespace: a statement routes to the database named
by the driver's `session(database="acme")` (the Bolt `db` field of RUN/BEGIN),
falling back to `--default-namespace`. The token is re-resolved against the
requested namespace per statement (`principal_for_in`), so namespace-scoped
tokens are enforced over Bolt exactly like the HTTP middleware; an explicit
transaction is pinned to the database named at BEGIN. (History: 2.2.x refused
the flag combination at boot; earlier versions silently never opened the Bolt
port.)

## Provisioning a namespace and a scoped token (Implemented)

There is **no integrated "create namespace + token" command** today. Provisioning
is two independent steps:

1. **Mint a namespace-scoped token** by editing the `--auth-tokens-file` JSON (or
   issuing a scoped JWT from your IdP).
2. **Materialize the namespace** — namespaces are created *lazily* on first
   access; there is no create/delete DDL.

### Namespace names

A namespace name must match `[a-z0-9][a-z0-9_-]{0,62}` (length 1..=63),
validated by `NamespaceId::new` (`crates/namidb-core/src/id.rs`). The
registry validates the same way on open (`registry.rs`), and the CLI exposes
the check directly:

```bash
namidb namespace-check acme        # ok: acme
namidb namespace-check 'Acme!'     # error: invalid namespace
```

### Static tokens: the real `--auth-tokens-file` format

The file is parsed into `TokenFile` / `TokenFileEntry`
(`crates/namidb-server/src/auth.rs`) by `AuthConfig::load_file`, at boot
and again on every hot reload (see *Adding or rotating a static token*,
below). The exact accepted shape:

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
  time. An empty secret is rejected (`auth.rs`).
- **`name`** (optional). A human label only; defaults to `token-<i>`
  (`auth.rs`). Surfaces as the `Principal.subject`.
- **`role`** (optional). `"read-write"` or `"read-only"`; defaults to
  `"read-write"` (`RoleSpec`, `auth.rs`). A `read-only` token has every
  write (and admin flush) rejected.
- **`namespaces`** (optional). This is the scoping field
  (`AuthToken.namespaces`, `auth.rs`):
  - **omitted or `null`** ⇒ *unscoped* — the token reaches **every** namespace
    (the back-compat default). This is what `"platform"` above is.
  - **a list** ⇒ scoped to exactly those namespaces.
  - **`[]` (empty array)** ⇒ denies **all** namespaces. This is almost never
    what you want; omit the key instead of writing `[]`.

A file with an empty `tokens` array is rejected outright — that would silently
disable auth; omit `--auth-tokens-file` to run open on purpose
(`auth.rs`).

Enforcement happens in `role_for_in` / `principal_for_in` →
`token_principal` (`auth.rs`): on the multi-tenant
path a scoped token grants its role only when the requested namespace is in its
set; an unscoped token grants always. (On the single-tenant path the
namespace is the empty-string sentinel `""`, which bypasses the scope check:
`role_for` / `principal_for` in `auth.rs`. Multi-tenant Bolt resolves the
token against the requested namespace per statement, like HTTP.)

### Worked example (HTTP, namespace-per-tenant)

Start a multi-tenant server. In `--multi-tenant` mode the `?ns=` part of
`--store` is **optional** (since 2.5.0; earlier versions demanded it and then
discarded the value) — the registry uses the bucket/root and routes each
namespace under a flat layout. If you do pass one it is ignored with a boot
warning: the fallback tenant is named by `--default-namespace`, never by the
URI. Boolean flags/envs accept `1/0/true/false/yes/no/on/off`
(`NAMIDB_MULTI_TENANT=1` works):

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
there is no explicit DDL (`registry.rs`). The first authorized request
above is enough to materialize `acme`. If you want to pre-create it out of band,
issue one write through the CLI against the same store (this opens and commits a
`WriterSession` for that namespace, see `crates/namidb-cli/src/main.rs`):

```bash
namidb run --store "s3://my-bucket?ns=acme&region=us-east-1" \
  'CREATE (:_Bootstrap {at: timestamp()})'
```

There is **no delete-namespace** command; reclaiming a namespace means deleting
its objects out of band.

### Backing up and restoring a namespace (Implemented)

Both copies run inside the serving process, through its own store, and are
disabled until `--backup-target-uri` names the allowlisted backup location.
One backup or restore runs at a time per process.

The prefix carries the store options too, e.g.
`--backup-target-uri 's3://my-backups/namidb?endpoint=http://minio:9000&region=us-east-1&allow_http=true'`.
A backup's `to` and a restore's `from` must use the same scheme and bucket,
sit at or below the prefix's path with no `.`, `..` or empty segments
(percent-encoded forms included), repeat every query parameter of the
prefix with the same value, and add nothing but `ns`. A tenant can neither
climb out of the prefix nor point the server's credentials at an endpoint
of its choosing. On `file://` the path must also resolve, symlinks
included, inside the prefix's directory.

```bash
# Back up `acme` (live; the namespace keeps serving).
curl -s -X POST https://host:8080/acme/v0/admin/backup \
  -H "Authorization: Bearer $OPERATOR_TOKEN" -H 'content-type: application/json' \
  -d '{"to": "s3://my-backups/namidb/acme-2026-10-31?ns=acme"}'

# Restore it into a NEW namespace `acme-restored`.
curl -s -X POST https://host:8080/acme-restored/v0/admin/restore \
  -H "Authorization: Bearer $OPERATOR_TOKEN" -H 'content-type: application/json' \
  -d '{"from": "s3://my-backups/namidb/acme-2026-10-31?ns=acme", "verify": true}'
# => {"source_version":0,"objects_copied":<n>,"bytes_copied":<n>}
```

Use the HTTP restore, not `namidb restore`, for a multi-tenant server on
S3: the server keeps namespaces at the bucket root, and an `s3://` URI
always resolves to `<prefix or tenants>/<ns>/`, so the CLI writes where the
server never reads. (On `file://` the two layouts agree.)

Restore only targets a namespace that does not exist. There is no fencing
and no `force`: the server answers 409 when the target is open or being
opened on this node, is already being restored, or has a manifest pointer
in the store, and while the copy runs every request that would open the
target is refused as temporarily unavailable (HTTP 503, a Bolt failure). A
restore can be retried as is: the pointer is written last, a target without
a pointer was never opened, so the restore first deletes whatever an
interrupted attempt left there, and a failed copy (a `verify` mismatch
included) deletes what it wrote before releasing the target. The copy runs
on its own task, so a client or proxy that disconnects does not abandon it
halfway. To replace a namespace,
restore into a new name and switch traffic to it, or stop serving the old
one and delete its objects out of band first. Other nodes are not checked:
the routing layer must not send requests for the target to another node
while the restore runs, for the same reason as in
[Running as a hosted service](#running-as-a-hosted-service).

The allowlist is process-wide, not per tenant. A token that may call admin
backup or restore on a namespace can read any backup under the prefix into
it, and write that namespace's backup anywhere under it. Give those calls
operator tokens, or run one process per tenant with its own prefix.

### JWT scoping (Implemented, `jwt` feature)

NamiDB is a JWT **verifier**, not an issuer: it validates a bearer JWT against an
external IdP's JWKS and never mints one. With `--multi-tenant` plus
`--jwt-namespaces-claim`, a JWT is scoped the same way a static token's
`namespaces` list scopes it.

Configure it (`crates/namidb-server/src/main.rs`):

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
  (`ACCEPTED_ALGS`, `jwt.rs`). Symmetric `HS*` and `none` are refused
  (`jwt.rs`).
- **`exp`** with 30s leeway (`jwt.rs`); optional `iss` / `aud` enforced only
  when configured (`jwt.rs`).
- **`sub`** becomes `Principal.subject` (`jwt.rs`).
- **groups claim** (`--jwt-groups-claim`, default `groups`) → role:
  `--jwt-write-group` wins over `--jwt-read-group` (`jwt.rs`).
- **namespaces claim** (`--jwt-namespaces-claim`): the scoping check at
  `jwt.rs`. When configured, the token must name the requested namespace
  in that claim; unconfigured ⇒ unscoped (any namespace). The claim value may be
  a **JSON array of strings or a single string** (`extract_group_strings`,
  `jwt.rs`).

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
(`crates/namidb-server/src/lib.rs`).

### Adding or rotating a static token: hot reload (since 2.5.0)

The server re-reads `--auth-tokens-file` on an interval
(`--auth-tokens-reload-interval`, default `10s`, `0s` disables) and swaps the
token set atomically — onboarding or rotating a tenant no longer restarts
anyone. Semantics to rely on:

- **Write the file atomically** (write a temp file, then `rename(2)`; keep it
  `0600`). A partial or malformed file is rejected and the LAST GOOD set keeps
  serving — a bad reload can never widen access, empty the set, or flip the
  server open. The next tick picks up the completed write.
- HTTP requests see a reload on the **next request** (both single- and
  multi-tenant). Multi-tenant Bolt re-resolves the principal **per statement**
  (pinned for the life of an explicit transaction).
- **Revocation does not terminate live single-tenant Bolt sessions**: their
  principal is cached at LOGON. Kill the connection (or restart) if immediate
  revocation of an active session is required; fresh LOGONs with the revoked
  token are refused as soon as the reload lands.
- Change detection is content-compare, not mtime, so `rename(2)` swaps and
  ConfigMap-style updates are picked up reliably.

(JWT key rotation likewise needs no restart — the JWKS refreshes on a timer.)

## Running as a hosted service

The engine can serve many tenants from one process, but a hosted service
needs a few rules around it. Some of them are flags, the rest belong in
whatever control plane sits in front.

**One owner per namespace.** A namespace has exactly one writer, fenced by an
epoch in its manifest. Any process that opens the namespace for writing claims
the next epoch and fences the previous holder: a second server pod, an old pod
that overlaps a new one during a rolling deploy, or an embedded Python client
pointed at the same bucket. Two writers on one namespace never add
throughput, they just fence each other. Pick one node per namespace and route
every request for it there.

**Use `--fence-policy=yield` when something outside decides ownership.** The
default, `steal`, makes a fenced server reopen and claim the namespace back,
which is right for a single box. When a control plane moves namespaces
between nodes, run every node with `yield` (`NAMIDB_FENCE_POLICY=yield`): the
new owner claims the epoch and the old one stops writing and reports its
writer degraded, so readiness goes 503. The yielded writer stays down until
the control plane restarts or reassigns it.

Yield only covers a writer that is already open. In `--multi-tenant` mode any
request for a namespace that is not open on a node opens it there, and that
open claims the epoch and fences the real owner, whatever policy either node
runs. Health checks count: `/<namespace>/v0/health` opens the namespace it
asks about, and the unprefixed `/v0/health` opens the header or default
namespace. So the routing layer must never send a namespace's requests,
probes included, to a node that does not own it. Use `/v0/livez` for node
liveness, since it opens nothing. A yielded namespace that gets evicted under
capacity pressure is no longer open either, so the next stray request for it
would steal it the same way.

**Turn on group commit.** `--group-commit-window` defaults to `0s`, which
commits every write on its own. Many clients writing the same namespace queue
on the writer lock and each one pays a full commit round trip. A window of
`2ms` to `5ms` lets writes that arrive together share one WAL append and one
manifest CAS. The cost is shared fate: if a group's commit fails, every write
in it fails.

**Reads stay off the writer.** HTTP reads, Bolt auto-commit reads and Bolt
transactions opened in read access mode (what `session.execute_read` sends)
all run on the published snapshot and never wait for the writer lock. Write
transactions hold the writer lock until COMMIT or ROLLBACK, so keep them short;
`--bolt-tx-timeout` rolls back one that goes idle.

**Keep one tenant from starving the rest.** The read admission gate
(`NAMIDB_MAX_CONCURRENT_QUERIES`) is shared by the whole process. Set
`NAMIDB_MAX_CONCURRENT_QUERIES_PER_NAMESPACE` to give each namespace its own
cap inside it. Namespaces also open independently: a slow cold open (large
WAL to replay, slow bucket) only delays requests for that namespace.

**What the control plane owns.** Tenant-to-node placement, the routing proxy
(the HTTP `/:namespace` path or `X-NamiDB-Namespace` header, the Bolt `db`
field), health-driven failover, issuing JWTs with a namespaces claim, rate
limits, connection caps, storage quotas, billing, namespace provisioning and
deletion, and scheduled backups and restores through the admin backup and
restore endpoints. The engine
makes ownership safe to move. It does not decide where a namespace lives.

## Proposed / Future

None of this exists yet; it is design intent.

### `namidb token mint` (proposed CLI)

The `namidb` CLI today has no token/tenant subcommand — its commands are
`version`, `namespace-check`, `parse`, `explain`, `run`, `load-vault`, `backup`,
`restore` (`crates/namidb-cli/src/main.rs`). A proposed subcommand would
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

This is provisioning sugar only. The running server picks up the edited file
on its next hot reload, and the command would not create the namespace:
namespaces remain lazy.

### Other future work

- **JWT minting / NamiDB as a key authority.** Out of scope today: NamiDB holds
  no signing key. Minting JWTs would mean generating/rotating a private key and
  serving a JWKS — a new trust boundary, RFC-sized.
- **Explicit namespace lifecycle.** A real create/list/delete API instead of
  lazy `get_or_open`, so provisioning and de-provisioning are first-class.
  (Admin restore is the one way today to create a namespace with content
  without opening it first.)

## Drawbacks and operational notes

- **Property-mode isolation is best-effort.** The global vector ANN returns
  cross-tenant neighbours; a post-filter by `tenant_id` can degrade recall, and a
  missing predicate leaks data. Do not treat it as a security boundary.
- **`namespaces: []` is a foot-gun.** An empty array denies every namespace.
  Omit the key for an unscoped token.
- **Static tokens reload on an interval, not instantly.** A new or revoked
  token takes effect on the next reload tick (`--auth-tokens-reload-interval`,
  default `10s`), and a live single-tenant Bolt session keeps the principal it
  logged on with.
- **Bolt multidatabase needs driver `database=` support.** Since 2.3.0 Bolt
  routes per statement by the `db` field; clients that cannot set a session
  database land on `--default-namespace`.
- **Capacity and eviction.** In `--multi-tenant` mode, `--max-namespaces`
  (default 100) caps concurrently open namespaces and idle ones are evicted
  oldest-first (`registry.rs`); `--namespace-idle-timeout` (default 1h)
  controls idleness. A namespace with a read in flight, such as an open Bolt
  read transaction, is never idle. Eviction drops the in-memory session, not
  the data.

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
