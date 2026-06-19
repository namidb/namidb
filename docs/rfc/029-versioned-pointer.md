# RFC 029: Create-only versioned manifest pointer

**Status:** accepted
**Author(s):** NamiDB team
**Created:** 2026-06-16
**Updated:** 2026-06-16
**Implements:** feat/s3b-versioned-pointer
**Amends:** RFC-001 §"CAS protocol for committing a new manifest" (step 5)

## Summary

The manifest commit protocol advanced the namespace by overwriting a single
mutable pointer object, `manifest/current.json`, with `PutMode::Update`
(HTTP `If-Match: <etag>`). This RFC replaces that mutable pointer with a
**Create-only monotonic family**, `manifest/pointer/p<N>.json`, where each
file is written exactly once with `PutMode::Create` (HTTP `If-None-Match: *`)
and the current pointer is the highest `N` present. After this change the
entire commit path depends on a **single** object-store conditional
primitive — PUT-if-absent — which is the one primitive every S3-compatible
store (and the local filesystem) supports natively.

## Motivation

NamiDB's pitch is "your S3 bucket is the database on any S3-compatible
store." The commit protocol is the load-bearing part of that claim, and it
depended on two distinct conditional primitives:

- **`If-None-Match: *`** (PUT-if-absent) for the write-once manifest bodies
  `manifest/v<N>.json`. **Universally supported**: S3, GCS, Azure Blob, R2,
  Tigris, Nutanix Objects, MinIO — and `LocalFileSystem` via `O_CREAT|O_EXCL`.
- **`If-Match: <etag>`** (conditional overwrite) for the `current.json`
  pointer CAS. **Spottily supported**: this is the primitive `LocalFileSystem`
  refuses with `NotImplemented`, forcing the `LocalFileObjectStore` advisory
  `flock` workaround (`crates/namidb-storage/src/local.rs`); and several
  S3-compatible stores (Garage, SeaweedFS, older MinIO builds) implement it
  unevenly.

Depending on `If-Match` for the linearization point means the portability of
the whole engine is gated on the *rarer* of the two primitives. The fix is to
express the pointer the same way the bodies are already expressed — as a
write-once Create-only family — so the commit path needs nothing rarer than
PUT-if-absent.

The field has converged on exactly this shape. Delta Lake commits
`_delta_log/NNNNN.json` PUT-if-absent and derives the current version as the
highest `N` from a LIST. SlateDB writes manifest files with `If-None-Match`
CAS ("first to create wins") plus a monotonic writer epoch for fencing — i.e.
*Create-only pointer family + epoch fence*, adopted precisely because S3 only
universally guarantees `If-None-Match`. This RFC adopts the SlateDB/Delta
model.

## Design

### On-disk layout

```text
manifest/
  v0000000000000000.json     # immutable manifest body, version 0   (Create)
  v0000000000000001.json     # immutable manifest body, version 1   (Create)
  ...
  pointer/
    p0000000000000000.json   # pointer → body v0                    (Create)
    p0000000000000001.json   # pointer → body v1                    (Create)
    ...
  current.json               # legacy single pointer (read-only fallback)
```

Each `pointer/p<N>.json` holds the same `ManifestPointer` body as before
(`{version, epoch, manifest_path}`) and is **write-once**: `N` equals the
manifest version it points at, so the pointer for version `N` is created in
the same logical step that creates body `v<N>.json`. The **current pointer is
the highest `N` that exists.**

`current.json` is no longer written by the commit path. It is retained only as
a read fallback for namespaces bootstrapped before this RFC and for snapshots
produced by the pre-RFC backup path.

### Commit protocol (amends RFC-001 step 5)

To advance from version `v` to `v+1`:

1. (unchanged) PUT `manifest/v<v+1>.json` with `PutMode::Create`. Losing this
   means another writer claimed the version: reload, fence-check, retry.
2. (changed) PUT `manifest/pointer/p<v+1>.json` with `PutMode::Create`.
   Losing this (`AlreadyExists`) is treated exactly like losing step 1: reload
   to discover whether we were merely raced (`ManifestCommitCas`, retry) or
   fenced by a higher epoch (`Fenced`, drop the writer).

Both phases now use the *same* primitive (`If-None-Match: *`). The pointer
file for version `N` is the linearization point: whoever creates `p<N>.json`
first owns version `N`.

The commit body/pointer split (`put_body` + `cas_pointer`) and its pipelining
against the WAL segment PUT in `WriterSession::commit_batch` are unchanged;
only the conditional primitive inside `cas_pointer` changed.

### Reading the current pointer

`load_current` resolves the pointer as follows:

1. LIST `manifest/pointer/`, parse each `p<N>.json`, take the maximum `N`.
2. **Forward HEAD probe**: from that maximum, HEAD `p<N+1>`, `p<N+2>`, …
   advancing while each exists (bounded). Some S3-compatible stores are only
   eventually consistent for LIST, but GET/HEAD of a *specific* key is
   read-after-write consistent on every store we target; the probe closes the
   window where a just-created pointer a writer needs as its base has not yet
   appeared in LIST. The probe is safe against gaps because the janitor keeps
   the family contiguous over `[horizon, current]` (see GC below).
3. GET `p<max>.json` → `ManifestPointer`, then GET the manifest body it names.
4. If the family is empty, fall back to GET `manifest/current.json` (legacy /
   backup-produced namespace). If that too is absent, surface the object
   store's `NotFound`, which `WriterSession::open` turns into a bootstrap.

`load_current` is not on the per-read hot path — reads serve from the
published in-memory snapshot (RFC-021); only `WriterSession::open`, commit
retries, the janitor, and backup call it — so the LIST + HEAD cost is
acceptable.

### Garbage collection

The janitor reclaims `pointer/p<N>.json` for `N < horizon` under the same
retention-horizon + `min_age` rule it already applies to manifest snapshots
`v<N>.json`, where `horizon` is the oldest manifest version any live reader is
pinned to (RFC-027). Keeping every pointer at or above the horizon makes the
family **contiguous over `[horizon, current]`**, which is what makes the
forward HEAD probe in step 2 gap-safe: a stale LIST can only lag *behind*
`current`, never skip a hole, so the probe always walks up to the true
current. Pointer reclaims are reported separately from manifest-snapshot
reclaims (`JanitorReport::pointer_files_reclaimed` / `pointer_bytes_freed`).

### `LocalFileObjectStore`

The advisory `flock` CAS in `LocalFileObjectStore::put_with_cas` existed solely
to emulate `If-Match` (`PutMode::Update`) for the pointer overwrite. After this
RFC the commit path issues no `PutMode::Update`, so the `flock` machinery is
**no longer load-bearing for commits**. It is retained for now as a faithful
`ObjectStore` implementation (and is still exercised by the crate's
`PutMode::Update` contract tests); fully removing it is a follow-up once we
confirm nothing external relies on local conditional overwrite.

### Backup / restore

`copy_namespace_snapshot` writes the destination pointer into the family
(`pointer/p0.json`) instead of `current.json`, and its no-clobber guard
detects a live destination by the family (or the legacy `current.json`). On an
`overwrite` restore it first clears any existing pointer family so the
renumbered version-0 pointer is the authoritative maximum (a leftover higher
`p<N>` would otherwise shadow the restore). The deeper backup hardening
(CAS the destination pointer, fence the restore to `current.next()`, a
`--verify` pass) is RFC-pending and tracked separately.

## Alternatives considered

- **Keep `current.json` as an advisory cached hint over the authoritative
  family** (write it via plain unconditional `Overwrite` on every commit, read
  it first to skip the LIST on hot reads). A legitimate later optimization;
  deferred because `load_current` is not hot, and a second pointer
  representation adds a stale-mirror failure mode for marginal benefit.
- **Iceberg-style external catalog CAS.** Regains strong serialization with no
  object-store conditional primitive at all, but adds an external dependency
  and breaks "the bucket *is* the database."
- **Do nothing / bless `LocalFileObjectStore`.** Cheapest, but keeps `If-Match`
  on the commit hot path and the advisory-lock workaround load-bearing — i.e.
  keeps portability gated on the rarer primitive.

## Drawbacks

- `load_current` now costs a LIST of the (GC-bounded, typically 1–2 entry)
  pointer directory plus one HEAD, versus a single GET of `current.json`.
  Mitigated by the family being kept small by the janitor and by `load_current`
  not being on the read hot path.
- The pointer family grows by one object per commit until the janitor reclaims
  it — the same space-amplification profile the manifest bodies already have,
  bounded by the same retention horizon.
- Namespaces written by post-RFC code no longer maintain `current.json`, so an
  in-place downgrade to pre-RFC code would not find a pointer. Forward-only
  migration is acceptable for the current alpha; a downgrade would require
  re-pointing `current.json` at the max `p<N>`.

## Open questions

- Whether to fold the advisory-`current.json`-hint optimization in once a
  workload demonstrates `load_current` LIST latency matters.
- Whether to fully delete the `LocalFileObjectStore` `flock` path or keep it as
  a general-purpose conditional-overwrite shim.

## References

- RFC-001 §"Manifest protocol", §"Epoch fencing", RFC-027 (retention horizon).
- Delta Lake transaction log (PUT-if-absent commit + LIST for current version).
- SlateDB manifest CAS (`If-None-Match` "first to create wins" + writer epoch).
- Object-store conditional-write capability matrix (verified 2026-06): the
  `s3a`/`s3b` audit in `docs/audit/2026-06-14-real-world-gaps.md`.
