# Restore points (point-in-time backup)

## What it gives you

Every durable change to a namespace (a commit, a flush, a compaction, a DDL
statement) publishes an immutable manifest version stamped with its
`created_at`. A version is a complete description of the namespace at that
moment: its SSTs and side-cars plus the WAL segments not yet flushed into
them. The namespace's state at an instant `T` is therefore the newest version
committed at or before `T`, and copying that version's closure reproduces the
namespace row for row as it was at `T`, whatever was written, deleted,
flushed or compacted afterwards.

With `NAMIDB_RESTORE_POINT_RETENTION` set, the server keeps every version of
the last `retention` and everything those versions reference:

- `GET /v0/admin/restore-points` lists the window (`earliest`, `latest`, and
  pages of points, newest first).
- `POST /v0/admin/backup` with `as_of` (an RFC 3339 instant or a manifest
  version) copies the namespace exactly as it was then. Restoring that backup
  into a new namespace (`POST /:namespace/v0/admin/restore`, or `namidb
  restore`) completes a point-in-time restore.

The HTTP contract is in [`crates/namidb-server/README.md`](../../crates/namidb-server/README.md).
Without retention, the window is just the current version.

## How history is kept

Normally the orphan sweep (`janitor.rs`) reclaims a manifest version, and
every object only it references, once no reader needs it and the object is
older than `NAMIDB_SWEEP_MIN_AGE`. Retention pins (`pin.rs`) are how a reader
outside the process holds history back. Each lease now has a scope:

| Scope | Holder | The sweep keeps |
|---|---|---|
| `floor` (default, and how older releases read every lease) | readers | every version from the lease's version to current, and everything they reference |
| `point` | a running backup | exactly the lease's version and its closure |
| `window` | the restore point window | every manifest body from the lease's version up, that version's closure, and every SST, side-car and WAL object written since the window opened (less `min_age`) |

The window is one lease per namespace, `manifest/pins/restore-points.json`.
Its version is the window's **floor**: the newest version committed at or
before `now - retention`, the state at the start of the window. The writer's
maintenance loop refreshes it before each sweep (`refresh_restore_point_lease`
in `restore_point.rs`), finding the new floor by binary search over the
retained manifests, so the floor only moves forward and the refresh costs
`O(log commits)` manifest reads.

Why keeping by write time is exact: an object that a version inside the
window references is either in the floor version's closure, or was first
referenced by a later version, and so uploaded after the floor committed
(less the upload-to-commit window that `min_age` already bounds for every
sweep), because a manifest never re-adds an object an earlier version
dropped. That lets the sweep honour a window of any number of versions by
loading one extra manifest, instead of loading every version in the window
on every sweep. A backup of an old point pins only that version (`point`),
for the same reason.

The lease expires `retention + max(2 × compaction interval, 15 min)` after
each refresh. If the writer stops refreshing it (the process stops, the
namespace is evicted, or retention is turned off without a release), every
point it protected has aged out of the window by the time it expires. On
refresh only an unexpired lease is trusted: once a lease lapses a sweep may
already have reclaimed history, so the window restarts at the current version
and fills up from there. It also starts empty the first time retention is
turned on. Setting the retention back to `0` deletes the lease when the
maintenance loop next starts.

Mixing releases is safe: an older janitor reads a `point` or `window` lease
as a `floor`, which keeps strictly more (it is only slower, since it loads
every version above the floor). A scope this build does not know is read as a
`floor` for the same reason.

## Storage cost

The window keeps everything the namespace writes during `retention + min_age`:

- **Manifest bodies:** one per commit. A body lists every SST, so its size
  grows with the namespace (a few KiB for a small namespace, more for one
  with thousands of SSTs). `commits in window × body size`.
- **WAL segments:** one per commit (one per group with group commit), the
  size of the batch.
- **Flush outputs and superseded SSTs:** every L0 SST flushed in the window,
  and every SST a compaction in the window replaced, stay until they age out.
  For a namespace that keeps rewriting its data through compaction this is
  roughly `retention × ingest rate × compaction write amplification`.

The sweep logs `restore_window_retained` / `restore_window_bytes`
(`JanitorReport`) for the objects kept only for the window. Two second-order
costs: the sweep, and a writer's cold open, LIST the `manifest/` and `wal/`
prefixes, which grow with the number of commits kept; and every point listed
by `GET /v0/admin/restore-points` is one manifest read. As a rule of thumb,
a namespace that ingests 1 GiB/day with a 7-day window and ~3× compaction
amplification keeps on the order of 20 GiB of history on top of its live data,
plus its manifests.

## Limits

- Restore points resolve to the newest version committed at or before the
  requested instant, by the writer's clock (`created_at`). Two versions
  committed in the same instant resolve to the newer.
- A backup at the very edge of the window can race the floor moving past
  it: the copy then fails with 409 (the pinned version was reclaimed) and a
  retry resolves the instant again.
- In multi-tenant mode a namespace's window advances only while it is open.
