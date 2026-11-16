# Storage format versions and upgrades

## The stamp

Every manifest version records the storage format it was written in, as
`storage_format` in the manifest JSON:

```json
"storage_format": {
  "version": 2,
  "min_reader_version": 2,
  "min_writer_version": 2,
  "written_by": "namidb 3.4.0"
}
```

- `version` is the format of the release that committed this manifest
  version.
- `min_reader_version` is the oldest format a reader must support to read
  this version and everything it references.
- `min_writer_version` is the oldest format a writer must support to commit
  on top of it. A writer that does not know a field drops it when it
  re-serialises the manifest, so a release may need to keep older writers
  out even when it lets them read.
- `written_by` names the release, for operators.

Requirements only grow: a release that commits on top of a manifest written
by a newer one keeps the newer `min_reader_version` and `min_writer_version`.
The stamp is plain JSON in `manifest/v<N>.json`, so tools outside the engine
can read the format of a namespace without linking it: follow
`manifest/current.json` (or the highest `manifest/pointer/p<N>.json`) to the
manifest body and read `storage_format`; a missing field means 3.3.1 or
older wrote it. The server reports the format it supports as
`storage_format` in `GET /v0/version`, so a deployment can compare the two
before routing a namespace to a server.

## Format numbers

| Format | Releases | Node SST layout |
|--------|----------|-----------------|
| 1 | 2.x | every ad-hoc property, embeddings included, in the `__overflow_json` text column; JSON exact records in `.nloc2` |
| 2 | 3.x | numeric arrays and byte strings in a binary `__binary_props` column; binary exact records |

Manifests committed by 3.3.1 and older carry no stamp. For those the engine
infers the format from the data: format 2 when any WAL segment or node SST
needs a 3.x reader, format 1 when everything can still be read by 2.x, and
no format at all for a namespace with no data. Only node SSTs and the WAL
changed between 2.x and 3.x; edge SSTs did not.

## What each access does

| Access | Manifest needs a newer reader | Manifest needs a newer writer | Format 1 data, unstamped |
|--------|-------------------------------|-------------------------------|--------------------------|
| Read (`read_only=True`, `ReaderSession`, backup) | refused | allowed | allowed, nothing written |
| Python `Client(uri)` | refused | refused | refused, unless `upgrade=True` |
| Server, CLI, Rust `WriterSession::open` | refused (server: 409, `code: "unsupported_storage_format"`) | refused | opened; new writes use format 2 |

A refusal happens before anything is claimed or committed, so the namespace
is left exactly as it was. The errors are
`Error::NewerStorageFormat` and `Error::StorageFormatUpgradeRequired` in
Rust, and `namidb.StorageFormatError` in Python. The janitor loads manifests
through the same check, so a sweep never deletes objects of a manifest it
cannot read.

Opening never rewrites existing SSTs. With the upgrade allowed, the writer's
first claim stamps the manifest, and from then on everything it writes (WAL
segments, flushed and compacted SSTs) is in the current format. Old SSTs
keep their layout until a compaction merges them with newer data or the
migration below rewrites them.

## Upgrading a 2.x namespace

1. Take a backup with the release you are leaving, or with the new one:
   `namidb backup --from <uri> --to <backup-uri>`. Once anything is written
   in the new format, 2.x can no longer read the namespace, and going back
   means restoring that backup.
2. Open it with the new release. In Python, pass `upgrade=True` once; the
   stamp it writes records the upgrade and later opens need no flag.
3. Optionally rewrite the old node SSTs, so reads stop parsing embeddings
   out of JSON text:

   ```bash
   namidb migrate --store "file:///var/lib/namidb?ns=prod" --dry-run
   namidb migrate --store "file:///var/lib/namidb?ns=prod"
   ```

   or from Python:

   ```python
   client = namidb.Client(uri, upgrade=True)
   client.storage_format()          # what is there, and how much is 2.x
   client.migrate(progress=print)   # rewrite it
   ```

## The migration

`migrate` rewrites one 2.x node SST at a time:

1. The SST is read and written out again in the current layout with exactly
   the same rows: same ids, LSNs, labels and properties, tombstones kept,
   nothing merged or collected, at the same level and key range, with fresh
   sidecars (locator, property pages, label index, unique and equality
   indexes). Search indexes are untouched; their corpus does not change.
2. The new SST is read back from the object store and compared row by row
   with the old one. The rewrite stops with an error on any difference.
3. One manifest commit swaps the old SST for the new one, under the writer
   lock, exactly like a compaction install.

Each SST is its own commit, so a crash, an error or Ctrl-C leaves every SST
either fully migrated or untouched, never both or neither. Running
`migrate` again lists the SSTs still in the 2.x layout and continues with
those. The objects of an attempt that never committed, and the replaced
SSTs, are unreferenced garbage for the orphan sweep. Reads keep running
during a migration, and writes too: only the install of each SST takes the
writer lock. In Python a compaction (`compact()`, or the one `flush()`
triggers) also keeps running: the migration holds the compaction lock for
one SST at a time, so a compaction waits for at most that SST. Migration needs free space for one rewritten SST at a time on
top of the namespace.

The migration is explicit only. A compaction does not rewrite a lone,
fully compacted 2.x SST on its own: for a vector corpus of many gigabytes
that is an unplanned full rewrite, and it would also make the namespace
unreadable to 2.x as a side effect of a maintenance tick.
