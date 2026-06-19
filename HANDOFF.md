# Handoff: s3b Versioned Pointer Implementation

**Branch:** `feat/s3b-versioned-pointer`
**Status:** Functionally complete, 5 bugs identified requiring fixes
**All tests:** 301 passing in `namidb-storage`

## Immediate Work: Fix 5 Bugs from Adversarial Review

The adversarial review (via `/adversarial-reviewer`) identified 5 bugs in the RFC-029 implementation:

### 1. [HIGH] Empty LIST misclassified as uninitialized (Finding 1)
**File:** `crates/namidb-storage/src/manifest.rs` - `load_pointer()`

**Problem:** On eventually-consistent-LIST stores, a transiently-empty/stale LIST makes an initialized namespace look empty, causing spurious bootstrap.

**Fix:** Before treating an empty family LIST as proof of non-existence, HEAD `manifest/pointer/p0.json` (and/or `manifest/v0.json`). GET/HEAD of a specific key is read-after-write consistent on every targeted store.

**Location:** Around line ~440 in `load_pointer()`, where empty LIST is returned.

### 2. [MEDIUM] Forward probe gap-safety false after GC (Finding 2)
**File:** `crates/namidb-storage/src/manifest.rs` - `probe_pointer_forward()`

**Problem:** The stated invariant ("a stale LIST can only lag behind current, never skip a hole") is false once the janitor has GC'd low pointers. A LIST that returns `[p5, p6, p7]` when p3 exists can cause the forward probe to skip p3-p4 and land on p5.

**Fix:** Document or adjust the forward probe to account for GC-created gaps. The comment claiming gap-safety needs updating.

**Location:** `probe_pointer_forward()` function and its doc comment.

### 3. [MEDIUM] Missing test for forward probe advancing branch (Finding 5)
**File:** `crates/namidb-storage/src/manifest.rs` - tests

**Problem:** No test coverage for the forward probe actually advancing past the initial LIST result.

**Fix:** Add a test that creates pointer versions, simulates a stale LIST (by mocking or delaying), and verifies the forward probe finds the newer version.

**Location:** In the `#[cfg(test)]` mod of `manifest.rs`.

### 4. [LOW] Overwrite restore leaves stale `current.json` (Finding 4)
**File:** `crates/namidb-storage/src/backup.rs` - `copy_namespace_snapshot()`

**Problem:** When restoring with `overwrite=true`, any legacy `current.json` at the destination is left intact alongside the new `pointer/p0.json`, creating ambiguity.

**Fix:** In the `overwrite` block that clears existing pointer family, also delete `current_pointer()` (legacy `current.json`) if it exists.

**Location:** Around lines 181-189 in `backup.rs`.

### 5. [LOW] MAX_PROBE=256 can return stale base (Finding 3)
**File:** `crates/namidb-storage/src/manifest.rs` - `load_pointer()`

**Problem:** With aggressive GC and a long stale LIST, `MAX_PROBE=256` might return a pointer that's no longer the current base, causing `WriterFence` to falsely flag `OrphanManifestBody`.

**Fix:** Either increase `MAX_PROBE` significantly, or add a verification HEAD of the returned pointer before accepting it as the base.

**Location:** `load_pointer()` where `MAX_PROBE` is used.

---

## Files Modified (Current State)

```
M crates/namidb-server/src/lib.rs
M crates/namidb-storage/src/backup.rs
M crates/namidb-storage/src/janitor.rs
M crates/namidb-storage/src/local.rs
M crates/namidb-storage/src/manifest.rs
M crates/namidb-storage/src/paths.rs
M docs/rfc/001-storage-engine.md
A docs/rfc/029-versioned-pointer.md
```

## Roadmap After s3b Completion

From `docs/audit/2026-06-14-real-world-gaps.md`, prioritized order:

### Next Wave (immediately after s3b):
- **Item 14:** Multi-tenant foundations (partition key routing)
- **Item 13:** Hybrid search integration (S3 + vector index)
- **Item 08:** CALL/YIELD procedure support + algorithms
- **Item 15:** OIDC/JWT auth + RBAC
- **Item 16:** Backup CAS + restore fence
- **Item 11:** Typed errors across server

### Later Wave:
- **Item 12:** ANN HNSW vector index

### Already Shipped:
- Item 0 (Release 0.18.0) - tags exist on main

## Commands for Next Agent

```bash
# Verify working tree
git status

# Run storage tests
cargo test -p namidb-storage

# Run clippy (with CI flags)
cargo clippy -p namidb-storage -- -D warnings -W clippy::all -W clippy::pedantic

# After bugs fixed, commit:
git add -A
git commit -m "fix(storage): address adversarial review findings in RFC-029

- Fix empty LIST misclassification (HEAD p0 before falling back)
- Fix overwrite restore leaving stale current.json
- Add test coverage for forward probe advancing
- Adjust MAX_PROBE for stale LIST + GC edge case
- Document forward probe gap-safety after GC"

# Push (awaiting user confirmation for org repo)
git push origin feat/s3b-versioned-pointer
```

## Context Notes

- **RFC-029 Design:** Create-only versioned pointer family replacing mutable `current.json` for S3 portability
- **Primitive Used:** `PutMode::Create` (If-None-Match:*) for both manifest bodies and pointers
- **Retired:** `PutMode::Update` (If-Match) dependency removed
- **Resolution:** LIST + forward HEAD probe to resolve current pointer
- **Janitor:** GC reclaims old pointers below retention horizon

---
Generated: 2025-01-19 (session handoff)
