//! Object-store key conventions for a namespace.
//!
//! Layout (everything under `<root_prefix>/<namespace>/`):
//!
//! ```text
//! manifest/current.json            (legacy pointer; read fallback only)
//! manifest/v00000001.json
//! manifest/v00000002.json
//! manifest/pointer/p00000001.json  (Create-only pointer family, RFC-029)
//! manifest/pointer/p00000002.json
//! wal/00000001.wal
//! wal/00000002.wal
//! sst/level0/01J5XY7K...-nodes-Person.parquet
//! sst/level0/01J5XY7K...-edges-KNOWS.csr
//! sst/level1/...
//! snapshots/<name>.json
//! ```
//!
//! We render manifest versions and WAL segments as zero-padded 16-digit
//! hex; SST identifiers as UUIDv7 strings. The padding keeps lexicographic
//! order aligned with numeric order so `object_store::list` returns
//! candidates in a predictable sequence.

use std::fmt::Write as _;

use namidb_core::NamespaceId;
use object_store::path::Path;

/// Wrapper around a root prefix + namespace that knows how to render every
/// canonical key the storage engine uses.
#[derive(Debug, Clone)]
pub struct NamespacePaths {
    root: String, // normalised: no leading/trailing slash
    namespace: NamespaceId,
}

impl NamespacePaths {
    /// `root_prefix` may be empty (everything lives at the bucket root) or
    /// contain slashes (e.g. `tenants/prod`). It is normalised on entry.
    pub fn new(root_prefix: impl Into<String>, namespace: NamespaceId) -> Self {
        let mut root = root_prefix.into();
        while root.starts_with('/') {
            root.remove(0);
        }
        while root.ends_with('/') {
            root.pop();
        }
        Self { root, namespace }
    }

    pub fn namespace(&self) -> &NamespaceId {
        &self.namespace
    }
    pub fn root_prefix(&self) -> &str {
        &self.root
    }

    /// `<root>/<ns>/` — used as a base prefix for `list` operations.
    pub fn namespace_prefix(&self) -> Path {
        self.join(&[])
    }

    pub fn manifest_dir(&self) -> Path {
        self.join(&["manifest"])
    }
    /// Legacy single mutable pointer (`manifest/current.json`). No longer
    /// written by the commit path after RFC-029; kept for reading namespaces
    /// bootstrapped before the pointer family and snapshots produced by the
    /// pre-RFC backup path.
    pub fn current_pointer(&self) -> Path {
        self.join(&["manifest", "current.json"])
    }
    pub fn manifest_version(&self, version: u64) -> Path {
        self.join(&["manifest", &format!("v{}.json", pad_hex(version, 16))])
    }
    /// Directory holding the Create-only versioned pointer family
    /// (`manifest/pointer/p<N>.json`, RFC-029). Used as the LIST prefix when
    /// resolving the current pointer (the highest `N` present).
    pub fn pointer_dir(&self) -> Path {
        self.join(&["manifest", "pointer"])
    }
    /// Pointer object for manifest `version` (`manifest/pointer/p<16hex>.json`,
    /// RFC-029). Created write-once with `PutMode::Create`; `version` matches
    /// the manifest body the pointer names.
    pub fn pointer_version(&self, version: u64) -> Path {
        self.join(&[
            "manifest",
            "pointer",
            &format!("p{}.json", pad_hex(version, 16)),
        ])
    }
    /// Directory holding retention pin leases (`manifest/pins/<uuid>.json`).
    /// Listed by the orphan sweep before it deletes anything; see
    /// [`crate::pin`].
    pub fn pins_dir(&self) -> Path {
        self.join(&["manifest", "pins"])
    }
    /// Lease object for one retention pin holder. `id` is the holder's own
    /// UUID; the sweep identifies leases by prefix, not by name.
    pub fn pin_object(&self, id: &str) -> Path {
        self.join(&["manifest", "pins", &format!("{id}.json")])
    }
    pub fn wal_dir(&self) -> Path {
        self.join(&["wal"])
    }
    pub fn wal_segment(&self, seq: u64) -> Path {
        self.join(&["wal", &format!("{}.wal", pad_hex(seq, 16))])
    }
    pub fn sst_dir(&self, level: u32) -> Path {
        self.join(&["sst", &format!("level{level}")])
    }
    pub fn sst_object(&self, level: u32, file_name: &str) -> Path {
        self.join(&["sst", &format!("level{level}"), file_name])
    }
    pub fn snapshots_dir(&self) -> Path {
        self.join(&["snapshots"])
    }
    pub fn snapshot(&self, name: &str) -> Path {
        self.join(&["snapshots", &format!("{name}.json")])
    }

    /// Path for the optional bincode-serialised memtable snapshot
    /// used to speed up cold starts (RFC-pending). When this object
    /// exists, `recover_memtable` consumes it instead of replaying the
    /// full WAL segment history.
    pub fn memtable_snapshot(&self) -> Path {
        self.join(&["memtable_snapshot.bin"])
    }

    fn join(&self, segments: &[&str]) -> Path {
        let mut buf = String::with_capacity(64);
        if !self.root.is_empty() {
            buf.push_str(&self.root);
            buf.push('/');
        }
        buf.push_str(self.namespace.as_str());
        for s in segments {
            buf.push('/');
            buf.push_str(s);
        }
        Path::from(buf)
    }
}

fn pad_hex(n: u64, width: usize) -> String {
    let mut out = String::with_capacity(width);
    write!(&mut out, "{n:0width$x}").expect("write to string never fails");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ns(name: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
    }

    #[test]
    fn well_known_paths() {
        let p = ns("acme");
        assert_eq!(
            p.current_pointer().as_ref(),
            "tenants/acme/manifest/current.json"
        );
        assert_eq!(
            p.manifest_version(1).as_ref(),
            "tenants/acme/manifest/v0000000000000001.json"
        );
        assert_eq!(p.pointer_dir().as_ref(), "tenants/acme/manifest/pointer");
        assert_eq!(
            p.pointer_version(1).as_ref(),
            "tenants/acme/manifest/pointer/p0000000000000001.json"
        );
        assert_eq!(p.pins_dir().as_ref(), "tenants/acme/manifest/pins");
        assert_eq!(
            p.pin_object("abc-123").as_ref(),
            "tenants/acme/manifest/pins/abc-123.json"
        );
        assert_eq!(
            p.wal_segment(42).as_ref(),
            "tenants/acme/wal/000000000000002a.wal"
        );
        assert_eq!(p.sst_dir(2).as_ref(), "tenants/acme/sst/level2");
        assert_eq!(
            p.snapshot("named").as_ref(),
            "tenants/acme/snapshots/named.json"
        );
    }

    #[test]
    fn empty_root_prefix_is_supported() {
        let p = NamespacePaths::new("", NamespaceId::new("acme").unwrap());
        assert_eq!(p.current_pointer().as_ref(), "acme/manifest/current.json");
    }

    #[test]
    fn leading_and_trailing_slashes_are_stripped() {
        let p = NamespacePaths::new("/tenants/prod/", NamespaceId::new("acme").unwrap());
        // Whatever `object_store::path::Path::from` does to internal slashes is
        // its business; we only promise that we don't end up with a `//` glued
        // at the boundary between root prefix and namespace.
        let rendered = p.current_pointer().as_ref().to_string();
        assert!(
            rendered.starts_with("tenants/prod/acme/"),
            "unexpected layout: {rendered}"
        );
        assert!(rendered.ends_with("manifest/current.json"));
    }
}
