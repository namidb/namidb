//! Factorized intermediate result representation (RFC-017).
//!
//! Replaces `Vec<Row>` between operators. Each operator pushes one
//! `FactorNode { parent: FactorIdx, slots: Vec<Slot> }` to a shared
//! `FactorArena` instead of cloning the per-row `BTreeMap`. Inherited
//! bindings are walked up the parent chain lazily; materialisation to a
//! flat `Row` happens only at sinks (`TopN`, `Aggregate`, `Distinct`,
//! final `Project`, etc.).
//!
//! Ownership shape: `FactorRowSet` owns its `FactorArena` by value.
//! Operators that consume one input emit a transformed `FactorRowSet`
//! by `push`-ing into the same arena. Operators that merge two inputs
//! (CrossProduct, HashJoin) `splice_from` the right side into the left
//! arena, offsetting indices. No `Arc` / `RefCell` — keeps everything
//! `Send` for the tokio executor.
//!
//! See `docs/rfc/017-factorization.md` for the full design.

use std::sync::Arc;

use super::row::Row;
use super::value::RuntimeValue;

/// Index into a [`FactorArena`]. `u32` keeps the per-node footprint at
/// 4 bytes (vs 8 for `usize`); 4G nodes per query is well over any
/// realistic working set.
pub type FactorIdx = u32;

/// Pre-allocated root index. `FactorArena::new` always materialises this
/// as the first (empty) node.
pub const FACTOR_ROOT: FactorIdx = 0;

/// One binding introduced at a specific level of the chain.
///
/// `name` is `Arc<str>` so the thousands-of-nodes a typical query produces
/// share the binding name without allocating it once per node.
#[derive(Debug, Clone, PartialEq)]
pub struct Slot {
 pub name: Arc<str>,
 pub value: RuntimeValue,
}

impl Slot {
 pub fn new(name: impl Into<Arc<str>>, value: RuntimeValue) -> Self {
 Self {
 name: name.into(),
 value,
 }
 }
}

/// One factor node — the bindings *added at this level* plus a pointer to
/// the parent node carrying earlier bindings. Walking the chain
/// root-ward yields the complete set of bindings for that leaf.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorNode {
 pub parent: Option<FactorIdx>,
 pub slots: Vec<Slot>,
}

/// Append-only pool of [`FactorNode`]s. Owned by exactly one
/// [`FactorRowSet`] at a time; merge two arenas by `splice_from`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FactorArena {
 nodes: Vec<FactorNode>,
}

impl FactorArena {
 /// Create a new arena with the empty root node at index
 /// [`FACTOR_ROOT`].
 pub fn new() -> Self {
 Self {
 nodes: vec![FactorNode {
 parent: None,
 slots: Vec::new(),
 }],
 }
 }

 /// The root index — always valid in a `new()`'d arena.
 pub fn root(&self) -> FactorIdx {
 FACTOR_ROOT
 }

 /// Number of nodes (incl. root).
 pub fn len(&self) -> usize {
 self.nodes.len()
 }

 /// `true` iff arena is empty (no root). Constructed arenas via `new()`
 /// are never empty; this only returns true for `FactorArena::default()`
 /// which constructs an empty pool.
 pub fn is_empty(&self) -> bool {
 self.nodes.is_empty()
 }

 /// Push a new node with `parent` and `slots`. Returns the new index.
 /// `parent` must be a valid prior index.
 pub fn push(&mut self, parent: FactorIdx, slots: Vec<Slot>) -> FactorIdx {
 debug_assert!(
 (parent as usize) < self.nodes.len(),
 "parent {} out of range (len={})",
 parent,
 self.nodes.len()
 );
 let idx = self.nodes.len() as FactorIdx;
 self.nodes.push(FactorNode {
 parent: Some(parent),
 slots,
 });
 idx
 }

 /// Borrow a node by index. Panics if out of range — callers should
 /// only ever pass indices obtained from `push` / `root`.
 pub fn node(&self, idx: FactorIdx) -> &FactorNode {
 &self.nodes[idx as usize]
 }

 /// Walk the parent chain `leaf → root` accumulating bindings into a
 /// flat [`Row`]. Child slots **shadow** parent slots with the same
 /// name (the deeper, more recent binding wins).
 ///
 /// If `projection` is `Some(&names)`, only slots whose `name` is in
 /// `names` are added to the row. This is the hook for column pruning
 /// at sinks (combined with RFC-015 projection pushdown).
 pub fn materialize(&self, leaf: FactorIdx, projection: Option<&[&str]>) -> Row {
 let mut row = Row::new();
 let mut cur = Some(leaf);
 while let Some(idx) = cur {
 let node = &self.nodes[idx as usize];
 for slot in &node.slots {
 if let Some(filter) = projection {
 if !filter.iter().any(|n| **n == *slot.name) {
 continue;
 }
 }
 // entry() + or_insert_with so the first occurrence (the
 // deepest, child-most) wins. Subsequent ancestor hits with
 // the same name are silently skipped.
 row.bindings
 .entry(slot.name.to_string())
 .or_insert_with(|| slot.value.clone());
 }
 cur = node.parent;
 }
 row
 }

 /// Look up a single binding by walking root-ward. Returns the first
 /// (deepest, child-most) hit, or `None` if no ancestor declares it.
 /// O(depth) — sufficient for LDBC-style depth ≤ 5 patterns.
 pub fn lookup_binding(&self, leaf: FactorIdx, name: &str) -> Option<&RuntimeValue> {
 let mut cur = Some(leaf);
 while let Some(idx) = cur {
 let node = &self.nodes[idx as usize];
 for slot in &node.slots {
 if &*slot.name == name {
 return Some(&slot.value);
 }
 }
 cur = node.parent;
 }
 None
 }

 /// Splice all nodes of `other` into `self`. Indices in `other` are
 /// shifted by the current `self.len()`, so the caller can translate
 /// foreign indices via the returned `offset`. The root of `other` is
 /// *not* skipped — callers re-parent foreign roots themselves via
 /// `splice_under` if they need to splice a sub-chain under a new
 /// parent.
 ///
 /// Returns the offset to add to a foreign `FactorIdx` to get its
 /// position in `self` after splicing.
 pub fn splice_from(&mut self, other: &FactorArena) -> FactorIdx {
 let offset = self.nodes.len() as FactorIdx;
 self.nodes.reserve(other.nodes.len());
 for node in &other.nodes {
 let translated_parent = node.parent.map(|p| p + offset);
 self.nodes.push(FactorNode {
 parent: translated_parent,
 slots: node.slots.clone(),
 });
 }
 offset
 }

 /// Re-parent a foreign chain so its top node points at `new_parent`
 /// in `self`. The foreign chain was already spliced via `splice_from`;
 /// `foreign_leaf` is its tip (already translated to `self`-relative
 /// index). Walk up until we hit the (translated) foreign root, and
 /// re-point that root's parent from `None`-translated-to-offset+ROOT
 /// to the supplied `new_parent`.
 ///
 /// Returns `foreign_leaf` unchanged — only the topology of the
 /// chain is mutated.
 ///
 /// Used by `cross_product_factor` to glue right-side leaves under
 /// each left-side leaf without copying right's nodes once per
 /// `(l, r)` pair.
 pub fn splice_under(&mut self, new_parent: FactorIdx, foreign_leaf: FactorIdx) -> FactorIdx {
 // Walk up the chain to find the node whose parent is the
 // translated foreign root. Its parent slot is what we mutate.
 // A "foreign root" is a node whose parent is None (the original
 // FACTOR_ROOT of the foreign arena, translated). We find any
 // ancestor with `parent == None` and re-parent it to `new_parent`.
 let mut cur = foreign_leaf;
 loop {
 let node = &self.nodes[cur as usize];
 match node.parent {
 Some(p) => {
 // Check if p is itself a root candidate (slots empty,
 // parent None — the translated foreign root). If so,
 // we want to bypass p and connect cur directly to
 // new_parent.
 if self.nodes[p as usize].parent.is_none()
 && self.nodes[p as usize].slots.is_empty()
 {
 self.nodes[cur as usize].parent = Some(new_parent);
 break;
 }
 cur = p;
 }
 None => {
 // cur is itself a root. Skip — root has no slots to
 // carry and shouldn't normally appear in a chain
 // produced by `push`.
 break;
 }
 }
 }
 foreign_leaf
 }
}

/// The output of an operator. Owned `arena` + the list of valid leaf
/// indices into it. Operators that take a single input typically
/// transform `(arena, leaves)` → `(arena', leaves')` where `arena'`
/// extends `arena`; binary operators splice the right arena into the
/// left one.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorRowSet {
 pub arena: FactorArena,
 pub leaves: Vec<FactorIdx>,
}

impl FactorRowSet {
 /// Construct an empty set with a fresh arena (just the root) and
 /// the root as the sole leaf — the canonical input for operators
 /// that emit rows from nothing (e.g. `LogicalPlan::Empty`).
 pub fn singleton_root() -> Self {
 let arena = FactorArena::new();
 let root = arena.root();
 Self {
 arena,
 leaves: vec![root],
 }
 }

 /// Construct from an iterator of flat rows. Each row becomes a leaf
 /// node whose slots are the row's bindings, parent = root. Useful
 /// for adapter points where a flat operator emits into the factor
 /// world.
 pub fn from_flat<I: IntoIterator<Item = Row>>(rows: I) -> Self {
 let mut arena = FactorArena::new();
 let root = arena.root();
 let mut leaves = Vec::new();
 for row in rows {
 let slots: Vec<Slot> = row
 .bindings
 .into_iter()
 .map(|(k, v)| Slot::new(Arc::from(k.as_str()), v))
 .collect();
 let idx = arena.push(root, slots);
 leaves.push(idx);
 }
 Self { arena, leaves }
 }

 /// Materialise every leaf into a flat `Row`. Used by sinks that do
 /// not benefit from staying factorised (Project final, simple
 /// RETURN). For `TopN` / `Aggregate` callers prefer the per-leaf
 /// `materialize` directly to avoid building the full Vec.
 pub fn materialize_all(&self, projection: Option<&[&str]>) -> Vec<Row> {
 self.leaves
 .iter()
 .map(|&l| self.arena.materialize(l, projection))
 .collect()
 }

 /// Number of leaves (output cardinality).
 pub fn cardinality(&self) -> usize {
 self.leaves.len()
 }
}

/// Read the `NAMIDB_FACTORIZE` env var. Empty / unset / "0" / "false" /
/// "no" / "off" → flat path. Any of "1" / "true" / "yes" / "on" (case-
/// insensitive) → factor path. Default while factorization is in progress: **off**.
///
/// The default flips to ON when all operators have a factor
/// implementation and parity tests are green). RFC-017 §"Open questions"
/// tracks the flip.
pub fn factorize_enabled() -> bool {
 match std::env::var("NAMIDB_FACTORIZE") {
 Ok(v) => matches!(
 v.trim().to_ascii_lowercase().as_str(),
 "1" | "true" | "yes" | "on"
 ),
 Err(_) => false,
 }
}

#[cfg(test)]
mod tests {
 use super::*;

 fn s(name: &str, v: RuntimeValue) -> Slot {
 Slot::new(Arc::from(name), v)
 }

 fn int(n: i64) -> RuntimeValue {
 RuntimeValue::Integer(n)
 }

 fn string(s: &str) -> RuntimeValue {
 RuntimeValue::String(s.into())
 }

 #[test]
 fn arena_root_is_empty() {
 let arena = FactorArena::new();
 assert_eq!(arena.len(), 1);
 let row = arena.materialize(arena.root(), None);
 assert!(row.bindings.is_empty());
 }

 #[test]
 fn single_push_then_materialize() {
 let mut arena = FactorArena::new();
 let root = arena.root();
 let idx = arena.push(root, vec![s("p", int(42))]);
 let row = arena.materialize(idx, None);
 assert_eq!(row.bindings.len(), 1);
 assert_eq!(row.bindings.get("p"), Some(&int(42)));
 }

 #[test]
 fn chain_inherits_parent() {
 let mut arena = FactorArena::new();
 let root = arena.root();
 let p = arena.push(root, vec![s("p", int(1))]);
 let f = arena.push(p, vec![s("f", int(2))]);
 let fof = arena.push(f, vec![s("fof", int(3))]);
 let row = arena.materialize(fof, None);
 assert_eq!(row.bindings.len(), 3);
 assert_eq!(row.bindings.get("p"), Some(&int(1)));
 assert_eq!(row.bindings.get("f"), Some(&int(2)));
 assert_eq!(row.bindings.get("fof"), Some(&int(3)));
 }

 #[test]
 fn materialize_with_projection() {
 let mut arena = FactorArena::new();
 let root = arena.root();
 let p = arena.push(root, vec![s("p", int(1)), s("p_age", int(30))]);
 let f = arena.push(p, vec![s("f", int(2)), s("f_age", int(40))]);
 let row = arena.materialize(f, Some(&["p", "f"]));
 assert_eq!(row.bindings.len(), 2);
 assert!(row.bindings.contains_key("p"));
 assert!(row.bindings.contains_key("f"));
 assert!(!row.bindings.contains_key("p_age"));
 assert!(!row.bindings.contains_key("f_age"));
 }

 #[test]
 fn child_shadows_parent() {
 let mut arena = FactorArena::new();
 let root = arena.root();
 // Parent binds `x` to 1, child rebinds it to 2 — Cypher WITH
 // semantics where a later binding overrides an earlier one with
 // the same name.
 let p = arena.push(root, vec![s("x", int(1))]);
 let c = arena.push(p, vec![s("x", int(2))]);
 let row = arena.materialize(c, None);
 assert_eq!(row.bindings.get("x"), Some(&int(2)));
 }

 #[test]
 fn lookup_binding_walks_parent_chain() {
 let mut arena = FactorArena::new();
 let root = arena.root();
 let p = arena.push(root, vec![s("p", string("Alice"))]);
 let f = arena.push(p, vec![s("f", string("Bob"))]);
 let fof = arena.push(f, vec![s("fof", string("Carol"))]);
 // Lookup at the leaf finds bindings from every ancestor.
 assert_eq!(arena.lookup_binding(fof, "p"), Some(&string("Alice")));
 assert_eq!(arena.lookup_binding(fof, "f"), Some(&string("Bob")));
 assert_eq!(arena.lookup_binding(fof, "fof"), Some(&string("Carol")));
 assert_eq!(arena.lookup_binding(fof, "missing"), None);
 // Lookup at the middle node only sees ancestors up to that node.
 assert_eq!(arena.lookup_binding(f, "p"), Some(&string("Alice")));
 assert_eq!(arena.lookup_binding(f, "f"), Some(&string("Bob")));
 assert_eq!(arena.lookup_binding(f, "fof"), None);
 }

 #[test]
 fn splice_from_translates_indices() {
 let mut left = FactorArena::new();
 let l_root = left.root();
 let l1 = left.push(l_root, vec![s("L", int(10))]);

 let mut right = FactorArena::new();
 let r_root = right.root();
 let r1 = right.push(r_root, vec![s("R", int(20))]);
 let _ = right.push(r1, vec![s("R2", int(21))]);

 let offset = left.splice_from(&right);
 // Left now contains its own 2 nodes plus right's 3 (root + r1 + r2).
 assert_eq!(left.len(), 2 + right.len());
 // Original left bindings still resolve from l1.
 assert_eq!(left.lookup_binding(l1, "L"), Some(&int(10)));
 // Translated right tip is at r2_translated = offset + 2.
 let r2_translated = offset + 2;
 assert_eq!(left.lookup_binding(r2_translated, "R"), Some(&int(20)));
 assert_eq!(left.lookup_binding(r2_translated, "R2"), Some(&int(21)));
 }

 #[test]
 fn splice_under_reparents_foreign_chain() {
 // Build left: root → l1{L=10}.
 let mut left = FactorArena::new();
 let l_root = left.root();
 let l1 = left.push(l_root, vec![s("L", int(10))]);

 // Build right: root → r1{R=20} → r2{R2=21}.
 let mut right = FactorArena::new();
 let r_root = right.root();
 let r1 = right.push(r_root, vec![s("R", int(20))]);
 let r2 = right.push(r1, vec![s("R2", int(21))]);

 // Splice right into left, then re-parent right's r2 under l1.
 let offset = left.splice_from(&right);
 let r2_translated = r2 + offset;
 left.splice_under(l1, r2_translated);

 // Now materialising r2_translated should see both L (from left)
 // and R/R2 (from right re-parented).
 let row = left.materialize(r2_translated, None);
 assert_eq!(row.bindings.len(), 3);
 assert_eq!(row.bindings.get("L"), Some(&int(10)));
 assert_eq!(row.bindings.get("R"), Some(&int(20)));
 assert_eq!(row.bindings.get("R2"), Some(&int(21)));
 }

 #[test]
 fn rowset_from_flat_round_trip() {
 let r1 = Row::new().with("a", int(1)).with("b", string("x"));
 let r2 = Row::new().with("a", int(2)).with("b", string("y"));
 let set = FactorRowSet::from_flat(vec![r1.clone(), r2.clone()]);
 assert_eq!(set.cardinality(), 2);
 let flat = set.materialize_all(None);
 // BTreeMap iteration ordering is by key — assert by exact match.
 assert_eq!(flat.len(), 2);
 assert!(flat.contains(&r1));
 assert!(flat.contains(&r2));
 }

 #[test]
 fn singleton_root_has_one_leaf() {
 let set = FactorRowSet::singleton_root();
 assert_eq!(set.cardinality(), 1);
 let rows = set.materialize_all(None);
 assert_eq!(rows.len(), 1);
 assert!(rows[0].bindings.is_empty());
 }

 #[test]
 fn cardinality_after_fanout() {
 // Simulate a single-hop Expand from p (1 row) producing 3
 // friend bindings. Cardinality should be 3.
 let mut set = FactorRowSet::singleton_root();
 let parent = set.leaves[0];
 let new_leaves: Vec<FactorIdx> = (0..3)
 .map(|i| set.arena.push(parent, vec![s("friend", int(i))]))
 .collect();
 set.leaves = new_leaves;
 assert_eq!(set.cardinality(), 3);
 let rows = set.materialize_all(None);
 assert_eq!(rows.len(), 3);
 // Each row has the friend binding with the expected value.
 let friends: Vec<i64> = rows
 .iter()
 .filter_map(|r| {
 r.bindings.get("friend").and_then(|v| match v {
 RuntimeValue::Integer(n) => Some(*n),
 _ => None,
 })
 })
 .collect();
 let mut sorted = friends;
 sorted.sort_unstable();
 assert_eq!(sorted, vec![0, 1, 2]);
 }
}
