//! Row type carried by the tree-walking executor.

use std::collections::BTreeMap;

use super::value::RuntimeValue;

/// Mapping from binding name → runtime value. Bindings introduced by
/// earlier clauses live until a `WITH` resets the scope or until the
/// outer projection of `RETURN` filters them out.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Row {
 pub bindings: BTreeMap<String, RuntimeValue>,
}

impl Row {
 pub fn new() -> Self {
 Self::default()
 }

 pub fn with(mut self, name: impl Into<String>, value: RuntimeValue) -> Self {
 self.bindings.insert(name.into(), value);
 self
 }

 pub fn get(&self, name: &str) -> Option<&RuntimeValue> {
 self.bindings.get(name)
 }

 pub fn set(&mut self, name: impl Into<String>, value: RuntimeValue) {
 self.bindings.insert(name.into(), value);
 }

 pub fn extend(&mut self, other: BTreeMap<String, RuntimeValue>) {
 self.bindings.extend(other);
 }
}

#[cfg(test)]
mod tests {
 use super::*;

 #[test]
 fn row_with_and_get() {
 let row = Row::new()
 .with("a", RuntimeValue::Integer(1))
 .with("b", RuntimeValue::String("x".into()));
 assert_eq!(row.get("a"), Some(&RuntimeValue::Integer(1)));
 assert_eq!(row.get("missing"), None);
 }
}
