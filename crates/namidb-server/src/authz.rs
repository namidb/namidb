//! Pre-execution authorization hook (RFC-015 Wave B).
//!
//! After a query is parsed and planned but BEFORE it executes (and before the
//! writer lock is taken), the dispatcher consults an [`AuthzHook`]: a policy
//! decision point that can DENY a request based on the authenticated
//! [`Principal`] and the [`LogicalPlan`]. This is strictly more expressive
//! than the binary `Role::allows_write` gate — a hook can deny reads, gate on
//! group membership, or inspect which labels a plan touches.
//!
//! The default [`NoOpAuthz`] allows everything, so wiring the hook is
//! behavior-preserving until a real policy is configured. A concrete
//! policy-engine impl (OPA/Cedar) lives behind a future `pdp` feature and MUST
//! fail closed (deny on any internal/transport error).

use std::fmt;

use async_trait::async_trait;
use namidb_query::LogicalPlan;

use crate::auth::Principal;

/// A denied authorization decision. Carries a human-readable reason that the
/// transport surfaces (HTTP 403 / Bolt `Forbidden`).
#[derive(Debug, Clone)]
pub struct Denied {
    pub reason: String,
}

impl Denied {
    pub fn new(reason: impl Into<String>) -> Self {
        Denied {
            reason: reason.into(),
        }
    }
}

impl fmt::Display for Denied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "authorization denied: {}", self.reason)
    }
}

impl std::error::Error for Denied {}

/// A policy decision point consulted before every query executes.
///
/// Implementations MUST be fail-closed: a real PDP that cannot reach its
/// policy engine returns `Err(Denied)`, never `Ok`.
#[async_trait]
pub trait AuthzHook: Send + Sync {
    /// Decide whether `principal` may run `plan`. `Ok(())` allows; `Err` denies.
    async fn check(&self, principal: &Principal, plan: &LogicalPlan) -> Result<(), Denied>;

    /// Decide whether `principal` may run a schema (DDL) operation. DDL is
    /// intercepted before planning, so it has no `LogicalPlan`; this is the
    /// hook entrypoint for the most-privileged operations (e.g.
    /// `CREATE VECTOR INDEX`). The default delegates nothing and allows — a
    /// real PDP overrides it to gate schema changes. `Ok(())` allows; `Err`
    /// denies.
    async fn check_schema(&self, _principal: &Principal, _op: SchemaOp<'_>) -> Result<(), Denied> {
        Ok(())
    }
}

/// A schema (DDL) operation presented to [`AuthzHook::check_schema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaOp<'a> {
    /// `CREATE VECTOR INDEX <name>` over `(label, property)`.
    CreateVectorIndex {
        name: &'a str,
        label: &'a str,
        property: &'a str,
    },
}

/// The default hook: allows every request. Wiring it is behavior-preserving —
/// the dispatcher calls `check` but the decision is always `Ok`.
#[derive(Debug, Default, Clone)]
pub struct NoOpAuthz;

#[async_trait]
impl AuthzHook for NoOpAuthz {
    async fn check(&self, _principal: &Principal, _plan: &LogicalPlan) -> Result<(), Denied> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Role;

    fn principal() -> Principal {
        Principal {
            subject: "tester".into(),
            role: Role::ReadWrite,
            groups: vec![],
        }
    }

    // A hook that denies everything — used to prove the dispatcher honours a
    // deny decision (including for reads, which allows_write cannot express).
    struct DenyAll;
    #[async_trait]
    impl AuthzHook for DenyAll {
        async fn check(&self, _p: &Principal, _plan: &LogicalPlan) -> Result<(), Denied> {
            Err(Denied::new("denied by policy"))
        }
    }

    #[tokio::test]
    async fn noop_allows_everything() {
        let plan = LogicalPlan::Empty;
        assert!(NoOpAuthz.check(&principal(), &plan).await.is_ok());
    }

    #[tokio::test]
    async fn deny_all_denies() {
        let plan = LogicalPlan::Empty;
        let err = DenyAll.check(&principal(), &plan).await.unwrap_err();
        assert!(err.to_string().contains("denied by policy"));
    }
}
