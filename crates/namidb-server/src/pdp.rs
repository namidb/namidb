//! External policy decision point (PDP) — an [`AuthzHook`] backed by an
//! OPA-style HTTP endpoint (RFC-015 Wave B), behind the `pdp` feature.
//!
//! Before a query executes, [`OpaAuthz`] POSTs a small JSON document
//! describing the request — the principal (subject, role, groups), the action
//! (`read` / `write` / a schema op), and the plan shape (operator names) — to
//! a configured policy endpoint. The policy returns whether to allow it. This
//! is how a deployment expresses rules richer than the built-in 1-bit
//! read/write gate (e.g. "group `analysts` may read but never write label
//! `Salary`", "only `admins` create indexes").
//!
//! **Fail-closed.** Any failure to obtain a definitive `allow: true` — a
//! network error, a non-2xx status, a malformed body, a missing/false `allow`
//! — denies the request. A PDP that cannot be consulted must never
//! fail-open, or it is not a security control. The pure decision over an
//! already-fetched response lives in [`decide`] so it is unit-tested without a
//! network.
//!
//! Request shape (OPA convention — the policy reads `input.*`):
//! ```json
//! { "input": {
//!     "subject": "alice", "role": "read-write", "groups": ["eng"],
//!     "action": "write", "operators": ["Create", "NodeScan"],
//!     "namespace": "acme"
//! } }
//! ```
//! Accepted responses (see [`decide`]) — point `--pdp-url` at whichever your
//! deployment uses:
//! - OPA `/v1/data/<path>` with an object rule: `{ "result": { "allow": true } }`
//! - OPA `/v1/data/<path>` with a boolean rule: `{ "result": true }`
//! - OPA `/v0/data/<path>` (raw value): a bare `true`
//! - a custom endpoint: `{ "allow": true }`
//!
//! Prefer the `/v1/data/<path>` API: it returns 200 with no `result` field for
//! an *undefined* decision (which [`decide`] treats as deny), whereas `/v0`
//! returns a 404 for an undefined document (also a deny, but noisier).

use std::time::Duration;

use async_trait::async_trait;
use namidb_query::LogicalPlan;

use crate::auth::Principal;
use crate::authz::{AuthzHook, Denied, SchemaOp};

/// An [`AuthzHook`] that delegates each decision to an external OPA-style
/// policy endpoint. Fail-closed.
#[derive(Debug, Clone)]
pub struct OpaAuthz {
    endpoint: String,
    client: reqwest::Client,
}

impl OpaAuthz {
    /// Build a PDP client for `endpoint` (the full policy decision URL, e.g.
    /// `http://opa:8181/v1/data/namidb/allow`). A short timeout bounds the
    /// added latency; a timeout is a fail-closed deny.
    pub fn new(endpoint: impl Into<String>) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()?;
        Ok(Self {
            endpoint: endpoint.into(),
            client,
        })
    }

    /// POST `input` and return the raw response body text, or an error (which
    /// the caller turns into a fail-closed deny).
    async fn query_policy(&self, input: serde_json::Value) -> anyhow::Result<String> {
        let resp = self
            .client
            .post(&self.endpoint)
            .json(&serde_json::json!({ "input": input }))
            .send()
            .await?;
        if !resp.status().is_success() {
            anyhow::bail!("policy endpoint returned status {}", resp.status());
        }
        Ok(resp.text().await?)
    }

    async fn evaluate(&self, input: serde_json::Value) -> Result<(), Denied> {
        match self.query_policy(input).await {
            Ok(body) => decide(&body),
            // Fail-closed: an unreachable / erroring PDP denies.
            Err(e) => Err(Denied::new(format!("policy endpoint unavailable: {e}"))),
        }
    }
}

/// The policy decision over an already-fetched response body. `Ok(())` only
/// when the body definitively evaluates to `allow == true`; every other case
/// (parse failure, missing field, `false`, OPA "undefined" empty object) is a
/// fail-closed [`Denied`].
///
/// Accepts the shapes real OPA endpoints actually return, so an operator can
/// point `--pdp-url` at either API without the decision silently becoming
/// deny-all:
/// - `/v1/data/<path>` with an object rule: `{"result":{"allow":true}}`
/// - `/v1/data/<path>` with a boolean rule: `{"result":true}`
/// - `/v0/data/<path>` (raw value): a bare `true` / `false`
/// - a custom endpoint: `{"allow":true}`
pub fn decide(body: &str) -> Result<(), Denied> {
    let v: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => return Err(Denied::new(format!("policy response not understood: {e}"))),
    };
    let allow = extract_allow(&v).unwrap_or(false);
    if allow {
        Ok(())
    } else {
        Err(Denied::new("denied by policy"))
    }
}

/// Pull a boolean allow-decision out of any accepted response shape, or `None`
/// when the body does not definitively decide (→ caller fail-closes).
fn extract_allow(v: &serde_json::Value) -> Option<bool> {
    match v {
        // Bare boolean (OPA /v0/data raw value, or a custom boolean endpoint).
        serde_json::Value::Bool(b) => Some(*b),
        serde_json::Value::Object(map) => {
            // `{"result": ...}` — recurse into the OPA /v1 envelope (the inner
            // value may itself be a bool or an {"allow":...} object).
            if let Some(inner) = map.get("result") {
                return extract_allow(inner);
            }
            // `{"allow": <bool>}` — require an actual boolean (a string/number
            // "true" must NOT allow).
            map.get("allow").and_then(serde_json::Value::as_bool)
        }
        _ => None,
    }
}

/// Build the `input` document for a query plan.
fn plan_input(principal: &Principal, plan: &LogicalPlan) -> serde_json::Value {
    let action = if plan.contains_write() { "write" } else { "read" };
    serde_json::json!({
        "subject": principal.subject,
        "role": role_str(principal),
        "groups": principal.groups,
        "action": action,
        "operators": operator_names(plan),
    })
}

fn role_str(principal: &Principal) -> &'static str {
    if principal.allows_write() {
        "read-write"
    } else {
        "read-only"
    }
}

/// Collect the distinct operator names in a plan tree (the policy can gate on
/// which operators appear, e.g. deny `Delete` for a group).
fn operator_names(plan: &LogicalPlan) -> Vec<&'static str> {
    let mut out = Vec::new();
    collect_ops(plan, &mut out);
    out.sort_unstable();
    out.dedup();
    out
}

fn collect_ops(plan: &LogicalPlan, out: &mut Vec<&'static str>) {
    out.push(plan.operator_name());
    for child in plan.children() {
        collect_ops(child, out);
    }
}

#[async_trait]
impl AuthzHook for OpaAuthz {
    async fn check(&self, principal: &Principal, plan: &LogicalPlan) -> Result<(), Denied> {
        self.evaluate(plan_input(principal, plan)).await
    }

    async fn check_schema(&self, principal: &Principal, op: SchemaOp<'_>) -> Result<(), Denied> {
        let input = match op {
            SchemaOp::CreateVectorIndex {
                name,
                label,
                property,
            } => serde_json::json!({
                "subject": principal.subject,
                "role": role_str(principal),
                "groups": principal.groups,
                "action": "schema",
                "schema_op": "create_vector_index",
                "index_name": name,
                "label": label,
                "property": property,
            }),
        };
        self.evaluate(input).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_on_every_accepted_true_shape() {
        // OPA /v1 object-rule envelope.
        assert!(decide(r#"{"result":{"allow":true}}"#).is_ok());
        assert!(decide(r#"{"result":{"allow":false}}"#).is_err());
        // OPA /v1 boolean-rule envelope.
        assert!(decide(r#"{"result":true}"#).is_ok());
        assert!(decide(r#"{"result":false}"#).is_err());
        // OPA /v0/data raw value (bare boolean).
        assert!(decide("true").is_ok());
        assert!(decide("false").is_err());
        // Custom bare-allow object.
        assert!(decide(r#"{"allow":true}"#).is_ok());
        assert!(decide(r#"{"allow":false}"#).is_err());
    }

    #[test]
    fn fail_closed_on_malformed_or_missing() {
        // Empty / missing allow / OPA "undefined" (empty result) → deny.
        assert!(decide("{}").is_err());
        assert!(decide(r#"{"result":{}}"#).is_err());
        assert!(decide("not json").is_err());
        assert!(decide("").is_err());
        // A truthy-looking non-bool must NOT allow.
        assert!(decide(r#"{"result":{"allow":"true"}}"#).is_err());
        assert!(decide(r#"{"allow":1}"#).is_err());
        assert!(decide(r#""true""#).is_err()); // bare string, not bool
        assert!(decide("1").is_err()); // bare number, not bool
    }

    fn principal() -> Principal {
        Principal {
            subject: "alice".into(),
            role: crate::auth::Role::ReadWrite,
            groups: vec!["eng".into()],
        }
    }

    fn read_plan() -> LogicalPlan {
        LogicalPlan::NodeScan {
            label: Some("Doc".into()),
            alias: "d".into(),
            predicates: vec![],
            projection: None,
        }
    }

    /// Spawn a one-shot OPA-style server returning `body` and yield its URL.
    /// Returns (url, shutdown sender). The caller drops the sender to stop it.
    async fn spawn_policy(body: &'static str) -> (String, tokio::sync::oneshot::Sender<()>) {
        use axum::routing::post;
        let app = axum::Router::new().route("/decide", post(move || async move { body }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await
                .unwrap();
        });
        (format!("http://{addr}/decide"), tx)
    }

    #[tokio::test]
    async fn opa_allows_when_policy_allows() {
        let (url, _guard) = spawn_policy(r#"{"result":{"allow":true}}"#).await;
        let hook = OpaAuthz::new(url).unwrap();
        assert!(hook.check(&principal(), &read_plan()).await.is_ok());
    }

    #[tokio::test]
    async fn opa_denies_when_policy_denies() {
        let (url, _guard) = spawn_policy(r#"{"result":{"allow":false}}"#).await;
        let hook = OpaAuthz::new(url).unwrap();
        // A READ denied by policy — the role gate would allow it; the PDP denies.
        let err = hook.check(&principal(), &read_plan()).await.unwrap_err();
        assert!(err.to_string().contains("denied by policy"));
    }

    #[tokio::test]
    async fn opa_fails_closed_when_endpoint_unreachable() {
        // Nothing is listening on this port → fail-closed deny, not allow.
        let hook = OpaAuthz::new("http://127.0.0.1:1/decide").unwrap();
        let err = hook.check(&principal(), &read_plan()).await.unwrap_err();
        assert!(err.to_string().contains("unavailable"));
    }

    #[tokio::test]
    async fn opa_check_schema_is_enforced() {
        let (url, _guard) = spawn_policy(r#"{"result":{"allow":false}}"#).await;
        let hook = OpaAuthz::new(url).unwrap();
        let op = SchemaOp::CreateVectorIndex {
            name: "ix",
            label: "Doc",
            property: "emb",
        };
        assert!(hook.check_schema(&principal(), op).await.is_err());
    }

    #[test]
    fn operator_names_are_collected_sorted_unique() {
        // A plan with two NodeScans + a Project should dedup NodeScan.
        let scan = || LogicalPlan::NodeScan {
            label: Some("Doc".into()),
            alias: "d".into(),
            predicates: vec![],
            projection: None,
        };
        let plan = LogicalPlan::CrossProduct {
            left: Box::new(scan()),
            right: Box::new(scan()),
        };
        let ops = operator_names(&plan);
        assert!(ops.contains(&"NodeScan"));
        assert!(ops.contains(&"CrossProduct"));
        // deduped: NodeScan appears once.
        assert_eq!(ops.iter().filter(|o| **o == "NodeScan").count(), 1);
    }
}
