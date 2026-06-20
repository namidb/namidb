//! OIDC/JWT bearer-token validation (RFC-015 Wave A), behind the `jwt` feature.
//!
//! A bearer token in `Authorization: Bearer <jwt>` (HTTP) or the Bolt
//! `credentials` field is validated against a JWKS URL — signature (RS/ES*),
//! `exp`, and the optional `iss`/`aud` — and a configured group claim is
//! mapped to a [`Role`]. The validator plugs into
//! [`crate::auth::AuthConfig::role_for`], so both protocols become JWT-aware
//! with no other changes.
//!
//! **Fail-closed:** any validation error (bad signature, expired token, wrong
//! issuer/audience, no configured group present) yields `None`, and the
//! request is rejected as unauthorized.
//!
//! **Algorithms:** only asymmetric (RS256/384/512, ES256/384) keys from the
//! JWKS are accepted. Symmetric HS* algs are refused: a JWKS carries public
//! keys, and accepting HMAC with a public "key" would be a signature-confusion
//! vulnerability.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use jsonwebtoken::{decode, decode_header, Algorithm, DecodingKey, Validation};
use serde::Deserialize;

use crate::auth::{Principal, Role};

/// The accepted asymmetric algorithms (JWKS = public keys).
const ACCEPTED_ALGS: &[Algorithm] = &[
    Algorithm::RS256,
    Algorithm::RS384,
    Algorithm::RS512,
    Algorithm::ES256,
    Algorithm::ES384,
];

/// JWT validation configuration, parsed from CLI flags.
#[derive(Debug, Clone)]
pub struct JwtConfig {
    pub jwks_url: String,
    pub issuer: Option<String>,
    pub audience: Option<String>,
    /// JWT claim holding the user's group list (default `groups`).
    pub groups_claim: String,
    /// Group that grants read-write (write) access.
    pub write_group: Option<String>,
    /// Group that grants read-only access.
    pub read_group: Option<String>,
    /// Optional JWT claim listing the namespaces the token may reach. `None`
    /// (default) = unscoped (any namespace). When set, a token grants its role
    /// only to namespaces named in the claim.
    pub namespaces_claim: Option<String>,
}

/// A live JWT validator: a JWKS key set (refreshed in the background) plus the
/// validation rules. Cheap to share via [`Arc`]; [`validate`](Self::validate)
/// is synchronous so it can run inside `AuthConfig::role_for`.
pub struct JwtValidator {
    config: JwtConfig,
    /// `kid` → decoding key. Guarded by a std RwLock so `validate` (sync) can
    /// read it without an `.await`; the background refresh task writes it.
    keys: Arc<std::sync::RwLock<HashMap<String, DecodingKey>>>,
}

impl std::fmt::Debug for JwtValidator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the key material.
        let n = self.keys.read().map(|m| m.len()).unwrap_or(0);
        f.debug_struct("JwtValidator")
            .field("jwks_url", &self.config.jwks_url)
            .field("issuer", &self.config.issuer)
            .field("audience", &self.config.audience)
            .field("groups_claim", &self.config.groups_claim)
            .field("key_count", &n)
            .finish()
    }
}

impl JwtValidator {
    /// Build the validator and fetch the JWKS once (fail-fast: a startup
    /// fetch error aborts boot before any listener binds).
    pub async fn new(config: JwtConfig) -> anyhow::Result<Self> {
        let keys = Arc::new(std::sync::RwLock::new(HashMap::new()));
        let v = Self { config, keys };
        v.refresh().await?;
        Ok(v)
    }

    /// Re-fetch the JWKS and swap in the new key set.
    pub async fn refresh(&self) -> anyhow::Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;
        let jwks: JwkSet = client.get(&self.config.jwks_url).send().await?.json().await?;
        let mut map = HashMap::new();
        for key in &jwks.keys {
            if let Ok(dk) = DecodingKey::from_jwk(key) {
                let kid = key.common.key_id.clone().unwrap_or_default();
                map.insert(kid, dk);
            }
        }
        if map.is_empty() {
            anyhow::bail!(
                "JWKS at {} parsed but yielded no usable signing keys",
                self.config.jwks_url
            );
        }
        *self.keys.write().expect("jwt key lock poisoned") = map;
        Ok(())
    }

    /// Spawn a background task that refreshes the JWKS every `interval`. Lets
    /// keys rotate without a restart; a failed refresh logs and retries next
    /// tick (the last good key set stays served). The task runs for the life
    /// of the process and dies with the runtime on exit.
    pub fn spawn_refresh(self: &Arc<Self>, interval: Duration) {
        if interval.is_zero() {
            return;
        }
        let me = Arc::clone(self);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.tick().await; // skip the immediate first tick (fetched at boot)
            loop {
                tick.tick().await;
                if let Err(e) = me.refresh().await {
                    tracing::warn!("JWKS refresh failed: {e:#}");
                }
            }
        });
    }

    /// Validate `token` and resolve the [`Role`] from its group claim, or
    /// `None` (fail-closed) on any failure. Namespace-agnostic (the single-
    /// tenant / Bolt path).
    pub fn validate(&self, token: &str) -> Option<Role> {
        self.validate_principal(token).map(|p| p.role)
    }

    /// Like [`validate`](Self::validate) but additionally requires the token's
    /// namespaces claim (when `namespaces_claim` is configured) to name
    /// `namespace`. Unconfigured → unscoped (back-compat). Used by the
    /// multi-tenant path.
    pub fn validate_in(&self, token: &str, namespace: &str) -> Option<Role> {
        self.validate_principal_in(token, namespace)
            .map(|p| p.role)
    }

    /// Full principal (subject + role + groups) for `token`, namespace-agnostic.
    pub fn validate_principal(&self, token: &str) -> Option<Principal> {
        self.validate_inner(token, None)
    }

    /// [`validate_principal`](Self::validate_principal) with a namespace scope.
    pub fn validate_principal_in(&self, token: &str, namespace: &str) -> Option<Principal> {
        self.validate_inner(token, Some(namespace))
    }

    fn validate_inner(&self, token: &str, namespace: Option<&str>) -> Option<Principal> {
        let header = decode_header(token).ok()?;
        if !ACCEPTED_ALGS.contains(&header.alg) {
            return None; // refuse symmetric / "none" / unknown algs
        }
        let validation = self.validation_for(header.alg);

        let keys = self.keys.read().expect("jwt key lock poisoned");
        match &header.kid {
            // A `kid` pins one key; still fall back to trying all on failure.
            Some(kid) => keys
                .get(kid)
                .and_then(|dk| self.try_decode(token, dk, &validation, namespace))
                .or_else(|| {
                    keys.values()
                        .find_map(|dk| self.try_decode(token, dk, &validation, namespace))
                }),
            None => keys
                .values()
                .find_map(|dk| self.try_decode(token, dk, &validation, namespace)),
        }
    }

    /// Decode against one key; on success, extract subject + group claim (and
    /// check the namespaces claim when a namespace is being checked), and map
    /// to a [`Principal`].
    fn try_decode(
        &self,
        token: &str,
        key: &DecodingKey,
        validation: &Validation,
        namespace: Option<&str>,
    ) -> Option<Principal> {
        // Decode to a generic JSON value so the claims can live anywhere.
        let data = decode::<serde_json::Value>(token, key, validation).ok()?;

        // Namespace scoping: if a namespace is being checked AND a namespaces
        // claim is configured, the token must name that namespace. Unconfigured
        // claim → unscoped (back-compat).
        if let (Some(ns), Some(claim)) = (namespace, &self.config.namespaces_claim) {
            let allowed = data
                .claims
                .get(claim)
                .and_then(extract_group_strings)
                .unwrap_or_default();
            if !allowed.iter().any(|n| n == ns) {
                return None;
            }
        }

        let groups = data
            .claims
            .get(&self.config.groups_claim)
            .and_then(extract_group_strings)?;
        // Write group wins over read group (most-permissive).
        let role = if let Some(wg) = &self.config.write_group {
            if groups.iter().any(|g| g == wg) {
                Role::ReadWrite
            } else if let Some(rg) = &self.config.read_group {
                if groups.iter().any(|g| g == rg) {
                    Role::ReadOnly
                } else {
                    return None;
                }
            } else {
                return None;
            }
        } else if let Some(rg) = &self.config.read_group {
            if groups.iter().any(|g| g == rg) {
                Role::ReadOnly
            } else {
                return None;
            }
        } else {
            return None;
        };

        let subject = data
            .claims
            .get("sub")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        Some(Principal {
            subject,
            role,
            groups,
        })
    }

    fn validation_for(&self, alg: Algorithm) -> Validation {
        let mut v = Validation::new(alg);
        // A 30s clock-skew leeway avoids flapping around `exp` boundaries.
        v.leeway = 30;
        if let Some(iss) = &self.config.issuer {
            v.set_issuer(&[iss]);
        }
        // `aud`/`iss` are opt-in: only enforce them when configured, so a
        // token that happens to carry an `aud` isn't rejected when we didn't
        // ask to check it.
        v.validate_aud = self.config.audience.is_some();
        if let Some(aud) = &self.config.audience {
            v.set_audience(&[aud]);
        }
        v
    }
}

/// Coerce a claim value into a list of group strings. Accepts a JSON array of
/// strings, or a single string (treated as a one-element list).
fn extract_group_strings(v: &serde_json::Value) -> Option<Vec<String>> {
    match v {
        serde_json::Value::Array(a) => Some(
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect(),
        ),
        serde_json::Value::String(s) => Some(vec![s.clone()]),
        _ => None,
    }
}

/// A JWKS document: `{"keys": [ {kty, kid, alg, n, e, ...}, ... ]}`.
/// `jsonwebtoken` ships its own `Jwk` type (`DecodingKey::from_jwk` consumes
/// it directly), so we deserialize into that and drop the custom shape.
#[derive(Debug, Deserialize)]
struct JwkSet {
    #[serde(default)]
    keys: Vec<jsonwebtoken::jwk::Jwk>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
    use serde::Serialize;

    /// A throwaway RSA-2048 keypair generated for these tests only.
    const PRIV_PEM: &str = include_str!("../tests/jwt_test_key.pem");
    const PUB_PEM: &str = include_str!("../tests/jwt_test_pub.pem");

    fn now_plus(secs: u64) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + secs
    }

    #[derive(Serialize)]
    struct Claims<'a> {
        exp: u64,
        iss: &'a str,
        aud: &'a str,
        groups: Vec<&'a str>,
    }

    fn mint(groups: &[&str]) -> String {
        let key = EncodingKey::from_rsa_pem(PRIV_PEM.as_bytes()).unwrap();
        let claims = Claims {
            exp: now_plus(3600),
            iss: "test-iss",
            aud: "test-aud",
            groups: groups.to_vec(),
        };
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some("test-key".to_string());
        encode(&header, &claims, &key).unwrap()
    }

    fn validator(
        write_group: Option<&str>,
        read_group: Option<&str>,
        issuer: Option<&str>,
        audience: Option<&str>,
    ) -> JwtValidator {
        let dk = DecodingKey::from_rsa_pem(PUB_PEM.as_bytes()).unwrap();
        let mut map = HashMap::new();
        map.insert("test-key".to_string(), dk);
        JwtValidator {
            config: JwtConfig {
                jwks_url: String::new(),
                issuer: issuer.map(String::from),
                audience: audience.map(String::from),
                groups_claim: "groups".into(),
                write_group: write_group.map(String::from),
                read_group: read_group.map(String::from),
                namespaces_claim: None,
            },
            keys: Arc::new(std::sync::RwLock::new(map)),
        }
    }

    #[test]
    fn write_group_token_is_read_write() {
        let v = validator(Some("admins"), Some("readers"), None, None);
        assert_eq!(v.validate(&mint(&["admins"])), Some(Role::ReadWrite));
    }

    #[test]
    fn read_group_token_is_read_only() {
        let v = validator(Some("admins"), Some("readers"), None, None);
        assert_eq!(v.validate(&mint(&["readers"])), Some(Role::ReadOnly));
    }

    #[test]
    fn write_group_wins_over_read_group() {
        let v = validator(Some("admins"), Some("readers"), None, None);
        assert_eq!(v.validate(&mint(&["readers", "admins"])), Some(Role::ReadWrite));
    }

    #[test]
    fn no_matching_group_is_denied() {
        let v = validator(Some("admins"), Some("readers"), None, None);
        assert_eq!(v.validate(&mint(&["other"])), None);
    }

    #[test]
    fn wrong_issuer_is_denied() {
        let v = validator(Some("admins"), None, Some("expected-iss"), None);
        assert_eq!(v.validate(&mint(&["admins"])), None);
    }

    #[test]
    fn wrong_audience_is_denied() {
        let v = validator(Some("admins"), None, None, Some("expected-aud"));
        assert_eq!(v.validate(&mint(&["admins"])), None);
    }

    #[test]
    fn tampered_signature_is_denied() {
        let v = validator(Some("admins"), None, None, None);
        let mut tok = mint(&["admins"]);
        // Flip the last character (in the signature segment) → sig mismatch.
        if let Some(last) = tok.pop() {
            tok.push(if last == 'A' { 'B' } else { 'A' });
        }
        assert_eq!(v.validate(&tok), None);
    }

    #[test]
    fn garbage_is_denied() {
        let v = validator(Some("admins"), None, None, None);
        assert_eq!(v.validate("not.a.jwt"), None);
        assert_eq!(v.validate(""), None);
    }

    #[test]
    fn extract_group_strings_shapes() {
        assert_eq!(
            extract_group_strings(&serde_json::json!(["a", "b"])),
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(
            extract_group_strings(&serde_json::json!("solo")),
            Some(vec!["solo".to_string()])
        );
        assert_eq!(extract_group_strings(&serde_json::json!(42)), None);
    }
}

