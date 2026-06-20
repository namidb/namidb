//! Bearer-token authentication with per-token roles.
//!
//! A namespace accepts a set of tokens, each granting either read-only or
//! read-write access. An empty set means "no auth" (open mode): the server
//! warns loudly at boot and serves every request as read-write. The same
//! [`AuthConfig`] backs both serving paths — the HTTP middleware and the Bolt
//! `Custom` authenticator — so a read-only token cannot write over either.
//!
//! Per-namespace token scoping is deliberately out of scope here: the server
//! serves one namespace, so there is nothing to route. It lands with
//! multi-namespace routing.

use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;

/// What a token may do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Reads only; any write query is rejected.
    ReadOnly,
    /// Reads and writes.
    ReadWrite,
}

impl Role {
    /// Whether this role may run a write (CREATE/MERGE/SET/REMOVE/DELETE) or a
    /// maintenance write (admin flush).
    pub fn allows_write(self) -> bool {
        matches!(self, Role::ReadWrite)
    }
}

/// The authenticated identity behind a request: who (subject), what they may
/// do (role), and what groups they belong to. Richer than [`Role`] (a 1-bit
/// projection) so a future authorization hook can make group/subject-aware
/// decisions. Role stays `Copy`; `Principal` composes it and is `Clone` (it
/// owns Strings), so existing `role.allows_write()` sites keep working via
/// [`Principal::allows_write`].
#[derive(Debug, Clone, PartialEq)]
pub struct Principal {
    /// Who the caller is — a static token's name, or a JWT `sub` claim.
    pub subject: String,
    /// The resolved permission level.
    pub role: Role,
    /// Group memberships (a JWT's group claim; empty for static tokens).
    pub groups: Vec<String>,
}

impl Principal {
    /// Convenience: same 1-bit decision as [`Role::allows_write`].
    pub fn allows_write(&self) -> bool {
        self.role.allows_write()
    }

    /// The principal granted in open mode (no auth): an anonymous read-write
    /// caller. Used by the middleware when `AuthConfig::is_open()`.
    pub fn anonymous_rw() -> Self {
        Principal {
            subject: "anonymous".to_string(),
            role: Role::ReadWrite,
            groups: Vec::new(),
        }
    }

    /// `true` if the principal belongs to `group`.
    pub fn in_group(&self, group: &str) -> bool {
        self.groups.iter().any(|g| g == group)
    }
}

/// One accepted token and the role it grants. The secret is never logged; the
/// `name` is a human label for diagnostics.
#[derive(Clone)]
struct AuthToken {
    name: String,
    secret: Arc<str>,
    role: Role,
    /// Namespaces this token may reach. `None` = unscoped (any namespace — the
    /// back-compat default); `Some(set)` = scoped to exactly those. An empty
    /// set denies every namespace. Only consulted by `role_for_in` (the
    /// multi-tenant path); the single-tenant/Bolt `role_for` ignores it.
    namespaces: Option<Vec<String>>,
}

impl std::fmt::Debug for AuthToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthToken")
            .field("name", &self.name)
            .field("role", &self.role)
            .field("namespaces", &self.namespaces)
            .field("secret", &"***")
            .finish()
    }
}

/// The process's accepted tokens. Empty = open (no auth).
#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    tokens: Vec<AuthToken>,
    /// Optional OIDC/JWT validator. When present, a bearer token is first
    /// attempted as a JWT (mapped from a group claim); static tokens are the
    /// fallback. Only compiled under the `jwt` feature.
    #[cfg(feature = "jwt")]
    jwt: Option<Arc<crate::jwt::JwtValidator>>,
}

impl AuthConfig {
    /// No tokens: every request is served as read-write.
    pub fn open() -> Self {
        Self::default()
    }

    /// A single read-write token — the back-compat `--auth-token` path.
    #[allow(clippy::needless_update)] // `..Default::default()` fills the jwt field under the `jwt` feature
    pub fn single_read_write(secret: impl Into<String>) -> Self {
        Self {
            tokens: vec![AuthToken {
                name: "auth-token".into(),
                secret: Arc::from(secret.into()),
                role: Role::ReadWrite,
                namespaces: None,
            }],
            ..Default::default()
        }
    }

    /// Load tokens from a JSON file:
    ///
    /// ```json
    /// { "tokens": [
    ///     { "name": "ci",        "token": "…", "role": "read-write" },
    ///     { "name": "dashboard", "token": "…", "role": "read-only"  }
    /// ] }
    /// ```
    ///
    /// `role` defaults to `read-write` and `name` to `token-<i>`. A file with
    /// an empty `tokens` array is rejected — that would silently disable auth;
    /// omit the flag to run open on purpose.
    #[allow(clippy::needless_update)] // `..Default::default()` fills the jwt field under the `jwt` feature
    pub fn load_file(path: &Path) -> anyhow::Result<Self> {
        let body = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading auth tokens file {}: {e}", path.display()))?;
        let file: TokenFile = serde_json::from_str(&body)
            .map_err(|e| anyhow::anyhow!("parsing auth tokens file {}: {e}", path.display()))?;
        if file.tokens.is_empty() {
            anyhow::bail!(
                "auth tokens file {} has no tokens; omit --auth-tokens-file to run without auth",
                path.display()
            );
        }
        let tokens = file
            .tokens
            .into_iter()
            .enumerate()
            .map(|(i, e)| {
                if e.token.is_empty() {
                    anyhow::bail!(
                        "auth tokens file {}: token #{i} has an empty secret",
                        path.display()
                    );
                }
                Ok(AuthToken {
                    name: e.name.unwrap_or_else(|| format!("token-{i}")),
                    secret: Arc::from(e.token),
                    role: e.role.into(),
                    namespaces: e.namespaces,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            tokens,
            ..Default::default()
        })
    }

    /// Attach a JWT validator. Bearer tokens are then first interpreted as
    /// JWTs; static tokens remain the fallback. (Only under `jwt`.)
    #[cfg(feature = "jwt")]
    pub fn with_jwt(mut self, jwt: Arc<crate::jwt::JwtValidator>) -> Self {
        self.jwt = Some(jwt);
        self
    }

    /// `true` when no auth is configured (no tokens and no JWT validator), so
    /// the server runs open.
    pub fn is_open(&self) -> bool {
        let jwt_active = self.jwt_active();
        self.tokens.is_empty() && !jwt_active
    }

    #[cfg(feature = "jwt")]
    fn jwt_active(&self) -> bool {
        self.jwt.is_some()
    }
    #[cfg(not(feature = "jwt"))]
    fn jwt_active(&self) -> bool {
        false
    }

    /// The role granted to `presented`, or `None` when nothing matches.
    /// Namespace-agnostic: used by the single-tenant path and Bolt (one
    /// namespace each), so a token's namespace scope is intentionally ignored.
    pub fn role_for(&self, presented: &str) -> Option<Role> {
        self.role_for_in(presented, "")
    }

    /// The full [`Principal`] for `presented` (namespace-agnostic), or `None`.
    /// The single resolution path for identity: `role_for` delegates here via
    /// `.map(|p| p.role)`. Static tokens yield `subject = token name`,
    /// `groups = []`; a JWT yields the `sub` claim and the full group list.
    pub fn principal_for(&self, presented: &str) -> Option<Principal> {
        self.principal_for_in(presented, "")
    }

    /// [`principal_for`](Self::principal_for) with a namespace scope: returns
    /// `None` when the matched token/JWT isn't scoped to `namespace`.
    #[cfg(feature = "jwt")]
    pub fn principal_for_in(&self, presented: &str, namespace: &str) -> Option<Principal> {
        let single_tenant = namespace.is_empty();
        if let Some(jwt) = &self.jwt {
            let ok = if single_tenant {
                jwt.validate_principal(presented)
            } else {
                jwt.validate_principal_in(presented, namespace)
            };
            if let Some(p) = ok {
                return Some(p);
            }
        }
        self.token_principal(presented, namespace)
    }

    #[cfg(not(feature = "jwt"))]
    pub fn principal_for_in(&self, presented: &str, namespace: &str) -> Option<Principal> {
        self.token_principal(presented, namespace)
    }

    /// Static-token resolution into a [`Principal`] (subject = the token's
    /// name, groups empty), honouring the namespace scope.
    fn token_principal(&self, presented: &str, namespace: &str) -> Option<Principal> {
        let single_tenant = namespace.is_empty();
        let mut granted: Option<&AuthToken> = None;
        for t in &self.tokens {
            if constant_time_eq(presented.as_bytes(), t.secret.as_bytes())
                && (single_tenant
                    || t.namespaces
                        .as_ref()
                        .map_or(true, |ns| ns.iter().any(|n| n == namespace)))
            {
                granted = Some(t);
            }
        }
        granted.map(|t| Principal {
            subject: t.name.clone(),
            role: t.role,
            groups: Vec::new(),
        })
    }

    /// The role granted to `presented` for `namespace`, or `None` when nothing
    /// matches OR the matched token is scoped to other namespaces. This is the
    /// multi-tenant auth predicate that closes the cross-namespace reach gap.
    ///
    /// A `namespace` of `""` (the single-tenant sentinel) matches any token —
    /// scoped tokens carry no meaning for a single-namespace server, so they
    /// grant their role there. For a real namespace, a scoped token grants its
    /// role only if `namespace` is in its set; an unscoped token grants always.
    ///
    /// The static-token walk uses a constant-time byte compare with no early
    /// return (the namespace check happens only on a secret match), so neither
    /// the token count nor the matching position leaks through timing.
    pub fn role_for_in(&self, presented: &str, namespace: &str) -> Option<Role> {
        // The empty-string namespace is the single-tenant sentinel: skip the
        // scope check entirely (scoped tokens are meaningless for one ns).
        let single_tenant = namespace.is_empty();
        #[cfg(feature = "jwt")]
        if let Some(jwt) = &self.jwt {
            let ok = if single_tenant {
                jwt.validate(presented)
            } else {
                jwt.validate_in(presented, namespace)
            };
            if let Some(role) = ok {
                return Some(role);
            }
        }
        let mut granted = None;
        for t in &self.tokens {
            if constant_time_eq(presented.as_bytes(), t.secret.as_bytes())
                && (single_tenant
                    || t.namespaces
                        .as_ref()
                        .map_or(true, |ns| ns.iter().any(|n| n == namespace)))
            {
                granted = Some(t.role);
            }
        }
        granted
    }

    /// Number of configured tokens, for the boot log.
    pub(crate) fn len(&self) -> usize {
        self.tokens.len()
    }
}

#[derive(Deserialize)]
struct TokenFile {
    tokens: Vec<TokenFileEntry>,
}

#[derive(Deserialize)]
struct TokenFileEntry {
    token: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    role: RoleSpec,
    /// Optional namespace scope. Omit (or null) for an unscoped token that
    /// reaches every namespace; list namespaces to restrict it. An empty
    /// array denies all namespaces.
    #[serde(default)]
    namespaces: Option<Vec<String>>,
}

#[derive(Deserialize, Default, Clone, Copy)]
enum RoleSpec {
    #[default]
    #[serde(rename = "read-write")]
    ReadWrite,
    #[serde(rename = "read-only")]
    ReadOnly,
}

impl From<RoleSpec> for Role {
    fn from(r: RoleSpec) -> Self {
        match r {
            RoleSpec::ReadWrite => Role::ReadWrite,
            RoleSpec::ReadOnly => Role::ReadOnly,
        }
    }
}

/// Constant-time byte equality for short shared secrets: walks every byte
/// regardless of where a mismatch is. A length difference returns early (the
/// length is not itself secret).
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        acc |= x ^ y;
    }
    acc == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_config_has_no_tokens() {
        let c = AuthConfig::open();
        assert!(c.is_open());
        assert_eq!(c.role_for("anything"), None);
    }

    #[test]
    fn single_token_grants_read_write() {
        let c = AuthConfig::single_read_write("s3cret");
        assert!(!c.is_open());
        assert_eq!(c.role_for("s3cret"), Some(Role::ReadWrite));
        assert_eq!(c.role_for("wrong"), None);
    }

    #[test]
    #[allow(clippy::needless_update)]
    fn role_for_in_enforces_namespace_scope() {
        // acme-key is scoped to ["acme"]; beta-key to ["beta"]; any-key unscoped.
        let c = AuthConfig {
            tokens: vec![
                AuthToken {
                    name: "acme".into(),
                    secret: Arc::from("acme-key"),
                    role: Role::ReadWrite,
                    namespaces: Some(vec!["acme".into()]),
                },
                AuthToken {
                    name: "beta".into(),
                    secret: Arc::from("beta-key"),
                    role: Role::ReadOnly,
                    namespaces: Some(vec!["beta".into()]),
                },
                AuthToken {
                    name: "any".into(),
                    secret: Arc::from("any-key"),
                    role: Role::ReadWrite,
                    namespaces: None,
                },
            ],
            ..Default::default()
        };

        // acme-key reaches acme but not beta.
        assert_eq!(c.role_for_in("acme-key", "acme"), Some(Role::ReadWrite));
        assert_eq!(c.role_for_in("acme-key", "beta"), None);
        // beta-key reaches beta (read-only) but not acme.
        assert_eq!(c.role_for_in("beta-key", "beta"), Some(Role::ReadOnly));
        assert_eq!(c.role_for_in("beta-key", "acme"), None);
        // Unscoped any-key reaches every namespace.
        assert_eq!(c.role_for_in("any-key", "acme"), Some(Role::ReadWrite));
        assert_eq!(c.role_for_in("any-key", "beta"), Some(Role::ReadWrite));
        assert_eq!(c.role_for_in("any-key", "zzz"), Some(Role::ReadWrite));
        // role_for (single-tenant sentinel) ignores scope: every token authenticates.
        assert_eq!(c.role_for("acme-key"), Some(Role::ReadWrite));
        assert_eq!(c.role_for("beta-key"), Some(Role::ReadOnly));
        // Wrong token is denied everywhere.
        assert_eq!(c.role_for_in("nope", "acme"), None);
    }

    #[test]
    #[allow(clippy::needless_update)]
    fn role_for_distinguishes_read_only_and_read_write() {
        let c = AuthConfig {
            tokens: vec![
                AuthToken {
                    name: "rw".into(),
                    secret: Arc::from("write-key"),
                    role: Role::ReadWrite,
                    namespaces: None,
                },
                AuthToken {
                    name: "ro".into(),
                    secret: Arc::from("read-key"),
                    role: Role::ReadOnly,
                    namespaces: None,
                },
            ],
            ..Default::default()
        };
        assert_eq!(c.role_for("write-key"), Some(Role::ReadWrite));
        assert_eq!(c.role_for("read-key"), Some(Role::ReadOnly));
        assert_eq!(c.role_for("nope"), None);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn load_file_parses_roles_and_defaults() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("namidb-auth-{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{ "tokens": [
                { "name": "ci", "token": "k1", "role": "read-write" },
                { "token": "k2", "role": "read-only" },
                { "token": "k3" }
            ] }"#,
        )
        .unwrap();
        let c = AuthConfig::load_file(&path).unwrap();
        assert_eq!(c.role_for("k1"), Some(Role::ReadWrite));
        assert_eq!(c.role_for("k2"), Some(Role::ReadOnly));
        assert_eq!(c.role_for("k3"), Some(Role::ReadWrite)); // role defaults
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_file_rejects_empty_token_set() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("namidb-auth-empty-{}.json", std::process::id()));
        std::fs::write(&path, r#"{ "tokens": [] }"#).unwrap();
        assert!(AuthConfig::load_file(&path).is_err());
        std::fs::remove_file(&path).ok();
    }
}
