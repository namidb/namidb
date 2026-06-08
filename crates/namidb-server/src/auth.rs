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

/// One accepted token and the role it grants. The secret is never logged; the
/// `name` is a human label for diagnostics.
#[derive(Clone)]
struct AuthToken {
    name: String,
    secret: Arc<str>,
    role: Role,
}

impl std::fmt::Debug for AuthToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthToken")
            .field("name", &self.name)
            .field("role", &self.role)
            .field("secret", &"***")
            .finish()
    }
}

/// The process's accepted tokens. Empty = open (no auth).
#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    tokens: Vec<AuthToken>,
}

impl AuthConfig {
    /// No tokens: every request is served as read-write.
    pub fn open() -> Self {
        Self::default()
    }

    /// A single read-write token — the back-compat `--auth-token` path.
    pub fn single_read_write(secret: impl Into<String>) -> Self {
        Self {
            tokens: vec![AuthToken {
                name: "auth-token".into(),
                secret: Arc::from(secret.into()),
                role: Role::ReadWrite,
            }],
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
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self { tokens })
    }

    /// `true` when no tokens are configured, so the server runs open.
    pub fn is_open(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The role granted to `presented`, or `None` when no token matches.
    ///
    /// Walks every token (no early return on a match) and uses a constant-time
    /// byte compare, so neither the number of tokens nor the matching position
    /// leaks through timing. A length mismatch still short-circuits, exactly as
    /// the single-token path always has. If two tokens share the same secret
    /// (a config mistake), the last one's role wins.
    pub fn role_for(&self, presented: &str) -> Option<Role> {
        let mut granted = None;
        for t in &self.tokens {
            if constant_time_eq(presented.as_bytes(), t.secret.as_bytes()) {
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
    fn role_for_distinguishes_read_only_and_read_write() {
        let c = AuthConfig {
            tokens: vec![
                AuthToken {
                    name: "rw".into(),
                    secret: Arc::from("write-key"),
                    role: Role::ReadWrite,
                },
                AuthToken {
                    name: "ro".into(),
                    secret: Arc::from("read-key"),
                    role: Role::ReadOnly,
                },
            ],
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
