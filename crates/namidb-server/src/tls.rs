//! TLS for the serving path (HTTP and Bolt).
//!
//! Loads a PEM certificate chain and private key once and shares a single
//! [`rustls::ServerConfig`] between the HTTP server (via `axum-server`) and
//! the Bolt listener (via a [`tokio_rustls::TlsAcceptor`]). When `--tls-cert`
//! / `--tls-key` are not configured the server stays plaintext, exactly as
//! before. The `ring` crypto provider is selected explicitly so the build
//! needs no aws-lc-rs C toolchain and there is no ambiguity about which
//! provider serves the handshake.

use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::ServerConfig;
use tokio_rustls::TlsAcceptor;

/// Build a shared rustls server config from PEM cert-chain and key files.
pub fn load_server_config(cert_path: &Path, key_path: &Path) -> Result<Arc<ServerConfig>> {
    let certs = load_certs(cert_path)?;
    let key = load_key(key_path)?;
    let config =
        ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .context("rustls: selecting protocol versions")?
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .context("rustls: the certificate and private key do not match or are invalid")?;
    Ok(Arc::new(config))
}

/// A `tokio_rustls` acceptor for the Bolt listener, built from the same
/// config the HTTP server uses.
pub fn acceptor(config: Arc<ServerConfig>) -> TlsAcceptor {
    TlsAcceptor::from(config)
}

fn load_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening TLS certificate {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let certs = rustls_pemfile::certs(&mut reader)
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("parsing TLS certificate {}", path.display()))?;
    anyhow::ensure!(!certs.is_empty(), "no certificates in {}", path.display());
    Ok(certs)
}

fn load_key(path: &Path) -> Result<PrivateKeyDer<'static>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening TLS private key {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let key = rustls_pemfile::private_key(&mut reader)
        .with_context(|| format!("parsing TLS private key {}", path.display()))?
        .ok_or_else(|| anyhow::anyhow!("no private key in {}", path.display()))?;
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_pem(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn loads_a_self_signed_cert_and_key() {
        let ck = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let cert = write_pem(&ck.cert.pem());
        let key = write_pem(&ck.key_pair.serialize_pem());
        assert!(load_server_config(cert.path(), key.path()).is_ok());
    }

    #[test]
    fn rejects_a_key_that_does_not_parse() {
        let ck = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let cert = write_pem(&ck.cert.pem());
        let bad = write_pem("-----BEGIN PRIVATE KEY-----\nnot base64\n-----END PRIVATE KEY-----\n");
        assert!(load_server_config(cert.path(), bad.path()).is_err());
    }

    #[test]
    fn rejects_a_missing_certificate_file() {
        let ck = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let key = write_pem(&ck.key_pair.serialize_pem());
        let missing = std::path::Path::new("/nonexistent/namidb-tls-cert.pem");
        assert!(load_server_config(missing, key.path()).is_err());
    }
}
