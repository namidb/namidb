//! Common URI → (`ObjectStore`, `NamespacePaths`) parser shared by the
//! CLI, Python bindings, and any future server binary.
//!
//! Supported schemes:
//!
//! - `memory://<namespace>` — ephemeral, single-process.
//! - `file:///abs/dir?ns=<namespace>` (or `file://./rel?ns=<ns>`) —
//!   local filesystem with manifest CAS via `flock` + atomic rename.
//! - `s3://<bucket>[/<prefix>]?ns=<ns>[&region=...][&endpoint=...][&allow_http=true|false]`
//!   — any S3-compatible service (AWS S3, Cloudflare R2, MinIO,
//!   Tigris, LocalStack, …). Credentials from the standard AWS env
//!   vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional
//!   `AWS_SESSION_TOKEN`).
//! - `gs://<bucket>[/<prefix>]?ns=<ns>[&service_account=/path/key.json]`
//!   — Google Cloud Storage. Credentials from
//!   `GOOGLE_APPLICATION_CREDENTIALS` env or `?service_account=`.
//! - `az://<account>/<container>[/<prefix>]?ns=<ns>[&endpoint=...][&allow_http=true][&use_emulator=true]`
//!   — Azure Blob Storage. Credentials from `AZURE_STORAGE_*` env vars.
//!
//! Frontends wrap [`parse_uri`] and surface any returned [`UriError`]
//! in their native error type (`PyValueError`, `anyhow::Error`, HTTP
//! 400, …).

use std::sync::Arc;

use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_core::NamespaceId;

use crate::local::LocalFileObjectStore;
use crate::paths::NamespacePaths;

/// Errors that [`parse_uri`] can produce.
#[derive(Debug, thiserror::Error)]
pub enum UriError {
    /// The URI uses a scheme we don't recognise.
    #[error(
        "unsupported URI scheme '{0}'. Supported: \
         `memory://<ns>`, `file:///abs?ns=<ns>`, \
         `s3://<bucket>[/<prefix>]?ns=<ns>`, `gs://<bucket>?ns=<ns>`, \
         `az://<account>/<container>?ns=<ns>`"
    )]
    UnsupportedScheme(String),
    /// The URI is syntactically malformed for its scheme.
    #[error("invalid URI '{uri}': {reason}")]
    Malformed { uri: String, reason: String },
    /// A required query parameter is missing.
    #[error("URI '{uri}' is missing required parameter: {param}")]
    MissingParam { uri: String, param: &'static str },
    /// The namespace failed validation.
    #[error("bad namespace '{name}': {reason}")]
    BadNamespace { name: String, reason: String },
    /// The underlying object-store client failed to initialise.
    #[error("failed to open backend for '{uri}': {source}")]
    BackendInit {
        uri: String,
        #[source]
        source: anyhow::Error,
    },
}

/// Parse a NamiDB storage URI and return a ready-to-use object-store
/// handle plus the namespace-rooted path layout.
pub fn parse_uri(uri: &str) -> Result<(Arc<dyn ObjectStore>, NamespacePaths), UriError> {
    if uri.starts_with("memory://") {
        return parse_memory(uri);
    }
    if uri.starts_with("file://") {
        return parse_file(uri);
    }
    if uri.starts_with("s3://") {
        return parse_s3(uri);
    }
    if uri.starts_with("gs://") {
        return parse_gcs(uri);
    }
    if uri.starts_with("az://") {
        return parse_azure(uri);
    }
    Err(UriError::UnsupportedScheme(uri.to_string()))
}

/// Parse a storage URI into just the object-store handle, WITHOUT
/// requiring the `?ns=` parameter.
///
/// Multi-tenant boot uses this: namespaces come from requests (and
/// `--default-namespace`), so the URI's `ns` is meaningless there — the
/// old code still demanded it and then threw it away. Store construction
/// is delegated to [`parse_uri`] verbatim (a placeholder namespace
/// satisfies the shared parser when the URI carries none), so the two
/// entry points can never drift on endpoint/credential handling. A
/// present `?ns=` remains accepted and ignored.
pub fn parse_store(uri: &str) -> Result<Arc<dyn ObjectStore>, UriError> {
    match parse_uri(uri) {
        Ok((store, _)) => Ok(store),
        Err(UriError::MissingParam { param: "ns", .. }) => {
            let synthetic = if uri.contains('?') {
                format!("{uri}&ns=default")
            } else {
                format!("{uri}?ns=default")
            };
            parse_uri(&synthetic).map(|(store, _)| store)
        }
        Err(e) => Err(e),
    }
}

fn parse_memory(uri: &str) -> Result<(Arc<dyn ObjectStore>, NamespacePaths), UriError> {
    let rest = uri
        .strip_prefix("memory://")
        .expect("caller checked memory://");
    if rest.is_empty() {
        return Err(UriError::Malformed {
            uri: uri.to_string(),
            reason: "memory:// requires a namespace (e.g. memory://acme)".into(),
        });
    }
    let namespace = NamespaceId::new(rest).map_err(|e| UriError::BadNamespace {
        name: rest.to_string(),
        reason: e.to_string(),
    })?;
    Ok((
        Arc::new(InMemory::new()),
        NamespacePaths::new("", namespace),
    ))
}

fn parse_file(uri: &str) -> Result<(Arc<dyn ObjectStore>, NamespacePaths), UriError> {
    let rest = uri.strip_prefix("file://").expect("caller checked file://");
    if rest.is_empty() {
        return Err(UriError::Malformed {
            uri: uri.to_string(),
            reason: "file:// requires a directory path \
                (e.g. file:///var/lib/namidb?ns=acme or file://./data?ns=acme)"
                .into(),
        });
    }
    let (path_part, query_part) = match rest.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (rest, None),
    };
    if path_part.is_empty() {
        return Err(UriError::Malformed {
            uri: uri.to_string(),
            reason: "file:// path is empty".into(),
        });
    }
    let root = std::path::PathBuf::from(path_part);

    let mut ns_str: Option<String> = None;
    if let Some(q) = query_part {
        for (key, val) in url::form_urlencoded::parse(q.as_bytes()) {
            if key == "ns" || key == "namespace" {
                ns_str = Some(val.into_owned());
            }
        }
    }
    let ns_str = ns_str.ok_or_else(|| UriError::MissingParam {
        uri: uri.to_string(),
        param: "ns",
    })?;
    let namespace = NamespaceId::new(&ns_str).map_err(|e| UriError::BadNamespace {
        name: ns_str,
        reason: e.to_string(),
    })?;
    let store = LocalFileObjectStore::new(&root).map_err(|e| UriError::BackendInit {
        uri: uri.to_string(),
        source: anyhow::anyhow!("{e}"),
    })?;
    Ok((
        Arc::new(store) as Arc<dyn ObjectStore>,
        NamespacePaths::new("", namespace),
    ))
}

/// Optional client tuning for the HTTP-backed stores (S3/GCS/Azure), from
/// env. UNSET = object_store's defaults (30s/request, 180s retry budget) —
/// shipped behavior is unchanged. Item 59's residual window is a single
/// in-flight op (the commit path aborts BETWEEN ops at its deadline), so an
/// operator who saw a multi-minute retry storm can bound it here:
/// `NAMIDB_STORE_REQUEST_TIMEOUT` / `NAMIDB_STORE_RETRY_TIMEOUT` (seconds,
/// or `<n>ms`) and `NAMIDB_STORE_MAX_RETRIES`. Applies to every op on the
/// store, maintenance included — size generously.
fn env_duration(name: &str) -> Option<std::time::Duration> {
    let raw = std::env::var(name).ok()?;
    let raw = raw.trim();
    let parsed = if let Some(ms) = raw.strip_suffix("ms") {
        ms.trim()
            .parse::<u64>()
            .ok()
            .map(std::time::Duration::from_millis)
    } else {
        let secs = raw.strip_suffix('s').unwrap_or(raw).trim();
        secs.parse::<u64>().ok().map(std::time::Duration::from_secs)
    };
    if parsed.is_none() {
        tracing::warn!(env = name, value = raw, "unparseable duration ignored");
    }
    parsed
}

fn store_retry_config() -> Option<object_store::RetryConfig> {
    let retry_timeout = env_duration("NAMIDB_STORE_RETRY_TIMEOUT");
    let max_retries = std::env::var("NAMIDB_STORE_MAX_RETRIES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok());
    if retry_timeout.is_none() && max_retries.is_none() {
        return None;
    }
    let mut config = object_store::RetryConfig::default();
    if let Some(t) = retry_timeout {
        config.retry_timeout = t;
    }
    if let Some(n) = max_retries {
        config.max_retries = n;
    }
    Some(config)
}

fn store_client_options() -> Option<object_store::ClientOptions> {
    env_duration("NAMIDB_STORE_REQUEST_TIMEOUT")
        .map(|t| object_store::ClientOptions::new().with_timeout(t))
}

fn parse_s3(uri: &str) -> Result<(Arc<dyn ObjectStore>, NamespacePaths), UriError> {
    let parsed = url::Url::parse(uri).map_err(|e| UriError::Malformed {
        uri: uri.to_string(),
        reason: e.to_string(),
    })?;
    let bucket = parsed.host_str().ok_or_else(|| UriError::Malformed {
        uri: uri.to_string(),
        reason: "s3:// requires a bucket name".into(),
    })?;
    let path_prefix = parsed.path().trim_start_matches('/');
    let store_prefix = if path_prefix.is_empty() {
        "tenants".to_string()
    } else {
        path_prefix.to_string()
    };

    let mut ns_str: Option<String> = None;
    let mut region: Option<String> = None;
    let mut endpoint: Option<String> = None;
    let mut allow_http = false;
    for (key, val) in parsed.query_pairs() {
        match key.as_ref() {
            "ns" | "namespace" => ns_str = Some(val.into_owned()),
            "region" => region = Some(val.into_owned()),
            "endpoint" => endpoint = Some(val.into_owned()),
            "allow_http" => allow_http = matches!(val.as_ref(), "true" | "1" | "yes"),
            _ => {}
        }
    }
    let ns_str = ns_str.ok_or_else(|| UriError::MissingParam {
        uri: uri.to_string(),
        param: "ns",
    })?;
    let namespace = NamespaceId::new(&ns_str).map_err(|e| UriError::BadNamespace {
        name: ns_str,
        reason: e.to_string(),
    })?;
    let mut builder = object_store::aws::AmazonS3Builder::from_env().with_bucket_name(bucket);
    if let Some(r) = region {
        builder = builder.with_region(r);
    }
    if let Some(ep) = endpoint {
        builder = builder.with_endpoint(ep);
    }
    if allow_http {
        builder = builder.with_allow_http(true);
    }
    if let Some(options) = store_client_options() {
        builder = builder.with_client_options(options);
    }
    if let Some(retry) = store_retry_config() {
        builder = builder.with_retry(retry);
    }
    let store = builder.build().map_err(|e| UriError::BackendInit {
        uri: uri.to_string(),
        source: anyhow::anyhow!("{e}"),
    })?;
    Ok((
        Arc::new(store) as Arc<dyn ObjectStore>,
        NamespacePaths::new(&store_prefix, namespace),
    ))
}

fn parse_gcs(uri: &str) -> Result<(Arc<dyn ObjectStore>, NamespacePaths), UriError> {
    let parsed = url::Url::parse(uri).map_err(|e| UriError::Malformed {
        uri: uri.to_string(),
        reason: e.to_string(),
    })?;
    let bucket = parsed.host_str().ok_or_else(|| UriError::Malformed {
        uri: uri.to_string(),
        reason: "gs:// requires a bucket name".into(),
    })?;
    let path_prefix = parsed.path().trim_start_matches('/');
    let store_prefix = if path_prefix.is_empty() {
        "tenants".to_string()
    } else {
        path_prefix.to_string()
    };

    let mut ns_str: Option<String> = None;
    let mut service_account: Option<String> = None;
    for (key, val) in parsed.query_pairs() {
        match key.as_ref() {
            "ns" | "namespace" => ns_str = Some(val.into_owned()),
            "service_account" => service_account = Some(val.into_owned()),
            _ => {}
        }
    }
    let ns_str = ns_str.ok_or_else(|| UriError::MissingParam {
        uri: uri.to_string(),
        param: "ns",
    })?;
    let namespace = NamespaceId::new(&ns_str).map_err(|e| UriError::BadNamespace {
        name: ns_str,
        reason: e.to_string(),
    })?;
    let mut builder =
        object_store::gcp::GoogleCloudStorageBuilder::from_env().with_bucket_name(bucket);
    if let Some(p) = service_account {
        builder = builder.with_service_account_path(p);
    }
    if let Some(options) = store_client_options() {
        builder = builder.with_client_options(options);
    }
    if let Some(retry) = store_retry_config() {
        builder = builder.with_retry(retry);
    }
    let store = builder.build().map_err(|e| UriError::BackendInit {
        uri: uri.to_string(),
        source: anyhow::anyhow!("{e}"),
    })?;
    Ok((
        Arc::new(store) as Arc<dyn ObjectStore>,
        NamespacePaths::new(&store_prefix, namespace),
    ))
}

fn parse_azure(uri: &str) -> Result<(Arc<dyn ObjectStore>, NamespacePaths), UriError> {
    let parsed = url::Url::parse(uri).map_err(|e| UriError::Malformed {
        uri: uri.to_string(),
        reason: e.to_string(),
    })?;
    let account = parsed.host_str().ok_or_else(|| UriError::Malformed {
        uri: uri.to_string(),
        reason: "az:// requires a storage account".into(),
    })?;
    let raw_path = parsed.path().trim_start_matches('/');
    let (container, remainder) = match raw_path.split_once('/') {
        Some((c, rest)) => (c, rest),
        None => (raw_path, ""),
    };
    if container.is_empty() {
        return Err(UriError::Malformed {
            uri: uri.to_string(),
            reason: "az:// requires a container: `az://<account>/<container>?ns=<ns>`".into(),
        });
    }
    let store_prefix = if remainder.is_empty() {
        "tenants".to_string()
    } else {
        remainder.to_string()
    };

    let mut ns_str: Option<String> = None;
    let mut endpoint: Option<String> = None;
    let mut allow_http = false;
    let mut use_emulator = false;
    for (key, val) in parsed.query_pairs() {
        match key.as_ref() {
            "ns" | "namespace" => ns_str = Some(val.into_owned()),
            "endpoint" => endpoint = Some(val.into_owned()),
            "allow_http" => allow_http = matches!(val.as_ref(), "true" | "1" | "yes"),
            "use_emulator" => use_emulator = matches!(val.as_ref(), "true" | "1" | "yes"),
            _ => {}
        }
    }
    let ns_str = ns_str.ok_or_else(|| UriError::MissingParam {
        uri: uri.to_string(),
        param: "ns",
    })?;
    let namespace = NamespaceId::new(&ns_str).map_err(|e| UriError::BadNamespace {
        name: ns_str,
        reason: e.to_string(),
    })?;
    let mut builder = object_store::azure::MicrosoftAzureBuilder::from_env()
        .with_account(account)
        .with_container_name(container);
    if let Some(ep) = endpoint {
        builder = builder.with_endpoint(ep);
    }
    if allow_http {
        builder = builder.with_allow_http(true);
    }
    if use_emulator {
        builder = builder.with_use_emulator(true);
    }
    if let Some(options) = store_client_options() {
        builder = builder.with_client_options(options);
    }
    if let Some(retry) = store_retry_config() {
        builder = builder.with_retry(retry);
    }
    let store = builder.build().map_err(|e| UriError::BackendInit {
        uri: uri.to_string(),
        source: anyhow::anyhow!("{e}"),
    })?;
    Ok((
        Arc::new(store) as Arc<dyn ObjectStore>,
        NamespacePaths::new(&store_prefix, namespace),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_ok() {
        let (_store, paths) = parse_uri("memory://acme").unwrap();
        assert_eq!(paths.namespace().as_str(), "acme");
    }

    #[test]
    fn memory_requires_namespace() {
        assert!(matches!(
            parse_uri("memory://"),
            Err(UriError::Malformed { .. })
        ));
    }

    #[test]
    fn file_requires_ns() {
        let dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", dir.path().display());
        assert!(matches!(
            parse_uri(&uri),
            Err(UriError::MissingParam { param: "ns", .. })
        ));
    }

    /// Multi-tenant boot path: the store opens without `?ns=`, a present
    /// one stays accepted, and `parse_uri` keeps requiring it (pinned by
    /// [`file_requires_ns`] above — single-tenant is unchanged).
    #[test]
    fn parse_store_does_not_require_ns() {
        let dir = tempfile::tempdir().unwrap();
        let bare = format!("file://{}", dir.path().display());
        assert!(parse_store(&bare).is_ok());
        assert!(parse_store(&format!("{bare}?ns=acme")).is_ok());
        assert!(matches!(
            parse_store("ftp://nope"),
            Err(UriError::UnsupportedScheme(_))
        ));
        // memory:// takes its namespace from the authority; an empty one
        // stays malformed even for the store-only entry point.
        assert!(parse_store("memory://demo").is_ok());
        assert!(matches!(
            parse_store("memory://"),
            Err(UriError::Malformed { .. })
        ));
    }

    #[test]
    fn file_ok() {
        let dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}?ns=acme", dir.path().display());
        let (_store, paths) = parse_uri(&uri).unwrap();
        assert_eq!(paths.namespace().as_str(), "acme");
    }

    #[test]
    fn unsupported_scheme_is_reported() {
        let err = parse_uri("ftp://nope").unwrap_err();
        assert!(matches!(err, UriError::UnsupportedScheme(_)));
    }
}
