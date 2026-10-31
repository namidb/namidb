//! Common URI → (`ObjectStore`, `NamespacePaths`) parser shared by the
//! CLI, Python bindings, and any future server binary.
//!
//! Supported schemes:
//!
//! - `memory://<namespace>` (or `memory://?ns=<namespace>`): ephemeral,
//!   single-process.
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
    // The namespace is the authority (`memory://acme`) or `?ns=` like the
    // other schemes (`memory://?ns=acme`); both is fine when they agree.
    let (host, query) = match rest.split_once('?') {
        Some((h, q)) => (h, Some(q)),
        None => (rest, None),
    };
    let mut ns_query: Option<String> = None;
    if let Some(q) = query {
        for (key, val) in url::form_urlencoded::parse(q.as_bytes()) {
            if key == "ns" || key == "namespace" {
                ns_query = Some(val.into_owned());
            } else {
                // memory:// takes nothing else, so a typo like `?nss=`
                // must not quietly open the host namespace.
                return Err(UriError::Malformed {
                    uri: uri.to_string(),
                    reason: format!("unknown memory:// parameter '{key}'"),
                });
            }
        }
    }
    let ns_str = match (host, ns_query) {
        ("", None) => {
            return Err(UriError::Malformed {
                uri: uri.to_string(),
                reason: "memory:// requires a namespace (e.g. memory://acme)".into(),
            });
        }
        (h, Some(q)) if !h.is_empty() && h != q => {
            return Err(UriError::Malformed {
                uri: uri.to_string(),
                reason: format!("memory:// namespace '{h}' conflicts with ?ns={q}"),
            });
        }
        (_, Some(q)) => q,
        (h, None) => h.to_string(),
    };
    let namespace = NamespaceId::new(&ns_str).map_err(|e| UriError::BadNamespace {
        name: ns_str.clone(),
        reason: e.to_string(),
    })?;
    // Deliberately NOT marked create-only for pinned reads (unlike file://):
    // InMemory ETags are stable per-instance counters with no churn to
    // tolerate, and overwriting a path is legal against this store, so the
    // `If-Match` pin still carries weight here.
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
    let store: Arc<dyn ObjectStore> = Arc::new(store);
    // NamiDB writes every local object exactly once under a fresh UUID name
    // (tmp + atomic rename; `O_CREAT|O_EXCL` for pointers), so pinned reads
    // may key by the immutable path. The alternative — `If-Match` on
    // `LocalFileSystem`'s `{inode:x}-{mtime:x}-{size:x}` ETag — fails healthy
    // reads on FUSE-backed bind mounts (macOS Docker Desktop gRPC-FUSE),
    // which churn the synthetic inode under an unmodified file.
    // `NAMIDB_LOCAL_ETAG_PIN=1` restores the ETag pin.
    crate::range_cache::mark_store_paths_immutable(&store);
    Ok((store, NamespacePaths::new("", namespace)))
}

/// Check that `uri` names a location inside the operator allowlist `prefix`
/// (the server's `--backup-target-uri`), returning why not otherwise.
///
/// The server's ambient credentials read and write wherever such a URI
/// says, so the comparison is on parsed components, never on raw strings:
///
/// - the scheme and the bucket / account / host must equal the prefix's;
/// - the path segments must start with the prefix's segments (a trailing `/`
///   on the prefix is ignored), and every segment after them must be
///   non-empty and neither `.` nor `..`, raw or percent-encoded;
/// - query parameters: `uri` may add only `ns` / `namespace` (once). Every
///   other parameter (`endpoint`, `region`, `allow_http`, `service_account`,
///   …) must appear in `uri` with exactly the prefix's value, and none may
///   be added — so the store options always come from the operator. An `ns`
///   on the prefix is ignored;
/// - `#`, `\`, whitespace and control characters are refused outright (the
///   URL parser strips or reinterprets some of them, e.g. a tab inside `..`);
/// - for `file://`, the canonicalized path (symlinks resolved, through its
///   deepest existing ancestor) must also sit inside the canonicalized
///   prefix path.
pub fn check_uri_within_prefix(uri: &str, prefix: &str) -> Result<(), String> {
    if let Some(c) = uri
        .chars()
        .find(|c| *c == '#' || *c == '\\' || c.is_whitespace() || c.is_control())
    {
        return Err(format!("character {c:?} is not allowed"));
    }
    let want = SplitUri::new(prefix).ok_or_else(|| "the allowlist is not a URI".to_string())?;
    let got = SplitUri::new(uri).ok_or_else(|| "not a `scheme://` URI".to_string())?;
    if got.scheme != want.scheme {
        return Err(format!(
            "scheme `{}` differs from the allowlist's `{}`",
            got.scheme, want.scheme
        ));
    }
    if got.authority != want.authority {
        return Err(format!(
            "bucket/host `{}` differs from the allowlist's `{}`",
            got.authority, want.authority
        ));
    }

    let want_segments = path_segments(want.path.trim_end_matches('/'));
    let got_segments = path_segments(got.path);
    if got_segments.len() < want_segments.len()
        || got_segments[..want_segments.len()] != want_segments[..]
    {
        return Err("path is not under the allowlist's path".into());
    }
    for segment in &got_segments[want_segments.len()..] {
        let decoded = percent_decode(segment);
        if decoded.is_empty() || decoded == "." || decoded == ".." {
            return Err(format!("path segment `{segment}` is not allowed"));
        }
        if decoded.contains(['/', '\\']) {
            return Err(format!("path segment `{segment}` encodes a separator"));
        }
    }

    let want_params = store_params(want.query, false)?;
    let got_params = store_params(got.query, true)?;
    for (key, value) in &got_params {
        match want_params.get(key) {
            None => return Err(format!("query parameter `{key}` is not in the allowlist")),
            Some(expected) if expected != value => {
                return Err(format!(
                    "query parameter `{key}` differs from the allowlist's value"
                ));
            }
            Some(_) => {}
        }
    }
    if let Some(key) = want_params.keys().find(|k| !got_params.contains_key(*k)) {
        return Err(format!(
            "query parameter `{key}` is required with the allowlist's value"
        ));
    }

    if got.scheme == "file" {
        let resolve = |path: &str| {
            canonicalize_lenient(std::path::Path::new(path))
                .map_err(|e| format!("cannot resolve `{path}`: {e}"))
        };
        let want_dir = resolve(want.path)?;
        let got_dir = resolve(got.path)?;
        if !got_dir.starts_with(&want_dir) {
            return Err(format!(
                "path resolves to `{}`, outside `{}`",
                got_dir.display(),
                want_dir.display()
            ));
        }
    }
    Ok(())
}

/// A URI cut into the parts [`check_uri_within_prefix`] compares, without
/// any normalisation. `file://` has no authority: everything up to the `?`
/// is the path, as [`parse_uri`] reads it.
struct SplitUri<'a> {
    scheme: &'a str,
    authority: &'a str,
    path: &'a str,
    query: Option<&'a str>,
}

impl<'a> SplitUri<'a> {
    fn new(uri: &'a str) -> Option<Self> {
        let (scheme, rest) = uri.split_once("://")?;
        let (base, query) = match rest.split_once('?') {
            Some((base, query)) => (base, Some(query)),
            None => (rest, None),
        };
        let (authority, path) = if scheme == "file" {
            ("", base)
        } else {
            match base.find('/') {
                Some(i) => (&base[..i], &base[i..]),
                None => (base, ""),
            }
        };
        Some(Self {
            scheme,
            authority,
            path,
            query,
        })
    }
}

fn path_segments(path: &str) -> Vec<&str> {
    let path = path.strip_prefix('/').unwrap_or(path);
    if path.is_empty() {
        Vec::new()
    } else {
        path.split('/').collect()
    }
}

/// `%XX` decoding of one path segment (no `+` handling: that is a query
/// convention). Invalid escapes are kept literally.
fn percent_decode(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
            if let Some(b) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The query parameters other than `ns` / `namespace`, decoded the way the
/// scheme parsers decode them. With `strict`, a repeated key (counting `ns`
/// and `namespace` as one) is an error: the parsers keep the last value,
/// which would let a request smuggle a second one past a comparison.
fn store_params(
    query: Option<&str>,
    strict: bool,
) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut params = std::collections::BTreeMap::new();
    let mut saw_ns = false;
    for (key, value) in url::form_urlencoded::parse(query.unwrap_or("").as_bytes()) {
        if key == "ns" || key == "namespace" {
            if strict && saw_ns {
                return Err("the namespace is given more than once".into());
            }
            saw_ns = true;
            continue;
        }
        if params
            .insert(key.clone().into_owned(), value.into_owned())
            .is_some()
            && strict
        {
            return Err(format!("query parameter `{key}` is given more than once"));
        }
    }
    Ok(params)
}

/// [`std::fs::canonicalize`] for a path that may not exist yet: resolve the
/// deepest existing ancestor and append the missing components (which,
/// not existing, cannot be symlinks).
fn canonicalize_lenient(path: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
    let mut existing = path.to_path_buf();
    let mut missing = Vec::new();
    loop {
        let probe = if existing.as_os_str().is_empty() {
            std::path::Path::new(".")
        } else {
            existing.as_path()
        };
        match probe.canonicalize() {
            Ok(mut resolved) => {
                for component in missing.iter().rev() {
                    resolved.push(component);
                }
                return Ok(resolved);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let (Some(name), Some(parent)) = (existing.file_name(), existing.parent()) else {
                    return Err(e);
                };
                missing.push(name.to_os_string());
                existing = parent.to_path_buf();
            }
            Err(e) => return Err(e),
        }
    }
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
    fn memory_accepts_ns_query() {
        for uri in [
            "memory://acme?ns=acme",
            "memory://?ns=acme",
            "memory://?namespace=acme",
        ] {
            let (_store, paths) = parse_uri(uri).unwrap();
            assert_eq!(paths.namespace().as_str(), "acme", "{uri}");
        }
        assert!(matches!(
            parse_uri("memory://acme?ns=other"),
            Err(UriError::Malformed { .. })
        ));
        assert!(matches!(
            parse_uri("memory://?ns="),
            Err(UriError::BadNamespace { .. })
        ));
        for uri in ["memory://?foo=bar", "memory://acme?nss=prod"] {
            let err = parse_uri(uri).expect_err(uri);
            assert!(
                err.to_string().contains("unknown memory:// parameter"),
                "{uri}: {err}"
            );
        }
        let (_store, paths) = parse_uri("memory://forensic_case").unwrap();
        assert_eq!(paths.namespace().as_str(), "forensic_case");
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

    /// A `file://` store is registered create-only at parse time, so pinned
    /// reads use the immutable-path generation instead of `If-Match` on
    /// `LocalFileSystem`'s `{inode,mtime,size}` ETag — which FUSE-backed
    /// bind mounts churn for unmodified files.
    #[tokio::test]
    async fn file_store_pins_reads_by_immutable_path() {
        use object_store::ObjectStoreExt as _;

        let dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}?ns=acme", dir.path().display());
        let (store, _paths) = parse_uri(&uri).unwrap();
        let location = object_store::path::Path::from(format!("sst/{}.sst", uuid::Uuid::now_v7()));
        store
            .put(
                &location,
                object_store::PutPayload::from_static(b"immutable-bytes"),
            )
            .await
            .unwrap();
        let meta = store.head(&location).await.unwrap();
        assert!(
            meta.e_tag.is_some(),
            "LocalFileSystem reports a stat-derived ETag; the pin must still ignore it"
        );

        let source =
            crate::range_cache::PinnedObjectRangeSource::from_create_only_meta(store, meta)
                .await
                .unwrap();
        assert_eq!(
            source.generation(),
            &crate::range_cache::PinnedObjectGeneration::ImmutablePath
        );
        assert_eq!(
            source.read_range(0..15).await.unwrap(),
            bytes::Bytes::from_static(b"immutable-bytes")
        );
    }

    /// The `--backup-target-uri` boundary compares parsed components: same
    /// scheme and bucket, a segment-wise path prefix with no dot or empty
    /// segments after it, and store options fixed by the operator.
    #[test]
    fn allowlist_accepts_namespaces_under_the_prefix() {
        let prefix = "s3://bkt/backups?endpoint=http://minio:9000&region=eu-west-1";
        for ok in [
            "s3://bkt/backups?ns=acme&endpoint=http://minio:9000&region=eu-west-1",
            // Order-independent, `namespace` alias, the prefix itself.
            "s3://bkt/backups/nightly/acme?region=eu-west-1&namespace=acme&endpoint=http://minio:9000",
            "s3://bkt/backups?region=eu-west-1&endpoint=http%3A%2F%2Fminio%3A9000&ns=a",
        ] {
            check_uri_within_prefix(ok, prefix).unwrap_or_else(|e| panic!("{ok}: {e}"));
        }
        // A trailing `/` on the prefix is ignored.
        check_uri_within_prefix("gs://b/p/x?ns=x", "gs://b/p/").unwrap();
        check_uri_within_prefix("az://acct/c/p/x?ns=x", "az://acct/c").unwrap();
    }

    #[test]
    fn allowlist_refuses_traversal_and_lookalike_paths() {
        let prefix = "s3://bkt/backups";
        for bad in [
            "s3://bkt/backups-evil/x?ns=x",
            "s3://bkt/back?ns=x",
            "s3://bkt?ns=x",
            "s3://bkt/backups/../other?ns=x",
            "s3://bkt/backups/./x?ns=x",
            "s3://bkt/backups/x/..?ns=x",
            "s3://bkt/backups//x?ns=x",
            "s3://bkt/backups/x/?ns=x",
            "s3://bkt/backups/%2e%2e/other?ns=x",
            "s3://bkt/backups/%2E%2e/other?ns=x",
            "s3://bkt/backups/.%2e/other?ns=x",
            "s3://bkt/backups/%2e?ns=x",
            "s3://bkt/backups/a%2fb?ns=x",
            "s3://bkt/backups/.\t./other?ns=x",
            "s3://bkt/backups/x#frag?ns=x",
            "s3://bkt/backups/..\\x?ns=x",
            // Another bucket, user-info, scheme.
            "s3://other/backups/x?ns=x",
            "s3://evil@bkt/backups/x?ns=x",
            "gs://bkt/backups/x?ns=x",
            "file:///bkt/backups/x?ns=x",
            "bkt/backups/x?ns=x",
        ] {
            assert!(
                check_uri_within_prefix(bad, prefix).is_err(),
                "{bad} must be refused"
            );
        }
    }

    #[test]
    fn allowlist_pins_the_store_options_to_the_prefix() {
        let bare = "s3://bkt/backups";
        for bad in [
            "s3://bkt/backups/x?ns=x&endpoint=http://10.0.0.5:9000",
            "s3://bkt/backups/x?ns=x&allow_http=true",
            "s3://bkt/backups/x?ns=x&region=us-east-1",
            "s3://bkt/backups/x?ns=x&ns=y",
            "s3://bkt/backups/x?ns=x&namespace=y",
        ] {
            assert!(check_uri_within_prefix(bad, bare).is_err(), "{bad}");
        }
        let pinned = "s3://bkt/backups?endpoint=http://minio:9000&allow_http=true";
        for bad in [
            // Missing, changed, repeated, or extra options.
            "s3://bkt/backups/x?ns=x",
            "s3://bkt/backups/x?ns=x&endpoint=http://minio:9000",
            "s3://bkt/backups/x?ns=x&endpoint=http://evil:9000&allow_http=true",
            "s3://bkt/backups/x?ns=x&endpoint=http://minio:9000&allow_http=false",
            "s3://bkt/backups/x?ns=x&endpoint=http://minio:9000&allow_http=true&endpoint=http://evil:9000",
            "s3://bkt/backups/x?ns=x&endpoint=http://minio:9000&allow_http=true&region=x",
        ] {
            let err = check_uri_within_prefix(bad, pinned).expect_err(bad);
            assert!(err.contains("query parameter"), "{bad}: {err}");
        }
        let gcs = "gs://bkt/backups";
        assert!(
            check_uri_within_prefix("gs://bkt/backups/x?ns=x&service_account=/etc/key", gcs)
                .is_err()
        );
    }

    #[test]
    fn allowlist_resolves_file_paths() {
        let dir = tempfile::tempdir().unwrap();
        let backups = dir.path().join("backups");
        let data = dir.path().join("data");
        std::fs::create_dir_all(&backups).unwrap();
        std::fs::create_dir_all(&data).unwrap();
        let prefix = format!("file://{}", backups.display());
        // Missing directories are fine: they do not exist to resolve yet.
        check_uri_within_prefix(&format!("{prefix}/new/deeper?ns=x"), &prefix).unwrap();
        for bad in [
            format!("{prefix}/../data?ns=x"),
            format!("{prefix}/%2e%2e/data?ns=x"),
            format!("{prefix}-evil?ns=x"),
            format!("file://{}?ns=x", data.display()),
        ] {
            assert!(check_uri_within_prefix(&bad, &prefix).is_err(), "{bad}");
        }
        // A symlink under the prefix that leads out of it.
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&data, backups.join("link")).unwrap();
            let err = check_uri_within_prefix(&format!("{prefix}/link/x?ns=x"), &prefix)
                .expect_err("symlink escape");
            assert!(err.contains("outside"), "{err}");
        }
    }

    #[test]
    fn unsupported_scheme_is_reported() {
        let err = parse_uri("ftp://nope").unwrap_err();
        assert!(matches!(err, UriError::UnsupportedScheme(_)));
    }
}
