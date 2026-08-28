//! Startup validation: misconfigurations must fail the boot loudly instead
//! of silently serving a degraded or wide-open server.
//!
//! Both refusals fire before the store is opened, so `run()` returns the
//! error immediately and no server task is left behind.

use std::time::Duration;

fn base_config(ns: &str) -> namidb_server::Config {
    namidb_server::Config {
        store_uri: format!("memory://{ns}"),
        listen: "127.0.0.1:0".parse().unwrap(),
        auth_token: Some("test-token".into()),
        auth_tokens_file: None,
        no_auth: false,
        #[cfg(feature = "jwt")]
        jwt: None,
        #[cfg(feature = "pdp")]
        pdp_url: None,
        flush_interval: Duration::ZERO,
        compaction_interval: Duration::ZERO,
        sweep_min_age: Duration::ZERO,
        sweep_delete: false,
        bolt_listen: None,
        bolt_max_message_bytes: 64 << 20,
        bolt_tx_timeout: Duration::ZERO,
        query_timeout: Duration::from_secs(30),
        write_timeout: Duration::from_secs(30),
        query_row_cap: 0,
        compaction_l0_trigger: 0,
        write_stall_l0: 0,
        write_stall_delay: Duration::ZERO,
        memtable_flush_bytes: 0,
        memtable_stall_bytes: 0,
        writer_lock_timeout: Duration::from_secs(5),
        tls_cert: None,
        tls_key: None,
        slow_query_threshold: Duration::ZERO,
        multi_tenant: false,
        default_namespace: ns.to_string(),
        max_namespaces: 100,
        namespace_idle_timeout: Duration::from_secs(3600),
    }
}

#[tokio::test]
async fn multi_tenant_with_bolt_listen_refuses_to_boot() {
    let mut config = base_config("startup-mt-bolt");
    config.multi_tenant = true;
    config.bolt_listen = Some("127.0.0.1:0".parse().unwrap());
    let err = namidb_server::run(config).await.unwrap_err();
    assert!(
        err.to_string()
            .contains("--bolt-listen is not supported with --multi-tenant"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn missing_auth_refuses_to_boot_without_explicit_no_auth() {
    let mut config = base_config("startup-open");
    config.auth_token = None;
    let err = namidb_server::run(config).await.unwrap_err();
    assert!(
        err.to_string().contains("no auth configured"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn empty_auth_token_refuses_to_boot() {
    let mut config = base_config("startup-empty-token");
    config.auth_token = Some(String::new());
    let err = namidb_server::run(config).await.unwrap_err();
    assert!(
        err.to_string().contains("--auth-token is empty"),
        "unexpected error: {err}"
    );
}
