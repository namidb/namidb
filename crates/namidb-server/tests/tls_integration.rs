//! End-to-end TLS on both serving paths: HTTPS for the REST API and TLS for
//! the Bolt listener. The client verifies the self-signed certificate for
//! real (it is added to a root store), so this exercises the full handshake,
//! not a skip-verification shortcut.

use std::sync::Arc;
use std::time::Duration;

use namidb_server::Config;
use rustls::pki_types::{CertificateDer, ServerName};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;

struct Certs {
    cert_path: tempfile::TempPath,
    key_path: tempfile::TempPath,
    cert_der: Vec<u8>,
}

fn self_signed() -> Certs {
    use std::io::Write;
    let ck = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
    let cert_der = ck.cert.der().to_vec();
    let mut cert_file = tempfile::NamedTempFile::new().unwrap();
    cert_file.write_all(ck.cert.pem().as_bytes()).unwrap();
    let mut key_file = tempfile::NamedTempFile::new().unwrap();
    key_file
        .write_all(ck.key_pair.serialize_pem().as_bytes())
        .unwrap();
    Certs {
        cert_path: cert_file.into_temp_path(),
        key_path: key_file.into_temp_path(),
        cert_der,
    }
}

async fn free_addr() -> std::net::SocketAddr {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    drop(l);
    addr
}

async fn wait_ready(addr: std::net::SocketAddr) {
    for _ in 0..100 {
        if TcpStream::connect(addr).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server port {addr} never came up");
}

#[tokio::test]
async fn serves_https_and_bolt_over_tls() {
    let certs = self_signed();
    let http = free_addr().await;
    let bolt = free_addr().await;

    let config = Config {
        store_uri: "memory://tls-it".into(),
        listen: http,
        auth_token: None,
        auth_tokens_file: None,
        #[cfg(feature = "jwt")]
        jwt: None,
        #[cfg(feature = "pdp")]
        pdp_url: None,
        flush_interval: Duration::ZERO,
        compaction_interval: Duration::ZERO,
        sweep_min_age: Duration::ZERO,
        sweep_delete: false,
        bolt_listen: Some(bolt),
        bolt_tx_timeout: Duration::ZERO,
        query_timeout: Duration::ZERO,
        write_timeout: Duration::ZERO,
        query_row_cap: 0,
        compaction_l0_trigger: 0,
        write_stall_l0: 0,
        write_stall_delay: Duration::ZERO,
        memtable_flush_bytes: 0,
        memtable_stall_bytes: 0,
        tls_cert: Some(certs.cert_path.to_path_buf()),
        tls_key: Some(certs.key_path.to_path_buf()),
        slow_query_threshold: Duration::ZERO,
        multi_tenant: false,
        default_namespace: "tls-it".to_string(),
        max_namespaces: 100,
        namespace_idle_timeout: Duration::from_secs(3600),
    };
    let task = tokio::spawn(async move {
        let _ = namidb_server::run(config).await;
    });
    wait_ready(http).await;
    wait_ready(bolt).await;

    // HTTPS: the liveness probe over TLS, verifying the cert against the
    // "localhost" SAN (127.0.0.1 is where the server bound).
    let client = reqwest::Client::builder()
        .add_root_certificate(reqwest::Certificate::from_der(&certs.cert_der).unwrap())
        .build()
        .unwrap();
    let resp = client
        .get(format!("https://localhost:{}/v0/livez", http.port()))
        .send()
        .await
        .expect("HTTPS request");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert!(resp.text().await.unwrap().contains("ok"));

    // Bolt over TLS: complete a verifying TLS handshake, then the Bolt
    // handshake on top of the TLS stream.
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(CertificateDer::from(certs.cert_der.clone()))
        .unwrap();
    let client_cfg = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_root_certificates(roots)
    .with_no_client_auth();
    let connector = TlsConnector::from(Arc::new(client_cfg));
    let tcp = TcpStream::connect(("127.0.0.1", bolt.port()))
        .await
        .unwrap();
    let name = ServerName::try_from("localhost").unwrap();
    let mut tls = connector
        .connect(name, tcp)
        .await
        .expect("Bolt TLS handshake");

    // 0x6060B017 magic + Bolt 5.4 proposal.
    let hs = [
        0x60, 0x60, 0xB0, 0x17, 0x00, 0x00, 0x04, 0x05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    tls.write_all(&hs).await.unwrap();
    let mut reply = [0u8; 4];
    tls.read_exact(&mut reply).await.unwrap();
    assert_eq!(reply, [0, 0, 4, 5], "Bolt 5.4 negotiated over TLS");

    task.abort();
}
