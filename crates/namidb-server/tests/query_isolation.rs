//! A long read must neither starve the server nor outlive its client.
//!
//! Before reads ran on threads of their own, a CPU-bound read executed
//! inline on a runtime worker and never yielded: with as many such reads as
//! workers, nothing else was answered (health checks included), and a client
//! that disconnected could not be noticed until the read finished, because
//! the task that would notice was the one running it.
//!
//! Single-test binary: [`namidb_server::running_isolated_reads`] is
//! process-wide, so a parallel test would disturb the counts.

use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

/// CPU-bound for minutes (a million rows, each summing a 5,000-element
/// list) while holding little memory: it only ever stops when cancelled.
const ENDLESS: &str =
    "UNWIND range(1, 1000000) AS i RETURN sum(reduce(s = 0, x IN range(1, 5000) | s + x)) AS t";

fn post(query: &str) -> Request<Body> {
    let body = serde_json::to_vec(&serde_json::json!({ "query": query })).unwrap();
    Request::builder()
        .method("POST")
        .uri("/v0/cypher")
        .header("content-type", "application/json")
        .header("content-length", body.len().to_string())
        .body(Body::from(body))
        .unwrap()
}

async fn wait_for(what: &str, limit: Duration, mut done: impl FnMut() -> bool) -> Duration {
    let started = Instant::now();
    while !done() {
        assert!(
            started.elapsed() < limit,
            "{what} did not happen within {limit:?}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    started.elapsed()
}

// One runtime worker: before the fix, the endless read took it and the
// server answered nothing else until the read finished.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn long_reads_keep_the_server_responsive_and_stop_when_the_client_leaves() {
    let (store, paths) = namidb_storage::parse_uri("memory://isolation").unwrap();
    let writer = namidb_storage::WriterSession::open(store, paths)
        .await
        .unwrap();
    let state = namidb_server::AppState::new(writer, None, "isolation".into())
        .with_query_timeout(Duration::ZERO)
        .with_query_row_cap(0);
    let app = namidb_server::build_router(state);

    // 1. While an endless read runs, other requests are still answered.
    let endless = tokio::spawn(app.clone().oneshot(post(ENDLESS)));
    wait_for("the endless read to start", Duration::from_secs(10), || {
        namidb_server::running_isolated_reads() == 1
    })
    .await;
    let started = Instant::now();
    let response = tokio::time::timeout(
        Duration::from_secs(10),
        app.clone().oneshot(post("RETURN 1 AS one")),
    )
    .await
    .expect("a trivial read must not wait for the endless one")
    .unwrap();
    let answered_in = started.elapsed();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert!(String::from_utf8_lossy(&body).contains("one"));
    assert!(
        answered_in < Duration::from_secs(2),
        "a trivial read took {answered_in:?} next to the endless one"
    );
    assert!(
        !endless.is_finished(),
        "the endless read finished on its own"
    );

    // 2. The client goes away (hyper drops the handler future on a
    // disconnect, as aborting the request task does here): the read itself
    // stops, promptly, not only its response.
    endless.abort();
    let _ = endless.await;
    let stopped_in = wait_for("the abandoned read to stop", Duration::from_secs(5), || {
        namidb_server::running_isolated_reads() == 0
    })
    .await;
    eprintln!("trivial read answered in {answered_in:?}; abandoned read stopped in {stopped_in:?}");
    assert!(
        stopped_in < Duration::from_secs(1),
        "the abandoned read kept running for {stopped_in:?}"
    );

    // 3. The server is idle and healthy afterwards.
    let response = app.clone().oneshot(post("RETURN 2 AS two")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
