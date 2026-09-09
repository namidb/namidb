//! Fourth field report, item 64: a server-side read admission queue.
//! Single-test binary (the gate is a process-wide OnceLock configured from
//! env, so this test must own the process — the project convention for
//! env-mutating tests).

use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_admission_bounds_concurrency_and_rejects_retryably() {
    // MUST run before anything touches the gate: one permit, 100ms window.
    std::env::set_var("NAMIDB_MAX_CONCURRENT_QUERIES", "1");
    std::env::set_var("NAMIDB_QUERY_ADMISSION_WAIT_MS", "100");

    let (store, paths) = namidb_storage::parse_uri("memory://admission").unwrap();
    let writer = namidb_storage::WriterSession::open(store, paths)
        .await
        .unwrap();
    let state = namidb_server::AppState::new(writer, None, "admission".into());
    let app = namidb_server::build_router(state);

    let post = |app: axum::Router, query: &'static str| async move {
        let body = serde_json::to_vec(&serde_json::json!({ "query": query })).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .header("content-length", body.len().to_string())
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    };

    // Deterministic occupancy: hold the single permit directly.
    let plan = {
        let q = namidb_query::parse("MATCH (n:X) RETURN count(n) AS n").unwrap();
        namidb_query::plan(&q, &namidb_query::StatsCatalog::empty()).unwrap()
    };
    let held = namidb_server::acquire_read_admission(&plan)
        .await
        .expect("first admission must succeed");

    // A read while the gate is full: retryable 503 naming the knob.
    let (status, body) = post(app.clone(), "RETURN 1 AS one").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert!(
        body.contains("too many concurrent queries")
            && body.contains("NAMIDB_MAX_CONCURRENT_QUERIES"),
        "rejection must be actionable: {body}"
    );

    // Writes are NOT gated here (they serialize on the writer lock).
    let (status, body) = post(app.clone(), "CREATE (:W {ok: true})").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "writes must pass the read gate: {body}"
    );

    // Releasing the permit re-admits reads within the wait window.
    drop(held);
    tokio::time::sleep(Duration::from_millis(10)).await;
    let (status, body) = post(app.clone(), "RETURN 1 AS one").await;
    assert_eq!(status, StatusCode::OK, "{body}");
}
