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

    let post_full = |app: axum::Router, query: &'static str| async move {
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
        let retry_after = response
            .headers()
            .get("retry-after")
            .map(|v| v.to_str().unwrap().to_string());
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (
            status,
            String::from_utf8_lossy(&bytes).into_owned(),
            retry_after,
        )
    };
    let post = |app: axum::Router, query: &'static str| async move {
        let (status, body, _) = post_full(app, query).await;
        (status, body)
    };

    // Deterministic occupancy: hold the single permit directly.
    let plan = {
        let q = namidb_query::parse("MATCH (n:X) RETURN count(n) AS n").unwrap();
        namidb_query::plan(&q, &namidb_query::StatsCatalog::empty()).unwrap()
    };
    let held = namidb_server::acquire_read_admission(&plan, None)
        .await
        .expect("first admission must succeed");

    // A read while the gate is full waits in line for the whole window,
    // then gets a retryable 503 naming the knob, with `Retry-After`.
    let waited = std::time::Instant::now();
    let (status, body, retry_after) = post_full(app.clone(), "RETURN 1 AS one").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert!(
        waited.elapsed() >= Duration::from_millis(100),
        "the refusal must come after the queue deadline, not at once"
    );
    assert!(
        body.contains("too many concurrent queries")
            && body.contains("NAMIDB_MAX_CONCURRENT_QUERIES")
            && body.contains("\"code\":\"overloaded\""),
        "rejection must be actionable: {body}"
    );
    assert_eq!(
        retry_after.as_deref(),
        Some("2"),
        "refusals carry Retry-After"
    );

    // Writes are NOT gated here (they serialize on the writer lock).
    let (status, body) = post(app.clone(), "CREATE (:W {ok: true})").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "writes must pass the read gate: {body}"
    );

    // A read that arrives while the gate is full is admitted as soon as the
    // slot frees up inside its window: queued, not refused.
    let queued = tokio::spawn(post(app.clone(), "RETURN 1 AS one"));
    tokio::time::sleep(Duration::from_millis(30)).await;
    drop(held);
    let (status, body) = queued.await.unwrap();
    assert_eq!(status, StatusCode::OK, "a queued read must run: {body}");

    // Releasing the permit re-admits reads within the wait window.
    tokio::time::sleep(Duration::from_millis(10)).await;
    let (status, body) = post(app.clone(), "RETURN 1 AS one").await;
    assert_eq!(status, StatusCode::OK, "{body}");
}
