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

    // Clients that disconnect while queued leave the line: with the slot
    // held, twenty queued requests are dropped; once the slot frees, a new
    // read is admitted at once instead of queueing behind dead waiters.
    let held = namidb_server::acquire_read_admission(&plan, None)
        .await
        .unwrap();
    let abandoned: Vec<_> = (0..20)
        .map(|_| tokio::spawn(post(app.clone(), "RETURN 1 AS one")))
        .collect();
    tokio::time::sleep(Duration::from_millis(30)).await;
    for request in &abandoned {
        request.abort();
    }
    for request in abandoned {
        assert!(request.await.unwrap_err().is_cancelled());
    }
    drop(held);
    let started = std::time::Instant::now();
    let (status, body) = post(app.clone(), "RETURN 1 AS one").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(started.elapsed() < Duration::from_millis(100));

    // A read waiting for memory headroom keeps its slot meanwhile (the
    // line does not move past it) and gives it back when it is refused.
    let starved = std::sync::Arc::new(
        namidb_server::memory::MemoryGovernor::new(1).with_admission_wait(Duration::from_secs(5)),
    );
    let waiting = tokio::spawn({
        let plan = plan.clone();
        let starved = std::sync::Arc::clone(&starved);
        async move {
            namidb_server::acquire_read_admission_with_memory(
                &plan,
                None,
                Some(&starved),
                tokio::time::Instant::now() + Duration::from_millis(300),
            )
            .await
            .map(drop)
        }
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    let soon = || tokio::time::Instant::now() + Duration::from_millis(50);
    assert_eq!(
        namidb_server::acquire_read_admission_until(&plan, None, soon())
            .await
            .map(drop),
        Err("NAMIDB_MAX_CONCURRENT_QUERIES"),
        "the slot stays with the read waiting for memory"
    );
    assert!(matches!(
        waiting.await.unwrap(),
        Err(namidb_server::AdmissionRefused::Memory(_))
    ));
    assert!(
        namidb_server::acquire_read_admission_until(&plan, None, soon())
            .await
            .is_ok(),
        "a memory refusal releases the slot"
    );

    // The admission window is never more than half the read timeout, and
    // the read timeout counts from arrival, queueing included.
    std::env::set_var("NAMIDB_QUERY_ADMISSION_WAIT_MS", "5000");
    assert_eq!(
        namidb_server::admission_window(Duration::ZERO),
        Duration::from_secs(5)
    );
    assert_eq!(
        namidb_server::admission_window(Duration::from_secs(2)),
        Duration::from_secs(1)
    );
    let (store, paths) = namidb_storage::parse_uri("memory://admission-timeout").unwrap();
    let writer = namidb_storage::WriterSession::open(store, paths)
        .await
        .unwrap();
    let timed = namidb_server::build_router(
        namidb_server::AppState::new(writer, None, "admission-timeout".into())
            .with_query_timeout(Duration::from_secs(2)),
    );
    let (status, body) = post(
        timed.clone(),
        "UNWIND range(1, 2000) AS i CREATE (:N {i: i})",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let held = namidb_server::acquire_read_admission(&plan, None)
        .await
        .unwrap();
    let started = std::time::Instant::now();
    let (status, body, retry_after) = post_full(timed.clone(), "RETURN 1 AS one").await;
    let waited = started.elapsed();
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(retry_after.as_deref(), Some("2"));
    assert!(
        waited >= Duration::from_secs(1) && waited < Duration::from_millis(2500),
        "refused after half the 2 s read timeout, not the 5 s wait: {waited:?}"
    );

    let started = std::time::Instant::now();
    let slow = tokio::spawn(post(
        timed.clone(),
        "MATCH (a:N), (b:N), (c:N) RETURN count(*) AS n",
    ));
    tokio::time::sleep(Duration::from_millis(900)).await;
    drop(held);
    let (status, body) = slow.await.unwrap();
    let elapsed = started.elapsed();
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert!(
        elapsed < Duration::from_millis(2700),
        "900 ms in the queue plus a 2 s timeout from admission would be \
         2.9 s; from arrival it is 2 s: {elapsed:?}"
    );
}
