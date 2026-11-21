//! The panic hook records a panic of any thread to `NAMIDB_PANIC_LOG`,
//! through `tracing`, and still runs the hook installed before it.
//!
//! The server binary aborts on panic; this test binary unwinds, so the
//! panicking thread can be joined and the record read back. One test per
//! binary: the hook and the tracing subscriber are process-global.
#![cfg(panic = "unwind")]

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

static PANIC_LINE: AtomicU32 = AtomicU32::new(0);
static PREVIOUS_HOOK_RAN: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn a_thread_panic_is_appended_logged_and_chained() {
    let capture = Capture::default();
    let writer = capture.clone();
    tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_ansi(false)
        .init();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("panic.log");
    std::fs::write(&path, "earlier record\n").unwrap();
    std::panic::set_hook(Box::new(|_| {
        PREVIOUS_HOOK_RAN.store(true, Ordering::SeqCst);
    }));
    namidb_server::panic_log::install(Some(path.clone()));

    let joined = std::thread::Builder::new()
        .name("panic-probe".into())
        .spawn(|| {
            PANIC_LINE.store(line!() + 1, Ordering::SeqCst);
            panic!("probe panic {}", 42);
        })
        .unwrap()
        .join();
    assert!(joined.is_err(), "the probe thread must have panicked");
    assert!(
        PREVIOUS_HOOK_RAN.load(Ordering::SeqCst),
        "the previous hook still runs"
    );

    let line = PANIC_LINE.load(Ordering::SeqCst);
    let location = format!("{}:{line}:", file!());
    let record = std::fs::read_to_string(&path).unwrap();
    assert!(record.starts_with("earlier record\n"), "appends: {record}");
    for needle in [
        "=== namidb-server panic at 20",
        "thread: panic-probe",
        &format!("location: {location}"),
        "message: probe panic 42",
        "backtrace:",
    ] {
        assert!(record.contains(needle), "missing {needle:?} in {record}");
    }

    let logged = String::from_utf8(capture.0.lock().unwrap().clone()).unwrap();
    assert!(logged.contains("ERROR"), "{logged}");
    assert!(logged.contains("panic: probe panic 42"), "{logged}");
    assert!(logged.contains("thread=panic-probe"), "{logged}");
    assert!(logged.contains(&location), "{logged}");
}
