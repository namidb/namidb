//! Process panic hook: record every panic before the process dies.
//!
//! The release profile builds with `panic = "abort"`, so a panic on any thread
//! ends the server. Without a hook the only trace is the default hook's line
//! on stderr, which is gone with the container, and no backtrace unless
//! `RUST_BACKTRACE` was set. [`install_from_env`] installs a hook that, for
//! every panic:
//!
//! 1. appends one record (UTC timestamp, thread, location, message and a
//!    forced backtrace) to the file named by [`PANIC_LOG_ENV`], when set, with
//!    a single `O_APPEND` write;
//! 2. logs the same fields through `tracing::error!`;
//! 3. calls the hook that was installed before it (the default one prints to
//!    stderr), so nothing that used to appear there is lost.
//!
//! The file comes first because it needs the least machinery: if the panic
//! came from inside the logging stack, the record is already on disk.
//!
//! The hook takes no lock of its own. The file is opened per record (no shared
//! handle to guard), and a thread that panics again while the hook runs on it
//! skips straight to the previous hook. Every field is truncated to a fixed
//! size, so a huge panic payload cannot make the hook allocate without bound.
//! The file rotates once to `<path>.1` past [`MAX_LOG_BYTES`].
//!
//! What a panic hook cannot see: a stack overflow (the runtime prints
//! "has overflowed its stack" and aborts without a panic) and an allocation
//! failure (`memory allocation of N bytes failed`). Both still reach stderr.

use std::cell::Cell;
use std::fmt::Write as _;
use std::io::Write as _;
use std::panic::PanicHookInfo;
use std::path::{Path, PathBuf};

/// Path of the file panic records are appended to. Unset or empty: panics
/// are logged and printed to stderr only.
pub const PANIC_LOG_ENV: &str = "NAMIDB_PANIC_LOG";

/// Size past which the file is renamed to `<path>.1` (replacing an older one)
/// before the next record, so a crash loop cannot fill the volume.
pub const MAX_LOG_BYTES: u64 = 16 * 1024 * 1024;

/// Longest panic message kept, in bytes.
const MAX_MESSAGE_BYTES: usize = 8 * 1024;
/// Longest backtrace kept, in bytes.
const MAX_BACKTRACE_BYTES: usize = 64 * 1024;
/// Longest thread name kept, in bytes.
const MAX_THREAD_NAME_BYTES: usize = 256;

thread_local! {
    /// Set while this thread runs the hook, so a panic inside it (a broken
    /// writer, say) does not recurse into the file and logging steps.
    static IN_HOOK: Cell<bool> = const { Cell::new(false) };
}

/// Install the hook, with the file from [`PANIC_LOG_ENV`]. Call once, after
/// the tracing subscriber is set up. Returns the file path in use.
pub fn install_from_env() -> Option<PathBuf> {
    let path = std::env::var_os(PANIC_LOG_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    install(path.clone());
    path
}

/// Install the hook, appending records to `path` when given. Chains to the
/// hook installed before it. A path that cannot be opened now is reported
/// once and still tried at panic time (the volume may mount late).
pub fn install(path: Option<PathBuf>) {
    if let Some(path) = &path {
        match open_append(path) {
            Ok(_) => {
                tracing::info!(path = %path.display(), "panic records are appended to this file")
            }
            Err(error) => tracing::warn!(
                path = %path.display(),
                %error,
                "panic log file is not writable; panics go to the log and stderr only"
            ),
        }
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let reentered = IN_HOOK.with(|flag| flag.replace(true));
        if !reentered {
            let record = PanicRecord::capture(info);
            if let Some(path) = &path {
                // Best effort: there is nowhere left to report a failure.
                let _ = append(path, &record.to_text());
            }
            tracing::error!(
                thread = %record.thread,
                location = %record.location,
                backtrace = %record.backtrace,
                "panic: {}",
                record.message
            );
            IN_HOOK.with(|flag| flag.set(false));
        }
        previous(info);
    }));
}

/// The fields of one panic, each bounded.
struct PanicRecord {
    timestamp: String,
    thread: String,
    location: String,
    message: String,
    backtrace: String,
}

impl PanicRecord {
    fn capture(info: &PanicHookInfo<'_>) -> Self {
        let payload = info.payload();
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("Box<dyn Any>");
        let location = info.location().map_or_else(
            || "<unknown>".to_owned(),
            |location| {
                format!(
                    "{}:{}:{}",
                    truncated(location.file(), MAX_THREAD_NAME_BYTES * 4),
                    location.line(),
                    location.column()
                )
            },
        );
        let thread = std::thread::current();
        let thread = match thread.name() {
            Some(name) => truncated(name, MAX_THREAD_NAME_BYTES).to_owned(),
            None => format!("{:?}", thread.id()),
        };
        let mut backtrace = String::new();
        let _ = write!(
            BoundedWriter::new(&mut backtrace, MAX_BACKTRACE_BYTES),
            "{}",
            std::backtrace::Backtrace::force_capture()
        );
        Self {
            timestamp: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            thread,
            location,
            message: truncated(message, MAX_MESSAGE_BYTES).to_owned(),
            backtrace,
        }
    }

    fn to_text(&self) -> String {
        format!(
            "=== namidb-server panic at {} (pid {})\nthread: {}\nlocation: {}\nmessage: {}\nbacktrace:\n{}\n",
            self.timestamp,
            std::process::id(),
            self.thread,
            self.location,
            self.message,
            self.backtrace.trim_end()
        )
    }
}

/// `value` cut to at most `max` bytes on a character boundary.
fn truncated(value: &str, max: usize) -> &str {
    if value.len() <= max {
        return value;
    }
    let mut end = max;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

/// A `fmt::Write` that keeps the first `max` bytes and drops the rest, so a
/// deep backtrace is never held in full.
struct BoundedWriter<'a> {
    out: &'a mut String,
    max: usize,
}

impl<'a> BoundedWriter<'a> {
    fn new(out: &'a mut String, max: usize) -> Self {
        Self { out, max }
    }
}

impl std::fmt::Write for BoundedWriter<'_> {
    fn write_str(&mut self, piece: &str) -> std::fmt::Result {
        let room = self.max.saturating_sub(self.out.len());
        if room == 0 {
            // Stop formatting: the rest would be dropped anyway.
            return Err(std::fmt::Error);
        }
        let kept = truncated(piece, room);
        self.out.push_str(kept);
        if kept.len() < piece.len() {
            self.out.push_str("\n<backtrace truncated>");
            self.max = 0;
            return Err(std::fmt::Error);
        }
        Ok(())
    }
}

fn open_append(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o640);
    }
    options.open(path)
}

/// Append `text` with one write call (`O_APPEND`, so records of threads
/// panicking at once do not overwrite each other).
fn append(path: &Path, text: &str) -> std::io::Result<()> {
    if std::fs::metadata(path).is_ok_and(|meta| meta.len() > MAX_LOG_BYTES) {
        let mut rotated = path.as_os_str().to_owned();
        rotated.push(".1");
        let _ = std::fs::rename(path, rotated);
    }
    open_append(path)?.write_all(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_keeps_char_boundaries() {
        assert_eq!(truncated("abc", 8), "abc");
        assert_eq!(truncated("abcdef", 3), "abc");
        // 'é' is two bytes: cutting inside it backs off to before it.
        assert_eq!(truncated("aé", 2), "a");
    }

    #[test]
    fn bounded_writer_stops_at_its_limit() {
        let mut out = String::new();
        let long = "x".repeat(100);
        let _ = write!(BoundedWriter::new(&mut out, 10), "{long}{long}");
        assert!(out.starts_with("xxxxxxxxxx\n<backtrace truncated>"));
        assert!(out.len() < 40);
    }

    #[test]
    fn append_rotates_past_the_size_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("panic.log");
        std::fs::write(&path, vec![b'a'; MAX_LOG_BYTES as usize + 1]).unwrap();
        append(&path, "fresh\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "fresh\n");
        let rotated = dir.path().join("panic.log.1");
        assert_eq!(std::fs::metadata(rotated).unwrap().len(), MAX_LOG_BYTES + 1);
    }
}
