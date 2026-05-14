//! Opt-in ORT profiling for the Canary encoder + decoder sessions.
//!
//! When `CANARY_PROFILE_DIR` is set to a writable directory, the
//! encoder and decoder `SessionBuilder`s call
//! [`ort::session::builder::SessionBuilder::with_profiling`], and the
//! struct `Drop` impls call
//! [`ort::session::Session::end_profiling`] to flush the JSON trace
//! before the session vanishes. ORT itself names the file
//! `<prefix>_<timestamp>.json`, so multiple loads in one process
//! (en + es) don't clobber each other.
//!
//! Used by issue #58 / PR #64 to investigate why INT8 weights ended
//! up slower than FP32 on the CI runner — the per-op timing in the
//! trace tells us whether the runtime picked
//! `MatMulIntegerToFloat` / `QLinearMatMul` (hardware-accelerated
//! INT8) or fell back to dequantize-then-FP32.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::builder::SessionBuilder;
use ort::session::Session;

/// Environment variable: when set to a directory path, ORT
/// profiling is enabled on every Canary session loaded thereafter.
const ENV_VAR: &str = "CANARY_PROFILE_DIR";

/// Apply profiling to a `SessionBuilder` if `CANARY_PROFILE_DIR`
/// is set. Returns the (possibly modified) builder and a boolean
/// flag the caller stores on its struct so `Drop` knows whether
/// to flush.
///
/// `kind` is a short label (e.g. `"encoder"`, `"decoder"`) used
/// as the filename prefix inside the profile dir.
pub(crate) fn apply(
    builder: SessionBuilder,
    kind: &str,
) -> Result<(SessionBuilder, bool), ort::Error> {
    let Some(dir) = profile_dir() else {
        return Ok((builder, false));
    };
    let prefix: PathBuf = dir.join(format!("canary-{kind}"));
    let builder = builder.with_profiling(&prefix)?;
    Ok((builder, true))
}

/// Flush the profile trace for a session being dropped. Called
/// from `CanaryEncoder` / `CanaryDecoder` `Drop` impls. Failures
/// are logged to stderr rather than propagated because `Drop`
/// can't return errors and a half-written trace is still useful.
pub(crate) fn flush(session: &Mutex<Session>, kind: &str) {
    let mut guard = match session.lock() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("[canary][profile] {kind} session mutex poisoned, skipping flush");
            return;
        }
    };
    match guard.end_profiling() {
        Ok(path) => eprintln!("[canary][profile] {kind} trace flushed to {path}"),
        Err(e) => eprintln!("[canary][profile] {kind} end_profiling failed: {e}"),
    }
}

fn profile_dir() -> Option<PathBuf> {
    let raw = std::env::var(ENV_VAR).ok()?;
    if raw.is_empty() {
        return None;
    }
    Some(Path::new(&raw).to_path_buf())
}
