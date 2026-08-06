//! Per-stage cancellation propagation (spec §11.2).
//!
//! `PipelineError::Cancelled` names the stage that was in flight; these
//! tests assert each stage is reachable and reports itself correctly.
//!
//! Cancellation arrives from somewhere else — a hotkey release, a UI
//! button — while the session runs, so these drive it from a spawned
//! task rather than inline. `Session::finish` closes the audio stream
//! and awaits in one step, which is exactly the ordering a caller has.

mod common;

use common::{silence_chunk, SlowContextProvider, SlowRefiner};
use euhadra::prelude::*;
use std::time::Duration;

/// Cancelling while ASR is still awaiting more audio surfaces a recording-stage error.
#[tokio::test]
async fn cancel_during_recording() {
    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("ignored"))
        .refiner(MockRefiner::passthrough())
        .context(MockContextProvider::new())
        .emitter(MockEmitter::new())
        .build()
        .unwrap();

    let session = pipeline.session();
    session.audio.send(silence_chunk()).await.unwrap();

    // Cancel while the session is still collecting audio. `finish`
    // closes the stream, but the token is already tripped, so the
    // collection loop takes the cancellation branch.
    session.cancel.cancel();

    let err = session
        .finish()
        .await
        .expect_err("cancellation should surface an error");
    assert!(
        matches!(err, PipelineError::Cancelled { during } if during == "recording"),
        "expected recording-stage error, got: {err}"
    );
}

/// Cancelling while the ContextProvider future is in flight surfaces a
/// context-stage error.
#[tokio::test]
async fn cancel_during_context() {
    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("anything"))
        .refiner(MockRefiner::passthrough())
        .context(SlowContextProvider {
            delay: Duration::from_secs(5),
        })
        .emitter(MockEmitter::new())
        .build()
        .unwrap();

    let session = pipeline.session();
    session.audio.send(silence_chunk()).await.unwrap();

    // Let the session get as far as the context fetch, which the
    // SlowContextProvider parks for 5s, then cancel from outside.
    let cancel = session.cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
    });

    let err = session
        .finish()
        .await
        .expect_err("cancellation should surface an error");
    assert!(
        matches!(err, PipelineError::Cancelled { during } if during == "context"),
        "expected context-stage error, got: {err}"
    );
}

/// Cancelling while the LLM refiner future is in flight surfaces a
/// refinement-stage error.
#[tokio::test]
async fn cancel_during_refinement() {
    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("anything"))
        .refiner(SlowRefiner {
            delay: Duration::from_secs(5),
        })
        .context(MockContextProvider::new())
        .emitter(MockEmitter::new())
        .build()
        .unwrap();

    let session = pipeline.session();
    session.audio.send(silence_chunk()).await.unwrap();

    // Let the session get as far as the context fetch, which the
    // SlowContextProvider parks for 5s, then cancel from outside.
    let cancel = session.cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
    });

    let err = session
        .finish()
        .await
        .expect_err("cancellation should surface an error");
    assert!(
        matches!(err, PipelineError::Cancelled { during } if during == "refinement"),
        "expected refinement-stage error, got: {err}"
    );
}
