//! Per-stage cancellation propagation (spec §11.2).
//!
//! The pipeline reports the stage at which cancellation landed via the error
//! message; these tests assert each stage is reachable and surfaces the
//! expected diagnostic.

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

    let (audio_tx, cancel, handle) = pipeline.session();

    // Send one chunk but DO NOT close the channel — MockAsr will keep
    // awaiting more audio, parking the pipeline in the recording stage.
    audio_tx.send(silence_chunk()).await.unwrap();
    cancel.cancel();

    let err = handle
        .await
        .expect("task must not panic")
        .expect_err("cancellation should surface an error");
    assert!(
        err.message.contains("cancelled during recording"),
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

    let (audio_tx, cancel, handle) = pipeline.session();
    audio_tx.send(silence_chunk()).await.unwrap();
    drop(audio_tx); // let ASR finish so we advance into context fetch

    // Give the runtime a moment to enter the context-fetch select!.
    tokio::time::sleep(Duration::from_millis(50)).await;
    cancel.cancel();

    let err = handle
        .await
        .expect("task must not panic")
        .expect_err("cancellation should surface an error");
    assert!(
        err.message.contains("cancelled during context"),
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

    let (audio_tx, cancel, handle) = pipeline.session();
    audio_tx.send(silence_chunk()).await.unwrap();
    drop(audio_tx);

    // Recording + context are instantaneous; this delay parks us inside
    // refinement before the cancel fires.
    tokio::time::sleep(Duration::from_millis(50)).await;
    cancel.cancel();

    let err = handle
        .await
        .expect("task must not panic")
        .expect_err("cancellation should surface an error");
    assert!(
        err.message.contains("cancelled during refinement"),
        "expected refinement-stage error, got: {err}"
    );
}
