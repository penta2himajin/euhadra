//! Backpressure / channel-sizing integration tests (spec §11.2).
//!
//! The pipeline uses bounded channels between stages; these tests confirm
//! that small channel sizes don't deadlock when the producer sends more
//! chunks than the channel can hold.

mod common;

use common::silence_chunk;
use euhadra::prelude::*;

/// With an audio channel of size 1, sending many chunks must serialise
/// cleanly through ASR rather than deadlock.
#[tokio::test]
async fn small_audio_channel_does_not_deadlock() {
    let emitter = MockEmitter::new();
    let outputs = emitter.outputs();

    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("ok"))
        .refiner(MockRefiner::passthrough())
        .context(MockContextProvider::new())
        .emitter(emitter)
        .audio_channel_size(1)
        .build()
        .unwrap();

    let session = pipeline.session();

    for _ in 0..32 {
        session.audio.send(silence_chunk()).await.unwrap();
    }
    let result = session.finish().await.unwrap();
    assert_eq!(result.raw_text, "ok");

    let buf = outputs.lock().await;
    assert_eq!(buf.len(), 1, "exactly one emission expected");
}

/// A pathologically small ASR-result channel must still complete a session.
/// MockAsr only sends a single final result, so this primarily exercises
/// the channel-construction path with size 1.
#[tokio::test]
async fn small_asr_channel_completes_session() {
    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("hello"))
        .refiner(MockRefiner::passthrough())
        .context(MockContextProvider::new())
        .emitter(MockEmitter::new())
        .asr_channel_size(1)
        .build()
        .unwrap();

    let session = pipeline.session();
    session.audio.send(silence_chunk()).await.unwrap();
    let result = session.finish().await.unwrap();
    assert_eq!(result.raw_text, "hello");
}
