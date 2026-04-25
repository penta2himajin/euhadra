//! Graceful-degradation behaviour (spec §11.2).

mod common;

use common::{send_one_and_close, SilentAsr};
use euhadra::prelude::*;

/// When the LLM refiner errors out, the pipeline must still emit something —
/// specifically the raw ASR text — instead of failing the whole session.
#[tokio::test]
async fn llm_failure_falls_back_to_raw_asr_text() {
    let emitter = MockEmitter::new();
    let outputs = emitter.outputs();

    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("raw dictation text"))
        .filter(SimpleFillerFilter::english())
        .processor(BasicPunctuationRestorer)
        .refiner(MockRefiner::failing("upstream timeout"))
        .context(MockContextProvider::new())
        .emitter(emitter)
        .build()
        .unwrap();

    let (audio_tx, _cancel, handle) = pipeline.session();
    send_one_and_close(audio_tx).await;

    let result = handle
        .await
        .expect("task must not panic")
        .expect("pipeline must succeed despite LLM failure");
    assert!(result.emit_result.success, "fallback emission must succeed");

    let buf = outputs.lock().await;
    assert_eq!(buf.len(), 1);
    let RefinementOutput::TextInsertion { text, .. } = &buf[0] else {
        panic!("expected TextInsertion fallback");
    };
    // Pipeline contract: on refiner failure the *raw* ASR text is emitted,
    // bypassing filters and processors — we should see "um"-free input
    // would still be the literal raw ASR string here.
    assert_eq!(
        text, "raw dictation text",
        "LLM-failure fallback must emit the raw ASR text verbatim"
    );
}

/// When the ASR adapter never produces a final result, the pipeline must
/// surface an explicit "no speech detected" error rather than hanging.
#[tokio::test]
async fn silent_asr_produces_no_speech_error() {
    let pipeline = Pipeline::builder()
        .asr(SilentAsr)
        .refiner(MockRefiner::passthrough())
        .context(MockContextProvider::new())
        .emitter(MockEmitter::new())
        .build()
        .unwrap();

    let (audio_tx, _cancel, handle) = pipeline.session();
    send_one_and_close(audio_tx).await;

    let err = handle
        .await
        .expect("task must not panic")
        .expect_err("empty ASR output must surface an error");
    assert!(
        err.message.contains("no speech detected"),
        "expected no-speech error, got: {err}"
    );
}
