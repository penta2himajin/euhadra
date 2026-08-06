//! Graceful-degradation behaviour (spec §11.2).

mod common;

use common::{send_one, SilentAsr};
use euhadra::prelude::*;

/// When the LLM refiner errors out, the pipeline must still emit
/// something rather than failing the whole session — and what it emits is
/// the Tier 1+2 text, not the raw ASR string.
///
/// This changed deliberately. The pipeline used to fall back to the raw
/// transcript, discarding filler removal and punctuation that had already
/// succeeded. That contradicted the premise the crate is built on: Tier
/// 1+2 output stands on its own, and the refiner is the optional polish.
/// Throwing away working stages because an optional one failed made the
/// degraded path worse than not configuring a refiner at all.
#[tokio::test]
async fn llm_failure_falls_back_to_processed_text() {
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

    let session = pipeline.session();
    send_one(&session.audio).await;

    let result = session
        .finish()
        .await
        .expect("pipeline must succeed despite LLM failure");
    assert!(result.emit_result.as_ref().unwrap().success, "fallback emission must succeed");

    let buf = outputs.lock().await;
    assert_eq!(buf.len(), 1);
    let RefinementOutput::TextInsertion { text, .. } = &buf[0] else {
        panic!("expected TextInsertion fallback");
    };
    assert_eq!(
        text, "Raw dictation text.",
        "the fallback must keep the Tier 1+2 work, not revert to raw ASR"
    );
    assert_ne!(
        text, &result.raw_text,
        "falling back to the raw transcript would discard successful stages"
    );

    // The refiner's failure is not silent: it is on the record even
    // though the session succeeded.
    assert!(
        result
            .diagnostics
            .failures
            .iter()
            .any(|f| f.stage == Stage::Refiner),
        "the refiner failure should be reported, got {:?}",
        result.diagnostics.failures
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

    let session = pipeline.session();
    send_one(&session.audio).await;

    let err = session
        .finish()
        .await
        .expect_err("empty ASR output must surface an error");
    assert!(
        matches!(err, PipelineError::NoSpeech),
        "expected NoSpeech, got: {err}"
    );
}
