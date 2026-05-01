//! End-to-end pipeline integration tests against the public API.
//!
//! These tests exercise the same shape of pipeline construction shown in the
//! README so that doc rot is caught by CI.

mod common;

use common::send_one_and_close;
use euhadra::prelude::*;

#[tokio::test]
async fn english_full_pipeline_no_llm() {
    let emitter = MockEmitter::new();
    let outputs = emitter.outputs();

    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("um I want to go to Boston no wait to Denver"))
        .filter(SimpleFillerFilter::english())
        .processor(SelfCorrectionDetector::new())
        .processor(BasicPunctuationRestorer)
        .refiner(MockRefiner::passthrough())
        .context(MockContextProvider::new())
        .emitter(emitter)
        .build()
        .unwrap();

    let (audio_tx, _cancel, handle) = pipeline.session();
    send_one_and_close(audio_tx).await;

    let result = handle.await.unwrap().unwrap();
    assert_eq!(
        result.raw_text, "um I want to go to Boston no wait to Denver",
        "raw_text should preserve the original ASR output"
    );

    let buf = outputs.lock().await;
    assert_eq!(buf.len(), 1, "exactly one emission expected");
    let RefinementOutput::TextInsertion { text, .. } = &buf[0] else {
        panic!("expected TextInsertion variant");
    };
    assert!(!text.contains("um"), "filler not removed: {text}");
    assert!(!text.contains("Boston"), "reparandum not removed: {text}");
    assert!(text.contains("Denver"), "repair missing: {text}");
    assert!(
        text.starts_with(|c: char| c.is_uppercase()),
        "missing capitalization: {text}"
    );
    assert!(text.ends_with('.'), "missing terminal period: {text}");
}

#[tokio::test]
async fn japanese_full_pipeline_no_llm() {
    let emitter = MockEmitter::new();
    let outputs = emitter.outputs();

    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("えーと、今日は天気がいい"))
        .filter(JapaneseFillerFilter::new())
        .processor(SelfCorrectionDetector::new())
        .processor(BasicPunctuationRestorer)
        .refiner(MockRefiner::passthrough())
        .context(MockContextProvider::new())
        .emitter(emitter)
        .build()
        .unwrap();

    let (audio_tx, _cancel, handle) = pipeline.session();
    send_one_and_close(audio_tx).await;

    handle.await.unwrap().unwrap();

    let buf = outputs.lock().await;
    let RefinementOutput::TextInsertion { text, .. } = &buf[0] else {
        panic!("expected TextInsertion variant");
    };
    assert!(!text.contains("えーと"), "Japanese filler not removed: {text}");
    assert!(text.contains("今日は天気がいい"), "content lost: {text}");
}

#[tokio::test]
async fn spanish_full_pipeline_no_llm() {
    // Exercises the full Spanish post-ASR stack:
    //   "o sea voy a Madrid perdón a Barcelona"
    //     ─SpanishFillerFilter─→ "voy a Madrid perdón a Barcelona"
    //     ─SelfCorrectionDetector─→ "voy a Barcelona"
    //     ─BasicPunctuationRestorer─→ "Voy a Barcelona."
    //
    // The combination of a multi-word filler ("o sea") and a
    // perdón-cue self-correction ("Madrid → Barcelona") is what
    // PR #35 enabled but had no end-to-end coverage for. This test
    // catches regressions where filter / detector / punctuation
    // ordering breaks the chain (e.g. punctuation inserted before
    // detection runs).
    let emitter = MockEmitter::new();
    let outputs = emitter.outputs();

    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("o sea voy a Madrid perdón a Barcelona"))
        .filter(SpanishFillerFilter::new())
        .processor(SelfCorrectionDetector::new())
        .processor(BasicPunctuationRestorer)
        .refiner(MockRefiner::passthrough())
        .context(MockContextProvider::new())
        .emitter(emitter)
        .build()
        .unwrap();

    let (audio_tx, _cancel, handle) = pipeline.session();
    send_one_and_close(audio_tx).await;

    let result = handle.await.unwrap().unwrap();
    assert_eq!(
        result.raw_text, "o sea voy a Madrid perdón a Barcelona",
        "raw_text should preserve the original ASR output"
    );

    let buf = outputs.lock().await;
    assert_eq!(buf.len(), 1, "exactly one emission expected");
    let RefinementOutput::TextInsertion { text, .. } = &buf[0] else {
        panic!("expected TextInsertion variant");
    };
    assert!(!text.contains("o sea"), "Spanish filler not removed: {text}");
    assert!(!text.contains("Madrid"), "reparandum not removed: {text}");
    assert!(!text.contains("perdón"), "correction cue not removed: {text}");
    assert!(text.contains("Barcelona"), "repair missing: {text}");
    assert!(
        text.starts_with(|c: char| c.is_uppercase()),
        "missing capitalization: {text}"
    );
    assert!(text.ends_with('.'), "missing terminal period: {text}");
}

#[tokio::test]
async fn pipeline_emits_uppercase_via_mock_refiner() {
    let emitter = MockEmitter::new();
    let outputs = emitter.outputs();

    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("hello world"))
        .refiner(MockRefiner::uppercase())
        .context(MockContextProvider::new())
        .emitter(emitter)
        .build()
        .unwrap();

    let (audio_tx, _cancel, handle) = pipeline.session();
    send_one_and_close(audio_tx).await;

    handle.await.unwrap().unwrap();

    let buf = outputs.lock().await;
    let RefinementOutput::TextInsertion { text, .. } = &buf[0] else {
        panic!("expected TextInsertion variant");
    };
    assert_eq!(text, "HELLO WORLD");
}

#[tokio::test]
async fn missing_required_component_fails_build() {
    let result = Pipeline::builder()
        .asr(MockAsr::new("test"))
        .refiner(MockRefiner::passthrough())
        // missing context + emitter
        .build();
    assert!(
        result.is_err(),
        "builder must reject incomplete configurations"
    );
}
