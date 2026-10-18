//! An adapter that declares a maximum input length must never be handed
//! more than that (#166). FluidAudio's Japanese model, for one, drops text
//! at its own chunk seams past 15 s, so the pipeline — which is the only
//! thing that knows where the utterances are — has to keep each call short.

use std::time::Duration;

use euhadra::mock::AsrCall;
use euhadra::prelude::*;
use euhadra::vad::EnergyVad;

const RATE: u32 = 16_000;
const MAX: Duration = Duration::from_secs(3);

fn synth(spans: &[(f32, bool)]) -> Vec<f32> {
    let mut out = Vec::new();
    for (seconds, voiced) in spans {
        let n = (seconds * RATE as f32) as usize;
        for i in 0..n {
            out.push(if *voiced {
                (i as f32 * 0.3).sin() * 0.4
            } else {
                0.0
            });
        }
    }
    out
}

fn chunks(samples: &[f32]) -> Vec<AudioChunk> {
    samples
        .chunks(RATE as usize / 10)
        .map(|c| AudioChunk {
            samples: c.to_vec(),
            sample_rate: RATE,
            channels: 1,
        })
        .collect()
}

/// Six 1.5 s utterances with 1 s pauses: 9 s of speech in all.
fn six_utterances() -> Vec<f32> {
    let mut spans = vec![(0.5, false)];
    for _ in 0..6 {
        spans.push((1.5, true));
        spans.push((1.0, false));
    }
    synth(&spans)
}

fn assert_all_fit(calls: &[AsrCall]) {
    let max = (MAX.as_secs_f64() * RATE as f64) as usize;
    assert!(!calls.is_empty(), "the adapter was never called");
    for (i, call) in calls.iter().enumerate() {
        assert!(
            call.samples <= max,
            "call {i} handed the adapter {} samples, over the {max}-sample limit; \
             all calls: {:?}",
            call.samples,
            calls.iter().map(|c| c.samples).collect::<Vec<_>>()
        );
    }
}

#[tokio::test]
async fn batch_speech_only_never_exceeds_the_limit() {
    let samples = six_utterances();
    let asr = RecordingAsr::new(["a"]).with_max_input(MAX);
    let calls = asr.calls();
    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    pipeline.transcribe(&chunks(&samples)).await.unwrap();

    let calls = calls.lock().unwrap();
    assert_all_fit(&calls);
    assert!(calls.len() > 1, "9 s of speech cannot fit one 3 s call");
}

#[tokio::test]
async fn session_speech_only_never_exceeds_the_limit() {
    let samples = six_utterances();
    let asr = RecordingAsr::new(["a"]).with_max_input(MAX);
    let calls = asr.calls();
    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    let session = pipeline.session();
    for chunk in chunks(&samples) {
        session.audio.send(chunk).await.unwrap();
    }
    session.finish().await.unwrap();

    assert_all_fit(&calls.lock().unwrap());
}

/// One utterance longer than the limit, with no pause to split at: the
/// segmenter's forced cut must leave room for its own padding.
#[tokio::test]
async fn one_long_utterance_is_cut_to_fit() {
    let samples = synth(&[(0.5, false), (8.0, true), (0.5, false)]);
    let asr = RecordingAsr::new(["a"]).with_max_input(MAX);
    let calls = asr.calls();
    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    pipeline.transcribe(&chunks(&samples)).await.unwrap();

    assert_all_fit(&calls.lock().unwrap());
}

/// The same through a live session, where each closed utterance is also
/// transcribed on its own for partials. Those calls bypass packing, so
/// only the tightened forced cut keeps them within the limit.
#[tokio::test]
async fn one_long_utterance_fits_in_session_partials_too() {
    let samples = synth(&[(0.5, false), (8.0, true), (0.5, false)]);
    let asr = RecordingAsr::new(["a"]).with_max_input(MAX);
    let calls = asr.calls();
    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    let session = pipeline.session();
    for chunk in chunks(&samples) {
        session.audio.send(chunk).await.unwrap();
    }
    session.finish().await.unwrap();

    let calls = calls.lock().unwrap();
    assert_all_fit(&calls);
    assert!(
        calls.len() > 2,
        "partials plus a final pass expected, got {}",
        calls.len()
    );
}

/// Without a detector there is nowhere safe to cut, and a silent cut
/// mid-word is the failure the limit exists to prevent.
#[tokio::test]
async fn over_the_limit_without_a_vad_is_an_error() {
    let samples = synth(&[(5.0, true)]);
    let asr = RecordingAsr::new(["a"]).with_max_input(MAX);
    let calls = asr.calls();
    let pipeline = Pipeline::builder().asr(asr).build().unwrap();

    let err = pipeline.transcribe(&chunks(&samples)).await.unwrap_err();
    assert!(
        matches!(err, PipelineError::InputTooLong { .. }),
        "got {err:?}"
    );
    assert!(
        calls.lock().unwrap().is_empty(),
        "the adapter must not be called"
    );
}

#[tokio::test]
async fn under_the_limit_without_a_vad_is_unchanged() {
    let samples = synth(&[(2.0, true)]);
    let asr = RecordingAsr::new(["a"]).with_max_input(MAX);
    let pipeline = Pipeline::builder().asr(asr).build().unwrap();
    assert_eq!(
        pipeline.transcribe(&chunks(&samples)).await.unwrap().text(),
        "a"
    );
}

/// Pieces are joined the way the language writes: Japanese has no word
/// spaces, so a space between two packs would be a visible defect that
/// `cer_lenient` cannot see.
#[tokio::test]
async fn packs_join_without_a_space_between_japanese_text() {
    let samples = six_utterances();
    let asr = RecordingAsr::new(["今日は", "晴れです", "散歩に", "行きます"]).with_max_input(MAX);
    let calls = asr.calls();
    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    let result = pipeline.transcribe(&chunks(&samples)).await.unwrap();
    let n = calls.lock().unwrap().len();
    let expected: String = ["今日は", "晴れです", "散歩に", "行きます"]
        .iter()
        .chain(std::iter::repeat(&"行きます"))
        .take(n)
        .copied()
        .collect();
    assert_eq!(result.text(), expected);
}

#[tokio::test]
async fn packs_join_with_a_space_between_english_text() {
    let samples = six_utterances();
    let asr = RecordingAsr::new(["one", "two", "three", "four"]).with_max_input(MAX);
    let calls = asr.calls();
    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    let result = pipeline.transcribe(&chunks(&samples)).await.unwrap();
    let n = calls.lock().unwrap().len();
    let expected = ["one", "two", "three", "four"]
        .iter()
        .chain(std::iter::repeat(&"four"))
        .take(n)
        .copied()
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(result.text(), expected);
}

/// The default — no declared limit — keeps today's single final pass.
#[tokio::test]
async fn no_declared_limit_keeps_one_final_pass() {
    let samples = six_utterances();
    let asr = RecordingAsr::new(["a"]);
    let calls = asr.calls();
    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    pipeline.transcribe(&chunks(&samples)).await.unwrap();
    assert_eq!(calls.lock().unwrap().len(), 1);
}
