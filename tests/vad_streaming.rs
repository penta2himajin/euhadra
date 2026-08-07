//! Voice activity detection wired into the pipeline: what reaches the
//! ASR adapter, and what reaches the caller before the session ends.
//!
//! The unit tests in `src/vad/` decide where the boundaries fall. These
//! decide what the pipeline does with them, which is a separate question
//! — a correct segmenter wired to the wrong final pass still hands the
//! model the silence it was meant to remove.
//!
//! [`EnergyVad`] rather than `EarshotVad` throughout: these are wiring
//! tests, and the level detector is deterministic on synthetic audio and
//! present in the default build.

use euhadra::prelude::*;
use euhadra::vad::{EnergyVad, SegmenterConfig, VadBackend, VadStream};

const RATE: u32 = 16_000;

/// Build a recording from `(seconds, voiced)` spans at 16 kHz.
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

/// Cut a buffer into 100 ms chunks, the way a capture device delivers it.
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

/// Feed a whole recording through a live session and wait for the result.
async fn drive(session: Session, audio: &[AudioChunk]) -> Result<SessionResult, PipelineError> {
    for chunk in audio {
        session.audio.send(chunk.clone()).await.unwrap();
    }
    session.finish().await
}

// ---------------------------------------------------------------------------
// What the ASR adapter is handed
// ---------------------------------------------------------------------------

/// The point of the whole exercise: silence must not reach the model.
#[tokio::test]
async fn speech_only_keeps_the_silence_away_from_the_asr() {
    let samples = synth(&[(2.0, false), (1.0, true), (2.0, false)]);
    let audio = chunks(&samples);

    let asr = RecordingAsr::new(["hello"]);
    let calls = asr.calls();

    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    let result = drive(pipeline.session(), &audio).await.unwrap();
    assert_eq!(result.text(), "hello");

    let calls = calls.lock().unwrap();
    let final_pass = calls.last().expect("the ASR must have been called");
    assert!(
        final_pass.samples < samples.len() / 2,
        "5 s captured with 1 s of speech in it, but the adapter was handed \
         {} of {} samples",
        final_pass.samples,
        samples.len()
    );
    assert!(
        final_pass.samples >= RATE as usize,
        "the speech itself must survive; got {} samples",
        final_pass.samples
    );
    assert_eq!(final_pass.sample_rate, RATE);
    assert_eq!(
        result.diagnostics.speech_segments.len(),
        1,
        "one utterance expected, got {:?}",
        result.diagnostics.speech_segments
    );
}

/// The conservative policy: detection runs, partials flow, and the final
/// text is byte-for-byte what a pipeline with no VAD would have produced.
#[tokio::test]
async fn whole_utterance_hands_the_asr_the_entire_recording() {
    let samples = synth(&[(2.0, false), (1.0, true), (2.0, false)]);
    let audio = chunks(&samples);

    let asr = RecordingAsr::new(["hello"]);
    let calls = asr.calls();

    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(EnergyVad::new())
        .final_pass(FinalPass::WholeUtterance)
        .build()
        .unwrap();

    drive(pipeline.session(), &audio).await.unwrap();

    let calls = calls.lock().unwrap();
    assert_eq!(
        calls.last().unwrap().samples,
        samples.len(),
        "WholeUtterance must not alter what the adapter sees"
    );
}

/// Without a VAD nothing about the existing path changes, including the
/// number of passes over the audio.
#[tokio::test]
async fn no_vad_means_one_pass_over_the_captured_audio() {
    let samples = synth(&[(2.0, false), (1.0, true), (2.0, false)]);
    let audio = chunks(&samples);

    let asr = RecordingAsr::new(["hello"]);
    let calls = asr.calls();

    let pipeline = Pipeline::builder().asr(asr).build().unwrap();
    let result = drive(pipeline.session(), &audio).await.unwrap();

    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1, "expected exactly one ASR pass");
    assert_eq!(calls[0].samples, samples.len());
    assert!(result.diagnostics.speech_segments.is_empty());
}

// ---------------------------------------------------------------------------
// Incremental output
// ---------------------------------------------------------------------------

#[tokio::test]
async fn one_partial_arrives_per_utterance_in_capture_order() {
    let samples = synth(&[
        (0.5, false),
        (1.0, true),
        (1.5, false), // past min_silence — a real boundary
        (1.0, true),
        (1.0, false),
    ]);
    let audio = chunks(&samples);

    let pipeline = Pipeline::builder()
        .asr(RecordingAsr::new(["first", "second", "whole"]))
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    let mut session = pipeline.session();
    let mut partials = std::mem::replace(&mut session.partials, tokio::sync::mpsc::channel(1).1);
    let collector = tokio::spawn(async move {
        let mut seen = Vec::new();
        while let Some(partial) = partials.recv().await {
            seen.push(partial);
        }
        seen
    });

    let result = drive(session, &audio).await.unwrap();
    let seen: Vec<Partial> = collector.await.unwrap();

    assert_eq!(seen.len(), 2, "expected two utterances, got {seen:?}");
    assert_eq!(seen[0].index, 0);
    assert_eq!(seen[1].index, 1);
    assert_eq!(seen[0].text, "first");
    assert_eq!(seen[1].text, "second");
    assert!(
        seen[0].end <= seen[1].start,
        "utterances must not overlap in time: {seen:?}"
    );
    assert_eq!(result.diagnostics.speech_segments.len(), 2);
}

/// Under `JoinSegments` the partials *are* the transcript, so the final
/// text has to be exactly their concatenation and there must be no extra
/// pass over the audio.
#[tokio::test]
async fn join_segments_returns_the_partials_and_nothing_more() {
    let samples = synth(&[
        (0.5, false),
        (1.0, true),
        (1.5, false),
        (1.0, true),
        (1.0, false),
    ]);
    let audio = chunks(&samples);

    let asr = RecordingAsr::new(["first", "second", "SHOULD NOT BE CALLED"]);
    let calls = asr.calls();

    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(EnergyVad::new())
        .final_pass(FinalPass::JoinSegments)
        .build()
        .unwrap();

    let result = drive(pipeline.session(), &audio).await.unwrap();
    assert_eq!(result.raw_text, "first second");
    assert_eq!(
        calls.lock().unwrap().len(),
        2,
        "JoinSegments must not re-transcribe the recording"
    );
}

/// Dropping the receiver is how a caller says it does not want partials,
/// and it must save the per-utterance pass rather than merely discard it.
#[tokio::test]
async fn dropping_the_partial_receiver_skips_per_utterance_transcription() {
    let samples = synth(&[(0.5, false), (1.0, true), (1.5, false), (1.0, true)]);
    let audio = chunks(&samples);

    let asr = RecordingAsr::new(["whole"]);
    let calls = asr.calls();

    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    let mut session = pipeline.session();
    drop(std::mem::replace(
        &mut session.partials,
        tokio::sync::mpsc::channel(1).1,
    ));

    let result = drive(session, &audio).await.unwrap();
    assert_eq!(result.text(), "whole");
    assert_eq!(
        calls.lock().unwrap().len(),
        1,
        "with nobody listening, only the final pass should run"
    );
}

// ---------------------------------------------------------------------------
// Degradation and edges
// ---------------------------------------------------------------------------

/// A recording with nothing in it is `NoSpeech`, not an empty transcript
/// and not a hallucinated one.
#[tokio::test]
async fn silence_only_is_reported_as_no_speech() {
    let audio = chunks(&synth(&[(3.0, false)]));

    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("Thanks for watching!"))
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    let err = drive(pipeline.session(), &audio).await.unwrap_err();
    assert!(
        matches!(err, PipelineError::NoSpeech),
        "expected NoSpeech, got {err}"
    );
}

/// A detector that cannot run must not take the session down with it.
/// The recording is transcribed as it was captured, and the failure is
/// reported the same way a filter's is.
#[tokio::test]
async fn a_rate_mismatch_degrades_to_the_unsegmented_path() {
    struct SixteenKOnly;
    impl VadBackend for SixteenKOnly {
        fn frame_size(&self) -> usize {
            256
        }
        fn required_sample_rate(&self) -> Option<u32> {
            Some(8_000)
        }
        fn start(&self) -> Box<dyn VadStream> {
            unreachable!("construction fails first")
        }
    }

    let samples = synth(&[(0.5, false), (1.0, true)]);
    let audio = chunks(&samples);

    let asr = RecordingAsr::new(["hello"]);
    let calls = asr.calls();

    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(SixteenKOnly)
        .build()
        .unwrap();

    let result = drive(pipeline.session(), &audio).await.unwrap();
    assert_eq!(result.text(), "hello");
    assert_eq!(calls.lock().unwrap()[0].samples, samples.len());

    let failure = result
        .diagnostics
        .failures
        .iter()
        .find(|f| f.stage == Stage::Vad)
        .expect("the disabled detector must be reported, not only logged");
    assert!(
        failure.reason.contains("8000"),
        "the reason should name the mismatch, got {:?}",
        failure.reason
    );
}

#[tokio::test]
async fn cancellation_beats_a_configured_detector() {
    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("will be cancelled"))
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    let session = pipeline.session();
    for chunk in chunks(&synth(&[(0.5, true)])) {
        session.audio.send(chunk).await.unwrap();
    }
    session.cancel.cancel();

    let err = session.finish().await.unwrap_err();
    assert!(
        matches!(err, PipelineError::Cancelled { .. }),
        "expected a cancellation, got {err}"
    );
}

// ---------------------------------------------------------------------------
// The file-input path
// ---------------------------------------------------------------------------

/// Detection sits ahead of the ASR adapter rather than inside microphone
/// capture, so a WAV gets the same treatment as a live recording. Burying
/// it in the `mic` feature would have left file input in the hole.
#[tokio::test]
async fn transcribe_segments_file_input_too() {
    let samples = synth(&[(2.0, false), (1.0, true), (2.0, false)]);
    let audio = chunks(&samples);

    let asr = RecordingAsr::new(["hello"]);
    let calls = asr.calls();

    let pipeline = Pipeline::builder()
        .asr(asr)
        .vad(EnergyVad::new())
        .build()
        .unwrap();

    let result = pipeline.transcribe(&audio).await.unwrap();
    assert_eq!(result.text(), "hello");
    assert_eq!(result.diagnostics.speech_segments.len(), 1);

    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1, "the batch path needs only the final pass");
    assert!(
        calls[0].samples < samples.len() / 2,
        "the silence should not have reached the adapter; got {} of {}",
        calls[0].samples,
        samples.len()
    );
}

/// A pause the speaker takes mid-sentence must not become a boundary.
/// This is the destructive direction: #134 measured Parakeet answering a
/// 3-second fragment with a fluent, complete-looking, wrong sentence.
#[tokio::test]
async fn a_mid_sentence_pause_does_not_become_two_utterances() {
    let samples = synth(&[
        (0.5, false),
        (1.0, true),
        (0.3, false), // a breath
        (1.0, true),
        (1.0, false),
    ]);

    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("one sentence"))
        .vad(EnergyVad::new())
        .segmenter_config(SegmenterConfig::default())
        .build()
        .unwrap();

    let result = pipeline.transcribe(&chunks(&samples)).await.unwrap();
    assert_eq!(
        result.diagnostics.speech_segments.len(),
        1,
        "a 300 ms pause is shorter than the default min_silence, got {:?}",
        result.diagnostics.speech_segments
    );
}
