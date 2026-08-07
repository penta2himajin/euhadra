//! Voice activity detection and utterance segmentation.
//!
//! euhadra hands whole recordings to the ASR adapter, so a 30-second
//! capture with 5 seconds of speech in it feeds 25 seconds of silence to
//! the model. Depending on the backend that produces hallucinated text
//! ("Thanks for watching", 「ご視聴ありがとうございました」). This module
//! is the segmentation half of the answer: find where speech is, and cut
//! the recording into utterances on the silences between.
//!
//! Two pieces, deliberately separated:
//!
//! - [`VadBackend`] answers "is this frame speech?" — `EarshotVad`
//!   (feature `vad`) with a small neural network, or [`EnergyVad`] with
//!   a level threshold and no dependency at all.
//! - [`Segmenter`] turns that per-frame answer into utterance boundaries.
//!   It holds the hysteresis, and it is the part that decides whether a
//!   pause is a breath or the end of a sentence.
//!
//! **Use `EarshotVad` unless the audio is not at 16 kHz.** [`EnergyVad`]
//! decides on loudness, which a keyboard and a slammed door also have;
//! `EarshotVad` costs one pure-Rust dependency with the network embedded
//! in it, so there is no runtime to link and no model to fetch.
//!
//! The split matters because the risk lives in the second piece.
//! Over-segmentation is destructive and unrecoverable — a sentence cut
//! in half is transcribed as two wrong sentences, and #134 measured what
//! short windows do to Parakeet (a 1–2s prefix produced the confident
//! hallucinations "Yeah." and 「あっ。」). Under-segmentation only costs
//! latency. So [`SegmenterConfig`] defaults are biased towards waiting,
//! and a better [`VadBackend`] does not by itself fix a badly tuned
//! [`Segmenter`].
//!
//! ```
//! use euhadra::vad::{EnergyVad, SegmenterConfig, segment_buffer};
//!
//! // 1s of silence, 1s of tone, 1s of silence, at 16 kHz.
//! let mut samples = vec![0.0f32; 16_000];
//! samples.extend((0..16_000).map(|i| (i as f32 * 0.1).sin() * 0.5));
//! samples.extend(vec![0.0f32; 16_000]);
//!
//! let segments = segment_buffer(
//!     &EnergyVad::new(),
//!     &samples,
//!     16_000,
//!     &SegmenterConfig::default(),
//! )?;
//! assert_eq!(segments.len(), 1);
//! # Ok::<(), euhadra::vad::VadError>(())
//! ```

use std::time::Duration;

mod energy;
mod segmenter;

pub use energy::EnergyVad;
pub use segmenter::{Segmenter, SegmenterConfig};

#[cfg(feature = "vad")]
mod earshot;

#[cfg(feature = "vad")]
pub use earshot::EarshotVad;

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// Scores audio frames for the presence of speech.
///
/// Implementations range from [`EnergyVad`] — no dependencies, no model —
/// to a neural detector such as Silero. The [`Segmenter`] consumes their
/// output identically, so swapping one for the other does not change how
/// utterance boundaries are decided.
pub trait VadBackend: Send + Sync {
    /// Samples per frame. Callers must hand [`VadStream::speech_probability`]
    /// exactly this many.
    fn frame_size(&self) -> usize;

    /// The sample rate the frames must be at, or `None` when the backend
    /// works at any rate. Neural detectors are trained at one rate and
    /// cannot be fed another; energy thresholding does not care.
    fn required_sample_rate(&self) -> Option<u32>;

    /// Begin a detection pass.
    ///
    /// A separate object rather than `&mut self` because recurrent
    /// detectors carry state across frames that must not leak from one
    /// utterance into the next, while the backend itself is shared
    /// (`Arc<dyn VadBackend>`) across sessions.
    fn start(&self) -> Box<dyn VadStream>;

    /// The score above which this backend means "speech".
    ///
    /// Calibration belongs to the backend, not to the segmentation
    /// policy: a probability is only comparable against a threshold that
    /// was chosen for the same scale. 0.5 is the default because it is
    /// what a calibrated detector produces, and `EnergyVad` is built to
    /// hit it exactly — but a backend is free to disagree, and
    /// `EarshotVad` does.
    ///
    /// Measured, not assumed. Getting this wrong is silent and
    /// expensive: `EarshotVad` scored at 0.5 lost whole utterances and
    /// cost +0.05 WER (en) / +0.13 CER (ja) against no detector at all,
    /// which read as "this backend is bad" rather than "this number is
    /// wrong". See `docs/benchmarks/vad_delta_wer.md`.
    fn default_threshold(&self) -> f32 {
        0.5
    }
}

/// One detection pass over consecutive frames of a single recording.
pub trait VadStream: Send {
    /// Probability in `0.0..=1.0` that `frame` contains speech.
    ///
    /// `frame` is [`VadBackend::frame_size`] samples long. Frames arrive
    /// in capture order; a stateful implementation may rely on that.
    fn speech_probability(&mut self, frame: &[f32]) -> f32;
}

// ---------------------------------------------------------------------------
// Segments
// ---------------------------------------------------------------------------

/// A stretch of audio the detector considered speech.
///
/// Bounds are sample indices into the buffer that was analysed, half-open
/// `[start, end)`, and already include [`SegmenterConfig::speech_pad`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SpeechSegment {
    pub start: usize,
    pub end: usize,
}

impl SpeechSegment {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// How long this segment runs at `sample_rate`.
    pub fn duration(&self, sample_rate: u32) -> Duration {
        if sample_rate == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(self.len() as f64 / sample_rate as f64)
    }

    /// Where this segment starts within the recording.
    pub fn offset(&self, sample_rate: u32) -> Duration {
        if sample_rate == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(self.start as f64 / sample_rate as f64)
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why voice activity detection could not run.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum VadError {
    /// The audio is at a rate the backend was not trained for. Not
    /// something to paper over by resampling silently: a neural detector
    /// fed the wrong rate returns plausible numbers that mean nothing.
    #[error("backend requires {required} Hz audio, got {actual} Hz")]
    SampleRate { required: u32, actual: u32 },

    /// The backend could not be initialised — a missing model file, an
    /// unreadable graph.
    #[error("failed to load VAD backend: {0}")]
    BackendLoad(String),
}

// ---------------------------------------------------------------------------
// Whole-buffer convenience
// ---------------------------------------------------------------------------

/// Find the speech segments in a complete recording.
///
/// The file-input counterpart to feeding a [`Segmenter`] frame by frame.
/// Returns segments in capture order; an empty vector means the detector
/// found no speech at all.
pub fn segment_buffer(
    backend: &dyn VadBackend,
    samples: &[f32],
    sample_rate: u32,
    config: &SegmenterConfig,
) -> Result<Vec<SpeechSegment>, VadError> {
    let mut segmenter = Segmenter::new(backend, sample_rate, config.clone())?;
    let frame_size = backend.frame_size();
    let mut stream = backend.start();

    let mut out = Vec::new();
    for frame in samples.chunks(frame_size) {
        // A trailing partial frame is zero-padded rather than dropped:
        // dropping it would silently shorten the last utterance, and a
        // backend that indexes by frame_size would panic on the short
        // slice.
        let probability = if frame.len() == frame_size {
            stream.speech_probability(frame)
        } else {
            let mut padded = frame.to_vec();
            padded.resize(frame_size, 0.0);
            stream.speech_probability(&padded)
        };
        if let Some(segment) = segmenter.push(probability) {
            out.push(segment);
        }
    }
    if let Some(segment) = segmenter.flush() {
        out.push(segment);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sr` seconds of alternating silence and tone, in `(silence, tone)`
    /// pairs given in seconds.
    fn synth(sample_rate: u32, spans: &[(f32, bool)]) -> Vec<f32> {
        let mut out = Vec::new();
        for (seconds, voiced) in spans {
            let n = (seconds * sample_rate as f32) as usize;
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

    #[test]
    fn silence_only_yields_no_segments() {
        let samples = synth(16_000, &[(2.0, false)]);
        let segments = segment_buffer(
            &EnergyVad::new(),
            &samples,
            16_000,
            &SegmenterConfig::default(),
        )
        .unwrap();
        assert!(
            segments.is_empty(),
            "silence must not produce an utterance, got {segments:?}"
        );
    }

    #[test]
    fn one_utterance_between_silences() {
        let samples = synth(16_000, &[(1.0, false), (1.5, true), (1.0, false)]);
        let segments = segment_buffer(
            &EnergyVad::new(),
            &samples,
            16_000,
            &SegmenterConfig::default(),
        )
        .unwrap();
        assert_eq!(segments.len(), 1, "got {segments:?}");

        // The segment should cover the tone. Padding widens it, so check
        // containment rather than equality.
        let seg = segments[0];
        assert!(
            seg.start <= 16_000 && seg.end >= 16_000 + 24_000,
            "segment {seg:?} does not cover the speech at samples 16000..40000"
        );
    }

    /// The property the defaults exist for: a pause shorter than
    /// `min_silence` is a breath, not an utterance boundary. Splitting
    /// here is the destructive failure #134 measured.
    #[test]
    fn short_pause_does_not_split_an_utterance() {
        let samples = synth(
            16_000,
            &[
                (1.0, false),
                (1.0, true),
                (0.2, false), // breath
                (1.0, true),
                (1.0, false),
            ],
        );
        let segments = segment_buffer(
            &EnergyVad::new(),
            &samples,
            16_000,
            &SegmenterConfig::default(),
        )
        .unwrap();
        assert_eq!(
            segments.len(),
            1,
            "a 200ms pause is shorter than the default min_silence and must \
             not cut the utterance, got {segments:?}"
        );
    }

    #[test]
    fn long_pause_splits_into_two_utterances() {
        let samples = synth(
            16_000,
            &[
                (0.5, false),
                (1.0, true),
                (1.5, false), // well past min_silence
                (1.0, true),
                (0.5, false),
            ],
        );
        let segments = segment_buffer(
            &EnergyVad::new(),
            &samples,
            16_000,
            &SegmenterConfig::default(),
        )
        .unwrap();
        assert_eq!(segments.len(), 2, "got {segments:?}");
        assert!(
            segments[0].end <= segments[1].start,
            "segments must not overlap: {segments:?}"
        );
    }

    #[test]
    fn speech_running_to_the_end_is_still_emitted() {
        let samples = synth(16_000, &[(0.5, false), (1.5, true)]);
        let segments = segment_buffer(
            &EnergyVad::new(),
            &samples,
            16_000,
            &SegmenterConfig::default(),
        )
        .unwrap();
        assert_eq!(
            segments.len(),
            1,
            "a segment still open at end of audio must be flushed, got {segments:?}"
        );
        assert!(segments[0].end >= samples.len() - 16_000 / 100);
    }

    #[test]
    fn segment_duration_and_offset() {
        let seg = SpeechSegment {
            start: 16_000,
            end: 32_000,
        };
        assert_eq!(seg.duration(16_000), Duration::from_secs(1));
        assert_eq!(seg.offset(16_000), Duration::from_secs(1));
        assert_eq!(seg.duration(0), Duration::ZERO);
    }
}
