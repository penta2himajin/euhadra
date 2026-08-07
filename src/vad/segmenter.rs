use std::time::Duration;

use super::{SpeechSegment, VadBackend, VadError};

/// How per-frame speech probabilities become utterance boundaries.
///
/// The defaults are deliberately asymmetric. Cutting an utterance that
/// was still going is destructive: the ASR sees a fragment, and #134
/// measured what a fragment does — a 3.0s prefix of an English utterance
/// produced the fluent, complete-looking, wrong "However, due to the slow
/// communication.", and 1–2s prefixes produced outright hallucinations.
/// Waiting too long only costs latency, and the whole-utterance pass
/// still runs at the end. So every knob here leans towards waiting.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SegmenterConfig {
    /// Probability at or above which a frame counts as speech.
    pub threshold: f32,

    /// How much continuous speech opens an utterance. Guards against a
    /// door slam or a keystroke starting one.
    pub min_speech: Duration,

    /// How much continuous silence closes one. **The hysteresis knob.**
    /// Raising it biases towards under-segmentation, which is the safe
    /// direction; Silero's own default of 100 ms is tuned for endpointing
    /// latency and cuts mid-sentence pauses, so this sits well above it.
    pub min_silence: Duration,

    /// Audio kept either side of the detected boundary. A detector fires
    /// slightly late on onset and slightly early on offset, and clipped
    /// word edges cost more accuracy than a little extra silence does.
    pub speech_pad: Duration,

    /// Force a cut after this much continuous speech, so a speaker who
    /// never pauses still gets incremental output. `None` waits for a
    /// real silence however long that takes.
    ///
    /// This is the one setting that deliberately over-segments, so it is
    /// set long enough that reaching it is unusual.
    pub max_speech: Option<Duration>,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech: Duration::from_millis(120),
            min_silence: Duration::from_millis(700),
            speech_pad: Duration::from_millis(200),
            max_speech: Some(Duration::from_secs(30)),
        }
    }
}

/// Turns a stream of per-frame speech probabilities into utterances.
///
/// Frame-driven and allocation-free: feed [`push`](Self::push) one
/// probability per frame in capture order, and it returns a
/// [`SpeechSegment`] on the frame where an utterance closes. Call
/// [`flush`](Self::flush) at end of audio for a segment still open.
///
/// It never looks at the audio itself, which is what lets the same
/// segmentation policy sit on top of any [`VadBackend`].
#[derive(Debug, Clone)]
pub struct Segmenter {
    threshold: f32,
    frame_size: usize,
    min_speech_frames: usize,
    min_silence_frames: usize,
    pad_samples: usize,
    max_speech_frames: Option<usize>,

    /// Frames pushed so far — the cursor into the recording.
    frame_index: usize,
    /// `Some(first speech frame)` once an utterance is open.
    open_at: Option<usize>,
    /// Consecutive speech frames seen while still below `min_speech`.
    speech_run: usize,
    /// Consecutive silence frames seen since the last speech frame.
    silence_run: usize,
    /// The last frame that scored as speech within the open utterance.
    last_speech: usize,
}

impl Segmenter {
    /// Configure a segmenter for `backend` running over audio at
    /// `sample_rate`.
    ///
    /// Fails when the backend declares a rate and the audio is at a
    /// different one — see [`VadError::SampleRate`].
    pub fn new(
        backend: &dyn VadBackend,
        sample_rate: u32,
        config: SegmenterConfig,
    ) -> Result<Self, VadError> {
        if let Some(required) = backend.required_sample_rate() {
            if required != sample_rate {
                return Err(VadError::SampleRate {
                    required,
                    actual: sample_rate,
                });
            }
        }
        let frame_size = backend.frame_size().max(1);
        let frames = |d: Duration| -> usize {
            let samples = d.as_secs_f64() * sample_rate as f64;
            (samples / frame_size as f64).ceil() as usize
        };
        Ok(Self {
            threshold: config.threshold,
            frame_size,
            // At least one frame of each, so a zero duration still means
            // "one frame of evidence" rather than "no evidence needed".
            min_speech_frames: frames(config.min_speech).max(1),
            min_silence_frames: frames(config.min_silence).max(1),
            pad_samples: (config.speech_pad.as_secs_f64() * sample_rate as f64) as usize,
            max_speech_frames: config.max_speech.map(|d| frames(d).max(1)),
            frame_index: 0,
            open_at: None,
            speech_run: 0,
            silence_run: 0,
            last_speech: 0,
        })
    }

    /// Feed one frame's speech probability.
    ///
    /// Returns a segment on the frame where an utterance closes, and
    /// `None` otherwise.
    pub fn push(&mut self, probability: f32) -> Option<SpeechSegment> {
        let is_speech = probability >= self.threshold;
        let index = self.frame_index;
        self.frame_index += 1;

        match self.open_at {
            // ── No utterance open ────────────────────────────────────
            None => {
                if is_speech {
                    self.speech_run += 1;
                    if self.speech_run >= self.min_speech_frames {
                        // Open at the first frame of the run, not the
                        // frame that crossed the threshold, or the
                        // beginning of every utterance is shaved off.
                        self.open_at = Some(index + 1 - self.speech_run);
                        self.last_speech = index;
                        self.silence_run = 0;
                    }
                } else {
                    self.speech_run = 0;
                }
                None
            }

            // ── Utterance open ───────────────────────────────────────
            Some(start) => {
                if is_speech {
                    self.last_speech = index;
                    self.silence_run = 0;
                } else {
                    self.silence_run += 1;
                    if self.silence_run >= self.min_silence_frames {
                        return Some(self.close(start, self.last_speech));
                    }
                }

                // The safety valve, checked after the silence rule so a
                // genuine boundary always wins over a forced one.
                if let Some(limit) = self.max_speech_frames {
                    if index + 1 - start >= limit {
                        return Some(self.close(start, index));
                    }
                }
                None
            }
        }
    }

    /// Close an utterance still open at end of audio.
    ///
    /// Without this, a recording that ends while the speaker is talking
    /// loses its final utterance entirely — the most common shape for
    /// hold-to-talk capture, where the key is released as the last word
    /// finishes.
    pub fn flush(&mut self) -> Option<SpeechSegment> {
        let start = self.open_at?;
        let end = self.last_speech;
        Some(self.close(start, end))
    }

    /// Whether an utterance is currently open.
    pub fn is_speaking(&self) -> bool {
        self.open_at.is_some()
    }

    /// Turn a frame range into a padded sample range and reset for the
    /// next utterance.
    fn close(&mut self, start_frame: usize, end_frame: usize) -> SpeechSegment {
        let start = (start_frame * self.frame_size).saturating_sub(self.pad_samples);
        let end = (end_frame + 1) * self.frame_size + self.pad_samples;

        self.open_at = None;
        self.speech_run = 0;
        self.silence_run = 0;
        SpeechSegment { start, end }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::EnergyVad;

    /// A segmenter over 16 kHz audio with the given config, framed the
    /// way `EnergyVad` frames (10 ms).
    fn segmenter(config: SegmenterConfig) -> Segmenter {
        Segmenter::new(&EnergyVad::new(), 16_000, config).unwrap()
    }

    fn no_padding() -> SegmenterConfig {
        SegmenterConfig {
            speech_pad: Duration::ZERO,
            ..SegmenterConfig::default()
        }
    }

    /// Drive `probs` through a segmenter and collect everything it emits.
    fn run(mut seg: Segmenter, probs: &[f32]) -> Vec<SpeechSegment> {
        let mut out: Vec<SpeechSegment> = probs.iter().filter_map(|p| seg.push(*p)).collect();
        out.extend(seg.flush());
        out
    }

    /// 10 ms frames at 16 kHz: 12 speech frames for min_speech (120 ms),
    /// 70 silence frames for min_silence (700 ms).
    fn frames(n: usize, p: f32) -> Vec<f32> {
        vec![p; n]
    }

    #[test]
    fn a_brief_transient_does_not_open_an_utterance() {
        // 5 frames = 50 ms, under the 120 ms min_speech.
        let mut probs = frames(20, 0.0);
        probs.extend(frames(5, 1.0));
        probs.extend(frames(100, 0.0));
        assert!(
            run(segmenter(no_padding()), &probs).is_empty(),
            "a 50ms transient is below min_speech and must not open an utterance"
        );
    }

    #[test]
    fn utterance_starts_at_the_first_speech_frame_not_the_confirming_one() {
        let mut probs = frames(10, 0.0);
        probs.extend(frames(30, 1.0));
        probs.extend(frames(100, 0.0));

        let segments = run(segmenter(no_padding()), &probs);
        assert_eq!(segments.len(), 1);
        // Frame 10 is the first speech frame; 160 samples per frame.
        assert_eq!(
            segments[0].start,
            10 * 160,
            "the utterance must start where speech started, not where \
             min_speech was satisfied"
        );
    }

    #[test]
    fn closing_excludes_the_trailing_silence() {
        let mut probs = frames(30, 1.0);
        probs.extend(frames(100, 0.0));

        let segments = run(segmenter(no_padding()), &probs);
        assert_eq!(segments.len(), 1);
        assert_eq!(
            segments[0].end,
            30 * 160,
            "the segment must end at the last speech frame, not after the \
             silence that confirmed the boundary"
        );
    }

    #[test]
    fn padding_widens_the_segment_on_both_sides() {
        let mut probs = frames(50, 0.0);
        probs.extend(frames(30, 1.0));
        probs.extend(frames(100, 0.0));

        let padded = run(segmenter(SegmenterConfig::default()), &probs);
        let bare = run(segmenter(no_padding()), &probs);
        assert_eq!(padded.len(), 1);
        assert!(
            padded[0].start < bare[0].start && padded[0].end > bare[0].end,
            "padded {:?} should be wider than unpadded {:?}",
            padded[0],
            bare[0]
        );
    }

    #[test]
    fn padding_cannot_push_the_start_below_zero() {
        // Speech from the very first frame, with 200 ms of padding to
        // subtract from an offset of zero.
        let mut probs = frames(30, 1.0);
        probs.extend(frames(100, 0.0));
        let segments = run(segmenter(SegmenterConfig::default()), &probs);
        assert_eq!(segments[0].start, 0);
    }

    #[test]
    fn max_speech_forces_a_cut_when_the_speaker_never_pauses() {
        let config = SegmenterConfig {
            max_speech: Some(Duration::from_secs(1)),
            speech_pad: Duration::ZERO,
            ..SegmenterConfig::default()
        };
        // 3 seconds of unbroken speech at 10 ms per frame.
        let segments = run(segmenter(config), &frames(300, 1.0));
        assert_eq!(
            segments.len(),
            3,
            "1s cap over 3s of continuous speech should force 3 segments, \
             got {segments:?}"
        );
    }

    #[test]
    fn no_max_speech_waits_indefinitely() {
        let config = SegmenterConfig {
            max_speech: None,
            speech_pad: Duration::ZERO,
            ..SegmenterConfig::default()
        };
        let segments = run(segmenter(config), &frames(6000, 1.0));
        assert_eq!(
            segments.len(),
            1,
            "with no cap, 60s of continuous speech is one utterance"
        );
    }

    /// A real boundary and the safety valve can fall on the same frame.
    /// The boundary must win: it is where the speaker actually stopped.
    #[test]
    fn a_real_boundary_wins_over_the_forced_one() {
        let config = SegmenterConfig {
            max_speech: Some(Duration::from_millis(500)),
            min_silence: Duration::from_millis(100),
            speech_pad: Duration::ZERO,
            ..SegmenterConfig::default()
        };
        let mut probs = frames(40, 1.0); // 400 ms
        probs.extend(frames(10, 0.0)); // 100 ms silence closes it at 500 ms
        probs.extend(frames(200, 0.0));

        let segments = run(segmenter(config), &probs);
        assert_eq!(segments.len(), 1, "got {segments:?}");
        assert_eq!(
            segments[0].end,
            40 * 160,
            "the segment should end where speech ended, not at the cap"
        );
    }

    #[test]
    fn flush_emits_an_utterance_still_open_at_end_of_audio() {
        let mut seg = segmenter(no_padding());
        for p in frames(30, 1.0) {
            assert!(seg.push(p).is_none());
        }
        assert!(seg.is_speaking());
        let flushed = seg.flush().expect("open utterance must be flushed");
        assert_eq!(flushed.end, 30 * 160);
        assert!(!seg.is_speaking());
        assert!(seg.flush().is_none(), "flushing twice must not duplicate");
    }

    #[test]
    fn flush_on_silence_emits_nothing() {
        let mut seg = segmenter(no_padding());
        for p in frames(50, 0.0) {
            assert!(seg.push(p).is_none());
        }
        assert!(seg.flush().is_none());
    }

    /// The defaults must not split on a pause a speaker takes
    /// mid-sentence. This is the asymmetry the module exists to encode,
    /// asserted on the config rather than through a backend so that
    /// changing a default breaks this test loudly.
    #[test]
    fn defaults_tolerate_a_half_second_mid_sentence_pause() {
        let mut probs = frames(30, 1.0);
        probs.extend(frames(50, 0.0)); // 500 ms — a breath
        probs.extend(frames(30, 1.0));
        probs.extend(frames(100, 0.0));

        let segments = run(segmenter(no_padding()), &probs);
        assert_eq!(
            segments.len(),
            1,
            "default min_silence must be longer than a mid-sentence pause; \
             got {segments:?}"
        );
    }

    #[test]
    fn sample_rate_mismatch_is_refused() {
        struct Fixed;
        impl VadBackend for Fixed {
            fn frame_size(&self) -> usize {
                512
            }
            fn required_sample_rate(&self) -> Option<u32> {
                Some(16_000)
            }
            fn start(&self) -> Box<dyn crate::vad::VadStream> {
                unreachable!("construction fails before a stream is needed")
            }
        }
        let err = Segmenter::new(&Fixed, 44_100, SegmenterConfig::default()).unwrap_err();
        assert!(
            matches!(
                err,
                VadError::SampleRate {
                    required: 16_000,
                    actual: 44_100
                }
            ),
            "got {err:?}"
        );
    }
}
