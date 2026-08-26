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
    /// Score at or above which a frame counts as speech.
    ///
    /// `None` — the default — takes
    /// [`VadBackend::default_threshold`], which is where the
    /// backend's own calibration lives. Set it only to override that,
    /// and expect the right value to differ per backend rather than
    /// being a property of the audio.
    pub threshold: Option<f32>,

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
    ///
    /// On the **left** (onset) side this is combined with [`Self::preroll`]:
    /// the rewind is `max(speech_pad, preroll)`, then clamped by the
    /// previous segment's end so neighbouring utterances do not overlap.
    /// On the **right** (offset) side only `speech_pad` applies.
    pub speech_pad: Duration,

    /// How far to rewind before the detected onset when stream history
    /// allows it.
    ///
    /// Soft onsets and leading consonants often score below threshold for
    /// tens to hundreds of milliseconds; `speech_pad` alone either under-
    /// covers them or, when raised, walks back into the previous utterance.
    /// `preroll` is the intended left-side reach; the boundary guard
    /// (`max(onset − N, previous end, 0)`) keeps that reach from eating
    /// the prior segment. The effective left margin is
    /// `max(speech_pad, preroll)` so the two knobs do not stack.
    pub preroll: Duration,

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
            threshold: None,
            min_speech: Duration::from_millis(120),
            min_silence: Duration::from_millis(700),
            speech_pad: Duration::from_millis(200),
            // Wider than speech_pad: catch late VAD onset without relying
            // on a larger pad that would ignore the previous boundary.
            preroll: Duration::from_millis(400),
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
    /// Left-side rewind in samples: `max(speech_pad, preroll)`.
    left_margin_samples: usize,
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
    /// Sample index of the previous segment's end (already padded). The
    /// next onset rewind must not start before this — otherwise the
    /// trailing pad of utterance N becomes the leading edge of N+1.
    boundary_guard: usize,
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
        let threshold = config.threshold.unwrap_or_else(|| backend.default_threshold());
        let frames = |d: Duration| -> usize {
            let samples = d.as_secs_f64() * sample_rate as f64;
            (samples / frame_size as f64).ceil() as usize
        };
        let to_samples = |d: Duration| -> usize {
            (d.as_secs_f64() * sample_rate as f64) as usize
        };
        let pad_samples = to_samples(config.speech_pad);
        let preroll_samples = to_samples(config.preroll);
        Ok(Self {
            threshold,
            frame_size,
            // At least one frame of each, so a zero duration still means
            // "one frame of evidence" rather than "no evidence needed".
            min_speech_frames: frames(config.min_speech).max(1),
            min_silence_frames: frames(config.min_silence).max(1),
            pad_samples,
            left_margin_samples: pad_samples.max(preroll_samples),
            max_speech_frames: config.max_speech.map(|d| frames(d).max(1)),
            frame_index: 0,
            open_at: None,
            speech_run: 0,
            silence_run: 0,
            last_speech: 0,
            boundary_guard: 0,
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
    ///
    /// Left bound: rewind by `max(speech_pad, preroll)` from the detected
    /// onset, then clamp so the previous segment's end is never
    /// re-included. Right bound: offset plus `speech_pad` only (no
    /// preroll on the trailing edge).
    fn close(&mut self, start_frame: usize, end_frame: usize) -> SpeechSegment {
        let onset = start_frame * self.frame_size;
        let start = onset
            .saturating_sub(self.left_margin_samples)
            .max(self.boundary_guard);
        let end = (end_frame + 1) * self.frame_size + self.pad_samples;

        self.open_at = None;
        self.speech_run = 0;
        self.silence_run = 0;
        self.boundary_guard = end;
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
            preroll: Duration::ZERO,
            ..SegmenterConfig::default()
        }
    }

    /// Pad only — no preroll — so tests that pin the legacy left margin
    /// are not shifted by the wider default preroll.
    fn pad_only(pad: Duration) -> SegmenterConfig {
        SegmenterConfig {
            speech_pad: pad,
            preroll: Duration::ZERO,
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

        let padded = run(segmenter(pad_only(Duration::from_millis(200))), &probs);
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
        let segments = run(segmenter(pad_only(Duration::from_millis(200))), &probs);
        assert_eq!(segments[0].start, 0);
    }

    #[test]
    fn max_speech_forces_a_cut_when_the_speaker_never_pauses() {
        let config = SegmenterConfig {
            max_speech: Some(Duration::from_secs(1)),
            speech_pad: Duration::ZERO,
            preroll: Duration::ZERO,
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
            preroll: Duration::ZERO,
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
            preroll: Duration::ZERO,
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

    /// A backend's own calibration is what an unset threshold means.
    /// `EarshotVad` scores on a different scale from `EnergyVad`, and
    /// assuming one number fits both cost +0.05 WER before it was
    /// measured — so the wiring that reads it is pinned here.
    #[test]
    fn an_unset_threshold_takes_the_backends_calibration() {
        struct Quiet;
        impl VadBackend for Quiet {
            fn frame_size(&self) -> usize {
                160
            }
            fn required_sample_rate(&self) -> Option<u32> {
                None
            }
            fn default_threshold(&self) -> f32 {
                0.2
            }
            fn start(&self) -> Box<dyn crate::vad::VadStream> {
                unreachable!("this test drives the segmenter directly")
            }
        }

        let config = no_padding();
        assert!(config.threshold.is_none(), "the default must be unset");

        // 0.3 is speech at the backend's 0.2 and silence at the
        // segmenter's own 0.5, so the two cannot be confused.
        let seg = Segmenter::new(&Quiet, 16_000, config).unwrap();
        let mut probs = frames(30, 0.3);
        probs.extend(frames(100, 0.0));
        assert_eq!(
            run(seg, &probs).len(),
            1,
            "0.3 should count as speech at the backend's threshold of 0.2"
        );
    }

    #[test]
    fn an_explicit_threshold_overrides_the_backend() {
        struct Quiet;
        impl VadBackend for Quiet {
            fn frame_size(&self) -> usize {
                160
            }
            fn required_sample_rate(&self) -> Option<u32> {
                None
            }
            fn default_threshold(&self) -> f32 {
                0.2
            }
            fn start(&self) -> Box<dyn crate::vad::VadStream> {
                unreachable!("this test drives the segmenter directly")
            }
        }

        let mut config = no_padding();
        config.threshold = Some(0.5);
        let seg = Segmenter::new(&Quiet, 16_000, config).unwrap();

        let mut probs = frames(30, 0.3);
        probs.extend(frames(100, 0.0));
        assert!(
            run(seg, &probs).is_empty(),
            "an explicit 0.5 must win over the backend's 0.2"
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

    // ── preroll + boundary guard (#144) ─────────────────────────────────

    /// With room before the onset, preroll reaches further back than
    /// speech_pad alone. This is the gap speech_pad cannot close without
    /// raising the pad (and then colliding with the previous utterance).
    #[test]
    fn preroll_rewinds_further_than_speech_pad_alone() {
        // 1 s silence, then speech — plenty of history before onset.
        let mut probs = frames(100, 0.0);
        probs.extend(frames(30, 1.0));
        probs.extend(frames(100, 0.0));

        let pad = Duration::from_millis(200);
        let with_pad = run(segmenter(pad_only(pad)), &probs);
        let with_preroll = run(
            segmenter(SegmenterConfig {
                speech_pad: pad,
                preroll: Duration::from_millis(400),
                ..SegmenterConfig::default()
            }),
            &probs,
        );
        assert_eq!(with_pad.len(), 1);
        assert_eq!(with_preroll.len(), 1);

        let onset = 100 * 160;
        assert_eq!(
            with_pad[0].start,
            onset - 3_200,
            "200 ms pad at 16 kHz is 3200 samples"
        );
        assert_eq!(
            with_preroll[0].start,
            onset - 6_400,
            "400 ms preroll must win over the 200 ms pad on the left"
        );
        assert_eq!(
            with_pad[0].end, with_preroll[0].end,
            "preroll must not change the trailing pad"
        );
    }

    /// Raising speech_pad without a boundary guard walks the second
    /// onset back into the first segment. That failure mode is what
    /// preroll + guard exist to avoid: keep a wide left margin, but
    /// never cross the previous end.
    #[test]
    fn boundary_guard_stops_preroll_at_the_previous_segment_end() {
        let config = SegmenterConfig {
            // Short silence so neighbours sit close; wide left margin.
            min_silence: Duration::from_millis(100),
            speech_pad: Duration::from_millis(200),
            preroll: Duration::from_millis(500),
            max_speech: None,
            ..SegmenterConfig::default()
        };

        // 300 ms speech, 100 ms silence (closes), 300 ms speech, silence.
        let mut probs = frames(30, 1.0);
        probs.extend(frames(10, 0.0));
        probs.extend(frames(30, 1.0));
        probs.extend(frames(100, 0.0));

        let segments = run(segmenter(config), &probs);
        assert_eq!(segments.len(), 2, "got {segments:?}");
        assert!(
            segments[1].start >= segments[0].end,
            "second segment must not start inside the first: {:?}",
            segments
        );
        assert_eq!(
            segments[1].start, segments[0].end,
            "with a 500 ms preroll into a ~100 ms gap, the guard should \
             pin the second start exactly at the first end"
        );
    }

    /// Without a boundary guard, a large left margin on the second
    /// utterance would start before the first utterance's end. The
    /// segmenter must refuse that overlap even when only `speech_pad`
    /// supplies the left margin (preroll zero).
    #[test]
    fn speech_pad_alone_would_overlap_neighbours_without_a_guard() {
        let pad_ms = 300u64;
        let speech_frames = 30;
        let gap_frames = 40; // 400 ms — closes after min_silence (100 ms)

        let end1_padded = speech_frames * 160 + (pad_ms as usize * 16);
        let onset2 = (speech_frames + gap_frames) * 160;
        let unguarded_start2 = onset2.saturating_sub(pad_ms as usize * 16);
        assert!(
            unguarded_start2 < end1_padded,
            "fixture broken: unguarded second start {unguarded_start2}              should fall inside first padded end {end1_padded}"
        );

        let config = SegmenterConfig {
            min_silence: Duration::from_millis(100),
            speech_pad: Duration::from_millis(pad_ms),
            preroll: Duration::ZERO,
            max_speech: None,
            ..SegmenterConfig::default()
        };
        let mut probs = frames(speech_frames, 1.0);
        probs.extend(frames(gap_frames, 0.0));
        probs.extend(frames(speech_frames, 1.0));
        probs.extend(frames(100, 0.0));

        let segments = run(segmenter(config), &probs);
        assert_eq!(segments.len(), 2, "got {segments:?}");
        assert!(
            segments[1].start >= segments[0].end,
            "even pad-only left margins must respect the boundary guard;              got {:?}",
            segments
        );
    }

    /// Left margin uses max(pad, preroll), not the sum — stacking would
    /// drag half a second of silence into every short utterance.
    #[test]
    fn preroll_and_speech_pad_do_not_stack_on_the_left() {
        let mut probs = frames(100, 0.0);
        probs.extend(frames(30, 1.0));
        probs.extend(frames(100, 0.0));

        let segments = run(
            segmenter(SegmenterConfig {
                speech_pad: Duration::from_millis(200),
                preroll: Duration::from_millis(400),
                ..SegmenterConfig::default()
            }),
            &probs,
        );
        let onset = 100 * 160;
        assert_eq!(
            segments[0].start,
            onset - 6_400,
            "left margin must be max(200,400)=400 ms, not 600 ms"
        );
    }

    #[test]
    fn defaults_expose_preroll_wider_than_speech_pad() {
        let d = SegmenterConfig::default();
        assert_eq!(d.speech_pad, Duration::from_millis(200));
        assert_eq!(d.preroll, Duration::from_millis(400));
        assert!(
            d.preroll > d.speech_pad,
            "preroll should outreach speech_pad so the left edge gains \
             from history rather than from a larger pad"
        );
    }
}
