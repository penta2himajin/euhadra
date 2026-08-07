use earshot::{DefaultPredictor, Detector};

use super::{VadBackend, VadStream};

/// Samples per frame. Fixed by the model: 256 samples, 16 ms at 16 kHz.
const FRAME_SIZE: usize = 256;

/// The only rate the network was trained at.
const SAMPLE_RATE: u32 = 16_000;

/// Detects speech with a small neural network.
///
/// The recommended [`VadBackend`]. Unlike
/// [`EnergyVad`](super::EnergyVad) it is deciding whether a frame *is
/// speech*, not whether it is louder than the room, so a keyboard, a
/// door or music do not open an utterance and a quiet speaker is not
/// gradually muted by the noise floor climbing to meet them.
///
/// Costs one pure-Rust dependency, [`earshot`], and nothing else: the
/// network is 40 KiB embedded in that crate, so there is no ONNX
/// runtime, no model file to fetch, and no weights for euhadra to
/// redistribute. Behind the `vad` feature.
///
/// # Rate
///
/// 16 kHz only. Feeding it anything else is refused at
/// [`Segmenter::new`](super::Segmenter) rather than resampled silently —
/// a network fed the wrong rate returns confident numbers that mean
/// nothing, and a caller who thinks detection is running is worse off
/// than one who knows it is not.
///
/// ```
/// use euhadra::vad::{EarshotVad, SegmenterConfig, segment_buffer};
///
/// let mut samples = vec![0.0f32; 16_000];
/// samples.extend((0..24_000).map(|i| (i as f32 * 0.1).sin() * 0.3));
/// samples.extend(vec![0.0f32; 16_000]);
///
/// let segments = segment_buffer(
///     &EarshotVad::new(),
///     &samples,
///     16_000,
///     &SegmenterConfig::default(),
/// )?;
/// # let _ = segments;
/// # Ok::<(), euhadra::vad::VadError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct EarshotVad;

impl EarshotVad {
    pub fn new() -> Self {
        Self
    }
}

impl VadBackend for EarshotVad {
    fn frame_size(&self) -> usize {
        FRAME_SIZE
    }

    fn required_sample_rate(&self) -> Option<u32> {
        Some(SAMPLE_RATE)
    }

    fn start(&self) -> Box<dyn VadStream> {
        // A fresh detector per pass: it carries state across frames, and
        // one utterance's must not colour the next.
        Box::new(EarshotStream {
            detector: Detector::default(),
        })
    }
}

struct EarshotStream {
    detector: Detector<DefaultPredictor>,
}

impl VadStream for EarshotStream {
    fn speech_probability(&mut self, frame: &[f32]) -> f32 {
        self.detector.predict_f32(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::{segment_buffer, SegmenterConfig, VadError};

    /// White-ish noise stands in for speech better than a pure tone: the
    /// network keys on spectral shape, and a single sinusoid is not what
    /// a voice looks like. Deterministic so the test cannot flake.
    fn voiced(n: usize) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..n)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let noise = (state >> 40) as f32 / 8388608.0 - 1.0;
                // An amplitude envelope in the syllabic range, so the
                // frame looks modulated rather than stationary.
                let envelope = 0.5 + 0.5 * (i as f32 * 0.0008).sin();
                noise * 0.3 * envelope
            })
            .collect()
    }

    #[test]
    fn digital_silence_yields_no_segments() {
        let segments = segment_buffer(
            &EarshotVad::new(),
            &vec![0.0f32; 32_000],
            16_000,
            &SegmenterConfig::default(),
        )
        .unwrap();
        assert!(
            segments.is_empty(),
            "silence must not produce an utterance, got {segments:?}"
        );
    }

    /// The backend is only useful if it scores *something* as speech;
    /// a detector stuck at zero would pass the silence test above and
    /// break the pipeline completely.
    #[test]
    fn a_modulated_signal_scores_above_silence() {
        let backend = EarshotVad::new();
        let signal = voiced(FRAME_SIZE * 100);
        let mut stream = backend.start();
        let voiced_max = signal
            .chunks(FRAME_SIZE)
            .map(|f| stream.speech_probability(f))
            .fold(0.0f32, f32::max);

        let mut stream = backend.start();
        let silent_max = vec![0.0f32; FRAME_SIZE * 100]
            .chunks(FRAME_SIZE)
            .map(|f| stream.speech_probability(f))
            .fold(0.0f32, f32::max);

        assert!(
            voiced_max > silent_max,
            "a modulated signal ({voiced_max}) should outscore silence ({silent_max})"
        );
    }

    #[test]
    fn a_detector_does_not_carry_state_between_passes() {
        let backend = EarshotVad::new();
        let signal = voiced(FRAME_SIZE * 40);

        let score = |backend: &EarshotVad| {
            let mut stream = backend.start();
            signal
                .chunks(FRAME_SIZE)
                .map(|f| stream.speech_probability(f))
                .sum::<f32>()
        };
        assert_eq!(
            score(&backend),
            score(&backend),
            "two passes over the same audio must score identically"
        );
    }

    #[test]
    fn a_non_16k_recording_is_refused_rather_than_resampled() {
        let err = segment_buffer(
            &EarshotVad::new(),
            &vec![0.0f32; 4_800],
            48_000,
            &SegmenterConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                VadError::SampleRate {
                    required: 16_000,
                    actual: 48_000
                }
            ),
            "got {err:?}"
        );
    }
}
