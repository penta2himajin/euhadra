//! Log-mel spectrogram for Whisper-large-v3-turbo.
//!
//! Reproduces `openai/whisper`'s `log_mel_spectrogram` exactly so the
//! encoder ONNX sees features in its training distribution. Matches
//! HF `WhisperFeatureExtractor` when `feature_size = 128`.
//!
//! ```text
//! sample_rate       = 16_000
//! n_fft             = 400      # 25 ms
//! hop_length        = 160      # 10 ms
//! n_mels            = 128       (large-v3 / large-v3-turbo)
//! window            = periodic Hann length 400
//! padding           = pad/trim to 30 s (480_000 samples) with zeros
//! stft mode         = center=True, reflect-pad n_fft/2 each side
//! mel scale         = Slaney (librosa `htk=False`)
//! mel norm          = Slaney (each filter integrates to 1)
//! log               = log10(clamp(power, 1e-10))
//! post-norm         = clamp to (max - 8), then (x + 4) / 4
//! ```
//!
//! Output shape: `(128, 3000)` row-major (n_mels × n_frames), ready
//! to feed into the encoder's `input_features` tensor after adding
//! the batch dim.

use std::f32::consts::PI;
use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

/// 16 kHz × 30 s = 480 000 samples.
pub const N_SAMPLES: usize = 480_000;
/// `N_SAMPLES / HOP_LENGTH` frames per 30 s window (matches the
/// encoder's `input_features` time axis after dropping the trailing
/// `stft[..., :-1]` frame).
pub const N_FRAMES: usize = 3_000;
/// Whisper FFT length (25 ms at 16 kHz).
pub const N_FFT: usize = 400;
/// Whisper STFT hop (10 ms at 16 kHz).
pub const HOP_LENGTH: usize = 160;
/// Mel band count for the large-v3 / large-v3-turbo family.
pub const N_MELS: usize = 128;

/// Precomputed window + filterbank + FFT plan; rebuild once per
/// adapter, never per utterance.
pub struct WhisperMel {
    window: Vec<f32>,
    mel_filters: Vec<MelFilter>,
    fft: Arc<dyn Fft<f32> + Send + Sync>,
}

#[derive(Debug, Clone)]
struct MelFilter {
    start_bin: usize,
    weights: Vec<f32>,
}

impl Default for WhisperMel {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperMel {
    pub fn new() -> Self {
        let window = periodic_hann(N_FFT);
        let mel_filters = build_slaney_mel_filters(16_000.0, N_FFT, N_MELS, 0.0, 8000.0);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        Self {
            window,
            mel_filters,
            fft,
        }
    }

    /// Produce the `(N_MELS, N_FRAMES)` row-major log-mel buffer for
    /// `samples` (16 kHz mono `f32`). Audio shorter than 30 s is
    /// zero-padded to 480 000 samples; longer audio is truncated.
    /// Callers that need to transcribe >30 s must chunk upstream.
    pub fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let mut audio = vec![0.0_f32; N_SAMPLES];
        let take = samples.len().min(N_SAMPLES);
        audio[..take].copy_from_slice(&samples[..take]);

        let padded = reflect_pad(&audio, N_FFT / 2);

        let n_bins = N_FFT / 2 + 1;
        let total_frames = (padded.len() - N_FFT) / HOP_LENGTH + 1;
        // Drop the trailing frame (matches torch.stft + `[..., :-1]`).
        let keep_frames = total_frames.saturating_sub(1).min(N_FRAMES);

        // Power spectrogram, mel × frame layout for cache-friendly
        // per-frame writes. Allocate `N_MELS × N_FRAMES` so the
        // tensor handed to the encoder always has the expected shape
        // even if the audio is shorter than 30 s (the trailing
        // frames stay at the post-norm floor and act as silence).
        let mut log_spec = vec![0.0_f32; N_MELS * N_FRAMES];
        let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];

        for f in 0..keep_frames {
            let start = f * HOP_LENGTH;
            for i in 0..N_FFT {
                buf[i] = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);

            for (m, filt) in self.mel_filters.iter().enumerate() {
                let mut energy = 0.0_f32;
                for (k, w) in filt.weights.iter().enumerate() {
                    let bin = filt.start_bin + k;
                    if bin < n_bins {
                        let c = buf[bin];
                        energy += (c.re * c.re + c.im * c.im) * *w;
                    }
                }
                // log10(clamp(power, 1e-10)).
                let clamped = energy.max(1e-10);
                log_spec[m * N_FRAMES + f] = clamped.log10();
            }
        }

        // The trailing `N_FRAMES - keep_frames` slots are still
        // zero (= log10(1)). Whisper's post-norm collapses those to
        // the per-spec floor below, which is the correct silence value.
        if keep_frames < N_FRAMES {
            // Re-init untouched cells to -inf so the max scan
            // doesn't pick up a 0.0 sentinel. We then floor-clamp
            // them below, leaving them at (max - 8).
            for m in 0..N_MELS {
                for f in keep_frames..N_FRAMES {
                    log_spec[m * N_FRAMES + f] = f32::NEG_INFINITY;
                }
            }
        }

        // max-clamp + rescale, matching openai/whisper:
        //   log_spec = max(log_spec, log_spec.max() - 8.0)
        //   log_spec = (log_spec + 4.0) / 4.0
        let mut peak = f32::NEG_INFINITY;
        for &v in &log_spec {
            if v > peak {
                peak = v;
            }
        }
        let floor = peak - 8.0;
        for v in log_spec.iter_mut() {
            if *v < floor {
                *v = floor;
            }
            *v = (*v + 4.0) / 4.0;
        }

        log_spec
    }
}

/// `numpy.pad(audio, half, mode='reflect')` — mirrors around the
/// boundary samples themselves (audio[0], audio[N-1]). For a length-N
/// input this returns length `N + 2*half`.
fn reflect_pad(audio: &[f32], half: usize) -> Vec<f32> {
    let n = audio.len();
    assert!(n > half, "reflect_pad needs n > half");
    let mut out = Vec::with_capacity(n + 2 * half);
    // Left reflect: audio[half], audio[half-1], ..., audio[1]
    for i in 0..half {
        out.push(audio[half - i]);
    }
    out.extend_from_slice(audio);
    // Right reflect: audio[n-2], audio[n-3], ..., audio[n-half-1]
    for i in 0..half {
        out.push(audio[n - 2 - i]);
    }
    out
}

/// `torch.hann_window(N, periodic=True)` — length N, denominator N
/// (not N-1). Matches openai/whisper's `torch.hann_window(N_FFT)`.
fn periodic_hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_sp = 200.0_f32 / 3.0;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4_f32).ln() / 27.0;
    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_sp = 200.0_f32 / 3.0;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4_f32).ln() / 27.0;
    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

/// `librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax, htk=False,
/// norm='slaney')`. Triangular bands on Slaney mel scale with
/// `2 / (right_hz - left_hz)` normalisation so each filter integrates
/// to 1 in the linear-frequency domain.
fn build_slaney_mel_filters(
    sample_rate: f32,
    n_fft: usize,
    n_mels: usize,
    low_freq: f32,
    high_freq: f32,
) -> Vec<MelFilter> {
    let n_bins = n_fft / 2 + 1;
    let bin_hz = sample_rate / n_fft as f32;

    let mel_lo = hz_to_mel_slaney(low_freq);
    let mel_hi = hz_to_mel_slaney(high_freq);
    let mel_step = (mel_hi - mel_lo) / (n_mels + 1) as f32;
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_lo + i as f32 * mel_step)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|m| mel_to_hz_slaney(*m)).collect();

    let mut filters = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let left_hz = hz_points[m];
        let centre_hz = hz_points[m + 1];
        let right_hz = hz_points[m + 2];
        let enorm = 2.0 / (right_hz - left_hz).max(1e-10);

        let mut start_bin: Option<usize> = None;
        let mut weights_full: Vec<f32> = Vec::new();
        for bin in 0..n_bins {
            let hz = bin as f32 * bin_hz;
            if hz <= left_hz || hz >= right_hz {
                continue;
            }
            let w_unnorm = if hz <= centre_hz {
                (hz - left_hz) / (centre_hz - left_hz).max(1e-10)
            } else {
                (right_hz - hz) / (right_hz - centre_hz).max(1e-10)
            };
            if !w_unnorm.is_finite() || w_unnorm <= 0.0 {
                continue;
            }
            let w = w_unnorm * enorm;
            if start_bin.is_none() {
                start_bin = Some(bin);
            }
            let s = start_bin.unwrap();
            while weights_full.len() < bin - s {
                weights_full.push(0.0);
            }
            weights_full.push(w);
        }
        filters.push(MelFilter {
            start_bin: start_bin.unwrap_or(0),
            weights: weights_full,
        });
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_hann_starts_at_zero_and_peaks_near_middle() {
        let w = periodic_hann(400);
        assert!(w[0].abs() < 1e-7, "{}", w[0]);
        assert!(w[200] > 0.999, "{}", w[200]);
        // periodic Hann does NOT return to zero at the last sample
        assert!(w[399] > 0.0 && w[399] < 0.01, "{}", w[399]);
    }

    #[test]
    fn reflect_pad_matches_numpy_example() {
        // numpy.pad([1,2,3,4,5], (2,2), 'reflect') = [3,2,1,2,3,4,5,4,3]
        let audio = [1.0, 2.0, 3.0, 4.0, 5.0];
        let got = reflect_pad(&audio, 2);
        assert_eq!(got, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn output_shape_is_n_mels_x_n_frames() {
        let mel = WhisperMel::new();
        let silence = vec![0.0_f32; N_SAMPLES];
        let out = mel.compute(&silence);
        assert_eq!(out.len(), N_MELS * N_FRAMES);
    }

    #[test]
    fn output_finite_and_in_post_norm_range_for_silence() {
        let mel = WhisperMel::new();
        let silence = vec![0.0_f32; N_SAMPLES];
        let out = mel.compute(&silence);
        // After max-clamp + (x+4)/4 the per-spec floor sits at
        // (max - 8 + 4) / 4. For pure silence max = log10(1e-10) = -10
        // so floor = (-18 + 4)/4 = -3.5 and every cell equals it.
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "cell {i} = {v}");
            assert!((-3.5..=1.0).contains(v), "cell {i} = {v}");
        }
    }

    #[test]
    fn short_audio_is_zero_padded_and_yields_full_grid() {
        let mel = WhisperMel::new();
        let one_sec = vec![0.0_f32; 16_000];
        let out = mel.compute(&one_sec);
        assert_eq!(out.len(), N_MELS * N_FRAMES);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn slaney_mel_known_anchors() {
        assert!(hz_to_mel_slaney(0.0).abs() < 1e-6);
        assert!((hz_to_mel_slaney(1000.0) - 15.0).abs() < 1e-4);
        for hz in [50.0_f32, 200.0, 500.0, 1000.0, 4000.0, 8000.0] {
            let rt = mel_to_hz_slaney(hz_to_mel_slaney(hz));
            assert!((rt - hz).abs() <= hz * 1e-4, "hz={hz} rt={rt}");
        }
    }

    #[test]
    fn filterbank_has_correct_count_and_no_empty_bands() {
        let filters = build_slaney_mel_filters(16_000.0, N_FFT, N_MELS, 0.0, 8000.0);
        assert_eq!(filters.len(), N_MELS);
        for (i, f) in filters.iter().enumerate() {
            let any_nonzero = f.weights.iter().any(|&w| w > 0.0);
            assert!(any_nonzero, "filter {i} is empty");
        }
    }
}
