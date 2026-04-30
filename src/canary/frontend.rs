//! Mel-spectrogram preprocessor for Canary-180M-Flash.
//!
//! Reproduces the parameters NeMo's
//! `AudioToMelSpectrogramPreprocessor` ships with for Canary, and
//! mirrors the numpy implementation in `onnx-asr`'s
//! `numpy_preprocessor.py`:
//!
//! ```text
//! sample_rate     = 16_000
//! n_fft           = 512
//! win_length      = 400      # 25 ms
//! hop_length      = 160      # 10 ms
//! n_mels          = 80
//! preemph         = 0.97
//! window          = hann
//! mel scale       = Slaney  (librosa filters.mel htk=False)
//! mel norm        = Slaney  (each filter integrates to 1)
//! log             = ln(power + 2**-24)
//! norm            = per-feature mean/var across valid frames
//! ```
//!
//! Output: `[num_frames, n_mels]` row-major buffer plus the
//! frame-count, both passed straight into the encoder ONNX graph
//! after a transpose to `[n_mels, num_frames]` with batch dim added.
//!
//! Test coverage is numerical-sanity only at this layer; an end-to-
//! end fidelity check (matching `onnx-asr` outputs on a real WAV)
//! lives in the encoder PR where we can compare full pipelines.

use std::f32::consts::PI;

/// Tunable parameters of the Canary mel preprocessor. The defaults
/// match the NeMo Canary checkpoint config — change them and the
/// encoder will see features it wasn't trained on.
#[derive(Debug, Clone)]
pub struct MelOpts {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub preemph: f32,
    pub log_zero_guard: f32,
    /// Lower edge of the mel filterbank (Hz). NeMo defaults to 0.
    pub low_freq: f32,
    /// Upper edge (Hz). 0 means `sample_rate / 2`.
    pub high_freq: f32,
}

impl MelOpts {
    /// Defaults that match `nvidia/canary-180m-flash` and the
    /// `onnx-asr` numpy reference.
    pub fn canary_default() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            n_mels: 80,
            preemph: 0.97,
            log_zero_guard: 2.0_f32.powi(-24),
            low_freq: 0.0,
            high_freq: 0.0,
        }
    }

    pub fn effective_high_freq(&self) -> f32 {
        if self.high_freq > 0.0 {
            self.high_freq
        } else {
            (self.sample_rate as f32) / 2.0
        }
    }
}

/// Pre-computed Hann window + mel filterbank, kept on the adapter
/// so we don't rebuild them per utterance.
pub struct MelFrontend {
    opts: MelOpts,
    window: Vec<f32>,
    mel_filters: Vec<MelFilter>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32> + Send + Sync>,
}

#[derive(Debug, Clone)]
struct MelFilter {
    /// Inclusive [start_bin, end_bin] in the FFT power spectrum.
    start_bin: usize,
    /// Filter weights for bins start_bin .. start_bin + weights.len().
    weights: Vec<f32>,
}

impl MelFrontend {
    pub fn new(opts: MelOpts) -> Self {
        let window = hann_window(opts.win_length);
        let mel_filters = build_slaney_mel_filters(
            opts.sample_rate as f32,
            opts.n_fft,
            opts.n_mels,
            opts.low_freq,
            opts.effective_high_freq(),
        );
        let mut planner = rustfft::FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(opts.n_fft);
        Self {
            opts,
            window,
            mel_filters,
            fft,
        }
    }

    pub fn n_mels(&self) -> usize {
        self.opts.n_mels
    }

    /// Compute log-mel features and apply per-feature mean/variance
    /// normalisation across time. Returns the `[num_frames, n_mels]`
    /// row-major buffer plus the frame count.
    pub fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let frame_len = self.opts.win_length;
        let hop = self.opts.hop_length;
        let n_mels = self.opts.n_mels;

        if samples.len() < frame_len {
            return (Vec::new(), 0);
        }

        // snip_edges-style framing: number of frames that fully fit
        // in the input. Onnx-asr's numpy preprocessor does the same
        // (no padding before / after).
        let num_frames = (samples.len() - frame_len) / hop + 1;
        let mut features = vec![0.0_f32; num_frames * n_mels];

        let mut buf = vec![rustfft::num_complex::Complex32::new(0.0, 0.0); self.opts.n_fft];
        let mut frame = vec![0.0_f32; frame_len];

        for f in 0..num_frames {
            let start = f * hop;
            frame.copy_from_slice(&samples[start..start + frame_len]);

            // Pre-emphasis: y[t] = x[t] - coeff * x[t-1].
            // The numpy reference sets the boundary as
            // y[0] = x[0] (no scaling), unlike Kaldi's
            // y[0] = (1-coeff)*x[0]. Match it here: walk backwards
            // and leave frame[0] alone.
            let coeff = self.opts.preemph;
            if coeff != 0.0 {
                for n in (1..frame_len).rev() {
                    frame[n] -= coeff * frame[n - 1];
                }
            }

            // Hann window
            for (s, w) in frame.iter_mut().zip(self.window.iter()) {
                *s *= *w;
            }

            // Pack into FFT buffer (zero-pad to n_fft).
            for (i, c) in buf.iter_mut().enumerate() {
                *c = if i < frame_len {
                    rustfft::num_complex::Complex32::new(frame[i], 0.0)
                } else {
                    rustfft::num_complex::Complex32::new(0.0, 0.0)
                };
            }
            self.fft.process(&mut buf);

            // Power spectrum (first n_fft/2+1 bins).
            let n_bins = self.opts.n_fft / 2 + 1;

            // Apply mel filters; log with zero-guard.
            for (m, filt) in self.mel_filters.iter().enumerate() {
                let mut energy = 0.0_f32;
                for (k, w) in filt.weights.iter().enumerate() {
                    let bin = filt.start_bin + k;
                    if bin < n_bins {
                        let c = buf[bin];
                        energy += (c.re * c.re + c.im * c.im) * *w;
                    }
                }
                features[f * n_mels + m] = (energy + self.opts.log_zero_guard).ln();
            }
        }

        // Per-feature CMVN: subtract the across-time mean and divide
        // by the across-time standard deviation, computed independently
        // per mel bin. Matches `onnx-asr`'s
        // `(features - mean) / (std + 1e-5)`.
        for m in 0..n_mels {
            let mut sum = 0.0_f32;
            for f in 0..num_frames {
                sum += features[f * n_mels + m];
            }
            let mean = sum / num_frames as f32;

            let mut sq = 0.0_f32;
            for f in 0..num_frames {
                let d = features[f * n_mels + m] - mean;
                sq += d * d;
            }
            let std = (sq / num_frames as f32).sqrt();
            let denom = std + 1e-5;

            for f in 0..num_frames {
                features[f * n_mels + m] = (features[f * n_mels + m] - mean) / denom;
            }
        }

        (features, num_frames)
    }
}

fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    // Periodic Hann to match numpy's `np.hanning(N)` (which is the
    // *symmetric* form: 0.5 * (1 - cos(2π i / (N-1)))).
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / denom).cos()))
        .collect()
}

/// Slaney-style mel scale (librosa.hz_to_mel with htk=False):
/// linear below 1000 Hz, log above.
fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_sp = 200.0_f32 / 3.0;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = (min_log_hz - 0.0) / f_sp;
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
    let min_log_mel = (min_log_hz - 0.0) / f_sp;
    let logstep = (6.4_f32).ln() / 27.0;

    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

/// Triangular mel filterbank with **Slaney normalisation** (each
/// filter integrates to 1.0 in the linear-frequency domain). This
/// matches `librosa.filters.mel(..., htk=False, norm='slaney')`,
/// which is the configuration NeMo's Canary checkpoints expect.
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
    // n_mels + 2 mel-equispaced points define n_mels triangular filters.
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

        // Slaney normalisation factor: 2 / (right_hz - left_hz).
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
            // Pad zeros for any gap between previously-emitted bins
            // and this one (extremely narrow filters at low mel).
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

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn hann_window_endpoints_are_zero() {
        let w = hann_window(400);
        // Symmetric Hann: w[0] = 0, w[N-1] = 0.
        assert!(approx_eq(w[0], 0.0, 1e-6), "{}", w[0]);
        assert!(approx_eq(w[399], 0.0, 1e-6), "{}", w[399]);
    }

    #[test]
    fn hann_window_peak_at_centre() {
        let w = hann_window(400);
        // Symmetric Hann peaks at the (N-1)/2 index.
        let mid = 199; // (400 - 1) / 2 with integer truncation
        assert!(w[mid] > 0.99 && w[mid] <= 1.0, "{}", w[mid]);
    }

    #[test]
    fn hann_window_short_sizes() {
        assert_eq!(hann_window(0), Vec::<f32>::new());
        assert_eq!(hann_window(1), vec![1.0]);
    }

    #[test]
    fn slaney_mel_known_anchors() {
        // hz_to_mel_slaney(0) = 0, hz_to_mel_slaney(1000) = 15
        // (the linear-to-log transition point).
        assert!(approx_eq(hz_to_mel_slaney(0.0), 0.0, 1e-6));
        assert!(approx_eq(hz_to_mel_slaney(1000.0), 15.0, 1e-4));

        // hz_to_mel_slaney(200) = 200 / (200/3) = 3
        assert!(approx_eq(hz_to_mel_slaney(200.0), 3.0, 1e-4));

        // Round-trip: mel_to_hz(hz_to_mel(x)) ≈ x for several anchors.
        for hz in [50.0_f32, 200.0, 500.0, 1000.0, 4000.0, 8000.0] {
            let rt = mel_to_hz_slaney(hz_to_mel_slaney(hz));
            assert!(approx_eq(rt, hz, hz * 1e-4), "hz={hz} rt={rt}");
        }
    }

    #[test]
    fn slaney_mel_filterbank_has_correct_count_and_nonempty() {
        let filters = build_slaney_mel_filters(16_000.0, 512, 80, 0.0, 8000.0);
        assert_eq!(filters.len(), 80);
        // Every filter should have at least one non-zero weight (with
        // 80 mels at 16 kHz / n_fft=512 the bin spacing of 31.25 Hz
        // is fine enough that no filter degenerates to zero bins).
        for (i, f) in filters.iter().enumerate() {
            let any_nonzero = f.weights.iter().any(|&w| w > 0.0);
            assert!(any_nonzero, "filter {i} is empty: {f:?}");
        }
    }

    #[test]
    fn slaney_mel_normalisation_keeps_filters_finite() {
        let filters = build_slaney_mel_filters(16_000.0, 512, 80, 0.0, 8000.0);
        for (i, f) in filters.iter().enumerate() {
            for (k, &w) in f.weights.iter().enumerate() {
                assert!(
                    w.is_finite() && w >= 0.0,
                    "filter {i} bin {k} = {w}"
                );
            }
        }
    }

    #[test]
    fn frontend_short_input_returns_empty() {
        let fe = MelFrontend::new(MelOpts::canary_default());
        // Less than win_length (400) → no frames possible.
        let (feats, n_frames) = fe.compute(&vec![0.0; 100]);
        assert!(feats.is_empty());
        assert_eq!(n_frames, 0);
    }

    #[test]
    fn frontend_output_shape_matches_n_mels_x_frames() {
        let fe = MelFrontend::new(MelOpts::canary_default());
        // 1 second of audio at 16 kHz → expect floor((16000 - 400) /
        // 160) + 1 = 98 frames.
        let samples = vec![0.0_f32; 16_000];
        let (feats, n_frames) = fe.compute(&samples);
        assert_eq!(n_frames, 98);
        assert_eq!(feats.len(), n_frames * 80);
    }

    #[test]
    fn frontend_output_is_finite_for_silence() {
        let fe = MelFrontend::new(MelOpts::canary_default());
        // Silence triggers the log-zero-guard branch and exercises
        // the divide-by-near-zero in CMVN. Output must still be
        // finite (no NaN / inf) — encoder will explode otherwise.
        let samples = vec![0.0_f32; 16_000];
        let (feats, _n) = fe.compute(&samples);
        for (i, x) in feats.iter().enumerate() {
            assert!(x.is_finite(), "feature {i} = {x}");
        }
    }

    #[test]
    fn frontend_cmvn_zero_means_per_feature() {
        let fe = MelFrontend::new(MelOpts::canary_default());
        // Wideband pseudo-noise so every mel bin sees non-trivial
        // variance — a pure tone would leave most mels at the
        // log-zero-guard floor where CMVN's `+ 1e-5` denom guard
        // amplifies float-noise into a non-zero residual mean.
        // Deterministic LCG so the test is reproducible without
        // pulling a `rand` dep.
        let mut state: u32 = 0xDEADBEEF;
        let n = 16_000_usize;
        let samples: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        let (feats, n_frames) = fe.compute(&samples);

        // After per-feature CMVN each mel bin should have
        // approximately zero mean across time. With wideband input
        // the std-guard contribution is small, so the residual is
        // float-rounding only.
        for m in 0..80 {
            let sum: f32 = (0..n_frames).map(|f| feats[f * 80 + m]).sum();
            let mean = sum / n_frames as f32;
            assert!(
                mean.abs() < 1e-3,
                "mel {m} mean after CMVN = {mean}"
            );
        }
    }

    #[test]
    fn frontend_default_opts_match_canary_card() {
        // Pin the public defaults so a stray Cargo edit can't ship
        // a Canary preprocessor with mismatched parameters.
        let o = MelOpts::canary_default();
        assert_eq!(o.sample_rate, 16_000);
        assert_eq!(o.n_fft, 512);
        assert_eq!(o.win_length, 400);
        assert_eq!(o.hop_length, 160);
        assert_eq!(o.n_mels, 80);
        assert!(approx_eq(o.preemph, 0.97, 1e-9));
        assert!(approx_eq(o.log_zero_guard, 2.0_f32.powi(-24), 1e-30));
    }
}
