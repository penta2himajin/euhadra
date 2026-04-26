//! Kaldi-compatible log Mel-filterbank feature extraction.
//!
//! Reproduces the configuration that FunASR's Paraformer-large pipes
//! through `kaldi_native_fbank.OnlineFbank`:
//!
//! - 16 kHz / mono input.
//! - 25 ms frame length, 10 ms shift, snip_edges = true.
//! - Pre-emphasis 0.97, Hamming window.
//! - 512-pt FFT, power spectrum.
//! - 80 triangular mel filters, low_freq = 20 Hz, high_freq = fs / 2.
//! - `dither = 0` (the offline runtime disables dithering for
//!   determinism — see `runtime/onnxruntime/src/paraformer.cpp`).
//! - log(power + ε) with ε = 1e-10.
//!
//! Output is a row-major `[num_frames, n_mels]` `Vec<f32>`.

use rustfft::{FftPlanner, num_complex::Complex32};
use std::f32::consts::PI;

#[derive(Debug, Clone)]
pub struct FbankOpts {
    pub sample_rate: u32,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    pub n_mels: usize,
    pub fft_size: usize,
    pub low_freq: f32,
    pub high_freq: f32,
    pub preemph_coeff: f32,
}

impl FbankOpts {
    pub fn paraformer_default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            n_mels: 80,
            fft_size: 512,
            low_freq: 20.0,
            high_freq: 0.0, // 0 → derive from sample_rate / 2
            preemph_coeff: 0.97,
        }
    }

    pub fn frame_len_samples(&self) -> usize {
        ((self.sample_rate as f32) * self.frame_length_ms / 1000.0).round() as usize
    }
    pub fn frame_shift_samples(&self) -> usize {
        ((self.sample_rate as f32) * self.frame_shift_ms / 1000.0).round() as usize
    }
    pub fn effective_high_freq(&self) -> f32 {
        if self.high_freq > 0.0 {
            self.high_freq
        } else {
            (self.sample_rate as f32) / 2.0
        }
    }
}

/// Pre-computed Hamming window + mel filterbank, kept on the adapter
/// so we don't rebuild the matrix per utterance.
pub struct Fbank {
    opts: FbankOpts,
    window: Vec<f32>,
    mel_filters: Vec<MelFilter>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32> + Send + Sync>,
}

#[derive(Debug, Clone)]
struct MelFilter {
    /// Inclusive [start_bin, end_bin] in the FFT power spectrum.
    start_bin: usize,
    weights: Vec<f32>,
}

impl Fbank {
    pub fn new(opts: FbankOpts) -> Self {
        let frame_len = opts.frame_len_samples();
        let window = hamming_window(frame_len);
        let mel_filters = build_mel_filters(
            opts.sample_rate as f32,
            opts.fft_size,
            opts.n_mels,
            opts.low_freq,
            opts.effective_high_freq(),
        );
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(opts.fft_size);
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

    /// Compute log-mel filterbank features. Returns the row-major
    /// `[num_frames, n_mels]` buffer plus the frame count.
    pub fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let frame_len = self.opts.frame_len_samples();
        let frame_shift = self.opts.frame_shift_samples();

        if samples.len() < frame_len {
            return (Vec::new(), 0);
        }

        // snip_edges = true: number of frames is floor((N - frame_len) / shift) + 1
        let num_frames = (samples.len() - frame_len) / frame_shift + 1;
        let n_mels = self.opts.n_mels;
        let mut out = Vec::with_capacity(num_frames * n_mels);

        let mut buf = vec![Complex32::new(0.0, 0.0); self.opts.fft_size];
        let mut frame = vec![0.0_f32; frame_len];

        for f in 0..num_frames {
            let start = f * frame_shift;
            frame.copy_from_slice(&samples[start..start + frame_len]);

            // Pre-emphasis: y[t] = x[t] - coeff * x[t-1] (Kaldi uses
            // x[0] - coeff*x[0] = (1-coeff)*x[0] for the first sample).
            let coeff = self.opts.preemph_coeff;
            if coeff != 0.0 {
                for n in (1..frame_len).rev() {
                    frame[n] -= coeff * frame[n - 1];
                }
                frame[0] -= coeff * frame[0];
            }

            // Window
            for (sample, w) in frame.iter_mut().zip(self.window.iter()) {
                *sample *= *w;
            }

            // Pack into FFT buffer (zero-pad)
            for (i, c) in buf.iter_mut().enumerate() {
                *c = if i < frame_len {
                    Complex32::new(frame[i], 0.0)
                } else {
                    Complex32::new(0.0, 0.0)
                };
            }
            self.fft.process(&mut buf);

            // Power spectrum, only the first fft_size/2 + 1 bins are unique.
            let n_bins = self.opts.fft_size / 2 + 1;
            let mut power = Vec::with_capacity(n_bins);
            for c in buf.iter().take(n_bins) {
                power.push(c.re * c.re + c.im * c.im);
            }

            // Apply mel filters + log.
            for filt in &self.mel_filters {
                let mut energy = 0.0_f32;
                for (k, w) in filt.weights.iter().enumerate() {
                    let bin = filt.start_bin + k;
                    if bin < power.len() {
                        energy += power[bin] * *w;
                    }
                }
                out.push((energy + 1e-10).ln());
            }
        }

        (out, num_frames)
    }
}

fn hamming_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.54 - 0.46 * ((2.0 * PI * i as f32) / denom).cos())
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    // Slaney / Kaldi-style mel scale: 1127 * ln(1 + hz/700).
    1127.0 * (1.0 + hz / 700.0).ln()
}
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

fn build_mel_filters(
    sample_rate: f32,
    fft_size: usize,
    n_mels: usize,
    low_freq: f32,
    high_freq: f32,
) -> Vec<MelFilter> {
    let n_bins = fft_size / 2 + 1;
    let bin_hz = sample_rate / fft_size as f32;

    let mel_lo = hz_to_mel(low_freq);
    let mel_hi = hz_to_mel(high_freq);
    let mel_step = (mel_hi - mel_lo) / (n_mels + 1) as f32;

    let mut filters = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let left_mel = mel_lo + m as f32 * mel_step;
        let centre_mel = mel_lo + (m + 1) as f32 * mel_step;
        let right_mel = mel_lo + (m + 2) as f32 * mel_step;
        let left_hz = mel_to_hz(left_mel);
        let centre_hz = mel_to_hz(centre_mel);
        let right_hz = mel_to_hz(right_mel);

        let mut start_bin = (left_hz / bin_hz).ceil() as isize;
        let end_bin = (right_hz / bin_hz).floor() as isize;
        if start_bin < 0 {
            start_bin = 0;
        }
        let start_bin = start_bin as usize;
        let end_bin = (end_bin as usize).min(n_bins.saturating_sub(1));

        if end_bin < start_bin {
            // Degenerate filter — emit a single zero weight so the
            // index arithmetic stays consistent.
            filters.push(MelFilter {
                start_bin,
                weights: vec![0.0],
            });
            continue;
        }

        let mut weights = vec![0.0_f32; end_bin - start_bin + 1];
        for (k, w) in weights.iter_mut().enumerate() {
            let bin = start_bin + k;
            let f = bin as f32 * bin_hz;
            *w = if f <= centre_hz {
                (f - left_hz) / (centre_hz - left_hz).max(1e-10)
            } else {
                (right_hz - f) / (right_hz - centre_hz).max(1e-10)
            };
            if !w.is_finite() || *w < 0.0 {
                *w = 0.0;
            }
        }
        filters.push(MelFilter { start_bin, weights });
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opts_paraformer_defaults_match_reference() {
        let o = FbankOpts::paraformer_default();
        assert_eq!(o.frame_len_samples(), 400);
        assert_eq!(o.frame_shift_samples(), 160);
        assert_eq!(o.n_mels, 80);
        assert_eq!(o.fft_size, 512);
        assert!((o.effective_high_freq() - 8000.0).abs() < 1e-3);
    }

    #[test]
    fn fbank_silence_has_correct_shape() {
        // 1 s of silence at 16 kHz with 25/10 ms frames (snip_edges)
        // → floor((16000-400)/160)+1 = 98 frames, 80 mels.
        let f = Fbank::new(FbankOpts::paraformer_default());
        let samples = vec![0.0_f32; 16_000];
        let (out, n) = f.compute(&samples);
        assert_eq!(n, 98);
        assert_eq!(out.len(), 98 * 80);
    }

    #[test]
    fn fbank_short_input_returns_no_frames() {
        let f = Fbank::new(FbankOpts::paraformer_default());
        let samples = vec![0.0_f32; 200]; // shorter than one frame
        let (out, n) = f.compute(&samples);
        assert_eq!(n, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn fbank_low_sine_has_more_low_band_energy_than_high() {
        // Energy from a 200 Hz sine should sit firmly in the bottom
        // mel bands. We compare the mean log-mel of the lowest 10
        // filters against the highest 10.
        let f = Fbank::new(FbankOpts::paraformer_default());
        let fs = 16_000.0_f32;
        let freq = 200.0_f32;
        let samples: Vec<f32> = (0..fs as usize)
            .map(|i| (2.0 * PI * freq * i as f32 / fs).sin() * 0.5)
            .collect();
        let (out, n) = f.compute(&samples);
        assert!(n > 0);
        let n_mels = 80usize;
        let mut low_mean = 0.0_f32;
        let mut high_mean = 0.0_f32;
        for frame in out.chunks_exact(n_mels) {
            low_mean += frame[..10].iter().sum::<f32>() / 10.0;
            high_mean += frame[n_mels - 10..].iter().sum::<f32>() / 10.0;
        }
        low_mean /= n as f32;
        high_mean /= n as f32;
        assert!(
            low_mean > high_mean + 5.0,
            "expected substantially more low-band energy: low={low_mean} high={high_mean}"
        );
    }

    #[test]
    fn hamming_window_endpoints_are_small() {
        let w = hamming_window(400);
        assert!((w[0] - 0.08).abs() < 1e-3);
        assert!((w[399] - 0.08).abs() < 1e-3);
        // Peak at the centre is ~1.0.
        let peak = w.iter().cloned().fold(f32::MIN, f32::max);
        assert!((peak - 1.0).abs() < 1e-3);
    }

    #[test]
    fn mel_filters_cover_full_bandwidth() {
        let filters = build_mel_filters(16_000.0, 512, 80, 20.0, 8000.0);
        assert_eq!(filters.len(), 80);
        // The last filter must touch a bin in the upper half of the
        // spectrum, otherwise our high-frequency calibration is off.
        let last = filters.last().unwrap();
        let last_bin = last.start_bin + last.weights.len() - 1;
        assert!(last_bin > 200, "last bin {last_bin} below expected upper range");
    }
}
