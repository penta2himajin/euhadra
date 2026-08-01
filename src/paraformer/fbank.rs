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

use rustfft::{num_complex::Complex32, FftPlanner};
use std::f32::consts::PI;

/// Analysis window shape.
///
/// Kaldi's own default is `Povey`; FunASR overrides it to `Hamming`,
/// which is why the two front-ends here disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    Hamming,
    Povey,
}

#[derive(Debug, Clone)]
pub struct FbankOpts {
    pub sample_rate: u32,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    pub n_mels: usize,
    pub fft_size: usize,
    pub low_freq: f32,
    /// Upper band edge, in Kaldi's convention: `> 0` is a literal
    /// frequency, `0` means Nyquist, and **negative means an offset
    /// below Nyquist** (`-400` on a 16 kHz model is 7600 Hz).
    pub high_freq: f32,
    pub preemph_coeff: f32,
    pub window: WindowType,
    /// `true` keeps only whole frames inside the signal; `false` centres
    /// the frames and reflects the signal at both edges, which yields
    /// `round(num_samples / shift)` frames instead of
    /// `floor((num_samples - len) / shift) + 1`.
    pub snip_edges: bool,
    /// How a mel energy is kept away from `log(0)`.
    ///
    /// The two front-ends here disagree, and only in the silent bands —
    /// which is exactly where it is easy not to notice. Kaldi clamps to
    /// `FLT_EPSILON`, flooring quiet bins at a shared `ln(ε) ≈ -15.94`;
    /// FunASR adds `1e-10` instead, so its quiet bins keep sliding down
    /// past -20. Feeding one model the other's floor changes every
    /// high-frequency bin of near-silence.
    pub log_floor: LogFloor,
}

/// See [`FbankOpts::log_floor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogFloor {
    /// `log(max(energy, floor))` — Kaldi / kaldi-native-fbank.
    Clamp(f32),
    /// `log(energy + offset)` — FunASR.
    Offset(f32),
}

impl LogFloor {
    fn apply(self, energy: f32) -> f32 {
        match self {
            LogFloor::Clamp(floor) => energy.max(floor).ln(),
            LogFloor::Offset(offset) => (energy + offset).ln(),
        }
    }
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
            window: WindowType::Hamming,
            snip_edges: true,
            log_floor: LogFloor::Offset(1e-10),
        }
    }

    /// What sherpa-onnx feeds the Dolphin CTC graph.
    ///
    /// `FeatureExtractorConfig` overrides only `low_freq`, `high_freq`,
    /// `dither` and `snip_edges`; everything else is a
    /// kaldi-native-fbank default, which is where the Povey window
    /// comes from. Pinned against the reference implementation by
    /// `tests/fixtures/dolphin_fbank_golden.json` — see
    /// `scripts/gen_dolphin_fbank_golden.py`.
    pub fn dolphin_default() -> Self {
        Self {
            high_freq: -400.0,
            window: WindowType::Povey,
            snip_edges: false,
            log_floor: LogFloor::Clamp(f32::EPSILON),
            ..Self::paraformer_default()
        }
    }

    pub fn frame_len_samples(&self) -> usize {
        ((self.sample_rate as f32) * self.frame_length_ms / 1000.0).round() as usize
    }
    pub fn frame_shift_samples(&self) -> usize {
        ((self.sample_rate as f32) * self.frame_shift_ms / 1000.0).round() as usize
    }
    pub fn effective_high_freq(&self) -> f32 {
        let nyquist = (self.sample_rate as f32) / 2.0;
        if self.high_freq > 0.0 {
            self.high_freq
        } else {
            // Kaldi treats a negative high_freq as an offset below
            // Nyquist, so `-400` at 16 kHz is 7600 Hz. Zero stays
            // "the whole band".
            nyquist + self.high_freq
        }
    }

    /// Frame count for `num_samples`, in Kaldi's two conventions.
    pub fn num_frames(&self, num_samples: usize) -> usize {
        let shift = self.frame_shift_samples();
        if self.snip_edges {
            let len = self.frame_len_samples();
            if num_samples < len {
                0
            } else {
                (num_samples - len) / shift + 1
            }
        } else if num_samples == 0 {
            0
        } else {
            // Round-half-up division: Kaldi's
            // `(num_samples + shift / 2) / shift`.
            (num_samples + shift / 2) / shift
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
        let window = match opts.window {
            WindowType::Hamming => hamming_window(frame_len),
            WindowType::Povey => povey_window(frame_len),
        };
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

        let num_frames = self.opts.num_frames(samples.len());
        if num_frames == 0 {
            return (Vec::new(), 0);
        }

        let n_mels = self.opts.n_mels;
        let mut out = Vec::with_capacity(num_frames * n_mels);

        let mut buf = vec![Complex32::new(0.0, 0.0); self.opts.fft_size];
        let mut frame = vec![0.0_f32; frame_len];

        for f in 0..num_frames {
            if self.opts.snip_edges {
                let start = f * frame_shift;
                frame.copy_from_slice(&samples[start..start + frame_len]);
            } else {
                // Frames are centred on `f * shift` rather than starting
                // there, so the first and last ones reach outside the
                // signal; Kaldi fills that by reflecting about the
                // boundary (`feature-window.cc`, `ExtractWindow`).
                let start = f as isize * frame_shift as isize
                    - (frame_len as isize - frame_shift as isize) / 2;
                for (i, slot) in frame.iter_mut().enumerate() {
                    *slot = samples[reflect(start + i as isize, samples.len())];
                }
            }

            // Remove DC offset — Kaldi / kaldi_native_fbank's
            // FrameExtractionOptions::remove_dc_offset defaults to true
            // and is applied per-frame BEFORE pre-emphasis. Skipping it
            // produces ~2× CER on Paraformer-large.
            let mean = frame.iter().sum::<f32>() / frame_len as f32;
            for s in frame.iter_mut() {
                *s -= mean;
            }

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
                out.push(self.opts.log_floor.apply(energy));
            }
        }

        (out, num_frames)
    }
}

/// Fold an out-of-range index back inside `[0, len)` by mirroring about
/// each boundary, repeatedly — a window can be wider than the signal.
fn reflect(mut i: isize, len: usize) -> usize {
    debug_assert!(len > 0);
    let n = len as isize;
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= n {
            i = 2 * n - 1 - i;
        } else {
            return i as usize;
        }
    }
}

fn povey_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    // Kaldi: pow(0.5 - 0.5 * cos(2*pi*i / (N-1)), 0.85) — a Hann window
    // raised to 0.85, so it reaches exactly zero at both endpoints.
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| (0.5 - 0.5 * ((2.0 * PI * i as f32) / denom).cos()).powf(0.85))
        .collect()
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

        // Kaldi / kaldi_native_fbank computes triangles in MEL space,
        // not Hz space — peak weight is at centre_mel and the slopes
        // are linear in mel. HTK does the opposite (Hz-space triangles)
        // which produces a different filterbank and hurts CER on
        // Mandarin-trained models.
        let mut start_bin: Option<usize> = None;
        let mut end_bin: usize = 0;
        let mut weights_full: Vec<f32> = Vec::new();
        for bin in 0..n_bins {
            let freq = bin as f32 * bin_hz;
            let mel = hz_to_mel(freq);
            if mel <= left_mel || mel >= right_mel {
                continue;
            }
            let w = if mel <= centre_mel {
                (mel - left_mel) / (centre_mel - left_mel).max(1e-10)
            } else {
                (right_mel - mel) / (right_mel - centre_mel).max(1e-10)
            };
            if !w.is_finite() || w <= 0.0 {
                continue;
            }
            if start_bin.is_none() {
                start_bin = Some(bin);
            }
            end_bin = bin;
            // Pad zeros for any gap between previously-emitted bins
            // and this one (extremely narrow filters at low mel).
            let s = start_bin.unwrap();
            while weights_full.len() < bin - s {
                weights_full.push(0.0);
            }
            weights_full.push(w);
        }

        match start_bin {
            Some(s) => {
                debug_assert_eq!(weights_full.len(), end_bin - s + 1);
                filters.push(MelFilter {
                    start_bin: s,
                    weights: weights_full,
                });
            }
            None => {
                // Degenerate filter — emit a single zero weight so the
                // index arithmetic stays consistent.
                filters.push(MelFilter {
                    start_bin: 0,
                    weights: vec![0.0],
                });
            }
        }
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
    fn fbank_dc_offset_is_removed_per_frame() {
        // A constant non-zero signal should look identical to silence
        // once we strip the DC component. Without `remove_dc_offset`
        // the FBANK of a 0.5 constant blows up by ~30 dB across all
        // bands — that's the bug we just fixed.
        let f = Fbank::new(FbankOpts::paraformer_default());
        let zeros = vec![0.0_f32; 16_000];
        let constant = vec![0.5_f32; 16_000];
        let (a, _) = f.compute(&zeros);
        let (b, _) = f.compute(&constant);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() < 1e-3,
                "dc removal failed: zeros={x} constant={y}"
            );
        }
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
    fn dolphin_defaults_differ_from_paraformer_where_they_should() {
        let d = FbankOpts::dolphin_default();
        assert_eq!(d.window, WindowType::Povey);
        assert!(!d.snip_edges);
        // -400 is "Nyquist minus 400 Hz", not "minus 400 Hz literally".
        assert!((d.effective_high_freq() - 7600.0).abs() < 1e-3);
        // Everything else must still track the Paraformer front-end.
        let p = FbankOpts::paraformer_default();
        assert_eq!(d.n_mels, p.n_mels);
        assert_eq!(d.fft_size, p.fft_size);
        assert_eq!(d.frame_len_samples(), p.frame_len_samples());
        assert_eq!(d.frame_shift_samples(), p.frame_shift_samples());
    }

    #[test]
    fn snip_edges_false_rounds_the_frame_count() {
        // 4000 samples at shift 160: (4000 + 80) / 160 = 25, against
        // snip_edges' floor((4000 - 400) / 160) + 1 = 23.
        assert_eq!(FbankOpts::dolphin_default().num_frames(4_000), 25);
        assert_eq!(FbankOpts::paraformer_default().num_frames(4_000), 23);
        // Shorter than one frame still yields frames when edges are
        // reflected rather than snipped.
        assert_eq!(FbankOpts::dolphin_default().num_frames(200), 1);
        assert_eq!(FbankOpts::paraformer_default().num_frames(200), 0);
        assert_eq!(FbankOpts::dolphin_default().num_frames(0), 0);
    }

    #[test]
    fn reflect_mirrors_about_both_boundaries() {
        assert_eq!(reflect(-1, 10), 0);
        assert_eq!(reflect(-3, 10), 2);
        assert_eq!(reflect(0, 10), 0);
        assert_eq!(reflect(9, 10), 9);
        assert_eq!(reflect(10, 10), 9);
        assert_eq!(reflect(12, 10), 7);
        // A window wider than the signal needs more than one fold.
        assert_eq!(reflect(-15, 10), 5);
        for i in -40..40 {
            assert!(reflect(i, 7) < 7, "escaped range at {i}");
        }
    }

    #[test]
    fn povey_window_reaches_zero_at_the_endpoints() {
        let w = povey_window(400);
        assert!(w[0].abs() < 1e-6, "{}", w[0]);
        assert!(w[399].abs() < 1e-6, "{}", w[399]);
        let peak = w.iter().cloned().fold(f32::MIN, f32::max);
        assert!((peak - 1.0).abs() < 1e-3);
        // Hann raised to 0.85 sits above plain Hann away from the peak.
        let hann = 0.5 - 0.5 * ((2.0 * PI * 100.0) / 399.0).cos();
        assert!(w[100] > hann, "{} vs {hann}", w[100]);
    }

    /// Rebuild the fixture's two-tone waveform.
    ///
    /// In f64, like the generator: by sample 4000 the 1750 Hz phase is
    /// ~2750 rad, and an f32 `sin()` there has lost enough precision to
    /// move quiet mel bins by whole nats. That would be a defect in the
    /// fixture, not in the front-end.
    fn golden_tones(golden: &serde_json::Value) -> Vec<f32> {
        let fs = golden["sample_rate"].as_f64().unwrap();
        let n = golden["num_samples"].as_u64().unwrap() as usize;
        let tones: Vec<(f64, f64)> = golden["tones"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                let t = t.as_array().unwrap();
                (t[0].as_f64().unwrap(), t[1].as_f64().unwrap())
            })
            .collect();
        (0..n)
            .map(|i| {
                tones
                    .iter()
                    .map(|(hz, amp)| amp * (2.0 * std::f64::consts::PI * hz * i as f64 / fs).sin())
                    .sum::<f64>() as f32
            })
            .collect()
    }

    /// Rebuild the fixture's wideband waveform.
    ///
    /// Integer LCG, so these samples are bit-identical to the
    /// generator's rather than merely close — otherwise a mismatch here
    /// would be ambiguous between "the front-end is wrong" and "the
    /// waveform is".
    fn golden_noise(golden: &serde_json::Value) -> Vec<f32> {
        let n = golden["num_samples"].as_u64().unwrap() as usize;
        let lcg = &golden["lcg"];
        let (mul, add, modulus) = (
            lcg["mul"].as_u64().unwrap(),
            lcg["add"].as_u64().unwrap(),
            lcg["mod"].as_u64().unwrap(),
        );
        let mut state = lcg["seed"].as_u64().unwrap();
        (0..n)
            .map(|_| {
                state = (mul * state + add) % modulus;
                (state as f64 / modulus as f64 - 0.5) as f32
            })
            .collect()
    }

    #[test]
    fn dolphin_front_end_matches_the_kaldi_reference() {
        // The whole point of the Dolphin config: a front-end that is
        // subtly wrong still produces correctly-shaped features and a
        // plausible transcript, so it is pinned against
        // kaldi-native-fbank rather than against a reading of the docs.
        // Regenerate with scripts/gen_dolphin_fbank_golden.py.
        let raw = include_str!("../../tests/fixtures/dolphin_fbank_golden.json");
        let golden: serde_json::Value = serde_json::from_str(raw).expect("fixture parses");

        let fbank = Fbank::new(FbankOpts::dolphin_default());
        let n_mels = 80usize;

        for (case, samples) in [
            ("tones", golden_tones(&golden)),
            ("noise", golden_noise(&golden)),
        ] {
            let expected = golden["cases"][case]["frames"].as_array().unwrap();
            let (out, n) = fbank.compute(&samples);
            assert_eq!(n, expected.len(), "{case}: frame count");

            let mut worst = 0.0_f32;
            for (f, (row, want)) in out.chunks_exact(n_mels).zip(expected).enumerate() {
                let want = want.as_array().unwrap();
                for (b, (got, w)) in row.iter().zip(want).enumerate() {
                    let w = w.as_f64().unwrap() as f32;
                    let diff = (got - w).abs();
                    worst = worst.max(diff);
                    assert!(diff < 2e-3, "{case} frame {f} bin {b}: got {got}, want {w}");
                }
            }
            // rustfft and Kaldi's real-FFT accumulate differently, so
            // exact equality is not available; this bound is ~4 orders
            // below the per-bin dynamic range, far tighter than any
            // window, framing or filterbank mistake could survive.
            assert!(worst < 2e-3, "{case}: worst deviation {worst}");
        }
    }

    #[test]
    fn mel_filters_cover_full_bandwidth() {
        let filters = build_mel_filters(16_000.0, 512, 80, 20.0, 8000.0);
        assert_eq!(filters.len(), 80);
        // The last filter must touch a bin in the upper half of the
        // spectrum, otherwise our high-frequency calibration is off.
        let last = filters.last().unwrap();
        let last_bin = last.start_bin + last.weights.len() - 1;
        assert!(
            last_bin > 200,
            "last bin {last_bin} below expected upper range"
        );
    }
}
