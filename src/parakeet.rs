//! Parakeet TDT ASR adapter — Rust-native ONNX inference.
//!
//! Implements the full pipeline: audio → mel spectrogram → FastConformer encoder
//! → TDT greedy decoding → text.  No Python or external process required.
//!
//! Requires `--features onnx` and the sherpa-onnx-exported INT8 model files:
//! encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx, tokens.txt

use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::Value;
use rustfft::{num_complex::Complex, FftPlanner};
use std::path::Path;
use tokio::sync::mpsc;

use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AsrResult, AudioChunk};

// ---------------------------------------------------------------------------
// Mel Spectrogram
// ---------------------------------------------------------------------------

/// Computes log-mel spectrogram matching NeMo's AudioToMelSpectrogramPreprocessor.
/// Internal computation uses f64 for precision; output is f32 for ONNX.
struct MelSpectrogram {
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    n_mels: usize,
    preemph: f64,
    dither: f64,
    mel_filterbank: Vec<Vec<f64>>,
    window: Vec<f64>,
}

impl MelSpectrogram {
    fn new(sample_rate: u32, n_mels: usize) -> Self {
        let n_fft = 512;
        let hop_length = (sample_rate as f64 * 0.01) as usize;
        let win_length = (sample_rate as f64 * 0.025) as usize;

        let window: Vec<f64> = (0..win_length)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / win_length as f64).cos())
            })
            .collect();

        let mel_filterbank = Self::compute_mel_filterbank(sample_rate, n_fft, n_mels);

        Self {
            n_fft,
            hop_length,
            win_length,
            n_mels,
            preemph: 0.97,
            dither: 0.00001,
            mel_filterbank,
            window,
        }
    }

    fn hz_to_mel(hz: f64) -> f64 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f64) -> f64 {
        700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
    }

    fn compute_mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f64>> {
        let fft_bins = n_fft / 2 + 1;
        let f_min = 0.0_f64;
        let f_max = sample_rate as f64 / 2.0;

        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);

        let mel_points: Vec<f64> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
            .collect();

        let hz_points: Vec<f64> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        let bin_points: Vec<f64> = hz_points
            .iter()
            .map(|&hz| hz * n_fft as f64 / sample_rate as f64)
            .collect();

        let mut filterbank = vec![vec![0.0_f64; fft_bins]; n_mels];

        for m in 0..n_mels {
            let f_left = bin_points[m];
            let f_center = bin_points[m + 1];
            let f_right = bin_points[m + 2];

            for k in 0..fft_bins {
                let kf = k as f64;
                if kf >= f_left && kf <= f_center && f_center > f_left {
                    filterbank[m][k] = (kf - f_left) / (f_center - f_left);
                } else if kf > f_center && kf <= f_right && f_right > f_center {
                    filterbank[m][k] = (f_right - kf) / (f_right - f_center);
                }
            }

            // Slaney normalization: divide by bandwidth (area normalization)
            let bandwidth = hz_points[m + 2] - hz_points[m];
            if bandwidth > 0.0 {
                for k in 0..fft_bins {
                    filterbank[m][k] *= 2.0 / bandwidth;
                }
            }
        }

        filterbank
    }

    /// Compute log-mel spectrogram from f32 audio samples.
    /// All internal computation is f64; output is f32 for ONNX.
    fn compute(&self, audio: &[f32]) -> Array2<f32> {
        let fft_bins = self.n_fft / 2 + 1;

        // Pre-emphasis (f64)
        let mut signal = vec![0.0_f64; audio.len()];
        signal[0] = audio[0] as f64;
        for i in 1..audio.len() {
            signal[i] = audio[i] as f64 - self.preemph * audio[i - 1] as f64;
        }

        // Dithering
        if self.dither > 0.0 {
            for s in signal.iter_mut() {
                *s += self.dither;
            }
        }

        // STFT
        let n_frames = if audio.len() >= self.win_length {
            (audio.len() - self.win_length) / self.hop_length + 1
        } else {
            0
        };

        if n_frames == 0 {
            return Array2::zeros((self.n_mels, 1));
        }

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(self.n_fft);

        let log_guard: f64 = 2.0_f64.powi(-24);
        let mut mel_f64 = vec![vec![0.0_f64; n_frames]; self.n_mels];

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;

            // Apply window and zero-pad to n_fft
            let mut fft_input: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); self.n_fft];
            for i in 0..self.win_length.min(audio.len() - start) {
                fft_input[i] = Complex::new(signal[start + i] * self.window[i], 0.0);
            }

            fft.process(&mut fft_input);

            // Power spectrum
            let power_spec: Vec<f64> = fft_input[..fft_bins]
                .iter()
                .map(|c| c.norm_sqr())
                .collect();

            // Apply mel filterbank + log
            for m in 0..self.n_mels {
                let mut energy = 0.0_f64;
                for k in 0..fft_bins {
                    energy += self.mel_filterbank[m][k] * power_spec[k];
                }
                mel_f64[m][frame_idx] = energy.max(log_guard).ln();
            }
        }

        // Per-feature normalization in f32 (matches PyTorch/NeMo training)
        let mut mel_spec = Array2::zeros((self.n_mels, n_frames));
        for m in 0..self.n_mels {
            // Convert to f32 first, then normalize — matches training distribution
            let row_f32: Vec<f32> = mel_f64[m].iter().map(|&v| v as f32).collect();

            let mut sum = 0.0_f32;
            let mut sum_sq = 0.0_f32;
            for &v in &row_f32 {
                sum += v;
                sum_sq += v * v;
            }
            let mean = sum / n_frames as f32;
            let variance = (sum_sq / n_frames as f32) - mean * mean;
            let std = variance.abs().sqrt().max(1e-5);

            for t in 0..n_frames {
                mel_spec[[m, t]] = (row_f32[t] - mean) / std;
            }
        }

        mel_spec
    }
}

// ---------------------------------------------------------------------------
// TDT Greedy Decoder — supports two model formats
// ---------------------------------------------------------------------------

/// Model format: split (sherpa-onnx) or combined (onnx-asr/grikdotnet)
enum DecoderFormat {
    /// sherpa-onnx: encoder.*.onnx + decoder.*.onnx + joiner.*.onnx
    Split {
        decoder: Session,
        joiner: Session,
    },
    /// onnx-asr: encoder.*.onnx + decoder_joint.*.onnx
    Combined {
        decoder_joint: Session,
    },
}

struct TdtDecoder {
    encoder: Session,
    format: DecoderFormat,
    tokens: Vec<String>,
    blank_id: i32,
    vocab_size: usize,
    num_durations: usize,
}

impl TdtDecoder {
    fn load(model_dir: &Path) -> Result<Self, AsrError> {
        // Auto-detect model files
        let find_file = |patterns: &[&str]| -> Option<std::path::PathBuf> {
            for pat in patterns {
                let p = model_dir.join(pat);
                if p.exists() { return Some(p); }
            }
            None
        };

        // Encoder (required)
        let enc_path = find_file(&[
            "encoder.fp16.onnx", "encoder.int8.onnx", "encoder.onnx",
            "encoder-model.fp16.onnx", "encoder-model.int8.onnx", "encoder-model.onnx",
        ]).ok_or_else(|| AsrError { message: "no encoder ONNX file found".into() })?;

        tracing::info!(path = %enc_path.display(), "loading encoder");
        let encoder = Session::builder()
            .and_then(|mut b| b.commit_from_file(&enc_path))
            .map_err(|e| AsrError { message: format!("load encoder: {e}") })?;

        // Detect format
        let combined_path = find_file(&[
            "decoder_joint.fp16.onnx", "decoder_joint.int8.onnx",
            "decoder_joint.onnx", "decoder_joint-model.fp16.onnx",
            "decoder_joint-model.int8.onnx", "decoder_joint-model.onnx",
        ]);

        let format = if let Some(dj_path) = combined_path {
            tracing::info!(path = %dj_path.display(), "loading combined decoder_joint");
            let decoder_joint = Session::builder()
                .and_then(|mut b| b.commit_from_file(&dj_path))
                .map_err(|e| AsrError { message: format!("load decoder_joint: {e}") })?;
            DecoderFormat::Combined { decoder_joint }
        } else {
            let dec_path = find_file(&["decoder.int8.onnx", "decoder.onnx"])
                .ok_or_else(|| AsrError { message: "no decoder ONNX file found".into() })?;
            let join_path = find_file(&["joiner.int8.onnx", "joiner.onnx"])
                .ok_or_else(|| AsrError { message: "no joiner ONNX file found".into() })?;

            tracing::info!("loading split decoder + joiner");
            let decoder = Session::builder()
                .and_then(|mut b| b.commit_from_file(&dec_path))
                .map_err(|e| AsrError { message: format!("load decoder: {e}") })?;
            let joiner = Session::builder()
                .and_then(|mut b| b.commit_from_file(&join_path))
                .map_err(|e| AsrError { message: format!("load joiner: {e}") })?;
            DecoderFormat::Split { decoder, joiner }
        };

        // Load tokens
        let tokens_path = find_file(&["tokens.txt", "vocab.txt"])
            .ok_or_else(|| AsrError { message: "no tokens/vocab file found".into() })?;
        let tokens_text = std::fs::read_to_string(&tokens_path)
            .map_err(|e| AsrError { message: format!("load tokens: {e}") })?;

        let mut tokens = Vec::new();
        for line in tokens_text.lines() {
            if let Some((tok, _)) = line.rsplit_once(' ') {
                tokens.push(tok.to_string());
            } else if !line.is_empty() {
                tokens.push(line.to_string());
            }
        }

        let vocab_size = tokens.len();
        let blank_id = (vocab_size - 1) as i32;
        let num_durations = 5;

        tracing::info!(vocab_size, blank_id, "Parakeet TDT decoder loaded");

        Ok(Self { encoder, format, tokens, blank_id, vocab_size, num_durations })
    }

    fn transcribe(&mut self, mel: &Array2<f32>) -> Result<String, AsrError> {
        let (n_mels, n_frames) = (mel.shape()[0], mel.shape()[1]);
        let mel_data: Vec<f32> = mel.iter().copied().collect();
        let mel_arr = Array3::from_shape_vec((1, n_mels, n_frames), mel_data)
            .map_err(|e| AsrError { message: format!("mel reshape: {e}") })?;
        let length_arr = Array1::from_vec(vec![n_frames as i64]);

        // Run encoder and immediately extract owned data
        let (enc_data, t_enc) = {
            let enc_out = self.encoder.run(vec![
                ("audio_signal", Value::from_array(mel_arr).map_err(|e| AsrError { message: format!("{e}") })?.into_dyn()),
                ("length", Value::from_array(length_arr).map_err(|e| AsrError { message: format!("{e}") })?.into_dyn()),
            ]).map_err(|e| AsrError { message: format!("encoder: {e}") })?;

            let data: Vec<f32> = enc_out[0].try_extract_array::<f32>()
                .map_err(|e| AsrError { message: format!("enc extract: {e}") })?
                .view().iter().copied().collect();
            let t: usize = enc_out[1].try_extract_array::<i64>()
                .map_err(|e| AsrError { message: format!("enc_len: {e}") })?
                .view().iter().next().copied().unwrap_or(0) as usize;
            (data, t)
        }; // enc_out dropped here, releasing borrow on self.encoder

        let enc_dim = 1024;
        if t_enc == 0 { return Ok(String::new()); }

        // TDT greedy decode
        let mut emitted: Vec<i32> = Vec::new();
        let mut prev = self.blank_id;
        let mut states1 = vec![0.0f32; 2 * 1 * 640];
        let mut states2 = vec![0.0f32; 2 * 1 * 640];
        let mut t = 0usize;

        while t < t_enc {
            for _ in 0..10 {
                let enc_frame: Vec<f32> = (0..enc_dim).map(|d| enc_data[d * t_enc + t]).collect();

                let (logits, new_s1, new_s2) = self.decode_step(
                    &enc_frame, prev, &states1, &states2,
                )?;

                states1 = new_s1;
                states2 = new_s2;

                let tok = logits[..self.vocab_size].iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i32;
                let dur = logits[self.vocab_size..].iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

                if tok == self.blank_id || tok == 0 {
                    t += dur.max(1);
                    break;
                } else {
                    emitted.push(tok);
                    prev = tok;
                }
            }
        }

        tracing::info!(n_tokens = emitted.len(), "TDT decoding complete");

        let text: String = emitted.iter()
            .filter_map(|&id| self.tokens.get(id as usize))
            .map(|t| t.replace('▁', " "))
            .collect::<String>()
            .trim()
            .to_string();

        Ok(text)
    }

    /// Run one decode step, returning (logits, new_states1, new_states2).
    fn decode_step(
        &mut self, enc_frame: &[f32], prev_token: i32,
        states1: &[f32], states2: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), AsrError> {
        let enc_dim = 1024;
        let dec_dim = 640;
        let ef = Array3::from_shape_vec((1, enc_dim, 1), enc_frame.to_vec()).unwrap();
        let tgt = Array2::from_shape_vec((1, 1), vec![prev_token]).unwrap();
        let tl = Array1::from_vec(vec![1_i32]);
        let s1 = Array3::from_shape_vec((2, 1, dec_dim), states1.to_vec()).unwrap();
        let s2 = Array3::from_shape_vec((2, 1, dec_dim), states2.to_vec()).unwrap();

        match &mut self.format {
            DecoderFormat::Combined { decoder_joint } => {
                let out = decoder_joint.run(vec![
                    ("encoder_outputs", Value::from_array(ef).unwrap().into_dyn()),
                    ("targets", Value::from_array(tgt).unwrap().into_dyn()),
                    ("target_length", Value::from_array(tl).unwrap().into_dyn()),
                    ("input_states_1", Value::from_array(s1).unwrap().into_dyn()),
                    ("input_states_2", Value::from_array(s2).unwrap().into_dyn()),
                ]).map_err(|e| AsrError { message: format!("decoder_joint: {e}") })?;

                let logits: Vec<f32> = out[0].try_extract_array::<f32>().unwrap().view().iter().copied().collect();
                let ns1: Vec<f32> = out[2].try_extract_array::<f32>().unwrap().view().iter().copied().collect();
                let ns2: Vec<f32> = out[3].try_extract_array::<f32>().unwrap().view().iter().copied().collect();
                Ok((logits, ns1, ns2))
            }
            DecoderFormat::Split { decoder, joiner } => {
                let dec_out = decoder.run(vec![
                    ("targets", Value::from_array(tgt).unwrap().into_dyn()),
                    ("target_length", Value::from_array(tl).unwrap().into_dyn()),
                    ("states.1", Value::from_array(s1).unwrap().into_dyn()),
                    ("onnx::Slice_3", Value::from_array(s2).unwrap().into_dyn()),
                ]).map_err(|e| AsrError { message: format!("decoder: {e}") })?;

                let dec_vec: Vec<f32> = dec_out[0].try_extract_array::<f32>().unwrap().view().iter().copied().collect();
                let ns1: Vec<f32> = dec_out[2].try_extract_array::<f32>().unwrap().view().iter().copied().collect();
                let ns2: Vec<f32> = dec_out[3].try_extract_array::<f32>().unwrap().view().iter().copied().collect();

                let df = Array3::from_shape_vec((1, dec_dim, 1), dec_vec[..dec_dim].to_vec()).unwrap();
                let join_out = joiner.run(vec![
                    ("encoder_outputs", Value::from_array(ef).unwrap().into_dyn()),
                    ("decoder_outputs", Value::from_array(df).unwrap().into_dyn()),
                ]).map_err(|e| AsrError { message: format!("joiner: {e}") })?;

                let logits: Vec<f32> = join_out[0].try_extract_array::<f32>().unwrap().view().iter().copied().collect();
                Ok((logits, ns1, ns2))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ParakeetAdapter — AsrAdapter implementation
// ---------------------------------------------------------------------------

/// Parakeet TDT 0.6B ASR adapter using ONNX Runtime.
///
/// Runs entirely in Rust: mel spectrogram → FastConformer encoder →
/// TDT greedy decoding.  No Python required.
///
/// # Usage
/// ```no_run
/// use euhadra::parakeet::ParakeetAdapter;
///
/// let asr = ParakeetAdapter::load("/path/to/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
///     .expect("failed to load model");
/// ```
pub struct ParakeetAdapter {
    mel: MelSpectrogram,
    decoder: std::sync::Mutex<TdtDecoder>,
}

impl ParakeetAdapter {
    /// Load from a directory containing encoder/decoder/joiner ONNX files + tokens.txt.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, AsrError> {
        let mel = MelSpectrogram::new(16000, 128);
        let decoder = TdtDecoder::load(model_dir.as_ref())?;
        Ok(Self {
            mel,
            decoder: std::sync::Mutex::new(decoder),
        })
    }
}

#[async_trait]
impl AsrAdapter for ParakeetAdapter {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        result_tx: mpsc::Sender<AsrResult>,
    ) -> Result<(), AsrError> {
        // Accumulate all audio
        let mut all_samples: Vec<f32> = Vec::new();
        while let Some(chunk) = audio_rx.recv().await {
            all_samples.extend(&chunk.samples);
        }

        if all_samples.is_empty() {
            return Err(AsrError { message: "no audio received".into() });
        }

        // Compute mel spectrogram
        let mel = self.mel.compute(&all_samples);
        tracing::info!(
            n_mels = mel.shape()[0],
            n_frames = mel.shape()[1],
            audio_samples = all_samples.len(),
            "mel spectrogram computed"
        );

        // Run TDT decoding (blocking — model inference is CPU-bound)
        let text = {
            let mut dec = self.decoder.lock().unwrap();
            dec.transcribe(&mel)?
        };

        if !text.is_empty() {
            result_tx
                .send(AsrResult {
                    text,
                    is_final: true,
                    confidence: 1.0,
                    timestamp: std::time::Duration::ZERO,
                })
                .await
                .map_err(|e| AsrError { message: format!("send: {e}") })?;
        }

        Ok(())
    }
}
