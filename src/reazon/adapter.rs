//! ReazonSpeech Zipformer transducer ASR adapter (Japanese path).
//!
//! ```text
//! audio f32  →  Kaldi FBANK (80 mel, Povey, snip_edges=false)  [= dolphin front-end]
//!            →  encoder.int8.onnx  (x [N,T,80], x_lens [N]) → encoder_out [N,T',512]
//!            →  greedy modified transducer (one emit per frame)
//!                 decoder(y [N,2]) → decoder_out [N,512]
//!                 joiner(encoder_out[t], decoder_out) → logit [N,V]
//!            →  tokens.txt + icefall byte_decode → text
//! ```
//!
//! Bundle layout (from `scripts/setup_reazon_ja.sh`):
//! `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`.
//!
//! Upstream weights: `reazon-research/reazonspeech-k2-v2` (Apache-2.0).
//! Export consumed here: k2-fsa/sherpa-onnx
//! `sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17`.
//!
//! Stopgap relative to shipping a dedicated sherpa-onnx binding: pure `ort`
//! greedy search matching sherpa-onnx `decoding_method=greedy_search`
//! (modified transducer — at most one non-blank per encoder frame).

use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::paraformer::fbank::{Fbank, FbankOpts};
use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AudioChunk, Transcript};

use super::vocab::Vocab;

const BLANK_ID: i64 = 0;
const CONTEXT_SIZE: usize = 2;
const INTRA_THREADS: usize = 1;

#[derive(Debug, Clone)]
pub struct ReazonConfig {
    pub encoder_file: String,
    pub decoder_file: String,
    pub joiner_file: String,
}

impl Default for ReazonConfig {
    fn default() -> Self {
        Self {
            encoder_file: "encoder.int8.onnx".into(),
            decoder_file: "decoder.int8.onnx".into(),
            joiner_file: "joiner.int8.onnx".into(),
        }
    }
}

pub struct ReazonAdapter {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    joiner: Mutex<Session>,
    fbank: Fbank,
    vocab: Vocab,
    model_dir: PathBuf,
}

impl ReazonAdapter {
    /// Load a bundle laid out by `scripts/setup_reazon_ja.sh`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, AsrError> {
        Self::load_with_config(model_dir, ReazonConfig::default())
    }

    pub fn load_with_config(
        model_dir: impl AsRef<Path>,
        cfg: ReazonConfig,
    ) -> Result<Self, AsrError> {
        let dir = model_dir.as_ref();
        let vocab = Vocab::load(dir.join("tokens.txt"))?;

        let encoder = load_session(&dir.join(&cfg.encoder_file), "reazon encoder")?;
        let decoder = load_session(&dir.join(&cfg.decoder_file), "reazon decoder")?;
        let joiner = load_session(&dir.join(&cfg.joiner_file), "reazon joiner")?;

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            joiner: Mutex::new(joiner),
            fbank: Fbank::new(FbankOpts::dolphin_default()),
            vocab,
            model_dir: dir.to_path_buf(),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<String, AsrError> {
        if samples.is_empty() {
            return Err(AsrError::NoAudio);
        }

        let (feats, n_frames) = self.fbank.compute(samples);
        if n_frames == 0 {
            return Err(AsrError::Inference(format!(
                "reazon: audio too short for one FBANK frame ({} samples)",
                samples.len()
            )));
        }

        let dim = self.fbank.n_mels();
        let x: Array3<f32> = Array3::from_shape_vec((1, n_frames, dim), feats).map_err(|e| {
            AsrError::Inference(format!("reazon x tensor shape: {e}"))
        })?;
        let x_lens: Array1<i64> = Array1::from(vec![n_frames as i64]);

        let x_val = Value::from_array(x)
            .map_err(|e| AsrError::Inference(format!("reazon x Value: {e}")))?;
        let x_lens_val = Value::from_array(x_lens)
            .map_err(|e| AsrError::Inference(format!("reazon x_lens Value: {e}")))?;

        let enc_out = {
            let mut enc = self
                .encoder
                .lock()
                .map_err(|e| AsrError::Inference(format!("reazon encoder lock: {e}")))?;
            let outputs = enc
                .run(vec![
                    ("x", x_val.into_dyn()),
                    ("x_lens", x_lens_val.into_dyn()),
                ])
                .map_err(|e| AsrError::Inference(format!("reazon encoder run: {e}")))?;
            let arr = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| AsrError::Inference(format!("reazon encoder_out: {e}")))?;
            // [1, T', 512]
            let shape = arr.shape().to_vec();
            if shape.len() != 3 || shape[0] != 1 {
                return Err(AsrError::Inference(format!(
                    "reazon {}: unexpected encoder_out shape {shape:?}",
                    self.model_dir.display()
                )));
            }
            Array3::<f32>::from_shape_vec((shape[0], shape[1], shape[2]), arr.iter().copied().collect())
                .map_err(|e| AsrError::Inference(format!("reazon encoder_out copy: {e}")))?
        };

        let t_enc = enc_out.shape()[1];
        let mut hyp: Vec<i64> = vec![BLANK_ID; CONTEXT_SIZE];
        let mut decoder_out = self.run_decoder(&hyp[hyp.len() - CONTEXT_SIZE..])?;
        let mut tokens: Vec<i64> = Vec::new();

        for t in 0..t_enc {
            let enc_t = enc_out.slice(ndarray::s![0, t, ..]).to_owned();
            let enc_row: Array2<f32> = enc_t.insert_axis(ndarray::Axis(0));
            let logit = self.run_joiner(&enc_row, &decoder_out)?;
            let y = argmax(&logit) as i64;
            if y != BLANK_ID {
                tokens.push(y);
                hyp.push(y);
                decoder_out = self.run_decoder(&hyp[hyp.len() - CONTEXT_SIZE..])?;
            }
        }

        Ok(self.vocab.decode_hyp(&tokens, BLANK_ID))
    }

    fn run_decoder(&self, context: &[i64]) -> Result<Array2<f32>, AsrError> {
        debug_assert_eq!(context.len(), CONTEXT_SIZE);
        let y: Array2<i64> = Array2::from_shape_vec((1, CONTEXT_SIZE), context.to_vec())
            .map_err(|e| AsrError::Inference(format!("reazon decoder y shape: {e}")))?;
        let y_val = Value::from_array(y)
            .map_err(|e| AsrError::Inference(format!("reazon decoder y Value: {e}")))?;
        let mut dec = self
            .decoder
            .lock()
            .map_err(|e| AsrError::Inference(format!("reazon decoder lock: {e}")))?;
        let outputs = dec
            .run(vec![("y", y_val.into_dyn())])
            .map_err(|e| AsrError::Inference(format!("reazon decoder run: {e}")))?;
        let arr = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| AsrError::Inference(format!("reazon decoder_out: {e}")))?;
        let shape = arr.shape().to_vec();
        if shape.len() != 2 || shape[0] != 1 {
            return Err(AsrError::Inference(format!(
                "reazon unexpected decoder_out shape {shape:?}"
            )));
        }
        Array2::from_shape_vec((shape[0], shape[1]), arr.iter().copied().collect())
            .map_err(|e| AsrError::Inference(format!("reazon decoder_out copy: {e}")))
    }

    fn run_joiner(&self, enc: &Array2<f32>, dec: &Array2<f32>) -> Result<Vec<f32>, AsrError> {
        let enc_val = Value::from_array(enc.clone())
            .map_err(|e| AsrError::Inference(format!("reazon joiner enc Value: {e}")))?;
        let dec_val = Value::from_array(dec.clone())
            .map_err(|e| AsrError::Inference(format!("reazon joiner dec Value: {e}")))?;
        let mut joi = self
            .joiner
            .lock()
            .map_err(|e| AsrError::Inference(format!("reazon joiner lock: {e}")))?;
        let outputs = joi
            .run(vec![
                ("encoder_out", enc_val.into_dyn()),
                ("decoder_out", dec_val.into_dyn()),
            ])
            .map_err(|e| AsrError::Inference(format!("reazon joiner run: {e}")))?;
        let arr = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| AsrError::Inference(format!("reazon logit: {e}")))?;
        Ok(arr.iter().copied().collect())
    }
}

fn load_session(path: &Path, label: &str) -> Result<Session, AsrError> {
    Session::builder()
        .map_err(|e| AsrError::ModelLoad(format!("{label} session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| AsrError::Inference(format!("{label} optimization: {e}")))?
        .with_intra_threads(INTRA_THREADS)
        .map_err(|e| AsrError::Inference(format!("{label} intra_threads: {e}")))?
        .with_inter_threads(1)
        .map_err(|e| AsrError::Inference(format!("{label} inter_threads: {e}")))?
        .commit_from_file(path)
        .map_err(|e| {
            AsrError::ModelLoad(format!("{label} load failed at {}: {e}", path.display()))
        })
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

#[async_trait]
impl AsrAdapter for ReazonAdapter {
    async fn transcribe(&self, audio: &[AudioChunk]) -> Result<Transcript, AsrError> {
        let all_samples = AudioChunk::concat(audio);
        if all_samples.is_empty() {
            return Err(AsrError::NoAudio);
        }
        tracing::info!(
            audio_samples = all_samples.len(),
            "transcribing with reazon-zipformer"
        );
        let text = self.transcribe_samples(&all_samples)?;
        Ok(Transcript::new(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_points_at_int8_bundle() {
        let cfg = ReazonConfig::default();
        assert_eq!(cfg.encoder_file, "encoder.int8.onnx");
        assert_eq!(cfg.decoder_file, "decoder.int8.onnx");
        assert_eq!(cfg.joiner_file, "joiner.int8.onnx");
    }

    #[test]
    fn missing_bundle_fails_at_load() {
        let Err(err) = ReazonAdapter::load("/nonexistent/reazon/bundle") else {
            panic!("loading a nonexistent bundle should fail");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("tokens") || msg.contains("load"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn smoke_transcribe_when_bundle_present() {
        let dir = std::path::Path::new("vendor/reazon_ja");
        if !dir.join("encoder.int8.onnx").is_file() {
            eprintln!("skip: vendor/reazon_ja not present");
            return;
        }
        let wav = std::path::Path::new("data/fleurs_subset/ja/audio/1731.wav");
        if !wav.is_file() {
            eprintln!("skip: fleurs 1731.wav not present");
            return;
        }
        let adapter = ReazonAdapter::load(dir).expect("load reazon");
        let samples = read_pcm16_wav(wav);
        let text = adapter.transcribe_samples(&samples).expect("decode");
        assert!(
            text.contains("ヨット") || text.contains("湖"),
            "unexpected hyp: {text}"
        );
    }

    #[test]
    fn smoke_longer_utterance_1893() {
        let dir = std::path::Path::new("vendor/reazon_ja");
        let wav = std::path::Path::new("data/fleurs_subset/ja/audio/1893.wav");
        if !dir.join("encoder.int8.onnx").is_file() || !wav.is_file() {
            return;
        }
        let adapter = ReazonAdapter::load(dir).unwrap();
        let chunk = crate::whisper_local::read_wav(wav).expect("read_wav");
        eprintln!("samples={} sr={} ch={}", chunk.samples.len(), chunk.sample_rate, chunk.channels);
        let text = adapter.transcribe_samples(&chunk.samples).unwrap();
        eprintln!("TEXT={text:?}");
        assert!(!text.is_empty(), "empty hyp for 1893 via read_wav");
    }

    fn read_pcm16_wav(path: &Path) -> Vec<f32> {
        use std::io::Read;
        let mut buf = Vec::new();
        std::fs::File::open(path)
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        assert!(&buf[0..4] == b"RIFF" && &buf[8..12] == b"WAVE");
        let mut pos = 12;
        let mut data: Option<Vec<f32>> = None;
        while pos + 8 <= buf.len() {
            let id = &buf[pos..pos + 4];
            let size = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap()) as usize;
            let start = pos + 8;
            let end = (start + size).min(buf.len());
            if id == b"data" {
                data = Some(
                    buf[start..end]
                        .chunks_exact(2)
                        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                        .collect(),
                );
            }
            pos = end + (size % 2);
        }
        data.expect("data chunk")
    }
}
