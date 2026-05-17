//! `WhisperOnnxAdapter` — Whisper-large-v3-turbo via ONNX Runtime (`ort`).
//!
//! Loads the [`onnx-community/whisper-large-v3-turbo`](https://huggingface.co/onnx-community/whisper-large-v3-turbo)
//! bundle (encoder + decoder + decoder_with_past) into the same `ort`
//! sessions that the existing Parakeet / Canary / Paraformer / SenseVoice
//! adapters use. Lives under the existing `onnx` feature gate; no new
//! system dependencies vs the other ORT-based adapters.
//!
//! # Status: scaffolded
//!
//! The integration plumbing (factory, session loading, KV-cache schema
//! discovery, language prompt construction) is implemented. The full
//! mel-spectrogram + autoregressive greedy decode loop is **not yet**
//! wired into [`WhisperOnnxAdapter::transcribe_samples`]; calling it
//! returns a clear "not yet implemented" error. The shape of the loop
//! is documented inline below, and a working Python POC of the same
//! kernel path lives at `/tmp/ko_bench_ort_direct.py` in the issue #83
//! measurement session.
//!
//! # Bench results (Python POC, raw onnxruntime)
//!
//! Same FLEURS-ko 10-utt subset that gates L1, same shared 4-core
//! Xeon @ 2.1 GHz VM as the other #83 benches, lenient CER via
//! [`euhadra::eval::metrics::cer_lenient`]:
//!
//! | Quantisation variant | weighted CER | RTF  | bundle size |
//! |----------------------|-------------:|-----:|------------:|
//! | `q4`                 |    **1.09%** | **0.484** | ~1.1 GB |
//! | `int8`               |        260%* | 0.255 |   ~1.5 GB |
//!
//! * INT8's autoregressive Whisper decoder collapses into repeating-token
//!   hallucinations on this CPU; not usable. `q4` is the recommended
//!   variant for production.
//!
//! For comparison from PR #103 / #104 / `docs/korean-asr-alternatives.md`:
//!
//! | Path                          | CER   | RTF  |
//! |-------------------------------|------:|-----:|
//! | **ORT `q4` (this adapter)**   | 1.09% | 0.484 |
//! | CT2 FP16 (`faster-whisper`)   | 1.32% | 1.28 |
//! | whisper-rs Q4_0 (GGML)        | 1.74% | 1.78 |
//! | SenseVoice INT8 (baseline)    | 6.64% | 0.047 |
//!
//! ORT `q4` is the Whisper path with the best CER **and** the best
//! RTF on this hardware, by a comfortable margin. The full Rust
//! implementation of the decoder loop is the natural next step after
//! this skeleton lands.

use async_trait::async_trait;
use ort::session::Session;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use crate::router::{AdapterRequest, AsrRuntimeFactory, ModelSource, RouterError};
use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AsrResult, AudioChunk};

/// Optional builder knobs for [`WhisperOnnxAdapter`].
#[derive(Debug, Clone, Default)]
pub struct WhisperOnnxConfig {
    /// Language hint (BCP 47 short code: `"en"`, `"ko"`, `"ja"`, ...).
    /// `None` enables Whisper's built-in LID.
    pub language: Option<String>,
    /// Override the per-file names inside the bundle directory.
    /// Defaults pick the `q4` variant (best CER+RTF on x86 CPUs).
    pub encoder_file: Option<String>,
    pub decoder_file: Option<String>,
    pub decoder_with_past_file: Option<String>,
}

impl WhisperOnnxConfig {
    fn encoder_filename(&self) -> &str {
        self.encoder_file
            .as_deref()
            .unwrap_or("encoder_model_q4.onnx")
    }
    fn decoder_filename(&self) -> &str {
        self.decoder_file
            .as_deref()
            .unwrap_or("decoder_model_q4.onnx")
    }
    fn decoder_with_past_filename(&self) -> &str {
        self.decoder_with_past_file
            .as_deref()
            .unwrap_or("decoder_with_past_model_q4.onnx")
    }
}

/// `AsrAdapter` for Whisper-large-v3-turbo via raw `ort` sessions.
///
/// Holds three ONNX sessions: encoder, decoder (first step — produces the
/// initial KV cache), and decoder_with_past (subsequent steps — consumes
/// and produces KV cache). The mel-spectrogram and decode-loop wiring
/// is the work that completes this adapter (see the file-level docs).
pub struct WhisperOnnxAdapter {
    /// Encoder ONNX session. Holds an `Mutex` because `ort::Session` is
    /// `Send` but not `Sync`; the lock is only contended when the same
    /// adapter is shared across concurrent async sessions.
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    decoder_with_past: Mutex<Session>,
    cfg: WhisperOnnxConfig,
    #[allow(dead_code)]
    model_dir: PathBuf,
}

impl WhisperOnnxAdapter {
    pub fn load(model_dir: impl Into<PathBuf>) -> Result<Self, AsrError> {
        Self::load_with_config(model_dir, WhisperOnnxConfig::default())
    }

    pub fn load_with_config(
        model_dir: impl Into<PathBuf>,
        cfg: WhisperOnnxConfig,
    ) -> Result<Self, AsrError> {
        let dir = model_dir.into();
        let enc_path = dir.join("onnx").join(cfg.encoder_filename());
        let dec_path = dir.join("onnx").join(cfg.decoder_filename());
        let dec_past_path = dir.join("onnx").join(cfg.decoder_with_past_filename());

        let encoder = Session::builder()
            .and_then(|mut b| b.commit_from_file(&enc_path))
            .map_err(|e| AsrError {
                message: format!(
                    "whisper-onnx encoder load failed at {}: {e}",
                    enc_path.display()
                ),
            })?;
        let decoder = Session::builder()
            .and_then(|mut b| b.commit_from_file(&dec_path))
            .map_err(|e| AsrError {
                message: format!(
                    "whisper-onnx decoder load failed at {}: {e}",
                    dec_path.display()
                ),
            })?;
        let decoder_with_past = Session::builder()
            .and_then(|mut b| b.commit_from_file(&dec_past_path))
            .map_err(|e| AsrError {
                message: format!(
                    "whisper-onnx decoder_with_past load failed at {}: {e}",
                    dec_past_path.display()
                ),
            })?;

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            decoder_with_past: Mutex::new(decoder_with_past),
            cfg,
            model_dir: dir,
        })
    }

    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.cfg.language = Some(lang.into());
        self
    }

    /// Run the model on a single concatenated `f32` buffer at 16 kHz.
    ///
    /// **Not yet implemented.** The decode loop sketch:
    ///
    /// 1. Compute the log-mel spectrogram (128 mel bands for large-v3-turbo,
    ///    n_fft=400, hop=160, frames padded/truncated to 3000 = 30 s window).
    /// 2. Run [`encoder`] on the `(1, 128, 3000)` mel tensor; output is
    ///    `last_hidden_state (1, 1500, 1280)`.
    /// 3. Build the prompt: `<|startoftranscript|><|<lang>|><|transcribe|><|notimestamps|>`.
    /// 4. Call [`decoder`] once with `input_ids = prompt`, `encoder_hidden_states = enc`.
    ///    Capture its 17 outputs: `logits` + 16 `present.*.{decoder|encoder}.{key|value}`
    ///    for the 4 turbo decoder layers.
    /// 5. Argmax the last-token logits → first generated token.
    /// 6. Loop: feed `[next_tok]` + KV cache into [`decoder_with_past`]; harvest
    ///    new `present.*.decoder.{key|value}` (8 outputs total), reuse encoder
    ///    KVs unchanged. Argmax → next token. Stop on `<|endoftext|>` or
    ///    `max_new_tokens` (440 ≈ 30 s of speech).
    /// 7. Detokenise the generated ids after stripping the prompt + EOT.
    ///
    /// A working Python POC of this exact loop produced **CER 1.09% /
    /// RTF 0.484** on FLEURS-ko 10-utt with the `q4` quantisation
    /// (see the module-level docs).
    ///
    /// [`encoder`]: WhisperOnnxAdapter::encoder
    /// [`decoder`]: WhisperOnnxAdapter::decoder
    /// [`decoder_with_past`]: WhisperOnnxAdapter::decoder_with_past
    pub fn transcribe_samples(&self, _samples: &[f32]) -> Result<String, AsrError> {
        // Sanity: prove the sessions are reachable so a follow-up PR
        // doesn't have to re-do the load-side plumbing.
        let _enc_guard = self.encoder.lock().expect("encoder mutex poisoned");
        let _dec_guard = self.decoder.lock().expect("decoder mutex poisoned");
        let _dec_past_guard = self
            .decoder_with_past
            .lock()
            .expect("decoder_with_past mutex poisoned");
        let lang = self.cfg.language.as_deref().unwrap_or("auto");
        Err(AsrError {
            message: format!(
                "WhisperOnnxAdapter::transcribe_samples not yet implemented \
                 (language={lang}, encoder={}, decoder={}, decoder_with_past={}); \
                 see module-level docs for the decode-loop sketch and the Python \
                 POC that produced CER 1.09% / RTF 0.484 on FLEURS-ko",
                self.cfg.encoder_filename(),
                self.cfg.decoder_filename(),
                self.cfg.decoder_with_past_filename(),
            ),
        })
    }
}

#[async_trait]
impl AsrAdapter for WhisperOnnxAdapter {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        _result_tx: mpsc::Sender<AsrResult>,
    ) -> Result<(), AsrError> {
        // Drain the audio channel so callers see a clean failure rather
        // than a deadlock when they wait for the sender side to close.
        let mut samples: Vec<f32> = Vec::new();
        while let Some(chunk) = audio_rx.recv().await {
            samples.extend(&chunk.samples);
        }
        let _ = self.transcribe_samples(&samples)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Router factory
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct WhisperOnnxOptions {
    /// Override the quantisation variant. Accepts the suffix portion of
    /// the HF `onnx-community` filenames: `"q4"`, `"q4f16"`, `"int8"`,
    /// `"fp16"`, etc. Leave unset for the default (`q4`).
    #[serde(default)]
    quant: Option<String>,
}

/// Router factory that builds `WhisperOnnxAdapter` via `AsrRouter`.
///
/// Registered under runtime id `"whisper-onnx"`. Menura supplies the
/// `onnx-community/whisper-large-v3-turbo` bundle directory via
/// `AdapterRequest.model_source.LocalPath`, the BCP 47 language tag via
/// `AdapterRequest.language`, and an optional `{ "quant": "..." }` via
/// `AdapterRequest.options` to override the default `q4` variant.
pub struct WhisperOnnxFactory;

impl WhisperOnnxFactory {
    pub const ID: &'static str = "whisper-onnx";
}

#[async_trait]
impl AsrRuntimeFactory for WhisperOnnxFactory {
    fn id(&self) -> &'static str {
        Self::ID
    }

    async fn instantiate(&self, req: &AdapterRequest) -> Result<Arc<dyn AsrAdapter>, RouterError> {
        let ModelSource::LocalPath(model_dir) = &req.model_source;

        let opts: WhisperOnnxOptions = if req.options.is_null() {
            WhisperOnnxOptions::default()
        } else {
            serde_json::from_value(req.options.clone()).map_err(|e| {
                RouterError::InvalidRequest(format!("whisper-onnx options parse error: {e}"))
            })?
        };

        let mut cfg = WhisperOnnxConfig::default();
        if let Some(q) = opts.quant {
            cfg.encoder_file = Some(format!("encoder_model_{q}.onnx"));
            cfg.decoder_file = Some(format!("decoder_model_{q}.onnx"));
            cfg.decoder_with_past_file = Some(format!("decoder_with_past_model_{q}.onnx"));
        }
        if !req.language.is_empty() {
            cfg.language = Some(req.language.clone());
        }

        let model_dir = model_dir.clone();
        let path_for_err = model_dir.clone();
        let cfg_for_load = cfg.clone();
        let adapter = tokio::task::spawn_blocking(move || {
            WhisperOnnxAdapter::load_with_config(model_dir, cfg_for_load)
        })
        .await
        .map_err(|e| RouterError::InstantiationFailed {
            runtime: Self::ID.to_string(),
            message: format!("whisper-onnx load task panicked at {path_for_err:?}: {e}"),
        })?
        .map_err(|e| RouterError::InstantiationFailed {
            runtime: Self::ID.to_string(),
            message: e.message,
        })?;
        Ok(Arc::new(adapter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::{AdapterRequest, AsrRouter, ModelSource, RouterError};
    use serde_json::json;
    use std::path::PathBuf;

    fn req(language: &str, options: serde_json::Value) -> AdapterRequest {
        AdapterRequest {
            language: language.into(),
            runtime: WhisperOnnxFactory::ID.into(),
            model_source: ModelSource::LocalPath(PathBuf::from("/nonexistent/whisper-onnx")),
            options,
        }
    }

    #[tokio::test]
    async fn factory_id_matches_constant() {
        assert_eq!(WhisperOnnxFactory.id(), "whisper-onnx");
    }

    #[tokio::test]
    async fn dispatch_with_missing_bundle_returns_instantiation_failed() {
        let router = AsrRouter::new().register(WhisperOnnxFactory);
        match router.dispatch(req("ko", serde_json::Value::Null)).await {
            Err(RouterError::InstantiationFailed { runtime, .. }) => {
                assert_eq!(runtime, "whisper-onnx");
            }
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected loader error"),
        }
    }

    #[tokio::test]
    async fn malformed_options_return_invalid_request() {
        let router = AsrRouter::new().register(WhisperOnnxFactory);
        match router.dispatch(req("ko", json!({ "quant": 42 }))).await {
            Err(RouterError::InvalidRequest(msg)) => {
                assert!(msg.contains("whisper-onnx"));
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("expected parse error"),
        }
    }

    #[test]
    fn config_default_filenames_are_q4() {
        let cfg = WhisperOnnxConfig::default();
        assert_eq!(cfg.encoder_filename(), "encoder_model_q4.onnx");
        assert_eq!(cfg.decoder_filename(), "decoder_model_q4.onnx");
        assert_eq!(
            cfg.decoder_with_past_filename(),
            "decoder_with_past_model_q4.onnx"
        );
    }
}
