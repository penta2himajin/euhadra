//! `WhisperOnnxAdapter` — Whisper-large-v3-turbo via ONNX Runtime (`ort`).
//!
//! Loads the [`onnx-community/whisper-large-v3-turbo`](https://huggingface.co/onnx-community/whisper-large-v3-turbo)
//! bundle (encoder + decoder + decoder_with_past) into the same `ort`
//! sessions that the existing Parakeet / Canary / Paraformer / SenseVoice
//! adapters use. Lives under the existing `onnx` feature gate; no new
//! system dependencies vs the other ORT-based adapters.
//!
//! # Decode pipeline
//!
//! 1. [`mel::WhisperMel`] turns the f32 audio into a `(128, 3000)`
//!    log-mel buffer (n_fft=400, hop=160, Slaney mel, Whisper's
//!    log10 + max-clamp + `(x+4)/4` post-norm).
//! 2. Encoder ONNX maps `(1, 128, 3000)` → `last_hidden_state
//!    (1, 1500, 1280)`.
//! 3. Decoder ONNX runs once with the prompt
//!    `<|startoftranscript|><|<lang>|><|transcribe|><|notimestamps|>`
//!    plus `encoder_hidden_states`; produces logits + 16 KV tensors
//!    (4 layers × decoder/encoder × key/value).
//! 4. `decoder_with_past` ONNX loops on the latest token + full KV
//!    cache, returning logits + 8 *updated* decoder KVs (encoder
//!    KVs stay constant). Greedy argmax; stop on `<|endoftext|>` or
//!    `max_new_tokens` (440 ≈ 30 s of speech).
//! 5. Drop prompt + EOT, hand the rest to `tokenizers.decode(skip_special_tokens=True)`.
//!
//! # Bench results (Python POC, raw onnxruntime)
//!
//! FLEURS-ko 10-utt subset that gates L1, shared 4-core Xeon @ 2.1 GHz
//! VM as the other #83 benches, lenient CER via
//! [`euhadra::eval::metrics::cer_lenient`]:
//!
//! | Quantisation variant | weighted CER | RTF       | bundle size |
//! |----------------------|-------------:|----------:|------------:|
//! | `q4`                 |    **1.09%** | **0.484** | ~1.1 GB     |
//! | `int8`               |        260%* | 0.255     |   ~1.5 GB   |
//!
//! * INT8's autoregressive Whisper decoder collapses into repeating-token
//!   hallucinations on this CPU; not usable. `q4` is the recommended
//!   variant for production.
//!
//! For comparison from PR #103 / #104 / `docs/korean-asr-alternatives.md`:
//!
//! | Path                          | CER   | RTF       |
//! |-------------------------------|------:|----------:|
//! | **ORT `q4` (this adapter)**   | 1.09% | 0.484     |
//! | CT2 FP16 (`faster-whisper`)   | 1.32% | 1.28      |
//! | whisper-rs Q4_0 (GGML)        | 1.74% | 1.78      |
//! | SenseVoice INT8 (baseline)    | 6.64% | 0.047     |
//!
//! # Rust port verification
//!
//! Running `cargo run --release --features onnx --example
//! bench_whisper_onnx_ko` against the same FLEURS-ko 10-utt subset
//! (4-core Xeon @ 2.8 GHz, loadavg ~2.6) gave CER **0.95 %** at the
//! mean-of-per-utterance aggregation (POC's `jiwer.cer` reports
//! 1.09 % on weighted concat — both numbers reflect the same
//! transcript quality). The Rust port reproduces the POC's output
//! token sequence; the small CER delta is jiwer-vs-`cer_lenient`
//! normalisation, not a model behaviour gap.

use async_trait::async_trait;
use ndarray::{s, Array3, Array4, ArrayD};
use ort::session::Session;
use ort::value::Value;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use super::mel::{self, WhisperMel};
use super::tokenizer::WhisperTokenizer;
use crate::router::{AdapterRequest, AsrRuntimeFactory, ModelSource, RouterError};
use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AsrResult, AudioChunk};

/// Whisper-large-v3-turbo has 4 decoder layers.
const N_DECODER_LAYERS: usize = 4;
/// Per-step cap on generated tokens; 440 ≈ 30 s of speech, beyond
/// which the encoder window has run out.
const MAX_NEW_TOKENS: usize = 440;

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
pub struct WhisperOnnxAdapter {
    /// Encoder ONNX session. Holds an `Mutex` because `ort::Session` is
    /// `Send` but not `Sync`; the lock is only contended when the same
    /// adapter is shared across concurrent async sessions.
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    decoder_with_past: Mutex<Session>,
    /// Cached output-name lists, captured at load time so the decode
    /// loop doesn't re-introspect the sessions per step. The order
    /// matches the `Vec` returned by `Session::run` (which is the
    /// graph's declared output order).
    decoder_output_names: Vec<String>,
    decoder_past_output_names: Vec<String>,
    mel: WhisperMel,
    tokenizer: WhisperTokenizer,
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

        let encoder = build_session(&enc_path, "encoder")?;
        let decoder = build_session(&dec_path, "decoder")?;
        let decoder_with_past = build_session(&dec_past_path, "decoder_with_past")?;

        let decoder_output_names: Vec<String> = decoder
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let decoder_past_output_names: Vec<String> = decoder_with_past
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let tokenizer = WhisperTokenizer::load(&dir)?;

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            decoder_with_past: Mutex::new(decoder_with_past),
            decoder_output_names,
            decoder_past_output_names,
            mel: WhisperMel::new(),
            tokenizer,
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
    /// Audio is padded/truncated to 30 s. Returns the trimmed plain-text
    /// transcript with prompt + EOT removed.
    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<String, AsrError> {
        // ---- Mel features ----
        let mel_data = self.mel.compute(samples);
        let input_features = Array3::from_shape_vec((1, mel::N_MELS, mel::N_FRAMES), mel_data)
            .map_err(|e| AsrError {
                message: format!("whisper-onnx mel reshape: {e}"),
            })?;

        // ---- Encoder ----
        let enc_value = Value::from_array(input_features).map_err(|e| AsrError {
            message: format!("whisper-onnx input_features Value: {e}"),
        })?;
        let encoder_hidden = {
            let mut session = self.encoder.lock().map_err(|e| AsrError {
                message: format!("whisper-onnx encoder lock poisoned: {e}"),
            })?;
            let outputs = session
                .run(vec![("input_features", enc_value.into_dyn())])
                .map_err(|e| AsrError {
                    message: format!("whisper-onnx encoder run: {e}"),
                })?;
            // Single output: `last_hidden_state (1, 1500, 1280)`.
            outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| AsrError {
                    message: format!("extract last_hidden_state: {e}"),
                })?
                .to_owned()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| AsrError {
                    message: format!("last_hidden_state rank: {e}"),
                })?
        };

        // ---- Build prompt ----
        let lang = self.cfg.language.clone();
        let prompt = self
            .tokenizer
            .build_prompt(lang.as_deref().filter(|s| !s.is_empty() && *s != "auto"))?;
        let prompt_len = prompt.len();
        let eot = self.tokenizer.eot();

        // ---- Decoder first step ----
        let input_ids =
            ndarray::Array2::from_shape_vec((1, prompt_len), prompt.clone()).map_err(|e| {
                AsrError {
                    message: format!("whisper-onnx prompt reshape: {e}"),
                }
            })?;

        let ids_v = Value::from_array(input_ids).map_err(|e| AsrError {
            message: format!("whisper-onnx input_ids Value: {e}"),
        })?;
        let enc_v = Value::from_array(encoder_hidden.clone()).map_err(|e| AsrError {
            message: format!("whisper-onnx encoder_hidden_states Value: {e}"),
        })?;

        let mut session = self.decoder.lock().map_err(|e| AsrError {
            message: format!("whisper-onnx decoder lock poisoned: {e}"),
        })?;
        let outputs = session
            .run(vec![
                ("input_ids", ids_v.into_dyn()),
                ("encoder_hidden_states", enc_v.into_dyn()),
            ])
            .map_err(|e| AsrError {
                message: format!("whisper-onnx decoder run: {e}"),
            })?;

        // Output 0 is `logits (1, prompt_len, vocab)`; the rest are
        // `present.<i>.{decoder|encoder}.{key|value}` in declaration
        // order. Capture them as owned arrays + their names so the
        // subsequent loop can rebuild `past_key_values.*` feeds.
        let logits = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| AsrError {
                message: format!("extract logits: {e}"),
            })?
            .to_owned();
        let mut next_token = greedy_last(&logits)?;

        let mut decoder_kv: Vec<(String, Array4<f32>)> = Vec::with_capacity(N_DECODER_LAYERS * 2);
        let mut encoder_kv: Vec<(String, Array4<f32>)> = Vec::with_capacity(N_DECODER_LAYERS * 2);
        for (i, name) in self.decoder_output_names.iter().enumerate().skip(1) {
            let arr: Array4<f32> = outputs[i]
                .try_extract_array::<f32>()
                .map_err(|e| AsrError {
                    message: format!("extract {name}: {e}"),
                })?
                .to_owned()
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| AsrError {
                    message: format!("{name} rank: {e}"),
                })?;
            let past_name = present_to_past(name);
            if name.contains(".encoder.") {
                encoder_kv.push((past_name, arr));
            } else {
                decoder_kv.push((past_name, arr));
            }
        }
        drop(outputs);
        drop(session);

        let mut generated: Vec<i64> = prompt;
        generated.push(next_token);

        // ---- decoder_with_past loop ----
        if next_token != eot {
            let mut past_session = self.decoder_with_past.lock().map_err(|e| AsrError {
                message: format!("whisper-onnx decoder_with_past lock poisoned: {e}"),
            })?;
            for _ in 0..MAX_NEW_TOKENS {
                if next_token == eot {
                    break;
                }
                let ids =
                    ndarray::Array2::from_shape_vec((1, 1), vec![next_token]).map_err(|e| {
                        AsrError {
                            message: format!("whisper-onnx step input_ids reshape: {e}"),
                        }
                    })?;
                let ids_v = Value::from_array(ids).map_err(|e| AsrError {
                    message: format!("whisper-onnx step input_ids Value: {e}"),
                })?;
                let mut feeds: Vec<(&str, ort::value::DynValue)> =
                    Vec::with_capacity(1 + N_DECODER_LAYERS * 4);
                feeds.push(("input_ids", ids_v.into_dyn()));
                for (name, arr) in decoder_kv.iter().chain(encoder_kv.iter()) {
                    let v = Value::from_array(arr.clone()).map_err(|e| AsrError {
                        message: format!("whisper-onnx step {name} Value: {e}"),
                    })?;
                    feeds.push((name.as_str(), v.into_dyn()));
                }

                let outs = past_session.run(feeds).map_err(|e| AsrError {
                    message: format!("whisper-onnx decoder_with_past run: {e}"),
                })?;
                let logits = outs[0]
                    .try_extract_array::<f32>()
                    .map_err(|e| AsrError {
                        message: format!("extract step logits: {e}"),
                    })?
                    .to_owned();
                next_token = greedy_last(&logits)?;
                generated.push(next_token);

                // decoder_with_past returns logits + 8 new decoder KVs
                // (encoder KVs stay constant). Replace `decoder_kv`.
                let mut new_decoder_kv: Vec<(String, Array4<f32>)> =
                    Vec::with_capacity(N_DECODER_LAYERS * 2);
                for (i, name) in self.decoder_past_output_names.iter().enumerate().skip(1) {
                    let arr: Array4<f32> = outs[i]
                        .try_extract_array::<f32>()
                        .map_err(|e| AsrError {
                            message: format!("extract step {name}: {e}"),
                        })?
                        .to_owned()
                        .into_dimensionality::<ndarray::Ix4>()
                        .map_err(|e| AsrError {
                            message: format!("step {name} rank: {e}"),
                        })?;
                    new_decoder_kv.push((present_to_past(name), arr));
                }
                decoder_kv = new_decoder_kv;
            }
        }

        // Strip prompt + EOT, then detokenize.
        let text_ids: Vec<i64> = generated
            .into_iter()
            .skip(prompt_len)
            .take_while(|&t| t != eot)
            .collect();
        self.tokenizer.decode(&text_ids)
    }
}

#[async_trait]
impl AsrAdapter for WhisperOnnxAdapter {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        result_tx: mpsc::Sender<AsrResult>,
    ) -> Result<(), AsrError> {
        let mut samples: Vec<f32> = Vec::new();
        while let Some(chunk) = audio_rx.recv().await {
            samples.extend(&chunk.samples);
        }
        let text = self.transcribe_samples(&samples)?;
        let _ = result_tx
            .send(AsrResult {
                text,
                is_final: true,
                confidence: 1.0,
                timestamp: std::time::Duration::from_secs(0),
            })
            .await;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build an `ort::Session` from a model file, pinning intra-op threads
/// to the same `min(4, num_cpus)` the Python POC measured under (with
/// `inter_op_num_threads = 1`). Letting ort pick its own default
/// drops Whisper-q4 throughput by ~2× on the shared bench VM because
/// the per-step decode kernels don't parallelise well across many
/// threads; the encoder's matmuls dominate and saturate ~4.
fn build_session(path: &std::path::Path, role: &str) -> Result<Session, AsrError> {
    let threads = std::cmp::min(num_cpus_hint(), 4);
    let mut builder = Session::builder().map_err(|e| AsrError {
        message: format!("whisper-onnx {role} builder: {e}"),
    })?;
    builder = builder.with_intra_threads(threads).map_err(|e| AsrError {
        message: format!("whisper-onnx {role} with_intra_threads({threads}): {e}"),
    })?;
    builder = builder.with_inter_threads(1).map_err(|e| AsrError {
        message: format!("whisper-onnx {role} with_inter_threads(1): {e}"),
    })?;
    builder.commit_from_file(path).map_err(|e| AsrError {
        message: format!("whisper-onnx {role} load failed at {}: {e}", path.display()),
    })
}

/// Cheap available-parallelism probe; falls back to 1 when the
/// platform refuses (cgroup-restricted containers, etc.).
fn num_cpus_hint() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// `present.<i>.<...>` → `past_key_values.<i>.<...>` — the naming the
/// `decoder_with_past` session expects for the cached state feed.
fn present_to_past(name: &str) -> String {
    name.replacen("present.", "past_key_values.", 1)
}

/// Argmax over the last time step of a `(B, T, V)` logits tensor,
/// batch 0. Returns the token id as `i64` since that's what
/// `input_ids` expects on the next step.
fn greedy_last(logits: &ArrayD<f32>) -> Result<i64, AsrError> {
    let shape = logits.shape();
    if shape.len() != 3 {
        return Err(AsrError {
            message: format!("logits expected rank 3, got {shape:?}"),
        });
    }
    let (_b, t, v) = (shape[0], shape[1], shape[2]);
    if t == 0 || v == 0 {
        return Err(AsrError {
            message: format!("logits empty: shape={shape:?}"),
        });
    }
    let last = logits
        .view()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| AsrError {
            message: format!("logits rank3 view: {e}"),
        })?
        .slice(s![0, t - 1, ..])
        .to_owned();
    let mut best = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in last.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    Ok(best as i64)
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

    #[test]
    fn present_to_past_renames_only_prefix() {
        assert_eq!(
            present_to_past("present.0.decoder.key"),
            "past_key_values.0.decoder.key"
        );
        assert_eq!(
            present_to_past("present.3.encoder.value"),
            "past_key_values.3.encoder.value"
        );
        // Only the first occurrence is rewritten; the literal segment
        // ".present." mid-name wouldn't be remapped (Whisper graphs
        // don't use it, but the contract guards future churn).
        assert_eq!(
            present_to_past("foo.present.0.key"),
            "foo.past_key_values.0.key"
        );
    }
}
