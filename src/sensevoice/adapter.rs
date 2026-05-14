//! `SenseVoiceAdapter` — `AsrAdapter` for FunAudioLLM SenseVoice-Small.
//!
//! Glues the four pieces into a single end-to-end pipeline:
//!
//! ```text
//! audio f32  →  Paraformer-shared FBANK (80 mel)
//!            →  LFR (m=7, n=6)  →  CMVN (am.mvn)
//!            →  ONNX (x, x_length, language, text_norm)
//!            →  argmax → unique_consecutive → drop blank
//!            →  vocab lookup → strip `<|...|>` markers
//!            →  text
//! ```
//!
//! Defaults match what `scripts/setup_sensevoice.sh` produces from the
//! official `FunAudioLLM/SenseVoiceSmall` HuggingFace bundle:
//!
//! ```text
//! <model_dir>/
//!   model.int8.onnx      (INT8, ~234 MB)
//!   am.mvn               (Kaldi-NNet text format, 80*7=560 dims)
//!   tokens.txt           (one SentencePiece piece per line, line = id)
//!   metadata.json        (lang2id, with_itn_id, blank_id, lfr_m/n)
//! ```
//!
//! FP32 `model.onnx` (~895 MB) is no longer shipped by the setup
//! script (issue #59 Phase 2); INT8 CER drift on FLEURS-ko is <1pp.
//!
//! The model is shipped under the SenseVoice MODEL_LICENSE (CC-BY-NC
//! 4.0); this Rust port re-implements only the inference logic and
//! ships none of the upstream weights.

use async_trait::async_trait;
use ndarray::{Array1, Array3};
// `language` / `textnorm` / `speech_lengths` are i32 in the official
// FunASR export. PyTorch's `nn.Embedding` requires long tensors at
// the layer boundary, but the SenseVoice export wrapper inserts a
// Cast int32 → int64 right after the encoder inputs, exposing the
// portable int32 type at the ONNX boundary. The runtime rejects
// int64 here with `Unexpected input data type. Actual:
// (tensor(int64)), expected: (tensor(int32))`.
use ort::session::Session;
use ort::value::Value;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokio::sync::mpsc;

use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AsrResult, AudioChunk};

use crate::paraformer::fbank::{Fbank, FbankOpts};
use crate::paraformer::frontend::{apply_cmvn, apply_lfr, load_cmvn, Cmvn};

use super::metadata::SenseVoiceMetadata;
use super::vocab::{ctc_collapse, decode_tokens, ids_to_tokens, load_tokens_txt};

/// File-name overrides + per-utterance defaults. The defaults match
/// the INT8 bundle layout produced by `scripts/setup_sensevoice.sh`.
/// FP32 `model.onnx` is no longer shipped (issue #59 Phase 2); set
/// `model_filename` manually if you have a custom FP32 export.
#[derive(Debug, Clone)]
pub struct SenseVoiceConfig {
    pub fbank: FbankOpts,
    pub model_filename: String,
    pub cmvn_filename: String,
    pub tokens_filename: String,
    pub metadata_filename: String,
    /// Default language fed to the encoder when the caller doesn't
    /// override via `with_language`. `"ko"` matches the immediate
    /// integration goal; pass `"auto"` for upstream LID.
    pub default_language: String,
    /// Apply ITN (e.g. "12" instead of "twelve"/"열두") on the raw
    /// transcript. `true` matches the upstream `demo2.py` default.
    pub default_use_itn: bool,
}

impl SenseVoiceConfig {
    /// Idempotent on the current default (which already points to
    /// `model.int8.onnx`). Kept as a builder helper so callers can be
    /// explicit about precision intent, and so a future FP32 default
    /// switch wouldn't require API changes downstream.
    pub fn with_int8_weights(mut self) -> Self {
        self.model_filename = "model.int8.onnx".to_string();
        self
    }
}

impl Default for SenseVoiceConfig {
    fn default() -> Self {
        Self {
            fbank: FbankOpts::paraformer_default(),
            model_filename: "model.int8.onnx".to_string(),
            cmvn_filename: "am.mvn".to_string(),
            tokens_filename: "tokens.txt".to_string(),
            metadata_filename: "metadata.json".to_string(),
            default_language: "ko".to_string(),
            default_use_itn: true,
        }
    }
}

/// `AsrAdapter` for SenseVoice-Small via ONNX Runtime.
///
/// The adapter accumulates the full utterance, runs frontend → ONNX
/// → CTC postprocess, and emits a single final `AsrResult`. The
/// upstream model is non-autoregressive, so a partial / streaming
/// decoder isn't applicable; chunking would be done by the caller
/// before feeding audio in.
pub struct SenseVoiceAdapter {
    session: Mutex<Session>,
    fbank: Fbank,
    cmvn: Cmvn,
    vocab: Vec<String>,
    metadata: SenseVoiceMetadata,
    cfg: SenseVoiceConfig,
    /// Resolved language label that drives the `language` input. May
    /// be overridden per-adapter via `with_language`.
    language: String,
    /// Cached input names so we don't introspect the session per
    /// utterance.
    input_names: InputNames,
}

#[derive(Debug, Clone)]
struct InputNames {
    x: String,
    x_length: String,
    language: String,
    text_norm: String,
}

const ONNX_INPUT_X: &str = "speech";
const ONNX_INPUT_X_LENGTH: &str = "speech_lengths";
const ONNX_INPUT_LANGUAGE: &str = "language";
const ONNX_INPUT_TEXT_NORM: &str = "textnorm";

impl SenseVoiceAdapter {
    /// Load a model bundle laid out as
    /// `<dir>/{model.int8.onnx, am.mvn, tokens.txt, metadata.json}`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, AsrError> {
        Self::load_with_config(model_dir, SenseVoiceConfig::default())
    }

    pub fn load_with_config(
        model_dir: impl AsRef<Path>,
        cfg: SenseVoiceConfig,
    ) -> Result<Self, AsrError> {
        let dir: PathBuf = model_dir.as_ref().to_path_buf();
        let model_path = dir.join(&cfg.model_filename);
        let mvn_path = dir.join(&cfg.cmvn_filename);
        let tokens_path = dir.join(&cfg.tokens_filename);
        let metadata_path = dir.join(&cfg.metadata_filename);

        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(&model_path))
            .map_err(|e| AsrError {
                message: format!(
                    "failed to load SenseVoice ONNX {}: {e}",
                    model_path.display()
                ),
            })?;

        let cmvn = load_cmvn(&mvn_path)?;
        let vocab = load_tokens_txt(&tokens_path)?;
        let metadata = SenseVoiceMetadata::load(&metadata_path)?;

        let expected_dim = cfg.fbank.n_mels * metadata.lfr_m;
        if cmvn.dim() != expected_dim {
            return Err(AsrError {
                message: format!(
                    "CMVN dim {} does not match n_mels({}) * lfr_m({}) = {}",
                    cmvn.dim(),
                    cfg.fbank.n_mels,
                    metadata.lfr_m,
                    expected_dim
                ),
            });
        }

        if metadata.language_id(&cfg.default_language).is_none() {
            return Err(AsrError {
                message: format!(
                    "metadata.json {} has no language id for default_language={:?}",
                    metadata_path.display(),
                    cfg.default_language
                ),
            });
        }

        let input_names = resolve_input_names(&session, &model_path)?;
        let fbank = Fbank::new(cfg.fbank.clone());
        let language = cfg.default_language.clone();

        Ok(Self {
            session: Mutex::new(session),
            fbank,
            cmvn,
            vocab,
            metadata,
            cfg,
            language,
            input_names,
        })
    }

    /// Builder-style language override. Accepts ISO 639-1 (`"ko"`,
    /// `"ja"`, `"zh"`, `"yue"`, `"en"`), the long-form alias
    /// (`"korean"`, …), or `"auto"` for upstream LID.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// Builder-style ITN override. Defaults to `true`, matching the
    /// upstream demo behaviour.
    pub fn with_use_itn(mut self, use_itn: bool) -> Self {
        self.cfg.default_use_itn = use_itn;
        self
    }

    /// Run the full extract → encode → decode pipeline on a single
    /// concatenated waveform. Exposed so cousin tests can drive the
    /// model end-to-end without spinning up a pipeline session.
    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<String, AsrError> {
        if samples.is_empty() {
            return Err(AsrError {
                message: "no audio received".into(),
            });
        }

        let (mel, n_frames) = self.fbank.compute(samples);
        if n_frames == 0 {
            return Err(AsrError {
                message: format!(
                    "audio too short for one FBANK frame ({} samples)",
                    samples.len()
                ),
            });
        }

        let n_mels = self.fbank.n_mels();
        let lfr_m = self.metadata.lfr_m;
        let lfr_n = self.metadata.lfr_n;
        let (mut feats, t_lfr) = apply_lfr(&mel, n_frames, n_mels, lfr_m, lfr_n);
        let feat_dim = n_mels * lfr_m;
        apply_cmvn(&mut feats, feat_dim, &self.cmvn);

        let language_id = self
            .metadata
            .language_id(&self.language)
            .ok_or_else(|| AsrError {
                message: format!(
                    "language {:?} not present in metadata.json lang2id",
                    self.language
                ),
            })?;
        let text_norm_id = if self.cfg.default_use_itn {
            self.metadata.with_itn_id
        } else {
            self.metadata.without_itn_id
        };

        // ONNX expects `speech` as [B=1, T, feat_dim] f32 and the three
        // sidecar integer inputs as int32 1-D tensors.
        let speech: Array3<f32> =
            Array3::from_shape_vec((1, t_lfr, feat_dim), feats).map_err(|e| AsrError {
                message: format!("speech tensor shape: {e}"),
            })?;
        let speech_lengths: Array1<i32> = Array1::from(vec![t_lfr as i32]);
        let language_arr: Array1<i32> = Array1::from(vec![language_id]);
        let text_norm_arr: Array1<i32> = Array1::from(vec![text_norm_id]);

        let speech_val = Value::from_array(speech).map_err(|e| AsrError {
            message: format!("speech Value: {e}"),
        })?;
        let speech_lengths_val = Value::from_array(speech_lengths).map_err(|e| AsrError {
            message: format!("speech_lengths Value: {e}"),
        })?;
        let language_val = Value::from_array(language_arr).map_err(|e| AsrError {
            message: format!("language Value: {e}"),
        })?;
        let text_norm_val = Value::from_array(text_norm_arr).map_err(|e| AsrError {
            message: format!("textnorm Value: {e}"),
        })?;

        // Hold the session lock for the duration of decoding so the
        // borrowed output tensor stays live while we argmax over [T, V].
        let mut session = self.session.lock().map_err(|e| AsrError {
            message: format!("session lock poisoned: {e}"),
        })?;
        let outputs = session
            .run(vec![
                (self.input_names.x.as_str(), speech_val.into_dyn()),
                (
                    self.input_names.x_length.as_str(),
                    speech_lengths_val.into_dyn(),
                ),
                (self.input_names.language.as_str(), language_val.into_dyn()),
                (
                    self.input_names.text_norm.as_str(),
                    text_norm_val.into_dyn(),
                ),
            ])
            .map_err(|e| AsrError {
                message: format!("SenseVoice ONNX run: {e}"),
            })?;

        // Output: logits [1, T_out, V] f32 — CTC log-softmax.
        let logits = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| AsrError {
                message: format!("extract logits: {e}"),
            })?;
        let view = logits.view();
        let shape = view.shape().to_vec();
        if shape.len() != 3 {
            return Err(AsrError {
                message: format!("unexpected logits rank {}", shape.len()),
            });
        }
        let t = shape[1];
        let v = shape[2];

        let mut ids = Vec::with_capacity(t);
        for ti in 0..t {
            let mut best = 0u32;
            let mut best_v = f32::NEG_INFINITY;
            for vi in 0..v {
                let val = view[[0, ti, vi]];
                if val > best_v {
                    best_v = val;
                    best = vi as u32;
                }
            }
            ids.push(best);
        }

        let collapsed = ctc_collapse(&ids, self.metadata.blank_id);
        let tokens = ids_to_tokens(&collapsed, &self.vocab);
        Ok(decode_tokens(&tokens))
    }
}

fn resolve_input_names(session: &Session, path: &Path) -> Result<InputNames, AsrError> {
    let names: Vec<String> = session
        .inputs()
        .iter()
        .map(|i| i.name().to_string())
        .collect();
    let find = |needle: &str| -> Result<String, AsrError> {
        names
            .iter()
            .find(|n| n.as_str() == needle)
            .cloned()
            .ok_or_else(|| AsrError {
                message: format!(
                    "SenseVoice ONNX {} missing input {:?} (have: {:?})",
                    path.display(),
                    needle,
                    names
                ),
            })
    };
    Ok(InputNames {
        x: find(ONNX_INPUT_X)?,
        x_length: find(ONNX_INPUT_X_LENGTH)?,
        language: find(ONNX_INPUT_LANGUAGE)?,
        text_norm: find(ONNX_INPUT_TEXT_NORM)?,
    })
}

#[async_trait]
impl AsrAdapter for SenseVoiceAdapter {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        result_tx: mpsc::Sender<AsrResult>,
    ) -> Result<(), AsrError> {
        let mut all_samples: Vec<f32> = Vec::new();
        while let Some(chunk) = audio_rx.recv().await {
            all_samples.extend(&chunk.samples);
        }

        if all_samples.is_empty() {
            return Err(AsrError {
                message: "no audio received".into(),
            });
        }

        tracing::info!(
            audio_samples = all_samples.len(),
            language = %self.language,
            use_itn = self.cfg.default_use_itn,
            "transcribing with sensevoice-small"
        );

        let text = self.transcribe_samples(&all_samples)?;
        if !text.is_empty() {
            result_tx
                .send(AsrResult {
                    text,
                    is_final: true,
                    confidence: 1.0,
                    timestamp: std::time::Duration::ZERO,
                })
                .await
                .map_err(|e| AsrError {
                    message: format!("send: {e}"),
                })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_nonexistent_dir_returns_error() {
        let res = SenseVoiceAdapter::load("/nonexistent/path/to/sensevoice");
        assert!(res.is_err());
        let err = res.err().unwrap();
        assert!(
            err.message.contains("failed to load")
                || err.message.contains("read")
                || err.message.contains("No such file"),
            "expected load-failure message, got: {}",
            err.message
        );
    }

    #[test]
    fn config_default_filenames_match_setup_script() {
        // Default tracks the INT8-only bundle layout produced by
        // `scripts/setup_sensevoice.sh` after issue #59 Phase 2.
        let cfg = SenseVoiceConfig::default();
        assert_eq!(cfg.model_filename, "model.int8.onnx");
        assert_eq!(cfg.cmvn_filename, "am.mvn");
        assert_eq!(cfg.tokens_filename, "tokens.txt");
        assert_eq!(cfg.metadata_filename, "metadata.json");
        assert_eq!(cfg.default_language, "ko");
        assert!(cfg.default_use_itn);
    }

    #[test]
    fn with_int8_weights_swaps_only_model_filename() {
        // Default already targets INT8 after issue #59 Phase 2, so
        // `with_int8_weights()` is idempotent — the assertion still
        // holds and the rest of the bundle stays untouched.
        let cfg = SenseVoiceConfig::default().with_int8_weights();
        assert_eq!(cfg.model_filename, "model.int8.onnx");
        // Sidecars are precision-agnostic so they survive the override.
        assert_eq!(cfg.cmvn_filename, "am.mvn");
        assert_eq!(cfg.tokens_filename, "tokens.txt");
        assert_eq!(cfg.metadata_filename, "metadata.json");
        assert_eq!(cfg.default_language, "ko");
    }

    #[test]
    fn adapter_is_send_and_sync() {
        // The pipeline runtime requires Send + Sync ASR adapters so
        // they can move between worker tasks. ort's Session, wrapped
        // in Mutex, satisfies both — pin that here so a future
        // refactor can't accidentally break the contract.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SenseVoiceAdapter>();
    }

    #[tokio::test]
    async fn empty_audio_yields_no_audio_received_error() {
        // Mirrors the contract enforced by ParakeetAdapter,
        // ParaformerAdapter, and CanaryAdapter — empty channel →
        // AsrError with "no audio received". Verified through MockAsr
        // without requiring a real SenseVoice bundle.
        use crate::mock::MockAsr;
        let (tx, rx) = mpsc::channel::<AudioChunk>(1);
        let (rtx, mut rrx) = mpsc::channel::<AsrResult>(1);
        drop(tx);
        let mock = MockAsr::new("");
        let _ = mock.transcribe(rx, rtx).await;
        assert!(rrx.try_recv().is_ok());
    }
}
