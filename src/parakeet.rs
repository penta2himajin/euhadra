//! Parakeet TDT ASR adapter — powered by parakeet-rs.
//!
//! Wraps the `parakeet-rs` crate for ONNX inference of NVIDIA Parakeet TDT models.
//! Handles mel spectrogram, FastConformer encoding, and TDT greedy decoding internally.
//!
//! Requires `--features onnx` and a model directory containing:
//! encoder-model.onnx, decoder_joint-model.onnx, vocab.txt
//!
//! Supports both English (parakeet-tdt-0.6b-v2) and multilingual models
//! including Japanese (parakeet-tdt-0.6b-v3).

use async_trait::async_trait;
use parakeet_rs::{ParakeetTDT, Transcriber};
use serde::Deserialize;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use crate::router::{AdapterRequest, AsrRuntimeFactory, ModelSource, RouterError};
use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AsrResult, AudioChunk};

/// Parakeet TDT ASR adapter using parakeet-rs.
///
/// Runs entirely in Rust via ONNX Runtime. No Python required.
///
/// # Usage
/// ```no_run
/// use euhadra::parakeet::ParakeetAdapter;
///
/// let asr = ParakeetAdapter::load("/path/to/parakeet-tdt-0.6b-v3")
///     .expect("failed to load model");
/// ```
pub struct ParakeetAdapter {
    model: Mutex<ParakeetTDT>,
}

impl ParakeetAdapter {
    /// Load with the default 128-mel preprocessor — matches
    /// `parakeet-tdt-0.6b-v2` and the multilingual European
    /// `parakeet-tdt-0.6b-v3`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, AsrError> {
        let model =
            ParakeetTDT::from_pretrained(model_dir.as_ref(), None).map_err(|e| AsrError {
                message: format!("failed to load ParakeetTDT model: {e}"),
            })?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    /// Load with an explicit mel-feature size. Use this for variants
    /// trained with a non-default preprocessor:
    ///
    /// - `nvidia/parakeet-tdt_ctc-0.6b-ja` (Japanese, Hybrid TDT-CTC) → **80**
    /// - `parakeet-tdt-0.6b-v2` / `parakeet-tdt-0.6b-v3` → **128** (same as `load`)
    ///
    /// Underlying support requires the fork at
    /// `penta2himajin/parakeet-rs@feature-size-injection`; see Cargo.toml.
    pub fn load_with_feature_size(
        model_dir: impl AsRef<Path>,
        feature_size: usize,
    ) -> Result<Self, AsrError> {
        let model =
            ParakeetTDT::from_pretrained_with_feature_size(model_dir.as_ref(), None, feature_size)
                .map_err(|e| AsrError {
                    message: format!(
                        "failed to load ParakeetTDT model (feature_size={feature_size}): {e}"
                    ),
                })?;
        Ok(Self {
            model: Mutex::new(model),
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
        // Accumulate all audio chunks
        let mut all_samples: Vec<f32> = Vec::new();
        while let Some(chunk) = audio_rx.recv().await {
            all_samples.extend(&chunk.samples);
        }

        if all_samples.is_empty() {
            return Err(AsrError {
                message: "no audio received".into(),
            });
        }

        let n_samples = all_samples.len();
        tracing::info!(audio_samples = n_samples, "transcribing with parakeet-rs");

        // Run transcription (CPU-bound)
        let result = {
            let mut model = self.model.lock().unwrap();
            model
                .transcribe_samples(all_samples, 16000, 1, None)
                .map_err(|e| AsrError {
                    message: format!("transcription failed: {e}"),
                })?
        };

        let text = result.text.trim().to_string();
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

// ---------------------------------------------------------------------------
// Router factory
// ---------------------------------------------------------------------------

/// Options accepted by `ParakeetFactory` via `AdapterRequest.options`.
///
/// `feature_size` selects between the 128-mel (default — v2/v3) and
/// 80-mel (ja Hybrid TDT-CTC) preprocessors. Leave unset to fall back
/// to `ParakeetAdapter::load`'s default 128-mel path.
#[derive(Debug, Default, Deserialize)]
struct ParakeetOptions {
    #[serde(default)]
    feature_size: Option<usize>,
}

/// Router factory that builds `ParakeetAdapter` via `AsrRouter`.
///
/// Registered under the runtime id `"parakeet"`. The language is
/// determined by the model variant itself — `parakeet-tdt-0.6b-v3`
/// covers en + EU 25, the `-ja` variant is Japanese-only — so the
/// factory does not consume `AdapterRequest.language` today. Menura's
/// `asr_models.toml` still routes per BCP 47 tag, just by selecting
/// different model bundles for different languages.
pub struct ParakeetFactory;

impl ParakeetFactory {
    pub const ID: &'static str = "parakeet";
}

#[async_trait]
impl AsrRuntimeFactory for ParakeetFactory {
    fn id(&self) -> &'static str {
        Self::ID
    }

    async fn instantiate(&self, req: &AdapterRequest) -> Result<Arc<dyn AsrAdapter>, RouterError> {
        let ModelSource::LocalPath(model_dir) = &req.model_source;

        let opts: ParakeetOptions = if req.options.is_null() {
            ParakeetOptions::default()
        } else {
            serde_json::from_value(req.options.clone()).map_err(|e| {
                RouterError::InvalidRequest(format!("parakeet options parse error: {e}"))
            })?
        };

        let adapter = match opts.feature_size {
            None => ParakeetAdapter::load(model_dir),
            Some(fs) => ParakeetAdapter::load_with_feature_size(model_dir, fs),
        }
        .map_err(|e| RouterError::InstantiationFailed {
            runtime: Self::ID.to_string(),
            message: e.message,
        })?;
        Ok(Arc::new(adapter))
    }
}

#[cfg(test)]
mod factory_tests {
    use super::*;
    use crate::router::{AdapterRequest, AsrRouter, ModelSource, RouterError};
    use serde_json::json;
    use std::path::PathBuf;

    fn req(options: serde_json::Value) -> AdapterRequest {
        AdapterRequest {
            language: "en".into(),
            runtime: ParakeetFactory::ID.into(),
            model_source: ModelSource::LocalPath(PathBuf::from("/nonexistent/parakeet/bundle")),
            options,
        }
    }

    #[tokio::test]
    async fn factory_id_matches_constant() {
        assert_eq!(ParakeetFactory.id(), "parakeet");
    }

    #[tokio::test]
    async fn dispatch_with_missing_bundle_returns_instantiation_failed() {
        let router = AsrRouter::new().register(ParakeetFactory);
        match router.dispatch(req(serde_json::Value::Null)).await {
            Err(RouterError::InstantiationFailed { runtime, .. }) => {
                assert_eq!(runtime, "parakeet");
            }
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected error when bundle dir does not exist"),
        }
    }

    #[tokio::test]
    async fn feature_size_80_reaches_load_with_feature_size_path() {
        let router = AsrRouter::new().register(ParakeetFactory);
        match router.dispatch(req(json!({ "feature_size": 80 }))).await {
            Err(RouterError::InstantiationFailed { runtime, message }) => {
                assert_eq!(runtime, "parakeet");
                // load_with_feature_size embeds the feature size in its
                // failure message; this confirms we went through that path
                // rather than the default `load`.
                assert!(
                    message.contains("feature_size=80"),
                    "expected feature_size=80 in error, got: {message}"
                );
            }
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected error when bundle dir does not exist"),
        }
    }

    #[tokio::test]
    async fn malformed_options_return_invalid_request() {
        let router = AsrRouter::new().register(ParakeetFactory);
        match router
            .dispatch(req(json!({ "feature_size": "not a number" })))
            .await
        {
            Err(RouterError::InvalidRequest(msg)) => {
                assert!(msg.contains("parakeet"));
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("expected error for malformed options"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn load_nonexistent_model_returns_error() {
        let result = ParakeetAdapter::load("/nonexistent/path/to/model");
        assert!(result.is_err(), "loading from nonexistent path should fail");
        let err = result.err().unwrap();
        assert!(
            err.message.contains("failed to load"),
            "error message should indicate load failure: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn transcribe_empty_audio_sends_empty_text() {
        // Verify the AsrAdapter contract: when audio channel closes immediately
        // with no chunks, the adapter drains and emits its result.
        let (tx, rx) = mpsc::channel::<AudioChunk>(1);
        let (result_tx, mut result_rx) = mpsc::channel::<AsrResult>(1);

        // Drop sender immediately → receiver gets None → no audio
        drop(tx);

        use crate::mock::MockAsr;
        use crate::traits::AsrAdapter;

        let mock = MockAsr::new("");
        let _ = mock.transcribe(rx, result_tx).await;
        // MockAsr with empty string still sends an AsrResult with empty text.
        // The real ParakeetAdapter would return an error for empty audio.
        let result = result_rx.try_recv();
        assert!(
            result.is_ok(),
            "MockAsr should send a result even with empty transcript"
        );
        assert!(result.unwrap().text.is_empty());
    }

    #[test]
    fn adapter_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ParakeetAdapter>();
    }
}
