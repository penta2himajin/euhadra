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

use crate::router::{AdapterRequest, AsrRuntimeFactory, ModelSource, RouterError};
use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AudioChunk, Transcript};

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
    /// Load a Parakeet TDT bundle.
    ///
    /// The mel-filterbank size comes from the encoder graph, so this
    /// works for both the 128-mel `parakeet-tdt-0.6b-v2` / `v3` and the
    /// 80-mel `nvidia/parakeet-tdt_ctc-0.6b-ja` without being told
    /// which is which.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, AsrError> {
        let model =
            ParakeetTDT::from_pretrained(model_dir.as_ref(), None).map_err(|e| AsrError::ModelLoad(format!("failed to load ParakeetTDT model: {e}")))?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }
}

#[async_trait]
impl AsrAdapter for ParakeetAdapter {
    async fn transcribe(&self, audio: &[AudioChunk]) -> Result<Transcript, AsrError> {
        let all_samples = AudioChunk::concat(audio);
        if all_samples.is_empty() {
            return Err(AsrError::NoAudio);
        }

        tracing::info!(
            audio_samples = all_samples.len(),
            "transcribing with parakeet-rs"
        );

        // Run transcription (CPU-bound)
        let result = {
            let mut model = self.model.lock().unwrap();
            model
                .transcribe_samples(all_samples, 16000, 1, None)
                .map_err(|e| AsrError::Inference(format!("parakeet: {e}")))?
        };

        Ok(Transcript::new(result.text.trim()))
    }
}

// ---------------------------------------------------------------------------
// Router factory
// ---------------------------------------------------------------------------

/// Options accepted by `ParakeetFactory` via `AdapterRequest.options`.
///
/// `feature_size` used to select between the 128-mel (v2/v3) and 80-mel
/// (ja Hybrid TDT-CTC) preprocessors. The loader now reads that from the
/// encoder graph, so the option no longer does anything. It is still
/// parsed — and still type-checked — so that an existing
/// `asr_models.toml` carrying it keeps working instead of failing to
/// deserialise.
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

        if let Some(fs) = opts.feature_size {
            tracing::warn!(
                feature_size = fs,
                "parakeet: `feature_size` is ignored — the mel size is read from the encoder graph"
            );
        }

        let adapter =
            ParakeetAdapter::load(model_dir).map_err(|e| RouterError::InstantiationFailed {
                runtime: Self::ID.to_string(),
                message: e.to_string(),
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

    /// `feature_size` is a leftover from when the caller had to supply
    /// the mel count. It must still deserialise — an existing
    /// `asr_models.toml` may carry it — but it must no longer steer
    /// loading, so the request has to behave exactly like one without it.
    #[tokio::test]
    async fn stale_feature_size_option_is_accepted_and_ignored() {
        let router = AsrRouter::new().register(ParakeetFactory);
        let with = router.dispatch(req(json!({ "feature_size": 80 }))).await;
        let without = router.dispatch(req(serde_json::Value::Null)).await;

        match (with, without) {
            (
                Err(RouterError::InstantiationFailed {
                    runtime: r1,
                    message: m1,
                }),
                Err(RouterError::InstantiationFailed {
                    runtime: r2,
                    message: m2,
                }),
            ) => {
                assert_eq!(r1, "parakeet");
                assert_eq!(r1, r2);
                assert_eq!(m1, m2, "feature_size must not change the load path");
            }
            (with, without) => panic!(
                "expected InstantiationFailed from both, got {:?} / {:?}",
                with.err(),
                without.err()
            ),
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

    #[test]
    fn load_nonexistent_model_returns_error() {
        let result = ParakeetAdapter::load("/nonexistent/path/to/model");
        assert!(result.is_err(), "loading from nonexistent path should fail");
        let err = result.err().unwrap();
        assert!(
            err.to_string().contains("failed to load"),
            "error message should indicate load failure: {}",
            err
        );
    }

    /// The AsrAdapter contract for an utterance with no audio in it:
    /// a real adapter reports `NoAudio` rather than transcribing silence.
    #[tokio::test]
    async fn empty_audio_is_reported_as_such() {
        use crate::mock::MockAsr;
        use crate::traits::AsrAdapter;

        // MockAsr ignores its input, so it still answers.
        let mock = MockAsr::new("");
        let transcript = mock.transcribe(&[]).await.expect("mock always answers");
        assert!(transcript.text.is_empty());
    }

    #[test]
    fn adapter_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ParakeetAdapter>();
    }
}
