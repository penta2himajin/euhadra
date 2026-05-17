//! `FasterWhisperAdapter` — Whisper inference via CTranslate2 (`ct2rs`).
//!
//! Loads a CTranslate2-format Whisper checkpoint (e.g.
//! `deepdml/faster-whisper-large-v3-turbo-ct2`) and runs inference
//! in-process through the vendored `libctranslate2` build. Compared to
//! the `whisper-rs` GGML path:
//!
//! - Model artefact is a CTranslate2 directory bundle (`model.bin` plus `config.json`, `tokenizer.json`, `vocabulary.json`, `preprocessor_config.json`), not a single GGML file.
//! - Inference engine is CTranslate2's hand-tuned CPU kernels rather than whisper.cpp's ggml. INT8 quantisation is selected at CT2 conversion time and stored inside the bundle.
//! - Same `AsrAdapter` contract: 16 kHz mono f32, one final result per session.
//!
//! Feature: `ctranslate2`. The `ct2rs` crate builds libctranslate2 from
//! source, so consumers need `cmake` + a C++17 compiler at the time
//! this feature is enabled.

use async_trait::async_trait;
use ct2rs::{Whisper, WhisperOptions};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::router::{AdapterRequest, AsrRuntimeFactory, ModelSource, RouterError};
use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AsrResult, AudioChunk};

/// Optional builder knobs for [`FasterWhisperAdapter`].
#[derive(Debug, Clone, Default)]
pub struct FasterWhisperConfig {
    /// Language hint forwarded to `Whisper::generate`. `None` enables
    /// Whisper's built-in LID.
    pub language: Option<String>,
    /// Whether to include word-level timestamps. Not used for euhadra's
    /// final-only `AsrAdapter` contract yet, but exposed for future
    /// streaming work.
    pub return_timestamps: bool,
}

/// `AsrAdapter` for CTranslate2-format Whisper models.
pub struct FasterWhisperAdapter {
    inner: Arc<Whisper>,
    cfg: FasterWhisperConfig,
    #[allow(dead_code)]
    model_dir: PathBuf,
}

impl FasterWhisperAdapter {
    /// Load a CTranslate2 Whisper bundle directory with default config.
    pub fn load(model_dir: impl Into<PathBuf>) -> Result<Self, AsrError> {
        Self::load_with_config(model_dir, FasterWhisperConfig::default())
    }

    pub fn load_with_config(
        model_dir: impl Into<PathBuf>,
        cfg: FasterWhisperConfig,
    ) -> Result<Self, AsrError> {
        let dir = model_dir.into();
        let whisper = Whisper::new(&dir, Default::default()).map_err(|e| AsrError {
            message: format!("failed to load ct2rs Whisper at {}: {e}", dir.display()),
        })?;
        Ok(Self {
            inner: Arc::new(whisper),
            cfg,
            model_dir: dir,
        })
    }

    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.cfg.language = Some(lang.into());
        self
    }

    pub fn sampling_rate(&self) -> usize {
        self.inner.sampling_rate()
    }

    /// Single-shot transcription on a mono f32 buffer at the model's
    /// expected sampling rate (16 kHz for all Whisper variants). Public
    /// so benchmarks can drive the adapter without `AsrAdapter` channel
    /// plumbing.
    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<String, AsrError> {
        let lang = self.cfg.language.as_deref();
        let segments = self
            .inner
            .generate(
                samples,
                lang,
                self.cfg.return_timestamps,
                &WhisperOptions::default(),
            )
            .map_err(|e| AsrError {
                message: format!("ct2rs Whisper::generate failed: {e}"),
            })?;
        Ok(segments.join("").trim().to_string())
    }
}

#[async_trait]
impl AsrAdapter for FasterWhisperAdapter {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        result_tx: mpsc::Sender<AsrResult>,
    ) -> Result<(), AsrError> {
        let expected_sr = self.sampling_rate() as u32;
        let mut samples: Vec<f32> = Vec::new();
        while let Some(chunk) = audio_rx.recv().await {
            if chunk.sample_rate != expected_sr {
                return Err(AsrError {
                    message: format!(
                        "ct2rs Whisper expects {expected_sr} Hz audio; got {} Hz",
                        chunk.sample_rate
                    ),
                });
            }
            samples.extend(&chunk.samples);
        }
        if samples.is_empty() {
            return Err(AsrError {
                message: "no audio received".into(),
            });
        }

        // CPU-bound — keep it off the async runtime executor.
        let inner = self.inner.clone();
        let cfg = self.cfg.clone();
        let text = tokio::task::spawn_blocking(move || -> Result<String, AsrError> {
            let segments = inner
                .generate(
                    &samples,
                    cfg.language.as_deref(),
                    cfg.return_timestamps,
                    &WhisperOptions::default(),
                )
                .map_err(|e| AsrError {
                    message: format!("ct2rs Whisper::generate failed: {e}"),
                })?;
            Ok(segments.join("").trim().to_string())
        })
        .await
        .map_err(|e| AsrError {
            message: format!("transcribe task panicked: {e}"),
        })??;

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
                    message: format!("failed to send result: {e}"),
                })?;
        }
        Ok(())
    }
}

/// Convenience wrapper for one-off transcriptions in examples / tests.
pub fn transcribe_samples(
    model_dir: impl AsRef<Path>,
    samples: &[f32],
    language: Option<&str>,
) -> Result<String, AsrError> {
    let cfg = FasterWhisperConfig {
        language: language.map(str::to_owned),
        return_timestamps: false,
    };
    FasterWhisperAdapter::load_with_config(model_dir.as_ref().to_path_buf(), cfg)?
        .transcribe_samples(samples)
}

// ---------------------------------------------------------------------------
// Router factory
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct FasterWhisperOptions {
    /// Emit timestamps along with the transcript. Defaults to false.
    #[serde(default)]
    timestamps: Option<bool>,
}

/// Router factory that builds `FasterWhisperAdapter` via `AsrRouter`.
///
/// Registered under runtime id `"faster-whisper"`. Menura passes the
/// CTranslate2 model directory via `AdapterRequest.model_source.LocalPath`
/// and the BCP 47 language tag via `AdapterRequest.language`.
pub struct FasterWhisperFactory;

impl FasterWhisperFactory {
    pub const ID: &'static str = "faster-whisper";
}

#[async_trait]
impl AsrRuntimeFactory for FasterWhisperFactory {
    fn id(&self) -> &'static str {
        Self::ID
    }

    async fn instantiate(&self, req: &AdapterRequest) -> Result<Arc<dyn AsrAdapter>, RouterError> {
        let ModelSource::LocalPath(model_dir) = &req.model_source;

        let opts: FasterWhisperOptions = if req.options.is_null() {
            FasterWhisperOptions::default()
        } else {
            serde_json::from_value(req.options.clone()).map_err(|e| {
                RouterError::InvalidRequest(format!("faster-whisper options parse error: {e}"))
            })?
        };

        let mut cfg = FasterWhisperConfig::default();
        if let Some(ts) = opts.timestamps {
            cfg.return_timestamps = ts;
        }
        if !req.language.is_empty() {
            cfg.language = Some(req.language.clone());
        }

        let model_dir = model_dir.clone();
        let path_for_err = model_dir.clone();
        let cfg_for_load = cfg.clone();
        let adapter = tokio::task::spawn_blocking(move || {
            FasterWhisperAdapter::load_with_config(model_dir, cfg_for_load)
        })
        .await
        .map_err(|e| RouterError::InstantiationFailed {
            runtime: Self::ID.to_string(),
            message: format!("faster-whisper load task panicked at {path_for_err:?}: {e}"),
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
            runtime: FasterWhisperFactory::ID.into(),
            model_source: ModelSource::LocalPath(PathBuf::from("/nonexistent/faster-whisper")),
            options,
        }
    }

    #[tokio::test]
    async fn factory_id_matches_constant() {
        assert_eq!(FasterWhisperFactory.id(), "faster-whisper");
    }

    #[tokio::test]
    async fn dispatch_with_missing_bundle_returns_instantiation_failed() {
        let router = AsrRouter::new().register(FasterWhisperFactory);
        match router.dispatch(req("ko", serde_json::Value::Null)).await {
            Err(RouterError::InstantiationFailed { runtime, .. }) => {
                assert_eq!(runtime, "faster-whisper");
            }
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected loader error"),
        }
    }

    #[tokio::test]
    async fn malformed_options_return_invalid_request() {
        let router = AsrRouter::new().register(FasterWhisperFactory);
        match router
            .dispatch(req("ko", json!({ "timestamps": "yes" })))
            .await
        {
            Err(RouterError::InvalidRequest(msg)) => {
                assert!(msg.contains("faster-whisper"));
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("expected parse error"),
        }
    }
}
