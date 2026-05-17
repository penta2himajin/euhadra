//! `WhisperRsAdapter` — in-process whisper.cpp via the `whisper-rs` crate.
//!
//! Replaces the subprocess-per-utterance pattern of `WhisperLocal` with a
//! Rust-callable inference loop. The model is loaded once on adapter
//! construction; each `transcribe()` call creates a fresh per-session
//! `WhisperState` and runs `full()` on the accumulated audio.
//!
//! Compared to `WhisperLocal`:
//!
//! - No `whisper-cli` binary required (kernels are vendored by
//!   `whisper-rs` and built from source at compile time).
//! - No subprocess startup cost per transcription. Model load is
//!   amortised across many sessions.
//! - Cancellation can be wired through the `abort_callback` parameter
//!   in the future; not exposed here yet.
//!
//! Same model artefact as `WhisperLocal`: any GGML/whisper.cpp
//! checkpoint loads, including the `large-v3-turbo` Q4_0 / Q5_0
//! quantisations that euhadra targets for Korean (see
//! `docs/korean-asr-alternatives.md`).

use async_trait::async_trait;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::router::{AdapterRequest, AsrRuntimeFactory, ModelSource, RouterError};
use crate::traits::{AsrAdapter, AsrError};
use crate::types::{AsrResult, AudioChunk};

/// Optional builder knobs for [`WhisperRsAdapter`]. All fields default
/// to values that match `whisper-cli`'s out-of-the-box behaviour.
#[derive(Debug, Clone)]
pub struct WhisperRsConfig {
    /// CTranslate2-style language hint (`"en"`, `"ja"`, `"ko"`, ...).
    /// `None` enables Whisper's built-in LID.
    pub language: Option<String>,
    /// CPU threads for the inference graph. `0` lets `whisper-rs`
    /// auto-detect (typically all available cores).
    pub threads: i32,
    /// Disable the temperature fallback loop (mirrors `--no-fallback`
    /// in `whisper-cli`).
    pub no_fallback: bool,
}

impl Default for WhisperRsConfig {
    fn default() -> Self {
        Self {
            language: None,
            threads: 4,
            no_fallback: false,
        }
    }
}

/// `AsrAdapter` backed by an in-process `whisper.cpp` GGML model.
///
/// The model is loaded eagerly in [`WhisperRsAdapter::load`] and shared
/// across all `transcribe()` calls. `WhisperContext` is `Send + Sync`,
/// so no further wrapping is needed.
pub struct WhisperRsAdapter {
    ctx: Arc<WhisperContext>,
    cfg: WhisperRsConfig,
    /// Stored only for diagnostics — the model is owned by `ctx`.
    #[allow(dead_code)]
    model_path: PathBuf,
}

impl WhisperRsAdapter {
    /// Load a GGML/whisper.cpp checkpoint at `model_path` with the
    /// default configuration.
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self, AsrError> {
        Self::load_with_config(model_path, WhisperRsConfig::default())
    }

    pub fn load_with_config(
        model_path: impl Into<PathBuf>,
        cfg: WhisperRsConfig,
    ) -> Result<Self, AsrError> {
        let path = model_path.into();
        let path_str = path.to_string_lossy().into_owned();
        let ctx_params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(&path_str, ctx_params).map_err(|e| AsrError {
            message: format!("failed to load whisper-rs model {path_str}: {e}"),
        })?;
        Ok(Self {
            ctx: Arc::new(ctx),
            cfg,
            model_path: path,
        })
    }

    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.cfg.language = Some(lang.into());
        self
    }

    /// Run the model on a single concatenated `f32` buffer. Public so
    /// benchmarks can drive the adapter without spinning up the
    /// `AsrAdapter` channel plumbing.
    pub fn transcribe_samples_public(&self, samples: &[f32]) -> Result<String, AsrError> {
        self.transcribe_samples(samples)
    }

    fn transcribe_samples(&self, samples: &[f32]) -> Result<String, AsrError> {
        let mut state = self.ctx.create_state().map_err(|e| AsrError {
            message: format!("failed to create whisper state: {e}"),
        })?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.cfg.threads);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params.set_no_context(true);
        if self.cfg.no_fallback {
            params.set_temperature_inc(0.0);
        }
        if let Some(lang) = self.cfg.language.as_deref() {
            params.set_language(Some(lang));
        }

        state.full(params, samples).map_err(|e| AsrError {
            message: format!("whisper-rs full() failed: {e}"),
        })?;

        let n = state.full_n_segments().map_err(|e| AsrError {
            message: format!("whisper-rs full_n_segments() failed: {e}"),
        })?;
        let mut out = String::new();
        for i in 0..n {
            let seg = state.full_get_segment_text(i).map_err(|e| AsrError {
                message: format!("whisper-rs full_get_segment_text({i}) failed: {e}"),
            })?;
            out.push_str(&seg);
        }
        Ok(out.trim().to_string())
    }
}

#[async_trait]
impl AsrAdapter for WhisperRsAdapter {
    async fn transcribe(
        &self,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        result_tx: mpsc::Sender<AsrResult>,
    ) -> Result<(), AsrError> {
        let mut samples: Vec<f32> = Vec::new();
        while let Some(chunk) = audio_rx.recv().await {
            // whisper.cpp expects 16 kHz mono f32. Callers are responsible
            // for resampling; if a mismatch slips through, fail fast so
            // diagnostics are clear.
            if chunk.sample_rate != 16000 {
                return Err(AsrError {
                    message: format!(
                        "whisper-rs expects 16 kHz audio; got {} Hz",
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
        let ctx = self.ctx.clone();
        let cfg = self.cfg.clone();
        let model_path = self.model_path.clone();
        let text = tokio::task::spawn_blocking(move || {
            WhisperRsAdapter {
                ctx,
                cfg,
                model_path,
            }
            .transcribe_samples(&samples)
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

/// Convenience: run the model on a single concatenated `f32` buffer.
/// Used by examples and microbenches that don't need the streaming
/// `AsrAdapter` plumbing.
pub fn transcribe_samples(
    model_path: impl AsRef<Path>,
    samples: &[f32],
    language: Option<&str>,
    threads: i32,
) -> Result<String, AsrError> {
    let cfg = WhisperRsConfig {
        language: language.map(str::to_owned),
        threads,
        no_fallback: false,
    };
    WhisperRsAdapter::load_with_config(model_path.as_ref().to_path_buf(), cfg)?
        .transcribe_samples(samples)
}

// ---------------------------------------------------------------------------
// Router factory
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct WhisperRsOptions {
    /// CPU threads. Defaults to 4 if absent.
    #[serde(default)]
    threads: Option<i32>,
    /// Disable the temperature fallback loop. Default: keep enabled.
    #[serde(default)]
    no_fallback: Option<bool>,
}

/// Router factory that builds `WhisperRsAdapter` via `AsrRouter`.
///
/// Registered under runtime id `"whisper-rs"`. Compared to the
/// `whisper-local` factory, this one loads the GGML model in-process
/// and avoids a `whisper-cli` subprocess per session.
pub struct WhisperRsFactory;

impl WhisperRsFactory {
    pub const ID: &'static str = "whisper-rs";
}

#[async_trait]
impl AsrRuntimeFactory for WhisperRsFactory {
    fn id(&self) -> &'static str {
        Self::ID
    }

    async fn instantiate(&self, req: &AdapterRequest) -> Result<Arc<dyn AsrAdapter>, RouterError> {
        let ModelSource::LocalPath(model_path) = &req.model_source;

        let opts: WhisperRsOptions = if req.options.is_null() {
            WhisperRsOptions::default()
        } else {
            serde_json::from_value(req.options.clone()).map_err(|e| {
                RouterError::InvalidRequest(format!("whisper-rs options parse error: {e}"))
            })?
        };

        let mut cfg = WhisperRsConfig::default();
        if let Some(t) = opts.threads {
            cfg.threads = t;
        }
        if let Some(nf) = opts.no_fallback {
            cfg.no_fallback = nf;
        }
        if !req.language.is_empty() {
            cfg.language = Some(req.language.clone());
        }

        // The whisper-rs load happens on a blocking thread because
        // it does real disk I/O + kernel allocation. Keep the async
        // runtime responsive.
        let model_path = model_path.clone();
        let path_for_err = model_path.clone();
        let cfg_for_load = cfg.clone();
        let adapter = tokio::task::spawn_blocking(move || {
            WhisperRsAdapter::load_with_config(model_path, cfg_for_load)
        })
        .await
        .map_err(|e| RouterError::InstantiationFailed {
            runtime: Self::ID.to_string(),
            message: format!("whisper-rs load task panicked at {path_for_err:?}: {e}"),
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
            runtime: WhisperRsFactory::ID.into(),
            model_source: ModelSource::LocalPath(PathBuf::from(
                "/nonexistent/whisper-rs/model.bin",
            )),
            options,
        }
    }

    #[tokio::test]
    async fn factory_id_matches_constant() {
        assert_eq!(WhisperRsFactory.id(), "whisper-rs");
    }

    #[tokio::test]
    async fn dispatch_with_missing_model_returns_instantiation_failed() {
        let router = AsrRouter::new().register(WhisperRsFactory);
        match router.dispatch(req("ko", serde_json::Value::Null)).await {
            Err(RouterError::InstantiationFailed { runtime, .. }) => {
                assert_eq!(runtime, "whisper-rs");
            }
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected loader error"),
        }
    }

    #[tokio::test]
    async fn malformed_options_return_invalid_request() {
        let router = AsrRouter::new().register(WhisperRsFactory);
        match router
            .dispatch(req("ko", json!({ "threads": "many" })))
            .await
        {
            Err(RouterError::InvalidRequest(msg)) => {
                assert!(msg.contains("whisper-rs"));
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("expected parse error"),
        }
    }
}
