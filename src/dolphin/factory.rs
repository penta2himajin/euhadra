//! Router factory wrapping `DolphinAdapter`.
//!
//! Registered under runtime id `"dolphin"`. Menura supplies the model
//! bundle directory via `AdapterRequest.model_source.LocalPath` (the
//! CTC bundle laid out by `scripts/setup_dolphin_ko.sh`) and an
//! optional `model_file` override via `AdapterRequest.options`.
//!
//! `AdapterRequest.language` is accepted and ignored. Dolphin's CTC
//! branch has no decoder prompt and no language input — the graph takes
//! `(x, x_len)` and nothing else, and its language tags (`<ko>`, `<ja>`
//! …) are output symbols the decoder may emit, not a selector. Menura
//! should still route per-language correctly through
//! `asr_models.toml`; the tag simply has nowhere to go at this layer,
//! the same as for `paraformer`.

use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

use crate::router::{AdapterRequest, AsrRuntimeFactory, ModelSource, RouterError};
use crate::traits::AsrAdapter;

use super::adapter::{DolphinAdapter, DolphinConfig};

/// Options accepted by `DolphinFactory` via `AdapterRequest.options`.
///
/// All fields are optional; missing fields fall back to
/// `DolphinConfig::default()`.
#[derive(Debug, Default, Deserialize)]
struct DolphinOptions {
    /// Model filename inside the bundle directory, for pointing at a
    /// non-default export (`model.onnx` rather than the INT8 graph, or
    /// a `base`/`medium` bundle laid out the same way).
    #[serde(default)]
    model_file: Option<String>,
}

/// Router factory that builds `DolphinAdapter` via `AsrRouter`.
pub struct DolphinFactory;

impl DolphinFactory {
    pub const ID: &'static str = "dolphin";
}

#[async_trait]
impl AsrRuntimeFactory for DolphinFactory {
    fn id(&self) -> &'static str {
        Self::ID
    }

    async fn instantiate(&self, req: &AdapterRequest) -> Result<Arc<dyn AsrAdapter>, RouterError> {
        let ModelSource::LocalPath(model_dir) = &req.model_source;

        let opts: DolphinOptions = if req.options.is_null() {
            DolphinOptions::default()
        } else {
            serde_json::from_value(req.options.clone()).map_err(|e| {
                RouterError::InvalidRequest(format!("dolphin options parse error: {e}"))
            })?
        };

        let mut cfg = DolphinConfig::default();
        if let Some(model_file) = opts.model_file {
            cfg.model_file = model_file;
        }

        let adapter = DolphinAdapter::load_with_config(model_dir, cfg).map_err(|e| {
            RouterError::InstantiationFailed {
                runtime: Self::ID.to_string(),
                message: e.message,
            }
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
            runtime: DolphinFactory::ID.into(),
            model_source: ModelSource::LocalPath(PathBuf::from("/nonexistent/dolphin/bundle")),
            options,
        }
    }

    #[tokio::test]
    async fn factory_id_matches_constant() {
        // This string is a published contract: Menura's asr_models.toml
        // names it, so changing it breaks deployed config silently.
        assert_eq!(DolphinFactory.id(), "dolphin");
    }

    #[tokio::test]
    async fn dispatch_with_missing_bundle_returns_instantiation_failed() {
        // Reaching the adapter loader is the point: an earlier
        // UnknownRuntime or InvalidRequest would mean the registration
        // itself is broken.
        let router = AsrRouter::new().register(DolphinFactory);
        match router.dispatch(req("ko", serde_json::Value::Null)).await {
            Err(RouterError::InstantiationFailed { runtime, message }) => {
                assert_eq!(runtime, "dolphin");
                assert!(message.contains("tokens.txt"), "unexpected message: {message}");
            }
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected error when bundle dir does not exist"),
        }
    }

    #[tokio::test]
    async fn malformed_options_return_invalid_request() {
        let router = AsrRouter::new().register(DolphinFactory);
        // `model_file` typed as a number instead of a string.
        match router.dispatch(req("ko", json!({ "model_file": 7 }))).await {
            Err(RouterError::InvalidRequest(msg)) => {
                assert!(msg.contains("dolphin options"), "unexpected message: {msg}");
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("expected error for malformed options"),
        }
    }

    #[tokio::test]
    async fn unknown_options_are_ignored_rather_than_rejected() {
        // Menura may carry keys this runtime does not know; rejecting
        // them would make every shared option block a breaking change.
        let router = AsrRouter::new().register(DolphinFactory);
        match router.dispatch(req("ko", json!({ "with_itn": true }))).await {
            Err(RouterError::InstantiationFailed { .. }) => {}
            Err(other) => panic!("expected the request to reach the loader, got {other:?}"),
            Ok(_) => panic!("expected the missing bundle to fail at load"),
        }
    }

    #[tokio::test]
    async fn an_empty_language_is_accepted() {
        // The graph has no language input, so a missing tag must not be
        // an error the way it would be for SenseVoice.
        let router = AsrRouter::new().register(DolphinFactory);
        match router.dispatch(req("", serde_json::Value::Null)).await {
            Err(RouterError::InstantiationFailed { .. }) => {}
            Err(other) => panic!("expected the request to reach the loader, got {other:?}"),
            Ok(_) => panic!("expected the missing bundle to fail at load"),
        }
    }
}
