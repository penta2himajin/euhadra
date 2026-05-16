//! Router factory wrapping `SenseVoiceAdapter`.
//!
//! Registered under runtime id `"sensevoice"`. Menura supplies the model
//! bundle directory via `AdapterRequest.model_source.LocalPath` (the
//! INT8 bundle laid out by `scripts/setup_sensevoice.sh`), the BCP 47
//! language tag via `AdapterRequest.language`, and an optional
//! `with_itn` override via `AdapterRequest.options`.

use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

use crate::router::{AdapterRequest, AsrRuntimeFactory, ModelSource, RouterError};
use crate::traits::AsrAdapter;

use super::adapter::{SenseVoiceAdapter, SenseVoiceConfig};

/// Options accepted by `SenseVoiceFactory` via `AdapterRequest.options`.
///
/// All fields are optional; missing fields fall back to
/// `SenseVoiceConfig::default()`. Today the only meaningful toggle is
/// `with_itn` because the upstream bundle is INT8-only (FP32 was
/// retired in issue #59 Phase 2).
#[derive(Debug, Default, Deserialize)]
struct SenseVoiceOptions {
    /// Apply inverse text normalisation on the raw transcript
    /// (e.g. `"열두"` → `"12"`). `None` leaves
    /// `SenseVoiceConfig::default_use_itn` untouched.
    #[serde(default)]
    with_itn: Option<bool>,
}

/// Router factory that builds `SenseVoiceAdapter` via `AsrRouter`.
pub struct SenseVoiceFactory;

impl SenseVoiceFactory {
    pub const ID: &'static str = "sensevoice";
}

#[async_trait]
impl AsrRuntimeFactory for SenseVoiceFactory {
    fn id(&self) -> &'static str {
        Self::ID
    }

    async fn instantiate(&self, req: &AdapterRequest) -> Result<Arc<dyn AsrAdapter>, RouterError> {
        let ModelSource::LocalPath(model_dir) = &req.model_source;

        let opts: SenseVoiceOptions = if req.options.is_null() {
            SenseVoiceOptions::default()
        } else {
            serde_json::from_value(req.options.clone()).map_err(|e| {
                RouterError::InvalidRequest(format!("sensevoice options parse error: {e}"))
            })?
        };

        let mut cfg = SenseVoiceConfig::default();
        if let Some(use_itn) = opts.with_itn {
            cfg.default_use_itn = use_itn;
        }

        let mut adapter = SenseVoiceAdapter::load_with_config(model_dir, cfg).map_err(|e| {
            RouterError::InstantiationFailed {
                runtime: Self::ID.to_string(),
                message: e.message,
            }
        })?;
        if !req.language.is_empty() {
            adapter = adapter.with_language(req.language.clone());
        }
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
            runtime: SenseVoiceFactory::ID.into(),
            model_source: ModelSource::LocalPath(PathBuf::from("/nonexistent/sensevoice/bundle")),
            options,
        }
    }

    #[tokio::test]
    async fn factory_id_matches_constant() {
        let f = SenseVoiceFactory;
        assert_eq!(f.id(), "sensevoice");
    }

    #[tokio::test]
    async fn dispatch_with_missing_bundle_returns_instantiation_failed() {
        let router = AsrRouter::new().register(SenseVoiceFactory);
        match router.dispatch(req("ko", serde_json::Value::Null)).await {
            Err(RouterError::InstantiationFailed { runtime, message }) => {
                assert_eq!(runtime, "sensevoice");
                assert!(
                    !message.is_empty(),
                    "expected non-empty error message from adapter load"
                );
            }
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected error when bundle dir does not exist"),
        }
    }

    #[tokio::test]
    async fn malformed_options_return_invalid_request() {
        let router = AsrRouter::new().register(SenseVoiceFactory);
        // `with_itn` typed as a string instead of bool.
        match router
            .dispatch(req("ko", json!({ "with_itn": "yes" })))
            .await
        {
            Err(RouterError::InvalidRequest(msg)) => {
                assert!(
                    msg.contains("sensevoice"),
                    "error message should mention sensevoice, got: {msg}"
                );
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("expected error for malformed options"),
        }
    }

    #[tokio::test]
    async fn well_formed_options_propagate_to_adapter_load() {
        // Even though the load fails (no bundle on disk), we should
        // still pass the parse step — i.e. NOT see InvalidRequest.
        let router = AsrRouter::new().register(SenseVoiceFactory);
        match router
            .dispatch(req("ja", json!({ "with_itn": false })))
            .await
        {
            Err(RouterError::InstantiationFailed { .. }) => {}
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected error when bundle dir does not exist"),
        }
    }
}
