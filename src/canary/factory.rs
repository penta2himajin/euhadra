//! Router factory wrapping `CanaryAdapter`.
//!
//! Registered under runtime id `"canary"`. Menura supplies the model
//! bundle directory via `AdapterRequest.model_source.LocalPath`, the
//! BCP 47 language tag via `AdapterRequest.language` (one of
//! `en`/`de`/`fr`/`es`), and an optional `int8` toggle via
//! `AdapterRequest.options` to load the quantised weights instead of
//! the FP32 ones.

use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

use crate::router::{AdapterRequest, AsrRuntimeFactory, ModelSource, RouterError};
use crate::traits::AsrAdapter;

use super::adapter::{CanaryAdapter, CanaryConfig};

/// Options accepted by `CanaryFactory` via `AdapterRequest.options`.
#[derive(Debug, Default, Deserialize)]
struct CanaryOptions {
    /// Load the INT8-quantised encoder/decoder pair instead of FP32.
    /// `None` keeps the `CanaryConfig::default()` filenames (FP32).
    #[serde(default)]
    int8: Option<bool>,
}

/// Router factory that builds `CanaryAdapter` via `AsrRouter`.
pub struct CanaryFactory;

impl CanaryFactory {
    pub const ID: &'static str = "canary";
}

#[async_trait]
impl AsrRuntimeFactory for CanaryFactory {
    fn id(&self) -> &'static str {
        Self::ID
    }

    async fn instantiate(&self, req: &AdapterRequest) -> Result<Arc<dyn AsrAdapter>, RouterError> {
        let ModelSource::LocalPath(model_dir) = &req.model_source;

        let opts: CanaryOptions = if req.options.is_null() {
            CanaryOptions::default()
        } else {
            serde_json::from_value(req.options.clone()).map_err(|e| {
                RouterError::InvalidRequest(format!("canary options parse error: {e}"))
            })?
        };

        let mut cfg = CanaryConfig::default();
        if opts.int8 == Some(true) {
            cfg = cfg.with_int8_weights();
        }

        let mut adapter = CanaryAdapter::load_with_config(model_dir, cfg).map_err(|e| {
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
            runtime: CanaryFactory::ID.into(),
            model_source: ModelSource::LocalPath(PathBuf::from("/nonexistent/canary/bundle")),
            options,
        }
    }

    #[tokio::test]
    async fn factory_id_matches_constant() {
        assert_eq!(CanaryFactory.id(), "canary");
    }

    #[tokio::test]
    async fn dispatch_with_missing_bundle_returns_instantiation_failed() {
        let router = AsrRouter::new().register(CanaryFactory);
        match router.dispatch(req("en", serde_json::Value::Null)).await {
            Err(RouterError::InstantiationFailed { runtime, .. }) => {
                assert_eq!(runtime, "canary");
            }
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected error when bundle dir does not exist"),
        }
    }

    #[tokio::test]
    async fn int8_option_is_accepted() {
        let router = AsrRouter::new().register(CanaryFactory);
        match router.dispatch(req("es", json!({ "int8": true }))).await {
            Err(RouterError::InstantiationFailed { .. }) => {}
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected loader error"),
        }
    }

    #[tokio::test]
    async fn malformed_options_return_invalid_request() {
        let router = AsrRouter::new().register(CanaryFactory);
        match router.dispatch(req("en", json!({ "int8": "yes" }))).await {
            Err(RouterError::InvalidRequest(msg)) => {
                assert!(msg.contains("canary"));
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("expected error for malformed options"),
        }
    }
}
