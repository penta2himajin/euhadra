//! Router factory wrapping `ReazonAdapter`.
//!
//! Registered under runtime id `"reazon"`. Menura supplies the model
//! bundle directory via `AdapterRequest.model_source.LocalPath` (the
//! Zipformer bundle laid out by `scripts/setup_reazon_ja.sh`).
//!
//! `AdapterRequest.language` is accepted and ignored — the transducer
//! has no language prompt; routing is done upstream via
//! `asr_models.toml`.

use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

use crate::router::{AdapterRequest, AsrRuntimeFactory, ModelSource, RouterError};
use crate::traits::AsrAdapter;

use super::adapter::{ReazonAdapter, ReazonConfig};

#[derive(Debug, Default, Deserialize)]
struct ReazonOptions {
    #[serde(default)]
    encoder_file: Option<String>,
    #[serde(default)]
    decoder_file: Option<String>,
    #[serde(default)]
    joiner_file: Option<String>,
}

/// Router factory that builds `ReazonAdapter` via `AsrRouter`.
pub struct ReazonFactory;

impl ReazonFactory {
    pub const ID: &'static str = "reazon";
}

#[async_trait]
impl AsrRuntimeFactory for ReazonFactory {
    fn id(&self) -> &'static str {
        Self::ID
    }

    async fn instantiate(&self, req: &AdapterRequest) -> Result<Arc<dyn AsrAdapter>, RouterError> {
        let ModelSource::LocalPath(model_dir) = &req.model_source;

        let opts: ReazonOptions = if req.options.is_null() {
            ReazonOptions::default()
        } else {
            serde_json::from_value(req.options.clone()).map_err(|e| {
                RouterError::InvalidRequest(format!("reazon options parse error: {e}"))
            })?
        };

        let mut cfg = ReazonConfig::default();
        if let Some(f) = opts.encoder_file {
            cfg.encoder_file = f;
        }
        if let Some(f) = opts.decoder_file {
            cfg.decoder_file = f;
        }
        if let Some(f) = opts.joiner_file {
            cfg.joiner_file = f;
        }

        let adapter = ReazonAdapter::load_with_config(model_dir, cfg).map_err(|e| {
            RouterError::InstantiationFailed {
                runtime: Self::ID.to_string(),
                message: e.to_string(),
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

    fn req(options: serde_json::Value) -> AdapterRequest {
        AdapterRequest {
            language: "ja".into(),
            runtime: ReazonFactory::ID.into(),
            model_source: ModelSource::LocalPath(PathBuf::from("/nonexistent/reazon/bundle")),
            options,
        }
    }

    #[tokio::test]
    async fn factory_id_matches_constant() {
        assert_eq!(ReazonFactory.id(), "reazon");
    }

    #[tokio::test]
    async fn dispatch_with_missing_bundle_returns_instantiation_failed() {
        let router = AsrRouter::new().register(ReazonFactory);
        match router.dispatch(req(serde_json::Value::Null)).await {
            Err(RouterError::InstantiationFailed { runtime, message }) => {
                assert_eq!(runtime, "reazon");
                assert!(
                    message.contains("tokens") || message.contains("load"),
                    "unexpected message: {message}"
                );
            }
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected error when bundle dir does not exist"),
        }
    }

    #[tokio::test]
    async fn malformed_options_return_invalid_request() {
        let router = AsrRouter::new().register(ReazonFactory);
        match router.dispatch(req(json!({ "encoder_file": 7 }))).await {
            Err(RouterError::InvalidRequest(msg)) => {
                assert!(msg.contains("reazon options"), "unexpected message: {msg}");
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("expected error for malformed options"),
        }
    }
}
