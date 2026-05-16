//! Router factory wrapping `ParaformerAdapter`.
//!
//! Registered under runtime id `"paraformer"`. Menura supplies the
//! model bundle directory via `AdapterRequest.model_source.LocalPath`.
//! Paraformer-large is a Chinese-only checkpoint, so the request's
//! language is ignored at the adapter layer; Menura should still route
//! per-language correctly via `asr_models.toml`.

use async_trait::async_trait;
use std::sync::Arc;

use crate::router::{AdapterRequest, AsrRuntimeFactory, ModelSource, RouterError};
use crate::traits::AsrAdapter;

use super::adapter::ParaformerAdapter;

/// Router factory that builds `ParaformerAdapter` via `AsrRouter`.
pub struct ParaformerFactory;

impl ParaformerFactory {
    pub const ID: &'static str = "paraformer";
}

#[async_trait]
impl AsrRuntimeFactory for ParaformerFactory {
    fn id(&self) -> &'static str {
        Self::ID
    }

    async fn instantiate(&self, req: &AdapterRequest) -> Result<Arc<dyn AsrAdapter>, RouterError> {
        let ModelSource::LocalPath(model_dir) = &req.model_source;

        // Reject non-empty options for now — there are no toggles
        // exposed today. A future `model_filename` override (e.g. for
        // the 8358 hot-word variant) can add fields here without
        // breaking existing callers.
        if !req.options.is_null()
            && !matches!(&req.options, serde_json::Value::Object(o) if o.is_empty())
        {
            return Err(RouterError::InvalidRequest(format!(
                "paraformer accepts no options today; got: {}",
                req.options
            )));
        }

        let adapter =
            ParaformerAdapter::load(model_dir).map_err(|e| RouterError::InstantiationFailed {
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

    fn req(options: serde_json::Value) -> AdapterRequest {
        AdapterRequest {
            language: "zh".into(),
            runtime: ParaformerFactory::ID.into(),
            model_source: ModelSource::LocalPath(PathBuf::from("/nonexistent/paraformer/bundle")),
            options,
        }
    }

    #[tokio::test]
    async fn factory_id_matches_constant() {
        assert_eq!(ParaformerFactory.id(), "paraformer");
    }

    #[tokio::test]
    async fn dispatch_with_missing_bundle_returns_instantiation_failed() {
        let router = AsrRouter::new().register(ParaformerFactory);
        match router.dispatch(req(serde_json::Value::Null)).await {
            Err(RouterError::InstantiationFailed { runtime, .. }) => {
                assert_eq!(runtime, "paraformer");
            }
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected error when bundle dir does not exist"),
        }
    }

    #[tokio::test]
    async fn empty_object_options_are_accepted() {
        let router = AsrRouter::new().register(ParaformerFactory);
        // {} should pass the options check and reach the loader.
        match router.dispatch(req(json!({}))).await {
            Err(RouterError::InstantiationFailed { .. }) => {}
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected loader error"),
        }
    }

    #[tokio::test]
    async fn unknown_options_return_invalid_request() {
        let router = AsrRouter::new().register(ParaformerFactory);
        match router.dispatch(req(json!({ "vocab_size": 8358 }))).await {
            Err(RouterError::InvalidRequest(msg)) => {
                assert!(msg.contains("paraformer"));
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("expected error for unknown options"),
        }
    }
}
