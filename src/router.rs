//! ASR runtime router.
//!
//! Menura side picks a `(runtime, language, model_source, options)` request
//! and dispatches it to the router; the router looks up the registered
//! factory for that runtime id and asks it to build an `AsrAdapter`. The
//! router itself owns no language-to-runtime mapping — that lives in
//! Menura's `asr_models.toml`. By the time a request reaches `dispatch`,
//! the runtime selection has already happened upstream.
//!
//! Several design knobs are deliberately left minimal in this initial
//! landing (see the parent tracking issue #84): `RouterError` exposes only
//! the variants we need to express today, `options` is raw JSON so factories
//! can iterate without locking the trait into a typed schema, and there is
//! no `supported_languages()` introspection. These can be added without
//! breaking existing factories.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::traits::AsrAdapter;

/// Where to load the model artefacts from.
///
/// Only local paths are supported today; remote sources (HF Hub, S3, ...)
/// can be added as additional variants without breaking existing factories.
#[derive(Debug, Clone)]
pub enum ModelSource {
    LocalPath(PathBuf),
}

/// A request to instantiate an `AsrAdapter` for a particular language and
/// runtime backend.
#[derive(Debug, Clone)]
pub struct AdapterRequest {
    /// BCP 47 language tag (e.g. `"ja"`, `"en-US"`).
    pub language: String,
    /// Factory id — must match an `AsrRuntimeFactory::id()` registered on
    /// the router.
    pub runtime: String,
    pub model_source: ModelSource,
    /// Runtime-specific configuration. Factories interpret this themselves;
    /// the router does not inspect it.
    pub options: Value,
}

/// Errors returned by the router.
#[derive(Debug)]
pub enum RouterError {
    /// No factory is registered under the requested runtime id.
    UnknownRuntime(String),
    /// The request failed validation before reaching a factory
    /// (e.g. unsupported `ModelSource` for this build).
    InvalidRequest(String),
    /// The factory accepted the request but failed to build the adapter.
    InstantiationFailed { runtime: String, message: String },
}

impl std::fmt::Display for RouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownRuntime(id) => write!(f, "unknown runtime: {id}"),
            Self::InvalidRequest(msg) => write!(f, "invalid request: {msg}"),
            Self::InstantiationFailed { runtime, message } => {
                write!(f, "failed to instantiate runtime {runtime}: {message}")
            }
        }
    }
}

impl std::error::Error for RouterError {}

/// A factory that knows how to build an `AsrAdapter` for a specific runtime.
///
/// Factories are usually thin wrappers around an existing adapter's
/// `load`/`new` constructor (see Phase 1 factory PRs: #B-1 / #B-2 / #B-3).
/// They are registered on the router at startup; Menura decides which
/// factory id to dispatch to based on its own language-to-runtime mapping.
#[async_trait]
pub trait AsrRuntimeFactory: Send + Sync {
    /// The runtime id this factory handles, e.g. `"whisper-local"` or
    /// `"sensevoice"`. Must be stable across releases — Menura's config
    /// keys off this string.
    fn id(&self) -> &'static str;

    async fn instantiate(&self, req: &AdapterRequest) -> Result<Arc<dyn AsrAdapter>, RouterError>;
}

/// A registry of `AsrRuntimeFactory`s, indexed by `id()`.
#[derive(Default)]
pub struct AsrRouter {
    factories: HashMap<&'static str, Arc<dyn AsrRuntimeFactory>>,
}

impl AsrRouter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a factory. Chainable; later registrations under the same
    /// id overwrite earlier ones.
    pub fn register<F>(mut self, factory: F) -> Self
    where
        F: AsrRuntimeFactory + 'static,
    {
        self.factories.insert(factory.id(), Arc::new(factory));
        self
    }

    pub async fn dispatch(&self, req: AdapterRequest) -> Result<Arc<dyn AsrAdapter>, RouterError> {
        let factory = self
            .factories
            .get(req.runtime.as_str())
            .ok_or_else(|| RouterError::UnknownRuntime(req.runtime.clone()))?
            .clone();
        factory.instantiate(&req).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{AsrAdapter, AsrError};
    use crate::types::{AsrResult, AudioChunk};
    use tokio::sync::mpsc;

    struct StubAdapter;
    #[async_trait]
    impl AsrAdapter for StubAdapter {
        async fn transcribe(
            &self,
            _audio_rx: mpsc::Receiver<AudioChunk>,
            _result_tx: mpsc::Sender<AsrResult>,
        ) -> Result<(), AsrError> {
            Ok(())
        }
    }

    struct StubFactory {
        id: &'static str,
    }

    #[async_trait]
    impl AsrRuntimeFactory for StubFactory {
        fn id(&self) -> &'static str {
            self.id
        }
        async fn instantiate(
            &self,
            _req: &AdapterRequest,
        ) -> Result<Arc<dyn AsrAdapter>, RouterError> {
            Ok(Arc::new(StubAdapter))
        }
    }

    struct FailingFactory;
    #[async_trait]
    impl AsrRuntimeFactory for FailingFactory {
        fn id(&self) -> &'static str {
            "failing"
        }
        async fn instantiate(
            &self,
            _req: &AdapterRequest,
        ) -> Result<Arc<dyn AsrAdapter>, RouterError> {
            Err(RouterError::InstantiationFailed {
                runtime: self.id().to_string(),
                message: "deliberate test failure".into(),
            })
        }
    }

    fn req(runtime: &str) -> AdapterRequest {
        AdapterRequest {
            language: "ja".into(),
            runtime: runtime.into(),
            model_source: ModelSource::LocalPath(PathBuf::from("/tmp/model")),
            options: Value::Null,
        }
    }

    #[tokio::test]
    async fn dispatch_returns_adapter_from_registered_factory() {
        let router = AsrRouter::new().register(StubFactory { id: "stub" });
        let adapter = router.dispatch(req("stub")).await;
        assert!(adapter.is_ok());
    }

    #[tokio::test]
    async fn dispatch_unknown_runtime_returns_unknown_runtime() {
        let router = AsrRouter::new();
        match router.dispatch(req("missing")).await {
            Err(RouterError::UnknownRuntime(id)) => assert_eq!(id, "missing"),
            Err(other) => panic!("expected UnknownRuntime, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[tokio::test]
    async fn dispatch_propagates_factory_failure() {
        let router = AsrRouter::new().register(FailingFactory);
        match router.dispatch(req("failing")).await {
            Err(RouterError::InstantiationFailed { runtime, message }) => {
                assert_eq!(runtime, "failing");
                assert!(message.contains("deliberate"));
            }
            Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[tokio::test]
    async fn register_is_chainable_and_dispatches_correct_factory() {
        let router = AsrRouter::new()
            .register(StubFactory { id: "alpha" })
            .register(StubFactory { id: "beta" });
        assert!(router.dispatch(req("alpha")).await.is_ok());
        assert!(router.dispatch(req("beta")).await.is_ok());
        match router.dispatch(req("gamma")).await {
            Err(RouterError::UnknownRuntime(_)) => {}
            Err(other) => panic!("expected UnknownRuntime, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn router_error_display_messages_include_context() {
        let unknown = RouterError::UnknownRuntime("foo".into()).to_string();
        assert!(unknown.contains("foo"));

        let failed = RouterError::InstantiationFailed {
            runtime: "bar".into(),
            message: "boom".into(),
        }
        .to_string();
        assert!(failed.contains("bar") && failed.contains("boom"));
    }
}
