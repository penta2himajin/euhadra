//! Integration tests: Parakeet / Paraformer / Canary factories via the
//! public router API.
//!
//! Each factory is registered on a fresh `AsrRouter` and dispatched.
//! Real model bundles are not available in CI, so the tests verify
//! that dispatch reaches the adapter loader (returning
//! `InstantiationFailed`) rather than failing earlier with
//! `UnknownRuntime` or `InvalidRequest`.

#![cfg(feature = "onnx")]

use euhadra::canary::CanaryFactory;
use euhadra::paraformer::ParaformerFactory;
use euhadra::parakeet::ParakeetFactory;
use euhadra::prelude::{AdapterRequest, AsrRouter, ModelSource, RouterError};
use std::path::PathBuf;

fn make_request(runtime: &str, language: &str) -> AdapterRequest {
    AdapterRequest {
        language: language.into(),
        runtime: runtime.into(),
        model_source: ModelSource::LocalPath(PathBuf::from("/nonexistent/bundle")),
        options: serde_json::Value::Null,
    }
}

#[tokio::test]
async fn router_dispatches_parakeet_factory() {
    let router = AsrRouter::new().register(ParakeetFactory);
    match router.dispatch(make_request("parakeet", "en")).await {
        Err(RouterError::InstantiationFailed { runtime, .. }) => {
            assert_eq!(runtime, "parakeet");
        }
        Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
        Ok(_) => panic!("expected loader error"),
    }
}

#[tokio::test]
async fn router_dispatches_paraformer_factory() {
    let router = AsrRouter::new().register(ParaformerFactory);
    match router.dispatch(make_request("paraformer", "zh")).await {
        Err(RouterError::InstantiationFailed { runtime, .. }) => {
            assert_eq!(runtime, "paraformer");
        }
        Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
        Ok(_) => panic!("expected loader error"),
    }
}

#[tokio::test]
async fn router_dispatches_canary_factory() {
    let router = AsrRouter::new().register(CanaryFactory);
    match router.dispatch(make_request("canary", "es")).await {
        Err(RouterError::InstantiationFailed { runtime, .. }) => {
            assert_eq!(runtime, "canary");
        }
        Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
        Ok(_) => panic!("expected loader error"),
    }
}

#[tokio::test]
async fn router_routes_three_factories_independently() {
    let router = AsrRouter::new()
        .register(ParakeetFactory)
        .register(ParaformerFactory)
        .register(CanaryFactory);

    for runtime in ["parakeet", "paraformer", "canary"] {
        match router.dispatch(make_request(runtime, "en")).await {
            Err(RouterError::InstantiationFailed { runtime: got, .. }) => {
                assert_eq!(got, runtime);
            }
            Err(other) => panic!("expected InstantiationFailed for {runtime}, got {other:?}"),
            Ok(_) => panic!("expected loader error for {runtime}"),
        }
    }
}
