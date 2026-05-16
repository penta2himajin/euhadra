//! Integration test: `SenseVoiceFactory` resolves via the public router API.
//!
//! Exercises the same dispatch path Menura uses. We can't load a real
//! SenseVoice bundle in CI, so the test verifies that dispatch reaches
//! the adapter loader (which fails with `InstantiationFailed`) rather
//! than dying earlier with `UnknownRuntime`.

#![cfg(feature = "onnx")]

use euhadra::prelude::{AdapterRequest, AsrRouter, ModelSource, RouterError};
use euhadra::sensevoice::SenseVoiceFactory;
use std::path::PathBuf;

#[tokio::test]
async fn router_dispatch_reaches_sensevoice_loader() {
    let router = AsrRouter::new().register(SenseVoiceFactory);

    let req = AdapterRequest {
        language: "ko".into(),
        runtime: "sensevoice".into(),
        model_source: ModelSource::LocalPath(PathBuf::from("/nonexistent/sensevoice/bundle")),
        options: serde_json::Value::Null,
    };

    match router.dispatch(req).await {
        Err(RouterError::InstantiationFailed { runtime, .. }) => {
            assert_eq!(runtime, "sensevoice");
        }
        Err(other) => panic!("expected InstantiationFailed, got {other:?}"),
        Ok(_) => panic!("expected error when bundle dir does not exist"),
    }
}
