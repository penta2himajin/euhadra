//! Integration test: WhisperLocalFactory resolves via the public router API.
//!
//! Exercises the same dispatch path that Menura uses — registering the
//! factory on a fresh `AsrRouter` and asking it to build an adapter from
//! a `AdapterRequest`. We can't run a real transcription here (no
//! whisper-cli binary in CI), so we stop after `instantiate()` returns.

use euhadra::prelude::{AdapterRequest, AsrRouter, ModelSource};
use euhadra::whisper_local::WhisperLocalFactory;
use serde_json::json;
use std::path::PathBuf;

#[tokio::test]
async fn router_dispatch_builds_whisper_local_adapter() {
    let router = AsrRouter::new().register(WhisperLocalFactory);

    let req = AdapterRequest {
        language: "ja".into(),
        runtime: "whisper-local".into(),
        model_source: ModelSource::LocalPath(PathBuf::from("/tmp/ggml-base.bin")),
        options: json!({ "cli_path": "/usr/bin/whisper-cli" }),
    };

    let adapter = router
        .dispatch(req)
        .await
        .expect("dispatch should succeed for a well-formed whisper-local request");

    // We can't transcribe without a real binary, but holding the trait
    // object proves the factory returned a valid `Arc<dyn AsrAdapter>`.
    drop(adapter);
}
