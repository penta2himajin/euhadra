#![cfg_attr(docsrs, feature(doc_auto_cfg))]

pub mod emitters;
pub mod filter;
pub mod paragraph;
pub mod phoneme;
pub mod pipeline;
pub mod prelude;
pub mod processor;
pub mod router;
pub mod traits;
pub mod types;
pub mod whisper_local;

// Internal machinery. `PipelineState` — the part a caller might observe —
// is re-exported from `types`; driving the state machine or comparing
// embeddings are not things a dependent does.
pub(crate) mod similarity;
pub(crate) mod state;

/// Mock adapters for testing pipelines built on euhadra's traits.
///
/// Available inside the crate's own tests, and to dependents that turn
/// on the `testing` feature.
#[cfg(any(test, feature = "testing"))]
pub mod mock;

/// WER/CER metrics, latency sampling and baseline gating for the
/// evaluation harness. Development tooling rather than library surface.
#[cfg(any(test, feature = "testing"))]
pub mod eval;

/// Microphone capture. Pulls in `cpal`, which links ALSA on Linux.
#[cfg(feature = "mic")]
pub mod mic;

#[cfg(feature = "onnx")]
pub mod embedding;

#[cfg(feature = "onnx")]
pub mod onnx_processing;

#[cfg(feature = "onnx")]
pub mod parakeet;

#[cfg(feature = "onnx")]
pub mod paraformer;

#[cfg(feature = "onnx")]
pub mod canary;

#[cfg(feature = "onnx")]
pub mod dolphin;

#[cfg(feature = "onnx")]
pub mod sensevoice;

#[cfg(feature = "onnx")]
pub mod whisper_onnx;
