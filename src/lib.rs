#![cfg_attr(docsrs, feature(doc_auto_cfg))]

pub mod eval;
pub mod filter;
pub mod mock;
pub mod paragraph;
pub mod phoneme;
pub mod pipeline;
pub mod prelude;
pub mod processor;
pub mod router;
pub mod similarity;
pub mod state;
pub mod traits;
pub mod types;
pub mod whisper_local;

/// Microphone capture. Pulls in `cpal`, which links ALSA on Linux.
#[cfg(feature = "mic")]
pub mod mic;

/// Clipboard-backed [`OutputEmitter`](traits::OutputEmitter) implementations.
#[cfg(feature = "clipboard")]
pub mod emitters;

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
