pub mod emitters;
pub mod eval;
pub mod filter;
pub mod mic;
pub mod mock;
pub mod paragraph;
pub mod phoneme;
pub mod pipeline;
pub mod prelude;
pub mod processor;
pub mod router;
pub mod state;
pub mod traits;
pub mod types;
pub mod whisper_local;

#[cfg(feature = "onnx")]
pub mod onnx_processing;

#[cfg(feature = "onnx")]
pub mod parakeet;

#[cfg(feature = "onnx")]
pub mod paraformer;

#[cfg(feature = "onnx")]
pub mod canary;

#[cfg(feature = "onnx")]
pub mod sensevoice;

#[cfg(feature = "onnx")]
pub mod whisper_onnx;
