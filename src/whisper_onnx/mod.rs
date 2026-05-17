//! Whisper-large-v3-turbo ONNX adapter.
//!
//! See [`adapter`] for the public `AsrAdapter` + `AsrRuntimeFactory`
//! impls and the high-level decode loop. [`mel`] computes the
//! Whisper-spec log-mel spectrogram and [`tokenizer`] loads the
//! `tokenizers`-format vocabulary plus special-token ids.

pub mod adapter;
pub mod mel;
pub mod tokenizer;

pub use adapter::{WhisperOnnxAdapter, WhisperOnnxConfig, WhisperOnnxFactory};
