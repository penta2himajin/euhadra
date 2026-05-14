//! NVIDIA Canary-180M-Flash ASR adapter (en / de / fr / es).
//!
//! Wraps the [`istupakov/canary-180m-flash-onnx`](https://huggingface.co/istupakov/canary-180m-flash-onnx)
//! ONNX export — encoder + decoder graphs only — and reproduces the
//! preprocessing + autoregressive decode loop from
//! [`onnx-asr`](https://github.com/istupakov/onnx-asr) in pure Rust:
//!
//! ```text
//! audio f32 → log-mel (Hann, 80 mel, n_fft=512, hop=160) → CMVN
//!           → encoder ONNX → encoder_embeddings, encoder_mask
//!           → autoregressive decoder loop with KV cache
//!             (greedy argmax, EOS-terminated)
//!           → SentencePiece detokenize → text
//! ```
//!
//! Architecture choice rationale and fallback plan are documented in
//! `docs/canary-integration.md`.
//!
//! Distribution: ONNX weights are downloaded by the user via
//! `scripts/setup_canary.sh`; this crate ships none of them.
//! Upstream license is CC-BY-4.0.

pub mod adapter;
pub mod decoder;
pub mod encoder;
pub mod frontend;
pub(crate) mod profiling;
pub mod vocab;

pub use adapter::{CanaryAdapter, CanaryConfig};
