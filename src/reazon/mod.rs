//! ReazonSpeech Zipformer transducer ASR adapter (Japanese routing path).
//!
//! ```text
//! audio f32  →  Kaldi FBANK (80 mel, Povey window, snip_edges=false)
//!            →  encoder / decoder / joiner INT8 ONNX (greedy modified RNNT)
//!            →  tokens.txt + icefall byte_decode → text
//! ```
//!
//! Chosen over `parakeet-tdt_ctc-0.6b-ja` for the shipping ja path after the
//! hayamimi speed compare (`docs/benchmarks/hayamimi_speed_compare.md`):
//! same host, FLEURS-ja, offline RTF ~0.017 vs Parakeet-ja ~0.032 (~2×),
//! at ~72 MB INT8 vs ~2.4 GB. Weights are Apache-2.0
//! (`reazon-research/reazonspeech-k2-v2`).
//!
//! This module re-implements the sherpa-onnx greedy transducer path in
//! pure `ort` and ships no weights; `scripts/setup_reazon_ja.sh` fetches
//! the k2-fsa Zipformer export.

pub mod adapter;
pub mod factory;
pub mod vocab;

pub use adapter::{ReazonAdapter, ReazonConfig};
pub use factory::ReazonFactory;
