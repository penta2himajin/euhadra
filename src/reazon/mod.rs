//! ReazonSpeech Zipformer transducer ASR adapter (lightweight Japanese option).
//!
//! ```text
//! audio f32  →  Kaldi FBANK (80 mel, Povey window, snip_edges=false)
//!            →  encoder / decoder / joiner INT8 ONNX (greedy modified RNNT)
//!            →  tokens.txt + icefall byte_decode → text
//! ```
//!
//! A lightweight alternative to `parakeet-tdt_ctc-0.6b-ja`, which is the
//! ja evaluation path: ~72 MB INT8 vs ~2.4 GB, Apache-2.0 weights
//! (`reazon-research/reazonspeech-k2-v2`), ~12× lower RTF on CI — but
//! FLEURS-ja CER 0.1048 vs 0.0724, and still 0.0885 with sherpa-onnx's
//! own FP32 decode, so it is not the accuracy default (#167).
//!
//! This module re-implements the sherpa-onnx greedy transducer path in
//! pure `ort` and ships no weights; `scripts/setup_reazon_ja.sh` fetches
//! the k2-fsa Zipformer export.

pub mod adapter;
pub mod factory;
pub mod vocab;

pub use adapter::{ReazonAdapter, ReazonConfig};
pub use factory::ReazonFactory;
