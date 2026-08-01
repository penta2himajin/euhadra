//! DataoceanAI Dolphin CTC ASR adapter (Korean routing path).
//!
//! ```text
//! audio f32  →  Kaldi FBANK (80 mel, Povey window, snip_edges=false)
//!            →  per-bin CMVN read from the graph's metadata_props
//!            →  ONNX (x, x_len)  →  log_probs [1, T, V]
//!            →  argmax → unique_consecutive → drop blank
//!            →  tokens.txt lookup → strip <...> control tags → text
//! ```
//!
//! Chosen over the incumbent `whisper-large-v3-turbo` in
//! `docs/korean-asr-alternatives.md` §I: 6.2× the throughput for 2.4×
//! the error, and 18× on a short utterance, because a CTC model has no
//! fixed 30-second window to pad. Two properties of that section carry
//! into this code rather than staying in the document:
//!
//! - **One intra-op thread** (`adapter::INTRA_THREADS`). Above one, the
//!   model does not reproduce itself — five runs at four threads gave
//!   five different transcripts (§I.1).
//! - **The front-end is not Paraformer's.** Povey window,
//!   `snip_edges = false`, `high_freq = -400` and a Kaldi-style log
//!   floor, all of which are silent when wrong. Pinned against
//!   kaldi-native-fbank by `FbankOpts::dolphin_default`'s golden test.
//!
//! Dolphin's code and weights are Apache-2.0
//! (<https://github.com/DataoceanAI/Dolphin>). This module re-implements
//! the inference path and ships no weights; `scripts/setup_dolphin_ko.sh`
//! fetches the k2-fsa CTC export.

pub mod adapter;
pub mod factory;
pub mod metadata;
pub mod vocab;

pub use adapter::{DolphinAdapter, DolphinConfig};
pub use factory::DolphinFactory;
pub use metadata::Cmvn;
