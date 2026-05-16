//! SenseVoice-Small ASR adapter (multilingual: zh / en / yue / ja / ko).
//!
//! This module ports the FunAudioLLM SenseVoice-Small inference path
//! to pure Rust with `ort`:
//!
//! ```text
//! audio f32  →  Kaldi FBANK (80 mel, shared with paraformer)
//!            →  LFR (m=7, n=6)  →  CMVN (am.mvn)
//!            →  ONNX (x, x_length, language, text_norm)
//!            →  argmax → unique_consecutive → drop blank
//!            →  tokens.txt lookup → strip <|...|> markers
//!            →  text
//! ```
//!
//! Default config matches `FunAudioLLM/SenseVoiceSmall` exported via
//! `scripts/setup_sensevoice.sh` (vocab ~25K, 80-mel × 7-frame LFR,
//! BPE word boundary U+2581).
//!
//! Korean is the immediate integration target, but the model also
//! supports Mandarin (`zh`), Cantonese (`yue`), English (`en`), and
//! Japanese (`ja`) — switch via `with_language("ja")` etc.
//!
//! The model weights ship under the FunASR Model Open Source
//! License Agreement v1.1
//! (https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE),
//! a permissive license allowing commercial use with attribution.
//! This Rust port re-implements the inference logic and ships none
//! of the upstream weights — `setup_sensevoice.sh` self-exports
//! from the official HuggingFace bundle.

pub mod adapter;
pub mod factory;
pub mod metadata;
pub mod vocab;

pub use adapter::{SenseVoiceAdapter, SenseVoiceConfig};
pub use factory::SenseVoiceFactory;
pub use metadata::SenseVoiceMetadata;
