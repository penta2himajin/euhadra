//! Paraformer-large ASR adapter (Chinese, FunASR-compatible ONNX).
//!
//! This module ports the reference FunASR pipeline
//! (`funasr_onnx.utils.frontend.WavFrontend` +
//! `funasr_onnx.paraformer_bin.Paraformer`) to pure Rust with `ort`:
//!
//! ```text
//! audio f32  →  Kaldi FBANK (80 mel)  →  LFR (m=7, n=6)  →  CMVN (am.mvn)
//!            →  ONNX encoder (speech, speech_lengths)
//!            →  argmax + valid_token_num crop
//!            →  tokens.json lookup  →  sentence_postprocess  →  text
//! ```
//!
//! Default config matches
//! `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx`
//! (vocab 8404, 80-mel × 7-frame LFR, predictor_bias=1).
//!
//! The model files (`model.onnx`, `am.mvn`, `tokens.json`) ship under
//! the [Model License](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE);
//! the FunASR project itself is MIT — see
//! `https://github.com/FunAudioLLM/SenseVoice/blob/main/LICENSE`. This
//! Rust port re-implements the inference logic and ships none of the
//! upstream weights.

pub mod adapter;
pub mod factory;
pub mod fbank;
pub mod frontend;
pub mod vocab;

pub use adapter::{ParaformerAdapter, ParaformerConfig};
pub use factory::ParaformerFactory;
