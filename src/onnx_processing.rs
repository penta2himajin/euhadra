//! ONNX-based text processing (feature-gated behind `onnx`).
//!
//! `cargo build --features onnx`
//!
//! A full non-LLM pipeline — ONNX ASR, punctuation and paragraph splitting.
//! Note that the embedder lives in [`crate::phoneme`], not here:
//!
//! ```no_run
//! use euhadra::prelude::*;
//! use euhadra::onnx_processing::OnnxPunctuationRestorer;
//! use euhadra::parakeet::ParakeetAdapter;
//! use euhadra::paragraph::ParagraphSplitter;
//! use euhadra::phoneme::OnnxTextEmbedder;
//!
//! # fn f() -> Result<(), Box<dyn std::error::Error>> {
//! let pipeline = PipelineBuilder::new()
//!     .asr(ParakeetAdapter::load("models/parakeet-tdt-0.6b-v3")?)
//!     .filter(FillerFilter::for_language(Language::English))
//!     .processor(SelfCorrectionDetector::new())
//!     .processor(OnnxPunctuationRestorer::load(
//!         "models/punct/model.onnx",
//!         "models/punct/tokenizer.json",
//!         OnnxPunctuationRestorer::default_labels(),
//!     )?)
//!     .processor(
//!         ParagraphSplitter::new()
//!             .with_embedder(OnnxTextEmbedder::load("models/bge-small-en")?),
//!     )
//!     .emitter(StdoutEmitter)
//!     .build()?;
//! # let _ = pipeline;
//! # Ok(())
//! # }
//! ```
//!
//! `no_run` because it loads model files; it is still compiled, so the
//! signatures above cannot drift.

use async_trait::async_trait;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use crate::processor::{Correction, CorrectionKind, ProcessError, ProcessResult, TextProcessor};
use crate::types::ContextSnapshot;

// ---------------------------------------------------------------------------
// OnnxPunctuationRestorer
// ---------------------------------------------------------------------------

/// Punctuation + capitalization via ONNX token classification model.
pub struct OnnxPunctuationRestorer {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    labels: Vec<String>,
}

impl OnnxPunctuationRestorer {
    pub fn load(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        labels: Vec<String>,
    ) -> Result<Self, ProcessError> {
        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(model_path.as_ref()))
            .map_err(|e| ProcessError::Unavailable(format!("load model: {e}")))?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_path.as_ref()).map_err(|e| ProcessError::Unavailable(format!("load tokenizer: {e}")))?;
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            labels,
        })
    }

    /// Default labels for felflare/bert-restore-punctuation.
    /// Compound format: first char = punctuation ('O'=none), second char = case ('U'=uppercase).
    pub fn default_labels() -> Vec<String> {
        [
            "OU", "OO", ".O", "!O", ",O", ".U", "!U", ",U", ":O", ":U", ";O", ";U", "'O", "'U",
            "-O",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Parse compound label into (punctuation_char, uppercase_next).
    fn parse_label(label: &str) -> (Option<char>, bool) {
        let chars: Vec<char> = label.chars().collect();
        if chars.len() < 2 {
            return (None, false);
        }
        let punct = if chars[0] == 'O' {
            None
        } else {
            Some(chars[0])
        };
        let uppercase = chars[1] == 'U';
        (punct, uppercase)
    }
}

#[async_trait]
impl TextProcessor for OnnxPunctuationRestorer {
    async fn process(
        &self,
        text: &str,
        _ctx: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(ProcessResult {
                text: String::new(),
                corrections: vec![],
            });
        }

        let mut session = self.session.lock().await;

        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| ProcessError::Failed(format!("tokenize: {e}")))?;

        let len = enc.get_ids().len();
        let ids =
            Array2::from_shape_vec((1, len), enc.get_ids().iter().map(|&x| x as i64).collect())
                .unwrap();
        let mask = Array2::from_shape_vec(
            (1, len),
            enc.get_attention_mask().iter().map(|&x| x as i64).collect(),
        )
        .unwrap();

        let ids_val = Value::from_array(ids).map_err(|e| ProcessError::Failed(format!("{e}")))?;
        let mask_val = Value::from_array(mask).map_err(|e| ProcessError::Failed(format!("{e}")))?;

        let outputs = session
            .run(vec![
                ("input_ids", ids_val.into_dyn()),
                ("attention_mask", mask_val.into_dyn()),
            ])
            .map_err(|e| ProcessError::Inference(format!("inference: {e}")))?;

        // Extract logits and copy to owned data before dropping session
        let logits = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| ProcessError::Failed(format!("extract: {e}")))?;
        let view = logits.view();
        let seq_len = view.shape()[1];
        let num_labels = view.shape()[2];

        // Copy logits to owned vec so we can drop session
        let logits_owned: Vec<f32> = view.iter().copied().collect();

        // Get word_ids before dropping
        let word_ids: Vec<Option<u32>> = enc.get_word_ids().to_vec();

        drop(outputs);
        drop(session);

        // Map subword → word (first subword per word)
        let mut word_labels: Vec<String> = vec!["O".into(); words.len()];
        let mut seen = vec![false; words.len()];

        for (ti, wid_opt) in word_ids.iter().enumerate() {
            if let Some(wid) = wid_opt {
                let w = *wid as usize;
                if w < words.len() && !seen[w] && ti < seq_len {
                    seen[w] = true;
                    let offset = ti * num_labels;
                    let best = (0..num_labels)
                        .max_by(|&a, &b| {
                            logits_owned[offset + a]
                                .partial_cmp(&logits_owned[offset + b])
                                .unwrap()
                        })
                        .unwrap();
                    if best < self.labels.len() {
                        word_labels[w] = self.labels[best].clone();
                    }
                }
            }
        }

        // Reconstruct with punctuation and capitalization from compound labels.
        // Label format: first char = punct after this word ('O'=none),
        //               second char = case of THIS word ('U'=uppercase, 'O'=original).
        let mut result = String::with_capacity(text.len() + words.len());
        let mut corrections = Vec::new();

        for (i, word) in words.iter().enumerate() {
            if !result.is_empty() {
                result.push(' ');
            }

            let (punct, should_upper) = Self::parse_label(&word_labels[i]);

            let mut w = word.to_string();
            if should_upper && w.chars().next().is_some_and(|c| c.is_alphabetic()) {
                let first_len = w.chars().next().unwrap().len_utf8();
                let first: String = w.chars().next().unwrap().to_uppercase().collect();
                if first != w[..first_len] {
                    corrections.push(Correction {
                        kind: CorrectionKind::Capitalized,
                        original: w[..first_len].to_string(),
                        replacement: first.clone(),
                    });
                }
                w = format!("{}{}", first, &w[first_len..]);
            }
            result.push_str(&w);

            if let Some(p) = punct {
                let ps = p.to_string();
                corrections.push(Correction {
                    kind: CorrectionKind::PunctuationInserted,
                    original: String::new(),
                    replacement: ps.clone(),
                });
                result.push_str(&ps);
            }
        }

        Ok(ProcessResult {
            text: result,
            corrections,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
