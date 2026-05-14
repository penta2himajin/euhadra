//! ONNX-based text processing (feature-gated behind `onnx`).
//!
//! `cargo build --features onnx`

use async_trait::async_trait;
use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use crate::filter::{FilterError, FilterResult, TextFilter};
use crate::processor::{Correction, CorrectionKind, ProcessError, ProcessResult, TextProcessor};
use crate::types::ContextSnapshot;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Embed a single text and return L2-normalized CLS vector.
fn embed_one(
    session: &mut Session,
    tokenizer: &Tokenizer,
    text: &str,
) -> Result<Array1<f32>, String> {
    let enc = tokenizer
        .encode(text, true)
        .map_err(|e| format!("tokenize: {e}"))?;
    let len = enc.get_ids().len();

    let ids = Array2::from_shape_vec((1, len), enc.get_ids().iter().map(|&x| x as i64).collect())
        .unwrap();
    let mask = Array2::from_shape_vec(
        (1, len),
        enc.get_attention_mask().iter().map(|&x| x as i64).collect(),
    )
    .unwrap();
    let tids = Array2::from_shape_vec(
        (1, len),
        enc.get_type_ids().iter().map(|&x| x as i64).collect(),
    )
    .unwrap();

    let ids_val = Value::from_array(ids).map_err(|e| format!("ids: {e}"))?;
    let mask_val = Value::from_array(mask).map_err(|e| format!("mask: {e}"))?;
    let tids_val = Value::from_array(tids).map_err(|e| format!("tids: {e}"))?;

    let outputs = session
        .run(vec![
            ("input_ids", ids_val.into_dyn()),
            ("attention_mask", mask_val.into_dyn()),
            ("token_type_ids", tids_val.into_dyn()),
        ])
        .map_err(|e| format!("run: {e}"))?;

    // [1, seq_len, hidden_dim] → CLS = [0, 0, :]
    let arr = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|e| format!("extract: {e}"))?;
    let view = arr.view();
    let hidden_dim = view.shape()[2];
    let cls: Vec<f32> = (0..hidden_dim).map(|i| view[[0, 0, i]]).collect();

    // L2 normalize
    let norm: f32 = cls.iter().map(|x| x * x).sum::<f32>().sqrt();
    let normalized: Vec<f32> = if norm > 0.0 {
        cls.iter().map(|x| x / norm).collect()
    } else {
        cls
    };
    Ok(Array1::from_vec(normalized))
}

// ---------------------------------------------------------------------------
// OnnxEmbeddingFilter
// ---------------------------------------------------------------------------

/// Filler removal using ONNX sentence embeddings (bge-small-en-v1.5, 384-dim).
pub struct OnnxEmbeddingFilter {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    filler_embeddings: Vec<Array1<f32>>,
    pure_fillers: Vec<String>,
    contextual_fillers: Vec<String>,
    pure_threshold: f32,
}

impl OnnxEmbeddingFilter {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, FilterError> {
        let dir = model_dir.as_ref();

        let mut session = Session::builder()
            .and_then(|mut b| b.commit_from_file(dir.join("model.onnx")))
            .map_err(|e| FilterError {
                message: format!("load model: {e}"),
            })?;

        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| FilterError {
                message: format!("load tokenizer: {e}"),
            })?;

        let pure_fillers: Vec<String> = ["um", "uh", "uhm", "umm", "hmm", "er", "ah", "eh"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let contextual_fillers: Vec<String> =
            ["so", "well", "basically", "actually", "literally", "right"]
                .iter()
                .map(|s| s.to_string())
                .collect();

        let mut filler_embeddings = Vec::new();
        for f in &pure_fillers {
            filler_embeddings.push(
                embed_one(&mut session, &tokenizer, f).map_err(|e| FilterError { message: e })?,
            );
        }

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            filler_embeddings,
            pure_fillers,
            contextual_fillers,
            pure_threshold: 0.82,
        })
    }

    fn max_filler_sim(&self, emb: &Array1<f32>) -> f32 {
        self.filler_embeddings
            .iter()
            .map(|f| emb.dot(f))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    fn is_sentence_initial(words: &[&str], idx: usize, removed: &[bool]) -> bool {
        if idx == 0 {
            return true;
        }
        for j in (0..idx).rev() {
            if removed[j] {
                continue;
            }
            return words[j].ends_with('.')
                || words[j].ends_with('!')
                || words[j].ends_with('?')
                || words[j].ends_with(',');
        }
        true
    }
}

#[async_trait]
impl TextFilter for OnnxEmbeddingFilter {
    async fn filter(&self, text: &str) -> Result<FilterResult, FilterError> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(FilterResult {
                text: String::new(),
                removed: vec![],
            });
        }

        let mut session = self.session.lock().await;
        let mut embeddings = Vec::with_capacity(words.len());
        for w in &words {
            embeddings.push(
                embed_one(&mut session, &self.tokenizer, w)
                    .map_err(|e| FilterError { message: e })?,
            );
        }
        drop(session);

        let n = words.len();
        let mut removed_flags = vec![false; n];
        let mut labels = Vec::new();

        for i in 0..n {
            let clean: String = words[i]
                .to_lowercase()
                .chars()
                .take_while(|c| c.is_alphanumeric())
                .collect();
            if self.max_filler_sim(&embeddings[i]) >= self.pure_threshold
                && self.pure_fillers.contains(&clean)
            {
                removed_flags[i] = true;
                labels.push(words[i].to_string());
            }
        }
        for i in 0..n {
            if removed_flags[i] {
                continue;
            }
            let clean: String = words[i]
                .to_lowercase()
                .chars()
                .take_while(|c| c.is_alphanumeric())
                .collect();
            if self.contextual_fillers.contains(&clean)
                && Self::is_sentence_initial(&words, i, &removed_flags)
            {
                removed_flags[i] = true;
                labels.push(words[i].to_string());
            }
        }

        let kept: Vec<&str> = words
            .iter()
            .enumerate()
            .filter(|(i, _)| !removed_flags[*i])
            .map(|(_, w)| *w)
            .collect();
        Ok(FilterResult {
            text: kept.join(" "),
            removed: labels,
        })
    }
}

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
            .map_err(|e| ProcessError {
                message: format!("load model: {e}"),
            })?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_path.as_ref()).map_err(|e| ProcessError {
                message: format!("load tokenizer: {e}"),
            })?;
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
            .map_err(|e| ProcessError {
                message: format!("tokenize: {e}"),
            })?;

        let len = enc.get_ids().len();
        let ids =
            Array2::from_shape_vec((1, len), enc.get_ids().iter().map(|&x| x as i64).collect())
                .unwrap();
        let mask = Array2::from_shape_vec(
            (1, len),
            enc.get_attention_mask().iter().map(|&x| x as i64).collect(),
        )
        .unwrap();

        let ids_val = Value::from_array(ids).map_err(|e| ProcessError {
            message: format!("{e}"),
        })?;
        let mask_val = Value::from_array(mask).map_err(|e| ProcessError {
            message: format!("{e}"),
        })?;

        let outputs = session
            .run(vec![
                ("input_ids", ids_val.into_dyn()),
                ("attention_mask", mask_val.into_dyn()),
            ])
            .map_err(|e| ProcessError {
                message: format!("inference: {e}"),
            })?;

        // Extract logits and copy to owned data before dropping session
        let logits = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| ProcessError {
                message: format!("extract: {e}"),
            })?;
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
