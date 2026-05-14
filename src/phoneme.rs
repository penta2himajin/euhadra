//! Phoneme-aware dictionary correction for ASR post-processing.
//!
//! Uses IPA phoneme representations to match ASR-misrecognized words against
//! a user-provided custom dictionary.  Words that sound similar to a dictionary
//! entry (low phoneme edit distance) are replaced with the correct spelling.
//!
//! # Example
//! ```text
//! Custom dictionary: {"useEffect": "juːsɪfɛkt"}
//! ASR output:        "use effect"  → phonemes "juːs ɪfɛkt"
//! Merged phonemes:   "juːsɪfɛkt"
//! Distance to "useEffect": 0  → match → replace
//! ```

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;

use crate::processor::{Correction, CorrectionKind, ProcessError, ProcessResult, TextProcessor};
use crate::types::ContextSnapshot;

// ---------------------------------------------------------------------------
// Phoneme edit distance
// ---------------------------------------------------------------------------

/// Levenshtein distance on Unicode codepoint sequences.
/// IPA characters (ɪ, ɛ, ʃ, ŋ, etc.) are single codepoints, so this
/// gives a meaningful phoneme-level edit distance.
fn phoneme_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());

    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Normalized phoneme similarity: 1.0 = identical, 0.0 = completely different.
fn phoneme_similarity(a: &str, b: &str) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let max_len = a.chars().count().max(b.chars().count());
    let dist = phoneme_distance(a, b);
    1.0 - (dist as f32 / max_len as f32)
}

/// Cosine similarity of two vectors (assumed L2-normalized → dot product).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .max(0.0)
}

// ---------------------------------------------------------------------------
// IPA Dictionary
// ---------------------------------------------------------------------------

/// A word-to-IPA mapping loaded from a JSON file (e.g., CMUdict IPA export).
pub struct IpaDictionary {
    entries: HashMap<String, String>,
}

impl IpaDictionary {
    /// Load from a JSON file: `{"word": "IPA string", ...}`
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ProcessError> {
        let data = std::fs::read_to_string(path.as_ref()).map_err(|e| ProcessError {
            message: format!("load IPA dict: {e}"),
        })?;
        let entries: HashMap<String, String> =
            serde_json::from_str(&data).map_err(|e| ProcessError {
                message: format!("parse IPA dict: {e}"),
            })?;
        tracing::info!(entries = entries.len(), "IPA dictionary loaded");
        Ok(Self { entries })
    }

    /// Create an empty dictionary.
    pub fn empty() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Look up the IPA pronunciation of a word (case-insensitive).
    pub fn lookup(&self, word: &str) -> Option<&str> {
        self.entries.get(&word.to_lowercase()).map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// Text Embedder (for composite scoring in Step C)
// ---------------------------------------------------------------------------

/// Computes a dense vector embedding of a text string.
/// Used together with phoneme distance for composite scoring.
pub trait TextEmbedder: Send + Sync {
    /// Return an L2-normalized embedding vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>, ProcessError>;
}

// ---------------------------------------------------------------------------
// Custom dictionary entry
// ---------------------------------------------------------------------------

/// An entry in the user's custom dictionary.
#[derive(Debug, Clone)]
pub struct CustomEntry {
    /// The correct spelling to emit (e.g., "useEffect").
    pub word: String,
    /// IPA phoneme string (e.g., "juːsɪfɛkt").
    pub phonemes: String,
    /// Pre-computed text embedding (populated by PhonemeCorrector::precompute_embeddings).
    pub embedding: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// G2P Backend (grapheme-to-phoneme for OOV words)
// ---------------------------------------------------------------------------

/// Converts a word's spelling to its IPA phoneme representation.
/// Used as a fallback when the word is not in the IPA dictionary.
pub trait G2pBackend: Send + Sync {
    fn phonemize(&self, word: &str) -> Result<String, ProcessError>;
}

// ---------------------------------------------------------------------------
// PhonemeCorrector — TextProcessor implementation
// ---------------------------------------------------------------------------

/// Corrects ASR-misrecognized words by matching their phonemes against a
/// user-provided custom dictionary.
///
/// The corrector works in three steps for each ASR word:
/// 1. Look up IPA phonemes in the base dictionary (CMUdict).
///    If not found, use the G2P backend (if available) to generate phonemes.
/// 2. Compare phonemes against each custom dictionary entry.
/// 3. If similarity exceeds threshold, replace the word.
///
/// For multi-word ASR errors (e.g., "use effect" for "useEffect"), the
/// corrector also tries merging adjacent words and comparing the merged
/// phoneme string.
pub struct PhonemeCorrector {
    ipa_dict: IpaDictionary,
    custom_entries: Vec<CustomEntry>,
    g2p: Option<Box<dyn G2pBackend>>,
    embedder: Option<Box<dyn TextEmbedder>>,
    /// Weight for phoneme similarity in composite score (0.0–1.0).
    /// Composite = alpha * phoneme_sim + (1-alpha) * text_sim.
    /// Default: 1.0 (phoneme only, no text embedding).
    pub alpha: f32,
    /// Minimum similarity to accept a match (0.0–1.0).
    /// Applied to the composite score when embedder is set.
    /// Default: 0.85
    pub threshold: f32,
    /// Maximum number of adjacent words to merge for compound matching.
    /// Default: 3
    pub max_merge: usize,
}

impl PhonemeCorrector {
    /// Create a new corrector with an IPA dictionary and custom entries.
    pub fn new(ipa_dict: IpaDictionary, custom_entries: Vec<CustomEntry>) -> Self {
        Self {
            ipa_dict,
            custom_entries,
            g2p: None,
            embedder: None,
            alpha: 1.0,
            threshold: 0.85,
            max_merge: 3,
        }
    }

    /// Builder: set similarity threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Builder: set a G2P backend for OOV phonemization.
    pub fn with_g2p(mut self, g2p: impl G2pBackend + 'static) -> Self {
        self.g2p = Some(Box::new(g2p));
        self
    }

    /// Builder: set a text embedder for composite scoring and precompute
    /// embeddings for all custom entries.
    pub fn with_embedder(mut self, embedder: impl TextEmbedder + 'static, alpha: f32) -> Self {
        // Precompute embeddings for custom entries
        for entry in &mut self.custom_entries {
            match embedder.embed(&entry.word) {
                Ok(emb) => entry.embedding = Some(emb),
                Err(e) => {
                    tracing::warn!(word = %entry.word, error = %e, "failed to embed custom entry")
                }
            }
        }
        self.embedder = Some(Box::new(embedder));
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Get the IPA string for a word.
    /// Tries the dictionary first; falls back to G2P if available.
    fn word_to_phonemes(&self, word: &str) -> Option<String> {
        // Strip punctuation for lookup
        let clean: String = word
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '\'')
            .collect();

        // Dictionary lookup first
        if let Some(ipa) = self.ipa_dict.lookup(&clean) {
            return Some(ipa.to_string());
        }

        // G2P fallback for OOV words
        if let Some(g2p) = &self.g2p {
            match g2p.phonemize(&clean) {
                Ok(ipa) if !ipa.is_empty() => {
                    tracing::debug!(word = %clean, phonemes = %ipa, "G2P fallback");
                    return Some(ipa);
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(word = %clean, error = %e, "G2P failed");
                }
            }
        }

        None
    }

    /// Find the best matching custom entry for given phonemes and optional text span.
    /// Uses composite scoring when embedder is available:
    ///   score = alpha * phoneme_sim + (1-alpha) * text_sim
    /// Returns (entry_index, score) or None if below threshold.
    fn best_match(&self, phonemes: &str, text_span: &str) -> Option<(usize, f32)> {
        let text_emb = if self.alpha < 1.0 {
            self.embedder.as_ref().and_then(|e| e.embed(text_span).ok())
        } else {
            None
        };

        let mut best: Option<(usize, f32)> = None;

        for (i, entry) in self.custom_entries.iter().enumerate() {
            let phon_sim = phoneme_similarity(phonemes, &entry.phonemes);

            let score = match (&text_emb, &entry.embedding) {
                (Some(span_emb), Some(entry_emb)) => {
                    let text_sim = cosine_similarity(span_emb, entry_emb);
                    self.alpha * phon_sim + (1.0 - self.alpha) * text_sim
                }
                _ => phon_sim,
            };

            if score >= self.threshold && (best.is_none() || score > best.unwrap().1) {
                best = Some((i, score));
            }
        }

        best
    }
}

#[async_trait]
impl TextProcessor for PhonemeCorrector {
    async fn process(
        &self,
        text: &str,
        _ctx: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError> {
        if self.custom_entries.is_empty() {
            return Ok(ProcessResult {
                text: text.to_string(),
                corrections: vec![],
            });
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(ProcessResult {
                text: String::new(),
                corrections: vec![],
            });
        }

        // Get phonemes for each word
        let word_phonemes: Vec<Option<String>> =
            words.iter().map(|w| self.word_to_phonemes(w)).collect();

        let mut result_words: Vec<String> = words.iter().map(|w| w.to_string()).collect();
        let mut consumed = vec![false; words.len()]; // track merged words
        let mut corrections = Vec::new();

        // Try merging adjacent words (for "use effect" → "useEffect")
        // Collect all candidate matches, then pick non-overlapping set with best scores.
        struct Candidate {
            start: usize,
            len: usize,
            entry_idx: usize,
            similarity: f32,
        }

        let mut candidates: Vec<Candidate> = Vec::new();

        for i in 0..words.len() {
            // Single word match
            if let Some(phonemes) = &word_phonemes[i] {
                if let Some((idx, sim)) = self.best_match(phonemes, words[i]) {
                    if words[i].to_lowercase() != self.custom_entries[idx].word.to_lowercase() {
                        candidates.push(Candidate {
                            start: i,
                            len: 1,
                            entry_idx: idx,
                            similarity: sim,
                        });
                    }
                }
            }

            // Multi-word merges
            for merge_len in 2..=self.max_merge.min(words.len() - i) {
                let window_phonemes: Option<String> = (i..i + merge_len)
                    .map(|j| word_phonemes[j].as_deref())
                    .collect::<Option<Vec<_>>>()
                    .map(|parts| parts.concat());

                if let Some(merged) = &window_phonemes {
                    let text_span: String = (i..i + merge_len)
                        .map(|j| words[j])
                        .collect::<Vec<_>>()
                        .join(" ");
                    if let Some((idx, sim)) = self.best_match(merged, &text_span) {
                        candidates.push(Candidate {
                            start: i,
                            len: merge_len,
                            entry_idx: idx,
                            similarity: sim,
                        });
                    }
                }
            }
        }

        // Greedy selection: sort by similarity (desc), then by shorter merge (prefer precise),
        // and pick non-overlapping candidates.
        candidates.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap()
                .then_with(|| a.len.cmp(&b.len))
        });

        for cand in &candidates {
            let end = cand.start + cand.len;
            // Check overlap with already consumed positions
            if (cand.start..end).any(|j| consumed[j]) {
                continue;
            }

            let original: Vec<&str> = (cand.start..end).map(|j| words[j]).collect();
            let original_str = original.join(" ");

            tracing::debug!(
                original = %original_str,
                replacement = %self.custom_entries[cand.entry_idx].word,
                similarity = cand.similarity,
                merge_len = cand.len,
                "phoneme match"
            );

            corrections.push(Correction {
                kind: CorrectionKind::DictionaryMatch,
                original: original_str,
                replacement: self.custom_entries[cand.entry_idx].word.clone(),
            });

            result_words[cand.start] = self.custom_entries[cand.entry_idx].word.clone();
            consumed[cand.start..end].fill(true);
        }

        // Rebuild text, skipping consumed (merged) words
        let final_words: Vec<&str> = result_words
            .iter()
            .enumerate()
            .filter(|(i, _)| !consumed[*i] || *i < words.len() && result_words[*i] != words[*i])
            .map(|(_, w)| w.as_str())
            .collect();

        Ok(ProcessResult {
            text: final_words.join(" "),
            corrections,
        })
    }
}

// ---------------------------------------------------------------------------
// ONNX G2P Backend (feature-gated)
// ---------------------------------------------------------------------------

/// ONNX-based grapheme-to-phoneme backend using DeepPhonemizer.
///
/// Converts arbitrary words (including OOV/proper nouns) to IPA phoneme strings
/// using a Transformer model with CTC decoding.  ~59MB ONNX model.
///
/// Requires `--features onnx` and model files:
/// - `g2p.onnx` — the DeepPhonemizer forward transformer
/// - `tokenizer.json` — character-to-index and index-to-phoneme mappings
#[cfg(feature = "onnx")]
pub struct OnnxG2p {
    session: std::sync::Mutex<ort::session::Session>,
    text_to_idx: HashMap<String, i64>,
    idx_to_phoneme: HashMap<i64, String>,
    lang_token: i64,
    char_repeats: usize,
}

#[cfg(feature = "onnx")]
impl OnnxG2p {
    /// Load from a directory containing `g2p.onnx` and `tokenizer.json`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, ProcessError> {
        let dir = model_dir.as_ref();

        let session = ort::session::Session::builder()
            .and_then(|mut b| b.commit_from_file(dir.join("g2p.onnx")))
            .map_err(|e| ProcessError {
                message: format!("load G2P model: {e}"),
            })?;

        let tok_data =
            std::fs::read_to_string(dir.join("tokenizer.json")).map_err(|e| ProcessError {
                message: format!("load tokenizer: {e}"),
            })?;
        let tok: serde_json::Value = serde_json::from_str(&tok_data).map_err(|e| ProcessError {
            message: format!("parse tokenizer: {e}"),
        })?;

        let text_to_idx: HashMap<String, i64> = tok["text_to_idx"]
            .as_object()
            .ok_or_else(|| ProcessError {
                message: "missing text_to_idx".into(),
            })?
            .iter()
            .map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(0)))
            .collect();

        let idx_to_phoneme: HashMap<i64, String> = tok["idx_to_phoneme"]
            .as_object()
            .ok_or_else(|| ProcessError {
                message: "missing idx_to_phoneme".into(),
            })?
            .iter()
            .map(|(k, v)| {
                (
                    k.parse::<i64>().unwrap_or(0),
                    v.as_str().unwrap_or("").to_string(),
                )
            })
            .collect();

        let lang_token = *text_to_idx.get("<en_us>").unwrap_or(&2);

        tracing::info!(
            text_symbols = text_to_idx.len(),
            phoneme_symbols = idx_to_phoneme.len(),
            "ONNX G2P loaded"
        );

        Ok(Self {
            session: std::sync::Mutex::new(session),
            text_to_idx,
            idx_to_phoneme,
            lang_token,
            char_repeats: 3,
        })
    }

    /// Tokenize a word: lang_token + char indices, each repeated char_repeats times.
    fn tokenize(&self, word: &str) -> Vec<i64> {
        let mut tokens = vec![self.lang_token];
        for c in word.to_lowercase().chars() {
            if let Some(&idx) = self.text_to_idx.get(&c.to_string()) {
                tokens.push(idx);
            }
        }
        // Repeat each token
        let mut repeated = Vec::with_capacity(tokens.len() * self.char_repeats);
        for t in &tokens {
            for _ in 0..self.char_repeats {
                repeated.push(*t);
            }
        }
        repeated
    }

    /// CTC decode: argmax → remove blanks (0) and consecutive duplicates → map to IPA.
    fn ctc_decode(&self, logits: &[f32], seq_len: usize, n_classes: usize) -> String {
        let mut decoded = Vec::new();
        let mut prev: Option<i64> = None;

        for t in 0..seq_len {
            let offset = t * n_classes;
            let best = (0..n_classes)
                .max_by(|&a, &b| logits[offset + a].partial_cmp(&logits[offset + b]).unwrap())
                .unwrap_or(0) as i64;

            // Skip CTC blank (0) and consecutive duplicates
            if best != 0 && Some(best) != prev {
                decoded.push(best);
            }
            prev = Some(best);
        }

        // Map to IPA symbols and strip special tokens
        decoded
            .iter()
            .filter_map(|&idx| self.idx_to_phoneme.get(&idx))
            .filter(|s| !s.starts_with('<'))
            .cloned()
            .collect::<String>()
    }
}

#[cfg(feature = "onnx")]
impl G2pBackend for OnnxG2p {
    fn phonemize(&self, word: &str) -> Result<String, ProcessError> {
        use ndarray::{Array1, Array2};
        use ort::value::Value;

        if word.is_empty() {
            return Ok(String::new());
        }

        let tokens = self.tokenize(word);
        let seq_len = tokens.len();

        let text = Array2::from_shape_vec((1, seq_len), tokens).map_err(|e| ProcessError {
            message: format!("shape: {e}"),
        })?;
        let start_index =
            Array2::from_shape_vec((1, 1), vec![0_i64]).map_err(|e| ProcessError {
                message: format!("shape: {e}"),
            })?;
        let text_len = Array1::from_vec(vec![seq_len as i64]);

        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(vec![
                (
                    "text",
                    Value::from_array(text)
                        .map_err(|e| ProcessError {
                            message: format!("{e}"),
                        })?
                        .into_dyn(),
                ),
                (
                    "start_index",
                    Value::from_array(start_index)
                        .map_err(|e| ProcessError {
                            message: format!("{e}"),
                        })?
                        .into_dyn(),
                ),
                (
                    "text_len",
                    Value::from_array(text_len)
                        .map_err(|e| ProcessError {
                            message: format!("{e}"),
                        })?
                        .into_dyn(),
                ),
            ])
            .map_err(|e| ProcessError {
                message: format!("G2P inference: {e}"),
            })?;

        let logits = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| ProcessError {
                message: format!("extract: {e}"),
            })?;
        let view = logits.view();
        let out_seq = view.shape()[1];
        let n_classes = view.shape()[2];
        let logits_flat: Vec<f32> = view.iter().copied().collect();

        drop(outputs);
        drop(session);

        Ok(self.ctc_decode(&logits_flat, out_seq, n_classes))
    }
}

// ---------------------------------------------------------------------------
// ONNX Text Embedder (feature-gated)
// ---------------------------------------------------------------------------

/// Text embedder using a sentence-transformer ONNX model (e.g., bge-small-en-v1.5).
///
/// Computes L2-normalized CLS embeddings for text strings.
/// Used by PhonemeCorrector for composite phoneme+semantic scoring.
///
/// Requires `--features onnx` and model files:
/// - `model.onnx` — BERT/bge sentence transformer
/// - `tokenizer.json` — HuggingFace tokenizer
#[cfg(feature = "onnx")]
pub struct OnnxTextEmbedder {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "onnx")]
impl OnnxTextEmbedder {
    /// Load from a directory containing `model.onnx` and `tokenizer.json`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, ProcessError> {
        let dir = model_dir.as_ref();
        let session = ort::session::Session::builder()
            .and_then(|mut b| b.commit_from_file(dir.join("model.onnx")))
            .map_err(|e| ProcessError {
                message: format!("load embedder: {e}"),
            })?;
        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| {
                ProcessError {
                    message: format!("load tokenizer: {e}"),
                }
            })?;
        tracing::info!("ONNX text embedder loaded");
        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
        })
    }
}

#[cfg(feature = "onnx")]
impl TextEmbedder for OnnxTextEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, ProcessError> {
        use ndarray::Array2;
        use ort::value::Value;

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
        let tids = Array2::from_shape_vec(
            (1, len),
            enc.get_type_ids().iter().map(|&x| x as i64).collect(),
        )
        .unwrap();

        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(vec![
                (
                    "input_ids",
                    Value::from_array(ids)
                        .map_err(|e| ProcessError {
                            message: format!("{e}"),
                        })?
                        .into_dyn(),
                ),
                (
                    "attention_mask",
                    Value::from_array(mask)
                        .map_err(|e| ProcessError {
                            message: format!("{e}"),
                        })?
                        .into_dyn(),
                ),
                (
                    "token_type_ids",
                    Value::from_array(tids)
                        .map_err(|e| ProcessError {
                            message: format!("{e}"),
                        })?
                        .into_dyn(),
                ),
            ])
            .map_err(|e| ProcessError {
                message: format!("embed: {e}"),
            })?;

        let arr = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| ProcessError {
                message: format!("extract: {e}"),
            })?;
        let view = arr.view();
        let hidden_dim = view.shape()[2];
        let cls: Vec<f32> = (0..hidden_dim).map(|i| view[[0, 0, i]]).collect();

        drop(outputs);
        drop(session);

        // L2 normalize
        let norm: f32 = cls.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            Ok(cls.iter().map(|x| x / norm).collect())
        } else {
            Ok(cls)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phoneme_distance_identical() {
        assert_eq!(phoneme_distance("həloʊ", "həloʊ"), 0);
    }

    #[test]
    fn test_phoneme_distance_one_edit() {
        // "ɪfɛkt" vs "ɛfɛkt" — one substitution
        assert_eq!(phoneme_distance("ɪfɛkt", "ɛfɛkt"), 1);
    }

    #[test]
    fn test_phoneme_similarity() {
        let sim = phoneme_similarity("juːsɪfɛkt", "juːsɪfɛkt");
        assert!((sim - 1.0).abs() < 1e-6);

        let sim2 = phoneme_similarity("juːs", "juːsɪfɛkt");
        assert!(sim2 < 0.7); // quite different lengths
    }

    #[test]
    fn test_phoneme_distance_empty() {
        assert_eq!(phoneme_distance("", "abc"), 3);
        assert_eq!(phoneme_distance("abc", ""), 3);
        assert_eq!(phoneme_distance("", ""), 0);
    }

    #[tokio::test]
    async fn test_corrector_single_word() {
        let dict = IpaDictionary::empty();
        let custom = vec![CustomEntry {
            word: "Kubernetes".into(),
            phonemes: "kuːbɝniːts".into(),
            embedding: None,
        }];
        // Simulate ASR producing "kuber nets" → phonemes not in empty dict
        // This tests the case where we can't look up phonemes — no crash
        let corrector = PhonemeCorrector::new(dict, custom);
        let ctx = ContextSnapshot::default();
        let result = corrector.process("kuber nets", &ctx).await.unwrap();
        // No phonemes found → no correction (graceful)
        assert_eq!(result.text, "kuber nets");
    }

    #[tokio::test]
    async fn test_corrector_merge_with_dict() {
        // Build a minimal IPA dict
        let mut entries = HashMap::new();
        entries.insert("use".into(), "juːs".into());
        entries.insert("effect".into(), "ɪfɛkt".into());
        entries.insert("java".into(), "dʒɑːvə".into());
        entries.insert("script".into(), "skrɪpt".into());
        let dict = IpaDictionary { entries };

        let custom = vec![
            CustomEntry {
                word: "useEffect".into(),
                phonemes: "juːsɪfɛkt".into(),
                embedding: None,
            },
            CustomEntry {
                word: "JavaScript".into(),
                phonemes: "dʒɑːvəskrɪpt".into(),
                embedding: None,
            },
        ];

        let corrector = PhonemeCorrector::new(dict, custom);
        let ctx = ContextSnapshot::default();

        // "use effect" should merge to "useEffect"
        let r = corrector.process("use effect", &ctx).await.unwrap();
        assert_eq!(r.text, "useEffect");
        assert_eq!(r.corrections.len(), 1);

        // "java script" should merge to "JavaScript"
        let r2 = corrector.process("java script", &ctx).await.unwrap();
        assert_eq!(r2.text, "JavaScript");

        // Mixed sentence
        let r3 = corrector
            .process("I called use effect in java script", &ctx)
            .await
            .unwrap();
        assert_eq!(r3.text, "I called useEffect in JavaScript");
    }

    #[tokio::test]
    async fn test_corrector_no_false_positive() {
        let mut entries = HashMap::new();
        entries.insert("use".into(), "juːs".into());
        entries.insert("the".into(), "ðə".into());
        entries.insert("computer".into(), "kəmpjuːtɝ".into());
        let dict = IpaDictionary { entries };

        let custom = vec![CustomEntry {
            word: "useEffect".into(),
            phonemes: "juːsɪfɛkt".into(),
            embedding: None,
        }];

        let corrector = PhonemeCorrector::new(dict, custom);
        let ctx = ContextSnapshot::default();

        // "use the computer" — "use" alone is too short to match "useEffect" phonemes
        let r = corrector.process("use the computer", &ctx).await.unwrap();
        assert_eq!(r.text, "use the computer");
        assert_eq!(r.corrections.len(), 0);
    }

    #[tokio::test]
    async fn test_corrector_empty_custom_dict() {
        let dict = IpaDictionary::empty();
        let corrector = PhonemeCorrector::new(dict, vec![]);
        let ctx = ContextSnapshot::default();

        let r = corrector.process("hello world", &ctx).await.unwrap();
        assert_eq!(r.text, "hello world");
        assert_eq!(r.corrections.len(), 0);
    }
}
