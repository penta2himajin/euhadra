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
//!
//! [`OnnxEntityRecognizer`] is not in that chain on purpose. It changes
//! no text — it reports what it found and passes the input through — so
//! it earns its place only once something consumes the entities. Use it
//! directly for that:
//!
//! ```no_run
//! use euhadra::onnx_processing::OnnxEntityRecognizer;
//!
//! # async fn f() -> Result<(), Box<dyn std::error::Error>> {
//! let ner = OnnxEntityRecognizer::load(
//!     "vendor/ner_distilbert/model.onnx",
//!     "vendor/ner_distilbert/tokenizer.json",
//!     OnnxEntityRecognizer::default_labels(),
//! )?;
//! for entity in ner.detect("Alice works at Google in New York").await? {
//!     println!("{} {:?}", entity.label.as_str(), entity.text);
//! }
//! # Ok(())
//! # }
//! ```

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
                        span: None,
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
                    span: None,
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

// ---------------------------------------------------------------------------
// OnnxEntityRecognizer
// ---------------------------------------------------------------------------

/// The entity classes a NER model assigns.
///
/// CoNLL-2003's four, which is what `dslim/distilbert-NER` and its
/// relatives emit. Deliberately not extended: a label this type does not
/// know is dropped rather than guessed at, so an unfamiliar model
/// produces fewer entities rather than wrong ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityLabel {
    Person,
    Location,
    Organisation,
    Misc,
}

impl EntityLabel {
    /// Parse the class half of a BIO tag (`PER`, `LOC`, `ORG`, `MISC`).
    fn from_class(class: &str) -> Option<Self> {
        match class {
            "PER" => Some(Self::Person),
            "LOC" => Some(Self::Location),
            "ORG" => Some(Self::Organisation),
            "MISC" => Some(Self::Misc),
            _ => None,
        }
    }

    /// The CoNLL class string, for diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Person => "PER",
            Self::Location => "LOC",
            Self::Organisation => "ORG",
            Self::Misc => "MISC",
        }
    }
}

/// One recognised entity, with codepoint offsets into the source text.
///
/// Offsets are codepoints rather than bytes, matching [`crate::types::Span`]
/// and the filter layer, so a caller can slice with `chars()` without
/// worrying about multi-byte boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entity {
    pub label: EntityLabel,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// A word with its codepoint span in the source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WordSpan {
    start: usize,
    end: usize,
}

/// Split on whitespace, keeping each word's codepoint span.
///
/// `str::split_whitespace` discards positions, and the recogniser needs
/// them to report offsets a caller can slice with.
fn word_spans(text: &str) -> Vec<WordSpan> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in text.chars().enumerate() {
        match (c.is_whitespace(), start) {
            (false, None) => start = Some(i),
            (true, Some(s)) => {
                out.push(WordSpan { start: s, end: i });
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push(WordSpan {
            start: s,
            end: text.chars().count(),
        });
    }
    out
}

/// Merge per-word BIO tags into entity spans.
///
/// Pure, so it is tested without loading a model — which matters,
/// because this is where NER output is most easily got wrong and the
/// inference call around it cannot be exercised in CI.
///
/// Handles the malformed sequences real models emit. An `I-` tag with no
/// preceding `B-` of the same class opens an entity anyway: dropping it
/// would silently lose a detection, and every NER decoder in practice
/// treats a stray `I-` as a beginning. A class change mid-entity closes
/// the current one and opens the next, rather than merging two
/// neighbouring entities of different types into one span.
fn merge_bio(text: &str, words: &[WordSpan], tags: &[&str]) -> Vec<Entity> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<Entity> = Vec::new();
    let mut open: Option<(EntityLabel, usize, usize)> = None;

    let flush = |open: &mut Option<(EntityLabel, usize, usize)>, out: &mut Vec<Entity>| {
        if let Some((label, start, end)) = open.take() {
            out.push(Entity {
                label,
                start,
                end,
                text: chars[start..end].iter().collect(),
            });
        }
    };

    for (i, word) in words.iter().enumerate() {
        let tag = tags.get(i).copied().unwrap_or("O");
        let (prefix, class) = match tag.split_once('-') {
            Some((p, c)) => (p, c),
            // "O", or anything that is not a BIO tag at all.
            None => {
                flush(&mut open, &mut out);
                continue;
            }
        };
        let Some(label) = EntityLabel::from_class(class) else {
            flush(&mut open, &mut out);
            continue;
        };

        match (prefix, open) {
            // Continuation of the same class.
            ("I", Some((cur, start, _))) if cur == label => {
                open = Some((label, start, word.end));
            }
            // B-, a class change, or a stray I- with nothing open.
            _ => {
                flush(&mut open, &mut out);
                open = Some((label, word.start, word.end));
            }
        }
    }
    flush(&mut open, &mut out);
    out
}

/// Named-entity recognition via an ONNX token-classification model.
///
/// Same shape as [`OnnxPunctuationRestorer`] — BERT-family token
/// classification, first-subword-per-word pooling — differing only in
/// what the labels mean. `dslim/distilbert-NER` is the reference model
/// (~65 MB INT8); any model emitting CoNLL-2003 BIO tags will do,
/// provided its label order is passed to [`Self::load`].
///
/// As a [`TextProcessor`] this changes nothing. It reports what it found
/// as [`CorrectionKind::EntityDetected`] corrections and returns the
/// text untouched, so it is safe to drop anywhere in a processor chain.
/// The structured output is [`Self::detect`].
pub struct OnnxEntityRecognizer {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    labels: Vec<String>,
}

impl OnnxEntityRecognizer {
    /// Load a model, its tokenizer, and the label list in id order.
    ///
    /// The labels must match the model's `id2label` ordering. Getting
    /// this wrong does not fail loudly — it mislabels every entity — so
    /// take the order from the model's `config.json` rather than
    /// assuming [`Self::default_labels`] applies.
    pub fn load(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        labels: Vec<String>,
    ) -> Result<Self, ProcessError> {
        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(model_path.as_ref()))
            .map_err(|e| ProcessError::Unavailable(format!("load model: {e}")))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| ProcessError::Unavailable(format!("load tokenizer: {e}")))?;
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            labels,
        })
    }

    /// Label order used by `dslim/distilbert-NER`.
    ///
    /// Taken from that model's `config.json#id2label`, not from the
    /// conventional CoNLL ordering — they differ, and the difference is
    /// silent. Written from memory this list had MISC where PER belongs,
    /// which would have relabelled every person as MISC without any
    /// error. `scripts/setup_ner.sh` prints the model's own order after
    /// downloading so the two can be compared.
    pub fn default_labels() -> Vec<String> {
        [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Recognise entities, with codepoint offsets into `text`.
    pub async fn detect(&self, text: &str) -> Result<Vec<Entity>, ProcessError> {
        let words = word_spans(text);
        if words.is_empty() {
            return Ok(Vec::new());
        }

        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| ProcessError::Failed(format!("tokenize: {e}")))?;

        let len = enc.get_ids().len();
        let ids = Array2::from_shape_vec(
            (1, len),
            enc.get_ids().iter().map(|&x| x as i64).collect::<Vec<_>>(),
        )
        .map_err(|e| ProcessError::Failed(format!("input_ids: {e}")))?;
        let mask = Array2::from_shape_vec(
            (1, len),
            enc.get_attention_mask()
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>(),
        )
        .map_err(|e| ProcessError::Failed(format!("attention_mask: {e}")))?;

        let ids_val = Value::from_array(ids).map_err(|e| ProcessError::Failed(format!("{e}")))?;
        let mask_val = Value::from_array(mask).map_err(|e| ProcessError::Failed(format!("{e}")))?;

        let mut session = self.session.lock().await;
        let outputs = session
            .run(vec![
                ("input_ids", ids_val.into_dyn()),
                ("attention_mask", mask_val.into_dyn()),
            ])
            .map_err(|e| ProcessError::Inference(format!("inference: {e}")))?;

        let logits = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| ProcessError::Failed(format!("extract: {e}")))?;
        let view = logits.view();
        let seq_len = view.shape()[1];
        let num_labels = view.shape()[2];
        let logits_owned: Vec<f32> = view.iter().copied().collect();
        let word_ids: Vec<Option<u32>> = enc.get_word_ids().to_vec();
        drop(outputs);
        drop(session);

        // First subword decides the word, matching how these models are
        // trained: only the first subword of a word carries the label,
        // the rest are ignored in the loss.
        let mut word_tags: Vec<String> = vec!["O".into(); words.len()];
        let mut seen = vec![false; words.len()];
        for (ti, wid) in word_ids.iter().enumerate() {
            let Some(w) = wid.map(|w| w as usize) else {
                continue;
            };
            if w >= words.len() || seen[w] || ti >= seq_len {
                continue;
            }
            seen[w] = true;
            let offset = ti * num_labels;
            let best = (0..num_labels)
                .max_by(|&a, &b| {
                    logits_owned[offset + a].total_cmp(&logits_owned[offset + b])
                })
                .unwrap_or(0);
            if let Some(label) = self.labels.get(best) {
                word_tags[w] = label.clone();
            }
        }

        let tags: Vec<&str> = word_tags.iter().map(|s| s.as_str()).collect();
        Ok(merge_bio(text, &words, &tags))
    }
}

#[async_trait]
impl TextProcessor for OnnxEntityRecognizer {
    async fn process(
        &self,
        text: &str,
        _ctx: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError> {
        let entities = self.detect(text).await?;
        Ok(ProcessResult {
            text: text.to_string(),
            corrections: entities
                .into_iter()
                .map(|e| Correction {
                    kind: CorrectionKind::EntityDetected,
                    original: e.text.clone(),
                    replacement: e.text,
                    span: None,
                })
                .collect(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn spans(text: &str, tags: &[&str]) -> Vec<Entity> {
        let words = word_spans(text);
        assert_eq!(
            words.len(),
            tags.len(),
            "test gave {} tags for {} words",
            tags.len(),
            words.len()
        );
        merge_bio(text, &words, tags)
    }

    #[test]
    fn word_spans_are_codepoint_offsets() {
        let text = "café de flore";
        let w = word_spans(text);
        let chars: Vec<char> = text.chars().collect();
        let surfaces: Vec<String> = w
            .iter()
            .map(|s| chars[s.start..s.end].iter().collect())
            .collect();
        assert_eq!(surfaces, ["café", "de", "flore"]);
        // Byte offsets would put "de" at 6, not 5.
        assert_eq!(w[1].start, 5);
    }

    #[test]
    fn word_spans_handle_leading_trailing_and_repeated_whitespace() {
        assert_eq!(word_spans("").len(), 0);
        assert_eq!(word_spans("   ").len(), 0);
        let w = word_spans("  a   b  ");
        assert_eq!(w.len(), 2);
        assert_eq!((w[0].start, w[0].end), (2, 3));
        assert_eq!((w[1].start, w[1].end), (6, 7));
    }

    #[test]
    fn single_word_entity() {
        let e = spans("I met Alice today", &["O", "O", "B-PER", "O"]);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].label, EntityLabel::Person);
        assert_eq!(e[0].text, "Alice");
        assert_eq!((e[0].start, e[0].end), (6, 11));
    }

    #[test]
    fn multi_word_entity_merges() {
        let e = spans(
            "we deployed to New York City",
            &["O", "O", "O", "B-LOC", "I-LOC", "I-LOC"],
        );
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].text, "New York City");
        assert_eq!(e[0].label, EntityLabel::Location);
    }

    /// Two entities of the same class in a row must stay separate: `B-`
    /// starts a new one even when the previous is still open.
    #[test]
    fn adjacent_entities_of_the_same_class_do_not_merge() {
        let e = spans("Alice Bob talked", &["B-PER", "B-PER", "O"]);
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].text, "Alice");
        assert_eq!(e[1].text, "Bob");
    }

    /// A class change mid-entity closes the current span. Merging across
    /// it would invent an entity spanning two types.
    #[test]
    fn class_change_closes_the_open_entity() {
        let e = spans("Paris Hilton stayed", &["B-LOC", "I-PER", "O"]);
        assert_eq!(e.len(), 2);
        assert_eq!((e[0].label, e[0].text.as_str()), (EntityLabel::Location, "Paris"));
        assert_eq!((e[1].label, e[1].text.as_str()), (EntityLabel::Person, "Hilton"));
    }

    /// Models emit `I-` with no preceding `B-`. Dropping it would lose a
    /// real detection, so it opens an entity instead.
    #[test]
    fn stray_i_tag_opens_an_entity() {
        let e = spans("call Kubernetes now", &["O", "I-ORG", "O"]);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].text, "Kubernetes");
        assert_eq!(e[0].label, EntityLabel::Organisation);
    }

    #[test]
    fn entity_running_to_end_of_text_is_closed() {
        let e = spans("deployed to Kubernetes", &["O", "O", "B-ORG"]);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].text, "Kubernetes");
        assert_eq!(e[0].end, "deployed to Kubernetes".chars().count());
    }

    /// An unfamiliar class is dropped rather than guessed at, and it
    /// closes whatever was open rather than silently extending it.
    #[test]
    fn unknown_class_is_dropped_and_closes_the_span() {
        let e = spans("Alice met Bob", &["B-PER", "B-GPE", "B-PER"]);
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].text, "Alice");
        assert_eq!(e[1].text, "Bob");
    }

    #[test]
    fn all_outside_yields_nothing() {
        assert!(spans("nothing to see here", &["O", "O", "O", "O"]).is_empty());
    }

    /// Multi-byte text must slice by codepoint, or the surface comes out
    /// mangled and the offsets are unusable.
    #[test]
    fn multibyte_entity_surface_is_intact() {
        let e = spans("私は東京に行った", &["B-LOC"]);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].text, "私は東京に行った");
    }

    /// Fewer tags than words is treated as trailing `O` rather than a
    /// panic — a truncated model output should lose detections, not
    /// take the process down.
    #[test]
    fn missing_trailing_tags_are_outside() {
        let words = word_spans("Alice met Bob");
        let e = merge_bio("Alice met Bob", &words, &["B-PER"]);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].text, "Alice");
    }

    #[test]
    fn default_labels_cover_the_conll_classes() {
        let labels = OnnxEntityRecognizer::default_labels();
        assert_eq!(labels.len(), 9);
        assert_eq!(labels[0], "O");
        for class in ["PER", "LOC", "ORG", "MISC"] {
            assert!(labels.contains(&format!("B-{class}")));
            assert!(labels.contains(&format!("I-{class}")));
            assert!(EntityLabel::from_class(class).is_some());
        }
    }
}
