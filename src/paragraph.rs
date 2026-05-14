//! Paragraph segmentation for dictation text.
//!
//! Splits continuous dictation output into paragraphs using two signals:
//! 1. **Semantic distance**: Consecutive sentences with low embedding cosine
//!    similarity are separated into different paragraphs (topic shift).
//! 2. **Maximum paragraph length**: Paragraphs exceeding N sentences are
//!    split at the point of lowest inter-sentence similarity.
//!
//! The embedding-based approach is language-agnostic and requires no
//! hand-crafted marker lists.

use async_trait::async_trait;

use crate::phoneme::TextEmbedder;
use crate::processor::{ProcessError, ProcessResult, TextProcessor};
use crate::types::{ContextSnapshot, FieldType};

// ---------------------------------------------------------------------------
// Sentence splitter (lightweight, rule-based)
// ---------------------------------------------------------------------------

/// Split text into sentences on `.` `!` `?` boundaries.
/// Preserves the delimiter attached to the sentence.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if matches!(c, '.' | '!' | '?' | '。' | '！' | '？') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Remaining text without terminal punctuation
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

// ---------------------------------------------------------------------------
// ParagraphSplitter
// ---------------------------------------------------------------------------

/// Splits dictation text into paragraphs using semantic similarity
/// and maximum-length constraints.
///
/// When an embedder is provided, consecutive sentences whose cosine
/// similarity falls below `similarity_threshold` are separated by a
/// paragraph break.  Without an embedder, only the max-sentences
/// constraint is applied.
///
/// Paragraph splitting is only applied for field types where it makes
/// sense (Document, EmailCompose).  For ChatMessage, Terminal, SearchBar,
/// the text passes through unchanged.
pub struct ParagraphSplitter {
    embedder: Option<Box<dyn TextEmbedder>>,
    /// Cosine similarity threshold below which a paragraph break is inserted.
    /// Lower = fewer breaks, higher = more breaks.
    /// Default: 0.5
    pub similarity_threshold: f32,
    /// Maximum number of sentences per paragraph.
    /// When exceeded, the paragraph is split at the point of lowest
    /// inter-sentence similarity (or at the midpoint if no embedder).
    /// Default: 8
    pub max_sentences: usize,
    /// The string used to separate paragraphs.
    /// Default: "\n\n"
    pub separator: String,
}

impl Default for ParagraphSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl ParagraphSplitter {
    /// Create a splitter with only max-sentence constraint (no embedder).
    pub fn new() -> Self {
        Self {
            embedder: None,
            similarity_threshold: 0.5,
            max_sentences: 8,
            separator: "\n\n".to_string(),
        }
    }

    /// Builder: set the text embedder for semantic segmentation.
    pub fn with_embedder(mut self, embedder: impl TextEmbedder + 'static) -> Self {
        self.embedder = Some(Box::new(embedder));
        self
    }

    /// Builder: set similarity threshold.
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Builder: set max sentences per paragraph.
    pub fn with_max_sentences(mut self, max: usize) -> Self {
        self.max_sentences = max;
        self
    }

    /// Check if paragraph splitting should be applied for this field type.
    fn should_split(field_type: &Option<FieldType>) -> bool {
        match field_type {
            None => true, // default: split
            Some(FieldType::Document) | Some(FieldType::EmailCompose) => true,
            Some(FieldType::ChatMessage)
            | Some(FieldType::Terminal)
            | Some(FieldType::SearchBar)
            | Some(FieldType::CodeEditor) => false,
            Some(FieldType::Generic) => true,
        }
    }

    /// Compute embeddings for each sentence, returning None for failures.
    fn embed_sentences(&self, sentences: &[String]) -> Vec<Option<Vec<f32>>> {
        let embedder = match &self.embedder {
            Some(e) => e,
            None => return vec![None; sentences.len()],
        };

        sentences.iter().map(|s| embedder.embed(s).ok()).collect()
    }

    /// Cosine similarity between two embedding vectors.
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x * y)
            .sum::<f32>()
            .max(0.0)
    }

    /// Compute inter-sentence similarities. Returns N-1 similarity values
    /// where result[i] = similarity(sentence[i], sentence[i+1]).
    fn inter_sentence_similarities(&self, embeddings: &[Option<Vec<f32>>]) -> Vec<Option<f32>> {
        if embeddings.len() < 2 {
            return vec![];
        }

        (0..embeddings.len() - 1)
            .map(|i| match (&embeddings[i], &embeddings[i + 1]) {
                (Some(a), Some(b)) => Some(Self::cosine_sim(a, b)),
                _ => None,
            })
            .collect()
    }

    /// Find paragraph break points given inter-sentence similarities.
    fn find_breaks(&self, n_sentences: usize, similarities: &[Option<f32>]) -> Vec<usize> {
        if n_sentences <= 1 {
            return vec![];
        }

        let mut breaks = Vec::new();

        // Phase 1: semantic breaks (similarity below threshold)
        for (i, sim) in similarities.iter().enumerate() {
            if let Some(s) = sim {
                if *s < self.similarity_threshold {
                    breaks.push(i + 1); // break BEFORE sentence i+1
                }
            }
        }

        // Phase 2: enforce max_sentences constraint on each resulting paragraph
        let mut final_breaks = Vec::new();
        let mut prev_break = 0;

        for &br in &breaks {
            // Check if the segment [prev_break..br] is too long
            self.split_long_segment(prev_break, br, similarities, &mut final_breaks);
            final_breaks.push(br);
            prev_break = br;
        }
        // Handle the last segment
        self.split_long_segment(prev_break, n_sentences, similarities, &mut final_breaks);

        final_breaks.sort();
        final_breaks.dedup();
        final_breaks
    }

    /// If a segment exceeds max_sentences, split at the lowest-similarity point.
    fn split_long_segment(
        &self,
        start: usize,
        end: usize,
        similarities: &[Option<f32>],
        breaks: &mut Vec<usize>,
    ) {
        let len = end - start;
        if len <= self.max_sentences {
            return;
        }

        // Find the lowest similarity point in this segment
        let mut min_sim = f32::INFINITY;
        let mut min_idx = start + len / 2; // fallback: midpoint

        for i in start..end.saturating_sub(1) {
            if i < similarities.len() {
                if let Some(s) = similarities[i] {
                    if s < min_sim {
                        min_sim = s;
                        min_idx = i + 1;
                    }
                }
            }
        }

        breaks.push(min_idx);

        // Recurse on both halves
        self.split_long_segment(start, min_idx, similarities, breaks);
        self.split_long_segment(min_idx, end, similarities, breaks);
    }
}

#[async_trait]
impl TextProcessor for ParagraphSplitter {
    async fn process(
        &self,
        text: &str,
        ctx: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError> {
        // Skip splitting for field types where it doesn't make sense
        if !Self::should_split(&ctx.field_type) {
            return Ok(ProcessResult {
                text: text.to_string(),
                corrections: vec![],
            });
        }

        let sentences = split_sentences(text);

        if sentences.len() <= 1 {
            return Ok(ProcessResult {
                text: text.to_string(),
                corrections: vec![],
            });
        }

        let embeddings = self.embed_sentences(&sentences);
        let similarities = self.inter_sentence_similarities(&embeddings);
        let breaks = self.find_breaks(sentences.len(), &similarities);

        if breaks.is_empty() {
            return Ok(ProcessResult {
                text: text.to_string(),
                corrections: vec![],
            });
        }

        // Reconstruct text with paragraph breaks
        let mut paragraphs: Vec<String> = Vec::new();
        let mut current = Vec::new();

        for (i, sentence) in sentences.iter().enumerate() {
            if breaks.contains(&i) && !current.is_empty() {
                paragraphs.push(current.join(" "));
                current.clear();
            }
            current.push(sentence.as_str());
        }
        if !current.is_empty() {
            paragraphs.push(current.join(" "));
        }

        let result = paragraphs.join(&self.separator);

        tracing::debug!(
            n_sentences = sentences.len(),
            n_paragraphs = paragraphs.len(),
            breaks = ?breaks,
            "paragraph splitting applied"
        );

        Ok(ProcessResult {
            text: result,
            corrections: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences_basic() {
        let s = split_sentences("Hello world. How are you? I am fine!");
        assert_eq!(s, vec!["Hello world.", "How are you?", "I am fine!"]);
    }

    #[test]
    fn test_split_sentences_no_terminal() {
        let s = split_sentences("Hello world");
        assert_eq!(s, vec!["Hello world"]);
    }

    #[test]
    fn test_split_sentences_japanese() {
        let s = split_sentences("今日は天気がいい。明日は雨だ。");
        assert_eq!(s, vec!["今日は天気がいい。", "明日は雨だ。"]);
    }

    #[test]
    fn test_split_sentences_empty() {
        let s = split_sentences("");
        assert!(s.is_empty());
    }

    #[tokio::test]
    async fn test_splitter_single_sentence() {
        let splitter = ParagraphSplitter::new();
        let ctx = ContextSnapshot::default();
        let r = splitter.process("Hello world.", &ctx).await.unwrap();
        assert_eq!(r.text, "Hello world.");
    }

    #[tokio::test]
    async fn test_splitter_skip_chat() {
        let splitter = ParagraphSplitter::new();
        let ctx = ContextSnapshot {
            field_type: Some(FieldType::ChatMessage),
            ..Default::default()
        };
        let text = "First sentence. Second sentence. Third sentence. Fourth. Fifth. Sixth. Seventh. Eighth. Ninth. Tenth.";
        let r = splitter.process(text, &ctx).await.unwrap();
        assert_eq!(r.text, text); // no change for chat
    }

    #[tokio::test]
    async fn test_splitter_max_sentences_no_embedder() {
        let splitter = ParagraphSplitter::new().with_max_sentences(3);
        let ctx = ContextSnapshot::default();
        let text = "One. Two. Three. Four. Five. Six.";
        let r = splitter.process(text, &ctx).await.unwrap();
        // Should split: 6 sentences > max 3
        assert!(
            r.text.contains("\n\n"),
            "Expected paragraph break in: {}",
            r.text
        );
    }

    #[tokio::test]
    async fn test_splitter_under_max_no_change() {
        let splitter = ParagraphSplitter::new().with_max_sentences(10);
        let ctx = ContextSnapshot::default();
        let text = "One. Two. Three.";
        let r = splitter.process(text, &ctx).await.unwrap();
        assert!(!r.text.contains("\n\n"));
    }

    #[tokio::test]
    async fn test_splitter_semantic_break() {
        // Create embeddings where sentences 1-2 are similar, 3 is different
        let emb_a = vec![1.0, 0.0, 0.0]; // topic A
        let emb_b = vec![0.0, 1.0, 0.0]; // topic B (orthogonal = similarity 0)

        // We need a mock that returns different embeddings per sentence.
        // Use a stateful approach.
        struct OrderedEmbedder {
            embeddings: std::sync::Mutex<std::collections::VecDeque<Vec<f32>>>,
        }
        impl TextEmbedder for OrderedEmbedder {
            fn embed(&self, _text: &str) -> Result<Vec<f32>, ProcessError> {
                let mut q = self.embeddings.lock().unwrap();
                Ok(q.pop_front().unwrap_or_else(|| vec![0.0; 3]))
            }
        }

        let embedder = OrderedEmbedder {
            embeddings: std::sync::Mutex::new(
                vec![emb_a.clone(), emb_a.clone(), emb_b.clone()].into(),
            ),
        };

        let splitter = ParagraphSplitter::new()
            .with_embedder(embedder)
            .with_similarity_threshold(0.5);
        let ctx = ContextSnapshot::default();

        let text = "Dogs are great pets. Cats are also wonderful. The stock market crashed today.";
        let r = splitter.process(text, &ctx).await.unwrap();

        // Should split before "The stock market..." (topic shift)
        assert!(
            r.text.contains("\n\n"),
            "Expected paragraph break in: {}",
            r.text
        );
        let parts: Vec<&str> = r.text.split("\n\n").collect();
        assert_eq!(parts.len(), 2);
    }
}
