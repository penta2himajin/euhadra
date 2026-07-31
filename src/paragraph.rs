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
/// When an embedder is provided, breaks are placed at *valleys* in the
/// adjacent-sentence similarity sequence — local minima that sit
/// measurably below their surroundings — rather than wherever
/// similarity falls under a fixed number. The distinction matters
/// because a raw cosine's absolute position is a property of the
/// embedding model, not of the sentences: the same pair scores ~0.2
/// higher on `granite-embedding-97m-multilingual-r2` than on
/// `bge-small-en-v1.5`. Comparing valleys to each other cancels that
/// offset out; comparing them to a constant does not.
///
/// Without an embedder, only the max-sentences constraint is applied.
///
/// Paragraph splitting is only applied for field types where it makes
/// sense (Document, EmailCompose).  For ChatMessage, Terminal, SearchBar,
/// the text passes through unchanged.
pub struct ParagraphSplitter {
    embedder: Option<Box<dyn TextEmbedder>>,
    /// Fraction of the document's deepest valley that a local minimum
    /// must reach to become a paragraph break.
    ///
    /// A **shape** parameter, not a location: it compares valleys to
    /// each other within one text, so it carries across embedding
    /// backends unchanged. The absolute cosine threshold it replaced
    /// did not — granite scores even unrelated strings at 0.62-0.75,
    /// so the old default of 0.5 sat below that backend's entire
    /// output range and the semantic path never fired at all.
    /// Default: 0.5
    pub depth_ratio: f32,
    /// Minimum spread (max - min) of adjacent-sentence similarities for
    /// a text to be considered to have any topic structure worth
    /// splitting on.
    ///
    /// Purely a degeneracy guard: without it, a relative rule always
    /// finds *some* lowest point and would split uniform text. It is a
    /// magnitude, but a magnitude of a *range* rather than of a
    /// position, so a backend that shifts all its similarities up or
    /// down does not move it.
    /// Default: 0.05
    pub min_similarity_range: f32,
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
            depth_ratio: 0.5,
            min_similarity_range: 0.05,
            max_sentences: 8,
            separator: "\n\n".to_string(),
        }
    }

    /// Builder: set the text embedder for semantic segmentation.
    pub fn with_embedder(mut self, embedder: impl TextEmbedder + 'static) -> Self {
        self.embedder = Some(Box::new(embedder));
        self
    }

    /// Builder: set the valley-depth ratio.
    pub fn with_depth_ratio(mut self, ratio: f32) -> Self {
        self.depth_ratio = ratio;
        self
    }

    /// Builder: set the minimum similarity spread required before any
    /// semantic split is attempted.
    pub fn with_min_similarity_range(mut self, range: f32) -> Self {
        self.min_similarity_range = range;
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

        // Phase 1: semantic breaks at similarity valleys.
        let breaks = self.semantic_breaks(similarities);

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


    /// Depth of the valley at `i`: how far it sits below the highest
    /// point reachable by walking outward in each direction without
    /// descending.
    ///
    /// This is the quantity that makes the rule portable. It is built
    /// entirely from *differences* between similarities, so adding a
    /// constant offset to every similarity — which is what changing
    /// embedding backend largely does — leaves it unchanged.
    fn valley_depth(sims: &[f32], i: usize) -> f32 {
        let mut left = sims[i];
        let mut j = i;
        while j > 0 && sims[j - 1] >= left {
            left = sims[j - 1];
            j -= 1;
        }

        let mut right = sims[i];
        let mut j = i;
        while j + 1 < sims.len() && sims[j + 1] >= right {
            right = sims[j + 1];
            j += 1;
        }

        (left - sims[i]) + (right - sims[i])
    }

    /// Whether `i` is a local minimum. The two ends count when they are
    /// lower than their single neighbour, so a topic shift at the very
    /// start or end of a dictation is not missed.
    fn is_local_min(sims: &[f32], i: usize) -> bool {
        let left_ok = i == 0 || sims[i - 1] > sims[i];
        let right_ok = i + 1 == sims.len() || sims[i + 1] > sims[i];
        left_ok && right_ok
    }

    /// Break points from valleys in the similarity sequence.
    ///
    /// Returns empty — deferring entirely to the max-sentences
    /// constraint — when there is nothing to judge against: fewer than
    /// two boundaries, any failed embedding, or a similarity spread too
    /// narrow to indicate structure.
    fn semantic_breaks(&self, similarities: &[Option<f32>]) -> Vec<usize> {
        if similarities.len() < 2 {
            return Vec::new();
        }
        // A failed embedding leaves a hole; rather than invent a value
        // for it and risk fabricating a valley, skip semantic splitting
        // and let max_sentences handle the text.
        let sims: Vec<f32> = match similarities.iter().copied().collect::<Option<Vec<f32>>>() {
            Some(v) => v,
            None => return Vec::new(),
        };

        let max = sims.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = sims.iter().copied().fold(f32::INFINITY, f32::min);
        if max - min < self.min_similarity_range {
            return Vec::new();
        }

        let candidates: Vec<(usize, f32)> = (0..sims.len())
            .filter(|&i| Self::is_local_min(&sims, i))
            .map(|i| (i, Self::valley_depth(&sims, i)))
            .collect();

        let deepest = candidates
            .iter()
            .map(|(_, d)| *d)
            .fold(f32::NEG_INFINITY, f32::max);
        if !deepest.is_finite() || deepest <= 0.0 {
            return Vec::new();
        }

        candidates
            .iter()
            .filter(|(_, d)| *d >= self.depth_ratio * deepest)
            .map(|(i, _)| i + 1) // break BEFORE sentence i+1
            .collect()
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

    // The `calibrated_similarity` constants these replace asserted that
    // a per-backend threshold table was internally consistent. The
    // table is gone; what mattered about it — that the split decision
    // survives a change of embedding backend — is now asserted
    // directly against the mechanism.

    fn depths(sims: &[f32]) -> Vec<f32> {
        (0..sims.len())
            .map(|i| ParagraphSplitter::valley_depth(sims, i))
            .collect()
    }

    #[test]
    fn valley_depth_is_invariant_under_a_uniform_shift() {
        // The portability property, stated as a test. granite sits
        // roughly 0.2 above bge-small for the same relationships; a
        // rule built from differences must not notice.
        let bge = [0.68, 0.66, 0.51, 0.67, 0.69];
        let granite: Vec<f32> = bge.iter().map(|s| s + 0.20).collect();

        for (a, b) in depths(&bge).iter().zip(depths(&granite).iter()) {
            assert!((a - b).abs() < 1e-5, "depth moved under shift: {a} vs {b}");
        }
    }

    #[test]
    fn valley_depth_measures_the_drop_from_both_sides() {
        // The walk climbs to the *peak* on each side, not merely to the
        // immediate neighbour: right of the valley it passes 0.87 and
        // carries on to 0.89.
        //   (0.88 - 0.71) + (0.89 - 0.71) = 0.17 + 0.18 = 0.35
        let sims = [0.88, 0.86, 0.71, 0.87, 0.89];
        let d = ParagraphSplitter::valley_depth(&sims, 2);
        assert!((d - 0.35).abs() < 1e-5, "got {d}");
    }

    #[test]
    fn a_monotone_decline_has_no_interior_valley() {
        // Falling similarity is a gradual topic drift, not a boundary.
        // Only the final point is a minimum, and it has no drop on its
        // right, so its depth comes from one side only.
        let sims = [0.9, 0.8, 0.7, 0.6];
        for i in 0..3 {
            assert!(!ParagraphSplitter::is_local_min(&sims, i), "index {i}");
        }
        assert!(ParagraphSplitter::is_local_min(&sims, 3));
    }

    #[test]
    fn ends_count_as_minima_so_edge_shifts_are_not_missed() {
        assert!(ParagraphSplitter::is_local_min(&[0.2, 0.9, 0.9], 0));
        assert!(ParagraphSplitter::is_local_min(&[0.9, 0.9, 0.2], 2));
    }

    #[test]
    fn same_breaks_on_both_backends_for_the_same_text() {
        let splitter = ParagraphSplitter::new();
        let bge: Vec<Option<f32>> = [0.68, 0.66, 0.51, 0.67, 0.69]
            .iter()
            .map(|s| Some(*s))
            .collect();
        let granite: Vec<Option<f32>> = bge.iter().map(|s| Some(s.unwrap() + 0.20)).collect();

        let a = splitter.semantic_breaks(&bge);
        let b = splitter.semantic_breaks(&granite);
        assert_eq!(a, b, "backend shift changed the break points");
        assert_eq!(a, vec![3], "expected the single valley at index 2");
    }

    #[test]
    fn the_old_absolute_rule_would_have_disagreed() {
        // Documents the defect being fixed. A 0.5 cosine threshold
        // fires on every bge-small boundary below it and on none of
        // the granite ones, for the very same text.
        const OLD_THRESHOLD: f32 = 0.5;
        let bge = [0.68, 0.66, 0.51, 0.67, 0.69];
        let granite: Vec<f32> = bge.iter().map(|s| s + 0.20).collect();

        assert_eq!(bge.iter().filter(|s| **s < OLD_THRESHOLD).count(), 0);
        assert_eq!(granite.iter().filter(|s| **s < OLD_THRESHOLD).count(), 0);

        // ...and at a threshold tuned so bge fires once, granite still
        // cannot fire at all, because its whole range sits above it.
        const BGE_TUNED: f32 = 0.6;
        assert_eq!(bge.iter().filter(|s| **s < BGE_TUNED).count(), 1);
        assert_eq!(granite.iter().filter(|s| **s < BGE_TUNED).count(), 0);
    }

    #[test]
    fn flat_similarity_does_not_split() {
        // Without the range guard a relative rule always finds *some*
        // lowest point, and uniform text would be split arbitrarily.
        let splitter = ParagraphSplitter::new();
        let flat: Vec<Option<f32>> = [0.90, 0.90, 0.89, 0.90].iter().map(|s| Some(*s)).collect();
        assert!(splitter.semantic_breaks(&flat).is_empty());
    }

    #[test]
    fn a_failed_embedding_disables_semantic_splitting() {
        let splitter = ParagraphSplitter::new();
        let holed = vec![Some(0.9), None, Some(0.2), Some(0.9)];
        assert!(splitter.semantic_breaks(&holed).is_empty());
    }

    #[test]
    fn a_single_boundary_is_not_enough_to_judge() {
        let splitter = ParagraphSplitter::new();
        assert!(splitter.semantic_breaks(&[Some(0.1)]).is_empty());
        assert!(splitter.semantic_breaks(&[]).is_empty());
    }

    #[test]
    fn shallow_valleys_are_dropped_relative_to_the_deepest() {
        // Two minima, one a third as deep as the other: the default
        // ratio of 0.5 keeps only the prominent one.
        let splitter = ParagraphSplitter::new();
        let sims: Vec<Option<f32>> = [0.90, 0.80, 0.90, 0.30, 0.90]
            .iter()
            .map(|s| Some(*s))
            .collect();
        assert_eq!(splitter.semantic_breaks(&sims), vec![4]);

        // Lowering the ratio admits the shallower one too.
        let permissive = ParagraphSplitter::new().with_depth_ratio(0.1);
        assert_eq!(permissive.semantic_breaks(&sims), vec![2, 4]);
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
            .with_depth_ratio(0.5);
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
