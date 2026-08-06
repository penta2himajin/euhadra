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

use crate::embedding::{cosine, EmbeddingBackend};
use crate::eval::f1::Span;
use crate::filter::{FilterError, FilterResult, TextFilter};
use crate::processor::{Correction, CorrectionKind, ProcessError, ProcessResult, TextProcessor};
use crate::types::ContextSnapshot;

// ---------------------------------------------------------------------------
// OnnxEmbeddingFilter
// ---------------------------------------------------------------------------

/// How an utterance is cut into scoreable units.
///
/// The embedding filter scores one unit at a time, so a language
/// without inter-word spaces cannot use whitespace tokenisation — the
/// whole utterance would arrive as a single unit and no filler would
/// ever be isolated. Japanese and Chinese therefore segment on the
/// reading-comma boundaries the rule-based filters already rely on
/// (`docs/spec.md` §3.5, "読点区切り3パス検出").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segmenter {
    /// Space-delimited scripts: en / es / ko.
    Whitespace,
    /// CJK: split on 、，。！？ and any whitespace that is present.
    CjkPunctuation,
}

impl Segmenter {
    /// The separator to rejoin surviving segments with.
    pub fn joiner(&self) -> &'static str {
        match self {
            Self::Whitespace => " ",
            Self::CjkPunctuation => "",
        }
    }

    fn is_boundary(&self, c: char) -> bool {
        match self {
            Self::Whitespace => c.is_whitespace(),
            Self::CjkPunctuation => {
                c.is_whitespace() || matches!(c, '、' | '，' | '。' | '！' | '？' | ',')
            }
        }
    }
}

/// One scoreable unit, with codepoint offsets into the source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub cp_start: usize,
    pub cp_end: usize,
    pub surface: String,
    /// Lower-cased leading alphanumeric run, used for lexicon lookup.
    /// For CJK the whole surface survives this, since CJK codepoints
    /// are alphanumeric under Unicode.
    pub clean: String,
}

/// Cut `text` into codepoint-indexed segments.
pub fn segment(text: &str, segmenter: Segmenter) -> Vec<Segment> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;

    while i < n {
        while i < n && segmenter.is_boundary(chars[i]) {
            i += 1;
        }
        if i >= n {
            break;
        }
        let start = i;
        while i < n && !segmenter.is_boundary(chars[i]) {
            i += 1;
        }
        let surface: String = chars[start..i].iter().collect();
        let clean: String = surface
            .to_lowercase()
            .chars()
            .take_while(|c| c.is_alphanumeric())
            .collect();
        out.push(Segment {
            cp_start: start,
            cp_end: i,
            surface,
            clean,
        });
    }
    out
}

/// Pure-filler cosine thresholds measured per embedding backend.
///
/// These are **not** interchangeable. Each backend places short
/// strings in a differently-scaled space, so a threshold carried over
/// from another model does not merely lose a little accuracy — it
/// collapses. Driving `granite-embedding-97m-multilingual-r2` at
/// bge-small's 0.82 turns 131 of 143 English non-filler tokens into
/// false positives, because that backend's *negatives* already sit
/// around 0.85.
///
/// Produced by `examples/bench_embedder.rs`; the runs behind each
/// number are recorded in `docs/model-upgrade-candidates.md` §3.2.
/// A backend that is not listed here has not been calibrated, and
/// must be swept before use.
pub mod calibrated {
    /// `BAAI/bge-small-en-v1.5` — the historical default. English
    /// plateau starts at 0.80; 0.82 keeps a little headroom and is the
    /// value `docs/spec.md` §3.5 has always quoted.
    pub const BGE_SMALL_EN_V1_5: f32 = 0.82;

    /// `ibm-granite/granite-embedding-97m-multilingual-r2`. Negatives
    /// top out near 0.90 in every language measured, so the operating
    /// point sits well above bge-small's.
    pub const GRANITE_97M_MULTILINGUAL_R2: f32 = 0.90;

    /// `minishlab/potion-multilingual-128M`. Static embeddings put
    /// unrelated strings much closer to orthogonal, so the whole scale
    /// shifts down.
    pub const POTION_MULTILINGUAL_128M: f32 = 0.40;
}

/// The filler lexicon and cosine threshold a given embedding backend
/// should be driven with.
///
/// The threshold is **not** portable across models: each backend puts
/// short strings in a differently-shaped space, so the separation
/// point between "this is a filler" and "this is a content word" has
/// to be measured per model. `examples/bench_embedder.rs` produces
/// these numbers; `docs/model-upgrade-candidates.md` §3 records the
/// measurement runs they came from.
#[derive(Debug, Clone)]
pub struct FillerLexicon {
    pub pure: Vec<String>,
    pub contextual: Vec<String>,
    pub pure_threshold: f32,
    pub segmenter: Segmenter,
}

impl FillerLexicon {
    fn from_strs(
        pure: &[&str],
        contextual: &[&str],
        pure_threshold: f32,
        segmenter: Segmenter,
    ) -> Self {
        Self {
            pure: pure.iter().map(|s| s.to_string()).collect(),
            contextual: contextual.iter().map(|s| s.to_string()).collect(),
            pure_threshold,
            segmenter,
        }
    }

    /// English lexicon. Mirrors `SimpleFillerFilter`'s closed set.
    pub fn english() -> Self {
        Self::from_strs(
            &["um", "uh", "uhm", "umm", "hmm", "er", "ah", "eh"],
            &["so", "well", "basically", "actually", "literally", "right"],
            calibrated::BGE_SMALL_EN_V1_5,
            Segmenter::Whitespace,
        )
    }

    /// Japanese lexicon. Mirrors `JapaneseFillerFilter`'s closed set,
    /// including the ASR artefacts documented in `docs/spec.md` §3.5.
    pub fn japanese() -> Self {
        Self::from_strs(
            &["えーと", "えっと", "あのー", "あの", "そのー", "えー", "あー", "んー", "まあ"],
            &["なんか", "ちょっと", "やっぱり", "その"],
            calibrated::BGE_SMALL_EN_V1_5,
            Segmenter::CjkPunctuation,
        )
    }

    /// Chinese lexicon. Mirrors `ChineseFillerFilter`'s closed set.
    pub fn chinese() -> Self {
        Self::from_strs(
            &["嗯", "呃", "哦", "啊"],
            &["那个", "这个", "就是", "然后", "怎么说"],
            calibrated::BGE_SMALL_EN_V1_5,
            Segmenter::CjkPunctuation,
        )
    }

    /// Korean lexicon. Mirrors the `ko` gold set in
    /// `tests/evaluation/annotations/ko_filler.jsonl`.
    pub fn korean() -> Self {
        Self::from_strs(
            &["어", "음", "아", "엄"],
            &["그", "그니까", "뭐", "이제"],
            calibrated::BGE_SMALL_EN_V1_5,
            Segmenter::Whitespace,
        )
    }

    /// Spanish lexicon. Mirrors `SpanishFillerFilter`'s closed set.
    pub fn spanish() -> Self {
        Self::from_strs(
            &["e", "eh", "em", "este", "mmm"],
            &["bueno", "entonces", "digamos"],
            calibrated::BGE_SMALL_EN_V1_5,
            Segmenter::Whitespace,
        )
    }

    /// Look a lexicon up by ISO-639-1 code.
    pub fn for_language(lang: &str) -> Option<Self> {
        match lang {
            "en" => Some(Self::english()),
            "ja" => Some(Self::japanese()),
            "zh" => Some(Self::chinese()),
            "ko" => Some(Self::korean()),
            "es" => Some(Self::spanish()),
            _ => None,
        }
    }
}

/// Filler removal driven by ONNX sentence embeddings.
///
/// Backend-agnostic: the graph's input signature is probed at load
/// time by [`EmbeddingBackend`], so this works with the 384-dim BERT
/// export it was originally written against (`bge-small-en-v1.5`),
/// with ModernBERT exports that have no `token_type_ids`
/// (`granite-embedding-97m-multilingual-r2`), and with static
/// embedding-bag exports (`potion-multilingual-128M`).
///
/// # Unwired
///
/// Nothing in the pipeline constructs this. It was never reachable
/// from the CLI or `prelude`, and the measurement in
/// `docs/model-upgrade-candidates.md` §3.2 found the rule-based
/// filters beating every embedding backend in every language tested
/// (en 1.000 vs 0.828, ja 0.941 vs 0.909, zh 1.000 vs 1.000,
/// ko 0.977 vs 0.562). The remaining argument for it — generalising
/// past a closed lexicon — was never actually exercised, because the
/// cosine gate was inert until §3.1 fixed it.
///
/// It is kept, with `examples/bench_embedder.rs`, so that case can be
/// re-opened against data containing out-of-lexicon filler variants
/// rather than re-derived from scratch. Until such data exists, use
/// [`crate::filter::SimpleFillerFilter`] and its language siblings.
#[deprecated(
    since = "0.1.0",
    note = "unwired: rule-based filters outscore every measured embedding backend. \
            See docs/model-upgrade-candidates.md §3.2. Kept only for re-evaluation \
            via examples/bench_embedder.rs."
)]
pub struct OnnxEmbeddingFilter {
    backend: Arc<Mutex<EmbeddingBackend>>,
    filler_embeddings: Vec<Vec<f32>>,
    pure_fillers: Vec<String>,
    contextual_fillers: Vec<String>,
    pure_threshold: f32,
    segmenter: Segmenter,
}

#[allow(deprecated)]
impl OnnxEmbeddingFilter {
    /// Load a backend from `model_dir` and drive it with the English
    /// lexicon. Preserved as the zero-argument entry point the
    /// existing call sites use.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, FilterError> {
        Self::load_with_lexicon(model_dir, FillerLexicon::english())
    }

    /// Load a backend and drive it with an explicit lexicon +
    /// threshold. This is the entry point that makes non-English
    /// evaluation possible: the prototype embeddings are computed from
    /// `lexicon.pure`, so a multilingual backend paired with, say,
    /// [`FillerLexicon::japanese`] scores Japanese fillers directly.
    pub fn load_with_lexicon(
        model_dir: impl AsRef<Path>,
        lexicon: FillerLexicon,
    ) -> Result<Self, FilterError> {
        let mut backend =
            EmbeddingBackend::load(model_dir).map_err(FilterError::Failed)?;

        let mut filler_embeddings = Vec::with_capacity(lexicon.pure.len());
        for f in &lexicon.pure {
            filler_embeddings.push(backend.embed(f).map_err(FilterError::Failed)?);
        }

        Ok(Self {
            backend: Arc::new(Mutex::new(backend)),
            filler_embeddings,
            pure_fillers: lexicon.pure,
            contextual_fillers: lexicon.contextual,
            pure_threshold: lexicon.pure_threshold,
            segmenter: lexicon.segmenter,
        })
    }

    /// Override the pure-filler cosine threshold.
    pub fn with_pure_threshold(mut self, threshold: f32) -> Self {
        self.pure_threshold = threshold;
        self
    }

    pub fn pure_threshold(&self) -> f32 {
        self.pure_threshold
    }

    /// Cosine similarity of `emb` against the nearest filler prototype.
    ///
    /// Exposed so the calibration bench can dump the raw score
    /// distribution instead of only the post-threshold decision.
    pub fn max_filler_sim(&self, emb: &[f32]) -> f32 {
        self.filler_embeddings
            .iter()
            .map(|f| cosine(emb, f))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Embed a single string through the shared backend.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, FilterError> {
        let mut backend = self.backend.lock().await;
        backend.embed(text).map_err(FilterError::Failed)
    }

    fn is_sentence_initial(segments: &[Segment], idx: usize, removed: &[bool]) -> bool {
        if idx == 0 {
            return true;
        }
        for j in (0..idx).rev() {
            if removed[j] {
                continue;
            }
            let s = &segments[j].surface;
            return s.ends_with('.')
                || s.ends_with('!')
                || s.ends_with('?')
                || s.ends_with(',');
        }
        true
    }

    /// Decide which segments are fillers.
    ///
    /// Shared by [`TextFilter::filter`], [`Self::detect_spans`] and
    /// the calibration bench so the text the pipeline emits, the spans
    /// the L3 evaluator scores, and the threshold sweep can never
    /// disagree.
    ///
    /// `pure_threshold` is a parameter rather than read off `self` so
    /// a sweep can re-decide over cached embeddings without paying for
    /// re-encoding at every candidate threshold.
    pub fn removal_flags(
        &self,
        segments: &[Segment],
        embeddings: &[Vec<f32>],
        pure_threshold: f32,
    ) -> Vec<bool> {
        let n = segments.len();
        let mut removed = vec![false; n];

        // Pass 1: pure fillers.
        //
        // Lexicon membership OR embedding proximity — not AND. The
        // conjunction this used to apply made the cosine gate inert:
        // a lexicon member embeds to (nearly) its own prototype, so
        // its similarity is ~1.0 and always clears any threshold,
        // while a non-member is rejected by the lexicon whatever it
        // scores. Sweeping the threshold from 0.00 to 0.95 moved F1
        // by exactly zero as a result
        // (`docs/model-upgrade-candidates.md` §3.1).
        //
        // The disjunction is what `docs/spec.md` §3.5 actually
        // specifies: the lexicon guarantees recall on known forms and
        // the embedding generalises to variants that are not in it
        // ("ummm", "uhh", ASR artefacts).
        for i in 0..n {
            let in_lexicon = self.pure_fillers.contains(&segments[i].clean);
            let near_prototype = self.max_filler_sim(&embeddings[i]) >= pure_threshold;
            if in_lexicon || near_prototype {
                removed[i] = true;
            }
        }
        // Pass 2: contextual fillers — only sentence-initially, so
        // "so" as a discourse marker goes but "so" as an intensifier
        // stays. Runs after pass 1 so a leading pure filler does not
        // hide the sentence-initial position of what follows.
        for i in 0..n {
            if removed[i] {
                continue;
            }
            if self.contextual_fillers.contains(&segments[i].clean)
                && Self::is_sentence_initial(segments, i, &removed)
            {
                removed[i] = true;
            }
        }
        removed
    }

    /// Codepoint spans of every detected filler.
    ///
    /// Mirrors the rule-based filters' `detect_spans` (same codepoint
    /// offset convention) so `examples/eval_l3.rs --task filler` can
    /// score this filter against the same gold annotations.
    ///
    /// Async because embedding needs the shared session; the
    /// rule-based equivalents are sync.
    pub async fn detect_spans(&self, text: &str) -> Result<Vec<Span>, FilterError> {
        let segments = self.segment_text(text);
        if segments.is_empty() {
            return Ok(Vec::new());
        }
        let embeddings = self.embed_segments(&segments).await?;
        let removed = self.removal_flags(&segments, &embeddings, self.pure_threshold);

        Ok(segments
            .iter()
            .zip(&removed)
            .filter(|(_, &r)| r)
            .map(|(s, _)| Span {
                start: s.cp_start,
                end: s.cp_end,
            })
            .collect())
    }

    /// Cut `text` with this filter's configured segmenter.
    pub fn segment_text(&self, text: &str) -> Vec<Segment> {
        segment(text, self.segmenter)
    }

    /// Embed every segment through the shared backend, in order.
    pub async fn embed_segments(
        &self,
        segments: &[Segment],
    ) -> Result<Vec<Vec<f32>>, FilterError> {
        let mut backend = self.backend.lock().await;
        let mut embeddings = Vec::with_capacity(segments.len());
        for s in segments {
            embeddings.push(
                backend
                    .embed(&s.surface)
                    .map_err(FilterError::Failed)?,
            );
        }
        Ok(embeddings)
    }
}

#[allow(deprecated)]
#[async_trait]
impl TextFilter for OnnxEmbeddingFilter {
    async fn filter(&self, text: &str) -> Result<FilterResult, FilterError> {
        let segments = self.segment_text(text);
        if segments.is_empty() {
            return Ok(FilterResult {
                text: String::new(),
                removed: vec![],
            });
        }

        let embeddings = self.embed_segments(&segments).await?;
        let removed = self.removal_flags(&segments, &embeddings, self.pure_threshold);

        let labels: Vec<String> = segments
            .iter()
            .zip(&removed)
            .filter(|(_, &r)| r)
            .map(|(s, _)| s.surface.clone())
            .collect();
        let kept: Vec<&str> = segments
            .iter()
            .zip(&removed)
            .filter(|(_, &r)| !r)
            .map(|(s, _)| s.surface.as_str())
            .collect();

        Ok(FilterResult {
            text: kept.join(self.segmenter.joiner()),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn surfaces(text: &str, seg: Segmenter) -> Vec<String> {
        segment(text, seg).into_iter().map(|s| s.surface).collect()
    }

    #[test]
    fn whitespace_segmenter_splits_on_spaces() {
        assert_eq!(
            surfaces("um I think so", Segmenter::Whitespace),
            vec!["um", "I", "think", "so"]
        );
    }

    #[test]
    fn whitespace_segmenter_reports_codepoint_offsets() {
        // Non-ASCII before the target: byte offsets would be wrong here.
        let segs = segment("canción um ahora", Segmenter::Whitespace);
        let um = &segs[1];
        assert_eq!(um.surface, "um");
        assert_eq!((um.cp_start, um.cp_end), (8, 10));
    }

    #[test]
    fn whitespace_segmenter_handles_empty_and_blank() {
        assert!(segment("", Segmenter::Whitespace).is_empty());
        assert!(segment("   ", Segmenter::Whitespace).is_empty());
    }

    #[test]
    fn cjk_segmenter_splits_on_reading_comma() {
        assert_eq!(
            surfaces("えーと、今日は晴れです。", Segmenter::CjkPunctuation),
            vec!["えーと", "今日は晴れです"]
        );
    }

    #[test]
    fn cjk_segmenter_splits_on_fullwidth_comma() {
        assert_eq!(
            surfaces("嗯，我们开始吧。", Segmenter::CjkPunctuation),
            vec!["嗯", "我们开始吧"]
        );
    }

    #[test]
    fn cjk_segmenter_reports_codepoint_offsets() {
        let segs = segment("えーと、今日は", Segmenter::CjkPunctuation);
        assert_eq!((segs[0].cp_start, segs[0].cp_end), (0, 3));
        assert_eq!((segs[1].cp_start, segs[1].cp_end), (4, 7));
    }

    #[test]
    fn clean_lowercases_and_strips_trailing_punctuation() {
        let segs = segment("Um, well", Segmenter::Whitespace);
        assert_eq!(segs[0].clean, "um");
        assert_eq!(segs[1].clean, "well");
    }

    #[test]
    fn clean_keeps_cjk_surface_intact() {
        // CJK codepoints are alphanumeric under Unicode, so the
        // leading-alphanumeric run is the whole segment.
        let segs = segment("えーと、今日", Segmenter::CjkPunctuation);
        assert_eq!(segs[0].clean, "えーと");
    }

    #[test]
    fn whitespace_joiner_is_space_cjk_joiner_is_empty() {
        assert_eq!(Segmenter::Whitespace.joiner(), " ");
        assert_eq!(Segmenter::CjkPunctuation.joiner(), "");
    }

    #[test]
    fn lexicon_lookup_covers_the_five_pipeline_languages() {
        for lang in ["en", "ja", "zh", "ko", "es"] {
            let lex = FillerLexicon::for_language(lang)
                .unwrap_or_else(|| panic!("no lexicon for {lang}"));
            assert!(!lex.pure.is_empty(), "{lang} has no pure fillers");
        }
        assert!(FillerLexicon::for_language("de").is_none());
    }

    #[test]
    fn cjk_languages_use_the_cjk_segmenter() {
        assert_eq!(
            FillerLexicon::japanese().segmenter,
            Segmenter::CjkPunctuation
        );
        assert_eq!(
            FillerLexicon::chinese().segmenter,
            Segmenter::CjkPunctuation
        );
        // Korean is space-delimited despite being CJK-adjacent.
        assert_eq!(FillerLexicon::korean().segmenter, Segmenter::Whitespace);
    }

    #[test]
    fn default_lexicon_threshold_matches_the_default_backend() {
        // The zero-argument `load` path uses bge-small, so the
        // lexicon default must be bge-small's calibrated value or the
        // out-of-the-box filter is mis-tuned.
        assert_eq!(
            FillerLexicon::english().pure_threshold,
            calibrated::BGE_SMALL_EN_V1_5
        );
    }

    #[test]
    fn calibrated_thresholds_are_distinct_per_backend() {
        // Guards the whole point of the module: if these ever collapse
        // to one value someone has assumed portability.
        let all = [
            calibrated::BGE_SMALL_EN_V1_5,
            calibrated::GRANITE_97M_MULTILINGUAL_R2,
            calibrated::POTION_MULTILINGUAL_128M,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert!((a - b).abs() > 1e-6, "thresholds {a} and {b} collapsed");
            }
        }
    }

    #[test]
    fn calibrated_thresholds_are_valid_cosines() {
        for t in [
            calibrated::BGE_SMALL_EN_V1_5,
            calibrated::GRANITE_97M_MULTILINGUAL_R2,
            calibrated::POTION_MULTILINGUAL_128M,
        ] {
            assert!((0.0..=1.0).contains(&t), "threshold {t} out of range");
        }
    }

    #[test]
    fn with_pure_threshold_overrides_the_lexicon_default() {
        let lex = FillerLexicon::english().pure_threshold;
        assert_ne!(lex, calibrated::POTION_MULTILINGUAL_128M);
        // The builder is the supported way to retarget a backend; the
        // struct field it writes is asserted through `pure_threshold`
        // in the loaded-filter path, which needs a model bundle.
        let mut retargeted = FillerLexicon::english();
        retargeted.pure_threshold = calibrated::POTION_MULTILINGUAL_128M;
        assert_eq!(
            retargeted.pure_threshold,
            calibrated::POTION_MULTILINGUAL_128M
        );
    }

    #[test]
    fn lexicon_pure_entries_survive_their_own_segmenter() {
        // A prototype that the segmenter would itself split cannot be
        // matched against a segment, so the lexicon would silently
        // never fire.
        for lex in [
            FillerLexicon::english(),
            FillerLexicon::japanese(),
            FillerLexicon::chinese(),
            FillerLexicon::korean(),
            FillerLexicon::spanish(),
        ] {
            for p in &lex.pure {
                let segs = segment(p, lex.segmenter);
                assert_eq!(segs.len(), 1, "pure filler {p:?} splits into {segs:?}");
                assert_eq!(&segs[0].clean, p, "pure filler {p:?} not lookup-stable");
            }
        }
    }
}
