//! Shared ONNX sentence-embedding backend (feature-gated behind `onnx`).
//!
//! Tier 1 (`OnnxEmbeddingFilter`) and Tier 2 (`OnnxTextEmbedder` for
//! `PhonemeCorrector`, `ParagraphSplitter`) both need "encode a short
//! string into an L2-normalised vector". They used to carry a private
//! copy of that code, and both copies hard-coded the input signature
//! of `BAAI/bge-small-en-v1.5` (`input_ids` + `attention_mask` +
//! `token_type_ids`, 3-D `last_hidden_state` output, CLS pooling).
//!
//! That assumption is not portable. Comparing embedding backends —
//! see `docs/model-upgrade-candidates.md` — turned up three distinct
//! graph shapes among otherwise interchangeable models:
//!
//! | Model | Inputs | Output |
//! |---|---|---|
//! | `bge-small-en-v1.5` (BERT) | `input_ids`, `attention_mask`, `token_type_ids` | `[b, seq, 384]` |
//! | `granite-embedding-97m-multilingual-r2` (ModernBERT) | `input_ids`, `attention_mask` | `[b, seq, 384]` |
//! | `potion-multilingual-128M` (Model2Vec static) | flat `input_ids`, `offsets` | `[b, 256]` |
//!
//! Feeding `token_type_ids` to the second shape is a hard load error,
//! and the third is not a transformer at all — it is an embedding-bag
//! lookup whose output is already pooled. So the signature is probed
//! from the loaded graph rather than assumed.

use std::path::Path;

use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;

/// How a loaded embedding graph wants to be fed, and what it returns.
///
/// Probed from the session's declared input names at load time by
/// [`EmbeddingSignature::from_input_names`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingSignature {
    /// Classic BERT: `input_ids` + `attention_mask` + `token_type_ids`,
    /// batch-shaped `[1, seq]`. Output `[1, seq, dim]`, CLS-pooled.
    BertLike,
    /// ModernBERT / RoBERTa-style: no segment ids. `input_ids` +
    /// `attention_mask`, `[1, seq]`. Output `[1, seq, dim]`, CLS-pooled.
    MaskOnly,
    /// Model2Vec static embeddings: a flat `input_ids` vector plus an
    /// `offsets` vector marking sequence starts (`nn.EmbeddingBag`
    /// convention). Output `[1, dim]` — already pooled, no CLS token.
    StaticBag,
}

impl EmbeddingSignature {
    /// Derive the signature from a session's input names.
    ///
    /// `offsets` is the discriminator for the static-bag shape because
    /// no transformer export declares it; `token_type_ids` then
    /// separates BERT from ModernBERT.
    pub fn from_input_names<S: AsRef<str>>(names: &[S]) -> Result<Self, String> {
        let has = |needle: &str| names.iter().any(|n| n.as_ref() == needle);

        if !has("input_ids") {
            return Err(format!(
                "embedding graph declares no `input_ids` input (got {:?})",
                names.iter().map(|n| n.as_ref()).collect::<Vec<_>>()
            ));
        }
        if has("offsets") {
            Ok(Self::StaticBag)
        } else if has("token_type_ids") {
            Ok(Self::BertLike)
        } else if has("attention_mask") {
            Ok(Self::MaskOnly)
        } else {
            Err(format!(
                "embedding graph has `input_ids` but neither `attention_mask` \
                 nor `offsets` (got {:?})",
                names.iter().map(|n| n.as_ref()).collect::<Vec<_>>()
            ))
        }
    }

    /// Whether the graph's output still carries a sequence axis and
    /// therefore needs pooling on our side.
    pub fn needs_pooling(&self) -> bool {
        !matches!(self, Self::StaticBag)
    }
}

/// L2-normalise in place, leaving an all-zero vector untouched.
pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity of two L2-normalised vectors (a plain dot product).
///
/// Returns 0.0 on a length mismatch rather than panicking: embedding
/// dimensions differ across backends (384 for bge/granite, 256 for
/// potion) and a mismatch means the caller mixed two models, which is
/// a configuration error we would rather surface as "no similarity"
/// than as a crash inside a filter.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Mutually unrelated strings used to measure a backend's similarity
/// floor. Spread across the pipeline's five languages and several
/// semantic fields so no two are plausibly related — the mean cosine
/// among them is then a reading of where that backend puts "nothing in
/// common", which is emphatically *not* zero.
const FLOOR_PROBES: &[&str] = &[
    "database migration",
    "strawberry jam",
    "オーケストラの練習",
    "台風が接近している",
    "감자를 삶았다",
    "el tipo de cambio",
    "quarterly revenue forecast",
    "他昨天去了医院",
    "shoelace",
    "volcanic ash cloud",
];

/// A loaded sentence-embedding model: session + tokenizer + the
/// signature probed from the graph.
pub struct EmbeddingBackend {
    session: Session,
    tokenizer: Tokenizer,
    signature: EmbeddingSignature,
    /// Lazily measured; see [`EmbeddingBackend::similarity_floor`].
    similarity_floor: Option<f32>,
}

impl EmbeddingBackend {
    /// Load `model.onnx` + `tokenizer.json` from `dir`.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(dir.join("model.onnx")))
            .map_err(|e| format!("load model: {e}"))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| format!("load tokenizer: {e}"))?;

        let names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let signature = EmbeddingSignature::from_input_names(&names)?;

        tracing::info!(
            ?signature,
            inputs = ?names,
            "ONNX embedding backend loaded"
        );
        Ok(Self {
            session,
            tokenizer,
            signature,
            similarity_floor: None,
        })
    }

    pub fn signature(&self) -> EmbeddingSignature {
        self.signature
    }

    /// Mean cosine this backend assigns to unrelated text.
    ///
    /// Cosine is not comparable across embedding models: the same
    /// "these are unrelated" verdict reads as ~0.45 on
    /// `bge-small-en-v1.5` and ~0.70 on
    /// `granite-embedding-97m-multilingual-r2`. Any consumer that
    /// blends a cosine with a differently-scaled quantity — as
    /// `PhonemeCorrector` blends it with a normalised edit distance —
    /// is therefore weighting the two terms differently on every
    /// backend, without saying so.
    ///
    /// Measuring the floor lets [`Self::rescale_similarity`] put every
    /// backend on the same scale, so the weight means one thing.
    ///
    /// Computed once, on first use, from [`FLOOR_PROBES`] — ten embeds,
    /// paid only by callers that rescale. Falls back to 0.0 (no
    /// rescaling) if the probes cannot be embedded, which keeps a
    /// degraded backend behaving as it did before rather than
    /// distorting scores with a bad floor.
    pub fn similarity_floor(&mut self) -> f32 {
        if let Some(f) = self.similarity_floor {
            return f;
        }

        let mut vectors = Vec::with_capacity(FLOOR_PROBES.len());
        for p in FLOOR_PROBES {
            match self.embed(p) {
                Ok(v) => vectors.push(v),
                Err(e) => {
                    tracing::warn!(probe = %p, error = %e, "floor probe failed");
                    self.similarity_floor = Some(0.0);
                    return 0.0;
                }
            }
        }

        let mut total = 0.0f32;
        let mut pairs = 0usize;
        for i in 0..vectors.len() {
            for j in (i + 1)..vectors.len() {
                total += cosine(&vectors[i], &vectors[j]);
                pairs += 1;
            }
        }
        let floor = if pairs == 0 {
            0.0
        } else {
            (total / pairs as f32).clamp(0.0, 0.99)
        };

        tracing::info!(floor, "measured embedding similarity floor");
        self.similarity_floor = Some(floor);
        floor
    }

    /// Map a raw cosine onto `[0, 1]` against this backend's own floor,
    /// so that "unrelated" reads as 0 and "identical" as 1 whichever
    /// model produced it.
    pub fn rescale_similarity(&mut self, cos: f32) -> f32 {
        crate::similarity::rescale(cos, self.similarity_floor())
    }

    /// Encode `text` into an L2-normalised embedding.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, String> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        if ids.is_empty() {
            return Err(format!("tokenizer produced no ids for {text:?}"));
        }
        let len = ids.len();

        let outputs = match self.signature {
            EmbeddingSignature::StaticBag => {
                // Flat token vector + a single offset marking the start
                // of our one and only sequence.
                let ids_arr = Array1::from_vec(ids);
                let offsets = Array1::from_vec(vec![0i64]);
                let ids_val =
                    Value::from_array(ids_arr).map_err(|e| format!("input_ids: {e}"))?;
                let off_val =
                    Value::from_array(offsets).map_err(|e| format!("offsets: {e}"))?;
                self.session
                    .run(vec![
                        ("input_ids", ids_val.into_dyn()),
                        ("offsets", off_val.into_dyn()),
                    ])
                    .map_err(|e| format!("run: {e}"))?
            }
            sig => {
                let ids_arr = Array2::from_shape_vec((1, len), ids)
                    .map_err(|e| format!("input_ids shape: {e}"))?;
                let mask = Array2::from_shape_vec(
                    (1, len),
                    enc.get_attention_mask().iter().map(|&x| x as i64).collect(),
                )
                .map_err(|e| format!("attention_mask shape: {e}"))?;
                let ids_val =
                    Value::from_array(ids_arr).map_err(|e| format!("input_ids: {e}"))?;
                let mask_val =
                    Value::from_array(mask).map_err(|e| format!("attention_mask: {e}"))?;

                let mut feed = vec![
                    ("input_ids", ids_val.into_dyn()),
                    ("attention_mask", mask_val.into_dyn()),
                ];
                if sig == EmbeddingSignature::BertLike {
                    let tids = Array2::from_shape_vec(
                        (1, len),
                        enc.get_type_ids().iter().map(|&x| x as i64).collect(),
                    )
                    .map_err(|e| format!("token_type_ids shape: {e}"))?;
                    let tids_val =
                        Value::from_array(tids).map_err(|e| format!("token_type_ids: {e}"))?;
                    feed.push(("token_type_ids", tids_val.into_dyn()));
                }
                self.session.run(feed).map_err(|e| format!("run: {e}"))?
            }
        };

        let arr = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("extract: {e}"))?;
        let view = arr.view();
        let shape = view.shape();

        let mut pooled: Vec<f32> = match shape.len() {
            // [1, seq, dim] → CLS token.
            3 => (0..shape[2]).map(|i| view[[0, 0, i]]).collect(),
            // [1, dim] → already pooled by the graph.
            2 => (0..shape[1]).map(|i| view[[0, i]]).collect(),
            other => {
                return Err(format!(
                    "unexpected embedding output rank {other} (shape {shape:?})"
                ))
            }
        };
        l2_normalize(&mut pooled);
        Ok(pooled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_bert_like_when_token_type_ids_present() {
        let names = ["input_ids", "attention_mask", "token_type_ids"];
        assert_eq!(
            EmbeddingSignature::from_input_names(&names).unwrap(),
            EmbeddingSignature::BertLike
        );
    }

    #[test]
    fn signature_mask_only_when_no_token_type_ids() {
        let names = ["input_ids", "attention_mask"];
        assert_eq!(
            EmbeddingSignature::from_input_names(&names).unwrap(),
            EmbeddingSignature::MaskOnly
        );
    }

    #[test]
    fn signature_static_bag_when_offsets_present() {
        let names = ["input_ids", "offsets"];
        assert_eq!(
            EmbeddingSignature::from_input_names(&names).unwrap(),
            EmbeddingSignature::StaticBag
        );
    }

    #[test]
    fn signature_offsets_wins_over_token_type_ids() {
        // Defensive: a static-bag export that also happens to declare
        // segment ids must still be fed the flat/offsets way.
        let names = ["input_ids", "token_type_ids", "offsets"];
        assert_eq!(
            EmbeddingSignature::from_input_names(&names).unwrap(),
            EmbeddingSignature::StaticBag
        );
    }

    #[test]
    fn signature_rejects_graph_without_input_ids() {
        let names = ["pixel_values"];
        let err = EmbeddingSignature::from_input_names(&names).unwrap_err();
        assert!(err.contains("input_ids"), "unexpected error: {err}");
    }

    #[test]
    fn signature_rejects_input_ids_only_graph() {
        let names = ["input_ids"];
        assert!(EmbeddingSignature::from_input_names(&names).is_err());
    }

    #[test]
    fn only_static_bag_skips_pooling() {
        assert!(EmbeddingSignature::BertLike.needs_pooling());
        assert!(EmbeddingSignature::MaskOnly.needs_pooling());
        assert!(!EmbeddingSignature::StaticBag.needs_pooling());
    }

    #[test]
    fn l2_normalize_makes_unit_length() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_leaves_zero_vector_alone() {
        let mut v = vec![0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn floor_probes_are_mutually_distinct() {
        // A duplicated probe would drag the measured floor toward 1.0
        // and flatten the rescaled range for everyone.
        for (i, a) in FLOOR_PROBES.iter().enumerate() {
            for b in &FLOOR_PROBES[i + 1..] {
                assert_ne!(a, b, "duplicate floor probe");
            }
        }
        assert!(FLOOR_PROBES.len() >= 4, "too few probes to average over");
    }

    #[test]
    fn cosine_of_identical_unit_vectors_is_one() {
        let a = vec![0.6, 0.8];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_mismatched_dimensions_is_zero_not_panic() {
        // 384-dim (bge/granite) vs 256-dim (potion): a config error,
        // surfaced as "no similarity" rather than a crash.
        assert_eq!(cosine(&[1.0, 0.0, 0.0], &[1.0, 0.0]), 0.0);
    }
}
