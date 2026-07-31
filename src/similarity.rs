//! Putting cosine similarities from different embedding models on a
//! common scale.
//!
//! A cosine's *ordering* within one model is meaningful; its *absolute
//! value* is not comparable across models. Measured on the pipeline's
//! own probes, "these two strings have nothing in common" reads as
//! ≈0.45 on `bge-small-en-v1.5` and ≈0.70 on
//! `granite-embedding-97m-multilingual-r2`.
//!
//! That matters wherever a cosine meets a number that is *not* a
//! cosine. `PhonemeCorrector` scores candidates as
//! `alpha * phoneme_sim + (1-alpha) * text_sim`, where `phoneme_sim` is
//! a normalised edit distance — genuinely on `[0, 1]` with 0 meaning
//! "nothing alike". Blending it with a raw cosine whose own zero point
//! sits at 0.70 means `alpha` weights the two terms differently on
//! every backend, which is why the `alpha = 0.7` in `docs/spec.md` §6.4
//! holds on one model and drops corrections on another.
//!
//! Rescaling against the backend's measured floor removes that: the
//! weight then means one thing everywhere.
//!
//! This module is deliberately outside the `onnx` feature gate — the
//! arithmetic is needed by `phoneme.rs`, which builds in the default
//! configuration.

/// Map `cos` from `[floor, 1]` onto `[0, 1]`, clamping outside it.
///
/// A `floor` of 0.0 makes this the identity on `[0, 1]`, which is the
/// behaviour for any embedder that has not measured one.
pub fn rescale(cos: f32, floor: f32) -> f32 {
    // A floor at or above 1.0 leaves no range to map onto, and NaN is
    // incomparable rather than merely large — both fall through to the
    // unscaled value rather than dividing by zero or worse.
    if matches!(floor.partial_cmp(&1.0), Some(std::cmp::Ordering::Less)) {
        ((cos - floor) / (1.0 - floor)).clamp(0.0, 1.0)
    } else {
        cos.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_the_floor_to_zero_and_one_to_one() {
        assert!((rescale(0.70, 0.70) - 0.0).abs() < 1e-6);
        assert!((rescale(1.00, 0.70) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn puts_two_backends_on_the_same_scale() {
        // The whole point: 0.45 and 0.70 describe the same "unrelated"
        // verdict, so a pair sitting the same fraction of the way up
        // each backend's range must come out equal.
        let bge = rescale(0.725, 0.45); // halfway between 0.45 and 1.0
        let granite = rescale(0.850, 0.70); // halfway between 0.70 and 1.0
        assert!((bge - granite).abs() < 1e-5, "{bge} vs {granite}");
        assert!((bge - 0.5).abs() < 1e-5);
    }

    #[test]
    fn clamps_below_the_floor() {
        // "Even less related than unrelated" is still just 0.
        assert_eq!(rescale(0.10, 0.70), 0.0);
        assert_eq!(rescale(-0.50, 0.70), 0.0);
    }

    #[test]
    fn a_zero_floor_is_the_identity() {
        // Embedders without a measured floor must behave exactly as
        // they did before rescaling existed.
        for c in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            assert!((rescale(c, 0.0) - c).abs() < 1e-6, "{c}");
        }
    }

    #[test]
    fn a_degenerate_floor_does_not_divide_by_zero() {
        assert!(rescale(0.8, 1.0).is_finite());
        assert!(rescale(0.8, 1.5).is_finite());
        assert!(rescale(0.8, f32::NAN).is_finite());
    }

    #[test]
    fn rescaling_preserves_ordering() {
        // It is a positive affine map, so it must not reorder
        // candidates — only restate them on a common scale.
        let raw = [0.60f32, 0.72, 0.85, 0.99];
        let scaled: Vec<f32> = raw.iter().map(|c| rescale(*c, 0.55)).collect();
        for w in scaled.windows(2) {
            assert!(w[0] < w[1], "ordering changed: {scaled:?}");
        }
    }
}
