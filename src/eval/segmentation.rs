//! Text-segmentation metrics for `ParagraphSplitter` evaluation.
//!
//! Span F1 is the wrong instrument here. A paragraph break placed one
//! sentence away from the gold one is nearly right, but exact-match F1
//! scores it as both a false positive and a false negative — doubly
//! wrong, and indistinguishable from a break placed at random. The
//! segmentation literature uses two window-based penalties instead:
//!
//! - **Pk** slides a window of width `k` (conventionally half the mean
//!   reference segment length) and counts how often the two ends
//!   disagree about lying in the same segment.
//! - **WindowDiff** slides the same window but compares the *number* of
//!   boundaries inside it, which fixes Pk's insensitivity to false
//!   positives and to near-misses.
//!
//! Both are penalties: **lower is better**, 0.0 is perfect.
//!
//! A segmentation is represented as the sentence indices at which a new
//! segment starts, excluding 0. For a 10-sentence document split after
//! sentences 3 and 7, that is `[4, 8]`.

/// Convert break indices into a per-position boundary mask of length
/// `n_sentences - 1`, where `mask[i]` is true when a boundary falls
/// between sentence `i` and `i + 1`.
fn boundary_mask(n_sentences: usize, breaks: &[usize]) -> Vec<bool> {
    if n_sentences == 0 {
        return Vec::new();
    }
    let mut mask = vec![false; n_sentences - 1];
    for &b in breaks {
        // A break "before sentence b" is a boundary at gap b-1.
        if b >= 1 && b <= mask.len() {
            mask[b - 1] = true;
        }
    }
    mask
}

/// Conventional window width: half the mean reference segment length,
/// clamped to at least 2 so the window spans a gap at all.
pub fn default_window(n_sentences: usize, reference: &[usize]) -> usize {
    if n_sentences == 0 {
        return 2;
    }
    let segments = reference.len() + 1;
    let mean_len = n_sentences as f64 / segments as f64;
    ((mean_len / 2.0).round() as usize).max(2)
}

/// Pk: probability that a randomly placed window of width `k`
/// disagrees with the reference about whether its endpoints are in the
/// same segment.
pub fn pk(n_sentences: usize, reference: &[usize], hypothesis: &[usize], k: usize) -> f64 {
    let r = boundary_mask(n_sentences, reference);
    let h = boundary_mask(n_sentences, hypothesis);
    if r.len() < k || k == 0 {
        return f64::NAN;
    }

    let mut disagreements = 0usize;
    let mut windows = 0usize;
    for start in 0..=(r.len() - k) {
        let r_same = !r[start..start + k].iter().any(|b| *b);
        let h_same = !h[start..start + k].iter().any(|b| *b);
        if r_same != h_same {
            disagreements += 1;
        }
        windows += 1;
    }
    if windows == 0 {
        f64::NAN
    } else {
        disagreements as f64 / windows as f64
    }
}

/// WindowDiff: fraction of windows where the hypothesis and reference
/// disagree on the *number* of boundaries.
pub fn window_diff(
    n_sentences: usize,
    reference: &[usize],
    hypothesis: &[usize],
    k: usize,
) -> f64 {
    let r = boundary_mask(n_sentences, reference);
    let h = boundary_mask(n_sentences, hypothesis);
    if r.len() < k || k == 0 {
        return f64::NAN;
    }

    let mut disagreements = 0usize;
    let mut windows = 0usize;
    for start in 0..=(r.len() - k) {
        let rc = r[start..start + k].iter().filter(|b| **b).count();
        let hc = h[start..start + k].iter().filter(|b| **b).count();
        if rc != hc {
            disagreements += 1;
        }
        windows += 1;
    }
    if windows == 0 {
        f64::NAN
    } else {
        disagreements as f64 / windows as f64
    }
}

/// Exact-match boundary counts, reported alongside the window metrics
/// because they are easier to reason about when diagnosing a segmenter
/// that fires too much or not at all.
pub fn boundary_counts(reference: &[usize], hypothesis: &[usize]) -> (usize, usize, usize) {
    let tp = hypothesis.iter().filter(|b| reference.contains(b)).count();
    let fp = hypothesis.len() - tp;
    let fn_ = reference.len() - tp;
    (tp, fp, fn_)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_segmentation_scores_zero() {
        let reference = vec![4, 8];
        assert_eq!(pk(12, &reference, &reference, 3), 0.0);
        assert_eq!(window_diff(12, &reference, &reference, 3), 0.0);
    }

    #[test]
    fn a_near_miss_is_penalised_less_than_a_wild_guess() {
        // The property that motivates using these metrics at all:
        // exact-match F1 gives both of these the same score of 0.
        let reference = vec![6];
        let near = window_diff(12, &reference, &[7], 3);
        let far = window_diff(12, &reference, &[1], 3);
        assert!(near < far, "near {near} should beat far {far}");
    }

    #[test]
    fn missing_every_boundary_is_penalised() {
        let reference = vec![4, 8];
        assert!(window_diff(12, &reference, &[], 3) > 0.0);
        assert!(pk(12, &reference, &[], 3) > 0.0);
    }

    #[test]
    fn window_diff_penalises_spurious_boundaries_pk_can_miss() {
        // Pk's blind spot: extra boundaries inside an already-split
        // window cost it nothing, while WindowDiff counts them.
        let reference = vec![6];
        let over = vec![5, 6, 7];
        assert!(
            window_diff(12, &reference, &over, 3) > 0.0,
            "WindowDiff should notice the extra boundaries"
        );
    }

    #[test]
    fn boundary_mask_maps_breaks_to_gaps() {
        // Break before sentence 4 is the gap between sentences 3 and 4,
        // i.e. index 3 in a 0-based gap array.
        let mask = boundary_mask(6, &[4]);
        assert_eq!(mask, vec![false, false, false, true, false]);
    }

    #[test]
    fn boundary_mask_ignores_out_of_range_breaks() {
        let mask = boundary_mask(4, &[0, 4, 99]);
        assert_eq!(mask, vec![false, false, false]);
    }

    #[test]
    fn default_window_is_half_the_mean_segment() {
        // 12 sentences, 3 segments → mean 4 → window 2.
        assert_eq!(default_window(12, &[4, 8]), 2);
        // 30 sentences, 3 segments → mean 10 → window 5.
        assert_eq!(default_window(30, &[10, 20]), 5);
    }

    #[test]
    fn default_window_never_degenerates_to_zero() {
        assert!(default_window(2, &[1]) >= 2);
        assert!(default_window(0, &[]) >= 2);
    }

    #[test]
    fn metrics_are_nan_when_the_window_does_not_fit() {
        assert!(pk(3, &[1], &[1], 10).is_nan());
        assert!(window_diff(3, &[1], &[1], 10).is_nan());
    }

    #[test]
    fn boundary_counts_are_exact_match() {
        let (tp, fp, fn_) = boundary_counts(&[4, 8], &[4, 9]);
        assert_eq!((tp, fp, fn_), (1, 1, 1));
    }

    #[test]
    fn boundary_counts_with_no_hypothesis() {
        assert_eq!(boundary_counts(&[4, 8], &[]), (0, 0, 2));
    }

    #[test]
    fn splitting_everywhere_is_worse_than_splitting_nowhere_here() {
        // Sanity on the baselines the evaluation compares against: with
        // sparse gold boundaries, over-splitting must not look good.
        let reference = vec![6];
        let none = window_diff(12, &reference, &[], 3);
        let all: Vec<usize> = (1..12).collect();
        assert!(
            window_diff(12, &reference, &all, 3) > none,
            "over-splitting should score worse than not splitting"
        );
    }
}
