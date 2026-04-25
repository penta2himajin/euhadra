//! Span-based F1 for L3 direct evaluation (Phase C-1).
//!
//! Two flavours:
//!
//! - **Strict span match**: a predicted span counts as a true positive
//!   only when its `(start, end)` exactly matches a gold span. Used for
//!   filler removal where the closed-class lexicon makes boundaries
//!   unambiguous.
//! - **Partial overlap match (IoU ≥ θ)**: a predicted span counts as a
//!   true positive when its character-level IoU with a gold span meets
//!   `iou_threshold`. Used for self-correction reparandum/repair spans
//!   where boundaries are inherently fuzzy (Switchboard-NXT-style
//!   inter-annotator κ 0.55–0.75).
//!
//! Both flavours produce the same `F1Stats` shape so callers can render
//! a single table.

use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Character-level intersection-over-union with another span.
    /// Returns 0.0 when either span is empty.
    pub fn iou(&self, other: &Span) -> f64 {
        if self.is_empty() || other.is_empty() {
            return 0.0;
        }
        let inter_start = self.start.max(other.start);
        let inter_end = self.end.min(other.end);
        if inter_start >= inter_end {
            return 0.0;
        }
        let inter = (inter_end - inter_start) as f64;
        let union = (self.len() + other.len()) as f64 - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F1Stats {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
}

impl F1Stats {
    pub fn from_counts(tp: usize, fp: usize, fn_: usize) -> Self {
        let precision = if tp + fp == 0 {
            f64::NAN
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_ == 0 {
            f64::NAN
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        let f1 = if precision.is_nan()
            || recall.is_nan()
            || (precision + recall) <= 0.0
        {
            f64::NAN
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        Self {
            precision,
            recall,
            f1,
            tp,
            fp,
            fn_,
        }
    }
}

/// Strict-match F1 over span sets. Each predicted span is counted as
/// TP iff there exists an exact `(start, end)` match in `gold`. Each
/// gold span unmatched by any predicted span is FN.
pub fn strict_f1(predicted: &[Span], gold: &[Span]) -> F1Stats {
    let gold_set: HashSet<(usize, usize)> =
        gold.iter().map(|s| (s.start, s.end)).collect();
    let pred_set: HashSet<(usize, usize)> =
        predicted.iter().map(|s| (s.start, s.end)).collect();
    let tp = pred_set.intersection(&gold_set).count();
    let fp = pred_set.len() - tp;
    let fn_ = gold_set.len() - tp;
    F1Stats::from_counts(tp, fp, fn_)
}

/// Partial-overlap F1: a predicted span counts as TP if it has IoU ≥
/// `iou_threshold` with at least one gold span; that gold span is then
/// "consumed" (greedy 1-to-1 matching, so two predictions can't both
/// claim the same gold span).
pub fn iou_f1(
    predicted: &[Span],
    gold: &[Span],
    iou_threshold: f64,
) -> F1Stats {
    let mut consumed = vec![false; gold.len()];
    let mut tp = 0;
    let mut fp = 0;
    for p in predicted {
        let mut best_idx = None;
        let mut best_iou = 0.0;
        for (i, g) in gold.iter().enumerate() {
            if consumed[i] {
                continue;
            }
            let iou = p.iou(g);
            if iou >= iou_threshold && iou > best_iou {
                best_iou = iou;
                best_idx = Some(i);
            }
        }
        if let Some(i) = best_idx {
            consumed[i] = true;
            tp += 1;
        } else {
            fp += 1;
        }
    }
    let fn_ = consumed.iter().filter(|c| !**c).count();
    F1Stats::from_counts(tp, fp, fn_)
}

/// Aggregate per-utterance counts into a single F1. Use this when you
/// have many utterances and want corpus-level F1 (micro-averaged).
pub fn aggregate(stats: &[F1Stats]) -> F1Stats {
    let tp = stats.iter().map(|s| s.tp).sum();
    let fp = stats.iter().map(|s| s.fp).sum();
    let fn_ = stats.iter().map(|s| s.fn_).sum();
    F1Stats::from_counts(tp, fp, fn_)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(a: usize, b: usize) -> Span {
        Span { start: a, end: b }
    }

    #[test]
    fn span_iou_basic() {
        assert!((s(0, 10).iou(&s(0, 10)) - 1.0).abs() < 1e-9);
        assert!((s(0, 10).iou(&s(5, 15)) - (5.0 / 15.0)).abs() < 1e-9);
        assert_eq!(s(0, 5).iou(&s(10, 15)), 0.0);
        assert_eq!(s(0, 0).iou(&s(0, 5)), 0.0);
    }

    #[test]
    fn strict_f1_perfect() {
        let p = vec![s(0, 2), s(5, 8)];
        let g = vec![s(0, 2), s(5, 8)];
        let r = strict_f1(&p, &g);
        assert_eq!(r.tp, 2);
        assert_eq!(r.fp, 0);
        assert_eq!(r.fn_, 0);
        assert!((r.f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn strict_f1_one_miss_one_extra() {
        // gold: [0..2, 5..8]; pred: [0..2, 9..10] → 1 TP, 1 FP, 1 FN
        let p = vec![s(0, 2), s(9, 10)];
        let g = vec![s(0, 2), s(5, 8)];
        let r = strict_f1(&p, &g);
        assert_eq!(r.tp, 1);
        assert_eq!(r.fp, 1);
        assert_eq!(r.fn_, 1);
        assert!((r.precision - 0.5).abs() < 1e-9);
        assert!((r.recall - 0.5).abs() < 1e-9);
        assert!((r.f1 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn iou_f1_partial_match() {
        // Pred slightly off but >= 0.5 IoU with gold → TP
        let p = vec![s(0, 4)];
        let g = vec![s(1, 5)]; // overlap 1..4 = 3, union 0..5 = 5, IoU = 0.6
        let r = iou_f1(&p, &g, 0.5);
        assert_eq!(r.tp, 1);
        assert_eq!(r.fp, 0);
        assert_eq!(r.fn_, 0);
    }

    #[test]
    fn iou_f1_below_threshold_is_miss() {
        let p = vec![s(0, 5)];
        let g = vec![s(4, 10)]; // overlap 1, union 10, IoU = 0.1
        let r = iou_f1(&p, &g, 0.5);
        assert_eq!(r.tp, 0);
        assert_eq!(r.fp, 1);
        assert_eq!(r.fn_, 1);
    }

    #[test]
    fn iou_f1_greedy_one_to_one() {
        // Two preds both overlap the same gold span; only the better
        // match should claim it.
        let p = vec![s(0, 4), s(0, 5)];
        let g = vec![s(0, 5)];
        let r = iou_f1(&p, &g, 0.5);
        assert_eq!(r.tp, 1);
        assert_eq!(r.fp, 1);
        assert_eq!(r.fn_, 0);
    }

    #[test]
    fn empty_gold_yields_nan_recall() {
        let r = strict_f1(&[s(0, 1)], &[]);
        assert_eq!(r.tp, 0);
        assert_eq!(r.fp, 1);
        assert_eq!(r.fn_, 0);
        assert!(r.recall.is_nan());
    }

    #[test]
    fn aggregate_micro_average() {
        let s1 = F1Stats::from_counts(2, 1, 1);
        let s2 = F1Stats::from_counts(3, 0, 2);
        let agg = aggregate(&[s1, s2]);
        assert_eq!(agg.tp, 5);
        assert_eq!(agg.fp, 1);
        assert_eq!(agg.fn_, 3);
    }
}
