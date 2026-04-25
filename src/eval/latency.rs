//! Latency sample collection with p50 / p95 reporting.
//!
//! Designed for noisy CI runners: we collect raw samples and report
//! median + p95 rather than mean, so a single outlier doesn't dominate.

use std::time::Duration;

/// Collected latency samples for one measurement point (e.g. "ASR stage" or
/// "E2E pipeline").
#[derive(Debug, Default, Clone)]
pub struct Samples {
    samples: Vec<Duration>,
}

impl Samples {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, d: Duration) {
        self.samples.push(d);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Compute p50 and p95 in milliseconds. Returns `None` if no samples
    /// have been recorded.
    pub fn summary(&self) -> Option<LatencySummary> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<u128> = self.samples.iter().map(|d| d.as_micros()).collect();
        sorted.sort_unstable();
        Some(LatencySummary {
            p50_ms: percentile(&sorted, 0.50),
            p95_ms: percentile(&sorted, 0.95),
            samples: sorted.len(),
        })
    }
}

/// Snapshot of latency statistics; serialised into `ci_baseline.json`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencySummary {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub samples: usize,
}

/// Linear-interpolation percentile over sorted microsecond samples,
/// returned in milliseconds.
fn percentile(sorted_us: &[u128], q: f64) -> f64 {
    debug_assert!(!sorted_us.is_empty());
    let n = sorted_us.len();
    if n == 1 {
        return sorted_us[0] as f64 / 1000.0;
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    let lo_us = sorted_us[lo] as f64;
    let hi_us = sorted_us[hi] as f64;
    (lo_us + (hi_us - lo_us) * frac) / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_summary_is_none() {
        let s = Samples::new();
        assert!(s.summary().is_none());
    }

    #[test]
    fn single_sample_p50_equals_p95() {
        let mut s = Samples::new();
        s.record(Duration::from_millis(100));
        let sum = s.summary().unwrap();
        assert!((sum.p50_ms - 100.0).abs() < 1e-6);
        assert!((sum.p95_ms - 100.0).abs() < 1e-6);
        assert_eq!(sum.samples, 1);
    }

    #[test]
    fn percentile_on_uniform_distribution() {
        let mut s = Samples::new();
        for ms in 1..=100 {
            s.record(Duration::from_millis(ms));
        }
        let sum = s.summary().unwrap();
        // p50 ≈ 50.5 ms, p95 ≈ 95.05 ms (linear interpolation)
        assert!((sum.p50_ms - 50.5).abs() < 0.5);
        assert!((sum.p95_ms - 95.05).abs() < 0.5);
    }

    #[test]
    fn p95_resists_single_outlier() {
        let mut s = Samples::new();
        for _ in 0..99 {
            s.record(Duration::from_millis(10));
        }
        s.record(Duration::from_millis(10_000));
        let sum = s.summary().unwrap();
        // p50 still ≈ 10 ms, p95 still around 10 ms (only one point at 10_000)
        assert!(sum.p50_ms < 15.0);
        assert!(sum.p95_ms < 100.0);
    }
}
