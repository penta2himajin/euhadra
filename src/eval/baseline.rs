//! `ci_baseline.json` schema, I/O, and regression gating.
//!
//! The baseline file lives at `docs/benchmarks/ci_baseline.json` and is
//! read by the `eval_l1_smoke` binary on every CI run. Discrepancies are
//! classified as `Pass` / `Warn` / `Fail` per the tolerances declared in
//! the same file (so the policy lives next to the numbers it constrains).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use crate::eval::latency::LatencySummary;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub schema_version: u32,
    pub generated: String,
    pub asr_model: String,
    pub languages: BTreeMap<String, LanguageBaseline>,
    pub tolerances: Tolerances,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageBaseline {
    pub samples: usize,
    /// `Some` for languages where WER is the primary metric (en),
    /// `None` for languages that report CER instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wer: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cer: Option<f64>,
    pub asr_latency_ms: LatencyRecord,
    pub e2e_latency_ms: LatencyRecord,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LatencyRecord {
    pub p50: f64,
    pub p95: f64,
}

impl From<LatencySummary> for LatencyRecord {
    fn from(s: LatencySummary) -> Self {
        Self {
            p50: round2(s.p50_ms),
            p95: round2(s.p95_ms),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Tolerances {
    /// Absolute WER/CER increase that triggers a warning, e.g. 0.05 = 5%
    /// absolute regression flagged.
    pub wer_absolute_warn: f64,
    pub wer_absolute_fail: f64,
    pub wer_relative_warn: f64,
    pub wer_relative_fail: f64,
    pub latency_p50_relative_warn: f64,
    pub latency_p50_relative_fail: f64,
    pub e2e_latency_p50_relative_warn: f64,
    pub e2e_latency_p50_relative_fail: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        // Tuned for GitHub-hosted Linux runners on the L1 smoke workload
        // (10 utterances × 3 languages, whisper-tiny). WER/CER are
        // deterministic given fixed model + audio, so we keep those
        // tight. Latency varies more across runners (~1.5–2× spread
        // even on the same hardware family), so initial latency
        // tolerances are intentionally generous; tighten in a follow-up
        // PR once we have a few CI runs of empirical data.
        Self {
            wer_absolute_warn: 0.05,
            wer_absolute_fail: 0.10,
            wer_relative_warn: 0.20,
            wer_relative_fail: 0.50,
            latency_p50_relative_warn: 1.00, // 2× slower → warn
            latency_p50_relative_fail: 2.00, // 3× slower → fail
            e2e_latency_p50_relative_warn: 0.50,
            e2e_latency_p50_relative_fail: 1.50,
        }
    }
}

impl Baseline {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

/// Outcome of comparing a single (language, metric) measurement against
/// the baseline.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Pass,
    Warn(String),
    Fail(String),
}

impl Verdict {
    pub fn is_fail(&self) -> bool {
        matches!(self, Verdict::Fail(_))
    }
}

/// Compare one language's measurement to its baseline entry. Returns one
/// `Verdict` per checked metric (wer/cer, asr latency, e2e latency); call
/// sites can summarise.
pub fn check_language(
    measured: &LanguageBaseline,
    baseline: &LanguageBaseline,
    tol: &Tolerances,
) -> Vec<(String, Verdict)> {
    let mut out = Vec::new();

    if let (Some(m), Some(b)) = (measured.wer, baseline.wer) {
        out.push(("wer".to_string(), check_error_rate(m, b, tol)));
    }
    if let (Some(m), Some(b)) = (measured.cer, baseline.cer) {
        out.push(("cer".to_string(), check_error_rate(m, b, tol)));
    }
    out.push((
        "asr_latency_p50_ms".to_string(),
        check_latency(
            measured.asr_latency_ms.p50,
            baseline.asr_latency_ms.p50,
            tol.latency_p50_relative_warn,
            tol.latency_p50_relative_fail,
        ),
    ));
    out.push((
        "e2e_latency_p50_ms".to_string(),
        check_latency(
            measured.e2e_latency_ms.p50,
            baseline.e2e_latency_ms.p50,
            tol.e2e_latency_p50_relative_warn,
            tol.e2e_latency_p50_relative_fail,
        ),
    ));
    out
}

fn check_error_rate(measured: f64, baseline: f64, tol: &Tolerances) -> Verdict {
    if measured.is_nan() || baseline.is_nan() {
        return Verdict::Fail(format!("NaN encountered: measured={measured}, baseline={baseline}"));
    }
    let abs_delta = measured - baseline;
    let rel_delta = if baseline > 0.0 { abs_delta / baseline } else { 0.0 };

    if abs_delta >= tol.wer_absolute_fail || rel_delta >= tol.wer_relative_fail {
        Verdict::Fail(format!(
            "{:.4} → {:.4} (Δ {:+.4}, rel {:+.1}%)",
            baseline,
            measured,
            abs_delta,
            rel_delta * 100.0
        ))
    } else if abs_delta >= tol.wer_absolute_warn || rel_delta >= tol.wer_relative_warn {
        Verdict::Warn(format!(
            "{:.4} → {:.4} (Δ {:+.4}, rel {:+.1}%)",
            baseline,
            measured,
            abs_delta,
            rel_delta * 100.0
        ))
    } else {
        Verdict::Pass
    }
}

fn check_latency(measured: f64, baseline: f64, warn_rel: f64, fail_rel: f64) -> Verdict {
    if baseline <= 0.0 {
        return Verdict::Pass;
    }
    let rel = (measured - baseline) / baseline;
    if rel >= fail_rel {
        Verdict::Fail(format!(
            "{:.1}ms → {:.1}ms ({:+.1}%)",
            baseline,
            measured,
            rel * 100.0
        ))
    } else if rel >= warn_rel {
        Verdict::Warn(format!(
            "{:.1}ms → {:.1}ms ({:+.1}%)",
            baseline,
            measured,
            rel * 100.0
        ))
    } else {
        Verdict::Pass
    }
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bl(wer: Option<f64>, cer: Option<f64>) -> LanguageBaseline {
        LanguageBaseline {
            samples: 10,
            wer,
            cer,
            asr_latency_ms: LatencyRecord { p50: 100.0, p95: 200.0 },
            e2e_latency_ms: LatencyRecord { p50: 150.0, p95: 250.0 },
        }
    }

    #[test]
    fn equal_measurement_passes_all_checks() {
        let baseline = bl(Some(0.20), None);
        let measured = baseline.clone();
        let tol = Tolerances::default();
        let results = check_language(&measured, &baseline, &tol);
        for (name, v) in &results {
            assert_eq!(v, &Verdict::Pass, "{name} expected Pass, got {v:?}");
        }
    }

    #[test]
    fn wer_absolute_regression_warns_then_fails() {
        let baseline = bl(Some(0.20), None);
        let mut measured = baseline.clone();
        let tol = Tolerances::default();

        measured.wer = Some(0.26); // +6 abs, +30% rel → warn (rel)
        let results = check_language(&measured, &baseline, &tol);
        let wer_v = &results.iter().find(|(k, _)| k == "wer").unwrap().1;
        assert!(matches!(wer_v, Verdict::Warn(_)), "got {wer_v:?}");

        measured.wer = Some(0.31); // +11 abs (>=10) → fail
        let results = check_language(&measured, &baseline, &tol);
        let wer_v = &results.iter().find(|(k, _)| k == "wer").unwrap().1;
        assert!(matches!(wer_v, Verdict::Fail(_)), "got {wer_v:?}");
    }

    #[test]
    fn wer_improvement_is_pass() {
        let baseline = bl(Some(0.20), None);
        let mut measured = baseline.clone();
        measured.wer = Some(0.10);
        let tol = Tolerances::default();
        let results = check_language(&measured, &baseline, &tol);
        let wer_v = &results.iter().find(|(k, _)| k == "wer").unwrap().1;
        assert_eq!(wer_v, &Verdict::Pass);
    }

    #[test]
    fn latency_regression_classified_correctly() {
        let baseline = bl(Some(0.20), None);
        let mut measured = baseline.clone();
        let tol = Tolerances::default();
        // Defaults: warn at +100% (2x), fail at +200% (3x)
        measured.asr_latency_ms.p50 = 250.0; // +150% → warn
        let results = check_language(&measured, &baseline, &tol);
        let v = &results.iter().find(|(k, _)| k == "asr_latency_p50_ms").unwrap().1;
        assert!(matches!(v, Verdict::Warn(_)), "got {v:?}");

        measured.asr_latency_ms.p50 = 350.0; // +250% → fail
        let results = check_language(&measured, &baseline, &tol);
        let v = &results.iter().find(|(k, _)| k == "asr_latency_p50_ms").unwrap().1;
        assert!(matches!(v, Verdict::Fail(_)), "got {v:?}");
    }

    #[test]
    fn round_trip_serde() {
        let mut langs = BTreeMap::new();
        langs.insert("en".to_string(), bl(Some(0.20), None));
        langs.insert("ja".to_string(), bl(None, Some(0.30)));
        let b = Baseline {
            schema_version: 1,
            generated: "2026-04-25T00:00:00Z".to_string(),
            asr_model: "ggml-tiny.en.bin".to_string(),
            languages: langs,
            tolerances: Tolerances::default(),
        };
        let json = serde_json::to_string(&b).unwrap();
        let parsed: Baseline = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.languages.len(), 2);
        assert_eq!(parsed.schema_version, 1);
    }
}
