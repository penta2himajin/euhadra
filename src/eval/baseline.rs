//! `ci_baseline*.json` schemas, I/O, and regression gating.
//!
//! Two baseline files live under `docs/benchmarks/`, each consumed by a
//! separate CI job:
//!
//! - `ci_baseline.json` — produced by `eval_l1_smoke` (Phase A-1):
//!   per-language WER/CER + ASR + E2E latency from a live whisper run
//!   on a FLEURS subset.
//! - `ci_baseline_layers.json` — produced by `eval_l1_fast` (Phase A-2):
//!   per-language layer ablation (ΔWER from running the pipeline with
//!   each post-ASR layer toggled on/off) and per-layer μ-benchmark
//!   latency.
//!
//! Both files are self-describing on tolerances so policy lives next to
//! the numbers it constrains.

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
    /// Real-Time Factor: ASR processing time divided by audio duration.
    /// `< 1.0` means the engine ran faster than real-time (required for
    /// streaming dictation). Stored separately from latency because it
    /// is hardware-normalised (RTF on a 5-second utterance is directly
    /// comparable to RTF on a 10-second utterance, while raw latency is
    /// not). `None` on legacy baselines that pre-date RTF reporting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rtf: Option<f64>,
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
    /// RTF regression tolerances (relative). RTF correlates with latency
    /// on the same model + runner pair, but tracking it independently
    /// catches the case where a faster runner masks a model-side
    /// slowdown.
    #[serde(default = "default_rtf_warn")]
    pub rtf_relative_warn: f64,
    #[serde(default = "default_rtf_fail")]
    pub rtf_relative_fail: f64,
}

fn default_rtf_warn() -> f64 {
    1.00
}
fn default_rtf_fail() -> f64 {
    2.00
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
            rtf_relative_warn: default_rtf_warn(),
            rtf_relative_fail: default_rtf_fail(),
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
    if let (Some(m), Some(b)) = (measured.rtf, baseline.rtf) {
        out.push((
            "rtf".to_string(),
            check_latency(m, b, tol.rtf_relative_warn, tol.rtf_relative_fail),
        ));
    }
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

// ---------------------------------------------------------------------------
// Layer baseline (Phase A-2: ablation + per-layer latency)
// ---------------------------------------------------------------------------

/// Schema for `docs/benchmarks/ci_baseline_layers.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerBaseline {
    pub schema_version: u32,
    pub generated: String,
    pub languages: BTreeMap<String, LanguageLayerBaseline>,
    pub tolerances: LayerTolerances,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageLayerBaseline {
    pub fixtures: usize,
    /// Mean error rate (WER for en, CER for ja/zh) of the full pipeline,
    /// then with each post-ASR layer disabled in turn. Layer keys are
    /// well-known identifiers: `full`, `without_filler`,
    /// `without_self_correction`, `without_punctuation`. Languages skip
    /// keys when the layer is not configured for them (e.g. zh has no
    /// filter today).
    pub ablation: BTreeMap<String, f64>,
    /// Median + p95 latency for each layer in isolation (μ-benchmark).
    pub layer_latency_us: BTreeMap<String, LatencyMicrosRecord>,
}

/// Per-layer μ-benchmark latency record. Reported in **microseconds**
/// because rule-based layers are sub-millisecond on typical CI runners
/// and millisecond rounding would erase the signal.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LatencyMicrosRecord {
    pub p50: f64,
    pub p95: f64,
}

impl From<LatencySummary> for LatencyMicrosRecord {
    fn from(s: LatencySummary) -> Self {
        Self {
            p50: round2(s.p50_ms * 1000.0),
            p95: round2(s.p95_ms * 1000.0),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LayerTolerances {
    /// Absolute ΔWER drift allowed before warning, e.g. 0.02 = +2 abs.
    pub ablation_absolute_warn: f64,
    pub ablation_absolute_fail: f64,
    /// Relative latency drift allowed before warning / failing.
    pub layer_latency_p50_relative_warn: f64,
    pub layer_latency_p50_relative_fail: f64,
}

impl Default for LayerTolerances {
    fn default() -> Self {
        // Ablation values are derived from the same fixture set every
        // run, so they are nearly deterministic. Layer μ-benchmark
        // latency on shared CI runners is noisy though; defaults are
        // generous on the first revision and can be tightened once we
        // have empirical CI data.
        Self {
            ablation_absolute_warn: 0.02,
            ablation_absolute_fail: 0.05,
            layer_latency_p50_relative_warn: 2.00,
            layer_latency_p50_relative_fail: 4.00,
        }
    }
}

impl LayerBaseline {
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

pub fn check_language_layers(
    measured: &LanguageLayerBaseline,
    baseline: &LanguageLayerBaseline,
    tol: &LayerTolerances,
) -> Vec<(String, Verdict)> {
    let mut out = Vec::new();

    for (key, m_val) in &measured.ablation {
        let Some(b_val) = baseline.ablation.get(key) else {
            out.push((
                format!("ablation/{key}"),
                Verdict::Warn(format!("no baseline entry for {key}")),
            ));
            continue;
        };
        let abs_delta = (m_val - b_val).abs();
        let v = if abs_delta >= tol.ablation_absolute_fail {
            Verdict::Fail(format!("{:.4} → {:.4} (|Δ| {:.4})", b_val, m_val, abs_delta))
        } else if abs_delta >= tol.ablation_absolute_warn {
            Verdict::Warn(format!("{:.4} → {:.4} (|Δ| {:.4})", b_val, m_val, abs_delta))
        } else {
            Verdict::Pass
        };
        out.push((format!("ablation/{key}"), v));
    }

    for (layer, m_lat) in &measured.layer_latency_us {
        let Some(b_lat) = baseline.layer_latency_us.get(layer) else {
            out.push((
                format!("latency/{layer}"),
                Verdict::Warn(format!("no baseline entry for {layer}")),
            ));
            continue;
        };
        let v = check_latency(
            m_lat.p50,
            b_lat.p50,
            tol.layer_latency_p50_relative_warn,
            tol.layer_latency_p50_relative_fail,
        );
        out.push((format!("latency/{layer}_p50_us"), v));
    }
    out
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
            rtf: Some(0.20),
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
    fn rtf_regression_classified() {
        let baseline = bl(Some(0.20), None);
        let mut measured = baseline.clone();
        let tol = Tolerances::default();
        // Defaults: warn at +100% (2× RTF), fail at +200% (3× RTF)
        measured.rtf = Some(0.50); // 0.20 → 0.50, +150% → warn
        let results = check_language(&measured, &baseline, &tol);
        let v = &results.iter().find(|(k, _)| k == "rtf").unwrap().1;
        assert!(matches!(v, Verdict::Warn(_)), "got {v:?}");

        measured.rtf = Some(0.70); // +250% → fail
        let results = check_language(&measured, &baseline, &tol);
        let v = &results.iter().find(|(k, _)| k == "rtf").unwrap().1;
        assert!(matches!(v, Verdict::Fail(_)), "got {v:?}");
    }

    #[test]
    fn rtf_missing_skips_check() {
        let baseline = bl(Some(0.20), None);
        let mut measured = baseline.clone();
        measured.rtf = None;
        let tol = Tolerances::default();
        let results = check_language(&measured, &baseline, &tol);
        assert!(results.iter().all(|(k, _)| k != "rtf"));
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

    fn lbl() -> LanguageLayerBaseline {
        let mut ablation = BTreeMap::new();
        ablation.insert("full".to_string(), 0.10);
        ablation.insert("without_filler".to_string(), 0.20);
        let mut layer_latency = BTreeMap::new();
        layer_latency.insert(
            "filler".to_string(),
            LatencyMicrosRecord { p50: 500.0, p95: 1000.0 },
        );
        LanguageLayerBaseline {
            fixtures: 25,
            ablation,
            layer_latency_us: layer_latency,
        }
    }

    #[test]
    fn layer_ablation_drift_classification() {
        let baseline = lbl();
        let mut measured = baseline.clone();
        let tol = LayerTolerances::default();

        // Same → pass
        let results = check_language_layers(&measured, &baseline, &tol);
        for (k, v) in &results {
            assert_eq!(v, &Verdict::Pass, "{k}: expected pass, got {v:?}");
        }

        // +0.03 absolute drift on `without_filler` → warn (>=0.02, <0.05)
        measured.ablation.insert("without_filler".to_string(), 0.23);
        let results = check_language_layers(&measured, &baseline, &tol);
        let v = &results
            .iter()
            .find(|(k, _)| k == "ablation/without_filler")
            .unwrap()
            .1;
        assert!(matches!(v, Verdict::Warn(_)), "got {v:?}");

        // +0.06 absolute drift → fail (>=0.05)
        measured.ablation.insert("without_filler".to_string(), 0.26);
        let results = check_language_layers(&measured, &baseline, &tol);
        let v = &results
            .iter()
            .find(|(k, _)| k == "ablation/without_filler")
            .unwrap()
            .1;
        assert!(matches!(v, Verdict::Fail(_)), "got {v:?}");
    }

    #[test]
    fn layer_latency_drift_classification() {
        let baseline = lbl();
        let mut measured = baseline.clone();
        let tol = LayerTolerances::default();

        // Default tolerances: warn 200%, fail 400%
        measured.layer_latency_us.insert(
            "filler".to_string(),
            LatencyMicrosRecord { p50: 1500.0, p95: 1500.0 }, // +200% → warn boundary
        );
        let results = check_language_layers(&measured, &baseline, &tol);
        let v = &results
            .iter()
            .find(|(k, _)| k == "latency/filler_p50_us")
            .unwrap()
            .1;
        assert!(matches!(v, Verdict::Warn(_)), "got {v:?}");

        measured.layer_latency_us.insert(
            "filler".to_string(),
            LatencyMicrosRecord { p50: 3000.0, p95: 3000.0 }, // +500% → fail
        );
        let results = check_language_layers(&measured, &baseline, &tol);
        let v = &results
            .iter()
            .find(|(k, _)| k == "latency/filler_p50_us")
            .unwrap()
            .1;
        assert!(matches!(v, Verdict::Fail(_)), "got {v:?}");
    }

    #[test]
    fn layer_baseline_round_trip_serde() {
        let mut langs = BTreeMap::new();
        langs.insert("en".to_string(), lbl());
        let b = LayerBaseline {
            schema_version: 1,
            generated: "2026-04-25T00:00:00Z".to_string(),
            languages: langs,
            tolerances: LayerTolerances::default(),
        };
        let json = serde_json::to_string(&b).unwrap();
        let parsed: LayerBaseline = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.languages.len(), 1);
    }
}
