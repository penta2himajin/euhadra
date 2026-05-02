//! L3 evaluation runner — Phase C (direct F1 + ablation).
//!
//! Two task modes (`--task`):
//!
//! - **`self-correction`** (Phase C-1): runs `SelfCorrectionDetector`
//!   against an annotated JSONL file and reports utterance-level +
//!   span-level F1. Used to measure how well the detector finds
//!   reparandum boundaries on hand-curated gold data.
//! - **`ablation`** (Phase C-2): replays a natural-speech fixture set
//!   through the post-ASR pipeline with each layer toggled on/off and
//!   reports ΔWER per configuration. Functionally identical to
//!   `eval_l1_fast` but pointed at richer fixtures (e.g. ReazonSpeech-
//!   derived, not the synthetic L1 set), so the same machinery is
//!   reused via a different `--fixtures-dir`.
//!
//! No committed baseline — L3 is research / release-time, not CI
//! regression. Output is stdout summary plus optional `--output JSON`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use euhadra::eval::annotations::load_jsonl as load_annotations;
use euhadra::eval::baseline::{LanguageLayerBaseline, LatencyMicrosRecord};
use euhadra::eval::f1::{F1Stats, Span, aggregate, iou_f1, strict_f1};
use euhadra::eval::fixtures::{Fixture, load_jsonl as load_fixtures};
use euhadra::eval::latency::Samples;
use euhadra::eval::metrics::{cer, wer};
use euhadra::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "L3: direct F1 (self-correction / filler) + ablation on natural speech")]
struct Cli {
    #[arg(long, value_enum)]
    task: Task,

    /// Language code (en / ja / zh). For `self-correction` task this
    /// determines the cue closed-set used to trim predicted spans.
    #[arg(long, default_value = "ja")]
    lang: String,

    /// Path to annotations JSONL (for `--task self-correction`) or
    /// fixtures JSONL (for `--task ablation`).
    #[arg(long)]
    input: PathBuf,

    /// IoU threshold for span-level F1 (self-correction only).
    #[arg(long, default_value_t = 0.5)]
    iou_threshold: f64,

    /// Optional report file (JSON).
    #[arg(long)]
    output: Option<PathBuf>,

    /// Print per-utterance predicted vs gold spans for the
    /// `self-correction` task. Useful when debugging boundary mismatches.
    #[arg(long)]
    verbose: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Task {
    SelfCorrection,
    Ablation,
    Filler,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}

async fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.task {
        Task::SelfCorrection => run_self_correction(&cli).await,
        Task::Ablation => run_ablation(&cli).await,
        Task::Filler => run_filler(&cli).await,
    }
}

// ---------------------------------------------------------------------------
// Task: self-correction (Phase C-1)
// ---------------------------------------------------------------------------

async fn run_self_correction(cli: &Cli) -> Result<(), String> {
    let annotations = load_annotations(&cli.input)
        .map_err(|e| format!("loading {}: {e}", cli.input.display()))?;
    if annotations.is_empty() {
        return Err(format!("annotation file {} is empty", cli.input.display()));
    }

    let detector = SelfCorrectionDetector::new();
    let ctx = ContextSnapshot::default();

    // Per-utterance fire/no-fire bookkeeping for utterance-level F1
    // and per-utterance span F1 for span-level aggregation.
    let mut utt_tp = 0usize;
    let mut utt_fp = 0usize;
    let mut utt_fn = 0usize;
    let mut utt_tn = 0usize;
    let mut span_stats: Vec<F1Stats> = Vec::new();
    let mut strict_stats: Vec<F1Stats> = Vec::new();

    let cues = cue_set_for(&cli.lang)?;

    for anno in &annotations {
        let result = detector
            .process(&anno.text, &ctx)
            .await
            .map_err(|e| format!("detector on {}: {e}", anno.utterance_id))?;

        // Predicted reparandum spans: derive from the diff between the
        // input and the corrected output, then trim the trailing cue
        // word so the span is reparandum-only (not reparandum + cue).
        let predicted: Vec<Span> = if result.corrections.is_empty() {
            Vec::new()
        } else {
            match diff_removed_span(&anno.text, &result.text) {
                Some(raw) => vec![trim_trailing_cue(&anno.text, raw, &cues)],
                None => Vec::new(),
            }
        };
        let gold: Vec<Span> = anno
            .repairs
            .iter()
            .map(|r| r.reparandum.span())
            .collect();

        // Utterance-level fire / no-fire (ignores span position).
        match (predicted.is_empty(), gold.is_empty()) {
            (false, false) => utt_tp += 1,
            (false, true) => utt_fp += 1,
            (true, false) => utt_fn += 1,
            (true, true) => utt_tn += 1,
        }

        // Span-level F1 (only meaningful when both sides have a span).
        if !predicted.is_empty() || !gold.is_empty() {
            let strict = strict_f1(&predicted, &gold);
            span_stats.push(iou_f1(&predicted, &gold, cli.iou_threshold));
            strict_stats.push(strict);

            if cli.verbose && (strict.fp > 0 || strict.fn_ > 0) {
                let chars: Vec<char> = anno.text.chars().collect();
                let span_text = |s: &Span| -> String {
                    chars
                        .get(s.start..s.end.min(chars.len()))
                        .map(|s| s.iter().collect::<String>())
                        .unwrap_or_default()
                };
                let pred_str: Vec<String> =
                    predicted.iter().map(|s| format!("{:?}={:?}", (s.start, s.end), span_text(s))).collect();
                let gold_str: Vec<String> =
                    gold.iter().map(|s| format!("{:?}={:?}", (s.start, s.end), span_text(s))).collect();
                if predicted != gold {
                    println!(
                        "  [diff] {} text={:?}\n         predicted={:?}\n         gold={:?}",
                        anno.utterance_id, anno.text, pred_str, gold_str,
                    );
                }
            }
        }
    }

    let utt_f1 = F1Stats::from_counts(utt_tp, utt_fp, utt_fn);
    let span_iou_agg = aggregate(&span_stats);
    let span_strict_agg = aggregate(&strict_stats);

    println!("=== L3 self-correction direct F1 ({}) ===", cli.lang);
    println!("annotations: {}", annotations.len());
    println!(
        "utterance-level   tp={} fp={} fn={} tn={}",
        utt_tp, utt_fp, utt_fn, utt_tn
    );
    println!(
        "  precision={}  recall={}  F1={}",
        fmt_pct(utt_f1.precision),
        fmt_pct(utt_f1.recall),
        fmt_pct(utt_f1.f1),
    );
    println!(
        "span-level (IoU≥{:.2})  tp={} fp={} fn={}  precision={} recall={} F1={}",
        cli.iou_threshold,
        span_iou_agg.tp,
        span_iou_agg.fp,
        span_iou_agg.fn_,
        fmt_pct(span_iou_agg.precision),
        fmt_pct(span_iou_agg.recall),
        fmt_pct(span_iou_agg.f1),
    );
    println!(
        "span-level (strict)    tp={} fp={} fn={}  precision={} recall={} F1={}",
        span_strict_agg.tp,
        span_strict_agg.fp,
        span_strict_agg.fn_,
        fmt_pct(span_strict_agg.precision),
        fmt_pct(span_strict_agg.recall),
        fmt_pct(span_strict_agg.f1),
    );

    if let Some(out) = &cli.output {
        let report = serde_json::json!({
            "task": "self-correction",
            "lang": cli.lang,
            "annotations": annotations.len(),
            "utterance_level": {
                "tp": utt_tp, "fp": utt_fp, "fn": utt_fn, "tn": utt_tn,
                "precision": utt_f1.precision, "recall": utt_f1.recall, "f1": utt_f1.f1,
            },
            "span_level_iou": {
                "iou_threshold": cli.iou_threshold,
                "tp": span_iou_agg.tp, "fp": span_iou_agg.fp, "fn": span_iou_agg.fn_,
                "precision": span_iou_agg.precision, "recall": span_iou_agg.recall, "f1": span_iou_agg.f1,
            },
            "span_level_strict": {
                "tp": span_strict_agg.tp, "fp": span_strict_agg.fp, "fn": span_strict_agg.fn_,
                "precision": span_strict_agg.precision, "recall": span_strict_agg.recall, "f1": span_strict_agg.f1,
            },
        });
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(out, serde_json::to_string_pretty(&report).unwrap())
            .map_err(|e| format!("write {}: {e}", out.display()))?;
        eprintln!("report written to {}", out.display());
    }
    Ok(())
}

fn cue_set_for(lang: &str) -> Result<Vec<&'static str>, String> {
    match lang {
        "ja" | "japanese" => Ok(ja_cue_set()),
        "es" | "spanish" => Ok(es_cue_set()),
        other => Err(format!(
            "self-correction task: --lang {other} not wired \
             (expected one of: ja, es)"
        )),
    }
}

fn ja_cue_set() -> Vec<&'static str> {
    vec![
        "いや",
        "じゃなくて",
        "じゃなく",
        "ではなく",
        "ていうか",
        "っていうか",
        "じゃない",
    ]
}

/// Mirrors `SelfCorrectionDetector::correction_cues_es` in
/// `src/processor.rs`. Order is unspecified — `trim_trailing_cue`
/// re-sorts longest-first internally so that `mejor dicho`
/// outranks `mejor` and `quiero decir` outranks `digo`.
fn es_cue_set() -> Vec<&'static str> {
    vec![
        "mejor dicho",
        "quiero decir",
        "o sea",
        "perdón",
        "mejor",
        "digo",
        "no es",
        "no",
    ]
}

/// Find the contiguous range that's present in `input` but absent from
/// `output`, by computing longest common prefix + suffix and reporting
/// what's between them. Operates on character offsets in `input`.
/// Returns `None` if the two strings are identical.
fn diff_removed_span(input: &str, output: &str) -> Option<Span> {
    let in_chars: Vec<char> = input.chars().collect();
    let out_chars: Vec<char> = output.chars().collect();

    // Longest common suffix.
    let mut suffix_len = 0;
    while suffix_len < in_chars.len()
        && suffix_len < out_chars.len()
        && in_chars[in_chars.len() - 1 - suffix_len]
            == out_chars[out_chars.len() - 1 - suffix_len]
    {
        suffix_len += 1;
    }
    let in_suffix_start = in_chars.len() - suffix_len;
    let out_suffix_start = out_chars.len() - suffix_len;

    // Longest common prefix, but never run past the suffix region.
    let mut prefix_len = 0;
    while prefix_len < in_suffix_start
        && prefix_len < out_suffix_start
        && in_chars[prefix_len] == out_chars[prefix_len]
    {
        prefix_len += 1;
    }

    if prefix_len < in_suffix_start {
        Some(Span {
            start: prefix_len,
            end: in_suffix_start,
        })
    } else {
        None
    }
}

/// Trim trailing separator + cue from a detected span so it
/// represents the reparandum only.
///
/// The detector's diff captures everything between input and output
/// that disappeared. The diff includes both the cue itself and any
/// separator (`、` for Japanese, whitespace + comma/period/etc. for
/// Spanish) immediately before *or after* the cue (e.g.
/// `"鈴木課長、じゃない、佐藤課長です"` → diff is
/// `"鈴木課長、じゃない、"` with both inner and trailing `、`;
/// `"voy mañana no voy hoy"` → diff is `"voy mañana no "` with a
/// trailing space). We:
///
/// 1. strip any trailing separator chars,
/// 2. strip the longest matching cue suffix (longest-first so that
///    `っていうか` outranks `ていうか`, `mejor dicho` outranks
///    `mejor`),
/// 3. strip trailing separator chars again to drop the separator
///    that sits between reparandum and the (just-removed) cue.
fn trim_trailing_cue(input: &str, raw: Span, cues: &[&str]) -> Span {
    let chars: Vec<char> = input.chars().collect();
    if raw.start >= raw.end || raw.end > chars.len() {
        return raw;
    }

    let is_sep = |c: char| {
        // JA separator + Spanish/English whitespace and punctuation
        // that the detector treats as a token boundary in its
        // shared-prefix check (see `detect_spanish` trim_chars).
        c == '、' || c.is_whitespace() || matches!(c, ',' | '.' | ';' | ':' | '!' | '?')
    };

    // Step 1: strip trailing separators.
    let mut end = raw.end;
    while end > raw.start && is_sep(chars[end - 1]) {
        end -= 1;
    }

    // Step 2: strip the longest matching cue suffix.
    let span_text: String = chars[raw.start..end].iter().collect();
    let mut sorted_cues: Vec<&&str> = cues.iter().collect();
    sorted_cues.sort_by_key(|c| std::cmp::Reverse(c.chars().count()));
    for cue in sorted_cues {
        if span_text.ends_with(*cue) {
            let cue_chars = cue.chars().count();
            end -= cue_chars;
            // Step 3: strip trailing separators again, now between
            // reparandum and the (just-removed) cue.
            while end > raw.start && is_sep(chars[end - 1]) {
                end -= 1;
            }
            break;
        }
    }
    Span {
        start: raw.start,
        end,
    }
}

fn fmt_pct(x: f64) -> String {
    if x.is_nan() {
        "n/a".to_string()
    } else {
        format!("{:.3}", x)
    }
}

// ---------------------------------------------------------------------------
// Task: filler — Tier 1 direct F1 against a token-span gold standard.
//
// Spanish only in v1 — driven by the CIEMPIESS Test transcripts that
// `scripts/build_es_filler_annotations.py` lifts into a structured
// JSONL (see PR for license posture). Other languages ship rule-based
// filters (`SimpleFillerFilter`, `JapaneseFillerFilter`,
// `ChineseFillerFilter`) but no codepoint-span emitter yet, so the
// strict-F1 evaluator can't compare against a gold annotation. Wire
// them up case-by-case as filter span emitters land.
// ---------------------------------------------------------------------------

async fn run_filler(cli: &Cli) -> Result<(), String> {
    type SpanDetector = Box<dyn Fn(&str) -> Vec<Span>>;
    let lang = cli.lang.as_str();
    let detect_spans: SpanDetector = match lang {
        "es" | "spanish" => {
            let filter = SpanishFillerFilter::new();
            Box::new(move |t| filter.detect_spans(t))
        }
        other => {
            return Err(format!(
                "filler task: --lang {other} not wired (es only in v1; \
                 other filters lack a codepoint span emitter)"
            ));
        }
    };

    let annotations = load_annotations(&cli.input)
        .map_err(|e| format!("loading {}: {e}", cli.input.display()))?;
    if annotations.is_empty() {
        return Err(format!("annotation file {} is empty", cli.input.display()));
    }

    let mut utt_tp = 0usize;
    let mut utt_fp = 0usize;
    let mut utt_fn = 0usize;
    let mut utt_tn = 0usize;
    let mut span_stats: Vec<F1Stats> = Vec::new();

    for anno in &annotations {
        let predicted = detect_spans(&anno.text);
        let gold: Vec<Span> = anno.fillers.iter().map(|f| f.span()).collect();

        // Utterance-level fire / no-fire (ignores positions): a single
        // predicted span counts as a fire regardless of how many gold
        // spans the utterance actually has.
        match (predicted.is_empty(), gold.is_empty()) {
            (false, false) => utt_tp += 1,
            (false, true) => utt_fp += 1,
            (true, false) => utt_fn += 1,
            (true, true) => utt_tn += 1,
        }

        // Span-level strict F1: closed-class lexicons make boundaries
        // unambiguous, so IoU-based scoring is unnecessary here.
        if !predicted.is_empty() || !gold.is_empty() {
            let stats = strict_f1(&predicted, &gold);
            span_stats.push(stats);

            if cli.verbose && (stats.fp > 0 || stats.fn_ > 0) {
                let chars: Vec<char> = anno.text.chars().collect();
                let span_text = |s: &Span| -> String {
                    chars
                        .get(s.start..s.end.min(chars.len()))
                        .map(|s| s.iter().collect::<String>())
                        .unwrap_or_default()
                };
                let pred_str: Vec<String> = predicted
                    .iter()
                    .map(|s| format!("{:?}={:?}", (s.start, s.end), span_text(s)))
                    .collect();
                let gold_str: Vec<String> = gold
                    .iter()
                    .map(|s| format!("{:?}={:?}", (s.start, s.end), span_text(s)))
                    .collect();
                println!(
                    "  [diff] {} text={:?}\n         predicted={:?}\n         gold={:?}",
                    anno.utterance_id, anno.text, pred_str, gold_str,
                );
            }
        }
    }

    let utt_f1 = F1Stats::from_counts(utt_tp, utt_fp, utt_fn);
    let span_agg = aggregate(&span_stats);

    println!("=== L3 filler direct F1 ({}) ===", cli.lang);
    println!("annotations: {}", annotations.len());
    println!(
        "utterance-level   tp={} fp={} fn={} tn={}",
        utt_tp, utt_fp, utt_fn, utt_tn
    );
    println!(
        "  precision={}  recall={}  F1={}",
        fmt_pct(utt_f1.precision),
        fmt_pct(utt_f1.recall),
        fmt_pct(utt_f1.f1),
    );
    println!(
        "span-level (strict)    tp={} fp={} fn={}  precision={} recall={} F1={}",
        span_agg.tp,
        span_agg.fp,
        span_agg.fn_,
        fmt_pct(span_agg.precision),
        fmt_pct(span_agg.recall),
        fmt_pct(span_agg.f1),
    );

    if let Some(out) = &cli.output {
        let report = serde_json::json!({
            "task": "filler",
            "lang": cli.lang,
            "annotations": annotations.len(),
            "utterance_level": {
                "tp": utt_tp, "fp": utt_fp, "fn": utt_fn, "tn": utt_tn,
                "precision": utt_f1.precision, "recall": utt_f1.recall, "f1": utt_f1.f1,
            },
            "span_level_strict": {
                "tp": span_agg.tp, "fp": span_agg.fp, "fn": span_agg.fn_,
                "precision": span_agg.precision, "recall": span_agg.recall, "f1": span_agg.f1,
            },
        });
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(out, serde_json::to_string_pretty(&report).unwrap())
            .map_err(|e| format!("write {}: {e}", out.display()))?;
        eprintln!("report written to {}", out.display());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Task: ablation (Phase C-2) — reuses the L1-fast machinery against a
// natural-speech fixture set.
// ---------------------------------------------------------------------------

async fn run_ablation(cli: &Cli) -> Result<(), String> {
    let fixtures = load_fixtures(&cli.input)
        .map_err(|e| format!("loading fixtures {}: {e}", cli.input.display()))?;
    if fixtures.is_empty() {
        return Err(format!("fixture file {} is empty", cli.input.display()));
    }

    let result = evaluate_ablation_for_lang(&cli.lang, &fixtures).await?;
    println!("=== L3 ablation ({}) ===", cli.lang);
    println!("fixtures: {}", result.fixtures);
    let primary = if cli.lang == "en" { "WER" } else { "CER" };
    for (cfg, er) in &result.ablation {
        println!("  ablation/{cfg:30}  {primary}={er:.4}");
    }
    for (layer, lat) in &result.layer_latency_us {
        println!(
            "  latency/{layer:30}  p50={:.1}μs p95={:.1}μs",
            lat.p50, lat.p95
        );
    }

    if let Some(out) = &cli.output {
        let json = serde_json::to_string_pretty(&result).map_err(|e| format!("json: {e}"))?;
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(out, json).map_err(|e| format!("write {}: {e}", out.display()))?;
        eprintln!("report written to {}", out.display());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct LayerConfig {
    name: &'static str,
    filter: bool,
    self_correction: bool,
    punctuation: bool,
}

const FULL: LayerConfig = LayerConfig {
    name: "full",
    filter: true,
    self_correction: true,
    punctuation: true,
};
const WITHOUT_FILLER: LayerConfig = LayerConfig {
    name: "without_filler",
    filter: false,
    self_correction: true,
    punctuation: true,
};
const WITHOUT_SC: LayerConfig = LayerConfig {
    name: "without_self_correction",
    filter: true,
    self_correction: false,
    punctuation: true,
};
const WITHOUT_PUNCT: LayerConfig = LayerConfig {
    name: "without_punctuation",
    filter: true,
    self_correction: true,
    punctuation: false,
};

async fn evaluate_ablation_for_lang(
    lang: &str,
    fixtures: &[Fixture],
) -> Result<LanguageLayerBaseline, String> {
    let configs: Vec<LayerConfig> = match lang {
        "en" | "ja" | "zh" => vec![FULL, WITHOUT_FILLER, WITHOUT_SC, WITHOUT_PUNCT],
        other => return Err(format!("unsupported lang {other}")),
    };

    let mut ablation = BTreeMap::new();
    for cfg in &configs {
        let er = mean_error_rate(lang, fixtures, cfg).await?;
        ablation.insert(cfg.name.to_string(), round4(er));
    }
    let layer_latency = bench_layer_latency(lang, fixtures, 10, 100).await;

    Ok(LanguageLayerBaseline {
        fixtures: fixtures.len(),
        ablation,
        layer_latency_us: layer_latency,
    })
}

async fn mean_error_rate(
    lang: &str,
    fixtures: &[Fixture],
    cfg: &LayerConfig,
) -> Result<f64, String> {
    let mut sum = 0.0;
    let mut counted = 0;

    for fix in fixtures {
        let pipeline = build_pipeline(lang, cfg, &fix.asr_hypothesis)?;
        let (audio_tx, _cancel, handle) = pipeline.session();
        audio_tx
            .send(AudioChunk {
                samples: vec![0.0; 160],
                sample_rate: 16000,
                channels: 1,
            })
            .await
            .map_err(|e| format!("send: {e}"))?;
        drop(audio_tx);
        let result = handle
            .await
            .map_err(|e| format!("join: {e}"))?
            .map_err(|e| format!("pipeline: {e}"))?;
        let RefinementOutput::TextInsertion { text, .. } = &result.output else {
            return Err("expected TextInsertion".into());
        };
        let er = match lang {
            "en" => wer(&fix.reference, text),
            _ => cer(&fix.reference, text),
        };
        if !er.is_nan() {
            sum += er;
            counted += 1;
        }
    }
    if counted == 0 {
        return Err("no scorable fixtures".into());
    }
    Ok(sum / counted as f64)
}

fn build_pipeline(lang: &str, cfg: &LayerConfig, hypothesis: &str) -> Result<Pipeline, String> {
    let mut builder = Pipeline::builder()
        .asr(MockAsr::new(hypothesis))
        .refiner(MockRefiner::passthrough())
        .context(MockContextProvider::new())
        .emitter(MockEmitter::new());
    if cfg.filter {
        builder = match lang {
            "en" => builder.filter(SimpleFillerFilter::english()),
            "ja" => builder.filter(JapaneseFillerFilter::new()),
            "zh" => builder.filter(ChineseFillerFilter::new()),
            other => return Err(format!("unsupported lang {other}")),
        };
    }
    if cfg.self_correction {
        builder = builder.processor(SelfCorrectionDetector::new());
    }
    if cfg.punctuation {
        builder = builder.processor(BasicPunctuationRestorer);
    }
    builder.build().map_err(|e| format!("build pipeline: {e}"))
}

async fn bench_layer_latency(
    lang: &str,
    fixtures: &[Fixture],
    warmup: usize,
    iters: usize,
) -> BTreeMap<String, LatencyMicrosRecord> {
    let mut out = BTreeMap::new();
    match lang {
        "en" => {
            let f = SimpleFillerFilter::english();
            out.insert(
                "filler".to_string(),
                bench_filter(&f, fixtures, warmup, iters).await,
            );
        }
        "ja" => {
            let f = JapaneseFillerFilter::new();
            out.insert(
                "filler".to_string(),
                bench_filter(&f, fixtures, warmup, iters).await,
            );
        }
        "zh" => {
            let f = ChineseFillerFilter::new();
            out.insert(
                "filler".to_string(),
                bench_filter(&f, fixtures, warmup, iters).await,
            );
        }
        _ => {}
    }
    let sc = SelfCorrectionDetector::new();
    out.insert(
        "self_correction".to_string(),
        bench_processor(&sc, fixtures, warmup, iters).await,
    );
    let punct = BasicPunctuationRestorer;
    out.insert(
        "punctuation".to_string(),
        bench_processor(&punct, fixtures, warmup, iters).await,
    );
    out
}

async fn bench_filter<F: TextFilter>(
    layer: &F,
    fixtures: &[Fixture],
    warmup: usize,
    iters: usize,
) -> LatencyMicrosRecord {
    for fix in fixtures.iter().take(warmup.min(fixtures.len())) {
        let _ = layer.filter(&fix.asr_hypothesis).await;
    }
    let mut samples = Samples::new();
    for _ in 0..iters {
        for fix in fixtures {
            let start = Instant::now();
            let _ = layer.filter(&fix.asr_hypothesis).await;
            samples.record(start.elapsed());
        }
    }
    samples.summary().expect("non-empty fixtures").into()
}

async fn bench_processor<P: TextProcessor>(
    layer: &P,
    fixtures: &[Fixture],
    warmup: usize,
    iters: usize,
) -> LatencyMicrosRecord {
    let ctx = ContextSnapshot::default();
    for fix in fixtures.iter().take(warmup.min(fixtures.len())) {
        let _ = layer.process(&fix.asr_hypothesis, &ctx).await;
    }
    let mut samples = Samples::new();
    for _ in 0..iters {
        for fix in fixtures {
            let start = Instant::now();
            let _ = layer.process(&fix.asr_hypothesis, &ctx).await;
            samples.record(start.elapsed());
        }
    }
    samples.summary().expect("non-empty fixtures").into()
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

