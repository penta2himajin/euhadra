//! L1 fast evaluation runner — Phase A-2.
//!
//! Replays per-language fixtures (`tests/evaluation/fixtures/{en,ja,zh,es,ko}.jsonl`)
//! through the post-ASR pipeline using `MockAsr` to inject the
//! pre-recorded hypothesis text. For each language we measure:
//!
//! 1. **Layer ablation**: WER/CER of the full pipeline, then with each
//!    post-ASR layer disabled in turn (filter, self-correction,
//!    punctuation). The delta between configurations is the layer's
//!    contribution to the final output.
//! 2. **Per-layer μ-benchmark latency**: each layer's `filter()` /
//!    `process()` is called directly on the fixture set with a 10-call
//!    warmup + 100 measured calls (criterion-style), reporting p50/p95
//!    in microseconds since rule-based layers are sub-millisecond.
//!
//! Output schema lives in `src/eval/baseline.rs` (`LayerBaseline`) and
//! is gated against `docs/benchmarks/ci_baseline_layers.json`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use euhadra::eval::baseline::{
    LanguageLayerBaseline, LatencyMicrosRecord, LayerBaseline, LayerTolerances, Verdict,
    check_language_layers,
};
use euhadra::eval::fixtures::{Fixture, load_jsonl};
use euhadra::eval::latency::Samples;
use euhadra::eval::metrics::{cer, wer};
use euhadra::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "L1 fast: layer ablation + per-layer μ-benchmark latency on text fixtures")]
struct Cli {
    #[arg(long, default_value = "tests/evaluation/fixtures")]
    fixtures_dir: PathBuf,

    #[arg(long, default_value = "docs/benchmarks/ci_baseline_layers.json")]
    baseline: PathBuf,

    #[arg(long, value_delimiter = ',', default_value = "en,ja,zh,es,ko")]
    langs: Vec<String>,

    /// Number of warmup calls before timed μ-benchmark calls.
    #[arg(long, default_value_t = 10)]
    warmup: usize,

    /// Number of timed μ-benchmark calls per (layer, fixture).
    #[arg(long, default_value_t = 100)]
    iters: usize,

    /// Write the measured numbers as the new baseline instead of comparing.
    #[arg(long)]
    update_baseline: bool,
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
    let mut measured: BTreeMap<String, LanguageLayerBaseline> = BTreeMap::new();

    for lang in &cli.langs {
        let path = cli.fixtures_dir.join(format!("{lang}.jsonl"));
        let fixtures =
            load_jsonl(&path).map_err(|e| format!("loading {}: {e}", path.display()))?;
        if fixtures.is_empty() {
            return Err(format!("fixture file {} is empty", path.display()));
        }

        let lang_result = evaluate_language(lang, &fixtures, cli.warmup, cli.iters).await?;
        print_language_result(lang, &lang_result);
        measured.insert(lang.clone(), lang_result);
    }

    if cli.update_baseline {
        let baseline = LayerBaseline {
            schema_version: 1,
            generated: now_marker(),
            languages: measured,
            tolerances: LayerTolerances::default(),
        };
        baseline
            .save(&cli.baseline)
            .map_err(|e| format!("writing baseline: {e}"))?;
        eprintln!("layer baseline written to {}", cli.baseline.display());
        return Ok(());
    }

    let baseline = LayerBaseline::load(&cli.baseline)
        .map_err(|e| format!("loading baseline {}: {e}", cli.baseline.display()))?;

    let mut any_fail = false;
    let mut any_warn = false;
    for (lang, m) in &measured {
        let Some(b) = baseline.languages.get(lang) else {
            eprintln!("warn: {lang} has no baseline entry");
            continue;
        };
        let verdicts = check_language_layers(m, b, &baseline.tolerances);
        for (metric, v) in verdicts {
            match v {
                Verdict::Pass => println!("  [{lang}] {metric}: pass"),
                Verdict::Warn(msg) => {
                    println!("  [{lang}] {metric}: WARN  {msg}");
                    any_warn = true;
                }
                Verdict::Fail(msg) => {
                    println!("  [{lang}] {metric}: FAIL  {msg}");
                    any_fail = true;
                }
            }
        }
    }

    if any_fail {
        return Err("regression detected (see FAIL entries above)".into());
    }
    if any_warn {
        eprintln!("note: warnings present but within hard-fail tolerance");
    }
    Ok(())
}

/// Which post-ASR layers are present in this configuration. Constructing
/// a pipeline with a layer omitted is what gives us the ablation delta.
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

const WITHOUT_SELF_CORR: LayerConfig = LayerConfig {
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

async fn evaluate_language(
    lang: &str,
    fixtures: &[Fixture],
    warmup: usize,
    iters: usize,
) -> Result<LanguageLayerBaseline, String> {
    // Configurations skip the layer flags that don't apply to this
    // language, so e.g. zh — which has no filter today — only reports
    // configurations that actually differ.
    let configs: Vec<LayerConfig> = match lang {
        "en" | "ja" | "zh" | "es" | "ko" => {
            vec![FULL, WITHOUT_FILLER, WITHOUT_SELF_CORR, WITHOUT_PUNCT]
        }
        other => return Err(format!("unsupported lang {other}")),
    };

    let mut ablation = BTreeMap::new();
    for cfg in &configs {
        let er = mean_error_rate(lang, fixtures, cfg).await?;
        ablation.insert(cfg.name.to_string(), round4(er));
    }

    let layer_latency = bench_layer_latency(lang, fixtures, warmup, iters).await;

    Ok(LanguageLayerBaseline {
        fixtures: fixtures.len(),
        ablation,
        layer_latency_us: layer_latency,
    })
}

/// Build a pipeline reflecting `cfg` and run every fixture through it.
/// Returns the mean WER (en) or CER (ja, zh) across the fixture set.
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

        // MockAsr ignores audio content; we just need to deliver one
        // chunk and close the channel for the session to advance.
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
            "en" | "es" => wer(&fix.reference, text),
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
            "es" => builder.filter(SpanishFillerFilter::new()),
            "ko" => builder.filter(SimpleFillerFilter::korean()),
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

/// μ-benchmark each post-ASR layer in isolation against the fixture set.
/// We hit the layer's public `filter()` / `process()` directly rather
/// than the full pipeline so that pipeline overhead doesn't smear the
/// per-layer signal.
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
        "es" => {
            let f = SpanishFillerFilter::new();
            out.insert(
                "filler".to_string(),
                bench_filter(&f, fixtures, warmup, iters).await,
            );
        }
        "ko" => {
            let f = SimpleFillerFilter::korean();
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

fn print_language_result(lang: &str, r: &LanguageLayerBaseline) {
    println!("[{lang}] fixtures={}", r.fixtures);
    // en / es are whitespace-segmented → WER; ja / zh use CER.
    let primary = if matches!(lang, "en" | "es") { "WER" } else { "CER" };
    for (cfg, er) in &r.ablation {
        println!("  ablation/{cfg:30}  {primary}={er:.4}");
    }
    for (layer, lat) in &r.layer_latency_us {
        println!(
            "  latency/{layer:30}  p50={:.1}μs p95={:.1}μs",
            lat.p50, lat.p95
        );
    }
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

fn now_marker() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch:{secs}")
}
