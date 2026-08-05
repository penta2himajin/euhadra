//! Embedding-backend calibration bench for the Tier 1 filler filter.
//!
//! `OnnxEmbeddingFilter` gates a lexicon hit on a cosine similarity
//! against filler prototypes. That threshold — 0.82 in `docs/spec.md`
//! §3.5, measured against `bge-small-en-v1.5` — is a property of the
//! embedding space, not of the task, so swapping the backend
//! invalidates it. This bench re-derives it.
//!
//! For every gold annotation in
//! `tests/evaluation/annotations/<lang>_filler.jsonl` it segments the
//! utterance, embeds each segment once, and labels the segment
//! positive when it overlaps a gold filler span. It then reports:
//!
//! - the **score distribution** of positives vs negatives (the
//!   separation `docs/spec.md` §3.5 claims), and
//! - a **threshold sweep** with strict-span P/R/F1 at each step, so
//!   the operating point is chosen from data rather than inherited.
//!
//! Usage:
//!
//! ```text
//! scripts/setup_embedders.sh                     # or EMBEDDER_MODEL=granite
//! cargo run --release --features onnx --example bench_embedder -- \
//!     --model-dir vendor/embedder_bge_small \
//!     --lang en \
//!     --annotations tests/evaluation/annotations/en_filler.jsonl
//! ```
//!
//! `--output` writes the same numbers as JSON for the record kept in
//! `docs/model-upgrade-candidates.md`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use euhadra::eval::annotations::load_jsonl as load_annotations;
use euhadra::eval::f1::{strict_f1, Span};
// `OnnxEmbeddingFilter` is deprecated and unwired — this bench is the
// one remaining entry point, kept so the retire-or-keep call can be
// re-made against better data. See docs/model-upgrade-candidates.md §4.
#[allow(deprecated)]
use euhadra::onnx_processing::{FillerLexicon, OnnxEmbeddingFilter};

#[derive(Parser, Debug)]
#[command(about = "Calibrate the Tier 1 embedding filler threshold for a given backend")]
struct Cli {
    /// Directory holding `model.onnx` + `tokenizer.json`.
    #[arg(long)]
    model_dir: PathBuf,

    /// Language code — selects the filler lexicon and segmenter.
    #[arg(long, default_value = "en")]
    lang: String,

    /// Gold filler annotations (JSONL).
    #[arg(long)]
    annotations: PathBuf,

    /// Threshold sweep bounds and step.
    #[arg(long, default_value_t = 0.50)]
    sweep_from: f32,
    #[arg(long, default_value_t = 0.99)]
    sweep_to: f32,
    #[arg(long, default_value_t = 0.01)]
    sweep_step: f32,

    /// Optional JSON report path.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Label for this run in the JSON report (defaults to the model
    /// directory's final component).
    #[arg(long)]
    label: Option<String>,
}

/// A segment paired with its similarity score and gold label.
struct Scored {
    surface: String,
    sim: f32,
    is_gold: bool,
}

fn percentile(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn overlaps(a: &Span, b: &Span) -> bool {
    a.start < b.end && b.start < a.end
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("[error] {e}");
        std::process::exit(1);
    }
}

#[allow(deprecated)]
async fn run() -> Result<(), String> {
    let cli = Cli::parse();

    let lexicon = FillerLexicon::for_language(&cli.lang)
        .ok_or_else(|| format!("no filler lexicon for language {:?}", cli.lang))?;
    let baseline_threshold = lexicon.pure_threshold;

    let label = cli.label.clone().unwrap_or_else(|| {
        cli.model_dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".into())
    });

    let load_start = Instant::now();
    let filter = OnnxEmbeddingFilter::load_with_lexicon(&cli.model_dir, lexicon)
        .map_err(|e| format!("loading {}: {}", cli.model_dir.display(), e.message))?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let annotations = load_annotations(&cli.annotations)
        .map_err(|e| format!("loading {}: {e}", cli.annotations.display()))?;
    if annotations.is_empty() {
        return Err(format!("{} is empty", cli.annotations.display()));
    }

    // --- Encode once, decide many times -------------------------------
    let mut scored: Vec<Scored> = Vec::new();
    let mut per_utt = Vec::new();
    let mut embed_micros: Vec<u128> = Vec::new();

    for anno in &annotations {
        let segments = filter.segment_text(&anno.text);
        if segments.is_empty() {
            continue;
        }

        let mut embeddings = Vec::with_capacity(segments.len());
        for s in &segments {
            let t0 = Instant::now();
            let e = filter
                .embed(&s.surface)
                .await
                .map_err(|e| format!("embedding {:?}: {}", s.surface, e.message))?;
            embed_micros.push(t0.elapsed().as_micros());
            embeddings.push(e);
        }

        let gold: Vec<Span> = anno
            .fillers
            .iter()
            .map(|f| Span {
                start: f.start,
                end: f.end,
            })
            .collect();

        for (s, emb) in segments.iter().zip(&embeddings) {
            let span = Span {
                start: s.cp_start,
                end: s.cp_end,
            };
            scored.push(Scored {
                surface: s.surface.clone(),
                sim: filter.max_filler_sim(emb),
                is_gold: gold.iter().any(|g| overlaps(&span, g)),
            });
        }
        per_utt.push((segments, embeddings, gold));
    }

    // --- Score distribution -------------------------------------------
    let mut pos: Vec<f32> = scored.iter().filter(|s| s.is_gold).map(|s| s.sim).collect();
    let mut neg: Vec<f32> = scored.iter().filter(|s| !s.is_gold).map(|s| s.sim).collect();
    pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    neg.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("== {label} / {} ==", cli.lang);
    println!("model_dir       : {}", cli.model_dir.display());
    println!("load            : {load_ms:.0} ms");
    println!(
        "segments        : {} ({} gold filler / {} other)",
        scored.len(),
        pos.len(),
        neg.len()
    );
    if pos.is_empty() || neg.is_empty() {
        return Err("need both positive and negative segments to calibrate".into());
    }
    println!(
        "gold  filler sim: min {:.4}  p50 {:.4}  max {:.4}",
        pos[0],
        percentile(&pos, 0.5),
        pos[pos.len() - 1]
    );
    println!(
        "other  token sim: min {:.4}  p50 {:.4}  max {:.4}",
        neg[0],
        percentile(&neg, 0.5),
        neg[neg.len() - 1]
    );
    let margin = pos[0] - neg[neg.len() - 1];
    println!("separation margin: {margin:+.4} (min(gold) − max(other))");

    // The non-filler segments that score closest to a filler
    // prototype: these are what a too-low threshold would delete.
    let mut confusable: Vec<&Scored> = scored.iter().filter(|s| !s.is_gold).collect();
    confusable.sort_by(|a, b| b.sim.partial_cmp(&a.sim).unwrap());
    let top: Vec<String> = confusable
        .iter()
        .take(5)
        .map(|s| format!("{:?}={:.3}", s.surface, s.sim))
        .collect();
    println!("most confusable : {}", top.join("  "));

    let mut lat = embed_micros.clone();
    lat.sort_unstable();
    if !lat.is_empty() {
        println!(
            "embed latency   : p50 {:.2} ms  p95 {:.2} ms  ({} calls)",
            lat[lat.len() / 2] as f64 / 1000.0,
            lat[((lat.len() as f64 * 0.95) as usize).min(lat.len() - 1)] as f64 / 1000.0,
            lat.len()
        );
    }

    // --- Threshold sweep ----------------------------------------------
    println!("\nthreshold   P       R       F1      TP  FP  FN");
    let mut sweep = BTreeMap::new();
    let mut best: Option<(f32, f64)> = None;

    let mut t = cli.sweep_from;
    while t <= cli.sweep_to + 1e-6 {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;

        for (segments, embeddings, gold) in &per_utt {
            let flags = filter.removal_flags(segments, embeddings, t);
            let predicted: Vec<Span> = segments
                .iter()
                .zip(&flags)
                .filter(|(_, &f)| f)
                .map(|(s, _)| Span {
                    start: s.cp_start,
                    end: s.cp_end,
                })
                .collect();
            let stats = strict_f1(&predicted, gold);
            tp += stats.tp;
            fp += stats.fp;
            fn_ += stats.fn_;
        }

        let p = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let r = if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        let f1 = if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        };

        let key = format!("{t:.2}");
        println!("  {key}      {p:.3}   {r:.3}   {f1:.3}   {tp:3} {fp:3} {fn_:3}");
        sweep.insert(key, serde_json::json!({"p": p, "r": r, "f1": f1}));

        // Strictly-greater keeps the *lowest* threshold among ties,
        // which is the conservative choice: it maximises recall
        // headroom for filler variants outside the gold set.
        // `Option::is_none_or` would read better but is only stable
        // since 1.82, above the crate's declared MSRV of 1.78.
        if best.map(|(_, bf1)| f1 > bf1).unwrap_or(true) {
            best = Some((t, f1));
        }
        t += cli.sweep_step;
    }

    let (best_t, best_f1) = best.unwrap();
    println!("\nbest threshold  : {best_t:.2} (F1 {best_f1:.3})");
    println!("current default : {baseline_threshold:.2}");

    if let Some(path) = &cli.output {
        let report = serde_json::json!({
            "label": label,
            "lang": cli.lang,
            "model_dir": cli.model_dir.display().to_string(),
            "load_ms": load_ms,
            "segments": {"total": scored.len(), "gold": pos.len(), "other": neg.len()},
            "similarity": {
                "gold": {"min": pos[0], "p50": percentile(&pos, 0.5), "max": pos[pos.len()-1]},
                "other": {"min": neg[0], "p50": percentile(&neg, 0.5), "max": neg[neg.len()-1]},
                "margin": margin,
            },
            "embed_latency_ms": {
                "p50": lat.get(lat.len()/2).map(|v| *v as f64/1000.0),
                "p95": lat.get((lat.len() as f64*0.95) as usize).map(|v| *v as f64/1000.0),
            },
            "sweep": sweep,
            "best": {"threshold": best_t, "f1": best_f1},
            "baseline_threshold": baseline_threshold,
        });
        std::fs::write(path, serde_json::to_string_pretty(&report).unwrap())
            .map_err(|e| format!("writing {}: {e}", path.display()))?;
        println!("wrote {}", path.display());
    }

    Ok(())
}
