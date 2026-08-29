//! Offline RTF bench for euhadra's shipping ASR models vs a shared
//! FLEURS manifest. Pair with `scripts/bench_hayamimi_asr.py` for the
//! head-to-head against hayamimi's sherpa-onnx catalog.
//!
//! Usage:
//!   cargo run --release --features onnx --example bench_shipping_asr -- \
//!     --kind canary --model-dir vendor/canary_en --language en \
//!     --manifest data/fleurs_subset/en/manifest.tsv \
//!     --audio-root data/fleurs_subset \
//!     --json-out /tmp/euhadra_en.json

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use euhadra::canary::{CanaryAdapter, CanaryConfig};
use euhadra::dolphin::{DolphinAdapter, DolphinConfig};
use euhadra::paraformer::ParaformerAdapter;
use euhadra::parakeet::ParakeetAdapter;
use euhadra::whisper_local::read_wav;
use serde::Serialize;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Kind {
    Canary,
    Parakeet,
    Paraformer,
    Dolphin,
}

#[derive(Parser)]
struct Cli {
    #[arg(long, value_enum)]
    kind: Kind,

    #[arg(long)]
    model_dir: PathBuf,

    /// ISO language for Canary (`en` / `es`). Ignored by other kinds.
    #[arg(long, default_value = "en")]
    language: String,

    #[arg(long)]
    manifest: PathBuf,

    #[arg(long)]
    audio_root: PathBuf,

    #[arg(long)]
    json_out: Option<PathBuf>,

    /// Warm-up passes before timing (default 1).
    #[arg(long, default_value_t = 1)]
    warmup: usize,
}

trait SyncAsr {
    fn name(&self) -> &str;
    fn transcribe(&self, samples: &[f32]) -> Result<String, String>;
}

struct CanaryWrap(CanaryAdapter);
impl SyncAsr for CanaryWrap {
    fn name(&self) -> &str {
        "euhadra/canary-180m-flash-int8"
    }
    fn transcribe(&self, samples: &[f32]) -> Result<String, String> {
        self.0
            .transcribe_samples(samples)
            .map_err(|e| e.to_string())
    }
}

struct ParakeetWrap(ParakeetAdapter);
impl SyncAsr for ParakeetWrap {
    fn name(&self) -> &str {
        "euhadra/parakeet-tdt_ctc-0.6b-ja"
    }
    fn transcribe(&self, samples: &[f32]) -> Result<String, String> {
        // ParakeetAdapter only exposes the async AsrAdapter path.
        let handle = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        let chunk = euhadra::types::AudioChunk {
            samples: samples.to_vec(),
            sample_rate: 16000,
            channels: 1,
        };
        handle
            .block_on(async {
                use euhadra::traits::AsrAdapter;
                self.0.transcribe(&[chunk]).await
            })
            .map(|t| t.text)
            .map_err(|e| e.to_string())
    }
}

struct ParaformerWrap(ParaformerAdapter);
impl SyncAsr for ParaformerWrap {
    fn name(&self) -> &str {
        "euhadra/paraformer-large"
    }
    fn transcribe(&self, samples: &[f32]) -> Result<String, String> {
        self.0
            .transcribe_samples(samples)
            .map_err(|e| e.to_string())
    }
}

struct DolphinWrap(DolphinAdapter);
impl SyncAsr for DolphinWrap {
    fn name(&self) -> &str {
        "euhadra/dolphin-small-ctc-int8"
    }
    fn transcribe(&self, samples: &[f32]) -> Result<String, String> {
        self.0
            .transcribe_samples(samples)
            .map_err(|e| e.to_string())
    }
}

#[derive(Serialize)]
struct Row {
    id: String,
    audio_s: f64,
    asr_s: f64,
    rtf: f64,
    hyp: String,
}

#[derive(Serialize)]
struct Report {
    side: &'static str,
    model: String,
    kind: String,
    language: String,
    threads_note: &'static str,
    n: usize,
    total_audio_s: f64,
    total_asr_s: f64,
    mean_rtf: f64,
    p50_asr_ms: f64,
    p95_asr_ms: f64,
    rows: Vec<Row>,
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ms.len() as f64 - 1.0) * p).round() as usize;
    sorted_ms[idx.min(sorted_ms.len() - 1)]
}

fn load_asr(cli: &Cli) -> Result<Box<dyn SyncAsr>, String> {
    match cli.kind {
        Kind::Canary => {
            let cfg = CanaryConfig::istupakov_default();
            let a = CanaryAdapter::load_with_config(&cli.model_dir, cfg)
                .map_err(|e| e.to_string())?
                .with_language(&cli.language);
            Ok(Box::new(CanaryWrap(a)))
        }
        Kind::Parakeet => {
            let a = ParakeetAdapter::load(&cli.model_dir).map_err(|e| e.to_string())?;
            Ok(Box::new(ParakeetWrap(a)))
        }
        Kind::Paraformer => {
            let a = ParaformerAdapter::load(&cli.model_dir).map_err(|e| e.to_string())?;
            Ok(Box::new(ParaformerWrap(a)))
        }
        Kind::Dolphin => {
            let a = DolphinAdapter::load_with_config(
                &cli.model_dir,
                DolphinConfig {
                    model_file: "model.int8.onnx".into(),
                },
            )
            .map_err(|e| e.to_string())?;
            Ok(Box::new(DolphinWrap(a)))
        }
    }
}

fn main() {
    let cli = Cli::parse();
    println!("loading {:?} from {}...", cli.kind, cli.model_dir.display());
    let t0 = Instant::now();
    let asr = load_asr(&cli).expect("load");
    println!("loaded {} in {:.1}s", asr.name(), t0.elapsed().as_secs_f64());

    let manifest = std::fs::read_to_string(&cli.manifest).expect("manifest");
    let rows: Vec<(String, PathBuf)> = manifest
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split('\t');
            let id = it.next().unwrap().to_string();
            let rel = it.next().unwrap();
            (id, cli.audio_root.join(rel))
        })
        .collect();

    if rows.is_empty() {
        panic!("empty manifest");
    }

    let warm = read_wav(&rows[0].1).expect("warm wav");
    for _ in 0..cli.warmup {
        let _ = asr.transcribe(&warm.samples).expect("warmup");
    }

    let mut out_rows = Vec::new();
    let mut total_audio = 0.0;
    let mut total_asr = 0.0;
    let mut asr_ms = Vec::new();

    for (id, path) in &rows {
        let chunk = read_wav(path).expect("wav");
        let dur = chunk.samples.len() as f64 / 16000.0;
        let t = Instant::now();
        let hyp = asr.transcribe(&chunk.samples).expect("transcribe");
        let asr_s = t.elapsed().as_secs_f64();
        total_audio += dur;
        total_asr += asr_s;
        asr_ms.push(asr_s * 1000.0);
        let rtf = asr_s / dur.max(1e-9);
        println!(
            "{id}: audio={dur:.2}s asr={:.0}ms rtf={rtf:.3} hyp={:?}",
            asr_s * 1000.0,
            hyp.chars().take(60).collect::<String>()
        );
        out_rows.push(Row {
            id: id.clone(),
            audio_s: dur,
            asr_s,
            rtf,
            hyp,
        });
    }

    asr_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let report = Report {
        side: "euhadra",
        model: asr.name().to_string(),
        kind: format!("{:?}", cli.kind).to_lowercase(),
        language: cli.language.clone(),
        threads_note: "ort default (adapter-controlled)",
        n: out_rows.len(),
        total_audio_s: total_audio,
        total_asr_s: total_asr,
        mean_rtf: total_asr / total_audio.max(1e-9),
        p50_asr_ms: percentile(&asr_ms, 0.50),
        p95_asr_ms: percentile(&asr_ms, 0.95),
        rows: out_rows,
    };

    println!(
        "SUMMARY model={} n={} audio={:.1}s asr={:.1}s mean_rtf={:.3} p50={:.0}ms p95={:.0}ms",
        report.model,
        report.n,
        report.total_audio_s,
        report.total_asr_s,
        report.mean_rtf,
        report.p50_asr_ms,
        report.p95_asr_ms
    );

    if let Some(path) = &cli.json_out {
        let s = serde_json::to_string_pretty(&report).expect("json");
        std::fs::write(path, s).expect("write json");
        println!("wrote {}", path.display());
    }
}
