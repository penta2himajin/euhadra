//! Endpoint latency on the live VAD path (#148).
//!
//! Measures **segment close → partial / final**, not the L1 full-file
//! ASR wall clock. Feed the same FLEURS subset the other evaluate jobs
//! use; VAD + Segmenter decide the utterances.
//!
//! ```text
//! cargo run --release --features onnx,vad,testing --example eval_endpoint -- \
//!     --canary-en-dir vendor/canary_en \
//!     --parakeet-ja-dir vendor/parakeet_ja \
//!     --baseline docs/benchmarks/ci_baseline_endpoint.json
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use clap::Parser;
use euhadra::canary::decoder::PrefixFormat;
use euhadra::canary::{CanaryAdapter, CanaryConfig};
use euhadra::eval::baseline::{
    check_endpoint_language, EndpointBaseline, EndpointLanguageBaseline, EndpointTolerances,
    LatencyRecord, Verdict,
};
use euhadra::eval::latency::Samples;
use euhadra::parakeet::ParakeetAdapter;
use euhadra::prelude::*;
use euhadra::vad::EarshotVad;
use euhadra::whisper_local::read_wav;

#[derive(Parser, Debug)]
#[command(about = "Endpoint latency: segment close → partial / final")]
struct Cli {
    /// Root of the FLEURS subset (containing `<lang>/manifest.tsv`).
    #[arg(long, default_value = "data/fleurs_subset")]
    data_dir: PathBuf,

    /// `canary-180m-flash` bundle for `en` (shipped configuration).
    #[arg(long)]
    canary_en_dir: Option<PathBuf>,

    /// `parakeet-tdt-0.6b-v3` alternative for `en`.
    #[arg(long)]
    parakeet_en_dir: Option<PathBuf>,

    /// `nvidia/parakeet-tdt_ctc-0.6b-ja` bundle for `ja`.
    #[arg(long)]
    parakeet_ja_dir: Option<PathBuf>,

    /// `canary-180m-flash` for `es` (same multilingual checkpoint as en).
    /// Falls back to `--canary-en-dir` when omitted.
    #[arg(long)]
    canary_es_dir: Option<PathBuf>,

    #[arg(long, value_delimiter = ',', default_value = "en,ja")]
    langs: Vec<String>,

    /// Cap utterances per language. 0 means all of them.
    #[arg(long, default_value_t = 0)]
    limit: usize,

    /// Final-pass policy. Default matches the library default.
    #[arg(long, default_value = "speech-only")]
    policy: String,

    /// Compare against this baseline (relative gate). Omit to measure only.
    #[arg(long)]
    baseline: Option<PathBuf>,

    /// Write a fresh baseline JSON (for intentional updates).
    #[arg(long)]
    write_baseline: Option<PathBuf>,
}

struct Row {
    audio_path: PathBuf,
}

fn load_manifest(data_dir: &Path, lang: &str) -> std::io::Result<Vec<Row>> {
    let raw = std::fs::read_to_string(data_dir.join(lang).join("manifest.tsv"))?;
    Ok(raw
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            let cols: Vec<&str> = line.splitn(3, '\t').collect();
            (cols.len() == 3).then(|| Row {
                audio_path: data_dir.join(cols[1]),
            })
        })
        .collect())
}

fn load_adapter(dir: &Path, lang: &str, canary: bool) -> Arc<dyn AsrAdapter> {
    if canary {
        let cfg = CanaryConfig::istupakov_default().with_int8_weights();
        let cfg = CanaryConfig {
            prefix_format: PrefixFormat::NemoCanary2,
            ..cfg
        };
        let adapter = CanaryAdapter::load_with_config(dir, cfg)
            .unwrap_or_else(|e| panic!("load canary from {}: {e}", dir.display()))
            .with_language(lang);
        Arc::new(adapter)
    } else {
        Arc::new(
            ParakeetAdapter::load(dir)
                .unwrap_or_else(|e| panic!("load parakeet from {}: {e}", dir.display())),
        )
    }
}

fn parse_policy(s: &str) -> FinalPass {
    match s {
        "speech-only" => FinalPass::SpeechOnly,
        "whole" => FinalPass::WholeUtterance,
        "join" => FinalPass::JoinSegments,
        other => panic!("unknown --policy {other:?} (speech-only|whole|join)"),
    }
}

fn main() {
    let cli = Cli::parse();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let policy = parse_policy(&cli.policy);
    let mut languages: BTreeMap<String, EndpointLanguageBaseline> = BTreeMap::new();
    let mut model_labels: Vec<String> = Vec::new();

    for lang in &cli.langs {
        let use_canary = match lang.as_str() {
            "en" => cli.canary_en_dir.is_some(),
            "es" => cli.canary_es_dir.is_some() || cli.canary_en_dir.is_some(),
            _ => false,
        };
        let model_dir = match lang.as_str() {
            "en" if use_canary => cli.canary_en_dir.clone(),
            "en" => cli.parakeet_en_dir.clone(),
            "ja" => cli.parakeet_ja_dir.clone(),
            "es" => cli
                .canary_es_dir
                .clone()
                .or_else(|| cli.canary_en_dir.clone()),
            other => {
                eprintln!("[skip] {other}: no model directory wired");
                continue;
            }
        };
        let Some(model_dir) = model_dir else {
            eprintln!("[skip] {lang}: model directory not supplied");
            continue;
        };

        let mut rows = load_manifest(&cli.data_dir, lang)
            .unwrap_or_else(|e| panic!("manifest for {lang}: {e}"));
        if cli.limit > 0 {
            rows.truncate(cli.limit);
        }
        let model_name = if use_canary {
            "canary-180m-flash-int8"
        } else {
            "parakeet"
        };
        eprintln!(
            "[{lang}] {} utterances, {model_name} @ {}",
            rows.len(),
            model_dir.display()
        );
        model_labels.push(format!("{lang}={model_name}"));

        let asr = load_adapter(&model_dir, lang, use_canary);
        let measured = runtime.block_on(measure_lang(asr, &rows, policy));
        eprintln!(
            "[{lang}] segments={}  partial p50={:.0}ms p95={:.0}ms  \
             final p50={:.0}ms p95={:.0}ms  rtf={}",
            measured.segments,
            measured.endpoint_to_partial_ms.p50,
            measured.endpoint_to_partial_ms.p95,
            measured.endpoint_to_final_ms.p50,
            measured.endpoint_to_final_ms.p95,
            measured
                .segment_rtf
                .map(|r| format!("{r:.3}"))
                .unwrap_or_else(|| "n/a".into())
        );
        languages.insert(lang.clone(), measured);
    }

    if languages.is_empty() {
        eprintln!("[FAIL] no languages measured");
        std::process::exit(1);
    }

    let generated = format!(
        "epoch:{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let fresh = EndpointBaseline {
        schema_version: 1,
        generated,
        asr_model: model_labels.join(" / "),
        languages: languages.clone(),
        tolerances: EndpointTolerances::default(),
    };

    if let Some(path) = &cli.write_baseline {
        fresh
            .save(path)
            .unwrap_or_else(|e| panic!("write baseline {}: {e}", path.display()));
        eprintln!("[done] wrote baseline {}", path.display());
    }

    if let Some(path) = &cli.baseline {
        let baseline = EndpointBaseline::load(path)
            .unwrap_or_else(|e| panic!("load baseline {}: {e}", path.display()));
        let mut failed = false;
        for (lang, measured) in &languages {
            let Some(entry) = baseline.languages.get(lang) else {
                eprintln!("[WARN] {lang}: no baseline entry");
                continue;
            };
            let results = check_endpoint_language(measured, entry, &baseline.tolerances);
            for (metric, verdict) in results {
                match verdict {
                    Verdict::Pass => {}
                    Verdict::Warn(msg) => eprintln!("[WARN] {lang}/{metric}: {msg}"),
                    Verdict::Fail(msg) => {
                        eprintln!("[FAIL] {lang}/{metric}: {msg}");
                        failed = true;
                    }
                }
            }
        }
        if failed {
            std::process::exit(1);
        }
        eprintln!("[ok] endpoint baseline gate passed");
    }
}

async fn measure_lang(
    asr: Arc<dyn AsrAdapter>,
    rows: &[Row],
    policy: FinalPass,
) -> EndpointLanguageBaseline {
    struct Shared(Arc<dyn AsrAdapter>);
    #[async_trait]
    impl AsrAdapter for Shared {
        async fn transcribe(&self, chunks: &[AudioChunk]) -> Result<Transcript, AsrError> {
            self.0.transcribe(chunks).await
        }
    }

    let mut partial_samples = Samples::new();
    let mut final_samples = Samples::new();
    let mut total_asr_secs = 0.0_f64;
    let mut total_speech_secs = 0.0_f64;
    let mut segment_count = 0usize;
    let mut utterance_count = 0usize;

    for row in rows {
        let audio = read_wav(&row.audio_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", row.audio_path.display()));

        let pipeline = Pipeline::builder()
            .asr(Shared(Arc::clone(&asr)))
            .vad(EarshotVad::new())
            .final_pass(policy)
            .build()
            .expect("pipeline");

        let mut session = pipeline.session();
        // Keep the partial receiver alive so per-utterance ASR (and thus
        // t0) runs; drain so the bounded channel cannot drop timings.
        let mut partials =
            std::mem::replace(&mut session.partials, tokio::sync::mpsc::channel(1).1);
        let drain = tokio::spawn(async move {
            let mut seen = Vec::new();
            while let Some(partial) = partials.recv().await {
                seen.push(partial);
            }
            seen
        });

        // Feed in capture-sized slices so the live segmenter sees the
        // same chunked arrival pattern as a mic path.
        let rate = audio.sample_rate.max(1);
        let step = (rate as usize / 10).max(1);
        for slice in audio.samples.chunks(step) {
            session
                .audio
                .send(AudioChunk {
                    samples: slice.to_vec(),
                    sample_rate: audio.sample_rate,
                    channels: audio.channels,
                })
                .await
                .expect("audio channel open");
        }

        let result = session
            .finish()
            .await
            .unwrap_or_else(|e| panic!("session: {e}"));
        let seen = drain.await.expect("partial drain");

        utterance_count += 1;
        for d in &result.diagnostics.endpoint_to_partial {
            partial_samples.record(*d);
            segment_count += 1;
        }
        if let Some(d) = result.diagnostics.endpoint_to_final {
            final_samples.record(d);
        }

        for partial in &seen {
            let Some(latency) = partial.endpoint_latency else {
                continue;
            };
            let Some(seg) = result.diagnostics.speech_segments.get(partial.index) else {
                continue;
            };
            total_speech_secs += seg.duration(rate).as_secs_f64();
            total_asr_secs += latency.as_secs_f64();
        }
    }

    let partial = partial_samples
        .summary()
        .unwrap_or_else(|| panic!("no endpoint_to_partial samples"));
    let final_sum = final_samples
        .summary()
        .unwrap_or_else(|| panic!("no endpoint_to_final samples"));
    let segment_rtf = if total_speech_secs > 0.0 {
        Some(round4(total_asr_secs / total_speech_secs))
    } else {
        None
    };

    EndpointLanguageBaseline {
        utterances: utterance_count,
        segments: segment_count,
        endpoint_to_partial_ms: LatencyRecord::from(partial),
        endpoint_to_final_ms: LatencyRecord::from(final_sum),
        segment_rtf,
    }
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}
