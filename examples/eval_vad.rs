//! ΔWER of voice activity detection, and the silence-hallucination probe.
//!
//! #133's acceptance criterion is not the detector's F1. It is whether
//! putting a detector in front of the ASR adapter improves the transcript
//! of a recording that contains silence, and whether hallucinated text
//! survives afterwards. Both are measured here against the FLEURS subset
//! already in the tree, with silence added artificially.
//!
//! **Synthetic silence measures an upper bound.** Digital zeros and
//! shaped noise are not what a room sounds like, and a real recording's
//! background is exactly what makes an energy detector fail. Read the
//! numbers as "no worse than this", not as field performance.
//!
//! ```text
//! cargo run --release --features onnx,vad --example eval_vad -- \
//!     --parakeet-en-dir vendor/parakeet_v3 \
//!     --parakeet-ja-dir vendor/parakeet_ja \
//!     --out docs/benchmarks/vad_delta_wer.json
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use euhadra::eval::metrics::{cer_lenient, wer_lenient};
use euhadra::parakeet::ParakeetAdapter;
use euhadra::prelude::*;
use euhadra::vad::{EarshotVad, EnergyVad, SegmenterConfig, VadBackend};
use euhadra::whisper_local::read_wav;

#[derive(Parser, Debug)]
#[command(about = "ΔWER with and without voice activity detection")]
struct Cli {
    /// Root of the FLEURS subset (containing `<lang>/manifest.tsv`).
    #[arg(long, default_value = "data/fleurs_subset")]
    data_dir: PathBuf,

    /// `parakeet-tdt-0.6b-v3` bundle, used for `en`.
    #[arg(long)]
    parakeet_en_dir: Option<PathBuf>,

    /// `nvidia/parakeet-tdt_ctc-0.6b-ja` bundle, used for `ja`.
    #[arg(long)]
    parakeet_ja_dir: Option<PathBuf>,

    #[arg(long, value_delimiter = ',', default_value = "en,ja")]
    langs: Vec<String>,

    /// Cap the number of utterances per language. 0 means all of them.
    #[arg(long, default_value_t = 0)]
    limit: usize,

    /// Silence prepended to every utterance, in seconds.
    #[arg(long, default_value_t = 5.0)]
    lead_silence: f32,

    /// Silence appended to every utterance, in seconds.
    #[arg(long, default_value_t = 5.0)]
    trail_silence: f32,

    /// Levels, in dBFS, at which to synthesise the added silence. −100 is
    /// digital zero. Quieter than about −45 is unrealistically clean; a
    /// real room sits nearer −50 to −40.
    #[arg(long, value_delimiter = ',', default_value = "-100,-45")]
    noise_db: Vec<f32>,

    /// Where to write the results as JSON.
    #[arg(long)]
    out: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

struct Row {
    id: String,
    audio_path: PathBuf,
    reference: String,
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
                id: cols[0].to_string(),
                audio_path: data_dir.join(cols[1]),
                reference: cols[2].to_string(),
            })
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Audio synthesis
// ---------------------------------------------------------------------------

/// Deterministic noise at `db` dBFS RMS. `-100` and below is returned as
/// digital zero, so "silence" and "quiet noise" are one code path.
fn silence(samples: usize, db: f32, seed: u64) -> Vec<f32> {
    if db <= -100.0 {
        return vec![0.0; samples];
    }
    // Uniform noise in [-a, a] has RMS a/sqrt(3).
    let amplitude = 10f32.powf(db / 20.0) * 3f32.sqrt();
    let mut state = seed | 1;
    (0..samples)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / 8_388_608.0 - 1.0;
            unit * amplitude
        })
        .collect()
}

/// The utterance with silence either side of it.
fn pad(chunk: &AudioChunk, lead: f32, trail: f32, db: f32, seed: u64) -> AudioChunk {
    let rate = chunk.sample_rate as f32;
    let mut samples = silence((lead * rate) as usize, db, seed);
    samples.extend_from_slice(&chunk.samples);
    samples.extend(silence((trail * rate) as usize, db, seed ^ 0xABCD_EF01));
    AudioChunk {
        samples,
        sample_rate: chunk.sample_rate,
        channels: chunk.channels,
    }
}

// ---------------------------------------------------------------------------
// Conditions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Detector {
    None,
    Energy,
    Earshot,
}

impl Detector {
    fn backend(&self) -> Option<Box<dyn VadBackend>> {
        match self {
            Detector::None => None,
            Detector::Energy => Some(Box::new(EnergyVad::new())),
            Detector::Earshot => Some(Box::new(EarshotVad::new())),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Detector::None => "none",
            Detector::Energy => "energy",
            Detector::Earshot => "earshot",
        }
    }
}

/// One row of the report.
#[derive(Debug, Default, Clone, serde::Serialize)]
struct Cell {
    /// WER for whitespace-delimited languages, CER otherwise.
    error_rate: f64,
    /// Error rate minus the clean-audio, no-detector run.
    delta: f64,
    /// Mean utterances found per recording. 1.0 is the ideal here: the
    /// FLEURS clips are single utterances, so anything above it is
    /// over-segmentation.
    segments: f64,
    /// Recordings whose transcript grew by more than a quarter against
    /// the clean run — the shape hallucinated filler takes.
    inflated: usize,
    seconds: f64,
}

fn main() {
    let cli = Cli::parse();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let mut report: BTreeMap<String, BTreeMap<String, Cell>> = BTreeMap::new();
    let mut probes: BTreeMap<String, String> = BTreeMap::new();

    for lang in &cli.langs {
        let model_dir = match lang.as_str() {
            "en" => cli.parakeet_en_dir.clone(),
            "ja" => cli.parakeet_ja_dir.clone(),
            other => {
                eprintln!("[skip] {other}: no model directory wired for this language");
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
        eprintln!("[{lang}] {} utterances, model {}", rows.len(), model_dir.display());

        let asr = std::sync::Arc::new(
            ParakeetAdapter::load(&model_dir)
                .unwrap_or_else(|e| panic!("load {}: {e}", model_dir.display())),
        );
        let by_chars = matches!(lang.as_str(), "ja" | "zh");

        let audio: Vec<(Row, AudioChunk)> = rows
            .into_iter()
            .map(|row| {
                let chunk = read_wav(&row.audio_path)
                    .unwrap_or_else(|e| panic!("read {}: {e}", row.audio_path.display()));
                (row, chunk)
            })
            .collect();

        // ── Reference: clean audio, no detector ─────────────────────────
        let clean = runtime.block_on(measure(
            &asr,
            &audio,
            |_, chunk| chunk.clone(),
            Detector::None,
            FinalPass::WholeUtterance,
            by_chars,
            None,
        ));
        eprintln!(
            "[{lang}] clean / no detector: {:.4} ({:.1}s)",
            clean.error_rate, clean.seconds
        );
        report
            .entry(lang.clone())
            .or_default()
            .insert("clean|none|whole".into(), clean.clone());

        // ── The hallucination probe: what the model says to silence ─────
        for &db in &cli.noise_db {
            let rate = audio[0].1.sample_rate;
            let seconds = cli.lead_silence + cli.trail_silence;
            let only = AudioChunk {
                samples: silence((seconds * rate as f32) as usize, db, 0x5EED),
                sample_rate: rate,
                channels: 1,
            };
            let text = runtime
                .block_on(asr.transcribe(std::slice::from_ref(&only)))
                .map(|t| t.text.trim().to_string())
                .unwrap_or_else(|e| format!("<error: {e}>"));
            eprintln!("[{lang}] {seconds:.0}s of {db} dBFS alone → {text:?}");
            probes.insert(format!("{lang}|{db}"), text);
        }

        // ── Every (noise, detector, policy) combination ─────────────────
        for &db in &cli.noise_db {
            let lead = cli.lead_silence;
            let trail = cli.trail_silence;
            let padder = move |index: usize, chunk: &AudioChunk| {
                pad(chunk, lead, trail, db, 0x1234_5678 ^ index as u64)
            };

            for detector in [Detector::None, Detector::Energy, Detector::Earshot] {
                let policies: &[FinalPass] = if detector == Detector::None {
                    &[FinalPass::WholeUtterance]
                } else {
                    &[FinalPass::SpeechOnly, FinalPass::JoinSegments]
                };
                for &policy in policies {
                    let mut cell = runtime.block_on(measure(
                        &asr,
                        &audio,
                        padder,
                        detector,
                        policy,
                        by_chars,
                        Some(&clean),
                    ));
                    cell.delta = cell.error_rate - clean.error_rate;
                    let key = format!("{db}|{}|{}", detector.label(), policy_label(policy));
                    eprintln!(
                        "[{lang}] {key}: {:.4} (Δ{:+.4}) segments {:.2} inflated {} ({:.1}s)",
                        cell.error_rate, cell.delta, cell.segments, cell.inflated, cell.seconds
                    );
                    report.entry(lang.clone()).or_default().insert(key, cell);
                }
            }
        }
    }

    let json = serde_json::json!({
        "note": "Synthetic silence measures an upper bound; a real room's \
                 background noise is what makes a level detector fail.",
        "lead_silence_s": cli.lead_silence,
        "trail_silence_s": cli.trail_silence,
        "silence_only_transcripts": probes,
        "results": report,
    });
    let rendered = serde_json::to_string_pretty(&json).expect("serialise report");
    if let Some(path) = &cli.out {
        std::fs::write(path, format!("{rendered}\n")).unwrap_or_else(|e| panic!("write out: {e}"));
        eprintln!("[done] wrote {}", path.display());
    } else {
        println!("{rendered}");
    }
}

fn policy_label(policy: FinalPass) -> &'static str {
    match policy {
        FinalPass::SpeechOnly => "speech-only",
        FinalPass::WholeUtterance => "whole",
        FinalPass::JoinSegments => "join",
        _ => "unknown",
    }
}

/// Run every utterance through one configuration and average the result.
async fn measure<F>(
    asr: &std::sync::Arc<ParakeetAdapter>,
    audio: &[(Row, AudioChunk)],
    prepare: F,
    detector: Detector,
    policy: FinalPass,
    by_chars: bool,
    clean: Option<&Cell>,
) -> Cell
where
    F: Fn(usize, &AudioChunk) -> AudioChunk,
{
    // `Pipeline` needs to own its adapter, so the shared `ParakeetAdapter`
    // is wrapped rather than cloned — loading the 2.4 GB bundle once per
    // configuration would dominate the runtime.
    struct Shared(std::sync::Arc<ParakeetAdapter>);
    #[async_trait::async_trait]
    impl AsrAdapter for Shared {
        async fn transcribe(&self, chunks: &[AudioChunk]) -> Result<Transcript, AsrError> {
            self.0.transcribe(chunks).await
        }
    }

    let mut builder = PipelineBuilder::new()
        .asr(Shared(std::sync::Arc::clone(asr)))
        .final_pass(policy)
        .segmenter_config(SegmenterConfig::default());
    if let Some(backend) = detector.backend() {
        builder = builder.vad(BoxedVad(backend));
    }
    let pipeline = builder.build().expect("pipeline");

    let started = Instant::now();
    let mut total = 0.0;
    let mut segments = 0.0;
    let mut inflated = 0;
    for (index, (row, chunk)) in audio.iter().enumerate() {
        let prepared = prepare(index, chunk);
        let result = pipeline.transcribe(std::slice::from_ref(&prepared)).await;
        let (text, found) = match result {
            Ok(result) => (
                result.text().to_string(),
                result.diagnostics.speech_segments.len(),
            ),
            // NoSpeech is a real outcome, not a crash: the detector found
            // nothing. Scored as an empty transcript so it costs the full
            // error rate rather than being quietly dropped.
            Err(e) => {
                eprintln!("  [{}] {e}", row.id);
                (String::new(), 0)
            }
        };
        total += if by_chars {
            cer_lenient(&row.reference, &text)
        } else {
            wer_lenient(&row.reference, &text)
        };
        segments += found as f64;
        if let Some(clean) = clean {
            let _ = clean;
            if text.chars().count() as f64 > row.reference.chars().count() as f64 * 1.25 {
                inflated += 1;
            }
        }
    }

    let n = audio.len().max(1) as f64;
    Cell {
        error_rate: total / n,
        delta: 0.0,
        segments: segments / n,
        inflated,
        seconds: started.elapsed().as_secs_f64(),
    }
}

/// `PipelineBuilder::vad` takes an owned backend; the runner picks one at
/// runtime and so holds a `Box<dyn VadBackend>`.
struct BoxedVad(Box<dyn VadBackend>);

impl VadBackend for BoxedVad {
    fn frame_size(&self) -> usize {
        self.0.frame_size()
    }
    fn required_sample_rate(&self) -> Option<u32> {
        self.0.required_sample_rate()
    }
    fn start(&self) -> Box<dyn euhadra::vad::VadStream> {
        self.0.start()
    }
}
