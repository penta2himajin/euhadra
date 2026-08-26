use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Duration;

use euhadra::emitters::ClipboardEmitter;
use euhadra::emitters::StdoutEmitter;
use euhadra::filter::{
    ChineseFillerFilter, JapaneseFillerFilter, SimpleFillerFilter, SpanishFillerFilter,
};
use euhadra::mic::{self, MicConfig};
use euhadra::pipeline::{FinalPass, Pipeline, PipelineBuilder, SessionResult};
use euhadra::processor::{
    BasicPunctuationRestorer, InverseTextNormalizer, SelfCorrectionDetector, SpokenFormNormalizer,
};
use euhadra::vad::{EarshotVad, SegmenterConfig};
use euhadra::whisper_local::{self, WhisperLocal};

#[derive(Parser)]
#[command(name = "euhadra", about = "Programmable voice input framework")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Transcribe an audio file through the full pipeline.
    Dictate {
        /// Path to WAV audio file (16-bit PCM).
        #[arg(short, long)]
        file: PathBuf,

        /// Path to whisper-cli binary.
        #[arg(long, default_value = "whisper-cli")]
        whisper_cli: PathBuf,

        /// Path to whisper GGML model file.
        #[arg(long)]
        model: PathBuf,

        /// Language hint (e.g. "en", "ja"). Omit for auto-detect.
        #[arg(short, long)]
        language: Option<String>,

        /// Skip filler removal entirely.
        #[arg(long, default_value_t = false)]
        no_filter: bool,

        /// Skip text processing (punctuation, self-correction detection).
        #[arg(long, default_value_t = false)]
        no_process: bool,

        /// Run the file through the live VAD session path (partials on stderr).
        ///
        /// Off by default for `dictate`: most WAVs are already trimmed.
        /// `record` enables VAD by default instead.
        #[arg(long, default_value_t = false)]
        vad: bool,

        /// Continuous silence (ms) that closes an utterance. Only with `--vad`.
        #[arg(long)]
        min_silence_ms: Option<u64>,

        /// Final transcript source when `--vad` is set.
        /// `speech-only` (default) | `whole` | `join`.
        #[arg(long, default_value = "speech-only")]
        final_pass: String,
    },

    /// Record from microphone and transcribe through the full pipeline.
    ///
    /// VAD + per-utterance partials are on by default (#147). Pass
    /// `--no-vad` for a single end-of-session transcript of the whole
    /// capture.
    Record {
        /// Path to whisper-cli binary.
        #[arg(long, default_value = "whisper-cli")]
        whisper_cli: PathBuf,

        /// Path to whisper GGML model file.
        #[arg(long)]
        model: PathBuf,

        /// Language hint (e.g. "en", "ja").
        #[arg(short, long)]
        language: Option<String>,

        /// Skip filler removal.
        #[arg(long, default_value_t = false)]
        no_filter: bool,

        /// Skip text processing.
        #[arg(long, default_value_t = false)]
        no_process: bool,

        /// Output to clipboard instead of stdout.
        #[arg(long, default_value_t = false)]
        clipboard: bool,

        /// Disable voice activity detection (whole capture → one final).
        #[arg(long, default_value_t = false)]
        no_vad: bool,

        /// Continuous silence (ms) that closes an utterance.
        /// Default matches [`SegmenterConfig`] (700). See
        /// `docs/benchmarks/endpointing_profiles.md`.
        #[arg(long)]
        min_silence_ms: Option<u64>,

        /// Final transcript source with VAD.
        /// `speech-only` (default) | `whole` | `join`.
        #[arg(long, default_value = "speech-only")]
        final_pass: String,
    },

    /// Transcribe a file with whisper only (no pipeline, no refinement).
    Transcribe {
        /// Path to WAV audio file.
        #[arg(short, long)]
        file: PathBuf,

        /// Path to whisper-cli binary.
        #[arg(long, default_value = "whisper-cli")]
        whisper_cli: PathBuf,

        /// Path to whisper GGML model file.
        #[arg(long)]
        model: PathBuf,

        /// Language hint.
        #[arg(short, long)]
        language: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "euhadra=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Transcribe {
            file,
            whisper_cli,
            model,
            language,
        } => {
            let text =
                whisper_local::transcribe_file(&whisper_cli, &model, &file, language.as_deref())
                    .await?;
            println!("{text}");
        }

        Commands::Dictate {
            file,
            whisper_cli,
            model,
            language,
            no_filter,
            no_process,
            vad,
            min_silence_ms,
            final_pass,
        } => {
            let audio =
                whisper_local::read_wav(&file).map_err(|e| format!("failed to read WAV: {e}"))?;

            let mut builder = Pipeline::builder();
            builder = attach_asr(builder, &whisper_cli, &model, language.as_deref());
            builder = attach_filters(builder, language.as_deref(), no_filter);
            builder = attach_processors(builder, language.as_deref(), no_process);
            builder = builder.emitter(StdoutEmitter);

            if vad {
                builder = attach_vad(builder, min_silence_ms, &final_pass)?;
                let pipeline = builder.build()?;
                let mut session = pipeline.session();
                let mut partials = std::mem::replace(
                    &mut session.partials,
                    tokio::sync::mpsc::channel(1).1,
                );
                let partial_task = tokio::spawn(async move {
                    while let Some(p) = partials.recv().await {
                        print_partial(&p);
                    }
                });
                session
                    .audio
                    .send(audio)
                    .await
                    .map_err(|_| "audio channel closed")?;
                let result = session.finish().await?;
                let _ = partial_task.await;
                report_emit(&result);
            } else {
                let pipeline = builder.build()?;
                let result = pipeline.transcribe(std::slice::from_ref(&audio)).await?;
                report_emit(&result);
            }
        }

        Commands::Record {
            whisper_cli,
            model,
            language,
            no_filter,
            no_process,
            clipboard,
            no_vad,
            min_silence_ms,
            final_pass,
        } => {
            let use_vad = !no_vad;
            if use_vad {
                eprintln!(
                    "Recording with VAD (utterance partials on stderr). Press Ctrl+C to stop."
                );
            } else {
                eprintln!("Recording from microphone... Press Ctrl+C to stop.");
            }

            let mut builder = Pipeline::builder();
            builder = attach_asr(builder, &whisper_cli, &model, language.as_deref());
            builder = attach_filters(builder, language.as_deref(), no_filter);
            builder = attach_processors(builder, language.as_deref(), no_process);

            if clipboard {
                builder = builder.emitter(ClipboardEmitter::new());
            } else {
                builder = builder.emitter(StdoutEmitter);
            }

            if use_vad {
                builder = attach_vad(builder, min_silence_ms, &final_pass)?;
            }

            let pipeline = builder.build()?;

            let (mut mic_rx, _mic_guard) =
                mic::record(MicConfig::default()).map_err(|e| format!("mic: {e}"))?;

            let mut session = pipeline.session();

            let partial_task = if use_vad {
                let mut partials = std::mem::replace(
                    &mut session.partials,
                    tokio::sync::mpsc::channel(1).1,
                );
                Some(tokio::spawn(async move {
                    while let Some(p) = partials.recv().await {
                        print_partial(&p);
                    }
                }))
            } else {
                None
            };

            let audio_tx = session.audio.clone();
            let bridge = tokio::spawn(async move {
                while let Some(chunk) = mic_rx.recv().await {
                    if audio_tx.send(chunk).await.is_err() {
                        break;
                    }
                }
            });

            tokio::signal::ctrl_c().await?;
            eprintln!("\nStopping...");

            drop(_mic_guard);
            let _ = bridge.await;

            let result = session.finish().await?;
            if let Some(task) = partial_task {
                let _ = task.await;
            }
            if clipboard {
                eprintln!("Text copied to clipboard.");
                let text = result.text();
                if !text.is_empty() {
                    eprintln!("Final: {text}");
                }
            }
            report_emit(&result);
        }
    }

    Ok(())
}

fn attach_asr(
    builder: PipelineBuilder,
    whisper_cli: &PathBuf,
    model: &PathBuf,
    language: Option<&str>,
) -> PipelineBuilder {
    let mut asr = WhisperLocal::new(whisper_cli, model);
    if let Some(lang) = language {
        asr = asr.with_language(lang);
    }
    builder.asr(asr)
}

fn attach_filters(
    builder: PipelineBuilder,
    language: Option<&str>,
    no_filter: bool,
) -> PipelineBuilder {
    if no_filter {
        return builder;
    }
    // Rule-based only. The embedding-backed alternatives were retired:
    // `filler_filter.py` gated a lexicon hit on a cosine threshold with
    // AND, which the calibration in docs/model-upgrade-candidates.md
    // §3.1 showed cannot discriminate, so it produced the same decisions
    // as the rule based filter while paying a Python subprocess and a
    // bge-small load per utterance.
    match language {
        Some("ja") | Some("japanese") => builder.filter(JapaneseFillerFilter::new()),
        Some("zh") | Some("chinese") => builder.filter(ChineseFillerFilter::new()),
        Some("es") | Some("spanish") => builder.filter(SpanishFillerFilter::new()),
        _ => builder.filter(SimpleFillerFilter::english()),
    }
}

fn attach_processors(
    builder: PipelineBuilder,
    language: Option<&str>,
    no_process: bool,
) -> PipelineBuilder {
    if no_process {
        return builder;
    }
    let lang = language.unwrap_or("en");
    builder
        .processor(SelfCorrectionDetector::new())
        .processor(SpokenFormNormalizer::new(lang))
        .processor(InverseTextNormalizer::new(lang))
        .processor(BasicPunctuationRestorer)
}

fn attach_vad(
    builder: PipelineBuilder,
    min_silence_ms: Option<u64>,
    final_pass: &str,
) -> Result<PipelineBuilder, Box<dyn std::error::Error>> {
    let mut config = SegmenterConfig::default();
    if let Some(ms) = min_silence_ms {
        config.min_silence = Duration::from_millis(ms);
    }
    let policy = parse_final_pass(final_pass)?;
    Ok(builder
        .vad(EarshotVad::new())
        .segmenter_config(config)
        .final_pass(policy))
}

fn parse_final_pass(s: &str) -> Result<FinalPass, String> {
    match s {
        "speech-only" => Ok(FinalPass::SpeechOnly),
        "whole" => Ok(FinalPass::WholeUtterance),
        "join" => Ok(FinalPass::JoinSegments),
        other => Err(format!(
            "unknown --final-pass {other:?} (speech-only|whole|join)"
        )),
    }
}

fn print_partial(p: &euhadra::pipeline::Partial) {
    let latency = p
        .endpoint_latency
        .map(|d| format!(" endpoint={}ms", d.as_millis()))
        .unwrap_or_default();
    eprintln!(
        "[partial {} {:.2}s–{:.2}s{}] {}",
        p.index,
        p.start.as_secs_f32(),
        p.end.as_secs_f32(),
        latency,
        p.text
    );
}

/// Surface an emitter failure on stderr.
///
/// `emit_result` is `None` when the pipeline has no emitter configured,
/// which is not a failure — there was simply nothing to deliver.
fn report_emit(result: &SessionResult) {
    let Some(emit) = &result.emit_result else {
        return;
    };
    if !emit.success {
        if let Some(err) = &emit.error {
            eprintln!("emit error: {err}");
        }
    }
}
