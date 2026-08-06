use clap::{Parser, Subcommand};
use std::path::PathBuf;

use euhadra::emitters::ClipboardEmitter;
use euhadra::filter::{
    ChineseFillerFilter, JapaneseFillerFilter, SimpleFillerFilter,
    SpanishFillerFilter,
};
use euhadra::mic::{self, MicConfig};
use euhadra::mock::StdoutEmitter;
use euhadra::pipeline::{Pipeline, SessionResult};
use euhadra::processor::{
    BasicPunctuationRestorer, InverseTextNormalizer, SelfCorrectionDetector, SpokenFormNormalizer,
};
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
    },

    /// Record from microphone and transcribe through the full pipeline.
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
        } => {
            // Load audio
            let audio =
                whisper_local::read_wav(&file).map_err(|e| format!("failed to read WAV: {e}"))?;

            // Build pipeline
            let mut builder = Pipeline::builder();

            // ASR
            let mut asr = WhisperLocal::new(&whisper_cli, &model);
            if let Some(ref lang) = language {
                asr = asr.with_language(lang);
            }
            builder = builder.asr(asr);

            // Filter — auto-select based on language.
            //
            // Rule-based only. The embedding-backed alternatives were
            // retired: `filler_filter.py` gated a lexicon hit on a cosine
            // threshold with AND, which the calibration in
            // docs/model-upgrade-candidates.md §3.1 showed cannot
            // discriminate, so it produced the same decisions as the rule
            // based filter while paying a Python subprocess and a
            // bge-small load per utterance.
            if !no_filter {
                builder = match language.as_deref() {
                    Some("ja") | Some("japanese") => builder.filter(JapaneseFillerFilter::new()),
                    Some("zh") | Some("chinese") => builder.filter(ChineseFillerFilter::new()),
                    Some("es") | Some("spanish") => builder.filter(SpanishFillerFilter::new()),
                    _ => builder.filter(SimpleFillerFilter::english()),
                };
            }

            // Processors — self-correction detection + punctuation
            if !no_process {
                builder = builder
                    .processor(SelfCorrectionDetector::new())
                    .processor(SpokenFormNormalizer::new(
                        language.as_deref().unwrap_or("en"),
                    ))
                    .processor(InverseTextNormalizer::new(
                        language.as_deref().unwrap_or("en"),
                    ))
                    .processor(BasicPunctuationRestorer);
            }

            // No refiner and no context provider: this is the LLM-free
            // Tier 1+2 path, and both are optional now.

            // Emitter — stdout
            builder = builder.emitter(StdoutEmitter);

            let pipeline = builder.build()?;

            // The whole file is already read, so this is the batch path.
            let result = pipeline.transcribe(std::slice::from_ref(&audio)).await?;

            report_emit(&result);
        }

        Commands::Record {
            whisper_cli,
            model,
            language,
            no_filter,
            no_process,
            clipboard,
        } => {
            eprintln!("Recording from microphone... Press Ctrl+C to stop.");

            // Build pipeline
            let mut builder = Pipeline::builder();

            // ASR
            let mut asr = WhisperLocal::new(&whisper_cli, &model);
            if let Some(ref lang) = language {
                asr = asr.with_language(lang);
            }
            builder = builder.asr(asr);

            // Filter
            if !no_filter {
                builder = match language.as_deref() {
                    Some("ja") | Some("japanese") => builder.filter(JapaneseFillerFilter::new()),
                    Some("zh") | Some("chinese") => builder.filter(ChineseFillerFilter::new()),
                    Some("es") | Some("spanish") => builder.filter(SpanishFillerFilter::new()),
                    _ => builder.filter(SimpleFillerFilter::english()),
                };
            }

            // Processors
            if !no_process {
                builder = builder
                    .processor(SelfCorrectionDetector::new())
                    .processor(SpokenFormNormalizer::new(
                        language.as_deref().unwrap_or("en"),
                    ))
                    .processor(InverseTextNormalizer::new(
                        language.as_deref().unwrap_or("en"),
                    ))
                    .processor(BasicPunctuationRestorer);
            }

            // No refiner and no context provider: this is the LLM-free
            // Tier 1+2 path, and both are optional now.

            // Emitter
            if clipboard {
                builder = builder.emitter(ClipboardEmitter::new());
            } else {
                builder = builder.emitter(StdoutEmitter);
            }

            let pipeline = builder.build()?;

            // Start mic capture
            let (mut mic_rx, _mic_guard) =
                mic::record(MicConfig::default()).map_err(|e| format!("mic: {e}"))?;

            // Live capture, so the session path: ASR starts while the
            // speaker is still talking.
            let session = pipeline.session();

            // Bridge mic → pipeline
            let audio_tx = session.audio.clone();
            let bridge = tokio::spawn(async move {
                while let Some(chunk) = mic_rx.recv().await {
                    if audio_tx.send(chunk).await.is_err() {
                        break;
                    }
                }
            });

            // Wait for Ctrl+C
            tokio::signal::ctrl_c().await?;
            eprintln!("\nStopping...");

            // Drop mic guard to stop recording (closes the channel)
            drop(_mic_guard);
            let _ = bridge.await;

            let result = session.finish().await?;
            if clipboard {
                eprintln!("Text copied to clipboard.");
            }
            report_emit(&result);
        }
    }

    Ok(())
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
