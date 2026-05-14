use clap::{Parser, Subcommand};
use std::path::PathBuf;

use euhadra::emitters::ClipboardEmitter;
use euhadra::filter::{
    ChineseFillerFilter, EmbeddingFillerFilter, JapaneseFillerFilter, SimpleFillerFilter,
    SpanishFillerFilter,
};
use euhadra::mic::{self, MicConfig};
use euhadra::mock::{MockContextProvider, MockRefiner, StdoutEmitter};
use euhadra::pipeline::Pipeline;
use euhadra::processor::{
    BasicPunctuationRestorer, InverseTextNormalizer, SelfCorrectionDetector,
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

        /// Path to filler_filter.py for embedding-based filler removal.
        /// If omitted, uses the simple keyword-based filter.
        #[arg(long)]
        filler_script: Option<PathBuf>,

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
            let text = whisper_local::transcribe_file(
                &whisper_cli,
                &model,
                &file,
                language.as_deref(),
            )
            .await?;
            println!("{text}");
        }

        Commands::Dictate {
            file,
            whisper_cli,
            model,
            language,
            filler_script,
            no_filter,
            no_process,
        } => {
            // Load audio
            let audio = whisper_local::read_wav(&file)
                .map_err(|e| format!("failed to read WAV: {e}"))?;

            // Build pipeline
            let mut builder = Pipeline::builder();

            // ASR
            let mut asr = WhisperLocal::new(&whisper_cli, &model);
            if let Some(ref lang) = language {
                asr = asr.with_language(lang);
            }
            builder = builder.asr(asr);

            // Filter — auto-select based on language
            if !no_filter {
                if let Some(ref script) = filler_script {
                    builder = builder.filter(EmbeddingFillerFilter::new(script));
                } else {
                    builder = match language.as_deref() {
                        Some("ja") | Some("japanese") => {
                            builder.filter(JapaneseFillerFilter::new())
                        }
                        Some("zh") | Some("chinese") => {
                            builder.filter(ChineseFillerFilter::new())
                        }
                        Some("es") | Some("spanish") => {
                            builder.filter(SpanishFillerFilter::new())
                        }
                        _ => builder.filter(SimpleFillerFilter::english()),
                    };
                }
            }

            // Processors — self-correction detection + punctuation
            if !no_process {
                builder = builder
                    .processor(SelfCorrectionDetector::new())
                    .processor(InverseTextNormalizer::new(
                        language.as_deref().unwrap_or("en"),
                    ))
                    .processor(BasicPunctuationRestorer);
            }

            // Refiner — passthrough for now (no LLM)
            builder = builder.refiner(MockRefiner::passthrough());

            // Context — manual for CLI
            builder = builder.context(MockContextProvider::new());

            // Emitter — stdout
            builder = builder.emitter(StdoutEmitter);

            let pipeline = builder.build()?;

            // Run session
            let (audio_tx, _cancel, handle) = pipeline.session();
            audio_tx.send(audio).await?;
            drop(audio_tx);

            let result = handle.await??;

            if !result.emit_result.success {
                if let Some(err) = result.emit_result.error {
                    eprintln!("emit error: {err}");
                }
            }
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
                    Some("ja") | Some("japanese") => {
                        builder.filter(JapaneseFillerFilter::new())
                    }
                    Some("zh") | Some("chinese") => {
                        builder.filter(ChineseFillerFilter::new())
                    }
                    Some("es") | Some("spanish") => {
                        builder.filter(SpanishFillerFilter::new())
                    }
                    _ => builder.filter(SimpleFillerFilter::english()),
                };
            }

            // Processors
            if !no_process {
                builder = builder
                    .processor(SelfCorrectionDetector::new())
                    .processor(InverseTextNormalizer::new(
                        language.as_deref().unwrap_or("en"),
                    ))
                    .processor(BasicPunctuationRestorer);
            }

            // Refiner
            builder = builder.refiner(MockRefiner::passthrough());

            // Context
            builder = builder.context(MockContextProvider::new());

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

            // Start pipeline session
            let (audio_tx, _cancel, handle) = pipeline.session();

            // Bridge mic → pipeline
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

            let result = handle.await??;
            if clipboard {
                eprintln!("Text copied to clipboard.");
            }
            if !result.emit_result.success {
                if let Some(err) = result.emit_result.error {
                    eprintln!("emit error: {err}");
                }
            }
        }
    }

    Ok(())
}
