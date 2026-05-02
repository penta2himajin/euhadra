//! L1 smoke evaluation runner.
//!
//! Reads each language's FLEURS subset manifest, runs each utterance
//! through the configured pipeline (`WhisperLocal` ASR + filler filter +
//! basic processors + passthrough refiner), records WER/CER + latency,
//! and either updates `docs/benchmarks/ci_baseline.json` (`--update-baseline`)
//! or compares against it and exits non-zero on regression.
//!
//! CLI:
//!   cargo run --release --example eval_l1_smoke -- \
//!       --whisper-cli vendor/whisper.cpp/build/bin/whisper-cli \
//!       --model-en   vendor/whisper.cpp/models/ggml-tiny.en.bin \
//!       --model-multi vendor/whisper.cpp/models/ggml-tiny.bin \
//!       --data-dir   data/fleurs_subset \
//!       --baseline   docs/benchmarks/ci_baseline.json
//!
//! Add `--update-baseline` on the first run (or after an intentional
//! quality change) to regenerate the file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use euhadra::eval::baseline::{
    Baseline, LanguageBaseline, LatencyRecord, Tolerances, Verdict, check_language,
};
use euhadra::eval::latency::Samples;
use euhadra::eval::metrics::{cer_lenient, wer_lenient};
#[cfg(feature = "onnx")]
use euhadra::canary::{CanaryAdapter, CanaryConfig};
#[cfg(feature = "onnx")]
use euhadra::canary::decoder::PrefixFormat;
#[cfg(feature = "onnx")]
use euhadra::parakeet::ParakeetAdapter;
#[cfg(feature = "onnx")]
use euhadra::paraformer::ParaformerAdapter;
#[cfg(feature = "onnx")]
use euhadra::sensevoice::SenseVoiceAdapter;
use euhadra::prelude::*;
use euhadra::whisper_local::{WhisperLocal, read_wav};

#[derive(Parser, Debug)]
#[command(about = "L1 smoke: FLEURS subset → ASR → pipeline → WER/CER + latency")]
struct Cli {
    /// Path to whisper-cli binary
    #[arg(long, env = "WHISPER_CLI_PATH")]
    whisper_cli: PathBuf,

    /// English-only model (ggml-tiny.en.bin)
    #[arg(long, env = "WHISPER_MODEL_EN")]
    model_en: PathBuf,

    /// Multilingual model (ggml-tiny.bin), used for ja/zh
    #[arg(long, env = "WHISPER_MODEL_MULTI")]
    model_multi: PathBuf,

    /// Root of FLEURS subset (containing `<lang>/manifest.tsv`)
    #[arg(long, env = "FLEURS_DATA_DIR", default_value = "data/fleurs_subset")]
    data_dir: PathBuf,

    /// Path to baseline JSON
    #[arg(long, default_value = "docs/benchmarks/ci_baseline.json")]
    baseline: PathBuf,

    /// Languages to evaluate (default: all five)
    #[arg(long, value_delimiter = ',', default_value = "en,ja,zh,es,ko")]
    langs: Vec<String>,

    /// Write the measured numbers as the new baseline instead of comparing.
    #[arg(long)]
    update_baseline: bool,

    /// Optional path to an `nvidia/parakeet-tdt_ctc-0.6b-ja` ONNX model
    /// directory (encoder-model.onnx, decoder_joint-model.onnx,
    /// vocab.txt, *.data). When provided, the `ja` language is run
    /// through ParakeetAdapter (80-mel) instead of WhisperLocal — gives
    /// dramatically better CER (~6–9% vs whisper-tiny's ~42%). Requires
    /// `--features onnx` at build time. When omitted, `ja` falls back
    /// to whisper.
    #[arg(long, env = "PARAKEET_JA_DIR")]
    parakeet_ja_dir: Option<PathBuf>,

    /// Optional path to a `parakeet-tdt-0.6b-v3` ONNX model directory
    /// (encoder-model.onnx, decoder_joint-model.onnx, vocab.txt,
    /// *.data). When provided, the `en` language is run through
    /// ParakeetAdapter (default 128-mel) instead of whisper-tiny.en.
    /// Smaller WER (~7.5% vs ~8.4%) and roughly 1.5× faster. Requires
    /// `--features onnx`.
    #[arg(long, env = "PARAKEET_EN_DIR")]
    parakeet_en_dir: Option<PathBuf>,

    /// Optional path to a FunASR Paraformer-large ONNX bundle
    /// (`model.onnx`, `am.mvn`, `tokens.json`). When provided, the
    /// `zh` language is run through `ParaformerAdapter` instead of
    /// whisper-tiny — gives dramatically better CER on Mandarin.
    /// Requires `--features onnx`.
    #[arg(long, env = "PARAFORMER_ZH_DIR")]
    paraformer_zh_dir: Option<PathBuf>,

    /// Optional path to an `istupakov/canary-180m-flash-onnx` bundle
    /// (`encoder-model.onnx`, `decoder-model.onnx`, `vocab.txt`; INT8
    /// pair `encoder-model.int8.onnx` / `decoder-model.int8.onnx`
    /// also accepted via the `with_int8_weights` config). When
    /// provided, the `es` language is run through `CanaryAdapter`
    /// — Canary-180M-Flash reports MLS-Spanish WER 3.17 % on the
    /// official model card, vs. whisper-tiny in the ~50 % range.
    /// Requires `--features onnx` at build time. Populate via
    /// `scripts/setup_canary_es.sh`.
    #[arg(long, env = "CANARY_ES_DIR")]
    canary_es_dir: Option<PathBuf>,

    /// Optional path to a `FunAudioLLM/SenseVoiceSmall` ONNX bundle
    /// (`model.onnx` or `model.int8.onnx`, `am.mvn`, `tokens.txt`,
    /// `metadata.json`). When provided, the `ko` language is run
    /// through `SenseVoiceAdapter` instead of whisper-tiny — the
    /// upstream model reports Korean CER far below whisper-tiny's
    /// multilingual baseline. Requires `--features onnx` at build
    /// time. Populate via `scripts/setup_sensevoice.sh`.
    #[arg(long, env = "SENSEVOICE_DIR")]
    sensevoice_dir: Option<PathBuf>,

    /// Print every utterance's reference / hypothesis / per-utterance
    /// WER + CER as the smoke runs. Useful when chasing residual
    /// errors (mismatched normalisation, OOV characters, traditional
    /// vs simplified Chinese, numeric formatting). Off by default to
    /// keep CI logs short.
    #[arg(long, default_value_t = false)]
    dump_utterances: bool,
}

#[derive(Debug)]
struct Manifest {
    rows: Vec<ManifestRow>,
}

#[derive(Debug)]
struct ManifestRow {
    #[allow(dead_code)]
    id: String,
    audio_path: PathBuf,
    reference: String,
}

fn load_manifest(data_dir: &Path, lang: &str) -> std::io::Result<Manifest> {
    let path = data_dir.join(lang).join("manifest.tsv");
    let raw = std::fs::read_to_string(&path)?;
    let mut rows = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue; // header
        }
        let cols: Vec<&str> = line.splitn(3, '\t').collect();
        if cols.len() != 3 {
            continue;
        }
        rows.push(ManifestRow {
            id: cols[0].to_string(),
            audio_path: data_dir.join(cols[1]),
            reference: cols[2].to_string(),
        });
    }
    Ok(Manifest { rows })
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Honour `RUST_LOG`/default to `euhadra=info` so adapter
    // tracing::debug events (e.g. SenseVoice CTC decode previews) are
    // visible when the operator opts in via `RUST_LOG=euhadra=debug`,
    // without spamming logs on a quiet baseline run.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "euhadra=info".parse().unwrap()),
        )
        .with_writer(std::io::stderr)
        .init();

    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}

async fn run() -> Result<(), String> {
    let cli = Cli::parse();

    let mut measured: BTreeMap<String, LanguageBaseline> = BTreeMap::new();

    let mut used_parakeet_ja = false;
    let mut used_parakeet_en = false;
    let mut used_paraformer_zh = false;
    let mut used_sensevoice_ko = false;

    for lang in &cli.langs {
        let manifest = load_manifest(&cli.data_dir, lang)
            .map_err(|e| format!("loading {lang} manifest: {e}"))?;
        if manifest.rows.is_empty() {
            return Err(format!("manifest for {lang} is empty — did the download script run?"));
        }

        // Build the pipeline ONCE per language. For ParakeetAdapter
        // this matters: model load is a multi-second ONNX session
        // initialisation that we don't want to pay per utterance.
        let model = if lang == "en" { &cli.model_en } else { &cli.model_multi };
        let pipeline = build_pipeline(
            &cli.whisper_cli,
            model,
            lang,
            cli.parakeet_ja_dir.as_deref(),
            cli.parakeet_en_dir.as_deref(),
            cli.paraformer_zh_dir.as_deref(),
            cli.canary_es_dir.as_deref(),
            cli.sensevoice_dir.as_deref(),
        )?;
        if lang == "ja" && cli.parakeet_ja_dir.is_some() {
            used_parakeet_ja = true;
        }
        if lang == "en" && cli.parakeet_en_dir.is_some() {
            used_parakeet_en = true;
        }
        if lang == "zh" && cli.paraformer_zh_dir.is_some() {
            used_paraformer_zh = true;
        }
        if lang == "ko" && cli.sensevoice_dir.is_some() {
            used_sensevoice_ko = true;
        }

        let lang_result = evaluate_language(lang, &manifest, &pipeline, cli.dump_utterances).await?;
        measured.insert(lang.clone(), lang_result);
    }

    // Print all per-language summaries together so they're easy to
    // find — `--dump-utterances` otherwise scatters them between
    // hundreds of ref/hyp lines.
    println!();
    for lang in &cli.langs {
        if let Some(r) = measured.get(lang) {
            print_language_result(lang, r);
        }
    }

    let asr_model_label = format!(
        "{en_model} (en) / {ja_model} (ja) / {zh_model} (zh) / {ko_model} (ko)",
        en_model = if used_parakeet_en { "parakeet-tdt-0.6b-v3" } else { "ggml-tiny.en" },
        ja_model = if used_parakeet_ja { "parakeet-tdt_ctc-0.6b-ja" } else { "ggml-tiny" },
        zh_model = if used_paraformer_zh { "paraformer-large-zh" } else { "ggml-tiny" },
        ko_model = if used_sensevoice_ko { "sensevoice-small" } else { "ggml-tiny" },
    );

    if cli.update_baseline {
        let baseline = Baseline {
            schema_version: 1,
            generated: chrono_now_iso8601(),
            asr_model: asr_model_label,
            languages: measured,
            tolerances: Tolerances::default(),
        };
        baseline
            .save(&cli.baseline)
            .map_err(|e| format!("writing baseline: {e}"))?;
        eprintln!("baseline written to {}", cli.baseline.display());
        return Ok(());
    }

    // Compare against existing baseline
    let baseline = Baseline::load(&cli.baseline)
        .map_err(|e| format!("loading baseline {}: {e}", cli.baseline.display()))?;

    let mut any_fail = false;
    let mut any_warn = false;
    for (lang, m) in &measured {
        let Some(b) = baseline.languages.get(lang) else {
            eprintln!("warn: {lang} has no baseline entry");
            continue;
        };
        let verdicts = check_language(m, b, &baseline.tolerances);
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

async fn evaluate_language(
    lang: &str,
    manifest: &Manifest,
    pipeline: &Pipeline,
    dump_utterances: bool,
) -> Result<LanguageBaseline, String> {
    let mut wer_acc = 0.0;
    let mut cer_acc = 0.0;
    let mut samples_counted = 0;
    let mut asr_latency = Samples::new();
    let mut e2e_latency = Samples::new();
    // RTF = ASR processing time / audio duration. Aggregated as the
    // mean across utterances (each utterance contributes its own RTF
    // observation).
    let mut total_asr_secs = 0.0_f64;
    let mut total_audio_secs = 0.0_f64;

    for row in &manifest.rows {
        let audio = read_wav(&row.audio_path)
            .map_err(|e| format!("read_wav {}: {e}", row.audio_path.display()))?;
        let audio_secs =
            audio.samples.len() as f64 / (audio.sample_rate as f64 * audio.channels as f64);

        let e2e_start = Instant::now();
        let (audio_tx, _cancel, handle) = pipeline.session();

        // ASR latency = time from "audio handed off" to "session result";
        // for this final-only pipeline the two are nearly identical, but
        // we measure them separately so we can split them out once we add
        // a streaming ASR adapter.
        let asr_start = Instant::now();
        audio_tx
            .send(audio)
            .await
            .map_err(|e| format!("send audio: {e}"))?;
        drop(audio_tx);

        // Per-utterance pipeline failures must not abort the whole
        // smoke run — ASR adapters legitimately produce zero text on
        // some inputs (e.g. autoregressive decoders that hit
        // max_sequence_length without emitting non-special tokens),
        // and treating that as fatal hides the per-utterance failure
        // mode behind a top-level error. Count empty hypotheses as
        // 100 % WER/CER (the reference is non-trivially missed) and
        // keep going.
        let result_or_err = handle
            .await
            .map_err(|e| format!("join: {e}"))?;
        let asr_elapsed = asr_start.elapsed();
        asr_latency.record(asr_elapsed);
        e2e_latency.record(e2e_start.elapsed());
        total_asr_secs += asr_elapsed.as_secs_f64();
        total_audio_secs += audio_secs;

        let result = match result_or_err {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "  [{lang} {idx:>2}] WARN pipeline returned no text \
                     ({e}); counting as 100 % error",
                    idx = samples_counted + 1
                );
                wer_acc += 1.0;
                cer_acc += 1.0;
                samples_counted += 1;
                continue;
            }
        };
        let hyp = &result.raw_text;
        // Live-ASR comparison: format-only differences between Whisper /
        // Parakeet / Paraformer output and the FLEURS gold transcript
        // (smart quotes, Chinese clause-comma, Arabic vs spelled
        // numerals) are noise here, not signal — use the lenient pass.
        let w = wer_lenient(&row.reference, hyp);
        let c = cer_lenient(&row.reference, hyp);
        if !w.is_nan() {
            wer_acc += w;
        }
        if !c.is_nan() {
            cer_acc += c;
        }
        samples_counted += 1;

        if dump_utterances {
            // Whitespace-segmented languages (en, es) report WER as
            // primary; logographic / non-segmented (ja, zh) and
            // Korean — where eojeol vs morpheme spacing varies and
            // Korean ASR papers default to CER — report CER.
            // Spanish matches the Canary model card convention (MLS
            // WER 3.17 %, MCV WER 4.90 %).
            let primary = match lang {
                "en" | "es" => format!("WER={:.4}", w),
                _ => format!("CER={:.4}", c),
            };
            // Emit on stderr so stdout stays parseable for the
            // baseline-comparison summary block.
            eprintln!("  [{lang} {idx:>2}] {primary}", idx = samples_counted);
            eprintln!("       ref: {}", row.reference);
            eprintln!("       hyp: {hyp}");
        }
    }

    let mean_wer = if samples_counted > 0 {
        wer_acc / samples_counted as f64
    } else {
        f64::NAN
    };
    let mean_cer = if samples_counted > 0 {
        cer_acc / samples_counted as f64
    } else {
        f64::NAN
    };

    let asr_summary = asr_latency.summary().ok_or("no asr samples")?;
    let e2e_summary = e2e_latency.summary().ok_or("no e2e samples")?;

    // For en / es we report WER, for ja / zh / ko we report CER.
    // Both are computed either way; only the "primary" metric is
    // stored in the baseline. Spanish is whitespace-segmented and
    // the Canary model card publishes WER (MLS 3.17 %, MCV-16.1
    // 4.90 %), so WER is the right axis to compare against. Korean
    // is space-separated at the eojeol level but spacing
    // conventions vary across speakers / corpora; Korean ASR papers
    // default to CER for that reason.
    let (wer_field, cer_field) = match lang {
        "en" | "es" => (Some(round4(mean_wer)), None),
        _ => (None, Some(round4(mean_cer))),
    };

    let rtf = if total_audio_secs > 0.0 {
        Some(round4(total_asr_secs / total_audio_secs))
    } else {
        None
    };

    Ok(LanguageBaseline {
        samples: samples_counted,
        wer: wer_field,
        cer: cer_field,
        asr_latency_ms: LatencyRecord::from(asr_summary),
        e2e_latency_ms: LatencyRecord::from(e2e_summary),
        rtf,
    })
}

// Each ASR backend has its own optional model-dir flag, and clippy's
// 7-arg default threshold only barely fits the original four; rather
// than refactor the wiring into a config struct just for this eval
// driver, accept the count and pin the allow here.
#[allow(clippy::too_many_arguments)]
fn build_pipeline(
    whisper_cli: &Path,
    model: &Path,
    lang: &str,
    parakeet_ja_dir: Option<&Path>,
    parakeet_en_dir: Option<&Path>,
    paraformer_zh_dir: Option<&Path>,
    canary_es_dir: Option<&Path>,
    sensevoice_dir: Option<&Path>,
) -> Result<Pipeline, String> {
    // Build the post-ASR stack first; ASR is bolted on per-language
    // because its concrete type depends on the model choice.
    let mut builder = Pipeline::builder()
        .processor(SelfCorrectionDetector::new())
        .processor(BasicPunctuationRestorer)
        .refiner(MockRefiner::passthrough())
        .context(MockContextProvider::new())
        .emitter(MockEmitter::new());
    builder = match lang {
        "en" => builder.filter(SimpleFillerFilter::english()),
        "ja" => builder.filter(JapaneseFillerFilter::new()),
        "zh" => builder.filter(ChineseFillerFilter::new()),
        "es" => builder.filter(SpanishFillerFilter::new()),
        // Korean filler filter is a follow-up. The English filter's
        // dictionary (um / uh / well / …) shares no surface form
        // with Hangul fillers (음 / 어 / 그 / …), so it acts as a
        // safe pass-through for Korean transcripts.
        "ko" => builder.filter(SimpleFillerFilter::english()),
        other => return Err(format!("unsupported language: {other}")),
    };

    builder = match lang {
        "ja" if parakeet_ja_dir.is_some() => {
            #[cfg(feature = "onnx")]
            {
                let dir = parakeet_ja_dir.unwrap();
                let asr = ParakeetAdapter::load_with_feature_size(dir, 80).map_err(|e| {
                    format!("load parakeet ja from {}: {e}", dir.display())
                })?;
                builder.asr(asr)
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(
                    "--parakeet-ja-dir requires --features onnx at build time".into(),
                );
            }
        }
        "en" if parakeet_en_dir.is_some() => {
            #[cfg(feature = "onnx")]
            {
                let dir = parakeet_en_dir.unwrap();
                // parakeet-tdt-0.6b-v3 (multilingual European, en included)
                // ships with the default 128-mel preprocessor — no
                // feature_size override needed; `load(dir)` does it.
                let asr = ParakeetAdapter::load(dir).map_err(|e| {
                    format!("load parakeet en from {}: {e}", dir.display())
                })?;
                builder.asr(asr)
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(
                    "--parakeet-en-dir requires --features onnx at build time".into(),
                );
            }
        }
        "zh" if paraformer_zh_dir.is_some() => {
            #[cfg(feature = "onnx")]
            {
                let dir = paraformer_zh_dir.unwrap();
                let asr = ParaformerAdapter::load(dir).map_err(|e| {
                    format!("load paraformer zh from {}: {e}", dir.display())
                })?;
                builder.asr(asr)
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(
                    "--paraformer-zh-dir requires --features onnx at build time".into(),
                );
            }
        }
        "ko" if sensevoice_dir.is_some() => {
            #[cfg(feature = "onnx")]
            {
                let dir = sensevoice_dir.unwrap();
                let asr = SenseVoiceAdapter::load(dir)
                    .map_err(|e| format!("load sensevoice from {}: {e}", dir.display()))?
                    .with_language("ko");
                builder.asr(asr)
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(
                    "--sensevoice-dir requires --features onnx at build time".into(),
                );
            }
        }
        "es" if canary_es_dir.is_some() => {
            #[cfg(feature = "onnx")]
            {
                let dir = canary_es_dir.unwrap();
                // Honour `CANARY_PREFIX_FORMAT={onnx-asr,nemo-canary2}`
                // so callers can sweep the prefix layout without a
                // rebuild. When unset, the cfg-default applies — that
                // default is `NemoCanary2` per `DEFAULT_PREFIX_FORMAT`.
                let mut cfg = CanaryConfig::istupakov_default();
                if let Ok(raw) = std::env::var("CANARY_PREFIX_FORMAT") {
                    cfg.prefix_format = match raw.as_str() {
                        "nemo-canary2" | "nemo_canary2" | "NemoCanary2" => {
                            PrefixFormat::NemoCanary2
                        }
                        "onnx-asr" | "onnx_asr" | "OnnxAsr" => PrefixFormat::OnnxAsr,
                        other => {
                            return Err(format!(
                                "unknown CANARY_PREFIX_FORMAT={other:?}; expected onnx-asr | nemo-canary2"
                            ));
                        }
                    };
                }
                // INT8 toggle — used by the post-v8 INT8
                // re-assessment sweep. When unset, FP32 weights are
                // loaded (the cfg default). Truthy values enable the
                // INT8 encoder + decoder weight bundles.
                if let Ok(raw) = std::env::var("CANARY_INT8") {
                    let truthy = matches!(raw.as_str(), "1" | "true" | "TRUE" | "yes");
                    if truthy {
                        cfg = cfg.with_int8_weights();
                    }
                }
                // Beam-search knobs — used by the post-v8 length-penalty
                // sweep. When unset, cfg-defaults apply (greedy).
                if let Ok(raw) = std::env::var("CANARY_BEAM_SIZE") {
                    cfg.beam_size = raw
                        .parse()
                        .map_err(|e| format!("CANARY_BEAM_SIZE: {e}"))?;
                }
                if let Ok(raw) = std::env::var("CANARY_LENGTH_PENALTY") {
                    cfg.length_penalty = raw
                        .parse()
                        .map_err(|e| format!("CANARY_LENGTH_PENALTY: {e}"))?;
                }
                let asr = CanaryAdapter::load_with_config(dir, cfg)
                    .map_err(|e| format!("load canary es from {}: {e}", dir.display()))?
                    .with_language("es");
                builder.asr(asr)
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(
                    "--canary-es-dir requires --features onnx at build time".into(),
                );
            }
        }
        _ => builder.asr(WhisperLocal::new(whisper_cli, model).with_language(lang)),
    };

    builder.build().map_err(|e| format!("build pipeline: {e}"))
}

fn print_language_result(lang: &str, r: &LanguageBaseline) {
    let primary = match lang {
        "en" | "es" => format!("WER={:.4}", r.wer.unwrap_or(f64::NAN)),
        _ => format!("CER={:.4}", r.cer.unwrap_or(f64::NAN)),
    };
    let rtf_str = match r.rtf {
        Some(rtf) => format!("RTF={rtf:.3}"),
        None => "RTF=n/a".to_string(),
    };
    println!(
        "[{lang}] n={}  {primary}  {rtf_str}  asr p50={:.0}ms p95={:.0}ms  e2e p50={:.0}ms p95={:.0}ms",
        r.samples,
        r.asr_latency_ms.p50,
        r.asr_latency_ms.p95,
        r.e2e_latency_ms.p50,
        r.e2e_latency_ms.p95,
    );
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

fn chrono_now_iso8601() -> String {
    // Avoid pulling in `chrono` as a dep just for one timestamp.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Crude UTC formatting: YYYY-MM-DDTHH:MM:SSZ from epoch seconds.
    format!("epoch:{secs}")
}
