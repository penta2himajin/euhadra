//! Smoke bench: `WhisperOnnxAdapter` on FLEURS-ko.
//!
//! Drives the Rust adapter end-to-end on the same 10-utterance FLEURS-ko
//! subset the Python POC measured, and reports CER + RTF so we can
//! verify the Rust port matches the POC numbers (CER 1.09% / RTF 0.484
//! with `q4` on a 4-core Xeon).
//!
//! Usage:
//!   cargo run --release --features onnx --example bench_whisper_onnx_ko -- \
//!       --model-dir /tmp/whisper-onnx-turbo-int8 \
//!       --manifest /tmp/fleurs/ko/manifest.tsv \
//!       --audio-root /tmp/fleurs

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use euhadra::eval::metrics::cer_lenient;
use euhadra::whisper_local::read_wav;
use euhadra::whisper_onnx::WhisperOnnxAdapter;

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    audio_root: PathBuf,
    #[arg(long, default_value = "ko")]
    language: String,
    /// Quantisation suffix applied to all three graphs unless one of
    /// the per-component overrides below is given.
    #[arg(long, default_value = "q4")]
    quant: String,
    /// Override the encoder's quantisation independently.
    ///
    /// Whisper-turbo moved almost all of its parameters into the
    /// encoder (4 decoder layers against large-v3's 32), and the
    /// encoder runs on a fixed 30-second window whatever the audio
    /// length. So encoder and decoder are worth quantising separately:
    /// the encoder is where the time goes, while the decoder is where
    /// `docs/korean-asr-alternatives.md` found int8 degenerating into
    /// repeated tokens.
    #[arg(long)]
    encoder_quant: Option<String>,
    /// Override both decoder graphs' quantisation independently.
    #[arg(long)]
    decoder_quant: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    let enc_q = cli.encoder_quant.clone().unwrap_or_else(|| cli.quant.clone());
    let dec_q = cli.decoder_quant.clone().unwrap_or_else(|| cli.quant.clone());
    // The unquantised export carries no suffix, so `fp32` is a sentinel
    // for "the plain graph" rather than a filename fragment.
    let graph = |stem: &str, q: &str| {
        if q == "fp32" {
            format!("{stem}.onnx")
        } else {
            format!("{stem}_{q}.onnx")
        }
    };
    let cfg = euhadra::whisper_onnx::WhisperOnnxConfig {
        encoder_file: Some(graph("encoder_model", &enc_q)),
        decoder_file: Some(graph("decoder_model", &dec_q)),
        decoder_with_past_file: Some(graph("decoder_with_past_model", &dec_q)),
        language: Some(cli.language.clone()),
    };

    println!(
        "loading {} (encoder={} decoder={})...",
        cli.model_dir.display(),
        enc_q,
        dec_q
    );
    let t0 = Instant::now();
    let adapter = WhisperOnnxAdapter::load_with_config(&cli.model_dir, cfg).expect("load");
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let manifest = std::fs::read_to_string(&cli.manifest).expect("read manifest");
    let rows: Vec<(String, PathBuf, String)> = manifest
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split('\t');
            let uid = it.next().unwrap().to_string();
            let audio_rel = it.next().unwrap();
            let reference = it.next().unwrap().to_string();
            (uid, cli.audio_root.join(audio_rel), reference)
        })
        .collect();

    // Warm-up pass on the first utterance.
    let warm = read_wav(&rows[0].1).expect("warm wav");
    let _ = adapter
        .transcribe_samples(&warm.samples)
        .expect("warm transcribe");

    let mut total_audio = 0.0_f64;
    let mut total_asr = 0.0_f64;
    let mut cer_acc = 0.0_f64;
    let mut counted = 0_usize;

    for (uid, wav_path, reference) in &rows {
        let chunk = read_wav(wav_path).expect("wav");
        let dur = chunk.samples.len() as f64 / 16000.0;
        let t = Instant::now();
        let hyp = adapter
            .transcribe_samples(&chunk.samples)
            .expect("transcribe");
        let asr_s = t.elapsed().as_secs_f64();
        let c = cer_lenient(reference, &hyp);
        total_audio += dur;
        total_asr += asr_s;
        if !c.is_nan() {
            cer_acc += c;
            counted += 1;
        }
        println!(
            "{uid}: audio={dur:.2}s asr={ms:.0}ms cer={c:.4} hyp={hyp:?}",
            ms = asr_s * 1000.0
        );
    }

    println!();
    println!("=== Whisper-ONNX-{} (Rust) ===", cli.quant);
    println!("  utterances    : {}", rows.len());
    println!("  total_audio_s : {:.2}", total_audio);
    println!("  total_asr_s   : {:.2}", total_asr);
    println!(
        "  cer_lenient   : {:.4}",
        if counted == 0 {
            f64::NAN
        } else {
            cer_acc / counted as f64
        }
    );
    println!("  rtf           : {:.4}", total_asr / total_audio);
}
