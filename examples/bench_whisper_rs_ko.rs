//! In-process whisper-rs bench on FLEURS-ko 10utt.
//!
//! Loads the GGML model once, transcribes each utt sequentially via
//! `WhisperRsAdapter::transcribe_samples`, times only the inference
//! call (no model load, no WAV I/O), and reports lenient CER + RTF
//! using `eval::metrics::cer_lenient`.
//!
//! Throwaway: not wired into CI. Lives next to the existing
//! `eval_l1_smoke` example for the duration of the #83 measurement
//! work. The flags pick the model file off the command line so the
//! same binary can compare Q5_0, Q4_0, Q8_0 etc.

use clap::Parser;
use euhadra::eval::metrics::cer_lenient;
use euhadra::whisper_rs::WhisperRsAdapter;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    /// Path to a GGML whisper checkpoint (e.g. ggml-large-v3-turbo-q5_0.bin).
    #[arg(long)]
    model: PathBuf,
    /// Path to the FLEURS-ko subset produced by
    /// `scripts/download_fleurs_subset.py --lang ko`.
    #[arg(long, default_value = "/tmp/fleurs/ko")]
    fleurs_dir: PathBuf,
    /// CPU threads. Matches the 4-thread baseline used by previous
    /// FLEURS-ko bench runs.
    #[arg(long, default_value_t = 4)]
    threads: i32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let manifest = args.fleurs_dir.join("manifest.tsv");
    let raw = std::fs::read_to_string(&manifest)?;
    let rows: Vec<(String, PathBuf, String)> = raw
        .lines()
        .skip(1) // header
        .filter_map(|line| {
            let mut parts = line.splitn(3, '\t');
            let id = parts.next()?.to_string();
            let rel = parts.next()?;
            let reference = parts.next()?.to_string();
            Some((id, args.fleurs_dir.parent().unwrap().join(rel), reference))
        })
        .collect();

    println!(
        "Loading {} (threads={}) ...",
        args.model.display(),
        args.threads
    );
    let t0 = Instant::now();
    let adapter = WhisperRsAdapter::load_with_config(
        args.model.clone(),
        euhadra::whisper_rs::WhisperRsConfig {
            language: Some("ko".into()),
            threads: args.threads,
            no_fallback: false,
        },
    )?;
    let load_s = t0.elapsed().as_secs_f64();
    println!("Loaded in {:.2}s", load_s);

    // Warm-up on the first utt
    let (_, first_wav, _) = &rows[0];
    let samples = euhadra::whisper_local::read_wav(first_wav).map_err(|e| anyhow_str(&e))?;
    let _ = adapter
        .transcribe_samples_public(&samples.samples)
        .map_err(|e| anyhow_str(&e.message))?;

    let mut total_audio = 0.0;
    let mut total_asr = 0.0;
    let mut total_cer_chars = 0.0;
    let mut total_ref_chars = 0usize;
    let mut latencies = Vec::new();

    for (id, wav_path, reference) in &rows {
        let chunk = euhadra::whisper_local::read_wav(wav_path).map_err(|e| anyhow_str(&e))?;
        let audio_s = chunk.samples.len() as f64 / chunk.sample_rate as f64;
        let t = Instant::now();
        let hyp = adapter
            .transcribe_samples_public(&chunk.samples)
            .map_err(|e| anyhow_str(&e.message))?;
        let asr_s = t.elapsed().as_secs_f64();

        let cer = cer_lenient(reference, &hyp);
        let ref_chars = reference.chars().count();
        total_cer_chars += cer * ref_chars as f64;
        total_ref_chars += ref_chars;
        total_audio += audio_s;
        total_asr += asr_s;
        latencies.push(asr_s * 1000.0);
        println!(
            "  {id}: audio={audio_s:.2}s asr={ms:.0}ms cer={cer:.4} hyp={hyp:.80?}",
            ms = asr_s * 1000.0
        );
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len() / 2];
    let p95_idx = ((latencies.len() - 1) as f64 * 0.95).round() as usize;
    let p95 = latencies[p95_idx];

    let weighted_cer = total_cer_chars / total_ref_chars as f64;
    println!(
        "\n=== {} (whisper-rs in-process, {} threads) ===",
        args.model.file_name().unwrap().to_string_lossy(),
        args.threads
    );
    println!("  n              : {}", rows.len());
    println!("  audio total    : {:.2}s", total_audio);
    println!("  asr total      : {:.2}s", total_asr);
    println!(
        "  weighted CER   : {:.4} ({:.2}%)",
        weighted_cer,
        weighted_cer * 100.0
    );
    println!("  RTF            : {:.3}", total_asr / total_audio);
    println!("  p50 / p95      : {:.0} / {:.0} ms", p50, p95);
    println!("  load (cold)    : {:.2}s", load_s);
    Ok(())
}

fn anyhow_str(s: &str) -> Box<dyn std::error::Error> {
    Box::<dyn std::error::Error>::from(s.to_string())
}
