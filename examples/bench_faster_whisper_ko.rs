//! FLEURS-ko bench for the `FasterWhisperAdapter` (CTranslate2 backend).
//!
//! Same shape as `bench_whisper_rs_ko`: loads the model once, runs each
//! utt sequentially, times only the inference call, reports lenient
//! CER + RTF via `eval::metrics::cer_lenient`. Throwaway for the
//! issue #83 measurement work.

use clap::Parser;
use euhadra::eval::metrics::cer_lenient;
use euhadra::faster_whisper::{FasterWhisperAdapter, FasterWhisperConfig};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    /// Directory containing the CTranslate2 Whisper bundle
    /// (e.g. `deepdml/faster-whisper-large-v3-turbo-ct2`).
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value = "/tmp/fleurs/ko")]
    fleurs_dir: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let manifest = args.fleurs_dir.join("manifest.tsv");
    let raw = std::fs::read_to_string(&manifest)?;
    let rows: Vec<(String, PathBuf, String)> = raw
        .lines()
        .skip(1)
        .filter_map(|line| {
            let mut parts = line.splitn(3, '\t');
            let id = parts.next()?.to_string();
            let rel = parts.next()?;
            let reference = parts.next()?.to_string();
            Some((id, args.fleurs_dir.parent().unwrap().join(rel), reference))
        })
        .collect();

    println!("Loading {} ...", args.model_dir.display());
    let t0 = Instant::now();
    let adapter = FasterWhisperAdapter::load_with_config(
        args.model_dir.clone(),
        FasterWhisperConfig {
            language: Some("ko".into()),
            return_timestamps: false,
        },
    )?;
    let load_s = t0.elapsed().as_secs_f64();
    println!(
        "Loaded in {:.2}s (sampling_rate={})",
        load_s,
        adapter.sampling_rate()
    );

    // Warm-up on first utt
    let (_, first_wav, _) = &rows[0];
    let samples = euhadra::whisper_local::read_wav(first_wav).map_err(|e| anyhow_str(&e))?;
    let _ = adapter
        .transcribe_samples(&samples.samples)
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
            .transcribe_samples(&chunk.samples)
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
        "\n=== {} (faster-whisper / ct2rs) ===",
        args.model_dir.file_name().unwrap().to_string_lossy()
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
