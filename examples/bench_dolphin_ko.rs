//! Smoke bench: `DolphinAdapter` on FLEURS-ko.
//!
//! The port's acceptance test is not "is the CER plausible" but "does
//! it reproduce the reference runtime". `docs/korean-asr-alternatives.md`
//! §I measured Dolphin through sherpa-onnx's Python bindings and chose
//! it on those numbers; this bench drives the Rust adapter over the same
//! manifest so the two can be diffed transcript by transcript.
//!
//! Pass `--hypotheses` to dump `id<TAB>hypothesis`, which is the same
//! two-column form `scripts/run_sherpa_ctc.py` emits and
//! `examples/score_hypotheses.rs` reads — so a byte-level `diff`
//! against the reference is one command.
//!
//! Usage:
//!   scripts/setup_dolphin_ko.sh
//!   cargo run --release --features onnx --example bench_dolphin_ko -- \
//!       --model-dir vendor/dolphin_ko \
//!       --manifest data/fleurs_subset/ko/manifest.tsv \
//!       --audio-root data/fleurs_subset

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use euhadra::dolphin::{DolphinAdapter, DolphinConfig};
use euhadra::eval::metrics::cer_lenient;
use euhadra::whisper_local::read_wav;

#[derive(Parser)]
struct Cli {
    /// Bundle directory: `model.int8.onnx` + `tokens.txt`.
    #[arg(long)]
    model_dir: PathBuf,

    /// `id<TAB>audio_path<TAB>reference` TSV (header row skipped).
    #[arg(long)]
    manifest: PathBuf,

    /// Prefix for the manifest's relative audio paths.
    #[arg(long)]
    audio_root: PathBuf,

    /// Model filename inside the bundle.
    #[arg(long, default_value = "model.int8.onnx")]
    model_file: String,

    /// Also write `id<TAB>hypothesis` here, for diffing against
    /// `scripts/run_sherpa_ctc.py`'s output.
    #[arg(long)]
    hypotheses: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();

    let cfg = DolphinConfig {
        model_file: cli.model_file.clone(),
    };
    println!(
        "loading {} ({})...",
        cli.model_dir.display(),
        cli.model_file
    );
    let t0 = Instant::now();
    let adapter = DolphinAdapter::load_with_config(&cli.model_dir, cfg).expect("load");
    println!(
        "loaded in {:.1}s (vocab {})",
        t0.elapsed().as_secs_f64(),
        adapter.vocab_size()
    );

    let manifest = std::fs::read_to_string(&cli.manifest).expect("read manifest");
    let rows: Vec<(String, PathBuf, String)> = manifest
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split('\t');
            let uid = it.next().expect("id column").to_string();
            let audio_rel = it.next().expect("audio_path column");
            let reference = it.next().expect("reference column").to_string();
            (uid, cli.audio_root.join(audio_rel), reference)
        })
        .collect();

    // Warm-up so the first utterance's timing isn't the allocator's.
    let warm = read_wav(&rows[0].1).expect("warm wav");
    let _ = adapter
        .transcribe_samples(&warm.samples)
        .expect("warm transcribe");

    let mut total_audio = 0.0_f64;
    let mut total_asr = 0.0_f64;
    let mut cer_acc = 0.0_f64;
    let mut counted = 0_usize;
    let mut dump = String::from("id\thypothesis\n");

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
        // A tab in the hypothesis would break the two-column contract.
        dump.push_str(&format!("{uid}\t{}\n", hyp.replace('\t', " ")));
        println!(
            "{uid}: audio={dur:.2}s asr={ms:.0}ms cer={c:.4} hyp={hyp:?}",
            ms = asr_s * 1000.0
        );
    }

    if let Some(path) = &cli.hypotheses {
        std::fs::write(path, &dump).expect("write hypotheses");
        println!("\nwrote {}", path.display());
    }

    println!();
    println!("=== Dolphin CTC (Rust) ===");
    println!("  utterances    : {}", rows.len());
    println!("  total_audio_s : {total_audio:.2}");
    println!("  total_asr_s   : {total_asr:.2}");
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
