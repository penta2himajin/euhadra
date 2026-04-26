//! L2 evaluation runner — Phase B (standard + Robust).
//!
//! Drives the heavyweight ASR benchmarks documented in
//! `docs/evaluation.md` §1.2 and §4. Designed to be invoked via
//! `cargo eval-l2` (alias in `.cargo/config.toml`) and to be runnable
//! end-to-end from a clean checkout: if the requested dataset is not
//! present under `--data-dir`, the runner shells out to
//! `scripts/download_l2_data.sh` (LibriSpeech / AISHELL-1 / MUSAN /
//! OpenSLR SLR26) or `scripts/download_l2_data.py` (ReazonSpeech via
//! HuggingFace) before evaluation.
//!
//! Why this lives in `examples/` rather than `src/`:
//! - `rustfft` is the only heavy build dependency, and we want default
//!   `cargo build` to stay lean. Pulling it in via `[dev-dependencies]`
//!   keeps that contract (examples have access to dev-deps).
//! - The L2 augmentation (noise mixing, RIR convolution) has no callers
//!   inside the library; promoting it would be premature.
//!
//! L2-Robust per §4.6:
//!   condition × {SNR ∈ {20, 10, 5, 0} dB for noise variants}
//!   condition ∈ {baseline, +noise, +reverb, +noise+reverb}
//!
//! Selection of (noise sample, RIR sample) per utterance is determined
//! by `(seed, utterance_id)`, so two invocations with the same seed mix
//! identical audio without committing the mixed waveforms (§5.2).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use euhadra::eval::metrics::{cer, wer};
use euhadra::prelude::*;
use euhadra::whisper_local::{WhisperLocal, read_wav};

// ---------------------------------------------------------------------------
// Audio augmentation (inline so the dev-dep `rustfft` doesn't leak into
// the library surface).
// ---------------------------------------------------------------------------
mod audio_aug {
    use rustfft::num_complex::Complex32;
    use rustfft::FftPlanner;
    use std::path::{Path, PathBuf};

    use euhadra::types::AudioChunk;

    /// Recursively collect WAV file paths under `root`. Used to build
    /// the noise / RIR pools at startup. Sorted for deterministic
    /// indexing under the same seed.
    pub fn collect_wavs(root: &Path) -> std::io::Result<Vec<PathBuf>> {
        if !root.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("pool root {} does not exist", root.display()),
            ));
        }
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir)? {
                let entry = entry?;
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|e| e == "wav") {
                    out.push(p);
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// Down-mix interleaved multi-channel audio to mono by averaging
    /// channels. Returns a new sample buffer.
    pub fn to_mono(chunk: &AudioChunk) -> Vec<f32> {
        if chunk.channels <= 1 {
            return chunk.samples.clone();
        }
        let ch = chunk.channels as usize;
        let frames = chunk.samples.len() / ch;
        let mut out = Vec::with_capacity(frames);
        for f in 0..frames {
            let mut sum = 0.0_f32;
            for c in 0..ch {
                sum += chunk.samples[f * ch + c];
            }
            out.push(sum / ch as f32);
        }
        out
    }

    /// Convolve `signal` with `rir` via FFT (length matched to next
    /// power of two ≥ output length). Both are mono f32 at the same
    /// sample rate; the caller is responsible for resampling. Returns
    /// `signal.len() + rir.len() - 1` samples.
    pub fn fft_convolve(signal: &[f32], rir: &[f32]) -> Vec<f32> {
        let out_len = signal.len() + rir.len().saturating_sub(1).max(1);
        let n = out_len.next_power_of_two().max(2);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        let mut s_buf: Vec<Complex32> = signal
            .iter()
            .map(|&x| Complex32::new(x, 0.0))
            .collect();
        s_buf.resize(n, Complex32::new(0.0, 0.0));

        let mut r_buf: Vec<Complex32> = rir
            .iter()
            .map(|&x| Complex32::new(x, 0.0))
            .collect();
        r_buf.resize(n, Complex32::new(0.0, 0.0));

        fft.process(&mut s_buf);
        fft.process(&mut r_buf);

        for (s, r) in s_buf.iter_mut().zip(r_buf.iter()) {
            *s *= *r;
        }
        ifft.process(&mut s_buf);

        let scale = 1.0 / n as f32;
        s_buf
            .into_iter()
            .take(out_len)
            .map(|c| c.re * scale)
            .collect()
    }

    /// Mix `noise` into `signal` so that the resulting SNR matches
    /// `snr_db`. Noise is repeated (or truncated) to match signal
    /// length. Output is clamped to ±1.0 to stay valid 16-bit PCM after
    /// re-quantisation by the caller.
    pub fn add_noise(signal: &mut [f32], noise: &[f32], snr_db: f64) {
        let signal_power = power(signal);
        let noise_power = power(noise);
        if noise_power <= 0.0 || signal_power <= 0.0 || noise.is_empty() {
            return;
        }
        let snr_linear = 10.0_f64.powf(snr_db / 10.0);
        let target_noise_power = signal_power / snr_linear;
        let scale = (target_noise_power / noise_power).sqrt() as f32;
        for (i, s) in signal.iter_mut().enumerate() {
            let n = noise[i % noise.len()] * scale;
            *s = (*s + n).clamp(-1.0, 1.0);
        }
    }

    /// Rescale `samples` so that the max absolute amplitude equals
    /// `target_peak`. No-op on silence.
    pub fn peak_normalise(samples: &mut [f32], target_peak: f32) {
        let mut peak = 0.0_f32;
        for &s in samples.iter() {
            let a = s.abs();
            if a > peak {
                peak = a;
            }
        }
        if peak < 1e-9 {
            return;
        }
        let scale = target_peak / peak;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    }

    fn power(samples: &[f32]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0_f64;
        for &s in samples {
            sum += (s as f64) * (s as f64);
        }
        sum / samples.len() as f64
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fft_convolve_with_unit_impulse_is_identity() {
            let signal: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
            let rir = vec![1.0, 0.0, 0.0, 0.0];
            let out = fft_convolve(&signal, &rir);
            for (i, expected) in signal.iter().enumerate() {
                let got = out[i];
                assert!((got - expected).abs() < 1e-4, "i={i} got={got} expected={expected}");
            }
        }

        #[test]
        fn fft_convolve_with_delta_at_offset_shifts_signal() {
            let signal: Vec<f32> = vec![1.0, 0.5, 0.25, 0.125];
            let rir = vec![0.0, 0.0, 1.0]; // delay by 2 samples
            let out = fft_convolve(&signal, &rir);
            assert!(out[0].abs() < 1e-4);
            assert!(out[1].abs() < 1e-4);
            assert!((out[2] - 1.0).abs() < 1e-4);
            assert!((out[3] - 0.5).abs() < 1e-4);
            assert!((out[4] - 0.25).abs() < 1e-4);
        }

        #[test]
        fn add_noise_hits_target_snr() {
            // Pure tone signal, white-ish noise, target 10 dB.
            let signal_orig: Vec<f32> = (0..16000)
                .map(|i| (i as f32 * 0.05).sin() * 0.5)
                .collect();
            let noise: Vec<f32> = (0..16000)
                .map(|i| ((i as f32 * 0.13).sin() + (i as f32 * 0.27).cos()) * 0.5)
                .collect();
            let mut signal = signal_orig.clone();
            add_noise(&mut signal, &noise, 10.0);

            // Reconstruct the noise component from (mixed - original)
            // and compare its power to the signal power.
            let resid_power: f64 = signal
                .iter()
                .zip(signal_orig.iter())
                .map(|(m, s)| (m - s) as f64)
                .map(|d| d * d)
                .sum::<f64>()
                / signal.len() as f64;
            let signal_power: f64 = signal_orig
                .iter()
                .map(|s| (*s as f64).powi(2))
                .sum::<f64>()
                / signal_orig.len() as f64;
            let achieved_db = 10.0 * (signal_power / resid_power).log10();
            assert!(
                (achieved_db - 10.0).abs() < 0.5,
                "expected ~10 dB SNR, got {achieved_db}"
            );
        }

        #[test]
        fn peak_normalise_scales_to_target() {
            let mut s = vec![0.1, -0.2, 0.05];
            peak_normalise(&mut s, 0.5);
            assert!((s[1].abs() - 0.5).abs() < 1e-6);
            assert!((s[0] - 0.25).abs() < 1e-6);
        }

        #[test]
        fn to_mono_averages_stereo() {
            let chunk = AudioChunk {
                samples: vec![1.0, 0.0, 0.5, -0.5, 0.2, 0.4],
                sample_rate: 16000,
                channels: 2,
            };
            let mono = to_mono(&chunk);
            assert_eq!(mono, vec![0.5, 0.0, 0.3]);
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "L2: standard + Robust ASR benchmark with WER/CER + RTF + latency")]
struct Cli {
    /// Logical dataset name. Each maps to `<data-dir>/<dataset>/manifest.tsv`
    /// produced by `scripts/download_l2_data.{sh,py}`.
    #[arg(long, value_enum)]
    dataset: Dataset,

    /// Augmentation condition. `baseline` runs the unmodified audio.
    #[arg(long, value_enum, default_value_t = Condition::Baseline)]
    condition: Condition,

    /// SNR (in dB) for `noise` / `noise_reverb` conditions. Ignored for
    /// `baseline` / `reverb`.
    #[arg(long, default_value_t = 10.0)]
    snr_db: f64,

    /// PRNG seed used to deterministically choose noise / RIR sample per
    /// utterance. Two runs with the same seed mix identical audio (§4.6).
    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, env = "WHISPER_CLI_PATH")]
    whisper_cli: PathBuf,

    /// English model (ggml-tiny.en.bin or larger). Used for LibriSpeech.
    #[arg(long, env = "WHISPER_MODEL_EN")]
    model_en: PathBuf,

    /// Multilingual model (ggml-tiny.bin or larger). Used for ja/zh.
    #[arg(long, env = "WHISPER_MODEL_MULTI")]
    model_multi: PathBuf,

    /// Root directory containing the per-dataset subfolders.
    #[arg(long, env = "L2_DATA_DIR", default_value = "data/l2")]
    data_dir: PathBuf,

    /// Directory containing MUSAN noise WAVs (`*.wav` recursively).
    /// Required for `--condition noise` / `noise_reverb`.
    #[arg(long, default_value = "data/l2/musan-noise")]
    noise_dir: PathBuf,

    /// Directory containing RIR WAVs (`*.wav` recursively). Required for
    /// `--condition reverb` / `noise_reverb`.
    #[arg(long, default_value = "data/l2/rir-slr26")]
    rir_dir: PathBuf,

    /// Cap evaluation at the first N utterances of the manifest (handy
    /// for smoke-testing without running the full 2,500-utterance set).
    #[arg(long)]
    max_utterances: Option<usize>,

    /// Where to write the JSON benchmark report. Stdout summary is
    /// always emitted.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Skip dataset download even if `--data-dir/<dataset>/manifest.tsv`
    /// is missing (fails fast instead).
    #[arg(long)]
    no_setup: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Dataset {
    LibrispeechTestClean,
    LibrispeechTestOther,
    Aishell1Test,
    ReazonspeechTest,
}

impl Dataset {
    fn dir_name(self) -> &'static str {
        match self {
            Dataset::LibrispeechTestClean => "librispeech-test-clean",
            Dataset::LibrispeechTestOther => "librispeech-test-other",
            Dataset::Aishell1Test => "aishell1-test",
            Dataset::ReazonspeechTest => "reazonspeech-test",
        }
    }

    fn language(self) -> &'static str {
        match self {
            Dataset::LibrispeechTestClean | Dataset::LibrispeechTestOther => "en",
            Dataset::Aishell1Test => "zh",
            Dataset::ReazonspeechTest => "ja",
        }
    }

    /// Setup script + arg used when the manifest is missing.
    fn setup_invocation(self) -> SetupInvocation {
        match self {
            Dataset::LibrispeechTestClean => SetupInvocation::Sh("librispeech-test-clean"),
            Dataset::LibrispeechTestOther => SetupInvocation::Sh("librispeech-test-other"),
            Dataset::Aishell1Test => SetupInvocation::Sh("aishell1-test"),
            Dataset::ReazonspeechTest => SetupInvocation::Py("reazonspeech-test"),
        }
    }
}

enum SetupInvocation {
    Sh(&'static str),
    Py(&'static str),
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum Condition {
    Baseline,
    Noise,
    Reverb,
    NoiseReverb,
}

impl Condition {
    fn label(self, snr_db: f64) -> String {
        match self {
            Condition::Baseline => "baseline".to_string(),
            Condition::Noise => format!("noise@{snr_db:.0}dB"),
            Condition::Reverb => "reverb".to_string(),
            Condition::NoiseReverb => format!("noise+reverb@{snr_db:.0}dB"),
        }
    }
    fn needs_noise(self) -> bool {
        matches!(self, Condition::Noise | Condition::NoiseReverb)
    }
    fn needs_reverb(self) -> bool {
        matches!(self, Condition::Reverb | Condition::NoiseReverb)
    }
}

#[derive(Debug)]
struct ManifestRow {
    id: String,
    audio_path: PathBuf,
    reference: String,
}

fn load_manifest(path: &Path) -> std::io::Result<Vec<ManifestRow>> {
    let raw = std::fs::read_to_string(path)?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut rows = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.splitn(3, '\t').collect();
        if cols.len() != 3 {
            continue;
        }
        rows.push(ManifestRow {
            id: cols[0].to_string(),
            audio_path: parent.join(cols[1]),
            reference: cols[2].to_string(),
        });
    }
    Ok(rows)
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}

async fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let dataset_dir = cli.data_dir.join(cli.dataset.dir_name());
    let manifest_path = dataset_dir.join("manifest.tsv");

    if !manifest_path.exists() {
        if cli.no_setup {
            return Err(format!(
                "{} missing and --no-setup was given. Run scripts/download_l2_data.{{sh,py}} first.",
                manifest_path.display()
            ));
        }
        run_setup(cli.dataset, &cli.data_dir)?;
    }

    let mut manifest = load_manifest(&manifest_path)
        .map_err(|e| format!("loading {}: {e}", manifest_path.display()))?;
    if let Some(cap) = cli.max_utterances {
        manifest.truncate(cap);
    }
    if manifest.is_empty() {
        return Err(format!("manifest {} is empty", manifest_path.display()));
    }

    // For Robust conditions we collect the noise / RIR pools once.
    let noise_pool = if cli.condition.needs_noise() {
        let pool = audio_aug::collect_wavs(&cli.noise_dir)
            .map_err(|e| format!("loading noise pool: {e}"))?;
        if pool.is_empty() {
            return Err(format!("noise pool at {} is empty", cli.noise_dir.display()));
        }
        Some(pool)
    } else {
        None
    };
    let rir_pool = if cli.condition.needs_reverb() {
        let pool = audio_aug::collect_wavs(&cli.rir_dir)
            .map_err(|e| format!("loading RIR pool: {e}"))?;
        if pool.is_empty() {
            return Err(format!("RIR pool at {} is empty", cli.rir_dir.display()));
        }
        Some(pool)
    } else {
        None
    };

    let lang = cli.dataset.language();
    let model = if lang == "en" { &cli.model_en } else { &cli.model_multi };
    let result = evaluate(
        &cli,
        lang,
        &manifest,
        model,
        noise_pool.as_deref(),
        rir_pool.as_deref(),
    )
    .await?;
    print_summary(&cli, &result);

    if let Some(out) = &cli.output {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let json = serde_json::to_string_pretty(&result.as_report(&cli)).map_err(|e| format!("json: {e}"))?;
        std::fs::write(out, json).map_err(|e| format!("write {}: {e}", out.display()))?;
        eprintln!("report written to {}", out.display());
    }
    Ok(())
}

fn run_setup(dataset: Dataset, data_dir: &Path) -> Result<(), String> {
    eprintln!(
        "[setup] {} not found; running download script",
        data_dir.join(dataset.dir_name()).display()
    );
    let (cmd_program, script, arg) = match dataset.setup_invocation() {
        SetupInvocation::Sh(name) => ("bash", "scripts/download_l2_data.sh", name),
        SetupInvocation::Py(name) => ("python3", "scripts/download_l2_data.py", name),
    };
    let status = Command::new(cmd_program)
        .arg(script)
        .arg(arg)
        .arg("--out")
        .arg(data_dir)
        .status()
        .map_err(|e| format!("spawn {script} {arg}: {e}"))?;
    if !status.success() {
        return Err(format!("{script} {arg} exited with {status}"));
    }
    Ok(())
}

#[derive(Debug)]
struct EvalResult {
    samples: usize,
    skipped: usize,
    mean_wer: f64,
    mean_cer: f64,
    rtf: f64,
    asr_p50_ms: f64,
    asr_p95_ms: f64,
    e2e_p50_ms: f64,
    e2e_p95_ms: f64,
}

impl EvalResult {
    fn as_report(&self, cli: &Cli) -> serde_json::Value {
        let lang = cli.dataset.language();
        let primary = if lang == "en" {
            ("wer", self.mean_wer)
        } else {
            ("cer", self.mean_cer)
        };
        serde_json::json!({
            "dataset": cli.dataset.dir_name(),
            "language": lang,
            "condition": cli.condition.label(cli.snr_db),
            "snr_db": if cli.condition.needs_noise() { Some(cli.snr_db) } else { None },
            "seed": cli.seed,
            "samples": self.samples,
            "skipped": self.skipped,
            primary.0: round4(primary.1),
            "rtf": round4(self.rtf),
            "asr_latency_ms": {"p50": round1(self.asr_p50_ms), "p95": round1(self.asr_p95_ms)},
            "e2e_latency_ms": {"p50": round1(self.e2e_p50_ms), "p95": round1(self.e2e_p95_ms)},
        })
    }
}

async fn evaluate(
    cli: &Cli,
    lang: &str,
    manifest: &[ManifestRow],
    model: &Path,
    noise_pool: Option<&[PathBuf]>,
    rir_pool: Option<&[PathBuf]>,
) -> Result<EvalResult, String> {
    let mut wer_sum = 0.0;
    let mut cer_sum = 0.0;
    let mut counted = 0;
    let mut skipped = 0;
    let mut asr_durations: Vec<f64> = Vec::new();
    let mut e2e_durations: Vec<f64> = Vec::new();
    let mut total_audio_secs = 0.0_f64;
    let mut total_asr_secs = 0.0_f64;

    for row in manifest {
        let mut audio = match read_wav(&row.audio_path) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("[skip] {}: {e}", row.id);
                skipped += 1;
                continue;
            }
        };
        let audio_secs =
            audio.samples.len() as f64 / (audio.sample_rate as f64 * audio.channels as f64);

        if cli.condition.needs_reverb() {
            let pool = rir_pool.expect("rir pool guaranteed");
            let rir_path = pick(pool, cli.seed, &row.id, "rir");
            let rir = read_wav(rir_path).map_err(|e| format!("read RIR {}: {e}", rir_path.display()))?;
            let rir_mono = audio_aug::to_mono(&rir);
            audio.samples = audio_aug::fft_convolve(&audio.samples, &rir_mono);
            // Convolution may push amplitude > 1.0; renormalise to keep
            // the WAV writer happy and the SNR math correct.
            audio_aug::peak_normalise(&mut audio.samples, 0.95);
        }
        if cli.condition.needs_noise() {
            let pool = noise_pool.expect("noise pool guaranteed");
            let noise_path = pick(pool, cli.seed, &row.id, "noise");
            let noise = read_wav(noise_path).map_err(|e| format!("read noise {}: {e}", noise_path.display()))?;
            let noise_mono = audio_aug::to_mono(&noise);
            audio_aug::add_noise(&mut audio.samples, &noise_mono, cli.snr_db);
        }

        let pipeline = build_pipeline(&cli.whisper_cli, model, lang)?;
        let e2e_start = Instant::now();
        let (audio_tx, _cancel, handle) = pipeline.session();
        let asr_start = Instant::now();
        audio_tx
            .send(audio)
            .await
            .map_err(|e| format!("send: {e}"))?;
        drop(audio_tx);
        let result = handle
            .await
            .map_err(|e| format!("join: {e}"))?
            .map_err(|e| format!("pipeline: {e}"))?;
        let asr_elapsed = asr_start.elapsed();
        let e2e_elapsed = e2e_start.elapsed();
        asr_durations.push(asr_elapsed.as_secs_f64() * 1000.0);
        e2e_durations.push(e2e_elapsed.as_secs_f64() * 1000.0);
        total_asr_secs += asr_elapsed.as_secs_f64();
        total_audio_secs += audio_secs;

        let hyp = &result.raw_text;
        let w = wer(&row.reference, hyp);
        let c = cer(&row.reference, hyp);
        if !w.is_nan() {
            wer_sum += w;
        }
        if !c.is_nan() {
            cer_sum += c;
        }
        counted += 1;
    }

    if counted == 0 {
        return Err("no scorable utterances".into());
    }

    let (asr_p50, asr_p95) = percentiles(&asr_durations);
    let (e2e_p50, e2e_p95) = percentiles(&e2e_durations);

    Ok(EvalResult {
        samples: counted,
        skipped,
        mean_wer: wer_sum / counted as f64,
        mean_cer: cer_sum / counted as f64,
        rtf: if total_audio_secs > 0.0 {
            total_asr_secs / total_audio_secs
        } else {
            f64::NAN
        },
        asr_p50_ms: asr_p50,
        asr_p95_ms: asr_p95,
        e2e_p50_ms: e2e_p50,
        e2e_p95_ms: e2e_p95,
    })
}

fn pick<'a>(pool: &'a [PathBuf], seed: u64, utt_id: &str, kind: &str) -> &'a PathBuf {
    let mut h = DefaultHasher::new();
    seed.hash(&mut h);
    utt_id.hash(&mut h);
    kind.hash(&mut h);
    let idx = (h.finish() as usize) % pool.len();
    &pool[idx]
}

fn build_pipeline(whisper_cli: &Path, model: &Path, lang: &str) -> Result<Pipeline, String> {
    let asr = WhisperLocal::new(whisper_cli, model).with_language(lang);
    let mut builder = Pipeline::builder()
        .asr(asr)
        .processor(SelfCorrectionDetector::new())
        .processor(BasicPunctuationRestorer)
        .refiner(MockRefiner::passthrough())
        .context(MockContextProvider::new())
        .emitter(MockEmitter::new());
    builder = match lang {
        "en" => builder.filter(SimpleFillerFilter::english()),
        "ja" => builder.filter(JapaneseFillerFilter::new()),
        "zh" => builder.filter(ChineseFillerFilter::new()),
        other => return Err(format!("unsupported lang {other}")),
    };
    builder.build().map_err(|e| format!("build pipeline: {e}"))
}

fn percentiles(samples: &[f64]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |q: f64| -> f64 {
        let n = s.len();
        if n == 1 {
            return s[0];
        }
        let pos = q * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - lo as f64;
        s[lo] + (s[hi] - s[lo]) * frac
    };
    (pct(0.50), pct(0.95))
}

fn print_summary(cli: &Cli, r: &EvalResult) {
    let lang = cli.dataset.language();
    let primary = if lang == "en" {
        format!("WER={:.4}", r.mean_wer)
    } else {
        format!("CER={:.4}", r.mean_cer)
    };
    println!(
        "[{}/{}] n={} skipped={} {} RTF={:.3} asr p50={:.0}ms p95={:.0}ms e2e p50={:.0}ms p95={:.0}ms",
        cli.dataset.dir_name(),
        cli.condition.label(cli.snr_db),
        r.samples,
        r.skipped,
        primary,
        r.rtf,
        r.asr_p50_ms,
        r.asr_p95_ms,
        r.e2e_p50_ms,
        r.e2e_p95_ms,
    );
    // Absolute-threshold warnings (independent of any per-runner
    // baseline). Same thresholds as `Tolerances::default()`.
    if r.rtf >= ABS_RTF_WARN {
        eprintln!(
            "[warn] RTF {:.3} ≥ {:.1} — slower than real-time, breaks streaming dictation",
            r.rtf, ABS_RTF_WARN
        );
    }
    if r.asr_p50_ms >= ABS_LATENCY_WARN_MS {
        eprintln!(
            "[warn] ASR p50 latency {:.0}ms ≥ {:.0}ms",
            r.asr_p50_ms, ABS_LATENCY_WARN_MS
        );
    }
    if r.e2e_p50_ms >= ABS_LATENCY_WARN_MS {
        eprintln!(
            "[warn] E2E p50 latency {:.0}ms ≥ {:.0}ms",
            r.e2e_p50_ms, ABS_LATENCY_WARN_MS
        );
    }
}

/// Absolute thresholds for "user-perceived too-slow" warnings. Same
/// numbers as `Tolerances::default()` so L1 and L2 share the contract.
const ABS_RTF_WARN: f64 = 1.0;
const ABS_LATENCY_WARN_MS: f64 = 1000.0;

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}
