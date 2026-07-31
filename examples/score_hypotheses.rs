//! Score an external ASR's output with euhadra's own metrics.
//!
//! Candidate backends are evaluated outside the Rust stack before
//! anyone commits to porting them — `docs/korean-asr-alternatives.md`
//! §A.1 measured transformers and whisper.cpp that way, and §A.2 the
//! ONNX quantisation variants. The risk in that workflow is scoring
//! the candidate with a *different* normaliser than the incumbent,
//! which silently makes the comparison meaningless.
//!
//! So the candidate runs wherever it runs, writes `id<TAB>hypothesis`,
//! and the numbers come from `eval::metrics` here — the same functions
//! the committed baselines use.
//!
//! ```text
//! python3 run_some_candidate.py > /tmp/hyp.tsv
//! cargo run --release --example score_hypotheses -- \
//!     --manifest data/fleurs_subset/ko/manifest.tsv \
//!     --hypotheses /tmp/hyp.tsv \
//!     --metric cer
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use euhadra::eval::metrics::{cer_lenient, wer_lenient};

#[derive(Parser, Debug)]
#[command(about = "Score an external ASR's hypotheses with euhadra's lenient metrics")]
struct Cli {
    /// `id<TAB>audio_path<TAB>reference` TSV, as written by
    /// `scripts/download_fleurs_subset.py` (header row skipped).
    #[arg(long)]
    manifest: PathBuf,

    /// `id<TAB>hypothesis` TSV. A header row is skipped when its first
    /// field is literally `id`.
    #[arg(long)]
    hypotheses: PathBuf,

    #[arg(long, value_enum, default_value_t = Metric::Cer)]
    metric: Metric,

    /// Print the per-utterance score as well as the aggregate.
    #[arg(long)]
    verbose: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Metric {
    Cer,
    Wer,
}

/// Parse `id<TAB>...` rows, taking column `value_col` as the value.
///
/// Split from the file read so the column indexing and header handling
/// are testable — a silent off-by-one here would compare hypotheses
/// against the wrong field and produce a plausible-looking number.
fn parse_tsv(raw: &str, value_col: usize) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.first() == Some(&"id") {
            continue; // header
        }
        let Some(id) = cols.first() else { continue };
        let value = cols.get(value_col).copied().unwrap_or("");
        out.insert((*id).to_string(), value.to_string());
    }
    out
}

fn read_tsv(path: &PathBuf, value_col: usize) -> Result<BTreeMap<String, String>, String> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    Ok(parse_tsv(&raw, value_col))
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();

    let references = read_tsv(&cli.manifest, 2)?;
    let hypotheses = read_tsv(&cli.hypotheses, 1)?;
    if references.is_empty() {
        return Err(format!("{} has no rows", cli.manifest.display()));
    }

    let score = |r: &str, h: &str| match cli.metric {
        Metric::Cer => cer_lenient(r, h),
        Metric::Wer => wer_lenient(r, h),
    };

    let mut scores = Vec::new();
    let mut missing = Vec::new();
    for (id, reference) in &references {
        match hypotheses.get(id) {
            // A missing hypothesis is scored as a total miss rather
            // than skipped: dropping it would quietly reward a
            // candidate that failed on some utterances.
            None => {
                missing.push(id.clone());
                scores.push((id.clone(), score(reference, "")));
            }
            Some(h) => scores.push((id.clone(), score(reference, h))),
        }
    }

    if cli.verbose {
        for (id, s) in &scores {
            println!("{id}\t{s:.4}");
        }
    }

    let finite: Vec<f64> = scores.iter().map(|(_, s)| *s).filter(|s| s.is_finite()).collect();
    let mean = if finite.is_empty() {
        f64::NAN
    } else {
        finite.iter().sum::<f64>() / finite.len() as f64
    };

    println!("utterances    : {}", references.len());
    if !missing.is_empty() {
        println!("missing hyps  : {} ({})", missing.len(), missing.join(", "));
    }
    println!(
        "{}_lenient   : {mean:.4}",
        match cli.metric {
            Metric::Cer => "cer",
            Metric::Wer => "wer",
        }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_tsv;

    #[test]
    fn manifest_column_two_is_the_reference() {
        let raw = "id\taudio_path\treference\n1\tko/audio/1.wav\t안녕하세요\n";
        let m = parse_tsv(raw, 2);
        assert_eq!(m.get("1").map(String::as_str), Some("안녕하세요"));
    }

    #[test]
    fn hypothesis_column_one_is_the_text() {
        let raw = "id\thypothesis\n1\t안녕하세요\n";
        let m = parse_tsv(raw, 1);
        assert_eq!(m.get("1").map(String::as_str), Some("안녕하세요"));
    }

    #[test]
    fn a_header_row_is_skipped_but_only_a_real_one() {
        // "id" as the first field means header; an utterance genuinely
        // called something else must survive.
        let m = parse_tsv("id\tx\nident\thello\n", 1);
        assert_eq!(m.len(), 1);
        assert_eq!(m.get("ident").map(String::as_str), Some("hello"));
    }

    #[test]
    fn a_missing_column_reads_as_empty_not_a_panic() {
        // A candidate that emitted nothing for an utterance writes a
        // bare id; that must score as a total miss, not crash.
        let m = parse_tsv("7\n", 1);
        assert_eq!(m.get("7").map(String::as_str), Some(""));
    }

    #[test]
    fn blank_lines_are_ignored() {
        let m = parse_tsv("1\ta\n\n\n2\tb\n", 1);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn tabs_inside_the_value_are_not_re_split() {
        // Column indexing takes the field at `value_col`; anything
        // after it belongs to later columns, so a hypothesis is never
        // silently truncated at its first tab when it is the last one.
        let m = parse_tsv("1\tone\ttwo\n", 1);
        assert_eq!(m.get("1").map(String::as_str), Some("one"));
    }
}
