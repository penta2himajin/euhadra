//! Embedding-backend smoke check for CI.
//!
//! Loads an ONNX sentence-embedding bundle and asserts the properties
//! every downstream Tier 1/2 consumer relies on, in each of the five
//! pipeline languages:
//!
//! - the graph loads and its input signature is recognised,
//! - every language produces a same-width, all-finite, unit-norm vector,
//! - a string is maximally similar to itself, and
//! - an unrelated string is measurably less similar than a related one.
//!
//! This is deliberately **not** an accuracy benchmark. No gold data
//! exists for the multilingual behaviour of `ParagraphSplitter` or
//! `PhonemeCorrector` (`docs/model-upgrade-candidates.md` §5), so the
//! check asserts only that the backend is functional per language —
//! enough to catch a broken CJK tokenizer, a bad quantised export, or
//! an input-signature regression, which is what silently degraded
//! before `EmbeddingBackend` probed the signature.
//!
//! Exits non-zero on the first failed assertion.
//!
//! ```text
//! EMBEDDER_MODEL=granite EMBEDDER_QUANT=int8 scripts/setup_embedders.sh
//! cargo run --release --features onnx --example check_embedder -- \
//!     --model-dir vendor/embedder_granite_97m
//! ```

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use euhadra::embedding::{cosine, EmbeddingBackend};

#[derive(Parser, Debug)]
#[command(about = "Assert an ONNX embedding backend is functional in all pipeline languages")]
struct Cli {
    /// Directory holding `model.onnx` + `tokenizer.json`.
    #[arg(long)]
    model_dir: PathBuf,

    /// Minimum margin by which the related pair must out-score the
    /// unrelated pair, per language.
    #[arg(long, default_value_t = 0.01)]
    min_margin: f32,
}

/// One language's probe: a term, something close to it, and something
/// unrelated. Kept short on purpose — Tier 1/2 embed words and single
/// sentences, not documents.
struct Probe {
    lang: &'static str,
    anchor: &'static str,
    related: &'static str,
    unrelated: &'static str,
}

const PROBES: &[Probe] = &[
    Probe {
        lang: "en",
        anchor: "database",
        related: "database server",
        unrelated: "strawberry jam",
    },
    Probe {
        lang: "ja",
        anchor: "データベース",
        related: "データベースサーバー",
        unrelated: "いちごジャム",
    },
    Probe {
        lang: "zh",
        anchor: "数据库",
        related: "数据库服务器",
        unrelated: "草莓果酱",
    },
    Probe {
        lang: "ko",
        anchor: "데이터베이스",
        related: "데이터베이스 서버",
        unrelated: "딸기 잼",
    },
    Probe {
        lang: "es",
        anchor: "base de datos",
        related: "servidor de base de datos",
        unrelated: "mermelada de fresa",
    },
];

/// Check a vector is usable: non-empty, all finite, unit L2 norm.
///
/// Returns the reason on failure so the caller can report which
/// language and string produced it.
fn check_vector(v: &[f32]) -> Result<(), String> {
    if v.is_empty() {
        return Err("empty embedding".into());
    }
    if let Some(bad) = v.iter().find(|x| !x.is_finite()) {
        return Err(format!("non-finite component {bad}"));
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if (norm - 1.0).abs() > 1e-3 {
        return Err(format!("not unit-norm (|v| = {norm:.6})"));
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[fail] {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();

    let load_start = Instant::now();
    let mut backend = EmbeddingBackend::load(&cli.model_dir)
        .map_err(|e| format!("loading {}: {e}", cli.model_dir.display()))?;
    println!(
        "backend  : {} ({:?}, loaded in {:.0} ms)",
        cli.model_dir.display(),
        backend.signature(),
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    let mut dim: Option<usize> = None;
    let mut failures = Vec::new();

    for probe in PROBES {
        let mut embed = |text: &str| -> Result<Vec<f32>, String> {
            backend
                .embed(text)
                .map_err(|e| format!("{}: embedding {text:?}: {e}", probe.lang))
        };

        let anchor = embed(probe.anchor)?;
        let related = embed(probe.related)?;
        let unrelated = embed(probe.unrelated)?;

        for (label, v) in [
            (probe.anchor, &anchor),
            (probe.related, &related),
            (probe.unrelated, &unrelated),
        ] {
            if let Err(why) = check_vector(v) {
                failures.push(format!("{}: {label:?}: {why}", probe.lang));
            }
        }

        // Width must agree across languages, or the downstream cosine
        // silently returns 0.0 for cross-language comparisons.
        match dim {
            None => dim = Some(anchor.len()),
            Some(d) if d != anchor.len() => failures.push(format!(
                "{}: dimension {} disagrees with {d}",
                probe.lang,
                anchor.len()
            )),
            _ => {}
        }

        let self_sim = cosine(&anchor, &anchor);
        if (self_sim - 1.0).abs() > 1e-3 {
            failures.push(format!("{}: self-similarity {self_sim:.6} != 1", probe.lang));
        }

        let related_sim = cosine(&anchor, &related);
        let unrelated_sim = cosine(&anchor, &unrelated);
        let margin = related_sim - unrelated_sim;
        let verdict = if margin >= cli.min_margin { "ok" } else { "FAIL" };
        println!(
            "  {:<3} dim={:<4} related={related_sim:.4}  unrelated={unrelated_sim:.4}  \
             margin={margin:+.4}  {verdict}",
            probe.lang,
            anchor.len()
        );
        if margin < cli.min_margin {
            failures.push(format!(
                "{}: related/unrelated margin {margin:+.4} below {:.4}",
                probe.lang, cli.min_margin
            ));
        }
    }

    if failures.is_empty() {
        println!("\nall {} languages ok", PROBES.len());
        Ok(())
    } else {
        for f in &failures {
            eprintln!("[fail] {f}");
        }
        Err(format!("{} check(s) failed", failures.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_vector_passes() {
        assert!(check_vector(&[0.6, 0.8]).is_ok());
    }

    #[test]
    fn empty_vector_fails() {
        assert!(check_vector(&[]).unwrap_err().contains("empty"));
    }

    #[test]
    fn non_finite_component_fails() {
        assert!(check_vector(&[f32::NAN, 1.0])
            .unwrap_err()
            .contains("non-finite"));
        assert!(check_vector(&[f32::INFINITY, 0.0])
            .unwrap_err()
            .contains("non-finite"));
    }

    #[test]
    fn unnormalised_vector_fails() {
        assert!(check_vector(&[3.0, 4.0]).unwrap_err().contains("unit-norm"));
    }

    #[test]
    fn probes_cover_every_pipeline_language() {
        let langs: Vec<&str> = PROBES.iter().map(|p| p.lang).collect();
        for expected in ["en", "ja", "zh", "ko", "es"] {
            assert!(langs.contains(&expected), "no probe for {expected}");
        }
    }

    #[test]
    fn probe_strings_are_distinct_within_a_language() {
        for p in PROBES {
            assert_ne!(p.anchor, p.related, "{}", p.lang);
            assert_ne!(p.anchor, p.unrelated, "{}", p.lang);
        }
    }
}
