//! Paragraph-segmentation evaluation for `ParagraphSplitter`.
//!
//! `docs/model-upgrade-candidates.md` §5.5 recorded that the splitter
//! had never been scored, in any language: the depth rule that replaced
//! its absolute threshold is provably portable across embedding
//! backends, but nothing established that the valleys it picks are the
//! *right* places to break. This closes that.
//!
//! Corpus from `scripts/download_paragraph_corpus.py`, in two shapes:
//! Choi-style synthetic concatenation (boundaries known by
//! construction) and single articles with their author's own paragraph
//! breaks. Metrics are Pk and WindowDiff — window penalties, **lower is
//! better** — because exact-match F1 treats a break one sentence off as
//! doubly wrong and so cannot distinguish "nearly right" from "random".
//!
//! Every run reports baselines alongside the real segmenters. Without
//! them the numbers mean nothing: splitting at a fixed interval is a
//! surprisingly strong segmenter, and a semantic rule that cannot beat
//! it is not earning its embedding pass.
//!
//! ```text
//! scripts/download_paragraph_corpus.py
//! EMBEDDER_MODEL=granite scripts/setup_embedders.sh
//! cargo run --release --features onnx --example eval_paragraph -- \
//!     --corpus data/paragraph_corpus/choi_en.jsonl \
//!     --embedder-dir vendor/embedder_granite_97m
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use clap::Parser;
use serde::Deserialize;

use euhadra::eval::segmentation::{boundary_counts, default_window, pk, window_diff};
use euhadra::paragraph::{split_sentences, ParagraphSplitter};
use euhadra::phoneme::TextEmbedder;
use euhadra::processor::ProcessError;

#[derive(Parser, Debug)]
#[command(about = "Score ParagraphSplitter against a paragraph-boundary corpus")]
struct Cli {
    /// JSONL from `scripts/download_paragraph_corpus.py`.
    #[arg(long)]
    corpus: PathBuf,

    /// Directory holding `model.onnx` + `tokenizer.json`. Without it
    /// only the embedding-free baselines run.
    #[arg(long)]
    embedder_dir: Option<PathBuf>,

    /// Cap on sentences per paragraph. Defaults high so the semantic
    /// rule is measured on its own rather than swamped by the length
    /// constraint; `uniform-k` is the baseline for that constraint.
    #[arg(long, default_value_t = 10_000)]
    max_sentences: usize,

    /// Optional JSON report path.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct Document {
    doc_id: String,
    lang: String,
    paragraphs: Vec<String>,
}

/// A document flattened into sentences, with the gold break indices
/// implied by its paragraph structure.
struct Prepared {
    doc_id: String,
    sentences: Vec<String>,
    gold: Vec<usize>,
}

fn prepare(doc: &Document) -> Option<Prepared> {
    let mut sentences = Vec::new();
    let mut gold = Vec::new();
    for para in &doc.paragraphs {
        let s = split_sentences(para);
        if s.is_empty() {
            continue;
        }
        if !sentences.is_empty() {
            gold.push(sentences.len());
        }
        sentences.extend(s);
    }
    // Too short to have a judgeable structure.
    if sentences.len() < 4 || gold.is_empty() {
        return None;
    }
    Some(Prepared {
        doc_id: doc.doc_id.clone(),
        sentences,
        gold,
    })
}

/// Memoising wrapper so every segmenter variant shares one pass of
/// embedding work. The corpus is a few thousand sentences and each
/// variant would otherwise re-encode all of them.
struct CachedEmbedder<E: TextEmbedder> {
    inner: E,
    cache: Mutex<HashMap<String, Vec<f32>>>,
}

impl<E: TextEmbedder> CachedEmbedder<E> {
    fn new(inner: E) -> Self {
        Self {
            inner,
            cache: Mutex::new(HashMap::new()),
        }
    }
}

impl<E: TextEmbedder> TextEmbedder for CachedEmbedder<E> {
    fn embed(&self, text: &str) -> Result<Vec<f32>, ProcessError> {
        if let Some(v) = self.cache.lock().unwrap().get(text) {
            return Ok(v.clone());
        }
        let v = self.inner.embed(text)?;
        self.cache
            .lock()
            .unwrap()
            .insert(text.to_string(), v.clone());
        Ok(v)
    }

    fn similarity_floor(&self) -> f32 {
        self.inner.similarity_floor()
    }
}

/// A shared handle to the cache, since `with_embedder` takes ownership
/// and every variant needs the same one.
#[derive(Clone)]
struct SharedCache(Arc<dyn TextEmbedder>);

impl TextEmbedder for SharedCache {
    fn embed(&self, text: &str) -> Result<Vec<f32>, ProcessError> {
        self.0.embed(text)
    }
    fn similarity_floor(&self) -> f32 {
        self.0.similarity_floor()
    }
}

// Baselines -----------------------------------------------------------

fn baseline_none(_n: usize) -> Vec<usize> {
    Vec::new()
}

fn baseline_uniform(n: usize, every: usize) -> Vec<usize> {
    (1..n).filter(|i| i % every == 0).collect()
}

/// Deterministic pseudo-random boundaries, matched in count to the
/// gold segmentation so it is a fair "right number, wrong places"
/// control rather than a degenerate one.
fn baseline_random(n: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut picked: Vec<usize> = Vec::new();
    let mut guard = 0;
    while picked.len() < count && guard < 1000 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let idx = 1 + (state >> 33) as usize % (n.saturating_sub(1)).max(1);
        if !picked.contains(&idx) {
            picked.push(idx);
        }
        guard += 1;
    }
    picked.sort_unstable();
    picked
}

#[derive(Default, Clone)]
struct Score {
    pk: Vec<f64>,
    wd: Vec<f64>,
    tp: usize,
    fp: usize,
    fn_: usize,
    predicted: usize,
}

impl Score {
    fn push(&mut self, n: usize, gold: &[usize], hyp: &[usize]) {
        let k = default_window(n, gold);
        let p = pk(n, gold, hyp, k);
        let w = window_diff(n, gold, hyp, k);
        if p.is_finite() {
            self.pk.push(p);
        }
        if w.is_finite() {
            self.wd.push(w);
        }
        let (tp, fp, fn_) = boundary_counts(gold, hyp);
        self.tp += tp;
        self.fp += fp;
        self.fn_ += fn_;
        self.predicted += hyp.len();
    }

    fn mean_pk(&self) -> f64 {
        mean(&self.pk)
    }
    fn mean_wd(&self) -> f64 {
        mean(&self.wd)
    }
    fn f1(&self) -> f64 {
        let p = ratio(self.tp, self.tp + self.fp);
        let r = ratio(self.tp, self.tp + self.fn_);
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        a as f64 / b as f64
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();

    let raw = std::fs::read_to_string(&cli.corpus)
        .map_err(|e| format!("reading {}: {e}", cli.corpus.display()))?;
    let docs: Vec<Document> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .map_err(|e| format!("parsing {}: {e}", cli.corpus.display()))?;

    let prepared: Vec<Prepared> = docs.iter().filter_map(prepare).collect();
    if prepared.is_empty() {
        return Err(format!("{} yielded no usable documents", cli.corpus.display()));
    }
    let lang = docs.first().map(|d| d.lang.clone()).unwrap_or_default();
    let total_sentences: usize = prepared.iter().map(|p| p.sentences.len()).sum();
    let total_gold: usize = prepared.iter().map(|p| p.gold.len()).sum();

    println!("== {} ({}) ==", cli.corpus.display(), lang);
    println!(
        "documents: {}  sentences: {}  gold boundaries: {}  mean segment: {:.1} sentences",
        prepared.len(),
        total_sentences,
        total_gold,
        total_sentences as f64 / (total_gold + prepared.len()) as f64
    );

    let mut results: Vec<(String, Score)> = Vec::new();

    // --- Embedding-free baselines -------------------------------------
    let mean_seg =
        (total_sentences as f64 / (total_gold + prepared.len()) as f64).round().max(2.0) as usize;

    let mut no_split = Score::default();
    let mut uniform = Score::default();
    let mut random = Score::default();
    for (i, p) in prepared.iter().enumerate() {
        let n = p.sentences.len();
        no_split.push(n, &p.gold, &baseline_none(n));
        uniform.push(n, &p.gold, &baseline_uniform(n, mean_seg));
        random.push(n, &p.gold, &baseline_random(n, p.gold.len(), i as u64 + 1));
    }
    results.push(("baseline: no split".into(), no_split));
    results.push((format!("baseline: uniform every {mean_seg}"), uniform));
    results.push(("baseline: random (gold count)".into(), random));

    // --- Embedding segmenters -----------------------------------------
    if let Some(dir) = &cli.embedder_dir {
        let embedder = euhadra::phoneme::OnnxTextEmbedder::load(dir)
            .map_err(|e| format!("loading embedder {}: {}", dir.display(), e.message))?;
        let cache: Arc<dyn TextEmbedder> =
            Arc::new(CachedEmbedder::new(embedder));

        let mut variants: Vec<(String, ParagraphSplitter)> = Vec::new();
        for ratio in [0.3f32, 0.5, 0.7] {
            variants.push((
                format!("depth ratio {ratio:.1}"),
                ParagraphSplitter::new()
                    .with_embedder(SharedCache(cache.clone()))
                    .with_depth_ratio(ratio)
                    .with_max_sentences(cli.max_sentences),
            ));
        }
        variants.push((
            "depth ratio 0.5 + centring".into(),
            ParagraphSplitter::new()
                .with_embedder(SharedCache(cache.clone()))
                .with_depth_ratio(0.5)
                .with_center_embeddings(true)
                .with_max_sentences(cli.max_sentences),
        ));

        for (name, splitter) in &variants {
            let mut score = Score::default();
            for p in &prepared {
                let hyp = splitter.breaks_for_sentences(&p.sentences);
                score.push(p.sentences.len(), &p.gold, &hyp);
            }
            results.push((name.clone(), score));
        }
    }

    println!("\n{:<32} {:>7} {:>7} {:>7} {:>9}", "segmenter", "Pk", "WD", "F1", "predicted");
    println!("{:-<32} {:->7} {:->7} {:->7} {:->9}", "", "", "", "", "");
    for (name, s) in &results {
        println!(
            "{:<32} {:>7.3} {:>7.3} {:>7.3} {:>9}",
            name,
            s.mean_pk(),
            s.mean_wd(),
            s.f1(),
            s.predicted
        );
    }
    println!("\nPk / WD are penalties — lower is better. gold boundaries: {total_gold}");

    // Name the documents the best embedding variant handles worst, so a
    // bad aggregate can be traced to whether it is systematic or a few
    // pathological texts.
    if cli.embedder_dir.is_some() {
        let splitter = ParagraphSplitter::new().with_max_sentences(cli.max_sentences);
        let mut worst: Vec<(f64, &str, usize, usize)> = prepared
            .iter()
            .map(|p| {
                let k = default_window(p.sentences.len(), &p.gold);
                let hyp = splitter.breaks_for_sentences(&p.sentences);
                (
                    window_diff(p.sentences.len(), &p.gold, &hyp, k),
                    p.doc_id.as_str(),
                    p.gold.len(),
                    hyp.len(),
                )
            })
            .filter(|(w, _, _, _)| w.is_finite())
            .collect();
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let sample: Vec<String> = worst
            .iter()
            .take(3)
            .map(|(w, id, g, h)| format!("{id} WD={w:.2} gold={g} pred={h}"))
            .collect();
        if !sample.is_empty() {
            println!("hardest documents (no-embedder reference): {}", sample.join("  |  "));
        }
    }

    if let Some(out) = &cli.output {
        let json = serde_json::json!({
            "corpus": cli.corpus.display().to_string(),
            "lang": lang,
            "documents": prepared.len(),
            "sentences": total_sentences,
            "gold_boundaries": total_gold,
            "results": results.iter().map(|(name, s)| serde_json::json!({
                "segmenter": name,
                "pk": s.mean_pk(),
                "window_diff": s.mean_wd(),
                "boundary_f1": s.f1(),
                "predicted": s.predicted,
                "tp": s.tp, "fp": s.fp, "fn": s.fn_,
            })).collect::<Vec<_>>(),
        });
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(out, serde_json::to_string_pretty(&json).unwrap())
            .map_err(|e| format!("write {}: {e}", out.display()))?;
        eprintln!("report written to {}", out.display());
    }

    Ok(())
}
