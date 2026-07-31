# Model Upgrade Candidates

A research log of models that could replace what euhadra ships today,
and of the measurements taken before recommending any of them. It
complements two neighbouring documents:

- [`model-licenses.md`](./model-licenses.md) — the canonical licence
  summary for what we already ship.
- [`korean-asr-alternatives.md`](./korean-asr-alternatives.md) — the
  same kind of log, scoped to the Korean ASR path (issue [#83]).

Survey data was collected 2026-07-31. Measurements in §3 were run in
the same session on a shared 4-core Xeon @ 2.1 GHz (the same class of
container the `#83` bench used), CPU execution provider.

[#83]: https://github.com/penta2himajin/euhadra/issues/83

---

## 1. Scope and conclusion

The survey covered every model slot in the pipeline. The short version:

| Slot | Shipping today | Verdict |
|---|---|---|
| ASR en / es | `canary-180m-flash` INT8 | Upgrade path exists (`canary-1b-v2`), not urgent |
| ASR ja | `parakeet-tdt_ctc-0.6b-ja` | **Keep** — still the strongest option measured |
| ASR zh | `paraformer-large` | Accuracy headroom exists |
| ASR ko | `whisper-large-v3-turbo` q4 | **Worst offender** — RTF 1.34 in `ci_baseline.json` |
| Tier 1/2 embedding | `bge-small-en-v1.5` | **English-only; measured and recalibrated in §3** |
| Tier 2 punctuation | `felflare/bert-restore-punctuation` | English-only, 2021 vintage |
| Tier 2 NER | *unimplemented* | Redesign before implementing |
| Tier 3 refiner | *unimplemented* | No change needed |

The headline finding is not in the survey but in the measurement: the
Tier 1 embedding filter's cosine threshold **had no effect on its
output at all** before this change (§3.1).

---

## 2. Survey, 2026-07

### 2.1 ASR

| Candidate | Licence | Languages | Why it matters | ONNX path |
|---|---|---|---|---|
| `Qwen3-ASR-0.6B` / `-1.7B` | Apache-2.0 | 52 incl. all five of ours | Single checkpoint covering en/ja/zh/es/ko — would collapse four backends into one | `sherpa-onnx` ships an INT8 export |
| `nvidia/canary-1b-v2` | CC-BY-4.0 | 25 European | Same family as our `canary/` adapter, so lowest porting cost | `istupakov/canary-1b-v2-onnx` |
| `nvidia/parakeet-tdt-0.6b-v3` | CC-BY-4.0 | 25 European | Throughput leader; could serve en+es from one backend | `istupakov/parakeet-tdt-0.6b-v3-onnx` |
| `FireRedASR-AED` | Apache-2.0 | zh, en | AISHELL-1 CER 0.55% vs Paraformer's 1.68% | Not published |
| `ibm-granite/granite-speech-4.1-2b` | Apache-2.0 | 6 incl. ja | Strong on clean speech, brittle on noisy/multi-speaker | Not published |
| `cohere-transcribe-03-2026` | **Disputed** | multi | Strong WER, Rust + ONNX ports exist | `onnx-community`, but see below |
| Voxtral Realtime / Transcribe 2 | Apache-2.0 / API-only | multi | Streaming; the accurate variant is API-only | — |

Published FLEURS numbers for Qwen3-ASR (technical report, Table A.2)
against our own `ci_baseline.json` figures:

| | en | es | zh | ja | ko |
|---|---|---|---|---|---|
| Qwen3-ASR-0.6B | 4.39 WER | 4.94 WER | 2.88 CER | 8.33 CER | 3.72 CER |
| Qwen3-ASR-1.7B | 3.35 WER | 3.36 WER | 2.41 CER | 5.20 CER | 2.57 CER |
| euhadra today | 4.39 WER | 5.28 WER | 5.53 CER | **3.31 CER** | 0.95 CER |

**These columns are not comparable.** Ours are 10-utterance FLEURS
subsets scored with `eval::metrics::cer_lenient`; theirs are full
FLEURS with their own normalisation. The table is a shortlist filter,
not a decision. Two things it does support:

- **ja should stay on `parakeet-tdt_ctc-0.6b-ja`.** A 2.5× gap does
  not close on normalisation differences.
- **zh has real headroom.** Both Qwen3-ASR and FireRedASR report
  Mandarin CER well below what `paraformer-large` gives us.

`cohere-transcribe-03-2026` is excluded from consideration until its
licence is settled: the CohereLabs repository declares Apache-2.0
while the CoreML and INT8 redistributions declare CC-BY-NC-4.0.
`docs/spec.md` §8 depends on a commercially-clean OSS default.

### 2.2 Tier 1 / Tier 2

This is where the largest gap sits, and it is not an accuracy gap —
it is a **coverage** gap. Both slots ship English-only models while
the pipeline advertises five languages. The gap bites hardest where
there is no rule-based fallback: `ParagraphSplitter` and
`PhonemeCorrector` are embedding-only, so outside English they do not
work at all. The filler filter is the opposite case — §3.2 measures
its rule-based implementations beating every embedding backend.

| Slot | Today | Candidate | Note |
|---|---|---|---|
| Embedding | `bge-small-en-v1.5` (2023, en) | `granite-embedding-97m-multilingual-r2` | Apache-2.0, **same 384 dims**, ja/zh/ko/es in its enhanced set. Measured in §3 |
| Embedding | — | `potion-multilingual-128M` | MIT, static (no transformer pass). Measured in §3 |
| Punctuation | `felflare/bert-restore-punctuation` (2021, en) | `1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase` | 47 languages, punctuation + truecase + sentence boundary in one ONNX graph |
| NER | unimplemented (DistilBERT-NER planned) | GLiNER / GLiNER2 | Zero-shot entity types fit §6.4's custom-dictionary use better than fixed PER/LOC/ORG/MISC. Rust ONNX runners exist (`gline-rs`, `gliner2-rs`) |

`LFM2.5-Embedding-350M` was evaluated on paper and **rejected**: it
omits Chinese from its 11 languages, requires asymmetric
`query:`/`document:` prefixes (our use is symmetric similarity), emits
1024 dims, and its LFM Open Licence v1.0 terminates commercial use
above $10M annual revenue — incompatible with `docs/spec.md` §8.

### 2.3 Tier 3

`LlamaCppRefiner` is still unimplemented and the survey found no
reason to revisit the plan. Gemma 3n E2B, Qwen3 0.6B–4B and SmolLM2
are all GGUF/llama.cpp-ready, which is what §5.2 assumes.

---

## 3. Measured: Tier 1 embedding backends

Three backends, four languages, scored against the gold filler
annotations in `tests/evaluation/annotations/<lang>_filler.jsonl`.
Reproduce with:

```bash
EMBEDDER_MODEL=all scripts/setup_embedders.sh
cargo run --release --features onnx --example bench_embedder -- \
    --model-dir vendor/embedder_granite_97m --lang ja \
    --annotations tests/evaluation/annotations/ja_filler.jsonl \
    --sweep-from 0.30 --sweep-to 1.00 --sweep-step 0.02
```

Raw output for all twelve runs:
[`benchmarks/embedder_calibration/results.json`](./benchmarks/embedder_calibration/results.json).

### 3.1 The threshold was inert

`docs/spec.md` §3.5 describes the embedding filter as detecting pure
fillers by cosine proximity, quoting a separation of ">0.82 for pure
fillers, <0.76 for ordinary words". The implementation did not do
that. Pass 1 read:

```rust
if max_filler_sim(emb) >= threshold && pure_fillers.contains(clean)
```

A conjunction of those two conditions cannot discriminate. Every
lexicon member embeds to (nearly) its own prototype, so its similarity
is ~1.0 and clears any threshold; every non-member is rejected by the
lexicon regardless of what it scores. The cosine gate could therefore
neither reject a dictionary hit nor accept anything else.

Sweeping the threshold across its entire usable range confirmed it,
identically for all three backends:

```
threshold   P       R       F1      TP  FP  FN
  0.00      1.000   0.706   0.828    12   0   5
  ...       (unchanged at every step)
  0.95      1.000   0.706   0.828    12   0   5
  1.00      1.000   0.353   0.522     6   0  11
```

F1 moved by exactly zero from 0.00 to 0.95. The only movement was at
1.00, where float equality starts excluding lexicon members whose
surface form carries punctuation. `potion-multilingual-128M`, whose
gold-segment similarities have a median of 0.27 against
`bge-small-en-v1.5`'s 0.77, scored *identically* to it at every step —
which is only possible if the similarity is not being consulted.

**Fix**: pass 1 now takes the disjunction — lexicon membership **or**
embedding proximity. The lexicon keeps recall on known forms; the
embedding generalises to variants outside it (`"ummm"`, `"uhh"`, ASR
artefacts), which is what §3.5 specifies. The threshold became a live
parameter, and the numbers below are the first real calibration of it.

### 3.2 Calibration results

Best strict-span F1 and the threshold that achieves it, per backend
and language:

| Language | `bge-small-en-v1.5` | `granite-97m-multilingual-r2` | `potion-multilingual-128M` |
|---|---|---|---|
| en | 0.828 @ 0.80 | 0.828 @ 0.90 | 0.828 @ 0.38 |
| ja | 0.824 @ 0.82 | **0.909 @ 0.88** | 0.882 @ 0.44 |
| zh | **1.000 @ 0.82** | 0.897 @ 0.84 | **1.000 @ 0.30** |
| ko | 0.562 @ 0.90 | 0.562 @ 0.88 | 0.556 @ 0.30 |

Separation margin (`min(gold) − max(other)`) and per-call embed
latency:

| Backend | Params | Dims | en margin | ja margin | embed p50 | load |
|---|---|---|---|---|---|---|
| `bge-small-en-v1.5` | 33M | 384 | −0.145 | −0.210 | 8.6 ms | 0.5 s |
| `granite-97m-multilingual-r2` | 97M | 384 | **−0.082** | **−0.093** | 10.1 ms | 2.1 s |
| `potion-multilingual-128M` | 128M static | 256 | −0.318 | −0.638 | **0.05 ms** | 4.6 s |

And the comparison that matters most — the **rule-based** filters,
scored on the same gold sets through the same L3 runner
(`eval_l3 --task filler`, no `--embedder-dir`):

| Language | Rule-based | Best embedding backend |
|---|---|---|
| en | **1.000** | 0.828 |
| ja | **0.941** | 0.909 (granite @ 0.88) |
| zh | **1.000** | 1.000 (bge-small, potion) |
| ko | **0.977** | 0.562 |

**The embedding filter does not beat the rule-based filters anywhere.**
That result is partly circular — these gold sets were curated around
the phenomena the rule-based filters target — but it is the data we
have, and it changes the recommendation: the case for a multilingual
embedding backend is *not* Tier 1 filler removal. See §4.

Four further things the backend comparison says:

1. **Thresholds are not portable, and the failure is not graceful.**
   Running granite at bge-small's 0.82 turns 131 of 143 English
   non-filler tokens into false positives — precision 0.084. The
   optimum is 0.80 / 0.90 / 0.38 for the three backends. This is the
   real migration cost, exactly as suspected; the code change is
   trivial by comparison.
2. **granite is the best backend for the languages that matter.** It
   wins ja outright (+0.085 F1 over bge-small) and has the tightest
   separation margin in both en and ja, meaning its decision boundary
   is the least fragile.
3. **bge-small's zh score is not an embedding result.** It reaches
   1.000 because single-character Mandarin fillers are all lexicon
   members; the disjunction returns them from the dictionary branch.
   An English-only model has no business scoring Mandarin, and the
   margin table shows why the number is not evidence of capability.
4. **potion is ~190× faster per call** at comparable F1 (it wins ja and
   zh against bge-small). For a Tier 1 layer whose latency budget is
   "milliseconds", a 0.05 ms lookup versus a 8–14 ms transformer pass
   is the difference between free and noticeable. Its slow *load* is a
   one-off and could be cut with a quantised export.

### 3.3 Limits of this measurement

Stated plainly, because the sample is small:

- **25 / 25 / 22 / 29 utterances** for en / ja / zh / ko. These are
  hand-curated fixtures, not a corpus. Treat the thresholds as
  starting points, not settled constants.
- **Recall is capped at 0.706 on en for every backend.** The five
  false negatives are multi-word gold spans (`"you know"`, `"i mean"`)
  that a one-segment-at-a-time scorer structurally cannot emit. That
  is a design limit of the embedding filter, not of any model, and it
  is why `SimpleFillerFilter` keeps its own bigram pass.
- **ko is weak across all three backends** (F1 ≈ 0.56). Korean fillers
  in the gold set are agglutinated into surrounding morphemes, so
  whitespace segmentation cannot isolate them. No embedding model
  fixes that; it needs a morphological segmenter.
- The bench measures **symmetric similarity**, which is what the
  filter does. Published MTEB figures for these models are
  **retrieval** scores. They do not transfer, which is precisely why
  this measurement had to exist.

---

## 4. Recommended next steps

In priority order, each small enough to be its own PR:

> **Status**: items 1 and 2 shipped — see §5 for the measurements they
> produced. Items 3–5 remain open.

1. ~~**Move `ParagraphSplitter` and `PhonemeCorrector` to
   `granite-embedding-97m-multilingual-r2`, not the filler filter.**~~
   This is the corrected priority. Those two Tier 2 consumers are
   English-locked with *no* rule-based fallback, so today they simply
   do not work for ja/zh/ko/es — that is the real coverage hole. The
   filler filter, by contrast, already has rule-based implementations
   that outscore every embedding backend measured (§3.2), so swapping
   its backend buys nothing on current evidence. Both consumers need
   their own thresholds swept the same way; neither is covered by
   this bench.
2. ~~**Decide what `OnnxEmbeddingFilter` is actually for.**~~
   Retired: deprecated and unwired, along with the Python
   `EmbeddingFillerFilter` path, which turned out to carry the same
   inert-AND-gate as §3.1 and to be a rule-based filter in all but
   cost. Filler removal is rule-based only now. On these
   gold sets it is strictly worse than the rule-based filters it sits
   beside. Its one defensible advantage is generalising past a closed
   lexicon — which the AND-bug (§3.1) meant it never did. Either
   evaluate it on data containing out-of-lexicon filler variants
   (where the disjunction should now pay off), or retire it. The
   bench added here is the tool for that call.
3. **Redesign the NER slot around GLiNER** before implementing the
   planned DistilBERT-NER. Unimplemented today, so it is the cheapest
   moment to change direction.
4. **Fix the ko filler path in the embedding filter** if it is kept.
   Neither a threshold nor a backend change moves it (0.56 for all
   three backends, against 0.977 rule-based); it needs morphological
   segmentation.
5. **Measure `Qwen3-ASR-0.6B` INT8 on the ko path** against the
   `#83` harness — the RTF 1.34 in `ci_baseline.json` is the only
   figure in the file that is unusable for dictation.

If `potion-multilingual-128M` is ever adopted for Tier 1, note the
secondary prize: it needs no transformer pass, so
`OnnxEmbeddingFilter` could leave the `onnx` feature gate entirely and
give the default build multilingual embedding at zero ML dependency,
in line with `docs/spec.md` §10.2. That is contingent on step 2
concluding the filter is worth keeping.

---

## 5. Measured: Tier 2 embedding consumers

§4 argued the case for a multilingual embedder is the Tier 2
consumers, not the filler filter. This section is that work.

### 5.1 The composite score shipped unmeasured

`PhonemeCorrector` blends phoneme distance with text-embedding
similarity as `alpha * phoneme_sim + (1-alpha) * text_sim`
(`docs/spec.md` §6.4). Its `alpha` defaults to **1.0**, and no
evaluation wired an embedder — `eval_l3 --task phoneme-correction`
constructed the corrector with `PhonemeCorrector::new(..)` and nothing
else. So the entire semantic half of the design had never been scored.
`--embedder-dir` / `--alpha` / `--threshold` now exist for that.

Same class of finding as §3.1: a documented mechanism that was inert
in practice.

### 5.2 Alpha calibration, and why the table is gone

Correction-pair F1 on `en_phoneme_correction.jsonl` (25 utterances,
13-word dictionary). The phoneme-only baseline is **F1 1.000** — the
gold set is saturated, so the embedding term cannot improve on it and
the question is only how much it costs.

The first pass measured the composite with the raw cosine, and found
the operating point was backend-specific: granite held 1.000 down to
alpha 0.60, `bge-small` needed 0.90, and at the **alpha = 0.70 that
`docs/spec.md` §6.4 works its example at** bge-small lost three of
nineteen corrections. That produced a per-backend `calibrated_alpha`
table.

The table was the wrong fix. `alpha` blends a normalised edit distance
(genuinely on `[0, 1]`, 0 meaning "nothing alike") with a raw cosine
whose own zero point sits wherever the model puts it — 0.45 on
bge-small, 0.70 on granite. So the same alpha weighted the two terms
differently on every backend. Rescaling the semantic term against the
backend's measured floor (`similarity::rescale`, floor measured once
per backend from `embedding::FLOOR_PROBES`) removes the cause instead
of tabulating the symptom:

| alpha | `bge-small-en-v1.5` | `granite-97m-multilingual-r2` | |
|---|---|---|---|
| 0.30 | 0.480 | 0.593 | diverge |
| 0.40 | 0.643 | 0.733 | diverge |
| 0.50 | 0.643 | 0.813 | diverge |
| 0.60 | 0.733 | 0.944 | diverge |
| **0.70** (spec §6.4) | **1.000** | **1.000** | **agree** |
| 0.80 | 0.950 | 0.950 | agree |
| 0.90 | 0.950 | 0.950 | agree |

The documented alpha now holds on both backends, and the CI gate at
alpha 0.70 passes on either — it failed on bge-small before.

**It is not a complete fix, and the table above says so.** Below alpha
0.70, where the semantic term starts to dominate, the two backends
still diverge. Rescaling is affine: it aligns where each model puts
"unrelated" but not the shape of the range above it. The honest claim
is that alpha is portable **in the region worth operating in**, not
that cosine has been made universal.

**Acceptance thresholds had to split.** Rescaling removes the free
points a 0.70 floor was contributing (0.3 × 0.70 = 0.21 of every
composite score), so the composite legitimately sits lower and the old
0.85 bar lost real matches. Measured: the composite wants **0.65** on
both backends, while phoneme-only still wants **0.85** — applying 0.65
there admits two false positives. These are two thresholds on two
different quantities, and unlike the alpha table, one value each covers
every backend. `PhonemeCorrector::threshold` and
`::composite_threshold`.

Raw sweep in
[`benchmarks/embedder_calibration/phoneme_alpha_sweep.json`](./benchmarks/embedder_calibration/phoneme_alpha_sweep.json).

### 5.3 ParagraphSplitter: from a threshold that cannot fire to a shape rule

`ParagraphSplitter` broke a paragraph where adjacent-sentence cosine
fell **below** `similarity_threshold`, default **0.5**. The
multilingual probes in `examples/check_embedder.rs` put granite's
similarity for deliberately *unrelated* short strings at **0.62–0.75**
across en/ja/zh/ko/es — above that threshold. On granite the semantic
path could never fire, silently reducing the splitter to its
max-sentences constraint.

Same root cause as §5.2, so the same treatment: stop comparing a raw
cosine to a constant. Breaks are now placed at **valleys** — local
minima in the similarity sequence, scored by how far they sit below the
peaks reachable on either side:

```
depth(i) = (left_peak − sim(i)) + (right_peak − sim(i))
```

Built from differences only, so a backend that shifts every similarity
up by 0.2 produces identical depths. Worked example, same text on two
backends:

| boundary | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| bge-small | 0.68 | 0.66 | **0.51** | 0.67 | 0.69 |
| granite | 0.88 | 0.86 | **0.71** | 0.87 | 0.89 |

An absolute rule cannot serve both: 0.5 fires on neither, and a
threshold tuned so bge-small fires once leaves granite's entire range
above it. Depth at index 2 is 0.35 on both. `ParagraphSplitter` now
takes a break where a minimum reaches `depth_ratio` (default 0.5) of
the document's deepest valley.

Two guards keep the relative rule honest. A purely relative criterion
always finds *some* lowest point, so uniform text would be split
arbitrarily: `min_similarity_range` (default 0.05) requires the text to
show some spread before any semantic split is attempted. And fewer than
two boundaries, or any failed embedding, defers entirely to
max-sentences rather than inventing a valley.

This replaces the provisional per-backend `calibrated_similarity`
constants an earlier revision of this PR introduced. It does **not**
resolve §5.5: the rule is portable and cannot silently go dead, but
whether the valleys it picks are the *right* places to break is still
unmeasured in every language.

### 5.4 Multilingual: what CI asserts, and what it does not

The new `evaluate (embedding backend)` job asserts the backend is
*functional* per language — same-width, finite, unit-norm vectors, and
a related string ranking above an unrelated one in each of en/ja/zh/ko/es.
`bge-small-en-v1.5` fails that check on **zh**, scoring 草莓果酱
("strawberry jam") *more* similar to 数据库 ("database") than
数据库服务器 ("database server") is, for a margin of −0.048. granite
passes all five with margins of +0.13 to +0.27.

It asserts nothing about multilingual **accuracy**, because there is
nothing to assert it against — see below.

### 5.5 The gold-data gap

Both Tier 2 consumers are limited by missing data, and no threshold or
backend choice changes that:

- ~~**No paragraph-boundary corpus exists at all.**~~ **Closed by §6.**
  `scripts/download_paragraph_corpus.py` builds one from Wikipedia and
  `examples/eval_paragraph.rs` scores against it in all five languages.
  The result is split: the valley rule finds real topic shifts well
  (§6.1) and does not reproduce author paragraphing (§6.2).
- **Phoneme-correction gold is English-only**, and saturated at F1
  1.000 by the phoneme-only path. It can detect a backend making
  things *worse* — which is exactly what it did in §5.2 — but it
  cannot demonstrate the semantic term making things better, and it
  says nothing about ja/zh/ko/es.

The paragraph half of that is now done (§6). The phoneme half is not,
and extending it beyond English needs per-language G2P and IPA tables
before any multilingual accuracy claim is possible there. So the
framing splits by consumer: for paragraph splitting there is now direct
multilingual evidence; for phoneme correction, granite remains a
strictly *safer* backend on the evidence available rather than a
demonstrated better one.

---

## 6. Measured: does the splitter break in the right places?

§5.3 replaced the splitter's absolute threshold with a valley rule and
proved it portable. §5.5 recorded what that still did not establish:
whether the valleys it picks are the *right* places to break. That was
unmeasured in every language because no paragraph-boundary corpus
existed. This section is that measurement.

**Corpus**: `scripts/download_paragraph_corpus.py` builds one from
Wikipedia (CC-BY-SA 4.0, fetched on demand, gitignored) in the two
shapes the segmentation literature uses:

- **`choi`** — Choi-style synthetic concatenation: six paragraphs drawn
  from six *different* articles, so boundaries are known by
  construction rather than annotated. 30 documents per language.
- **`author`** — one article's own paragraphs in order, with the gold
  boundaries where its author put them. Harder and noisier, since
  paragraph breaks are partly stylistic.

**Metrics**: Pk and WindowDiff, both window penalties where **lower is
better**. Exact-match F1 is the wrong instrument — it scores a break
one sentence off as both a false positive and a false negative, so it
cannot tell "nearly right" from "random".

Reproduce with:

```bash
scripts/download_paragraph_corpus.py
EMBEDDER_MODEL=granite scripts/setup_embedders.sh
cargo run --release --features onnx --example eval_paragraph -- \
    --corpus data/paragraph_corpus/choi_ja.jsonl \
    --embedder-dir vendor/embedder_granite_97m
```

Raw reports:
[`benchmarks/paragraph_segmentation/`](./benchmarks/paragraph_segmentation/).

### 6.1 Synthetic topic shifts: the rule works, in all five languages

WindowDiff, `granite-embedding-97m-multilingual-r2`:

| segmenter | en | ja | zh | ko | es |
|---|---|---|---|---|---|
| baseline: no split | 0.460 | 0.459 | 0.445 | 0.425 | 0.466 |
| baseline: uniform (per-corpus k) | 0.509 | 0.508 | 0.484 | 0.446 | 0.487 |
| baseline: random (gold count) | 0.528 | 0.552 | 0.509 | 0.516 | 0.522 |
| depth ratio 0.3 | 0.162 | 0.244 | 0.249 | 0.249 | 0.291 |
| **depth ratio 0.5** (default) | **0.122** | **0.154** | **0.178** | **0.194** | **0.244** |
| depth ratio 0.7 | 0.172 | 0.252 | 0.191 | 0.245 | 0.304 |
| depth ratio 0.5 + centring | 0.160 | 0.180 | 0.162 | 0.231 | 0.328 |

Two to four times better than every baseline, in every language, and it
predicts close to the right number of boundaries (133–172 against 150
gold). **The valleys are at real topic changes.** This is also the
first evidence of any kind that the splitter works outside English.

The uniform baseline scoring *worse* than not splitting at all is worth
noting on its own: the gold segments are not uniform in length, so the
`max_sentences` constraint alone would be a poor segmenter. The
semantic path is carrying the result.

**depth ratio 0.5 wins in all five languages**, which is the shipped
default — chosen before this corpus existed, and now supported by it.

### 6.2 Author paragraph structure: it does not reproduce that

Same corpus builder, real articles:

| segmenter | en | ja | zh | ko | es |
|---|---|---|---|---|---|
| baseline: no split | 0.515 | 0.432 | 0.427 | 0.434 | 0.487 |
| baseline: random (gold count) | 0.561 | 0.490 | 0.464 | 0.460 | 0.529 |
| depth ratio 0.5 | 0.499 | 0.391 | 0.366 | 0.368 | 0.454 |
| depth ratio 0.5 + centring | 0.421 | 0.365 | 0.403 | 0.395 | 0.467 |

On English it is 0.499 against 0.515 for **not splitting at all** —
inside the noise. CJK does better (0.366–0.391 against 0.427–0.434) but
the margin is nothing like §6.1's.

Stated plainly: **the semantic signal does not recover an author's
paragraphing.** That is not surprising — paragraph breaks inside one
article are largely stylistic and length-driven, and the topic barely
moves across them — but it is a real limit and it bounds what this
layer should be claimed to do.

Which task matters more depends on the use case. A user dictating who
finishes one subject and starts another is the §6.1 situation, and the
splitter handles it well. Matching a writer's stylistic paragraphing is
the §6.2 situation, and it does not. For a dictation tool the first is
the competence worth having, but the second is what a user comparing
output against their own writing would notice.

### 6.3 Centring does not pay

Subtracting the document's mean sentence embedding — the cheapest
member of the whitening family, the standard first move against
embedding anisotropy — helps in 4 of the 10 cells above and hurts in 6,
with no pattern by language or task. It stays available
(`with_center_embeddings`) and stays **off by default**.

The reason it was worth testing anyway: anisotropy is why unrelated
text scores 0.6–0.7 in the first place, and correcting it at the vector
level would have improved every consumer at once rather than each
consumer's scoring rule separately. On this evidence it does not, for
this consumer. §5.2's residual — that affine rescaling does not align
the backends below alpha 0.60 — remains open, and a stronger vector-level
correction (ZCA / all-but-the-top) or a distribution-free score mapping
(empirical CDF) are the untested candidates.

### 6.4 What is still not measured

- Only `granite-embedding-97m-multilingual-r2` was run. The rule is
  provably backend-invariant under a uniform shift (§5.3), but that is
  an argument, not a second measurement.
- Wikipedia prose is not dictation. Sentences are longer, better formed
  and more topically coherent than ASR output, so §6.1 is likely
  optimistic for the real input.
- 30 synthetic documents and 14–20 articles per language. Small enough
  that a few-point difference between neighbouring `depth_ratio` values
  should not be read as meaningful; the gap to the baselines is what
  the sample supports.
- **Not run in CI**, deliberately. The corpus is fetched over the
  network from a third party, which does not belong in a per-PR gate:
  a Wikipedia outage or an edit to one of the source articles would
  turn into a red build unrelated to the change under review. It is a
  release-time / research measurement, like the rest of L3
  (`docs/evaluation.md` §1.3), and the committed reports under
  `benchmarks/paragraph_segmentation/` are the record.
- `min_similarity_range` (0.05) is unvalidated. It exists because a
  purely relative rule always finds *some* lowest point and would split
  uniform text, so *something* absolute has to stop it — but no
  measurement here distinguishes 0.05 from 0.02 or 0.10. It is the
  weakest number in the splitter.
- The synthetic task joins paragraphs from unrelated articles, so its
  topic shifts are sharper than the ones a dictating user produces when
  moving between related subjects. §6.1 is an upper bound on that axis
  too, not just on prose quality.
