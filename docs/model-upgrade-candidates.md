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

### 5.2 Alpha calibration

Correction-pair F1 on `en_phoneme_correction.jsonl` (25 utterances,
13-word dictionary) at the default threshold 0.85. The phoneme-only
baseline is **F1 1.000** — the gold set is saturated, so the embedding
term cannot improve on it and the question is only how much it costs.

| alpha | `bge-small-en-v1.5` | `granite-97m-multilingual-r2` |
|---|---|---|
| 0.50 | 0.643 | 0.973 |
| 0.60 | 0.733 | **1.000** |
| 0.70 | 0.882 | **1.000** |
| 0.80 | 0.973 | **1.000** |
| 0.90 | **1.000** | **1.000** |
| 1.00 (phoneme only) | 1.000 | 1.000 |

Precision stayed at 1.000 everywhere; only recall moved, as dilution
pushed real matches under the threshold.

**granite holds F1 1.000 down to alpha 0.60; bge-small needs 0.90.**
That gap has a direct consequence: `docs/spec.md` §6.4 works its
example at **alpha = 0.70**, where bge-small loses three of nineteen
corrections (F1 0.882) and granite loses none. The documented
configuration was broken on the shipped backend.

This is the measured argument for the migration, and it is a stronger
one than multilingual coverage — because unlike coverage, it can be
verified against data that exists today. Recorded as
`phoneme::calibrated_alpha`; raw sweep in
[`benchmarks/embedder_calibration/phoneme_alpha_sweep.json`](./benchmarks/embedder_calibration/phoneme_alpha_sweep.json).

INT8 costs one correction at alpha 0.70 (F1 0.973) and needs alpha
≥ 0.75 to hold 1.000, which is why CI pulls the fp32 bundle.

### 5.3 ParagraphSplitter: a default that cannot fire

`ParagraphSplitter` breaks a paragraph when adjacent-sentence cosine
similarity falls **below** `similarity_threshold`, default **0.5**.
The multilingual probes in `examples/check_embedder.rs` put granite's
similarity for deliberately *unrelated* short strings at **0.62–0.75**
across en/ja/zh/ko/es — above the threshold. On granite the semantic
path would therefore never fire, silently degrading the splitter to
its max-sentences constraint.

`paragraph::calibrated_similarity` records provisional thresholds
(0.65 bge-small, 0.80 granite) to prevent that. They are marked
provisional in the source for a reason: see §5.5.

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

- **No paragraph-boundary corpus exists at all.** `ParagraphSplitter`
  has never been scored, in any language. §5.3's thresholds are
  inferred from probe behaviour, not fitted.
- **Phoneme-correction gold is English-only**, and saturated at F1
  1.000 by the phoneme-only path. It can detect a backend making
  things *worse* — which is exactly what it did in §5.2 — but it
  cannot demonstrate the semantic term making things better, and it
  says nothing about ja/zh/ko/es.

Building a paragraph-boundary gold set, and extending phoneme
correction beyond English (which needs per-language G2P and IPA
tables), are the prerequisites for any real multilingual accuracy
claim. Until then the honest framing is: granite is a strictly safer
backend on the evidence available, not a demonstrated better one.
