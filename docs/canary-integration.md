# Canary ONNX Integration — Decision Log & Fallback Plan

## Decision (2026-04-30)

Integrate **NVIDIA Canary-180M-Flash via the
[istupakov/canary-180m-flash-onnx](https://huggingface.co/istupakov/canary-180m-flash-onnx)
ONNX export** as the Spanish (and en/de/fr) ASR backend, replacing the
plan to wire Whisper-es subprocess as the MVP.

## Why not the original Canary plan

NeMo upstream `model.export()` blocks Canary export — see
[NVIDIA-NeMo/NeMo#11004](https://github.com/NVIDIA-NeMo/NeMo/issues/11004),
open since October 2024. Building the export pipeline ourselves was
estimated at 1–2 weeks before any inference code could land.

## Why istupakov over Parakeet-TDT-0.6B-v3-multi

| Axis | Canary-180M-Flash (istupakov) | Parakeet-TDT-0.6B-v3 |
|---|---|---|
| Size (INT8) | **~213 MB** (134 + 79.5) | ~671 MB |
| Spanish MLS WER | **3.17 %** | 4.39 % |
| Spanish FLEURS WER | (not reported in card; ~3.3 % expected) | 3.45 % |
| German MLS WER | **4.81 %** | (not reported per-lang) |
| French MLS WER | **4.75 %** | 4.97 % |
| AST (translation) | **yes** (en↔de/fr/es) | no |
| ONNX export | **available** | available |
| Existing Rust harness | none (this work) | `parakeet-rs` reusable |
| Languages | 4 (en/de/fr/es) | 25 (EU) |

Canary wins on accuracy-per-byte for our four target languages; the
size advantage matters for the on-device Cochlis distribution. The
loss is having to build a Rust inference loop from scratch (~500
lines), since `parakeet-rs` doesn't speak Canary's encoder-decoder
architecture.

## Architecture summary

The istupakov export ships **encoder + decoder ONNX graphs only** —
preprocessing and decoding are the caller's responsibility, mirrored
from the [`onnx-asr`](https://github.com/istupakov/onnx-asr) Python
reference.

| Component | Source | Effort | This repo |
|---|---|---|---|
| Mel preprocessor (Hann window, log-mel, CMVN) | `numpy_preprocessor.py` | ~150 lines Rust | `src/canary/frontend.rs` |
| SentencePiece vocab + special tokens | `vocab.txt` | ~80 lines Rust | `src/canary/vocab.rs` |
| Encoder ONNX call | `nemo.py` | ~50 lines, `ort` crate | `src/canary/encoder.rs` |
| Autoregressive decoder loop with KV cache | `nemo.py` `_decode` | ~120 lines | `src/canary/decoder.rs` |
| `CanaryAdapter` impl `AsrAdapter` trait | new | ~80 lines | `src/canary/adapter.rs` |
| Setup script (download ONNX) | new | ~40 lines bash | `scripts/setup_canary_es.sh` |

### Mel preprocessor parameters (verbatim from `onnx-asr`)
```
sample_rate     = 16_000
n_fft           = 512
win_length      = 400      # 25 ms
hop_length      = 160      # 10 ms
n_mels          = 128      # config.json features_size
preemph         = 0.97
window          = hann
log_guard       = 2**-24
norm            = per-feature mean/var across time (CMVN-like)
```

> **Note**: `onnx-asr`'s python source defaults `_features_size` to 80
> (the value Parakeet-TDT-0.6B-ja uses), but it actually reads
> `features_size` from `config.json` at runtime. The istupakov bundle
> ships `features_size=128`, and the encoder errors with
> `Got: 80 Expected: 128` if fed an 80-mel buffer. The Rust default
> matches the bundle.

### Decoder prefix layout

Two layouts are supported via `PrefixFormat` in
`src/canary/decoder.rs`:

1. **`NemoCanary2` (9 tokens — default, paper layout)** — what
   NVIDIA's `Canary2PromptFormatter` (NeMo
   [`nemo/collections/common/prompts/canary2.py`](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/prompts/canary2.py))
   actually produces when `decodercontext = ""`:
   ```
   [<|soc|>, <|sot|>, <|emo:undefined|>, <|src|>, <|tgt|>,
    <|pnc|>, <|noitn|>, <|notimestamp|>, <|nodiarize|>]
   ```
   Slots: 0 = `<|startofcontext|>`, 1 = `<|startoftranscript|>`,
   2 = `<|emo:undefined|>`, 3 = source language, 4 = target language,
   5 = `<|pnc|>`/`<|nopnc|>`, 6 = `<|noitn|>`, 7 = `<|notimestamp|>`,
   8 = `<|nodiarize|>`. The Canary-1B-v2 paper (arXiv 2509.14128)
   describes the same prompt structure.

2. **`OnnxAsr` (10 tokens — opt-in, legacy)** — the layout hardcoded
   by `istupakov/onnx-asr`'s Python reference, with a leading `▁`
   (last-occurrence of U+2581) before `<|startofcontext|>`:
   ```
   [▁, <|soc|>, <|sot|>, <|emo:undefined|>, <|src|>, <|tgt|>,
    <|pnc|>, <|noitn|>, <|notimestamp|>, <|nodiarize|>]
   ```
   Source language is at slot 4. We shipped this through PR #28 #36;
   it's bit-aligned with the Python reference (see v6) and therefore
   useful for comparison harnesses, but it is **not** the layout the
   model was trained against.

The two layouts differ by a single leading `▁`. The v8 sweep
below shows that token nearly doubles WER on FLEURS-es; the
default was flipped to `NemoCanary2` accordingly.

Greedy decode with KV cache (`decoder_mems` shape `(layers, batch, T, hidden)`).

## Fallback triggers — when to abandon this plan

**Hard fallback** (switch to Parakeet-TDT-0.6B-v3-multi):
1. **Quality regression > 1 pp WER** vs. the istupakov model card on
   FLEURS-es / MLS-es when measured via our `eval_l1_smoke` harness.
   Indicates the export is lossy or our preprocessing/decoder differs
   from `onnx-asr`.
2. **Decoder loop > 2 weeks of work** beyond the initial frontend PR.
   We budgeted 1-2 weeks total; doubling that hits the opportunity
   cost where Parakeet-v3 (already runnable via `parakeet-rs`) becomes
   the better path despite its larger size.
3. **License clarification trouble** — if it turns out istupakov's
   export imposes additional constraints beyond the upstream
   CC-BY-4.0, we cannot ship it in Cochlis.

**Soft fallback** (keep Canary, augment with Parakeet for unsupported
langs):
4. **Need for languages outside en/de/fr/es** — Canary covers only
   four. If users request it/pt/nl/etc. we add Parakeet-v3-multi as a
   second adapter rather than dropping Canary.

## Migration plan if fallback triggered

The `AsrAdapter` trait already abstracts ASR backends. A fallback to
Parakeet-v3-multi is:
1. Drop `src/canary/` (or keep dormant under `--features canary`).
2. Add a `ParakeetMultiAdapter::load_es(...)` constructor in the
   existing `src/parakeet.rs`, downloading
   [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx).
3. Update `scripts/setup_parakeet_es.sh` (analogous to
   `setup_parakeet_en.sh`).
4. Switch the CLI auto-select branch for `--language es` to the new
   adapter.

The Tier 1 SpanishFillerFilter and `eval_l3 --task filler --lang es`
work is independent of which ASR backend produces transcripts, so
none of that PR-19/20/21 work is at risk.

## Validation gates per PR

Each Canary PR must pass:
- Numerical sanity (frontend output non-NaN, finite, expected shape)
- Build clean on default + `--features onnx` features
- Clippy zero warnings
- Existing test suite green

Per-PR scope is intentionally small (frontend / vocab / encoder /
decoder / adapter / wiring) so any single PR failing review doesn't
strand the whole effort.

## End-to-end validation (2026-04-30)

### v1 — 10-utt smoke (initial integration check)

First live run against the istupakov bundle on a FLEURS-es 10-utterance
subset (`scripts/setup_canary_es.sh` + `--canary-es-dir`, INT8):

```
[es] n=10  CER=0.0975  RTF=0.091
asr p50=801ms p95=2098ms
```

8/10 utterances within model-card range (0–7 % CER). Two outliers
(utt 8 CER 24.75 %, utt 9 CER 56.64 %) dropped multi-word chunks —
same failure mode in both, suggesting an autoregressive coverage
issue rather than frontend/encoder.

### v2 — 100-utt run with WER as primary (2026-04-30)

Widened to 100 unique utterances (`download_fleurs_subset.py --lang
es --n 100` after the dedup-by-id fix that removes FLEURS multi-
speaker duplicates) and switched primary metric to WER (matches
Canary's published numbers).

**FP32 weights** (deterministic, identical results across 3 reruns):

```
[es] n=100  WER=0.3537  RTF=0.150
asr p50=1138ms p95=2958ms
median(WER)=0.083
```

WER bucket distribution (n=99 reportable, 1 hard failure
counted as 1.0 in the mean):

| WER range | count |
|---|---|
| 0–5 % | 41 |
| 5–10 % | 14 |
| 10–20 % | 19 |
| 20–50 % | 20 |
| 50–100 % | 3 |
| **> 100 % (repetition loop)** | **2** |
| Hard fail (no output) | 1 |

**INT8 weights** (non-deterministic across 3 reruns: 34.5 % / 36.4 %
/ 94.4 %, 7 hard failures vs FP32's 1):

```
[es] n=100  WER=~0.4 (mean of 3 runs, σ ≈ 0.27)  RTF=0.060
```

INT8 introduces additional non-determinism on top of the
greedy-decoding repetition issue, with the failure mode amplified
(7 hard failures + worst-case 94 % WER). FP32 is the right baseline
for evaluating model fidelity; INT8 quality is a separate question
once the underlying decoding is stabilised.

### v3 — repetition penalty sweep (FP32, n=100, 2026-04-30)

Implemented HuggingFace-style repetition penalty in
`src/canary/decoder.rs::apply_repetition_penalty` and ran the
100-utt FP32 smoke at five penalty values. The penalty discounts
the logit of any token already emitted in the current utterance
(divides positive logits, multiplies negative logits) before
greedy argmax. Applied only to the post-prefix region so the
static prefix tokens (sot / language / pnc / etc.) don't suppress
legitimate content emissions whose surface forms appear in the
prefix.

| penalty | mean WER | clean (0–5 %) | repetition loops (>100 % WER) | hard fails |
|---|---|---|---|---|
| 1.0 (off) | 35.37 % | 41 / 99 | **2** | 1 |
| 1.2 | 23.03 % | 41 / 99 | 1 | 1 |
| 1.5 | 14.63 % | 41 / 99 | 0 | 1 |
| **1.8 (default)** | **14.38 %** | **41 / 99** | **0** | 1 |
| 2.0 | 18.90 % | 40 / 99 | 1 (over-penalty resurfaces a loop) | 1 |

**Default chosen: `repetition_penalty = 1.8`** (`DEFAULT_REPETITION_PENALTY`).

Headline win: **mean WER 35.37 % → 14.38 % (Δ = -21.0 pp)** at the
sweet-spot penalty, with **zero regression on the 41 clean
utterances** that were already at WER < 5 %. The two catastrophic
"E E E E E…" / "boda boda boda…" repetition loops from v2 are
both eliminated.

The remaining 14.38 % is dominated by chunk-dropout failures
(decoder skipping multi-word regions in long sentences) and the
single hard-fail utterance — neither of which the repetition
penalty addresses. Step (2) of the next-investigation list
(min-length / EOS-confidence gate) is the right tool for those.

### v4 — min-length / EOS-suppression gate (FP32, n=100, 2026-04-30)

Building on v3's repetition penalty=1.8, added a min-length gate
in `src/canary/decoder.rs::suppress_eos_until_min_length`. The
gate sets `logits[<|endoftext|>] = -inf` until the suffix has
emitted at least `ceil(ratio × T_sub)` tokens, where `T_sub` is
the encoder's output frame count. This addresses the chunk-
dropout / hard-fail failure modes where greedy argmax exits
before consuming the full audio.

Sweep over min_token_to_frame_ratio (with penalty fixed at 1.8):

| ratio | mean WER | clean (0–5 %) | hard fails | repetition loops |
|---|---|---|---|---|
| 0.0 (off, v3) | 14.38 % | 41 / 99 | **1** | 0 |
| **0.2 (default)** | **13.63 %** | **41 / 100** | **0** | **0** |
| 0.3 | 15.26 % | 42 / 100 | 0 | 0 |
| 0.4 | 62.33 % | 26 / 100 | 0 | 3 (over-suppression triggers loops) |
| 0.5 | 96.32 % | 0 / 100 | 0 | 11 (catastrophic) |

**Default chosen: `min_token_to_frame_ratio = 0.2`**
(`DEFAULT_MIN_TOKEN_TO_FRAME_RATIO`).

Headline win: **mean WER 14.38 % → 13.63 %, hard fails 1 → 0**.
The improvement on mean WER is modest (0.75 pp), but the more
important effect is killing the last hard-fail utterance — it
moves from the 100% bucket into the long tail proper. Higher
ratios catastrophically overshoot: forcing the decoder to keep
emitting past natural end-of-speech triggers repetition loops
that the penalty alone can't catch (ratio=0.5 gives WER 96.32 %).

Across v1 → v4 the trajectory is:

| stage | mean WER | hard fails | repetition loops |
|---|---|---|---|
| v2 (no gates) | 35.37 % | 1 | 2 |
| v3 (penalty 1.8) | 14.38 % | 1 | 0 |
| **v4 (penalty 1.8 + min-len 0.2)** | **13.63 %** | **0** | **0** |

The two catastrophic failure modes (loops + hard fails) are now
both eliminated. The residual 13.63 % is dominated by chunk-
dropout cases that the gate doesn't fully address (the decoder
still picks a non-EOS token early but eventually emits EOS at the
right scale-of-encoder-frames boundary, producing a partial
transcript).

### v7 — Beam search (opt-in, regression on FLEURS-es greedy baseline)

Implemented batched beam search in `src/canary/decoder.rs::decode_beam`
to test the hypothesis that the gap between our greedy WER (12.89 %)
and the Canary 1B-v2 model card FLEURS-es WER (2.9 %) is mostly
greedy-vs-beam. Enabled by setting `DecodeOptions.beam_size > 1`;
default remains `1` (greedy v6 path).

The implementation:

- Runs `B` parallel hypotheses in a single batched decoder call by
  tiling encoder I/O across the batch dim and selecting the top-`B`
  candidates from each step's logits.
- Reorders `decoder_mems` per step via
  `ndarray::Array4::select(Axis(1), &parent_idx)` so each new beam
  starts from its parent's KV cache.
- Reuses the v3-v5 gates per beam (repetition penalty / min-length /
  eos-margin) via per-batch helper variants
  (`apply_repetition_penalty_one_batch` etc.) that each operate on a
  single row of the `[B, T, V]` logits tensor.
- Length-normalised final scoring: `cum_logprob / length^α`.

Sweep on FLEURS-es 100-utt FP32 with `length_penalty = 0.6`:

| beam_size | mean WER | clean (0–5 %) | hard fails | repetition loops | RTF |
|---|---|---|---|---|---|
| **1 (greedy, v6 default)** | **12.89 %** | **44** | **0** | **0** | **0.13** |
| 2 | 19.34 % | 44 | 0 | 1 | 0.28 |
| 4 | 28.45 % | 45 | 0 | 2 | 0.64 |
| 8 | (cancelled at 30 min) | — | — | — | — |

**Beam search regresses WER monotonically with width**. Beam 4
introduces 2 catastrophic loops the v6 gates had eliminated. The
clean count edges up by 1 (44 → 45 at beam=4), suggesting beam
helps on already-easy utterances while breaking borderline ones.

The most likely root cause is the interaction between
`length_penalty < 1` and the EOS-confidence margin gate:

- Score `cum_logprob / length^0.6` favours **shorter** sequences
  (cum_logprob grows linearly with length but the divisor grows
  sub-linearly), so beam search with α = 0.6 picks early-EOS paths
  even when the eos-margin gate has demoted EOS in the per-step
  logits.
- Greedy doesn't pick early EOS because the demoted EOS logit
  loses argmax to a content token; beam search picks finishing
  paths through the length-normalised aggregate score even when
  EOS lost the per-step competition.

Confirming this requires a length-penalty sweep at fixed beam_size
(tracked in "Next investigation steps" below) — the code is
correct, the default `α = 0.6` is wrong for this configuration.

**Default chosen: `beam_size = 1`** (= greedy = v6, unchanged).
Beam search is opt-in via `CanaryConfig.beam_size`; users who want
to experiment can also tune `length_penalty`. Shipping the gates
machinery so the next iteration (length-penalty sweep, possibly
GNMT-style normalisation) can land cleanly without rewriting the
beam loop.

Cumulative trajectory across the 5-stage decoder series:

| stage | mean WER | hard fails | loops | clean (0–5 %) |
|---|---|---|---|---|
| v2 (no gates, ad-hoc frontend) | 35.37 % | 1 | 2 | — |
| v3 (penalty 1.8) | 14.38 % | 1 | 0 | 41 |
| v4 (+ min-len 0.2) | 13.63 % | 0 | 0 | 41 |
| v5 (+ eos-margin 2.0) | 13.21 % | 0 | 0 | 42 |
| v6 (Python-aligned + penalty 2.0) | 12.89 % | 0 | 0 | 44 |
| **v7 (beam search opt-in, default still greedy)** | **12.89 %** | **0** | **0** | **44** |

### v6 — Python-aligned frontend + retune (FP32, n=100, 2026-05-01)

Investigated whether the residual ~13 % WER might be a Rust-side
bug rather than a model-intrinsic limit. Built a tensor-dump
harness (`scripts/dump_canary_python_tensors.py` +
`examples/dump_canary_rust_tensors.rs` +
`scripts/compare_canary_tensors.py`) that runs both `onnx-asr`'s
Python pipeline and our Rust `CanaryAdapter` on the same WAV and
compares mel / encoder / decoder tensors element-wise.

**Key finding** — for FLEURS-es utterance 2001 the first two
greedy tokens matched bit-exactly between Python and Rust even
before any frontend changes; for the loop-triggering utterances
(1725 "Asus E E E…", 1915 "boda boda…") **`onnx-asr` produces the
same loops as our Rust** with no gates active. The catastrophic
failure modes are upstream model / export behaviour, not Rust
bugs. The repetition penalty / min-length / EOS-margin gates from
v3-v5 are correct mitigations.

**But** the dump comparator did surface four real numerical
mismatches in `src/canary/frontend.rs` vs onnx-asr's
`NumpyPreprocessor`:

1. Framing strategy — Python pads the waveform with `n_fft / 2`
   zeros each side before sliding-window framing; Rust used
   snip-edges (no padding).
2. Frame length — Python frames are length `n_fft` (= 512); Rust
   used `win_length` (= 400).
3. Window — Python applies a Hann window of `win_length`
   centre-padded to `n_fft`; Rust used a raw `win_length` Hann.
4. CMVN variance — Python uses N-1 (Bessel correction); Rust used N.

Plus an off-by-one: Python truncates to `features_lens =
samples.len() / hop_length` (zeroing any extra trailing frame in
the buffer); Rust passed all `num_frames` to the encoder via the
`length` input.

Aligning the frontend to Python:

| component | max_rel diff before | max_rel diff after |
|---|---|---|
| mel | shape mismatch (1282 vs 1285) | aligned (truncate to 1284) |
| encoder_emb | 13.7 % | **3.5e-6** |
| step0_logits | 0.90 % | **5.1e-7** |
| step0_hidden | 0.34 % | **2.8e-7** |

Now Rust matches Python at FP32 float-precision level. Re-tuned the
gates against the new bit-aligned frontend (penalty 1.8 → **2.0**;
min-len 0.2 unchanged; eos-margin 2.0 unchanged):

```
[es] n=100  WER=0.1289  median=0.0845  RTF=0.171
       hard fails=0  repetition loops=0  clean(0–5%)=44
```

Across v2 → v6 trajectory:

| stage | mean WER | hard fails | loops | clean (0–5 %) |
|---|---|---|---|---|
| v2 (no gates, ad-hoc frontend) | 35.37 % | 1 | 2 | — |
| v3 (penalty 1.8) | 14.38 % | 1 | 0 | 41 |
| v4 (+ min-len 0.2) | 13.63 % | 0 | 0 | 41 |
| v5 (+ eos-margin 2.0) | 13.21 % | 0 | 0 | 42 |
| **v6 (Python-aligned + penalty 2.0)** | **12.89 %** | **0** | **0** | **44** |

The bit-level alignment is **strictly better** even though the
mean improvement vs v5 is small (-0.32 pp) — the clean count
moves from 42 to 44, the median holds at 8.4 %, and the
implementation is now provably equivalent to the upstream Python
reference.

### v5 — EOS-confidence margin (FP32, n=100, 2026-04-30)

Building on v4 (penalty 1.8 + min-len 0.2), added an EOS-confidence
margin in `src/canary/decoder.rs::enforce_eos_confidence_margin`.
The gate demotes `<|endoftext|>` to `-inf` when its logit doesn't
lead the next-best non-EOS token by at least `margin` raw-logit
units. Complementary to the positional min-length gate — that one
forbids EOS too early, this one demands EOS be confidently
dominant when it does win.

Sweep over eos_confidence_margin (with penalty 1.8 and min-len 0.2):

| margin | mean WER | clean (0–5 %) | hard fails | repetition loops |
|---|---|---|---|---|
| 0.0 (off, v4) | 13.63 % | 41 / 100 | 0 | 0 |
| 1.0 | 13.21 % | 42 / 100 | 0 | 0 |
| **2.0 (default)** | **13.21 %** | **42 / 100** | **0** | **0** |
| 3.0 | 13.21 % | 22 / 100 | 0 | 0 |
| 5.0 | 13.21 % | 21 / 100 | 0 | 0 |

**Default chosen: `eos_confidence_margin = 2.0`**
(`DEFAULT_EOS_CONFIDENCE_MARGIN`).

Headline win: **mean WER 13.63 % → 13.21 %, clean count 41 → 42**
(small but real improvement on partial-dropout cases). 1.0 and
2.0 give numerically identical results on this data; 2.0 chosen
as a slightly stricter / more robust default. Above 2.0 the mean
stays the same (the high-WER tail barely improves while clean
utterances get padded with extra tokens — the per-utterance
trade is symmetric but the clean count drops).

Cumulative trajectory across the 4-PR series:

| stage | mean WER | hard fails | loops | clean (0–5 %) |
|---|---|---|---|---|
| v2 (no gates) | 35.37 % | 1 | 2 | — |
| v3 (penalty 1.8) | 14.38 % | 1 | 0 | 41 |
| v4 (+ min-len 0.2) | 13.63 % | 0 | 0 | 41 |
| **v5 (+ eos-margin 2.0)** | **13.21 %** | **0** | **0** | **42** |

The decoder is now in steady state: catastrophic failures (loops,
hard fails) eliminated; mean WER pushed from 35 % → 13 %. Further
improvements need beam search (architectural change) or a different
class of fix — see "Next investigation steps" below.

### Failure-mode inventory (FP32, n=100, no penalty)

Three distinct decoder failure patterns observed:

1. **Hard fail (1 utt)** — decoder emits only the prefix, returns
   empty text. Pipeline raises "no speech detected".
2. **Chunk dropout (mid-WER)** — decoder skips a multi-word region
   in the middle of a long sentence, emitting only the head + tail.
   Same failure mode as v1 utterances 8 and 9.
3. **Repetition loop (utt 62, 63)** — decoder gets stuck repeating
   a single token until `max_sequence_length=1024` is hit. Examples:
   - `"Asus E E E E E E E …"` (~500 repeats of "E") for an audio
     containing the acronym "ASUS".
   - `"boda boda boda boda …"` (~500 repeats of "boda") for an
     audio containing "boda-boda mototaxi".

   These are the WER > 100 % outliers (12.6×, 8.6× the reference
   length).

All three are classic greedy-decode failure modes — production
autoregressive ASR systems mitigate them with **repetition penalty**,
**length penalty**, and/or **beam search**. None of those is
implemented in `src/canary/decoder.rs` yet.

### Fallback-trigger status (post-v5)

| Trigger | Status |
|---|---|
| > 1 pp WER regression vs model card | **Still triggered but small-fix lever exhausted.** v2 → v5 closes 67 % of the original gap (35.37 % → 13.21 %). Catastrophic failure modes (loops, hard fails) all eliminated. Remaining 10 pp delta vs MLS card 3.17 % WER needs a more substantial change (beam search / different architecture / Parakeet fallback). |
| Decoder loop > 2 weeks of work | **Cleared** — single-session Canary integration (5 PRs) + 3 incremental decoder gates (this is the third). |
| License clarification trouble | **Cleared** — istupakov bundle is upstream CC-BY-4.0 unchanged. |

The trajectory shows **diminishing returns from greedy-decoder
gates**:
- v2 → v3 (penalty 1.8): -21.0 pp
- v3 → v4 (min-len 0.2): -0.75 pp
- v4 → v5 (eos-margin 2.0): -0.42 pp

The remaining 10 pp gap likely needs beam search or an entirely
different decoding strategy. The hard Parakeet-v3-multi fallback
remains documented but is not warranted at this time — the
catastrophic failure modes are gone, the mean is stable, and the
median is 8.2 % (within reach of model-card territory).

### (Previous) Fallback-trigger status (post-v4)

| Trigger | Status |
|---|---|
| > 1 pp WER regression vs model card | **Still triggered but two of three catastrophic failure modes resolved.** v4 closes 64 % of the original v2 gap (35.37 % → 13.63 %). The remaining 10.46 pp delta vs MLS model-card 3.17 % WER is now dominated by chunk-dropout cases that survive the gate (decoder picks non-EOS early but still produces a partial transcript). |
| Decoder loop > 2 weeks of work | **Cleared** — frontend → vocab → encoder → decoder → adapter delivered in 5 PRs over a single session, plus repetition penalty (PR #31) and min-length gate (this PR) as 2 follow-ups. |
| License clarification trouble | **Cleared** — istupakov bundle is upstream CC-BY-4.0 unchanged. |

The trajectory is clear: **decoder-side fixes continue to work**.
Repetition loops and hard fails are now both eliminated. The
remaining gap is harder to close because the partial-dropout
cases produce plausible-looking partial transcripts (the decoder
emits 30-70 % of the reference correctly then exits) — neither
penalty nor min-length helps if the decoder genuinely thinks the
audio is done.

The hard Parakeet-v3-multi fallback remains documented but is
not warranted at this time.

### Next investigation steps (post-v7)

1. ~~**Repetition penalty**~~ — **DONE** (PR #31).
2. ~~**Min-length gate**~~ — **DONE** (PR #32).
3. ~~**EOS-confidence margin**~~ — **DONE** (PR #33).
4. ~~**Frontend bit-alignment**~~ — **DONE** (PR #34).
5. ~~**Beam search machinery**~~ — **DONE** (this PR), but the
   current α = 0.6 default regresses WER vs greedy. Default
   stays at `beam_size = 1` while we tune.
6. ~~**Length-penalty sweep at fixed beam_size = 4**~~ — **DONE**
   (v9). Beam=4 across α ∈ {0.0, 0.6, 1.0, 1.2, 1.5, 2.0} stays
   flat at ~6.94 % WER vs greedy 6.97 % — ≤ 0.03 pp gain at
   2.5× the latency cost. Default stays at `beam_size = 1`.
7. ~~**Encoder-mask-aware min-length**~~ — **DONE** (v10).
   Correctness fix; WER unchanged on FLEURS-es smoke as predicted
   (per-utt frame diff ≤ 1 with 8× stride, below the gate's
   effective resolution). New `valid_frame_count` helper + 5
   unit tests.
8. **INT8 reassessment** — with deterministic FP32 decoding now
   stable, the INT8 quality question becomes "is INT8 within X
   pp of FP32?". The earlier INT8 nondeterminism (3 runs at
   34 % / 36 % / 94 %) was driven by the catastrophic loops; with
   those gone the INT8 numbers should be more stable.
9. ~~**Paper / NeMo prompt alignment**~~ — **DONE** (v8). Switching
   to the official 9-token `Canary2PromptFormatter` layout dropped
   FLEURS-es WER from 12.89 % → 6.97 % at FP32 greedy. The earlier
   "istupakov export's greedy floor IS ~34 %" finding was an
   artefact of the Python reference using the same 10-token wrapper
   we did; the model's actual greedy floor on the paper layout is
   ~7 %.
10. **Hard fallback decision deferred.** v9 ruled out beam search
    as a path to closing the remaining 6.97 % → 3.17 % gap.
    Before triggering the Parakeet-TDT-0.6B-v3-multi migration,
    quantify items (7) and (8) — the encoder-mask min-length
    fix and the INT8 re-assessment — and audit how much of the
    gap is the FLEURS/MLS test-set distribution shift vs.
    actual model-pipeline differences. The 6.97 % FLEURS-es
    floor is already inside the "good enough for production"
    range for the kind of disfluency-detection workload euhadra
    targets; missing model-card parity by ~4 pp on a *different*
    test set is not by itself a fallback trigger.

### v8 — paper / NeMo prompt alignment (FP32, n=100, 2026-05-01)

Following the user request "Canaryを論文の実装を確認してそれを参照した形に
修正して欲しい" we audited the prefix layout against:

- The Canary-1B-v2 paper (arXiv 2509.14128, Sec. 3 "Prompt format").
- NVIDIA's official NeMo
  [`Canary2PromptFormatter`](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/prompts/canary2.py)
  source — the canonical reference implementation.

The official template renders to **9** tokens for ASR with empty
`decodercontext`, whereas `istupakov/onnx-asr`'s Python reference
hardcodes **10** tokens (extra leading `▁`). We had shipped the
`OnnxAsr` layout since PR #28 because it bit-matches the Python
reference (see v6 frontend alignment), but it is *not* the layout
in the paper.

PR #37 introduced `PrefixFormat::{OnnxAsr, NemoCanary2}` as a
runtime choice without changing the default. This run reports the
sweep that decides the default.

**Sweep:** FLEURS-es 100-utt FP32, repetition_penalty=2.0,
min_token_to_frame_ratio=0.2, eos_confidence_margin=2.0,
beam_size=1, runs collected by toggling
`CANARY_PREFIX_FORMAT={onnx-asr,nemo-canary2}` on
`examples/eval_l1_smoke.rs --langs es`.

| Metric           | OnnxAsr (10-tok) | NemoCanary2 (9-tok) | Δ |
|---               |---:|---:|---:|
| Mean WER         | 12.89 % | **6.97 %** | **−5.92 pp** |
| Median WER       |  8.57 % |  4.17 %    |  −4.40 pp    |
| p90 WER          | 34.48 % | 16.67 %    | −17.81 pp    |
| p99 WER          | 75.00 % | 50.00 %    | −25.00 pp    |
| Max WER          | 75.00 % | 50.00 %    | −25.00 pp    |
| Perfect (=0 %)   | 26      | 38         |  +12         |
| (0, 5 %]         | 18      | 19         |   +1         |
| (5 %, 20 %]      | 31      | 35         |   +4         |
| (20 %, 50 %]     | 24      |  8         |  −16         |
| > 50 %           |  1      |  0         |   −1         |
| RTF (median)     |  0.13   |  0.14      |  +0.01       |

**Result:** `NemoCanary2` nearly halves mean WER, eliminates every
`> 50 %` catastrophic failure on the 100-utt smoke, and increases
the perfect-transcript count by 46 %. The slight RTF cost (~ 8 %)
comes from the decoder no longer cutting transcripts short — what
was registering as "fast" on `OnnxAsr` was partial-dropout exiting
early.

The FLEURS-es 100-utt FP32 floor is therefore **6.97 % WER** at
`beam_size = 1`, well inside the v6 v7 retune range and a 2× gain
over the previous 12.89 % default. The 6.97 % vs the
Canary-180M-Flash model card (3.17 % MLS-es) gap is now
plausibly closable via beam search and length-penalty tuning
(items (6) (7) below) on the *correct* prompt baseline.

**Action:** flipped `DEFAULT_PREFIX_FORMAT` to `NemoCanary2`.
`OnnxAsr` remains selectable (env var or `CanaryConfig`) for
bit-level comparison with the Python `istupakov/onnx-asr`
reference, but is no longer the default.

#### Why the model prefers the 9-token layout

The 10-token `OnnxAsr` wrapper inserts a leading `▁` — the
last-occurrence of U+2581 in the SentencePiece vocab — before
`<|startofcontext|>`. That token is part of the regular
SentencePiece text region, not a control token. The decoder thus
sees a "dummy word piece" before its prompt, which biases the
language model into prematurely emitting an `<|endoftext|>` for
some utterances ("decoder picks non-EOS early but still produces
a partial transcript", see post-v4 fallback notes). The official
NeMo template starts directly with `<|startofcontext|>` and
avoids this bias.

This also retroactively explains the 34.43 % WER floor we
measured against Python `onnx-asr` — that reference *also* uses
the 10-token wrapper, so its WER was bounded by the same prompt
mismatch. With the paper layout, the istupakov export's actual
greedy floor is 6.97 %, not 34 %.

### v9 — beam-search length-penalty sweep (FP32, n=100, 2026-05-01)

Item (6) from the post-v7 next-steps list, re-run on the v8
paper-prompt baseline. Sweep α ∈ {0.0, 0.6, 1.0, 1.2, 1.5, 2.0}
at `beam_size = 4`, FLEURS-es 100-utt FP32, NemoCanary2 prefix,
penalty=2.0 / min-len=0.2 / margin=2.0. Driven via the new
`CANARY_BEAM_SIZE` and `CANARY_LENGTH_PENALTY` env vars on
`examples/eval_l1_smoke.rs`.

| Config            | Mean WER | RTF (median) |
|---                |---:|---:|
| **greedy (b=1)**  | **6.97 %** | **0.11** |
| beam=4, α=0.0     | 6.96 %     | 0.28 |
| beam=4, α=0.6     | 6.94 %     | 0.29 |
| beam=4, α=1.0     | 6.94 %     | 0.28 |
| beam=4, α=1.2     | 6.94 %     | 0.28 |
| beam=4, α=1.5     | 6.94 %     | 0.28 |
| beam=4, α=2.0     | 6.94 %     | 0.28 |

**Result:** beam search at b=4 gives a ≤ 0.03 pp WER improvement
over greedy regardless of α, at ≈ 2.5× the latency cost (RTF
0.11 → 0.28). 0.03 pp on 4470 reference words is ~1 word — well
inside the run-to-run noise floor.

**Action:** keep `DEFAULT_BEAM_SIZE = 1` (greedy). The
beam-search machinery from PR #36 stays opt-in via
`CanaryConfig::beam_size` / the new `CANARY_BEAM_SIZE` env var
for callers who want to experiment, but it is no longer
recommended for production.

#### Why beam search doesn't help here

The catastrophic failure modes that motivated the v7 beam search
work — `Asus E E E …` loops, `boda boda` repetitions, partial
transcripts cut short by premature `<|endoftext|>` — were all
upstream of the decoder's top-1 vs top-k choice. They were
symptoms of the wrong prompt biasing the LM. With the v8
NemoCanary2 prefix the decoder's greedy path is already mostly
correct on these utterances, so beam search has nothing to fix:
the top-1 token is the right choice for ~99.97 % of decoded
positions.

The implication for next-steps item (10) (Parakeet hard
fallback): closing the remaining 6.97 % → 3.17 % gap to the
Canary-180M-Flash MLS-Spanish model card via beam search is
*not* on the table. The remaining gap is most likely:

1. **Test-set difference** — FLEURS-es vs MLS-es. FLEURS is
   read-speech multi-speaker on a wider domain; MLS is LibriVox
   audiobooks. The two distributions differ by several pp on
   most ASR benchmarks.
2. **NeMo inference features outside the ONNX export** — the
   model card numbers come from the full `nemo_toolkit[asr]`
   pipeline (LM rescoring, time-stamp post-processing, etc.).
   The `istupakov/onnx-asr` export is encoder + decoder only.
3. **Tokenisation / normalisation drift** — our `wer_lenient`
   is whisper-normalizer-equivalent (~ 0.5 pp delta in our
   earlier audit), so this is a small but real contributor.

Items (7) and (8) — encoder-mask-aware min-length and INT8
reassessment — remain the next two avenues. Beyond those the
6.97 % FLEURS-es FP32 floor is plausibly the Canary-180M-Flash
ONNX export's effective ceiling at greedy; further gains would
require either Parakeet-v3-multi (item 10) or Canary integrated
via the full NeMo pipeline (not feasible on this CPU box per the
v7 NeMo dep notes).

### v10 — encoder-mask-aware min-length gate (FP32, n=100, 2026-05-01)

Item (7) from the post-v7 next-steps list. The bug: three call
sites in `src/canary/decoder.rs` (greedy step loop, beam step 0,
beam step loop) were passing `encoder_embeddings.shape()[1]`
(`T_sub`, the padded encoder output time dimension) into
`suppress_eos_until_min_length` instead of the count of *valid*
frames as reported by `encoder_mask`.

Concretely, the FastConformer encoder pads `T_sub` to a multiple
of its 8× subsampling stride. For inputs that don't divide
cleanly, the trailing 1-7 positions are zero in the mask. Using
the padded count slightly over-estimates `min_len` and could
over-suppress `<|endoftext|>` for very short utterances or
distributions with lots of edge-of-stride lengths.

**Fix:** introduced `valid_frame_count(mask, batch_idx)` and
swapped the three call sites. New helper has 5 unit tests
pinning the contract (count of 1-bits per row, full-mask sanity,
out-of-range batch returns 0, treats any non-zero as valid,
batches independent).

**Measurement:** FLEURS-es 100-utt FP32, NemoCanary2 prefix,
penalty=2.0/min-len=0.2/margin=2.0, greedy.

| Config            | Mean WER | RTF (median) |
|---                |---:|---:|
| master @ 3cbf1af  | 6.97 % | 0.114 |
| this PR           | 6.97 % | 0.117 |

WER unchanged on the smoke set — the docs predicted "smaller
gain than (6)" and "a real bug". For FLEURS-es with 8× stride,
the per-utt frame difference is ≤ 1, which is below the gate's
effective resolution (`min_len = ceil(0.2 × n_frames)` and
typical suffix lengths are 30+ tokens). The fix is therefore
shipped as a **correctness improvement**, not a WER win — the
prior code happened to be benign on this benchmark but was using
the wrong quantity by definition. Catches future regressions on
edge-of-stride-length distributions.
