# Contributing to euhadra

Thanks for looking. This document covers what euhadra needs most, and how to
work on it.

## The most useful thing you can do right now

**Review the evaluation gold sets in your native language.**

euhadra gates quality on annotated ground truth — filler spans, self-correction
spans, phoneme corrections. Those annotations were drafted by Claude and have
**not been reviewed by native speakers**. Everything measured in English,
Japanese, Chinese and Korean rests on unverified annotations.

That is a real risk, not a formality. A defect that silently disabled Spanish
filler removal for any punctuated input survived for months, because the
Spanish filter was only ever checked against unpunctuated transcripts. Gold
that nobody has read can hide the same class of problem.

If you speak any of these languages, reviewing a few hundred annotated
utterances is worth more to this project than a feature.

Start at [`tests/evaluation/annotations/guidelines.md`](tests/evaluation/annotations/guidelines.md).
It defines the schema, the span conventions, per-language judgement rules, and
what §5 specifically asks reviewers to check.

## Adding a language

Text processing is per-language work. A filler lexicon is hand-written, and
punctuation and self-correction behave differently per script. euhadra will not
claim a language it cannot measure, so a new language needs both an
implementation and a way to check it.

### 1. Check whether a verbatim corpus already exists

Some speech corpora transcribe disfluencies directly — fillers spelled out,
repetitions kept, partial words preserved. Where one exists, gold can be
derived rather than annotated by hand. Spanish works this way: CIEMPIESS Test
spells fillers into the transcript, and `scripts/build_es_filler_annotations.py`
lifts them into structured spans.

Check [`docs/evaluation.md`](docs/evaluation.md) §2.5 for the corpora already
surveyed before annotating anything by hand.

### 2. Mind the licence before you annotate

This is the constraint people hit last and should hit first.

euhadra is MIT / Apache-2.0. Annotations derived from a **CC-BY-SA** corpus
carry ShareAlike, so committing them into this tree would propagate that licence
into the project. Annotations derived from a **non-commercial** corpus cannot
ship at all.

Two workable paths:

- **Corpus permits redistribution** (CC-BY, CC0, permissive): commit the
  generated JSONL under `tests/evaluation/annotations/`, alongside the script
  that regenerates it.
- **Corpus is CC-BY-SA** — commit only the generator. It must write to
  `data/cache/` (gitignored), fetch the corpus at run time, and be wired into
  CI so the score is computed rather than stored. Only the resulting metrics get
  committed, under `docs/benchmarks/`; measurements are facts and carry no
  ShareAlike obligation.

If the corpus is non-commercial or has unclear terms, say so in the issue before
doing the work. It is better to find another corpus than to produce annotations
that cannot be used.

### 3. What a language contribution consists of

- A filler filter matching the script's segmentation — whitespace, `、`, `，`.
  Route it through `FillerFilter::for_language` and add the `Language` variant;
  do not pair a whitespace tokeniser with a script that has no whitespace.
- Gold annotations, or a generator plus CI wiring per the licence rules above.
- A baseline entry so regressions are detectable.
- Tests. At minimum: the filler is removed, the content survives, and
  filler-free text passes through byte-for-byte.

A language that arrives without a way to measure it will be marked unmeasured in
the README rather than merged as supported.

## Development

```bash
cargo build                   # default build, no ML runtime
cargo build --features onnx   # ONNX ASR and text processing
cargo test                    # unit + integration
cargo clippy --all-targets    # lint
```

On Debian/Ubuntu, `cpal` needs ALSA headers for the `mic` and `cli` features:

```bash
sudo apt-get install -y pkg-config libasound2-dev
```

Feature combinations worth checking before opening a PR, since they compile
different code: `--features testing`, `--features clipboard`, `--features cli`,
`--features onnx`.

Optional pre-push hook running fmt and clippy:

```bash
git config core.hooksPath git-hooks
```

## Working style

**Tests come first.** Write a failing test, make it pass, then tidy. When a test
fails, fix the code rather than the test — deleting or skipping a test to get
green is not acceptable here.

**Do not weaken production code to satisfy a test.** If a test is genuinely
wrong, say why in the PR rather than quietly adjusting it.

**Rule-based text processing is a stopgap.** Implementations like
`BasicPunctuationRestorer` and `SpokenFormNormalizer` stand in until a dedicated
ONNX model exists. Say so in the doc comment when you add one.

**Keep the default build lean.** Anything pulling an ML runtime or a system
library belongs behind a feature flag. The default build has no ONNX and no
system dependencies, and that is deliberate.

## Pull requests

- [Conventional Commits](https://www.conventionalcommits.org/) for titles:
  `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`, with an
  optional scope (`feat(processor):`).
- Fill in [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md)
  and link the issue with `Closes #N`.
- Open as ready for review, not draft.
- Tests passing and zero warnings before you push.
- Do not change CI configuration without saying why in the PR.

## Scope

euhadra is a library, not an application. It ships adapter traits; native OS
integration and LLM refinement are seams for a consuming application to fill,
not things euhadra links in. See [`docs/spec.md`](docs/spec.md) §2.2 for why,
and the open issues for what is deliberately deferred.

If you are unsure whether something belongs in the core, open an issue before
building it.
