#!/usr/bin/env python3
"""
Build Spanish Tier 1 filler annotations from CIEMPIESS Test transcripts.

CIEMPIESS Test (`ciempiess/ciempiess_test` on HuggingFace, CC-BY-SA 4.0)
contains 8h of spontaneous Mexican Spanish radio with verbatim
orthographic transcripts. Fillers are *spelled out* in the
`normalized_text` field rather than tagged: tokens like "e", "este",
"pus", repetitions ("a la a la"), and partial words ("sie siempre")
appear as ordinary tokens. This script lifts those orthographic signals
into structured `FillerSpan` annotations matching the schema in
`src/eval/annotations.rs`, so the pipeline's `SpanishFillerFilter` can
be measured against an F1 ground truth.

License posture (see `docs/evaluation.md`):

  CIEMPIESS Test is CC-BY-SA-4.0. To avoid SA propagation into the
  euhadra source tree, this script is intended to be run in CI against
  a *cached* copy of the dataset, with the resulting JSONL kept under
  `data/cache/` (gitignored) and consumed only as transient input to
  `cargo eval-l3`. We do NOT commit the structured annotations, the
  raw transcripts, or any audio. F1 *scores* derived from the run are
  factual data and are not subject to SA.

Usage:
  scripts/build_es_filler_annotations.py \\
      --out data/cache/es_filler_annotations.jsonl

Run unit tests (no external deps required):
  scripts/build_es_filler_annotations.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


# -------------------------------------------------------------------
# Filler lexicons (Mexican Spanish, biased toward CIEMPIESS register)
# -------------------------------------------------------------------

# Tier A: pure fillers — always disfluencies regardless of position.
# Only entries here that are not also valid Spanish words.
PURE_FILLERS = frozenset(
    {
        # vowel-only hesitations
        "e", "eh", "ehh", "ehhh", "eee", "eeh", "eeeh", "ehm",
        # nasal hesitations
        "mm", "mmm", "hmm",
        # "ah" family
        "ah", "aah", "ahh",
        # "oh" family
        "oh",
    }
)

# Tier B: multi-word fillers — discourse markers that are almost always
# fillers when they occur as a fixed sequence.
MULTI_FILLERS = (
    ("o", "sea"),
)

# Tokens we never flag as a "partial word" (i.e. a prefix-of-next-word
# disfluency) because they are common Spanish content/function words
# that legitimately precede longer words. Without POS tags this lexical
# stoplist is the cheapest way to keep precision high.
PARTIAL_STOPLIST = frozenset(
    {
        "y", "a", "o", "u", "e",
        "de", "en", "el", "la", "lo", "los", "las", "le", "les",
        "un", "una", "uno", "unos", "unas",
        "se", "te", "me", "nos", "os",
        "no", "ni", "es", "ya", "si", "sí",
        "por", "con", "para", "del", "al", "que", "qué",
        "su", "sus", "mi", "mis", "tu", "tus",
    }
)

# Repetitions of these tokens are emphasis (rhetorical) rather than
# disfluency. Currently empty — radio data shows even "muy muy muy" is
# usually a hesitation pattern. Add cautiously.
REPETITION_STOPLIST: frozenset[str] = frozenset()


# -------------------------------------------------------------------
# Detection algorithm
# -------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    text: str
    start: int  # codepoint offset, half-open [start, end)
    end: int


@dataclass(frozen=True)
class FillerSpan:
    start: int
    end: int
    label: str

    def to_json(self) -> dict:
        return {"start": self.start, "end": self.end, "label": self.label}


def tokenize(text: str) -> List[Token]:
    """Whitespace-tokenize, tracking codepoint offsets."""
    out: List[Token] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        out.append(Token(text[start:i], start, i))
    return out


def detect_fillers(text: str) -> List[FillerSpan]:
    """
    Return filler spans (codepoint offsets) discovered in `text`.

    Spans never overlap. Detection passes are applied in this order so
    that more-specific patterns claim tokens before less-specific ones:

      1. Multi-word fillers   ("o sea")
      2. 2-token repetitions  ("a la a la" — flag the FIRST pair)
      3. 1-token repetitions  ("del del" — flag the FIRST occurrence)
      4. Partial words        ("sie siempre" — flag "sie")
      5. Pure fillers         ("e", "eh", "mm", ...)

    Inputs are expected to be the lowercased, punctuation-free
    `normalized_text` shape used by CIEMPIESS Test. Behaviour on punctuated
    text is unspecified.
    """
    tokens = tokenize(text)
    n = len(tokens)
    covered = [False] * n
    spans: List[FillerSpan] = []

    # ---- 1. Multi-word fillers ------------------------------------
    for phrase in MULTI_FILLERS:
        plen = len(phrase)
        i = 0
        while i + plen <= n:
            if all(
                not covered[i + k] and tokens[i + k].text.lower() == phrase[k]
                for k in range(plen)
            ):
                spans.append(
                    FillerSpan(
                        start=tokens[i].start,
                        end=tokens[i + plen - 1].end,
                        label=" ".join(phrase),
                    )
                )
                for k in range(plen):
                    covered[i + k] = True
                i += plen
            else:
                i += 1

    # ---- 2. 2-token immediate repetitions -------------------------
    i = 0
    while i + 3 < n:
        if any(covered[i + k] for k in range(4)):
            i += 1
            continue
        a0 = tokens[i].text.lower()
        a1 = tokens[i + 1].text.lower()
        b0 = tokens[i + 2].text.lower()
        b1 = tokens[i + 3].text.lower()
        if a0 == b0 and a1 == b1 and len(a0) + len(a1) >= 3:
            spans.append(
                FillerSpan(
                    start=tokens[i].start,
                    end=tokens[i + 1].end,
                    label=f"rep2:{a0} {a1}",
                )
            )
            covered[i] = covered[i + 1] = True
            i += 2  # skip the kept pair, allow further detection after it
        else:
            i += 1

    # ---- 3. 1-token immediate repetitions -------------------------
    for j in range(1, n):
        if covered[j] or covered[j - 1]:
            continue
        prev = tokens[j - 1].text.lower()
        cur = tokens[j].text.lower()
        if (
            prev == cur
            and len(cur) >= 2
            and cur not in REPETITION_STOPLIST
        ):
            spans.append(
                FillerSpan(
                    start=tokens[j - 1].start,
                    end=tokens[j - 1].end,
                    label=f"rep:{cur}",
                )
            )
            covered[j - 1] = True

    # ---- 4. Partial / abandoned words -----------------------------
    for j in range(n - 1):
        if covered[j] or covered[j + 1]:
            continue
        a = tokens[j].text.lower()
        b = tokens[j + 1].text.lower()
        if (
            len(a) >= 2
            and len(b) - len(a) >= 2
            and b.startswith(a)
            and a not in PARTIAL_STOPLIST
        ):
            spans.append(
                FillerSpan(
                    start=tokens[j].start,
                    end=tokens[j].end,
                    label=f"partial:{a}",
                )
            )
            covered[j] = True

    # ---- 5. Pure fillers ------------------------------------------
    for j in range(n):
        if covered[j]:
            continue
        t = tokens[j].text.lower()
        if t in PURE_FILLERS:
            spans.append(
                FillerSpan(
                    start=tokens[j].start,
                    end=tokens[j].end,
                    label=t,
                )
            )
            covered[j] = True

    spans.sort(key=lambda s: (s.start, s.end))
    return spans


# -------------------------------------------------------------------
# Dataset loader (HuggingFace `ciempiess/ciempiess_test`)
# -------------------------------------------------------------------


def _import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        print(
            f"missing dependency: {e}\n"
            "install with: pip install 'datasets>=2.21.0'",
            file=sys.stderr,
        )
        sys.exit(2)
    return load_dataset


def stream_ciempiess_test(
    name: str,
    split: str,
    limit: Optional[int],
) -> Iterable[Tuple[str, str]]:
    """Yield (utterance_id, normalized_text) tuples from a CIEMPIESS HF
    dataset. Uses streaming so we never download audio (we only need the
    transcript field). HF caches the metadata in ~/.cache/huggingface/."""
    load_dataset = _import_datasets()
    print(f"[ciempiess] loading {name} split={split} (streaming)", file=sys.stderr)
    ds = load_dataset(name, split=split, streaming=True)
    n_emitted = 0
    for sample in ds:
        if limit is not None and n_emitted >= limit:
            break
        uid = str(sample.get("audio_id") or sample.get("id") or n_emitted)
        text = sample.get("normalized_text")
        if not isinstance(text, str) or not text.strip():
            continue
        yield uid, text
        n_emitted += 1
    print(f"[ciempiess] emitted {n_emitted} utterances", file=sys.stderr)


def build_annotations(
    rows: Iterable[Tuple[str, str]],
    out_path: Path,
) -> dict:
    """Write JSONL annotations and return summary stats."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_with_filler = 0
    n_filler_spans = 0
    label_counts: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for uid, text in rows:
            spans = detect_fillers(text)
            anno = {
                "utterance_id": uid,
                "text": text,
                "fillers": [s.to_json() for s in spans],
            }
            f.write(json.dumps(anno, ensure_ascii=False) + "\n")
            n_total += 1
            if spans:
                n_with_filler += 1
            n_filler_spans += len(spans)
            for s in spans:
                # bucket by category prefix (rep / rep2 / partial / pure)
                cat = s.label.split(":", 1)[0] if ":" in s.label else "pure"
                label_counts[cat] = label_counts.get(cat, 0) + 1
    return {
        "out_path": str(out_path),
        "utterances_total": n_total,
        "utterances_with_filler": n_with_filler,
        "filler_spans_total": n_filler_spans,
        "label_counts": label_counts,
    }


# -------------------------------------------------------------------
# Self-test (no external deps)
# -------------------------------------------------------------------


def _assert_spans(text: str, expected: Sequence[Tuple[int, int, str]]) -> None:
    got = [(s.start, s.end, s.label) for s in detect_fillers(text)]
    assert got == list(expected), (
        f"\n  text     = {text!r}"
        f"\n  expected = {expected}"
        f"\n  got      = {got}"
    )


def self_test() -> None:
    # 1. Pure filler at sentence start (CIEMPIESS pattern: lone "e")
    _assert_spans(
        "e fuera del del aire",
        [
            (0, 1, "e"),
            (8, 11, "rep:del"),
        ],
    )

    # 2. 2-token repetition takes precedence over 1-token rep
    _assert_spans(
        "a la a la a las saga siempre",
        [(0, 4, "rep2:a la")],
    )

    # 3. Partial / abandoned word
    _assert_spans(
        "sie siempre estuvo leyendo",
        [(0, 3, "partial:sie")],
    )

    # 4. Multi-word filler "o sea"
    _assert_spans(
        "o sea ahora las mujeres",
        [(0, 5, "o sea")],
    )

    # 5. Three-fold repetition: "muy muy muy" → flag first two
    _assert_spans(
        "muy muy muy ligada con",
        [
            (0, 3, "rep:muy"),
            (4, 7, "rep:muy"),
        ],
    )

    # 6. Clean speech yields no spans
    _assert_spans("como una de las partes importantes es el capitulado", [])

    # 7. "este este libro" — repetition fires; bare "este libro" stays clean
    _assert_spans(
        "este este libro",
        [(0, 4, "rep:este")],
    )
    _assert_spans("este libro", [])

    # 8. Stoplist guards partial-word rule from common-prefix false positives
    #    "se sevir" — `se` is in PARTIAL_STOPLIST → no partial
    _assert_spans("se sevir mañana", [])

    # 9. UTF-8 / accented-character offsets are codepoint-based
    _assert_spans(
        "e canción mañana",
        [(0, 1, "e")],
    )
    # "perdón" contains a non-ASCII codepoint; verify offset arithmetic
    spans = detect_fillers("e perdón")
    assert spans == [FillerSpan(0, 1, "e")], spans

    # 10. Pure filler embedded mid-utterance
    _assert_spans(
        "hace rato comentaba e fuera del aire",
        [(20, 21, "e")],
    )

    # 11. Empty / single-token / whitespace edge cases
    assert detect_fillers("") == []
    assert detect_fillers("   ") == []
    assert detect_fillers("hola") == []

    # 12. tokenize() round-trip preserves offsets
    text = "e  hola   mundo"  # multiple spaces
    toks = tokenize(text)
    assert [(t.text, t.start, t.end) for t in toks] == [
        ("e", 0, 1),
        ("hola", 3, 7),
        ("mundo", 10, 15),
    ]
    for t in toks:
        assert text[t.start : t.end] == t.text

    # 13. build_annotations() round-trips through JSON correctly
    import io
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "es.jsonl"
        rows = [
            ("u1", "e fuera del del aire"),
            ("u2", "como una de las partes"),
        ]
        stats = build_annotations(rows, out)
        assert stats["utterances_total"] == 2
        assert stats["utterances_with_filler"] == 1
        assert stats["filler_spans_total"] == 2
        loaded = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        assert loaded[0]["fillers"] == [
            {"start": 0, "end": 1, "label": "e"},
            {"start": 8, "end": 11, "label": "rep:del"},
        ]
        assert loaded[1]["fillers"] == []

    # 14. Span text matches token text (sanity)
    text = "e fuera del del aire"
    for s in detect_fillers(text):
        seg = text[s.start : s.end]
        # span should be a non-empty substring of `text`
        assert seg and not seg.startswith(" ") and not seg.endswith(" "), seg

    print("self-test: OK", file=sys.stderr)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Build Spanish Tier 1 filler annotations from CIEMPIESS Test "
            "transcripts. Output is intended as transient CI input only "
            "(see SA-safety note in the module docstring)."
        )
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/cache/es_filler_annotations.jsonl"),
        help="output JSONL path (default: data/cache/es_filler_annotations.jsonl)",
    )
    p.add_argument(
        "--dataset",
        default="ciempiess/ciempiess_test",
        help="HuggingFace dataset name (default: ciempiess/ciempiess_test)",
    )
    p.add_argument(
        "--split",
        default="train",
        help="dataset split (default: train; ciempiess_test only ships one split)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap on number of utterances (default: full split)",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="run unit tests against the detection algorithm and exit",
    )
    args = p.parse_args(argv)

    if args.self_test:
        self_test()
        return 0

    rows = stream_ciempiess_test(args.dataset, args.split, args.limit)
    stats = build_annotations(rows, args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
