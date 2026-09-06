#!/usr/bin/env python3
"""Punctuation bake-off: BasicPunctuationRestorer vs XLM-R ONNX.

Scores the rule-based Tier 2 stopgap against the multilingual candidate
named in `docs/model-upgrade-candidates.md` §2.2
(`1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase`).

This is deliberately a Python harness, not a Rust example:

- The candidate's ONNX graph does not match `OnnxPunctuationRestorer`
  (SentencePiece + 4 heads vs BERT token-classification). A native
  adapter is follow-up work; measuring through the upstream
  `punctuators` package avoids baking a buggy port into the number.
- Synthetic gold construction already lives in Python
  (`scripts/build_punct_annotations.py`), same as the paragraph corpus.

Metrics (per language, micro-averaged):

- **punct F1** — slot F1 over punctuation marks between non-punct
  characters. This is what Tier 2 is for; WER/CER ablation understates
  it because `BasicPunctuationRestorer` only appends a terminal.
- **terminal accuracy** — does the hypothesis end with the same
  terminal mark as gold (。/./？/?/！/!).
- **latency p50/p95** — wall time per utterance.

Usage:
    scripts/build_punct_annotations.py --langs en,ja
    scripts/setup_punct_xlmr.sh
    scripts/eval_punctuation.py \
        --annotations data/punct_eval \
        --backends basic,xlmr \
        --xlmr-dir vendor/punct_xlmr \
        --output docs/benchmarks/punctuation/bakeoff.json

Requires: pip install punctuators onnxruntime sentencepiece omegaconf
(for the `xlmr` backend only; `basic` needs nothing beyond stdlib).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# ---------------------------------------------------------------------------
# Punctuation inventory + slot F1
# ---------------------------------------------------------------------------

# Marks we score. Keep this aligned with `build_punct_annotations.py`.
PUNCT_CHARS = set(".!?,:;。、？！，；：")
TERMINALS = set(".。？！?!")


def skeleton_and_slots(text: str) -> tuple[str, list[str]]:
    """Split `text` into a punct-free skeleton and per-char punct slots.

    `slots[i]` is the punctuation string that follows skeleton char i.
    Leading punctuation (rare in our gold) is dropped so skeletons
    still align between gold and hyp.
    """
    skeleton_chars: list[str] = []
    slots: list[str] = []
    pending = ""
    for ch in text:
        if ch in PUNCT_CHARS:
            if skeleton_chars:
                # Accumulate so "。”" style doubles still count as one
                # slot's content; we score char-by-char inside the slot.
                slots[-1] = slots[-1] + ch
            else:
                pending += ch  # leading — ignore for alignment
            continue
        if ch.isspace():
            # Preserve spaces in the skeleton so Latin word boundaries
            # survive; they are not punctuation.
            skeleton_chars.append(ch)
            slots.append("")
            continue
        skeleton_chars.append(ch)
        slots.append("")
    return "".join(skeleton_chars), slots


def normalize_skeleton(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


@dataclass
class SlotCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "SlotCounts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_slots(gold: str, hyp: str) -> SlotCounts:
    g_skel, g_slots = skeleton_and_slots(gold)
    h_skel, h_slots = skeleton_and_slots(hyp)
    if normalize_skeleton(g_skel) != normalize_skeleton(h_skel):
        # Soft fallback: score over the shorter aligned prefix and
        # treat the tail as FN/FP. Truecasers that also rewrite tokens
        # would hit this; the XLM-R candidate should not on our gold.
        n = min(len(g_slots), len(h_slots))
    else:
        n = min(len(g_slots), len(h_slots))

    counts = SlotCounts()
    for i in range(n):
        g = list(g_slots[i])
        h = list(h_slots[i])
        # Multiset compare of punct chars in this slot.
        for ch in PUNCT_CHARS:
            gc = g.count(ch)
            hc = h.count(ch)
            counts.tp += min(gc, hc)
            counts.fp += max(0, hc - gc)
            counts.fn += max(0, gc - hc)
    # Unaligned tails.
    for i in range(n, len(g_slots)):
        counts.fn += sum(1 for ch in g_slots[i] if ch in PUNCT_CHARS)
    for i in range(n, len(h_slots)):
        counts.fp += sum(1 for ch in h_slots[i] if ch in PUNCT_CHARS)
    return counts


def terminal_of(text: str) -> str | None:
    t = text.rstrip()
    if not t:
        return None
    ch = t[-1]
    return ch if ch in TERMINALS else None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _classify_script(text: str) -> str:
    latin = hangul = hanzi = kana = 0
    for c in text:
        o = ord(c)
        if c.isascii() and c.isalpha():
            latin += 1
        elif 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            hangul += 1
        elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            hanzi += 1
        elif 0x3040 <= o <= 0x30FF:
            kana += 1
    if kana > 0 and kana + hanzi >= latin and kana + hanzi >= hangul:
        return "kana"
    if hangul > latin and hangul > hanzi:
        return "hangul"
    if hanzi > latin and hanzi > hangul:
        return "hanzi"
    if latin > 0:
        return "latin"
    return "other"


def basic_restore(text: str) -> str:
    """Python mirror of `BasicPunctuationRestorer` in src/processor.rs.

    Kept intentionally dumb — the point of the bake-off is to quantify
    how far the stopgap sits below a real model, not to polish the
    stopgap mid-measurement.
    """
    if not text:
        return text
    script = _classify_script(text)
    capitalise = script == "latin"
    out: list[str] = []
    capitalize_next = capitalise
    for ch in text:
        if capitalize_next and ch.isalpha():
            out.append(ch.upper())
            capitalize_next = False
        else:
            out.append(ch)
            if capitalise and ch in ".!?":
                capitalize_next = True
            elif ch == "。":
                capitalize_next = False
    result = "".join(out)
    trimmed = result.rstrip()
    if not trimmed:
        return result
    want = {
        "latin": ".",
        "hangul": ".",
        "hanzi": "。",
        "kana": "。",
        "other": None,
    }[script]
    if want is None:
        return result
    last = trimmed[-1]
    already = {
        ".": set(".!?'\"。"),
        "。": set("。！？.!?"),
    }[want]
    if last not in already:
        result = trimmed + want
    return result


def load_xlmr(model_dir: Path):
    try:
        # Config is not re-exported from punctuators.models.__init__.
        from punctuators.models import PunctCapSegModelONNX
        from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX
    except ImportError as exc:
        raise SystemExit(
            "xlmr backend needs the `punctuators` package "
            "(pip install punctuators onnxruntime sentencepiece omegaconf): "
            f"{exc}"
        ) from exc

    # Hub bundle filenames match the package defaults (sp.model /
    # model.onnx / config.yaml). Directory mode avoids a second download.
    cfg = PunctCapSegConfigONNX(directory=str(model_dir))
    # CPU-only in CI / cloud agents; CUDA is fine if the user has it but
    # we do not want a hard dependency on a GPU provider list failing.
    return PunctCapSegModelONNX(cfg, ort_providers=["CPUExecutionProvider"])


def make_xlmr_fn(model_dir: Path) -> Callable[[str], str]:
    model = load_xlmr(model_dir)

    def restore(text: str) -> str:
        # `infer` returns List[List[str]] (sentences) when SBD is on.
        # Join with the language-appropriate separator: space for
        # Latin, empty for CJK. Detect from the *input* script so we
        # do not invent spaces inside Japanese.
        out = model.infer([text], apply_sbd=True)
        sents: list[str] = out[0] if out else []
        if not sents:
            return text
        script = _classify_script(text)
        joiner = "" if script in ("kana", "hanzi") else " "
        return joiner.join(s.strip() for s in sents if s.strip())

    return restore


BACKEND_BUILDERS = {
    "basic": lambda _args: basic_restore,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class BackendResult:
    name: str
    by_lang: dict = field(default_factory=dict)


def load_annotations(path: Path) -> list[dict]:
    if path.is_dir():
        rows = []
        # Two-letter lang files only (`en_punct.jsonl`), not smoke fixtures.
        for f in sorted(path.glob("[a-z][a-z]_punct.jsonl")):
            rows.extend(load_annotations(f))
        return rows
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = int(round((len(ys) - 1) * p))
    return ys[k]


def run_backend(
    name: str,
    restore: Callable[[str], str],
    rows: Iterable[dict],
) -> BackendResult:
    by_lang_counts: dict[str, SlotCounts] = defaultdict(SlotCounts)
    by_lang_term_ok: dict[str, list[int]] = defaultdict(list)
    by_lang_latency: dict[str, list[float]] = defaultdict(list)
    by_lang_n: dict[str, int] = defaultdict(int)

    for row in rows:
        lang = row["lang"]
        gold = row["gold"]
        inp = row["input"]
        t0 = time.perf_counter()
        hyp = restore(inp)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        by_lang_latency[lang].append(dt_ms)
        by_lang_counts[lang].add(score_slots(gold, hyp))
        g_term, h_term = terminal_of(gold), terminal_of(hyp)
        by_lang_term_ok[lang].append(1 if g_term and g_term == h_term else 0)
        by_lang_n[lang] += 1

    by_lang = {}
    for lang, counts in sorted(by_lang_counts.items()):
        lat = by_lang_latency[lang]
        term = by_lang_term_ok[lang]
        by_lang[lang] = {
            "n": by_lang_n[lang],
            "punct_precision": round(counts.precision, 4),
            "punct_recall": round(counts.recall, 4),
            "punct_f1": round(counts.f1, 4),
            "terminal_accuracy": round(sum(term) / len(term), 4) if term else 0.0,
            "latency_ms": {
                "p50": round(percentile(lat, 0.50), 2),
                "p95": round(percentile(lat, 0.95), 2),
                "mean": round(sum(lat) / len(lat), 2) if lat else 0.0,
            },
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
        }
    return BackendResult(name=name, by_lang=by_lang)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/punct_eval"),
        help="JSONL file or directory of *_punct.jsonl",
    )
    ap.add_argument(
        "--backends",
        default="basic,xlmr",
        help="Comma-separated: basic, xlmr",
    )
    ap.add_argument(
        "--xlmr-dir",
        type=Path,
        default=Path(os.environ.get("PUNCT_XLMR_DIR", "vendor/punct_xlmr")),
    )
    ap.add_argument("--langs", default="", help="Optional lang filter, e.g. en,ja")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    rows = load_annotations(args.annotations)
    if not rows:
        print(f"[error] no annotations under {args.annotations}", file=sys.stderr)
        return 2
    if args.langs:
        keep = {x.strip() for x in args.langs.split(",") if x.strip()}
        rows = [r for r in rows if r["lang"] in keep]
    print(f"[eval] {len(rows)} utterances")

    backends: list[tuple[str, Callable[[str], str]]] = []
    for name in [x.strip() for x in args.backends.split(",") if x.strip()]:
        if name == "basic":
            backends.append((name, basic_restore))
        elif name == "xlmr":
            if not (args.xlmr_dir / "model.onnx").exists():
                print(
                    f"[error] {args.xlmr_dir}/model.onnx missing; "
                    "run scripts/setup_punct_xlmr.sh first",
                    file=sys.stderr,
                )
                return 3
            print(f"[eval] loading xlmr from {args.xlmr_dir}…")
            backends.append((name, make_xlmr_fn(args.xlmr_dir)))
        else:
            print(f"[error] unknown backend: {name}", file=sys.stderr)
            return 2

    report = {
        "annotations": str(args.annotations),
        "n": len(rows),
        "backends": {},
    }
    for name, fn in backends:
        print(f"[eval] backend={name}")
        result = run_backend(name, fn, rows)
        report["backends"][name] = result.by_lang
        for lang, m in result.by_lang.items():
            print(
                f"  {lang}: F1={m['punct_f1']:.3f} "
                f"(P={m['punct_precision']:.3f} R={m['punct_recall']:.3f}) "
                f"term={m['terminal_accuracy']:.3f} "
                f"p50={m['latency_ms']['p50']:.1f}ms"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"[ok] wrote {args.output}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
