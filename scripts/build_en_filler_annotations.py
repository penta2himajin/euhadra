#!/usr/bin/env python3
"""
Build English Tier 1 filler annotations.

Generates `tests/evaluation/annotations/en_filler.jsonl` from a
structured Python source. Each entry is a tuple `(id, text,
filler_words)` where `filler_words` is the list of surface forms
the script then locates inside `text` to produce `FillerSpan`
records (`{start, end, label}` codepoint offsets).

The eval target is `SimpleFillerFilter::english()` in
`src/filter.rs`; cue surface forms are taken from its
`pure_fillers` / `contextual_fillers` / `multi_fillers` lists:

    pure:       um, uh, uhm, umm, hmm, er, ah, eh
    contextual: so, well, basically, actually, literally, right
    multi:      you know, i mean, you see, sort of, kind of

`SimpleFillerFilter::detect_spans` only emits contextual fillers
at sentence-initial positions; the gold annotations follow the
same rule so the F1 evaluator measures real false-positive risk
rather than penalising mid-sentence content uses of the words.

Usage:
  scripts/build_en_filler_annotations.py \\
    --out tests/evaluation/annotations/en_filler.jsonl

  scripts/build_en_filler_annotations.py --verify-only

License: same as the rest of euhadra.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Entry:
    utterance_id: str
    text: str
    fillers: List[str]


def entry(uid: str, text: str, fillers: Optional[List[str]] = None) -> Entry:
    return Entry(utterance_id=uid, text=text, fillers=fillers or [])


# -----------------------------------------------------------------
# Annotation entries — 8 pure / 4 contextual / 5 multi-word /
# 5 clean / 3 edge-case (contextual mid-sentence content uses).
# -----------------------------------------------------------------

ENTRIES: List[Entry] = [
    # ---- Pure fillers (always removed) ----
    entry("en_filler_pure_001", "um I think we should deploy now", ["um"]),
    entry("en_filler_pure_002", "uh let me check the logs", ["uh"]),
    entry("en_filler_pure_003", "the build is um running", ["um"]),
    entry("en_filler_pure_004", "please uh review the patch", ["uh"]),
    entry("en_filler_pure_005", "hmm that doesn't look right", ["hmm"]),
    entry("en_filler_pure_006", "er the meeting was rescheduled", ["er"]),
    entry("en_filler_pure_007", "ah I see what you mean", ["ah"]),
    entry("en_filler_pure_008", "eh maybe we should try again", ["eh"]),

    # ---- Contextual fillers (sentence-initial only) ----
    entry("en_filler_ctx_001", "so I think the test is passing", ["so"]),
    entry("en_filler_ctx_002", "well the deployment finished", ["well"]),
    entry("en_filler_ctx_003", "basically we need to rewrite the whole thing", ["basically"]),
    entry("en_filler_ctx_004", "actually the API returns JSON", ["actually"]),

    # ---- Multi-word fillers ----
    entry("en_filler_multi_001", "you know we should deploy now", ["you know"]),
    entry("en_filler_multi_002", "i mean the build is failing", ["i mean"]),
    entry("en_filler_multi_003", "you see the issue is in the cache", ["you see"]),
    entry("en_filler_multi_004", "it's sort of working but not really", ["sort of"]),
    entry("en_filler_multi_005", "we have kind of a problem with the schema", ["kind of"]),

    # ---- Clean controls (no fillers) ----
    entry("en_filler_clean_001", "the API returns a JSON response over HTTP"),
    entry("en_filler_clean_002", "configure the database connection in the environment"),
    entry("en_filler_clean_003", "deploy the new version after all tests pass"),
    entry("en_filler_clean_004", "check the error log for the root cause"),
    entry("en_filler_clean_005", "the algorithm runs in linear time on the input"),

    # ---- Edge cases — contextual fillers used as content words.
    # `SimpleFillerFilter::detect_spans` should NOT mark these because
    # they're not at sentence-initial position. Gold has empty
    # `fillers` so the F1 evaluator scores precision against the
    # corrector's contextual-skip behaviour.
    entry("en_filler_edge_001", "I think the deployment finished well", []),
    entry("en_filler_edge_002", "the API returns JSON so we can parse it", []),
    entry("en_filler_edge_003", "it works actually as advertised", []),
]


# -----------------------------------------------------------------
# Renderer + verifier
# -----------------------------------------------------------------

def find_span(text: str, needle: str, start_search: int) -> tuple[int, int]:
    """Locate `needle` in `text` starting at codepoint offset
    `start_search`. Returns `[start, end)` codepoint span."""
    pos = text.find(needle, start_search)
    if pos < 0:
        raise ValueError(
            f"needle {needle!r} not found in {text!r} after offset {start_search}"
        )
    pre = text[:pos]
    cp_start = len(pre)
    cp_end = cp_start + len(needle)
    return cp_start, cp_end


def emit_entry(e: Entry) -> dict:
    fillers_out = []
    cursor = 0
    for label in e.fillers:
        s, end = find_span(e.text, label, cursor)
        cursor = end
        fillers_out.append({"start": s, "end": end, "label": label})
    out: dict = {
        "utterance_id": e.utterance_id,
        "text": e.text,
    }
    if fillers_out:
        out["fillers"] = fillers_out
    return out


def verify_entry(rendered: dict) -> List[str]:
    errors: List[str] = []
    text = rendered["text"]
    for f in rendered.get("fillers", []):
        s, e = f["start"], f["end"]
        if s < 0 or e > len(text) or s >= e:
            errors.append(
                f"{rendered['utterance_id']} filler ({s}, {e}) out of bounds"
            )
            continue
        slice_ = text[s:e]
        if slice_ != f["label"]:
            errors.append(
                f"{rendered['utterance_id']} slice {slice_!r} != label {f['label']!r}"
            )
    return errors


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSONL path. Defaults to stdout.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Render entries and re-slice spans; report drifts.")
    args = parser.parse_args(argv)

    rendered = [emit_entry(e) for e in ENTRIES]

    errors: List[str] = []
    for r in rendered:
        errors.extend(verify_entry(r))
    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1

    if args.verify_only:
        print(f"OK: {len(rendered)} entries verified", file=sys.stderr)
        return 0

    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rendered) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body, encoding="utf-8")
        print(f"wrote {len(rendered)} entries → {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
