#!/usr/bin/env python3
"""
Build English Tier 2 self-correction annotations.

Generates `tests/evaluation/annotations/en_self_correction.jsonl`
from the structured Python source below. Each entry is a small
tuple `(id, text, reparandum, interregnum, repair)`; the script
string-searches the text to assign codepoint offsets `[start, end)`,
then emits the JSONL schema documented in
`tests/evaluation/annotations/guidelines.md`.

Mirrors the workflow used for `ja_self_correction.jsonl` /
`es_self_correction.jsonl`. The English cue closed-set is taken
from `src/processor.rs::SelfCorrectionDetector::correction_cues_en`:

    no, wait, sorry, i mean, actually, rather, no wait, or rather

`SelfCorrectionDetector::detect_english` requires a shared content
word between `reparandum` and `repair` (see
`count_shared_prefix_from_end`); each entry below is constructed so
the detector actually fires.

Usage:
  scripts/build_en_self_correction_annotations.py
    --out tests/evaluation/annotations/en_self_correction.jsonl

  scripts/build_en_self_correction_annotations.py --verify-only

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
class Repair:
    reparandum: str
    interregnum: str
    repair: str


@dataclass
class Entry:
    utterance_id: str
    text: str
    repairs: List[Repair]


def entry(uid: str, text: str, repairs: Optional[List[Repair]] = None) -> Entry:
    return Entry(utterance_id=uid, text=text, repairs=repairs or [])


def repair(reparandum: str, interregnum: str, repair_text: str) -> Repair:
    return Repair(reparandum=reparandum, interregnum=interregnum, repair=repair_text)


# -----------------------------------------------------------------
# Self-correction entries — covers all 8 cues from
# `correction_cues_en`, plus 5 clean controls and 2 edge cases.
# -----------------------------------------------------------------

ENTRIES: List[Entry] = [
    # ---- Cue: `no wait` (5) ----
    entry("en_self_corr_no_wait_001",
          "I will deploy to staging no wait I will deploy to production",
          [repair("I will deploy to staging", "no wait",
                  "I will deploy to production")]),
    entry("en_self_corr_no_wait_002",
          "the meeting is at three no wait the meeting is at four",
          [repair("the meeting is at three", "no wait",
                  "the meeting is at four")]),
    entry("en_self_corr_no_wait_003",
          "send the report to Alice no wait send the report to Bob",
          [repair("send the report to Alice", "no wait",
                  "send the report to Bob")]),
    entry("en_self_corr_no_wait_004",
          "I went to Paris no wait I went to Rome",
          [repair("I went to Paris", "no wait", "I went to Rome")]),
    entry("en_self_corr_no_wait_005",
          "the bug is in the cache no wait the bug is in the queue",
          [repair("the bug is in the cache", "no wait",
                  "the bug is in the queue")]),

    # ---- Cue: `or rather` (4) ----
    entry("en_self_corr_or_rather_001",
          "merge into the main branch or rather merge into the develop branch",
          [repair("merge into the main branch", "or rather",
                  "merge into the develop branch")]),
    entry("en_self_corr_or_rather_002",
          "we need ten servers or rather we need eight servers",
          [repair("we need ten servers", "or rather",
                  "we need eight servers")]),
    entry("en_self_corr_or_rather_003",
          "the deadline is Monday or rather the deadline is Tuesday",
          [repair("the deadline is Monday", "or rather",
                  "the deadline is Tuesday")]),
    entry("en_self_corr_or_rather_004",
          "she lives in Tokyo or rather she lives in Osaka",
          [repair("she lives in Tokyo", "or rather",
                  "she lives in Osaka")]),

    # ---- Cue: `i mean` (5) ----
    entry("en_self_corr_i_mean_001",
          "the build failed i mean the build succeeded",
          [repair("the build failed", "i mean", "the build succeeded")]),
    entry("en_self_corr_i_mean_002",
          "please move the artifacts i mean please copy the artifacts",
          [repair("please move the artifacts", "i mean",
                  "please copy the artifacts")]),
    entry("en_self_corr_i_mean_003",
          "the API returns XML i mean the API returns JSON",
          [repair("the API returns XML", "i mean",
                  "the API returns JSON")]),
    entry("en_self_corr_i_mean_004",
          "I have two cats i mean I have three cats",
          [repair("I have two cats", "i mean", "I have three cats")]),
    entry("en_self_corr_i_mean_005",
          "she works at Google i mean she works at Microsoft",
          [repair("she works at Google", "i mean",
                  "she works at Microsoft")]),

    # ---- Cue: `actually` (4) ----
    entry("en_self_corr_actually_001",
          "the test passes actually the test fails",
          [repair("the test passes", "actually", "the test fails")]),
    entry("en_self_corr_actually_002",
          "we deploy on Friday actually we deploy on Thursday",
          [repair("we deploy on Friday", "actually",
                  "we deploy on Thursday")]),
    entry("en_self_corr_actually_003",
          "I had two coffees actually I had three coffees",
          [repair("I had two coffees", "actually",
                  "I had three coffees")]),
    entry("en_self_corr_actually_004",
          "the file is in src actually the file is in tests",
          [repair("the file is in src", "actually",
                  "the file is in tests")]),

    # ---- Cue: `rather` (3) ----
    entry("en_self_corr_rather_001",
          "let's eat at six rather let's eat at seven",
          [repair("let's eat at six", "rather", "let's eat at seven")]),
    entry("en_self_corr_rather_002",
          "I prefer tea rather I prefer coffee",
          [repair("I prefer tea", "rather", "I prefer coffee")]),
    entry("en_self_corr_rather_003",
          "we need eight workers rather we need twelve workers",
          [repair("we need eight workers", "rather",
                  "we need twelve workers")]),

    # ---- Cue: `sorry` (4) ----
    entry("en_self_corr_sorry_001",
          "the office is on the third floor sorry the office is on the fourth floor",
          [repair("the office is on the third floor", "sorry",
                  "the office is on the fourth floor")]),
    entry("en_self_corr_sorry_002",
          "we have ten members sorry we have twelve members",
          [repair("we have ten members", "sorry",
                  "we have twelve members")]),
    entry("en_self_corr_sorry_003",
          "the test runs in five seconds sorry the test runs in seven seconds",
          [repair("the test runs in five seconds", "sorry",
                  "the test runs in seven seconds")]),
    entry("en_self_corr_sorry_004",
          "the meeting room is A sorry the meeting room is B",
          [repair("the meeting room is A", "sorry",
                  "the meeting room is B")]),

    # ---- Cue: `wait` (3) ----
    entry("en_self_corr_wait_001",
          "the file is README wait the file is CHANGELOG",
          [repair("the file is README", "wait",
                  "the file is CHANGELOG")]),
    entry("en_self_corr_wait_002",
          "the train arrives at noon wait the train arrives at one",
          [repair("the train arrives at noon", "wait",
                  "the train arrives at one")]),
    entry("en_self_corr_wait_003",
          "the answer is 42 wait the answer is 43",
          [repair("the answer is 42", "wait", "the answer is 43")]),

    # ---- Cue: `no` (3) — bare 'no' as correction marker ----
    # Requires shared content word between reparandum and repair
    # (count_shared_prefix_from_end >= 1).
    entry("en_self_corr_no_001",
          "I went to Madrid no I went to Barcelona",
          [repair("I went to Madrid", "no", "I went to Barcelona")]),
    entry("en_self_corr_no_002",
          "we use Python no we use Rust",
          [repair("we use Python", "no", "we use Rust")]),
    entry("en_self_corr_no_003",
          "the bug is in line ten no the bug is in line twelve",
          [repair("the bug is in line ten", "no",
                  "the bug is in line twelve")]),

    # ---- Clean controls (5): no self-correction ----
    entry("en_clean_001", "the API returns a JSON response over HTTP"),
    entry("en_clean_002", "configure the database connection in the environment file"),
    entry("en_clean_003", "deploy the new version after all tests pass"),
    entry("en_clean_004", "check the error log for the root cause of the failure"),
    entry("en_clean_005", "the algorithm runs in linear time on the input size"),

    # ---- Edge cases (2): negation that should NOT be flagged ----
    # 'no' as plain negation without a repair phrase that shares a
    # content word with the prior content. The detector requires
    # `count_shared_prefix_from_end >= 1` on the post-cue side.
    entry("en_edge_001", "no one came to the meeting because of the storm"),
    entry("en_edge_002", "this is not something we can discuss right now"),
]


# -----------------------------------------------------------------
# Codepoint offsetter — finds each repair's substrings in `text`
# (in order) and yields half-open Unicode-codepoint spans.
# -----------------------------------------------------------------

def find_span(text: str, needle: str, start_search: int) -> tuple[int, int]:
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
    repairs_out = []
    cursor = 0
    for r in e.repairs:
        rep_s, rep_e = find_span(e.text, r.reparandum, cursor)
        cursor = rep_e
        int_s, int_e = find_span(e.text, r.interregnum, cursor)
        cursor = int_e
        rpr_s, rpr_e = find_span(e.text, r.repair, cursor)
        cursor = rpr_e
        repairs_out.append({
            "reparandum": {"start": rep_s, "end": rep_e},
            "interregnum": {"start": int_s, "end": int_e},
            "repair": {"start": rpr_s, "end": rpr_e},
            "type": "substitution",
        })
    return {
        "utterance_id": e.utterance_id,
        "text": e.text,
        "repairs": repairs_out,
    }


def verify_entry(rendered: dict) -> List[str]:
    errors: List[str] = []
    text = rendered["text"]
    for i, r in enumerate(rendered["repairs"]):
        for kind in ("reparandum", "interregnum", "repair"):
            s, e = r[kind]["start"], r[kind]["end"]
            if s < 0 or e > len(text) or s >= e:
                errors.append(
                    f"{rendered['utterance_id']} repair[{i}].{kind} span "
                    f"({s}, {e}) out of bounds for text len {len(text)}"
                )
                continue
            slice_ = text[s:e]
            if not slice_.strip():
                errors.append(
                    f"{rendered['utterance_id']} repair[{i}].{kind} span "
                    f"({s}, {e}) is whitespace-only"
                )
    return errors


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSONL path. If omitted, prints to stdout.")
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
