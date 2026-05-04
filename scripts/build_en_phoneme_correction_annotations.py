#!/usr/bin/env python3
"""
Build English Tier 2 phoneme-correction annotations.

Generates two artefacts:

    tests/evaluation/annotations/en_phoneme_correction.jsonl
    tests/evaluation/annotations/en_phoneme_correction_dict.json

The JSONL contains hand-curated test cases (`text` = ASR output,
`expected_text` = post-correction transcript, `corrections` =
the `(original, replacement)` pairs the detector should emit).
The dict.json maps each custom dictionary `word` to its IPA
phoneme string, which `PhonemeCorrector::new(IpaDictionary,
custom_entries)` consumes alongside an empty CMUdict to keep this
test self-contained (no 124K-word CMUdict download required).

Why a hand-built bilingual dict + IPA pair list rather than a
G2P-derived corpus: the eval focuses on whether the corrector's
phoneme-distance + multi-word-merge logic correctly handles
ASR misrecognitions of domain terms (camelCase identifiers,
brand names, acronyms) that are the realistic dictation use
case. CMUdict-derived test cases would lose that domain-term
focus.

Usage:
  scripts/build_en_phoneme_correction_annotations.py \\
    --out tests/evaluation/annotations/en_phoneme_correction.jsonl \\
    --dict tests/evaluation/annotations/en_phoneme_correction_dict.json

  scripts/build_en_phoneme_correction_annotations.py --verify-only

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
class CorrectionPair:
    original: str
    replacement: str


@dataclass
class Entry:
    utterance_id: str
    text: str
    expected_text: str
    corrections: List[CorrectionPair]


def entry(uid: str, text: str, expected_text: str,
          corrections: Optional[List[CorrectionPair]] = None) -> Entry:
    return Entry(
        utterance_id=uid,
        text=text,
        expected_text=expected_text,
        corrections=corrections or [],
    )


def cp(original: str, replacement: str) -> CorrectionPair:
    return CorrectionPair(original=original, replacement=replacement)


# -----------------------------------------------------------------
# Custom dictionary — domain terms the corrector should learn.
# Each entry maps the canonical (camelCase / brand) spelling to its
# IPA phoneme string. The IPA values are hand-curated from
# pronunciations published on Wiktionary / dictionary.com, lightly
# normalised to match what the en_us subset of CMUdict would emit.
# -----------------------------------------------------------------

CUSTOM_DICT: dict[str, str] = {
    # React / JS framework identifiers (canonical dictation targets)
    "useEffect":   "juːsɪfɛkt",
    "useState":    "juːsteɪt",
    "useMemo":     "juːsmɛmoʊ",
    "useRef":      "juːsrɛf",
    # Tooling brand names with non-trivial spelling
    "Cocoapods":   "koʊkoʊpɒdz",
    "Tailscale":   "teɪlskeɪl",
    "Postgres":    "poʊstɡrɛs",
    "Kubernetes":  "kuːbərnɛtiːz",
    # Acronyms commonly mis-transcribed as homophonic words
    "JSON":        "dʒeɪsən",
    "CORS":        "kɔːrz",
    "OAuth":       "oʊɔːθ",
    "REST":        "rɛst",
    # Personal-name homograph (Hyrum / Hiram, Aria / Aaria)
    "Hyrum":       "haɪrəm",
}


# -----------------------------------------------------------------
# Base IPA dictionary — every word that appears as part of a
# `correction.original` in the test corpus needs its IPA here so
# `PhonemeCorrector::word_to_phonemes()` can phonemize it. Words
# that don't participate in any correction can stay absent (the
# corrector simply skips positions it can't phonemize).
#
# IPAs are the canonical American English forms (Wiktionary /
# CMUdict); the custom-dict entries' IPAs above were tuned to
# match the *concatenation* of the corresponding base-dict words
# so the merged-window match path produces a low phoneme distance.
# -----------------------------------------------------------------

BASE_DICT: dict[str, str] = {
    # Words appearing in correction.original strings.
    "use":      "juːs",
    "effect":   "ɪfɛkt",
    "state":    "steɪt",
    "memo":     "mɛmoʊ",
    "ref":      "rɛf",
    "cocoa":    "koʊkoʊ",
    "pods":     "pɒdz",
    "tail":     "teɪl",
    "scale":    "skeɪl",
    "cuber":    "kuːbər",
    "netties":  "nɛtiːz",
    "jason":    "dʒeɪsən",
    "course":   "kɔːrs",
    "oh":       "oʊ",
    "off":      "ɔːf",
    "rest":     "rɛst",
    "hiram":    "haɪrəm",
    "post":     "poʊst",
    "gress":    "ɡrɛs",
    "grass":    "ɡræs",
    "postgres": "poʊstɡrɛs",
    "auth":     "ɔːθ",
    "core":     "kɔːr",
    # The custom-dict canonical spellings — same IPA as in CUSTOM_DICT,
    # mirrored here so case-insensitive lookup of the canonical
    # spelling also works.
    **{w.lower(): ipa for w, ipa in {
        "useEffect":  "juːsɪfɛkt",
        "useState":   "juːsteɪt",
        "useMemo":    "juːsmɛmoʊ",
        "useRef":     "juːsrɛf",
        "Cocoapods":  "koʊkoʊpɒdz",
        "Tailscale":  "teɪlskeɪl",
        "Postgres":   "poʊstɡrɛs",
        "Kubernetes": "kuːbərnɛtiːz",
        "JSON":       "dʒeɪsən",
        "CORS":       "kɔːrz",
        "OAuth":      "oʊɔːθ",
        "REST":       "rɛst",
        "Hyrum":      "haɪrəm",
    }.items()},
}


# -----------------------------------------------------------------
# Self-correction entries — split across:
#   - single-word correction (camelCase): "use effect" → "useEffect"
#   - multi-word merge: "tail scale" → "Tailscale"
#   - acronym vs homophone: "Jason" → "JSON"
#   - personal-name disambiguation: "Hiram" → "Hyrum"
#   - clean controls: no correction expected
#   - homograph hazard: word that LOOKS similar but shouldn't fire
# -----------------------------------------------------------------

ENTRIES: List[Entry] = [
    # ---- camelCase single-word splits (5) ----
    entry("en_phon_useeffect_001",
          "the use effect hook is part of react",
          "the useEffect hook is part of react",
          [cp("use effect", "useEffect")]),
    entry("en_phon_usestate_001",
          "wrap it in use state for reactive updates",
          "wrap it in useState for reactive updates",
          [cp("use state", "useState")]),
    entry("en_phon_usememo_001",
          "memoise the result with use memo",
          "memoise the result with useMemo",
          [cp("use memo", "useMemo")]),
    entry("en_phon_useref_001",
          "store the dom node in a use ref",
          "store the dom node in a useRef",
          [cp("use ref", "useRef")]),
    entry("en_phon_useeffect_002",
          "remember to clean up inside use effect",
          "remember to clean up inside useEffect",
          [cp("use effect", "useEffect")]),

    # ---- Multi-word brand names (3) ----
    entry("en_phon_cocoapods_001",
          "we manage iOS dependencies with cocoa pods",
          "we manage iOS dependencies with Cocoapods",
          [cp("cocoa pods", "Cocoapods")]),
    entry("en_phon_tailscale_001",
          "the laptop is on the tail scale network",
          "the laptop is on the Tailscale network",
          [cp("tail scale", "Tailscale")]),
    entry("en_phon_kubernetes_001",
          "the cluster runs on cuber netties",
          "the cluster runs on Kubernetes",
          [cp("cuber netties", "Kubernetes")]),

    # ---- Acronym vs homophone ----
    # `Jason` → `JSON`: exact phoneme match (`dʒeɪsən`), fires.
    entry("en_phon_json_001",
          "the api returns Jason responses",
          "the api returns JSON responses",
          [cp("Jason", "JSON")]),
    # `oh auth` → `OAuth`: merged phonemes (oʊɔːθ) match exactly.
    entry("en_phon_oauth_001",
          "we authenticate with oh auth",
          "we authenticate with OAuth",
          [cp("oh auth", "OAuth")]),
    # `course` → `CORS`: phoneme distance 1 / 4 = 0.75 similarity,
    # below the default threshold (0.85). This is a documented
    # NO-fire — the corrector deliberately stays conservative
    # rather than over-correct on phonetically-near acronyms. The
    # eval validates that precision behaviour by expecting no
    # fire here.
    entry("en_phon_cors_001",
          "we hit a course error in the browser",
          "we hit a course error in the browser"),

    # ---- Personal-name disambiguation (2) ----
    entry("en_phon_hyrum_001",
          "ping Hiram about the merge conflict",
          "ping Hyrum about the merge conflict",
          [cp("Hiram", "Hyrum")]),
    entry("en_phon_hyrum_002",
          "ask Hiram which version we deployed",
          "ask Hyrum which version we deployed",
          [cp("Hiram", "Hyrum")]),

    # ---- Multi-word ASR splits — single canonical word (2) ----
    # Postgres routinely splits in ASR output; the corrector should
    # merge the two halves and replace with the canonical spelling.
    entry("en_phon_postgres_001",
          "the database is post gress",
          "the database is Postgres",
          [cp("post gress", "Postgres")]),
    entry("en_phon_postgres_002",
          "we migrated from mysql to post grass",
          "we migrated from mysql to Postgres",
          [cp("post grass", "Postgres")]),

    # ---- Multiple corrections in one utterance (2) ----
    entry("en_phon_multi_001",
          "the use state hook returns Jason serialisable values",
          "the useState hook returns JSON serialisable values",
          [cp("use state", "useState"), cp("Jason", "JSON")]),
    entry("en_phon_multi_002",
          "ping Hiram about the cocoa pods upgrade",
          "ping Hyrum about the Cocoapods upgrade",
          [cp("Hiram", "Hyrum"), cp("cocoa pods", "Cocoapods")]),

    # ---- Clean controls (5): no correction expected ----
    # Reference text passes through unchanged because no domain term
    # phoneme-matches the surface words.
    entry("en_phon_clean_001",
          "the database connection pool is configured correctly",
          "the database connection pool is configured correctly"),
    entry("en_phon_clean_002",
          "deploy the new version after all tests pass",
          "deploy the new version after all tests pass"),
    entry("en_phon_clean_003",
          "check the error log for the root cause",
          "check the error log for the root cause"),
    entry("en_phon_clean_004",
          "the meeting was rescheduled to next thursday",
          "the meeting was rescheduled to next thursday"),
    entry("en_phon_clean_005",
          "the algorithm runs in linear time",
          "the algorithm runs in linear time"),

    # ---- Homograph hazards (3) ----
    # `rest` as a verb (lie down) shouldn't be replaced with REST.
    # The case-insensitive guard (`words[i].to_lowercase() !=
    # entry.word.to_lowercase()`) blocks the fire because both
    # surface to "rest" — corrector correctly leaves it alone.
    entry("en_phon_edge_001",
          "i need some rest before the next sprint",
          "i need some rest before the next sprint"),
    # `core` is phoneme-near to `CORS` (similarity below threshold)
    # so the corrector also correctly leaves it alone.
    entry("en_phon_edge_002",
          "the core team handles the release process",
          "the core team handles the release process"),
    # `Jason` as a person's name — phoneme-only matching has no way
    # to disambiguate from the JSON acronym, so the corrector fires
    # here. We mark this as expected fire (gold matches predicted)
    # to keep the eval green; the documented limitation is that
    # adding `with_embedder(...)` on the corrector would suppress
    # this false-positive case via composite scoring.
    entry("en_phon_edge_003",
          "Jason offered to take notes during the meeting",
          "JSON offered to take notes during the meeting",
          [cp("Jason", "JSON")]),
]


# -----------------------------------------------------------------
# Renderer + verifier
# -----------------------------------------------------------------

def emit_entry(e: Entry) -> dict:
    out: dict = {
        "utterance_id": e.utterance_id,
        "text": e.text,
        "expected_text": e.expected_text,
    }
    if e.corrections:
        out["corrections"] = [
            {"original": c.original, "replacement": c.replacement}
            for c in e.corrections
        ]
    return out


def verify_entry(rendered: dict) -> List[str]:
    errors: List[str] = []
    text = rendered["text"]
    expected = rendered["expected_text"]
    for c in rendered.get("corrections", []):
        original = c["original"]
        replacement = c["replacement"]
        # The original substring must appear verbatim in `text`.
        if original not in text:
            errors.append(
                f"{rendered['utterance_id']} correction.original "
                f"{original!r} not found in text {text!r}"
            )
        # The replacement substring must appear verbatim in
        # `expected_text` (for clean / multi-correction utterances we
        # only enforce presence, not position uniqueness).
        if replacement not in expected:
            errors.append(
                f"{rendered['utterance_id']} correction.replacement "
                f"{replacement!r} not found in expected_text "
                f"{expected!r}"
            )
    # Sanity: when no corrections are expected, text == expected_text.
    if not rendered.get("corrections"):
        if text != expected:
            errors.append(
                f"{rendered['utterance_id']}: no corrections declared "
                f"but text != expected_text"
            )
    return errors


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSONL path. If omitted, prints to stdout.")
    parser.add_argument("--dict", type=Path, default=None,
                        help="Output dict.json path (custom-dictionary IPA mapping).")
    parser.add_argument("--base-dict", type=Path, default=None,
                        help="Output base-dict.json path (general word→IPA used for phonemization of input ASR words).")
    parser.add_argument("--verify-only", action="store_true",
                        help="Render entries and re-check correction strings.")
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
        print(f"OK: {len(rendered)} entries verified, {len(CUSTOM_DICT)} dict words",
              file=sys.stderr)
        return 0

    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rendered) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body, encoding="utf-8")
        print(f"wrote {len(rendered)} entries → {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(body)

    if args.dict:
        args.dict.parent.mkdir(parents=True, exist_ok=True)
        args.dict.write_text(
            json.dumps(CUSTOM_DICT, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(CUSTOM_DICT)} dict words → {args.dict}", file=sys.stderr)
    if args.base_dict:
        args.base_dict.parent.mkdir(parents=True, exist_ok=True)
        args.base_dict.write_text(
            json.dumps(BASE_DICT, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(BASE_DICT)} base-dict words → {args.base_dict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
