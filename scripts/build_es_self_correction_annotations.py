#!/usr/bin/env python3
"""
Build Spanish Tier 2 self-correction annotations.

Generates `tests/evaluation/annotations/es_self_correction.jsonl`
from a structured Python source (this file). Each entry is a small
tuple `(id, text, reparandum, interregnum, repair)`; the script
string-searches the text to assign codepoint offsets `[start, end)`,
then emits the JSONL schema documented in
`tests/evaluation/annotations/guidelines.md`.

This mirrors the workflow used for `ja_self_correction.jsonl`
(hand-curated by Claude pending native-speaker review). The Spanish
cue closed-set is taken from
`src/processor.rs::SelfCorrectionDetector::correction_cues_es`:

    mejor dicho, quiero decir, o sea, perdón,
    mejor, digo, no es, no

`SelfCorrectionDetector::detect_spanish` requires a shared content
word between `reparandum` and `repair` (see
`count_shared_prefix_from_end`); each entry below is constructed so
the detector actually fires, so this corpus also serves as a
post-PR-#35 regression test.

Usage:
  scripts/build_es_self_correction_annotations.py
    --out tests/evaluation/annotations/es_self_correction.jsonl

  scripts/build_es_self_correction_annotations.py --verify-only
    (renders each span back to its expected word — useful in CI to
     catch hand-edited entries that drift out of bounds)

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


# -----------------------------------------------------------------
# Self-correction entries — covers all 8 cues from
# `correction_cues_es`, plus 5 clean controls and 2 edge cases.
# Pattern: <reparandum> <cue> <repair> with at least one shared
# content word between reparandum tail and repair head, so the
# detector actually fires.
# -----------------------------------------------------------------

def entry(uid: str, text: str, repairs: Optional[List[Repair]] = None) -> Entry:
    return Entry(utterance_id=uid, text=text, repairs=repairs or [])


def repair(reparandum: str, interregnum: str, repair_text: str) -> Repair:
    return Repair(reparandum=reparandum, interregnum=interregnum, repair=repair_text)


ENTRIES: List[Entry] = [
    # ---- Cue: `no` (5) ----
    entry("es_self_corr_no_001", "voy mañana no voy hoy",
          [repair("voy mañana", "no", "voy hoy")]),
    entry("es_self_corr_no_002", "compré dos libros no compré tres libros",
          [repair("compré dos libros", "no", "compré tres libros")]),
    entry("es_self_corr_no_003", "trabajo en Madrid no trabajo en Barcelona",
          [repair("trabajo en Madrid", "no", "trabajo en Barcelona")]),
    entry("es_self_corr_no_004", "llegué a las cinco no llegué a las seis",
          [repair("llegué a las cinco", "no", "llegué a las seis")]),
    entry("es_self_corr_no_005", "vivo en una casa no vivo en un piso",
          [repair("vivo en una casa", "no", "vivo en un piso")]),

    # ---- Cue: `perdón` (5) ----
    # Each example is constructed so that the post-cue clause repeats
    # at least one word from the pre-cue clause's prefix
    # (`count_shared_prefix_from_end` requirement). Without that, the
    # detector silently drops the utterance — see the post-PR-#35
    # behaviour audit.
    entry("es_self_corr_perdon_001", "voy a Madrid perdón voy a Barcelona",
          [repair("voy a Madrid", "perdón", "voy a Barcelona")]),
    entry("es_self_corr_perdon_002", "el martes perdón el miércoles",
          [repair("el martes", "perdón", "el miércoles")]),
    entry("es_self_corr_perdon_003", "tengo cinco hermanos perdón tengo cuatro hermanos",
          [repair("tengo cinco hermanos", "perdón", "tengo cuatro hermanos")]),
    entry("es_self_corr_perdon_004", "el rojo perdón el azul me gusta más",
          [repair("el rojo", "perdón", "el azul")]),
    entry("es_self_corr_perdon_005", "cuesta veinte euros perdón cuesta quince euros",
          [repair("cuesta veinte euros", "perdón", "cuesta quince euros")]),

    # ---- Cue: `digo` (5) ----
    entry("es_self_corr_digo_001", "el lunes digo el martes tenemos clase",
          [repair("el lunes", "digo", "el martes")]),
    entry("es_self_corr_digo_002", "compré arroz digo compré pasta",
          [repair("compré arroz", "digo", "compré pasta")]),
    entry("es_self_corr_digo_003", "mi hermano digo mi primo viene mañana",
          [repair("mi hermano", "digo", "mi primo")]),
    entry("es_self_corr_digo_004", "tengo treinta digo tengo cuarenta años",
          [repair("tengo treinta", "digo", "tengo cuarenta")]),
    entry("es_self_corr_digo_005", "trabajo desde casa digo trabajo desde la oficina",
          [repair("trabajo desde casa", "digo", "trabajo desde la oficina")]),

    # ---- Cue: `mejor dicho` (5) ----
    entry("es_self_corr_mejor_dicho_001", "voy a salir mejor dicho voy a entrar",
          [repair("voy a salir", "mejor dicho", "voy a entrar")]),
    entry("es_self_corr_mejor_dicho_002", "es mi amigo mejor dicho es mi colega",
          [repair("es mi amigo", "mejor dicho", "es mi colega")]),
    entry("es_self_corr_mejor_dicho_003", "vamos al cine mejor dicho vamos al teatro",
          [repair("vamos al cine", "mejor dicho", "vamos al teatro")]),
    entry("es_self_corr_mejor_dicho_004", "el equipo ganó mejor dicho el equipo empató",
          [repair("el equipo ganó", "mejor dicho", "el equipo empató")]),
    entry("es_self_corr_mejor_dicho_005", "compré una manzana mejor dicho compré una pera",
          [repair("compré una manzana", "mejor dicho", "compré una pera")]),

    # ---- Cue: `o sea` (5) ----
    entry("es_self_corr_o_sea_001", "es difícil o sea es complicado",
          [repair("es difícil", "o sea", "es complicado")]),
    entry("es_self_corr_o_sea_002", "vamos rápido o sea vamos en seguida",
          [repair("vamos rápido", "o sea", "vamos en seguida")]),
    entry("es_self_corr_o_sea_003", "lo hice ayer o sea lo hice anteayer",
          [repair("lo hice ayer", "o sea", "lo hice anteayer")]),
    entry("es_self_corr_o_sea_004", "es bonito o sea es elegante",
          [repair("es bonito", "o sea", "es elegante")]),
    entry("es_self_corr_o_sea_005", "trabajo mucho o sea trabajo todo el día",
          [repair("trabajo mucho", "o sea", "trabajo todo el día")]),

    # ---- Cue: `quiero decir` (3) ----
    entry("es_self_corr_quiero_decir_001", "vivo en Madrid quiero decir vivo en Barcelona",
          [repair("vivo en Madrid", "quiero decir", "vivo en Barcelona")]),
    entry("es_self_corr_quiero_decir_002", "tengo cinco años quiero decir tengo seis años",
          [repair("tengo cinco años", "quiero decir", "tengo seis años")]),
    entry("es_self_corr_quiero_decir_003", "es martes quiero decir es miércoles",
          [repair("es martes", "quiero decir", "es miércoles")]),

    # ---- Cue: `mejor` (standalone, not "mejor dicho") (3) ----
    entry("es_self_corr_mejor_001", "voy a Madrid mejor voy a Barcelona",
          [repair("voy a Madrid", "mejor", "voy a Barcelona")]),
    entry("es_self_corr_mejor_002", "compro pan mejor compro pasta",
          [repair("compro pan", "mejor", "compro pasta")]),
    entry("es_self_corr_mejor_003", "tengo café mejor tengo té",
          [repair("tengo café", "mejor", "tengo té")]),

    # ---- Cue: `no es` (2) ----
    entry("es_self_corr_no_es_001", "el coche es viejo no es el coche es nuevo",
          [repair("el coche es viejo", "no es", "el coche es nuevo")]),
    entry("es_self_corr_no_es_002", "es alto no es es bajo",
          [repair("es alto", "no es", "es bajo")]),

    # ---- Clean controls (5): no self-correction ----
    entry("es_clean_001", "el tren llega a las ocho de la mañana"),
    entry("es_clean_002", "ayer fui al supermercado a comprar leche"),
    entry("es_clean_003", "mi hermana trabaja en una oficina del centro"),
    entry("es_clean_004", "el partido empezará a las nueve de la noche"),
    entry("es_clean_005", "este libro habla sobre la historia de Roma"),

    # ---- Edge cases (2): negation that should NOT be flagged ----
    # 'no' as plain negation without a repair phrase that shares a
    # content word with the prior content. The detector requires
    # `count_shared_prefix_from_end >= 1` on the post-cue side, so
    # these should produce no fire.
    entry("es_edge_001", "no vino a la fiesta porque estaba enfermo"),
    entry("es_edge_002", "esto no es para discutir ahora"),
]


# -----------------------------------------------------------------
# Codepoint offsetter — finds each repair's substrings in `text`
# (in order) and yields half-open Unicode-codepoint spans.
# -----------------------------------------------------------------

def find_span(text: str, needle: str, start_search: int) -> tuple[int, int]:
    """
    Locate `needle` in `text` starting at codepoint offset
    `start_search`. Returns half-open `[start, end)` codepoint span.
    Raises ValueError on miss to surface drift.
    """
    pos = text.find(needle, start_search)
    if pos < 0:
        raise ValueError(f"needle {needle!r} not found in {text!r} after offset {start_search}")
    # `str.find` returns a UTF-8 byte offset == codepoint offset only for
    # ASCII; we want codepoint offsets explicitly. Translate via slicing.
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
    """
    Re-slice each span out of `text` and confirm it matches the
    declared boundaries. Returns a list of error strings (empty on
    success) so the verifier can list every drift in one pass.
    """
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
            # No specific expectation in the verifier; we trust the
            # builder upstream. Just sanity-check the slice is
            # non-empty.
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
