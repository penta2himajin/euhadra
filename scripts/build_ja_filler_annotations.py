#!/usr/bin/env python3
"""
Build Japanese Tier 1 filler annotations.

Generates `tests/evaluation/annotations/ja_filler.jsonl` from a
structured Python source. Each entry is a tuple `(id, text,
filler_segments)` where `filler_segments` is the list of `、`-
delimited segments the script then locates inside `text` to
produce `FillerSpan` records (codepoint offsets of the *trimmed*
segment, matching `JapaneseFillerFilter::detect_spans`).

The eval target is `JapaneseFillerFilter` in `src/filter.rs`;
cue surface forms are taken from its `pure_fillers` /
`contextual_fillers` lists:

    pure:       えーと, えっと, えー, あー, うーん, うん, ああ, ええ
                + ASR misrecognition aliases 映像 / 映映
    contextual: あの, まあ, その, なんか, ほら, やっぱり

`detect_spans` strips contextual fillers in three contexts: at
sentence-initial position, immediately after a `。` boundary, or
when the segment IS the filler word in its entirety. The gold
annotations follow those rules.

Usage:
  scripts/build_ja_filler_annotations.py \\
    --out tests/evaluation/annotations/ja_filler.jsonl

  scripts/build_ja_filler_annotations.py --verify-only

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
# Annotation entries — covers pure / contextual / standalone /
# clean / edge-case mid-sentence demonstrative uses.
# -----------------------------------------------------------------

ENTRIES: List[Entry] = [
    # ---- Pure fillers (always removed) ----
    entry("ja_filler_pure_001", "えーと、明日の予定を確認しましょう", ["えーと"]),
    entry("ja_filler_pure_002", "うーん、難しい問題ですね", ["うーん"]),
    entry("ja_filler_pure_003", "あー、忘れていました", ["あー"]),
    entry("ja_filler_pure_004", "えっと、来週の会議は何時ですか", ["えっと"]),
    entry("ja_filler_pure_005", "うん、了解しました", ["うん"]),
    entry("ja_filler_pure_006", "ああ、なるほど", ["ああ"]),
    entry("ja_filler_pure_007", "ええ、その通りです", ["ええ"]),

    # ---- ASR misrecognition aliases (whisper conversions) ----
    entry("ja_filler_pure_008", "映像、会議室を予約しました", ["映像"]),
    entry("ja_filler_pure_009", "映映、ちょっと待ってください", ["映映"]),

    # ---- Contextual fillers — sentence-initial ----
    entry("ja_filler_ctx_001", "あの、お立ち合いの中に", ["あの"]),
    entry("ja_filler_ctx_002", "まあ、なんとかなるでしょう", ["まあ"]),
    entry("ja_filler_ctx_003", "なんか、変な動きしてる", ["なんか"]),
    entry("ja_filler_ctx_004", "その、つまり、こういうことです", ["その"]),
    entry("ja_filler_ctx_005", "ほら、思い出しましたか", ["ほら"]),
    entry("ja_filler_ctx_006", "やっぱり、最初の案で進めましょう", ["やっぱり"]),

    # ---- Contextual fillers — after 。 boundary ----
    entry("ja_filler_after_period_001",
          "確認しました。あの、結果は問題ないです",
          ["あの"]),
    entry("ja_filler_after_period_002",
          "テストが終わりました。まあ、もう一度走らせます",
          ["まあ"]),

    # ---- Standalone contextual filler segment ----
    # The whole 、-segment IS just the filler word, with no other
    # content — detect_spans's pass 3 catches these.
    entry("ja_filler_standalone_001",
          "拙者親方と申すは、あの、お立ち合いの中に",
          ["あの"]),

    # ---- Clean controls — no fillers ----
    entry("ja_filler_clean_001", "明日の朝十時に駅前で待ち合わせしましょう"),
    entry("ja_filler_clean_002", "新しいバージョンをリリースする前にテストを実行する"),
    entry("ja_filler_clean_003", "データベースの接続設定を環境変数で管理する"),
    entry("ja_filler_clean_004", "アルゴリズムは入力サイズに対して線形時間で動作する"),
    entry("ja_filler_clean_005", "会議は午後三時に開始する予定です"),

    # ---- Edge cases — contextual fillers used as demonstratives /
    # content words mid-sentence. detect_spans should NOT fire on
    # these because they're not standalone segments and not at
    # sentence-initial position.
    entry("ja_filler_edge_001", "拙者親方と申すは、あの人が来た", []),
    entry("ja_filler_edge_002", "なんか嫌な予感がする", []),
]


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
