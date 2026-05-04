#!/usr/bin/env python3
"""
Build Korean Tier 1 filler annotations.

Generates `tests/evaluation/annotations/ko_filler.jsonl` from a
structured Python source. Each entry is a tuple `(id, text,
filler_words)` where `filler_words` is the list of surface forms
the script then locates inside `text` to produce `FillerSpan`
records (codepoint offsets matching `SimpleFillerFilter::korean()`'s
`detect_spans` output).

The eval target is `SimpleFillerFilter::korean()` in
`src/filter.rs`; cue surface forms are taken from its
`pure_fillers` / `contextual_fillers` / `multi_fillers` lists:

    pure:       음, 어, 아, 으, 엄, 응
    contextual: 그, 저, 막, 약간, 뭐, 저기, 글쎄, 그러니까,
                그니까, 그래서, 어쨌든, 그게, 뭐랄까
    multi:      음 그러니까, 그게 그러니까, 뭐랄까 그

Korean uses whitespace tokenisation at the eojeol level, same
code path as English / Spanish (not the comma-segmentation used
by ja / zh). `detect_spans` only emits contextual fillers at
sentence-initial positions.

Usage:
  scripts/build_ko_filler_annotations.py \\
    --out tests/evaluation/annotations/ko_filler.jsonl

  scripts/build_ko_filler_annotations.py --verify-only

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


ENTRIES: List[Entry] = [
    # ---- Pure fillers (always removed) ----
    entry("ko_filler_pure_001", "음 회의 시간을 변경합시다", ["음"]),
    entry("ko_filler_pure_002", "어 로그를 확인해 주세요", ["어"]),
    entry("ko_filler_pure_003", "아 서버를 다시 시작하겠습니다", ["아"]),
    entry("ko_filler_pure_004", "으 이건 맞는 것 같습니다", ["으"]),
    entry("ko_filler_pure_005", "엄 패치를 검토해 주세요", ["엄"]),
    entry("ko_filler_pure_006", "응 알겠습니다", ["응"]),
    entry("ko_filler_pure_007", "빌드가 어 진행 중입니다", ["어"]),
    entry("ko_filler_pure_008", "API가 음 JSON을 반환합니다", ["음"]),

    # ---- Contextual fillers — sentence-initial only ----
    entry("ko_filler_ctx_001", "그러니까 다음 주에 만나자", ["그러니까"]),
    entry("ko_filler_ctx_002", "그니까 회의를 시작하자", ["그니까"]),
    entry("ko_filler_ctx_003", "저기 잠깐 시간 있어요", ["저기"]),
    entry("ko_filler_ctx_004", "그래서 어떻게 됐어요", ["그래서"]),
    entry("ko_filler_ctx_005", "막 시작했어요", ["막"]),
    entry("ko_filler_ctx_006", "약간 어려운 부분이 있어요", ["약간"]),
    entry("ko_filler_ctx_007", "어쨌든 결과는 좋았습니다", ["어쨌든"]),
    entry("ko_filler_ctx_008", "글쎄 잘 모르겠어요", ["글쎄"]),
    entry("ko_filler_ctx_009", "뭐 그렇게 보면 맞아요", ["뭐"]),
    entry("ko_filler_ctx_010", "뭐랄까 좀 애매해요", ["뭐랄까"]),

    # ---- Multi-word fillers ----
    entry("ko_filler_multi_001", "음 그러니까 회의를 시작하자",
          ["음 그러니까"]),
    entry("ko_filler_multi_002", "그게 그러니까 다음 주에 합시다",
          ["그게 그러니까"]),
    entry("ko_filler_multi_003", "뭐랄까 그 모호한 점이 있어요",
          ["뭐랄까 그"]),

    # ---- Clean controls — no fillers ----
    entry("ko_filler_clean_001",
          "내일 오후 세 시에 회의를 시작하기로 했습니다"),
    entry("ko_filler_clean_002",
          "데이터베이스 연결을 환경 변수로 설정해 주세요"),
    entry("ko_filler_clean_003",
          "테스트가 모두 통과한 후에 배포를 진행합니다"),
    entry("ko_filler_clean_004",
          "오류 로그에서 자세한 원인을 확인할 수 있습니다"),
    entry("ko_filler_clean_005",
          "알고리즘은 입력 크기에 대해 선형 시간으로 동작합니다"),

    # ---- Edge cases — contextual fillers used as content words.
    # `그` is a demonstrative ("that"), `저` is a demonstrative or
    # personal pronoun, `뭐` is "what". Mid-sentence they're content,
    # not fillers, so detect_spans should NOT fire.
    entry("ko_filler_edge_001", "나는 그 사람을 만났어요", []),
    entry("ko_filler_edge_002", "그 책은 정말 재미있어요", []),
    entry("ko_filler_edge_003", "이게 뭐 하는 거예요", []),
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
