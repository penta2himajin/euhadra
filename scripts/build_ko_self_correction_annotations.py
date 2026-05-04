#!/usr/bin/env python3
"""
Build Korean Tier 2 self-correction annotations.

Generates `tests/evaluation/annotations/ko_self_correction.jsonl`
from the structured Python source below. Each entry is a small
tuple `(id, text, reparandum, interregnum, repair)`; the script
string-searches the text to assign codepoint offsets `[start, end)`,
then emits the JSONL schema documented in
`tests/evaluation/annotations/guidelines.md`.

Mirrors the workflow used for the ja / es / en / zh annotation
sets. The Korean cue closed-set is taken from
`src/processor.rs::SelfCorrectionDetector::correction_cues_ko`:

    그게 아니라, 그게 아니고, 잘못 말했다, 잘못 말했네,
    아 잠깐, 잠깐만, 아니에요, 아니라, 아니야, 아니

`SelfCorrectionDetector::detect_korean` uses the Japanese-style
comma-segmentation strategy with eojeol-boundary cue checks: the
reparandum is the last `,`/`.`/`?`/`!`-segment before the cue, and
the cue must sit at an eojeol (whitespace) boundary so `아니`
doesn't fire inside `아니에요`-as-predicate.

For the FLEURS-ko / SenseVoice case (no internal commas in ASR
output), the whole pre-cue clause is a single segment → drop it
entirely → output is just the repair. Most entries below follow
that short-utterance pattern.

Usage:
  scripts/build_ko_self_correction_annotations.py
    --out tests/evaluation/annotations/ko_self_correction.jsonl

  scripts/build_ko_self_correction_annotations.py --verify-only

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
# Self-correction entries — covers the 10 cues from
# `correction_cues_ko`, plus 5 clean controls and 3 edge cases
# (negation predicate `아니에요` at sentence end, `아니라` as a
# nominal modifier, `아니` inside a longer Hangul morpheme).
# -----------------------------------------------------------------

ENTRIES: List[Entry] = [
    # ---- Cue: `그게 아니라` (5) ----
    entry("ko_self_corr_geuge_anira_001",
          "내가 갈게 그게 아니라 네가 갈게",
          [repair("내가 갈게", "그게 아니라", "네가 갈게")]),
    entry("ko_self_corr_geuge_anira_002",
          "오늘 회의 그게 아니라 내일 회의",
          [repair("오늘 회의", "그게 아니라", "내일 회의")]),
    entry("ko_self_corr_geuge_anira_003",
          "삼층에서 만나자 그게 아니라 사층에서 만나자",
          [repair("삼층에서 만나자", "그게 아니라",
                  "사층에서 만나자")]),
    entry("ko_self_corr_geuge_anira_004",
          "버그가 캐시에 있어 그게 아니라 큐에 있어",
          [repair("버그가 캐시에 있어", "그게 아니라",
                  "큐에 있어")]),
    entry("ko_self_corr_geuge_anira_005",
          "메인 브랜치에 머지 그게 아니라 개발 브랜치에 머지",
          [repair("메인 브랜치에 머지", "그게 아니라",
                  "개발 브랜치에 머지")]),

    # ---- Cue: `그게 아니고` (3) ----
    entry("ko_self_corr_geuge_anigo_001",
          "다섯 명이 와요 그게 아니고 일곱 명이 와요",
          [repair("다섯 명이 와요", "그게 아니고",
                  "일곱 명이 와요")]),
    entry("ko_self_corr_geuge_anigo_002",
          "월요일 출시 그게 아니고 화요일 출시",
          [repair("월요일 출시", "그게 아니고", "화요일 출시")]),
    entry("ko_self_corr_geuge_anigo_003",
          "그 사람은 의사야 그게 아니고 그 사람은 변호사야",
          [repair("그 사람은 의사야", "그게 아니고",
                  "그 사람은 변호사야")]),

    # ---- Cue: `잘못 말했다` (3) ----
    entry("ko_self_corr_jalmot_malhaetda_001",
          "빌드가 실패했다 잘못 말했다 빌드가 통과했다",
          [repair("빌드가 실패했다", "잘못 말했다",
                  "빌드가 통과했다")]),
    entry("ko_self_corr_jalmot_malhaetda_002",
          "오후 두 시 잘못 말했다 오후 세 시",
          [repair("오후 두 시", "잘못 말했다", "오후 세 시")]),
    entry("ko_self_corr_jalmot_malhaetda_003",
          "예산은 백만 원 잘못 말했다 예산은 이백만 원",
          [repair("예산은 백만 원", "잘못 말했다",
                  "예산은 이백만 원")]),

    # ---- Cue: `잘못 말했네` (2) ----
    entry("ko_self_corr_jalmot_malhaenne_001",
          "수요일이야 잘못 말했네 목요일이야",
          [repair("수요일이야", "잘못 말했네", "목요일이야")]),
    entry("ko_self_corr_jalmot_malhaenne_002",
          "한 시간 걸려 잘못 말했네 두 시간 걸려",
          [repair("한 시간 걸려", "잘못 말했네",
                  "두 시간 걸려")]),

    # ---- Cue: `잠깐만` (4) ----
    entry("ko_self_corr_jamkkanman_001",
          "회의실은 삼층 잠깐만 사층입니다",
          [repair("회의실은 삼층", "잠깐만", "사층입니다")]),
    entry("ko_self_corr_jamkkanman_002",
          "보고서는 김 씨에게 잠깐만 박 씨에게",
          [repair("보고서는 김 씨에게", "잠깐만",
                  "박 씨에게")]),
    entry("ko_self_corr_jamkkanman_003",
          "내일 만나요 잠깐만 모레 만나요",
          [repair("내일 만나요", "잠깐만", "모레 만나요")]),
    entry("ko_self_corr_jamkkanman_004",
          "다섯 시에 시작 잠깐만 여섯 시에 시작",
          [repair("다섯 시에 시작", "잠깐만",
                  "여섯 시에 시작")]),

    # ---- Cue: `아 잠깐` (2) ----
    entry("ko_self_corr_a_jamkkan_001",
          "왼쪽으로 가세요 아 잠깐 오른쪽으로 가세요",
          [repair("왼쪽으로 가세요", "아 잠깐",
                  "오른쪽으로 가세요")]),
    entry("ko_self_corr_a_jamkkan_002",
          "이메일 보냈어 아 잠깐 이메일 안 보냈어",
          [repair("이메일 보냈어", "아 잠깐",
                  "이메일 안 보냈어")]),

    # ---- Cue: `아니라` (3) ----
    entry("ko_self_corr_anira_001",
          "파일을 옮겨 주세요 아니라 복사해 주세요",
          [repair("파일을 옮겨 주세요", "아니라",
                  "복사해 주세요")]),
    entry("ko_self_corr_anira_002",
          "어제 출발 아니라 오늘 출발",
          [repair("어제 출발", "아니라", "오늘 출발")]),
    entry("ko_self_corr_anira_003",
          "그 영화는 재밌어 아니라 그 영화는 지루해",
          [repair("그 영화는 재밌어", "아니라",
                  "그 영화는 지루해")]),

    # ---- Cue: `아니야` (2) ----
    entry("ko_self_corr_aniya_001",
          "스무 명이 왔어 아니야 서른 명이 왔어",
          [repair("스무 명이 왔어", "아니야",
                  "서른 명이 왔어")]),
    entry("ko_self_corr_aniya_002",
          "내가 한 일이야 아니야 네가 한 일이야",
          [repair("내가 한 일이야", "아니야",
                  "네가 한 일이야")]),

    # ---- Cue: `아니` (4) — bare single-eojeol cue ----
    # Korean's most common short-form correction: `X 아니 Y`, where Y
    # replaces X entirely.
    entry("ko_self_corr_ani_001",
          "8시에 만나자 아니 9시에 만나자",
          [repair("8시에 만나자", "아니", "9시에 만나자")]),
    entry("ko_self_corr_ani_002",
          "서울에 갈게 아니 부산에 갈게",
          [repair("서울에 갈게", "아니", "부산에 갈게")]),
    entry("ko_self_corr_ani_003",
          "두 명이 와 아니 세 명이 와",
          [repair("두 명이 와", "아니", "세 명이 와")]),
    entry("ko_self_corr_ani_004",
          "오늘 갈게 아니 내일 갈게",
          [repair("오늘 갈게", "아니", "내일 갈게")]),

    # ---- Clean controls (5): no self-correction ----
    entry("ko_clean_001", "내일 오후 세 시에 회의를 시작하기로 했습니다"),
    entry("ko_clean_002", "데이터베이스 연결을 환경 변수로 설정해 주세요"),
    entry("ko_clean_003", "테스트가 모두 통과한 후에 배포를 진행합니다"),
    entry("ko_clean_004", "오류 로그에서 자세한 원인을 확인할 수 있습니다"),
    entry("ko_clean_005", "알고리즘은 입력 크기에 대해 선형 시간으로 동작합니다"),

    # ---- Edge cases (3) — should NOT be flagged ----
    # `아니에요` as sentence-final negation predicate (cue at end of
    # utterance, no repair clause follows → detect_korean skips
    # because after_cue is empty).
    entry("ko_edge_001", "이것은 제 가방이 아니에요"),
    # `아니라` as a nominal modifier "not just X but Y" — different
    # syntactic role, but the detector will still match the cue
    # surface form. We mark this as a clean entry (no repairs) so
    # the F1 evaluator can flag it as a false positive if the
    # detector fires.
    entry("ko_edge_002", "지식이 아니라 지혜가 필요합니다"),
    # `아니` inside a longer Hangul morpheme → the eojeol-boundary
    # check should prevent firing.
    entry("ko_edge_003", "그 분은 아니지만 비슷한 사람이에요"),
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
