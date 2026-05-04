#!/usr/bin/env python3
"""
Build Chinese Tier 1 filler annotations.

Generates `tests/evaluation/annotations/zh_filler.jsonl` from a
structured Python source. Each entry is a tuple `(id, text,
filler_segments)` where each segment is a `，`-delimited filler
string the script then locates inside `text` to produce
`FillerSpan` records (codepoint offsets of the *trimmed* segment,
matching `ChineseFillerFilter::detect_spans`).

The eval target is `ChineseFillerFilter` in `src/filter.rs`; cue
surface forms are taken from its `pure_fillers` /
`contextual_fillers` lists:

    pure:       嗯, 呃, 哦, 唉, 呀
    contextual: 那个, 这个, 就是, 然后, 怎么说

`detect_spans` strips contextual fillers in three contexts: at
sentence-initial position, immediately after a `。` boundary, or
when the segment IS the filler in its entirety.

Usage:
  scripts/build_zh_filler_annotations.py \\
    --out tests/evaluation/annotations/zh_filler.jsonl

  scripts/build_zh_filler_annotations.py --verify-only

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
    entry("zh_filler_pure_001", "嗯，我们需要更新数据库", ["嗯"]),
    entry("zh_filler_pure_002", "呃，明天发布新版本", ["呃"]),
    entry("zh_filler_pure_003", "哦，我懂了", ["哦"]),
    entry("zh_filler_pure_004", "唉，又出问题了", ["唉"]),
    entry("zh_filler_pure_005", "呀，差点忘了", ["呀"]),
    entry("zh_filler_pure_006", "嗯，我觉得这样可以", ["嗯"]),

    # ---- Contextual fillers — sentence-initial ----
    entry("zh_filler_ctx_001", "那个，测试全部通过了", ["那个"]),
    entry("zh_filler_ctx_002", "这个，我们今天部署到生产环境", ["这个"]),
    entry("zh_filler_ctx_003", "就是，先把日志收集好", ["就是"]),
    entry("zh_filler_ctx_004", "然后，我们需要更新数据库", ["然后"]),
    entry("zh_filler_ctx_005", "怎么说，加一个索引就行了", ["怎么说"]),

    # ---- Standalone contextual filler segment ----
    entry("zh_filler_standalone_001",
          "我觉得，那个，我们需要更新数据库",
          ["那个"]),
    entry("zh_filler_standalone_002",
          "测试通过了，然后，可以合并代码",
          ["然后"]),

    # ---- Contextual fillers — after 。 boundary ----
    # Detected only when the previous comma-segment ends with 。 —
    # i.e. text shape "X。，cue，Y" rather than the more natural
    # "X。cue，Y". The latter doesn't trigger because the filter
    # splits only on `，`, not on `。`.
    entry("zh_filler_after_period_001",
          "我们已经讨论过了。，那个，下次再说",
          ["那个"]),

    # ---- Clean controls — no fillers ----
    entry("zh_filler_clean_001", "我们需要更新数据库以支持新功能"),
    entry("zh_filler_clean_002", "明天发布新版本到生产环境"),
    entry("zh_filler_clean_003", "测试全部通过了，可以合并代码"),
    entry("zh_filler_clean_004", "请检查日志文件中的错误信息"),
    entry("zh_filler_clean_005", "配置文件已经修改并提交到仓库"),

    # ---- Edge cases — contextual fillers used as demonstratives /
    # content words mid-sentence. detect_spans should NOT fire on
    # these because they're not standalone segments and not at
    # sentence-initial position.
    entry("zh_filler_edge_001", "那个人是新来的同事", []),
    entry("zh_filler_edge_002", "这个项目下个月上线", []),
    entry("zh_filler_edge_003", "我们先试试这个方案然后再决定", []),
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
