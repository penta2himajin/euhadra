#!/usr/bin/env python3
"""
Build Chinese Tier 2 self-correction annotations.

Generates `tests/evaluation/annotations/zh_self_correction.jsonl`
from the structured Python source below. Each entry is a small
tuple `(id, text, reparandum, interregnum, repair)`; the script
string-searches the text to assign codepoint offsets `[start, end)`,
then emits the JSONL schema documented in
`tests/evaluation/annotations/guidelines.md`.

Mirrors the workflow used for the ja / es / en annotation sets.
The Chinese cue closed-set is taken from
`src/processor.rs::SelfCorrectionDetector::correction_cues_zh`:

    我的意思是, 确切地说, 应该说, 我是说, 不对, 不是, 算了

`SelfCorrectionDetector::detect_chinese` mirrors the Japanese
comma-segmentation pattern: the reparandum is the last
clause-comma segment (、 / ，) before the cue. Each entry below is
constructed with a leading `，` on the reparandum side so the
detector actually fires.

Usage:
  scripts/build_zh_self_correction_annotations.py
    --out tests/evaluation/annotations/zh_self_correction.jsonl

  scripts/build_zh_self_correction_annotations.py --verify-only

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
# Self-correction entries — covers all 7 cues from
# `correction_cues_zh`, plus 5 clean controls and 2 edge cases.
# Pattern mirrors the ja annotations: clause-comma-delimited
# segments (X，cue，Y) where X is the reparandum and Y is the repair.
# -----------------------------------------------------------------

ENTRIES: List[Entry] = [
    # ---- Cue: `不对` (5) ----
    entry("zh_self_corr_bu_dui_001",
          "我们明天开会，不对，后天开会",
          [repair("我们明天开会", "不对", "后天开会")]),
    entry("zh_self_corr_bu_dui_002",
          "会议室在三楼，不对，在四楼",
          [repair("会议室在三楼", "不对", "在四楼")]),
    entry("zh_self_corr_bu_dui_003",
          "我去上海，不对，我去北京",
          [repair("我去上海", "不对", "我去北京")]),
    entry("zh_self_corr_bu_dui_004",
          "项目下个月发布，不对，下下个月发布",
          [repair("项目下个月发布", "不对", "下下个月发布")]),
    entry("zh_self_corr_bu_dui_005",
          "需要五个人，不对，需要七个人",
          [repair("需要五个人", "不对", "需要七个人")]),

    # ---- Cue: `不是` (5) ----
    entry("zh_self_corr_bu_shi_001",
          "他是医生，不是，他是工程师",
          [repair("他是医生", "不是", "他是工程师")]),
    entry("zh_self_corr_bu_shi_002",
          "我去西安，不是，我去成都",
          [repair("我去西安", "不是", "我去成都")]),
    entry("zh_self_corr_bu_shi_003",
          "答案是十二，不是，答案是十三",
          [repair("答案是十二", "不是", "答案是十三")]),
    entry("zh_self_corr_bu_shi_004",
          "用 Python 实现，不是，用 Rust 实现",
          [repair("用 Python 实现", "不是", "用 Rust 实现")]),
    entry("zh_self_corr_bu_shi_005",
          "下午三点开始，不是，下午四点开始",
          [repair("下午三点开始", "不是", "下午四点开始")]),

    # ---- Cue: `我是说` (5) ----
    entry("zh_self_corr_wo_shi_shuo_001",
          "我们星期二开会，我是说，我们星期三开会",
          [repair("我们星期二开会", "我是说", "我们星期三开会")]),
    entry("zh_self_corr_wo_shi_shuo_002",
          "费用是一千元，我是说，费用是一千二百元",
          [repair("费用是一千元", "我是说", "费用是一千二百元")]),
    entry("zh_self_corr_wo_shi_shuo_003",
          "请把报告发给小李，我是说，请把报告发给小王",
          [repair("请把报告发给小李", "我是说", "请把报告发给小王")]),
    entry("zh_self_corr_wo_shi_shuo_004",
          "他在北京工作，我是说，他在上海工作",
          [repair("他在北京工作", "我是说", "他在上海工作")]),
    entry("zh_self_corr_wo_shi_shuo_005",
          "会在四楼开，我是说，会在五楼开",
          [repair("会在四楼开", "我是说", "会在五楼开")]),

    # ---- Cue: `我的意思是` (4) ----
    entry("zh_self_corr_wo_de_yi_si_shi_001",
          "测试通过了，我的意思是，测试失败了",
          [repair("测试通过了", "我的意思是", "测试失败了")]),
    entry("zh_self_corr_wo_de_yi_si_shi_002",
          "他下个月来，我的意思是，他下下个月来",
          [repair("他下个月来", "我的意思是", "他下下个月来")]),
    entry("zh_self_corr_wo_de_yi_si_shi_003",
          "数据库出问题，我的意思是，缓存出问题",
          [repair("数据库出问题", "我的意思是", "缓存出问题")]),
    entry("zh_self_corr_wo_de_yi_si_shi_004",
          "需要十台服务器，我的意思是，需要十二台服务器",
          [repair("需要十台服务器", "我的意思是",
                  "需要十二台服务器")]),

    # ---- Cue: `应该说` (4) ----
    entry("zh_self_corr_ying_gai_shuo_001",
          "直接推送，应该说，用合并请求",
          [repair("直接推送", "应该说", "用合并请求")]),
    entry("zh_self_corr_ying_gai_shuo_002",
          "他是经理，应该说，他是总监",
          [repair("他是经理", "应该说", "他是总监")]),
    entry("zh_self_corr_ying_gai_shuo_003",
          "结果不太好，应该说，结果非常糟糕",
          [repair("结果不太好", "应该说", "结果非常糟糕")]),
    entry("zh_self_corr_ying_gai_shuo_004",
          "项目已经完成，应该说，项目接近完成",
          [repair("项目已经完成", "应该说", "项目接近完成")]),

    # ---- Cue: `确切地说` (3) ----
    entry("zh_self_corr_que_qie_de_shuo_001",
          "今天很热，确切地说，今天三十五度",
          [repair("今天很热", "确切地说", "今天三十五度")]),
    entry("zh_self_corr_que_qie_de_shuo_002",
          "需要一段时间，确切地说，需要两个星期",
          [repair("需要一段时间", "确切地说",
                  "需要两个星期")]),
    entry("zh_self_corr_que_qie_de_shuo_003",
          "不少人参加，确切地说，超过一百人参加",
          [repair("不少人参加", "确切地说",
                  "超过一百人参加")]),

    # ---- Cue: `算了` (3) ----
    entry("zh_self_corr_suan_le_001",
          "我们去看电影，算了，我们去吃饭",
          [repair("我们去看电影", "算了", "我们去吃饭")]),
    entry("zh_self_corr_suan_le_002",
          "先重启数据库，算了，先重启服务",
          [repair("先重启数据库", "算了", "先重启服务")]),
    entry("zh_self_corr_suan_le_003",
          "他来接你，算了，我自己开车",
          [repair("他来接你", "算了", "我自己开车")]),

    # ---- Clean controls (5): no self-correction ----
    entry("zh_clean_001", "我们需要更新数据库以支持新功能"),
    entry("zh_clean_002", "明天发布新版本到生产环境"),
    entry("zh_clean_003", "测试全部通过了，可以合并代码"),
    entry("zh_clean_004", "请检查日志文件中的错误信息"),
    entry("zh_clean_005", "配置文件已经修改并提交到仓库"),

    # ---- Edge cases (2): plain `不是`/`不对` as negation ----
    # These usages are predicate-only (sentence-final negation) and
    # should NOT be flagged because there's no repair clause after.
    entry("zh_edge_001", "这个答案不对"),
    entry("zh_edge_002", "我不是工程师"),
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
