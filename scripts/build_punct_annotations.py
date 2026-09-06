#!/usr/bin/env python3
"""Build synthetic punctuation-restoration annotations.

`docs/spec.md` §11.4: Tier 2 punctuation gold can be synthesised by
stripping punctuation from already-clean text — the stripped form is
the input, the original is the reference. No human annotation needed.

This script builds that gold for the languages euhadra ships
(en / ja / zh / ko / es), writing JSONL under `data/punct_eval/`
(gitignored; Wikipedia text is CC-BY-SA 4.0 and is fetched on demand,
same policy as `scripts/download_paragraph_corpus.py`).

Each line:
  {
    "id": "ja_0003",
    "lang": "ja",
    "gold": "今日は雨でした。午後は晴れます。",
    "input": "今日は雨でした午後は晴れます"
  }

`input` is `gold` with punctuation removed and, for cased scripts,
lowercased — matching what an ASR transcript typically looks like
before Tier 2.

Usage:
    scripts/build_punct_annotations.py
    scripts/build_punct_annotations.py --langs ja,en --per-lang 40
    PUNCT_EVAL_DIR=/tmp/punct scripts/build_punct_annotations.py

Licensing (informational — see docs/model-licenses.md):
  - Wikipedia article text: CC-BY-SA 4.0
    Declaration: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use
    License text: https://creativecommons.org/licenses/by-sa/4.0/legalcode
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request

UA = "euhadra-eval/0.1 (https://github.com/penta2himajin/euhadra)"
OUT_DIR = os.environ.get("PUNCT_EVAL_DIR", "data/punct_eval")
SEED = 20260906

# Punctuation the restorer is scored on. Quotes / brackets are left in
# place so the skeleton still aligns; we only strip marks the Tier 2
# slot is supposed to insert.
PUNCT_CHARS = set(
    ".!?,:;。、？！，；："
)

# Enough articles that a few dozen punctuated sentences fall out after
# filtering. Deliberately mixed domains so the bake-off is not a
# single-register artefact.
ARTICLES = {
    "en": [
        "Kubernetes", "Photosynthesis", "Baroque music", "Volcano",
        "Coffee", "Neutrino", "Bicycle", "Antarctica", "Chess",
        "Lighthouse", "Vaccine", "Jazz", "Submarine", "Typewriter",
    ],
    "ja": [
        "日本の鉄道", "味噌", "浮世絵", "台風", "将棋", "和紙",
        "火山", "落語", "温泉", "醤油", "新幹線", "桜", "茶道",
        "折り紙",
    ],
    "zh": [
        "围棋", "长城", "茶", "青花瓷", "京剧", "丝绸之路",
        "针灸", "水稻", "书法", "台风", "火药", "造纸",
    ],
    "ko": [
        "김치", "한글", "태권도", "부산", "한강", "온돌",
        "판소리", "고려청자", "세종대왕", "제주도",
    ],
    "es": [
        "Flamenco", "Quinoa", "Aqueducto de Segovia", "Guitarra",
        "Café", "Volcán", "Ajedrez", "Submarino", "Faro",
        "Vacuna",
    ],
}

LANG_TO_WIKI = {
    "en": "en",
    "ja": "ja",
    "zh": "zh",
    "ko": "ko",
    "es": "es",
}


def fetch_extract(lang: str, title: str) -> str:
    wiki = LANG_TO_WIKI[lang]
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exintro": "0",
            "redirects": "1",
            "format": "json",
            "titles": title,
        }
    )
    url = f"https://{wiki}.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract") or ""
    return ""


def _strip_invisible(text: str) -> str:
    """Drop zero-width / BOM chars (common in Wikipedia extracts).

    They break skeleton alignment in the punct bake-off without being
    visible in diffs — especially costly for Spanish.
    """
    for ch in ("\u200b", "\ufeff", "\u200c", "\u200d"):
        text = text.replace(ch, "")
    return text


def split_sentences(lang: str, text: str) -> list[str]:
    """Rough sentence split — good enough for gold synthesis.

    We keep only sentences that already carry terminal punctuation, so
    the reference is something a restorer can be scored against.
    """
    text = _strip_invisible(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if lang in ("ja", "zh"):
        parts = re.split(r"(?<=[。？！])", text)
    else:
        parts = re.split(r"(?<=[.?!])\s+", text)
    out = []
    for p in parts:
        s = p.strip()
        if len(s) < 8:
            continue
        if not any(c in s for c in ".。？！?!"):
            continue
        # Drop headings / list debris.
        if s.startswith("=") or s.startswith("*"):
            continue
        out.append(s)
    return out


def strip_punct(lang: str, text: str) -> str:
    chars = [c for c in text if c not in PUNCT_CHARS]
    out = "".join(chars)
    out = re.sub(r" {2,}", " ", out).strip()
    if lang in ("en", "es"):
        out = out.lower()
    return out


def build_lang(lang: str, per_lang: int, rng: random.Random) -> list[dict]:
    gold_sents: list[str] = []
    for title in ARTICLES[lang]:
        try:
            extract = fetch_extract(lang, title)
        except Exception as exc:  # noqa: BLE001 — keep going on one article
            print(f"[warn] {lang}/{title}: {exc}", file=sys.stderr)
            time.sleep(1)
            continue
        gold_sents.extend(split_sentences(lang, extract))
        time.sleep(0.2)
        if len(gold_sents) >= per_lang * 3:
            break

    # Prefer sentences that also contain an internal comma/、 so the
    # bake-off is not just "did you append a period".
    def richness(s: str) -> int:
        return sum(1 for c in s if c in ",、，")

    gold_sents = sorted(set(gold_sents), key=lambda s: (-richness(s), -len(s)))
    picked = gold_sents[:per_lang]
    rng.shuffle(picked)

    rows = []
    for i, gold in enumerate(picked):
        inp = strip_punct(lang, gold)
        if not inp or inp == gold:
            continue
        rows.append(
            {
                "id": f"{lang}_{i:04d}",
                "lang": lang,
                "gold": gold,
                "input": inp,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--langs",
        default="en,ja,zh,ko,es",
        help="Comma-separated language codes",
    )
    ap.add_argument("--per-lang", type=int, default=40)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    for lang in langs:
        if lang not in ARTICLES:
            print(f"[error] unsupported lang: {lang}", file=sys.stderr)
            return 2

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(SEED)
    summary = {}
    for lang in langs:
        print(f"[build] {lang}…")
        rows = build_lang(lang, args.per_lang, rng)
        path = os.path.join(args.out_dir, f"{lang}_punct.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[lang] = {"n": len(rows), "path": path}
        print(f"[ok] {lang}: {len(rows)} → {path}")

    meta = os.path.join(args.out_dir, "manifest.json")
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "seed": SEED,
                "per_lang": args.per_lang,
                "langs": summary,
                "note": "Synthetic punct gold from Wikipedia extracts; "
                "CC-BY-SA 4.0 text is not redistributed via git.",
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[ok] manifest → {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
