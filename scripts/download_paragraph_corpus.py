#!/usr/bin/env python3
"""Build a paragraph-segmentation corpus from Wikipedia.

`ParagraphSplitter` has never been scored, in any language
(`docs/model-upgrade-candidates.md` §5.5). This produces the data to
score it with, in the two shapes the text-segmentation literature uses:

  choi_<lang>.jsonl    Choi-style synthetic concatenation. Each document
                       is K paragraphs drawn from K *different* articles,
                       so the segment boundaries are known by
                       construction rather than annotated. This is the
                       classic benchmark shape (Choi 2000 concatenated
                       Brown-corpus paragraphs the same way).

  author_<lang>.jsonl  One article's own paragraphs, in order. The gold
                       boundaries are where its author put them. Harder
                       and noisier than the synthetic task — paragraph
                       breaks are partly stylistic — but closer to what
                       dictation output actually looks like.

Both files hold paragraphs as text, not sentences: the evaluator splits
them with the same `split_sentences` the splitter itself uses, so gold
boundary indices cannot drift from the segmenter's own tokenisation.

Output goes under `data/paragraph_corpus/`, which is gitignored. The
text is CC-BY-SA 4.0 (Wikipedia) and is fetched on demand rather than
redistributed, matching the policy in `docs/evaluation.md` §1.

Usage:
    scripts/download_paragraph_corpus.py
    PARAGRAPH_CORPUS_DIR=/tmp/pc scripts/download_paragraph_corpus.py
    PARAGRAPH_CORPUS_DOCS=40 scripts/download_paragraph_corpus.py

Licensing (informational — see docs/model-licenses.md):
  - Wikipedia article text: CC-BY-SA 4.0
    Declaration: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use
    License text: https://creativecommons.org/licenses/by-sa/4.0/legalcode
"""

import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

UA = "euhadra-eval/0.1 (https://github.com/penta2himajin/euhadra)"

OUT_DIR = os.environ.get("PARAGRAPH_CORPUS_DIR", "data/paragraph_corpus")
N_DOCS = int(os.environ.get("PARAGRAPH_CORPUS_DOCS", "30"))
# Paragraphs per synthetic document. Choi used 10; dictation output is
# shorter, so 6 keeps the documents in a plausible length range while
# still giving the relative rule a distribution to judge against.
SEGMENTS_PER_DOC = 6
MIN_PARA_CHARS = 250
SEED = 20260731

# Deliberately spread across unrelated domains: the synthetic task is
# only meaningful if paragraphs from different articles really are about
# different things.
ARTICLES = {
    "en": [
        "Kubernetes", "Photosynthesis", "Baroque music", "Volcano",
        "Byzantine Empire", "Coffee", "Neutrino", "Bicycle",
        "Sourdough", "Antarctica", "Chess", "Monsoon",
        "Lighthouse", "Vaccine", "Origami", "Glacier",
        "Jazz", "Submarine", "Coral reef", "Typewriter",
    ],
    "ja": [
        "日本の鉄道", "味噌", "浮世絵", "台風",
        "将棋", "和紙", "火山", "落語",
        "温泉", "竹", "灯台", "醤油",
        "能", "新幹線", "桜", "漆",
        "相撲", "茶道", "琵琶湖", "折り紙",
    ],
    "zh": [
        "围棋", "长城", "茶", "青花瓷",
        "熊猫", "京剧", "丝绸之路", "针灸",
        "水稻", "书法", "竹", "台风",
        "故宫", "中医", "陶瓷", "黄河",
        "太极拳", "造纸术", "月饼", "风筝",
    ],
    "ko": [
        "김치", "한글", "태권도", "판소리",
        "제주도", "인삼", "온돌", "한복",
        "고려청자", "된장", "거문고", "탈춤",
        "불국사", "막걸리", "씨름", "한지",
        "청계천", "비빔밥", "설악산", "가야금",
    ],
    "es": [
        "Flamenco", "Volcán", "Café", "Gaudí",
        "Amazonas", "Tango", "Aceite de oliva", "Machu Picchu",
        "Paella", "Glaciar", "Cerámica", "Guitarra",
        "Desierto de Atacama", "Vino", "Ajedrez", "Coral",
        "Molino de viento", "Faro", "Bicicleta", "Origami",
    ],
}


def fetch_extract(lang, title):
    """Plain-text extract of one article, or '' if unavailable."""
    q = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "format": "json", "redirects": 1, "titles": title,
    })
    url = f"https://{lang}.wikipedia.org/w/api.php?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


def paragraphs_of(extract):
    """Substantial prose paragraphs, dropping section headings."""
    out = []
    for line in extract.split("\n"):
        line = line.strip()
        if not line or line.startswith("=="):
            continue
        if len(line) < MIN_PARA_CHARS:
            continue
        out.append(line)
    return out


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    for lang, titles in ARTICLES.items():
        per_article = []
        for title in titles:
            try:
                paras = paragraphs_of(fetch_extract(lang, title))
            except Exception as e:  # noqa: BLE001 - report and carry on
                print(f"[warn] {lang}/{title}: {e}", file=sys.stderr)
                continue
            if len(paras) >= 2:
                per_article.append((title, paras))
            time.sleep(0.2)  # be polite to the API

        if len(per_article) < SEGMENTS_PER_DOC:
            print(
                f"[warn] {lang}: only {len(per_article)} usable articles, "
                f"need {SEGMENTS_PER_DOC} — skipping",
                file=sys.stderr,
            )
            continue

        # --- Choi-style synthetic concatenation -----------------------
        choi_path = os.path.join(OUT_DIR, f"choi_{lang}.jsonl")
        with open(choi_path, "w", encoding="utf-8") as f:
            for i in range(N_DOCS):
                picked = rng.sample(per_article, SEGMENTS_PER_DOC)
                paragraphs = [rng.choice(paras) for _title, paras in picked]
                f.write(json.dumps({
                    "doc_id": f"choi_{lang}_{i:03d}",
                    "lang": lang,
                    "sources": [t for t, _ in picked],
                    "paragraphs": paragraphs,
                }, ensure_ascii=False) + "\n")

        # --- Author's own paragraph structure -------------------------
        author_path = os.path.join(OUT_DIR, f"author_{lang}.jsonl")
        with open(author_path, "w", encoding="utf-8") as f:
            written = 0
            for title, paras in per_article:
                if len(paras) < 3:
                    continue
                f.write(json.dumps({
                    "doc_id": f"author_{lang}_{written:03d}",
                    "lang": lang,
                    "sources": [title],
                    "paragraphs": paras[:8],
                }, ensure_ascii=False) + "\n")
                written += 1

        print(f"{lang}: {len(per_article)} articles → "
              f"{N_DOCS} choi docs, {written} author docs")


if __name__ == "__main__":
    main()
