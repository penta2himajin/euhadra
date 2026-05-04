# Annotation Guidelines — L3 Direct Evaluation

ガイドライン版本: 0.1 (2026-04-25、初版)

このディレクトリの JSONL ファイルは euhadra の Tier 1 (filler 除去) /
Tier 2 (自己訂正検出) 層を **直接評価 (F1)** で測るための ground-truth
annotation です。`docs/evaluation.md` §5.6.1 のスキーマに従います。

## 1. ファイル構成

| ファイル | 言語 | タスク | 出典 / 由来 |
|---|---|---|---|
| `ja_self_correction.jsonl` | ja | self-correction span (reparandum / interregnum / repair) | 自社制作 (Claude による下書き、人手レビュー前) |
| `es_self_correction.jsonl` | es | self-correction span (reparandum / interregnum / repair) | 自社制作 (Claude による下書き、人手レビュー前) — `scripts/build_es_self_correction_annotations.py` で再生成可能 |
| `en_self_correction.jsonl` | en | self-correction span (reparandum / interregnum / repair) | 自社制作 (Claude による下書き、人手レビュー前) — `scripts/build_en_self_correction_annotations.py` で再生成可能 |
| `zh_self_correction.jsonl` | zh | self-correction span (reparandum / interregnum / repair) | 自社制作 (Claude による下書き、人手レビュー前) — `scripts/build_zh_self_correction_annotations.py` で再生成可能 |
| `ko_self_correction.jsonl` | ko | self-correction span (reparandum / interregnum / repair) | 自社制作 (Claude による下書き、人手レビュー前) — `scripts/build_ko_self_correction_annotations.py` で再生成可能 |
| `en_phoneme_correction.jsonl` + `en_phoneme_correction_dict.json` + `en_phoneme_correction_base_dict.json` | en | phoneme-correction (Tier 2 PhonemeCorrector direct F1) | 自社制作 (Claude による下書き、人手レビュー前) — `scripts/build_en_phoneme_correction_annotations.py` で再生成可能。Annotation JSONL は `text` / `expected_text` / `corrections` フィールド、dict.json は domain-term の `{word: ipa}`、base_dict.json は input 単語を音素化するための CMUdict 代替 |
| `en_filler.jsonl` | en | filler span (Tier 1 SimpleFillerFilter::english direct F1) | 自社制作 (Claude による下書き、人手レビュー前) — `scripts/build_en_filler_annotations.py` で再生成可能 |
| `ja_filler.jsonl` | ja | filler span (Tier 1 JapaneseFillerFilter direct F1) | 自社制作 (Claude による下書き、人手レビュー前) — `scripts/build_ja_filler_annotations.py` で再生成可能 |
| `zh_filler.jsonl` | zh | filler span (Tier 1 ChineseFillerFilter direct F1) | 自社制作 (Claude による下書き、人手レビュー前) — `scripts/build_zh_filler_annotations.py` で再生成可能 |
| `ko_filler.jsonl` | ko | filler span (Tier 1 SimpleFillerFilter::korean direct F1) | 自社制作 (Claude による下書き、人手レビュー前) — `scripts/build_ko_filler_annotations.py` で再生成可能 |

(self-correction: ja / es / en / zh / ko の 5 言語完備。filler span: en / ja / zh / es / ko の 5 言語完備 — es は CIEMPIESS Test ベース、他 4 言語は手作り。phoneme-correction: en のみ — ja / zh / ko / es は別 G2P + IPA 辞書を要するため将来拡張。)

## 2. スキーマ要約

各行は 1 発話の JSON オブジェクト:

```json
{
  "utterance_id": "ja_self_corr_time_001",
  "text": "明日、いや明後日に会議があります",
  "fillers": [],
  "repairs": [
    {
      "reparandum":  {"start": 0, "end": 2},
      "interregnum": {"start": 3, "end": 5},
      "repair":      {"start": 5, "end": 8},
      "type": "substitution"
    }
  ]
}
```

- `start` / `end` は **文字オフセット** (Unicode scalar value 数)、半開区間 `[start, end)`
- `text` は post-ASR の transcript (現状の `JapaneseFillerFilter` /
  `SelfCorrectionDetector` が想定する 「、」区切りフォーマット)
- `repairs` 空配列 = 「自己訂正は無い」(クリーン or 否定の edge case)
- `repair.type` の closed set: `substitution` / `insertion` / `deletion` / `abandoned`

## 3. ja self-correction の判定規則

### 3.1 cue 語 (closed set)

`SelfCorrectionDetector` 実装に揃える:

- `いや`
- `じゃなくて`
- `じゃなく`
- `ではなく`
- `ていうか`
- `っていうか`
- `じゃない` (※否定との曖昧性に注意、§3.4)

### 3.2 span 境界の取り方

形 `<reparandum>、<cue><repair>` を典型として:

- **reparandum**: 訂正で消したい語句のみ。前後の助詞 / 接続詞は含めない
  - 例: 「明日、いや明後日」→ reparandum は `明日` (「明日に」「明日の」ではない)
- **interregnum**: cue 語そのものだけ。直前の `、` は **含めない**
  - 例: 「明日、いや明後日」→ interregnum は `いや` (`、いや` ではない)
  - 理由: `、` は ASR/転記由来の構造記号で、disfluency 自体ではない
- **repair**: 訂正後の語句のみ。後続の助詞は含めない
  - 例: 「明日、いや明後日に会議」→ repair は `明後日` (「明後日に」ではない)

### 3.3 `type` の使い分け

- `substitution`: 大半のケース。reparandum を repair で置換
- `insertion`: reparandum 0 文字、何かを追加で言い直す (現状未収録)
- `deletion`: 撤回のみ (現状未収録)
- `abandoned`: 言いかけて止めた、repair 0 文字 (現状未収録)

`substitution` 以外は v0.1 では収録していない。需要に応じて拡張。

### 3.4 否定 vs 訂正の境界 (edge case)

`じゃない` は否定 (negation) と訂正 cue の両方の用法がある。判定基準:

- **訂正**: `<前段>、じゃない<後段>` の形。`<後段>` が `<前段>` を **置換する** 内容語
  - 例: 「鈴木課長、じゃない、佐藤課長です」→ repair = 佐藤課長
- **否定**: `<前段>じゃない` で文/節を終える、または後段が前段を否定するだけで置換しない
  - 例: 「それは赤いリンゴじゃない」→ repair なし
  - 例: 「それは赤いリンゴじゃない、青いリンゴだ」→ これは曖昧。**収録しない** (annotator 間で IAA が低くなるケースは除外)

`いや` も同様で、文頭の感動詞 (「いや、それは違います」) は cue ではない。
判定基準: cue の **直前に reparandum (内容語) が無い** なら disfluency ではなく感動詞。

### 3.5 1 発話に複数の自己訂正

複数 repair は `repairs` 配列に並べる。境界が重ならない限り独立に評価する。
v0.1 では収録していない (1 発話 1 repair)。

## 4. es self-correction の判定規則

### 4.1 cue 語 (closed set)

`SelfCorrectionDetector::correction_cues_es` (`src/processor.rs`) と一致:

- `mejor dicho`
- `quiero decir`
- `o sea`
- `perdón`
- `mejor`
- `digo`
- `no es`
- `no` (※否定との曖昧性に注意、§4.4)

### 4.2 span 境界の取り方

形 `<reparandum> <cue> <repair>` を典型として (空白区切り):

- **reparandum**: 訂正で消したい語句のみ。前後の助詞 / 句読点は含めない
  - 例: 「voy mañana no voy hoy」→ reparandum は `voy mañana` (空白を含めない)
- **interregnum**: cue 語そのものだけ。前後の空白 / 句読点は含めない
  - 例: 「voy mañana no voy hoy」→ interregnum は `no` (` no ` ではない)
  - 理由: 空白は語境界記号で disfluency 自体ではない
- **repair**: 訂正後の語句のみ。文の残りの部分は含めない
  - 例: 「el rojo perdón el azul me gusta más」→ repair は `el azul` (`el azul me gusta más` ではない)

### 4.3 detector の `count_shared_prefix_from_end` 制約

`SelfCorrectionDetector::detect_spanish` は **reparandum の末尾と repair の先頭で
最低 1 単語が共有されている** ことを要求する (`min_shared_words = 1`)。これは
「pure な否定 (`el gato no come pescado`)」を訂正と誤判定しないための保険。

ja の場合は cue (`いや` 等) の存在自体が十分だが、es では否定 / 訂正 cue が同じ
単語 (`no`) を共有するため、共有プレフィックス制約で曖昧性を解いている。

実用的には、訂正の自然な発話パターン (「voy a Madrid perdón **voy** a Barcelona」)
で repair 句の冒頭が reparandum の冒頭を反復することが多く、この制約は
ほぼ無料で満たされる。アノテーション側でも、各エントリで repair が
reparandum と少なくとも 1 単語を共有するように構成している。

### 4.4 否定 vs 訂正の境界 (edge case)

`no` は否定 (negation) と訂正 cue の両方の用法がある。判定基準:

- **訂正**: `<前段> no <後段>` の形で、`<後段>` の冒頭が `<前段>` の冒頭と
  同じ語を反復する。
  - 例: 「voy mañana no voy hoy」→ repair = `voy hoy`
- **否定**: `no` の後段が前段と語を共有しない、または `no` が前段を否定するだけ
  - 例: 「no vino a la fiesta porque estaba enfermo」→ repair なし (cue 直後に
    `no` の対象が無く、共有プレフィックスも無い)

§3.4 の ja `じゃない` と同じく、annotator 間で IAA が低くなるケース (両用法とも
取れる例) は **収録しない** 方針。

### 4.5 1 発話に複数の自己訂正

ja と同様、v0.1 では収録していない (1 発話 1 repair)。

## 5. 現状の v0.1 内容

`ja_self_correction.jsonl`:
- **40 発話** (33 自己訂正 + 5 クリーン + 2 否定 edge case)
- カバー: time/date (5)、place names (5)、people (5)、numbers/counters (5)、
  verbs/states (5)、phrase-level (3)、`じゃない` を訂正 cue として用いる例 (3)、
  長文文脈 (2)、clean controls (5)、edge cases (2)
- cue 別: `いや` (10)、`じゃなくて` (5)、`ではなく` (5+1 long)、`じゃなく` (3)、
  `じゃない` (3 訂正 + 1 否定 edge case)、`ていうか` (2)、`っていうか` (1)
- 全エントリ `substitution` 型のみ

すべて Claude が下書き、ネイティブ話者の人手レビュー **未実施**。
PR レビューで以下を確認してほしい:

1. cue 語の使い方が自然か (機械的すぎないか)
2. span 境界の判定が §3.2 のルールに従っているか
3. edge case (`edge_001`, `edge_002`) の判定が妥当か
4. クリーン文に違和感がないか
5. カテゴリのバリエーションが適切か (追加すべき類型はあるか)

修正提案は JSONL の該当行を直接編集する形で歓迎します。

`es_self_correction.jsonl`:
- **40 発話** (33 自己訂正 + 5 クリーン + 2 否定 edge case)
- cue 別: `no` (5)、`perdón` (5)、`digo` (5)、`mejor dicho` (5)、`o sea` (5)、
  `quiero decir` (3)、`mejor` (3)、`no es` (2)
- 全エントリ `substitution` 型のみ
- 構造: 各 self-correction エントリで repair の冒頭が reparandum の冒頭と
  少なくとも 1 単語を共有 (§4.3 の `count_shared_prefix_from_end` 制約)

すべて Claude が下書き、ネイティブ話者の人手レビュー **未実施**。
JA と同じく PR レビューで cue 用法の自然さ / span 境界 / edge case 判定 /
クリーン文の自然さ / カテゴリのバリエーションを確認してほしい。

再生成は `scripts/build_es_self_correction_annotations.py` で実施できる。
スクリプト末尾の `ENTRIES` リストを編集して `--out` 指定で出力。
`--verify-only` で各 span のオフセットを再スライスして整合性を検証する
(CI で叩くと、手編集による drift を即座に検出できる)。

## 6. 検証方法

各エントリの span が意図通りの文字列を切り出せるか手元確認:

```bash
python3 -c "
import json
for line in open('tests/evaluation/annotations/ja_self_correction.jsonl'):
    a = json.loads(line)
    for r in a.get('repairs', []):
        rep = a['text'][r['reparandum']['start']:r['reparandum']['end']]
        inter = a['text'][r['interregnum']['start']:r['interregnum']['end']]
        repair = a['text'][r['repair']['start']:r['repair']['end']]
        print(f\"{a['utterance_id']:30}  rep={rep!r}  inter={inter!r}  repair={repair!r}\")
"
```

評価実行:

```bash
# Japanese
cargo run --example eval_l3 -- --task self-correction --lang ja \
    --input tests/evaluation/annotations/ja_self_correction.jsonl
# Spanish
cargo run --example eval_l3 -- --task self-correction --lang es \
    --input tests/evaluation/annotations/es_self_correction.jsonl
```

`--verbose` で fp / fn ケースの per-utterance な diff を表示できる。

## 7. 参照規約

- **Switchboard Disfluency Annotation Stylebook** (Meteer et al. 1995) — reparandum/interregnum/repair 構造の定義原典
- **CSJ 転記マニュアル** (NINJAL) — `(F)` / `(D)` 慣例
- 詳細リンクは `docs/evaluation.md` §5.6.2 を参照
