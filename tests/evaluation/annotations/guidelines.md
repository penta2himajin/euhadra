# Annotation Guidelines — L3 Direct Evaluation

ガイドライン版本: 0.1 (2026-04-25、初版)

このディレクトリの JSONL ファイルは euhadra の Tier 1 (filler 除去) /
Tier 2 (自己訂正検出) 層を **直接評価 (F1)** で測るための ground-truth
annotation です。`docs/evaluation.md` §5.6.1 のスキーマに従います。

## 1. ファイル構成

| ファイル | 言語 | タスク | 出典 / 由来 |
|---|---|---|---|
| `ja_self_correction.jsonl` | ja | self-correction span (reparandum / interregnum / repair) | 自社制作 (Claude による下書き、人手レビュー前) |

(現状 ja のみ。zh / en は §6 Phase C-3 / D で追加予定。)

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

## 4. 現状の v0.1 内容

`ja_self_correction.jsonl`:
- **35 発話** (28 自己訂正 + 5 クリーン + 2 否定 edge case)
- カバー: time/date (5)、place names (5)、people (5)、numbers/counters (5)、
  verbs/states (5)、phrase-level (3)、clean controls (5)、edge cases (2)
- 全エントリ `substitution` 型のみ

すべて Claude が下書き、ネイティブ話者の人手レビュー **未実施**。
PR レビューで以下を確認してほしい:

1. cue 語の使い方が自然か (機械的すぎないか)
2. span 境界の判定が §3.2 のルールに従っているか
3. edge case (`edge_001`, `edge_002`) の判定が妥当か
4. クリーン文に違和感がないか
5. カテゴリのバリエーションが適切か (追加すべき類型はあるか)

修正提案は JSONL の該当行を直接編集する形で歓迎します。

## 5. 検証方法

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
cargo eval-l3 -- --task self-correction --lang ja
```

## 6. 参照規約

- **Switchboard Disfluency Annotation Stylebook** (Meteer et al. 1995) — reparandum/interregnum/repair 構造の定義原典
- **CSJ 転記マニュアル** (NINJAL) — `(F)` / `(D)` 慣例
- 詳細リンクは `docs/evaluation.md` §5.6.2 を参照
