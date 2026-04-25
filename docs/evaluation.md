# Evaluation Datasets and Testing Policy

euhadra のパイプラインを評価する際に使用する ASR テストデータセットとその使用方針を定める。`docs/spec.md` §9 で定義した MVP スコープに対応するレイヤー (ASR Adapter / Tier 1 TextFilter / Tier 2 TextProcessor / Tier 3 LlmRefiner) を、どのデータセットでどう検証するかをここに集約する。

このドキュメントの位置付け:
- 評価の **方針** を定めるものであって、ベンチマーク結果のレポートではない。実測値は別途 `docs/benchmarks/` 配下に蓄積する想定 (本 PR では追加しない)
- 評価コードと評価データを **同一リポジトリに混在させない**。データは外部から都度 download する
- ライセンスが OSS 配布物 (Cochlis 等) と衝突しうるデータセットを明示的に切り分ける

---

## 1. テストスイートの 3 層構造

評価コストとカバー範囲のトレードオフを踏まえ、以下の 3 層に分けて運用する。

### 1.1 L1 — スモークテスト (CI 用、毎回実行)

**目的**: パイプライン全体が壊れていないことの確認。CER / WER の trend tracking。

**データ**:
- FLEURS の en / ja / zh 各 100 発話程度のサブセット
- 軽量 (合計数百 MB)、CC-BY 4.0、HuggingFace `google/fleurs` から取得

**期待実行時間**: 数分以内 (CPU でも実行可能なサイズ)

**メトリクス**:
- ASR レベル: WER (en) / CER (ja, zh)
- パイプライン全体: 例外なく完走するか (smoke 確認)

### 1.2 L2 — 標準ベンチマーク (リリース毎、手動実行)

**目的**: ASR Adapter 単体性能を SOTA と比較。パイプライン全体の品質回帰を検出。

**データ**:

| 言語 | データセット | 取得元 |
|------|-------------|-------|
| en | LibriSpeech `test-clean` / `test-other` | OpenSLR、CC-BY 4.0 |
| ja | ReazonSpeech `test` または Common Voice ja `test` | HuggingFace |
| zh | AISHELL-1 `test` | OpenSLR、Apache-2.0 |

**期待実行時間**: 数時間 (GPU 推奨)

**メトリクス**:
- WER / CER
- RTF (Real-Time Factor) — Cochlis のオンデバイス要件確認用

### 1.3 L3 — euhadra 固有検証 (Tier 1+2 評価)

**目的**: ASR 後処理 (filler 除去・自己訂正検出・句読点・段落分割・固有名詞補正) が実際に効果を出しているかの定量検証。汎用 ASR ベンチマークでは測れない euhadra 独自の品質を測る。

**データ**:

| 検証対象 | データセット | 言語 |
|---------|-------------|------|
| filler / phonological process / self-correction | Buckeye | en |
| filler / 自己訂正 | CSJ (入手できれば) | ja |
| 連続発話 + 段落分割 | TED-LIUM 3 | en |
| 会議・自然発話 | WenetSpeech `test-meeting` | zh |

**期待実行時間**: 半日〜1 日

**メトリクス**: §4 のレイヤー別マッピングを参照。専用評価指標 (filler 除去 F1、自己訂正検出 F1、段落境界 P/R) を実装する必要がある。

---

## 2. 言語別データセット一覧

### 2.1 多言語横断

| データセット | 言語数 | 内容 | ライセンス | 用途 |
|-------------|-------|-----|-----------|------|
| FLEURS | 102 (16 語族) | 読み上げ、FLoRes-101 由来 | CC-BY 4.0 | L1 スモーク、多言語比較 |
| Common Voice 22 | 124+ locales | クラウドソース読み上げ | CC0 1.0 | L1 / L2、ライセンス自由 |
| MLS | 8 (英独蘭仏西伊葡波) | LibriVox オーディオブック | CC-BY 4.0 | 大規模学習・評価 |
| VoxPopuli | 16 欧州言語 | 議会演説 | CC-BY-NC 4.0 | **非商用評価のみ** |

**注**: VoxPopuli は CC-BY-NC のため、商用利用 (Cochlis 等) を意識した評価では使わない。学術用途・OSS 評価レポートのみ。

### 2.2 日本語

| データセット | 内容 | ライセンス | 用途 |
|-------------|-----|-----------|------|
| ReazonSpeech v2 | TV 録音、自然発話 | 著作権法第 30 条の 4 (機械学習用途のみ) | L2、ベンチマーク参考値 |
| CSJ (日本語話し言葉コーパス) | 講演・対話、filler/自己訂正含む | NII 経由、有償 | L3 (入手できれば) |
| JSUT | クリーン読み上げ (1 名話者) | CC-BY-SA 4.0 | TTS 寄り、ASR 補助 |
| Common Voice ja | クラウドソース | CC0 1.0 | L2、ネイティブ性に注意 |
| FLEURS ja | 読み上げ | CC-BY 4.0 | L1 |

**ReazonSpeech 用途制限**: 著作権法第 30 条の 4 に基づき機械学習目的のみ。**商用提供物 (Cochlis のリリース成果物) には絶対に同梱しない**。評価レポートでスコアを公表する際も、データ取得経路と用途を明記する。

### 2.3 英語

| データセット | 内容 | ライセンス | 用途 |
|-------------|-----|-----------|------|
| LibriSpeech | オーディオブック読み上げ | CC-BY 4.0 | L2 標準ベースライン |
| TED-LIUM 3 | TED トーク | CC-BY-NC-ND 3.0 | **非商用のみ**、L3 段落分割 |
| Switchboard | 電話会話、filler 多 | LDC 有償 | L3 (入手できれば) |
| Buckeye | 自然会話、IPA 音素アノテ | 研究用無償 | L3、Speciphonorm と共用 |
| AMI Meeting | 会議 | CC-BY 4.0 | 重なり発話・ノイズ評価 |
| CHiME-6/7/8 | 家庭・遠距離 mic、雑音 | 多くは無償 | AudioPreprocessor 評価 |
| GigaSpeech | マルチドメイン 10000h | Apache-2.0 | 大規模、領域多様性 |
| Common Voice en | クラウドソース | CC0 1.0 | アクセント多様性 |

**TED-LIUM 3 注意**: CC-BY-NC-ND は非営利かつ改変禁止。Cochlis 商用ベンチマーク用途には不可。OSS 評価レポートのみ。

**Buckeye の戦略的価値**: euhadra (Tier 1+2) と Speciphonorm (phonological normalization) の両研究で使えるため、**共通評価ハーネス**を `tests/evaluation/buckeye_runner.rs` (仮) に集約する設計を推奨。

### 2.4 中国語

| データセット | 内容 | ライセンス | 用途 |
|-------------|-----|-----------|------|
| AISHELL-1 | クリーン読み上げ | Apache-2.0 | L2 標準ベースライン |
| AISHELL-2 | 拡大版 1000h | Apache-2.0 | 大規模ベースライン |
| WenetSpeech | YouTube + Podcast、自然発話 | CC-BY 4.0 | L3 自然発話 |
| HKUST | Mandarin 電話会話 | LDC 有償 | 会話、L3 候補 |
| KeSpeech | 8 方言 + Mandarin | 学術利用 | 方言頑健性 |
| FLEURS zh | 読み上げ | CC-BY 4.0 | L1 |

**Disfluency データセットの不足**: 中国語で揃った disfluency 注釈コーパスは事実上存在しない。Tier 1 (filler 除去) / Tier 2 (自己訂正検出) の中国語検証は、WenetSpeech `test-meeting` の自然発話に対して **自前の軽量アノテーション** を付ける前提で進める。

---

## 3. レイヤー別マッピング

`docs/spec.md` §3 で定義した各レイヤーに対して、評価方法と推奨データを示す。

### 3.1 ASR Adapter (`AsrAdapter` trait)

**評価対象**: WhisperLocal、ParakeetAdapter、(将来) Whisper Cloud

**メトリクス**:
- WER (en) / CER (ja, zh)
- RTF (Real-Time Factor)
- レイテンシ (initial token latency / final transcript latency)

**推奨データ**:
- L1: FLEURS en/ja/zh
- L2: LibriSpeech test-other / ReazonSpeech-test / AISHELL-1 test
- ノイズ耐性: CHiME-6 (en)、Common Voice 低品質クリップ

### 3.2 Tier 1 — TextFilter (filler removal)

**評価対象**: SimpleFillerFilter、JapaneseFillerFilter、OnnxEmbeddingFilter

**メトリクス**:
- Filler removal **F1** (precision, recall を別途報告)
- False positive rate (内容語を filler と誤判定する率)
- 処理レイテンシ

**ground truth の作り方**:
- Buckeye: filler annotation を直接利用
- CSJ: X-JToBI / 短単位品詞情報から filler 抽出
- ReazonSpeech / WenetSpeech: 自前で軽量アノテーション (100-500 発話)

**推奨データ**:
- en: Buckeye、Switchboard (NXT annotation)
- ja: CSJ (入手できれば)、ReazonSpeech サブセット + 自前アノテ
- zh: WenetSpeech サブセット + 自前アノテ

### 3.3 Tier 2 — SelfCorrectionDetector

**評価対象**: SelfCorrectionDetector (英語/日本語ルール)

**メトリクス**:
- Self-correction detection **F1**
- Reparandum 削除の正確性 (削除後テキストが意図通りか)

**ground truth**:
- Switchboard NXT (英): disfluency annotation 標準
- CSJ (日): repair 構造のアノテーション
- 中国語: 揃ったコーパスなし、自前アノテ前提

**推奨データ**:
- en: Switchboard NXT
- ja: CSJ
- zh: 自前アノテーションのみ

### 3.4 Tier 2 — PhonemeCorrector

**評価対象**: PhonemeCorrector (CMUdict + G2P + 複合スコア)

**メトリクス**:
- 固有名詞・技術用語の正確な認識率
- False correction rate (本来正しい語を誤って置換する率)

**評価設計**: 専用ベンチマークが存在しないため、自前で構築する必要あり。
- TED-LIUM の科学技術トークから固有名詞リストを抽出
- 同じ音声に対して、custom dictionary を渡した場合と渡さない場合の WER 差分を測定
- 日本語: ReazonSpeech のニュースセグメント (固有名詞多)、JNAS (新聞読み上げ)

### 3.5 Tier 2 — ParagraphSplitter

**評価対象**: ParagraphSplitter (埋め込み距離 + 最大文数制約)

**メトリクス**:
- Paragraph boundary **F1** (人手アノテーションとの一致)
- WindowDiff、Pk (segmentation 専用メトリクス)

**推奨データ**:
- en: TED-LIUM 3 (講演スクリプトの段落構造を ground truth に流用)
- ja: CSJ 講演、ReazonSpeech のニュース番組 (放送台本に段落あり)
- zh: WenetSpeech `test-meeting`

### 3.6 Tier 3 — LlmRefiner

**評価対象**: (将来) LlamaCppRefiner、Cloud refiner

**評価設計**: LLM の出力品質評価は別軸 (UTMOS, Subjective MOS, 人手評価)。MVP では実装後に検討。

### 3.7 全パイプライン (E2E)

**評価対象**: PipelineBuilder で構築した完全なパイプライン

**メトリクス**:
- 最終出力テキストと ground truth の **編集距離 (CER/WER)**
- ユーザー知覚レイテンシ (発話終了から出力確定まで)
- LLM なし構成 (Tier 1+2 のみ) の品質スコア — euhadra の MVP ゴール「LLM なしで 80 点」の検証

**推奨データ**:
- 軽量: FLEURS 3 言語サブセット
- 自然発話: TED-LIUM 3 + ReazonSpeech-test + WenetSpeech test-meeting

---

## 4. ライセンス・使用上の注意

### 4.1 商用提供物 (Cochlis / talkadict) との分離

以下のデータセットは **商用ベンチマークでの公表 NG**:
- VoxPopuli (CC-BY-NC)
- TED-LIUM 3 (CC-BY-NC-ND)
- ReazonSpeech (用途制限)

OSS 評価レポート (GitHub README、論文) では使用可だが、商用ドキュメント・マーケティング素材には流用しない。

### 4.2 評価コードと評価データの分離

- 評価コードは `tests/evaluation/` 配下にチェックイン
- データセットは **リポジトリにコミットしない**。ダウンロードスクリプト (`scripts/download_eval_data.sh`) のみ提供
- HuggingFace datasets 経由で取得できるものは `datasets.load_dataset(...)` を使う
- LDC 等の有償データは、ユーザー側で別途取得 (CSJ、Switchboard、HKUST)

### 4.3 ReazonSpeech の表記

ReazonSpeech を使った評価結果を公開する際は以下を明記:
- データ取得経路 (HuggingFace `reazon-research/reazonspeech`)
- 利用目的 (ASR モデル評価、機械学習研究)
- 著作権法第 30 条の 4 に基づく利用であること

### 4.4 評価レポートの公開ポリシー

- L1 / L2 のスコアは README.md の Quality Status セクションに掲載 (CC-BY のもののみ)
- L3 の Tier 1+2 評価結果は別途研究レポート (talkadict 論文等) としてまとめる
- 競合製品 (Aqua Voice、Wispr Flow 等) との比較は、利用規約上問題ない範囲で行う

---

## 5. 推奨テストスイート (2026 Q2 開始時点)

優先度順に実装する:

### Phase A — L1 スモーク (即実装)

```
tests/evaluation/
├── l1_smoke/
│   ├── fleurs_en.rs
│   ├── fleurs_ja.rs
│   └── fleurs_zh.rs
└── common/
    └── metrics.rs  (WER/CER 実装)
```

CI で毎 PR 実行。WER/CER の閾値を回帰検出に使う。

### Phase B — L2 標準ベンチマーク (Phase A 完了後)

```
tests/evaluation/l2_benchmark/
├── librispeech_test.rs
├── reazonspeech_test.rs
└── aishell1_test.rs
```

リリース前に手動実行、結果を `docs/benchmarks/YYYY-MM-DD.md` に記録。

### Phase C — L3 euhadra 固有検証 (Tier 1+2 評価)

```
tests/evaluation/l3_euhadra/
├── filler_removal_buckeye.rs       (F1 評価)
├── self_correction_swbd.rs         (F1 評価)
├── paragraph_split_tedlium.rs      (Pk, WindowDiff 評価)
└── phoneme_correction_custom.rs    (固有名詞辞書効果測定)
```

talkadict 研究レポートの基礎データとなる。

---

## 6. 将来検討

### 6.1 AudioPreprocessor 評価 (新 trait 追加時)

VAD・ノイズ抑制・ステム分離を導入する場合、以下を評価軸に追加:
- VAD: trigger latency (発話開始検出の遅延)、clipping rate (発話冒頭欠損率)
- 雑音抑制: PESQ, STOI, DNSMOS
- 評価データ: URGENT 2025 challenge データ (7 種類の歪み × 多言語)

### 6.2 パーソナル VAD / target speaker extraction

複数話者環境での dictation 精度評価。LExt / VoiceFilter 系を入れる場合に追加検討。

### 6.3 Code-switching

en-ja / en-zh の code-switched 入力。CS-FLEURS (52 言語、113 ペア) が候補。Cochlis の海外展開時に重要となる。

### 6.4 Long-form ASR

数十分の長尺音声に対する pipeline 全体の安定性評価。GigaSpeech、Libriheavy が候補。

### 6.5 共通評価ハーネス

Buckeye のように euhadra と Speciphonorm の両方で使うデータセットについては、両プロジェクト共通の評価ランタイムを切り出すことを検討。`euhadra-eval` のような独立 crate にする可能性あり。

---

## 7. 参考リンク

- FLEURS: https://huggingface.co/datasets/google/fleurs
- Common Voice: https://commonvoice.mozilla.org
- ReazonSpeech: https://huggingface.co/datasets/reazon-research/reazonspeech
- LibriSpeech: https://www.openslr.org/12
- AISHELL-1: https://www.openslr.org/33
- WenetSpeech: https://wenet.org.cn/WenetSpeech/
- Buckeye: https://buckeyecorpus.osu.edu/
- TED-LIUM 3: https://www.openslr.org/51
- URGENT 2025: https://urgent-challenge.github.io/urgent2025/
