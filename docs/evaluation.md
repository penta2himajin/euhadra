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

**目的**: パイプライン全体が壊れていないことの確認 + ASR 精度・レイテンシの trend tracking。

**構成 (2 ジョブ並列)**:

1. **`evaluate-asr` ジョブ — ASR live スモーク**
   - FLEURS の en / ja / zh から各 10 発話を実音声で whisper-tiny に通す
   - 計測値: **WER (en) / CER (ja, zh)**、**E2E ユーザー知覚レイテンシ p50/p95**、**ASR 段レイテンシ p50/p95**
   - 期待実行時間: 3–5 分 (whisper.cpp ビルド + tiny モデル + FLEURS subset を CI キャッシュ)
2. **`evaluate-fast` ジョブ — 層 ablation + 各層レイテンシ μ-benchmark** (本 PR ではまだ未実装、後続 PR で追加)
   - Mock ASR で transcript 注入、Tier 1+2 の各層を on/off で 200 件 × 3 言語走らせる
   - 期待実行時間: 30 秒以内

**データ**:
- FLEURS の en / ja / zh 各 10 発話程度のサブセット
- 軽量 (合計 ~30 MB のオーディオ)、CC-BY 4.0、HuggingFace `google/fleurs` から取得
- ダウンロードスクリプト: `scripts/download_fleurs_subset.py`

**メトリクス**:
- ASR レベル: WER (en) / CER (ja, zh)
- パイプライン全体: 例外なく完走するか (smoke 確認) + E2E latency
- 各 ASR 段のレイテンシ
- 後続 PR で追加: 各 Tier 1+2 層の ablation ΔWER + 個別レイテンシ

**回帰検出**: `docs/benchmarks/ci_baseline.json` を ground truth として保存し、PR ごとに比較。閾値超過で fail (詳細は §3.8)。基準値の意図的更新は同 PR 内で `ci_baseline.json` を書き換え。

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

なお、L2 にはクリーン条件版 (上記) と **ロバスト条件版 (L2-Robust)** が存在する。L2-Robust の詳細は §4 を参照。

### 1.3 L3 — euhadra 固有検証 (Tier 1+2 評価)

**目的**: ASR 後処理 (filler 除去・自己訂正検出・句読点・段落分割・固有名詞補正) が実際に効果を出しているかの定量検証。汎用 ASR ベンチマークでは測れない euhadra 独自の品質を測る。

**評価方法**: §3.0 の **2 軸設計** に従い、各レイヤーを **直接評価 (F1)** と **Ablation 評価 (ΔWER/CER)** の両方で測定する。

**データ**:

| 軸 | 検証対象 | データセット | 言語 |
|----|---------|-------------|------|
| 直接 (F1) | filler / phonological process / self-correction | Buckeye | en |
| 直接 (F1) | filler / 自己訂正 | CSJ (有償または学術無償)、CEJC (NINJAL、free edition + 有償版) | ja |
| 直接 (F1) | filler | MagicData-RAMC (OpenSLR 123、無償)、CS2W (text-only、CC-BY-SA) | zh |
| 直接 (F1) | 段落分割 | TED-LIUM 3 (CC-BY-NC-ND、OSS のみ) | en |
| Ablation (ΔWER) | 各層の最終出力寄与 | §3.7 の E2E 推奨データを流用 | en / ja / zh |

**期待実行時間**: 半日〜1 日

**メトリクス**: §3 のレイヤー別マッピングを参照。専用評価指標 (filler 除去 F1、自己訂正検出 F1、段落境界 P/R、各層の ablation ΔWER/CER) を実装する必要がある。

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
| CSJ (日本語話し言葉コーパス) | 講演・対話 660h、`(F)` filler / `(D)` reparandum / `(D2)` 置換タグ + UniDic POS「感動詞-フィラー」、Core 500K words に係り受け+節境界 | NII / NINJAL 経由 (学術無償申請可、商用は有償 — §5.5) | **L3 直接 F1 のゴールドスタンダード** (filler / self-correction 両用) |
| **CEJC (日常会話コーパス)** | NINJAL、200h・1,675 話者の **日常会話** (講演ではない)、`F`/`D` 依存ラベル、UD Japanese-CEJC、2022 年公開 | **NINJAL Corpus Portal: free edition + 有償フル版** | **L3 直接 F1**、CSJ より dictation の現実分布に近い |
| JSUT | クリーン読み上げ (1 名話者) | CC-BY-SA 4.0 | TTS 寄り、ASR 補助 |
| Common Voice ja | クラウドソース | CC0 1.0 | L2、ネイティブ性に注意 |
| FLEURS ja | 読み上げ | CC-BY 4.0 | L1 |

**ReazonSpeech 用途制限**: 著作権法第 30 条の 4 に基づき機械学習目的のみ。**商用提供物 (Cochlis のリリース成果物) には絶対に同梱しない**。評価レポートでスコアを公表する際も、データ取得経路と用途を明記する。

**CSJ vs CEJC の使い分け**:
- **CSJ**: 講演・模擬講演中心、filler 密度高め、disfluency 注釈が体系的。**長文・準フォーマル発話の評価**に適する
- **CEJC**: 日常会話中心、自然な repair / overlap / fillers、CSJ より新しく自由度の高いライセンス階層。**日常 dictation シーンの評価**に近い
- 両方利用可能なら CEJC を主、CSJ を補助 (歴史的比較用) に使う構成を推奨

### 2.3 英語

| データセット | 内容 | ライセンス | 用途 |
|-------------|-----|-----------|------|
| LibriSpeech | オーディオブック読み上げ | CC-BY 4.0 | L2 標準ベースライン |
| TED-LIUM 3 | TED トーク | CC-BY-NC-ND 3.0 | **非商用のみ**、L3 段落分割 |
| Switchboard | 電話会話、filler 多 | LDC 有償 | L3 (入手できれば) |
| Buckeye | 自然会話、IPA 音素アノテ | 研究用無償 | L3、Speciphonorm と共用 |
| AMI Meeting | 会議 | CC-BY 4.0 | 重なり発話・ノイズ評価 |
| CHiME-6/7/8 | 家庭・遠距離 mic、雑音 | 多くは無償 | §4 ロバスト性評価 |
| GigaSpeech | マルチドメイン 10000h | Apache-2.0 | 大規模、領域多様性 |
| Common Voice en | クラウドソース | CC0 1.0 | アクセント多様性 |

**TED-LIUM 3 注意**: CC-BY-NC-ND は非営利かつ改変禁止。Cochlis 商用ベンチマーク用途には不可。OSS 評価レポートのみ。

**Buckeye の戦略的価値**: euhadra (Tier 1+2) と Speciphonorm (phonological normalization) の両研究で使えるため、**共通評価ハーネス**を `tests/evaluation/buckeye_runner.rs` (仮) に集約する設計を推奨。

### 2.4 中国語

| データセット | 内容 | ライセンス | 用途 |
|-------------|-----|-----------|------|
| AISHELL-1 | クリーン読み上げ | Apache-2.0 | L2 標準ベースライン |
| AISHELL-2 | 拡大版 1000h | Apache-2.0 | 大規模ベースライン |
| WenetSpeech | YouTube + Podcast、自然発話 | CC-BY 4.0 | L3 自然発話、ablation |
| **MagicData-RAMC** | Magic Data Tech (2022)、180h・663 話者の Mandarin 会話、**表層 disfluency マークアップ** (partial words / repetitions / hesitations)、train/dev/test split あり | **OpenSLR 123、無償、商用フレンドリー** (MagicHub 商用ライセンスあり) | **L3 直接 F1 (filler) の zh 主データ** |
| **CS2W (Chinese Spoken-to-Written)** | EMNLP 2023、7,237 文の **spoken→written 並列テキスト**。disfluency 除去、文法エラー、ASR エラー、口語表現の 4 種を含む。**音声なし** | **CC-BY-SA 4.0、GitHub 公開** | **Tier 1 のテキストレベル F1 単体テスト** (音声不要、CI に乗せやすい) |
| HKUST | Mandarin 電話会話 | LDC 有償 | 会話、L3 候補 |
| AISHELL-4 (会議) | 211 セッション 118h、`[laugh]/[cough]` 等の非言語マーカー、**専用 disfluency タグなし** | 無償 (再配布要許可) | 会議 ASR、disfluency 用途は限定的 |
| KeSpeech | 8 方言 + Mandarin | 学術利用 | 方言頑健性 |
| FLEURS zh | 読み上げ | CC-BY 4.0 | L1 |

**Disfluency データセット状況の更新 (2026 Q2)**: 当初「揃った disfluency 注釈コーパスは事実上存在しない」と書いていたが、**MagicData-RAMC (2022 公開)** が表層 disfluency マークアップ済みの 180h 会話コーパスを無償 + 商用フレンドリーに提供しているため、**Tier 1 (filler 除去) の zh 直接 F1 評価はこれが第一選択**となる。

**ただし、中国語には Switchboard NXT 様の reparandum/repair 構造アノテを持つコーパスは依然存在しない**。Tier 2 (自己訂正検出) の zh 直接評価には、**MagicData-RAMC のクリーンな transcript の上に span を貼る自前アノテ** が必要 (§5.6 参照)。これは WenetSpeech `test-meeting` の生音声から始めるより遥かに低コスト。

**CS2W の使い所**: Tier 1 の **テキストレベル単体テスト** に最適。音声を回さずに「filler 除去ロジックが spoken→written 並列の差分を再現できるか」を測れる。CI で `cargo test` から呼び出せる軽量チェックとして組み込み可能。

---

## 3. レイヤー別マッピング

`docs/spec.md` §3 で定義した各レイヤーに対して、評価方法と推奨データを示す。

### 3.0 評価方法の 2 軸設計

各レイヤー (§3.2〜§3.5) は **直接評価 (F1)** と **Ablation 評価 (ΔWER/CER)** の **2 軸** で評価する。両者は冗長ではなく異なる質問に答えるため、**組み合わせて運用する**。

**直接評価 (F1)**: ground truth annotation 付きデータを使い、**層単体のロジック正確性** を測る。例: Buckeye 上で filler removal の precision / recall / F1 を計算。Layer-specific corpus が必要なため、データ取得コストが高い言語/層が存在する。

**Ablation 評価 (ΔWER/CER)**: §3.7 の E2E データに対して、対象層を **on / off** で走らせ、最終 WER/CER の差分で **「層が最終出力にどれだけ寄与するか」** を測る。Ground truth は ASR 用 reference transcript のみで足りるため、F1 用の専用 annotation は不要。

**併用する理由 — F1 と ΔWER の乖離が診断シグナル**:

| パターン | 解釈 |
|---|---|
| F1 高 + ΔWER 改善 | 層は正しく動き、最終出力にも寄与している (理想形) |
| F1 高 + ΔWER ほぼゼロ | 層は正しく動くが最終出力に効いていない (実装過剰、削除候補) |
| F1 低 + ΔWER 改善 | 偶然の効果 — 検出精度が低くても結果として ASR エラーを打ち消している (脆弱、要調査) |
| F1 低 + ΔWER 悪化 | 層が壊れている、または前段との相互作用で品質を落としている (緊急対応) |

どちらか一方では捕まえられない問題 (例: 内容語を filler と誤判定して消したが偶然 WER が改善する) が、両軸を見ることで検出できる。

**両軸のデータ対応**:

| 軸 | データ要件 | 推奨データ (en / ja / zh) |
|---|---|---|
| 直接評価 (F1) | 層固有の annotation 付き | en: Buckeye / ja: CSJ・CEJC / zh: MagicData-RAMC・CS2W |
| Ablation (ΔWER) | ASR 用 reference transcript | §3.7 の E2E 推奨データ (FLEURS / TED-LIUM 3 / ReazonSpeech / WenetSpeech-meeting) |

**読み上げ音声で ablation が空打ちになる注意**: FLEURS / LibriSpeech / AISHELL-1 のような **読み上げ音声には filler や自己訂正がほぼ含まれない**。Tier 1 / Tier 2 self-correction の ablation で意味のある ΔWER を観測するには、自然発話を含むデータ (TED-LIUM 3 講演 / ReazonSpeech TV / WenetSpeech meeting) を使う必要がある。読み上げ音声での ablation は smoke 確認 (例外なく完走するか) にしか使えない。

§3.2〜§3.5 では各層について両軸の **メトリクス・ground truth・推奨データ** を別セクションで示す。§3.1 (ASR Adapter) と §3.7 (E2E) は性質上、直接評価のみ。§3.4 (PhonemeCorrector) は **専用 F1 corpus が存在しないため ablation 主体**。§3.6 (LlmRefiner) は MVP では評価対象外。

### 3.1 ASR Adapter (`AsrAdapter` trait)

**評価対象**: WhisperLocal、ParakeetAdapter、(将来) Whisper Cloud

**メトリクス**:
- WER (en) / CER (ja, zh)
- RTF (Real-Time Factor)
- レイテンシ (initial token latency / final transcript latency)

**推奨データ**:
- L1: FLEURS en/ja/zh
- L2: LibriSpeech test-other / ReazonSpeech-test / AISHELL-1 test
- ノイズ耐性: §4 を参照

### 3.2 Tier 1 — TextFilter (filler removal)

**評価対象**: SimpleFillerFilter、JapaneseFillerFilter、OnnxEmbeddingFilter

#### 直接評価 (F1)

**メトリクス**:
- Filler removal **F1** (precision / recall を別途報告)
- False positive rate (内容語を filler と誤判定する率)
- 処理レイテンシ

**Ground truth の作り方**:
- Buckeye: filler annotation を直接利用 (en、研究無償)
- CSJ: 短単位品詞 = `感動詞-フィラー` + `(F)` / `(D)` タグから機械抽出 (ja)
- CEJC: F / D 依存ラベルから抽出 (ja、free edition でも一部利用可)
- MagicData-RAMC: 表層 disfluency マークアップを利用 (zh、無償、OpenSLR 123)
- CS2W: spoken→written 並列の disfluency 除去差分をテキストレベル F1 に変換 (zh、音声不要)
- ReazonSpeech / WenetSpeech-meeting: 自前で軽量アノテーション (100–500 発話、§5.6 でコスト見積)

**推奨データ**:
- en: Buckeye (free)、Switchboard NXT (LDC 有償、入手可能なら追加)
- ja: **CEJC (free academic edition)** または **CSJ (有償または学術無償)**、補助に ReazonSpeech サブセット + 自前アノテ
- zh: **MagicData-RAMC test (無償、OpenSLR 123)**、**CS2W (text-only、CI 単体テスト用)**、補助に WenetSpeech-meeting + 自前アノテ

#### Ablation 評価 (ΔWER/CER)

**メトリクス**: filter on / off の最終 WER/CER 差分。Filter on 時の WER 改善 (ΔWER < 0) が期待値。改善幅が直接評価の F1 と一致しない場合は §3.0 のパターン表で診断。

**Ground truth**: ASR 用 reference transcript のみ。専用 annotation 不要。

**推奨データ**: §3.7 の E2E 推奨データから **自然発話を含むものを優先**。読み上げ音声 (FLEURS 等) では filler が存在しないため空打ちになる。
- en: TED-LIUM 3 (CC-BY-NC-ND、OSS 評価のみ)
- ja: ReazonSpeech-test (§30-4 用途制限)
- zh: WenetSpeech `test-meeting` (CC-BY 4.0)

### 3.3 Tier 2 — SelfCorrectionDetector

**評価対象**: SelfCorrectionDetector (英語/日本語ルール)

#### 直接評価 (F1)

**メトリクス**:
- Self-correction detection **F1** (reparandum span 検出の precision / recall)
- Reparandum 削除の正確性 (削除後テキストが意図通りか — span boundary の評価)

**Ground truth**:
- Switchboard NXT (en): disfluency annotation の標準。LDC 有償
- CSJ (ja): repair 構造のアノテーション (`(D)` / `(D2)`)
- CEJC (ja): F / D 依存ラベル (CSJ より新しく日常会話寄り)
- 中国語: **Switchboard NXT 様の reparandum/repair 構造アノテを持つコーパスは存在しない**。MagicData-RAMC のクリーンな transcript 上に span を貼る **自前アノテが必須** (§5.6)

**推奨データ**:
- en: Switchboard NXT (LDC 有償、入手可能なら)、または en TED-LIUM 3 への自前 span アノテ
- ja: CEJC (主)、CSJ (補助、長文・準フォーマル比較用)
- zh: **MagicData-RAMC サブセット + 自前 span アノテ** (~5h、§5.6 で ¥100–200k 見積)

**Inter-annotator agreement の留意点**: Reparandum span boundary は低 IAA タスク (英 Switchboard でも κ ≈ 0.55–0.75)。ja / zh で自前アノテする場合は **二重アノテ + 調停** を前提に設計し、単一アノテ単独の F1 を信頼しすぎない。

#### Ablation 評価 (ΔWER/CER)

**メトリクス**: detector on / off の最終 WER/CER 差分。

**Ground truth**: ASR 用 reference transcript のみ。

**推奨データ**: §3.7 E2E データ。**自然発話必須** (読み上げには自己訂正がない):
- en: TED-LIUM 3 (講演中の言い直しを含む)
- ja: ReazonSpeech-test
- zh: WenetSpeech `test-meeting`

### 3.4 Tier 2 — PhonemeCorrector

**評価対象**: PhonemeCorrector (CMUdict + G2P + 複合スコア)

**評価軸**: 専用 F1 corpus が存在しないため、**Ablation 主体** (§3.0 の 2 軸のうち F1 軸はスキップ)。「dictionary を渡した / 渡さない」での ΔWER がそのまま効果指標。

**メトリクス**:
- 固有名詞・技術用語の正確な認識率 (with-dict と without-dict の WER 差分)
- False correction rate (本来正しい語を誤って置換する率 — without-dict より with-dict の WER が悪化したケース)

**評価設計**:
- TED-LIUM の科学技術トークから固有名詞リストを抽出 → custom dictionary として注入
- 同じ音声に対して dictionary を渡した場合と渡さない場合の WER 差分を測定
- 日本語: ReazonSpeech のニュースセグメント (固有名詞多)、JNAS (新聞読み上げ)
- 中国語: WenetSpeech の人名・地名・専門用語を含むセグメント

### 3.5 Tier 2 — ParagraphSplitter

**評価対象**: ParagraphSplitter (埋め込み距離 + 最大文数制約)

#### 直接評価 (F1)

**メトリクス**:
- Paragraph boundary **F1** (人手アノテーションとの一致)
- WindowDiff、Pk (segmentation 専用メトリクス)

**Ground truth**: 各データセットの **既存の段落構造** (講演スクリプト・放送台本) を ground truth として流用。F1 用専用アノテは不要なケースが多い。

**推奨データ**:
- en: TED-LIUM 3 (講演スクリプトの段落構造)
- ja: CSJ 講演、CEJC、ReazonSpeech のニュース番組 (放送台本に段落あり)
- zh: WenetSpeech `test-meeting`

#### Ablation 評価 (ΔWER/CER)

ParagraphSplitter は文字列の最終 WER に直接影響しない (改行の追加のみ) ため、典型的な ΔWER 指標は使えない。代わりに以下を測る:
- **読みやすさプロキシ**: 段落区切り後の **段落あたりの文数分布** (理想分布との KL 距離)
- **下流タスク影響**: LlmRefiner を後続させた場合の出力品質変化 (Tier 3 評価との合流点)

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

### 3.8 レイテンシ評価とバジェット

「処理レイテンシ」は §3.1〜§3.7 の各層で個別に測るほか、**E2E ユーザー知覚レイテンシ** = 発話終了から出力確定までの合計時間 を主指標とする。**処理レイテンシ ≠ テスト実行時間** に注意 (§1.x の「期待実行時間」はテストの wall clock、レイテンシは pipeline 内部の各段階の処理時間)。

**バジェット (target、Phase 1 MVP、whisper-tiny + rule-based 構成、Linux CPU 想定)**:

| 段 | p50 target | p95 target | 備考 |
|----|-----------|-----------|------|
| ASR (whisper-tiny) | 1× audio (RTF 1.0 以下) | 1.5× | 5 秒音声で p50 ≤ 5 秒 |
| Tier 1 filter | < 1 ms | < 5 ms | rule-based、ONNX は cold start 別計測 |
| Tier 2 SelfCorrection | < 1 ms | < 5 ms | rule-based |
| Tier 2 PhonemeCorrector | < 50 ms | < 100 ms | dictionary サイズ依存 |
| Tier 2 ParagraphSplitter | < 5 ms | < 20 ms | rule-based、ONNX embedder は別 |
| Tier 3 LlmRefiner | (Phase 1 範囲外) | — | MVP では Mock のみ計測 |
| **E2E** | **音声長 + 1 秒** | **音声長 + 2 秒** | post-ASR は 1 秒以内が目安 |

**測定方法論**:
- **環境**: Linux x86_64 CPU、GitHub-hosted runner (`ubuntu-latest`、2-core)
- **percentile**: median (p50) と p95 の両方を記録。p50 は安定性、p95 は worst-case
- **サンプル数**:
  - ASR live (FLEURS subset): 各言語 10 発話 → ASR latency 30 サンプル
  - 層 μ-benchmark (後続 PR): 各層 100 calls × 言語 → 300 サンプル
- **ウォームアップ**: μ-benchmark は 10 回捨て + 100 回計測 (criterion 流)。ASR live はウォームアップなし (cold start も含めた実測値)
- **モデルロード時間**: ONNX/whisper モデルは「セッション開始時に発生する一回限りのコスト」として E2E latency とは別計測

**回帰検出ポリシー** (`docs/benchmarks/ci_baseline.json` を baseline として):

| メトリクス | warning | hard fail |
|---|---|---|
| WER / CER (絶対値) | > baseline + 5% (絶対) | > baseline + 10% |
| WER / CER (相対) | regression > 20% | > 50% |
| 各層 latency p50 | regression > 50% | > 100% (2× 遅い) |
| E2E latency p50 | regression > 25% | > 50% |
| ASR live スモーク完走 | — | 例外で落ちたら即 fail |

**ノイズ吸収のため**:
- 30 発話程度では WER stderr ~5%、絶対閾値判定は粗いため warning のみで運用
- latency は **rolling N コミット (例: 直近 5 PR) の中央値** を baseline とすると spike 耐性が上がる (本 PR ではまだ実装せず、将来検討)
- runner ノイズが疑われる失敗は再実行して再現確認

**`ci_baseline.json` の schema**:

```json
{
  "schema_version": 1,
  "generated": "2026-04-25T...",
  "asr_model": "ggml-tiny.en.bin / ggml-tiny.bin",
  "languages": {
    "en": {
      "samples": 10,
      "wer": 0.20,
      "cer": null,
      "asr_latency_ms":  {"p50": 800, "p95": 1100},
      "e2e_latency_ms":  {"p50": 850, "p95": 1200}
    },
    "ja": { "samples": 10, "wer": null, "cer": 0.30, "...": "..." },
    "zh": { "samples": 10, "wer": null, "cer": 0.28, "...": "..." }
  },
  "tolerances": {
    "wer_absolute_warn":   0.05,
    "wer_absolute_fail":   0.10,
    "wer_relative_warn":   0.20,
    "wer_relative_fail":   0.50,
    "latency_p50_relative_warn": 0.50,
    "latency_p50_relative_fail": 1.00,
    "e2e_latency_p50_relative_warn": 0.25,
    "e2e_latency_p50_relative_fail": 0.50
  }
}
```

---

## 4. ロバスト性評価 (ノイズ・残響加算)

クリーンな読み上げ音声での WER/CER だけでは、euhadra が想定する実環境 (オフィス、自宅、カフェ、移動中) での品質を反映できない。クリーン音声に **実環境ノイズと残響を加算** したテストデータを使い、SNR 別の劣化曲線を測定する。

### 4.1 目的

- 実環境ノイズ下での ASR Adapter 性能の定量化
- AudioPreprocessor (将来導入) 導入前後の効果測定基準
- Cochlis のターゲット環境 (静かなオフィス〜騒がしいカフェ) で動作することの確認
- Whisper の hallucination が SNR 低下とともにどう増えるかの可視化

### 4.2 ノイズデータセット

| データセット | サイズ | カバレッジ | ライセンス | 推奨度 |
|-------------|-------|-----------|-----------|-------|
| **MUSAN** | ~11 GB | 12言語 speech、複数ジャンル music、技術系/非技術系 noise | CC-BY 4.0 / CC0 (一部) | **★★★** デフォルト |
| **DEMAND** | 数 GB | 15 環境 × 16ch (カフェ、駅、車内、オフィス、公園 等) | CC-BY-SA 3.0 | ★★ ShareAlike 注意 |
| **DNS Challenge noise** | ~58 GB | Audioset 150 class × 60K clips + Freesound + DEMAND ~10K、計 181h、VoIP 寄り | 元データのライセンス継承 | ★★ 大規模だが要確認 |
| **QUT-NOISE** | 100 分 | カフェ、車内、リビング、ストリート、リバーサイド | non-commercial research | △ 商用 NG |
| **Audioset (filtered)** | 巨大 | ~200万 10秒クリップ、~600 audio events | YouTube 依存、ラベルは CC-BY | △ 商用 NG (YouTube 規約) |
| **Freesound (FSD系)** | 巨大 | コミュニティ提供、多様 | CC 各種 (混在) | △ ライセンス確認必須 |
| **NOISEX-92** | 15min × 15種 | factory, tank, jet, babble 等 (古典) | 研究用無償 | △ 古典、限定的 |

**MUSAN がデフォルト推奨**: CC ライセンス、Kaldi/ESPnet 等のレシピで標準採用、サイズ適度、商用ベンチマークまで通せる。標準的な augmentation 設定は **SNR 平均 12 dB / 分散 8 dB の Gaussian sampling** (Multi-condition training の慣例)。

### 4.3 RIR (残響シミュレーション)

| データセット | 内容 | ライセンス |
|-------------|-----|-----------|
| **OpenSLR SLR26** | RIRs and noises、~2.1 GB、DNS Challenge にも統合 | Apache-2.0 系 |
| **OpenSLR SLR28** | ~2.3 GB、small/medium/large room の RIR | Apache-2.0 系 |
| **DNS Challenge RIR** | 3,076 real + ~115,000 synthetic | Microsoft、要確認 |
| **AIR database** | 実測 RIR (オフィス、教会、車内) | RWTH Aachen、研究用 |
| **BUT ReverbDB** | 多様な実環境 | Brno、研究用 |
| **REVERB Challenge** | 模擬 + 実測 | 公開 |

**Synthetic RIR の自前生成**: `pyroomacoustics`、RIR-Generator、gpuRIR などで自前生成可能。room dimension と T60 をパラメータで振れるので、実 RIR より制御しやすい。再現性のあるテストには synthetic RIR が向く。

### 4.4 ミックス済みベンチマーク

評価のためだけなら、ノイズ加算済みの公式ベンチマークを使う方が楽。

| データセット | 内容 | 用途 |
|-------------|-----|------|
| **VoiceBank + DEMAND** | clean (VCTK) + DEMAND noise 既ミックス、SNR {15, 10, 5, 0 dB} | speech enhancement の標準 |
| **CHiME-3/4** | 4ch、4 環境 (cafe, bus, pedestrian, street) | distant ASR |
| **CHiME-6/7/8** | 家庭内、distant mic | 家庭環境 ASR |
| **WHAM!** | WSJ-mix + ambient noise | cocktail party (将来検討) |
| **DNS Challenge dev/blind test** | リアル + simulated 混在、speaker variety、device variety、T60 統制 | 包括評価 |
| **URGENT 2024/2025** | ~1300h speech + ~250h noise + RIR、7 種の歪み (additive noise, reverberation, clipping, bandwidth extension, codec artifacts, packet loss, wind noise) | 多言語 universal SE |

### 4.5 シミュレーションツール

ノイズ加算自体は単純な信号処理だが、再現性を持つパイプラインが既に存在する。euhadra でゼロから書く必要はない。

- **ESPnet** の noise recipe — Kaldi 系の標準 mix script。SNR sampling、RIR convolution、speed perturb 全部入り
- **DNS Challenge `noisyspeech_synthesizer.py`** — manifest 形式で再現可能な mix 生成
- **URGENT 2024/2025** — `meta.tsv` で simulation parameter を記述、parallel 実行可能、`generate_data_param.py` + `simulate_data_from_param.py` の 2 段構成
- **audiomentations / torch_audiomentations** — Python の augmentation library。RIR convolution / additive noise / reverb / pitch shift 等を pipeline 化
- **pyroomacoustics** — RIR の物理シミュレーション (image source method, ray tracing)

### 4.6 評価設計 (L2-Robust)

L2 標準ベンチマークを以下の 4 条件 × 4 SNR ステップで拡張する:

| 条件 | 加工内容 |
|-----|---------|
| `baseline` | クリーン音声そのまま (§1.2 と同じ) |
| `+noise` | baseline × MUSAN noise (SNR={20, 10, 5, 0} dB) |
| `+reverb` | baseline × OpenSLR SLR26 RIR (small / medium / large room) |
| `+noise+reverb` | baseline × MUSAN × RIR (フル現実条件) |

**SNR 設定の根拠** (Cochlis 想定環境):

| SNR | 想定環境 |
|----|---------|
| 20 dB | 静かなオフィス、自室 |
| 10 dB | 家庭、図書館、静かなカフェ |
| 5 dB | 騒がしいカフェ、移動中 |
| 0 dB | 騒音と音声が同レベル、実用境界 |

**再現性のための制約**:
- ノイズサンプルと RIR の選択は固定 seed で決定
- mix manifest を `tests/evaluation/manifests/l2_robust.tsv` に commit (ノイズ・RIR ファイルそのものは commit しない、選択 ID のみ)
- 評価ごとに manifest を再生成しない (regression 比較を可能にする)

### 4.7 メトリクス

**downstream metric (主)**:
- WER / CER
- 各 SNR ステップでの WER/CER 劣化曲線 (clean を基準にした relative degradation)

**音質 metric (補助、AudioPreprocessor 評価時)**:
- **DNSMOS** (non-intrusive、reference 不要、リアル録音にも適用可)
- **PESQ** (intrusive、reference 必要、speech enhancement 標準)
- **STOI** (intrusive、了解度評価)

DNS Challenge / URGENT で標準的に使われる音質 metric。WER/CER だけでは「ASR が落ちなくても聞き取り辛い」状況を捉えられないため、AudioPreprocessor を入れる場合は両方測る。

### 4.8 ライセンス・商用利用の整理

主要データセットの商用評価可否:

| データセット | OSS 評価 | 商用ベンチマーク (Cochlis) |
|-------------|---------|------------------------|
| MUSAN | ✓ | ✓ (CC-BY 4.0) |
| DEMAND | ✓ | △ ShareAlike が成果物に波及する可能性、要法務確認 |
| DNS Challenge | ✓ | △ "AS IS" basis、各サブデータの元ライセンス継承 |
| Audioset | ✓ | △ YouTube 規約依存 |
| OpenSLR SLR26/28 (RIR) | ✓ | ✓ |
| URGENT 2024/2025 | ✓ | △ 一部 WSJ 含むため LDC 経由必要 |
| CHiME | ✓ | △ 元データに有償部分あり |

**商用ベンチマークの推奨組合せ**: **MUSAN noise + OpenSLR SLR26/28 RIR + 自前 pyroomacoustics シミュレーション** が、Cochlis のマーケティング素材まで通せる。

DEMAND と DNS Challenge は OSS 評価レポートでは使えるが、Cochlis 公式数値には使わない方針。

---

## 5. ライセンス・使用上の注意

### 5.1 商用提供物 (Cochlis / talkadict) との分離

以下のデータセットは **商用ベンチマークでの公表 NG**:
- VoxPopuli (CC-BY-NC)
- TED-LIUM 3 (CC-BY-NC-ND)
- ReazonSpeech (用途制限)
- DEMAND、Audioset、QUT-NOISE 等のノイズ系の一部

OSS 評価レポート (GitHub README、論文) では使用可だが、商用ドキュメント・マーケティング素材には流用しない。

### 5.2 評価コードと評価データの分離

- 評価コードは `tests/evaluation/` 配下にチェックイン
- データセットは **リポジトリにコミットしない**。ダウンロードスクリプト (`scripts/download_eval_data.sh`) のみ提供
- HuggingFace datasets 経由で取得できるものは `datasets.load_dataset(...)` を使う
- LDC 等の有償データは、ユーザー側で別途取得 (CSJ、Switchboard、HKUST)
- §4.6 の mix manifest は ID のみ commit、音声は commit しない

### 5.3 ReazonSpeech の表記

ReazonSpeech を使った評価結果を公開する際は以下を明記:
- データ取得経路 (HuggingFace `reazon-research/reazonspeech`)
- 利用目的 (ASR モデル評価、機械学習研究)
- 著作権法第 30 条の 4 に基づく利用であること

### 5.4 評価レポートの公開ポリシー

- L1 / L2 のスコアは README.md の Quality Status セクションに掲載 (CC-BY のもののみ)
- L2-Robust は商用 OK の組合せ (MUSAN + OpenSLR RIR) のみ Cochlis 数値として公表
- L3 の Tier 1+2 評価結果は別途研究レポート (talkadict 論文等) としてまとめる
- 競合製品 (Aqua Voice、Wispr Flow 等) との比較は、利用規約上問題ない範囲で行う

### 5.5 商用ライセンスの価格 (2026 年現在、要再確認)

L3 直接評価で商用利用も視野に入れる場合の価格情報。**契約締結時には公式に最新料金を確認すること** (本表は計画用の参考値):

| データセット | 商用条件 | 概算価格 |
|---|---|---|
| CSJ | NINJAL 商用ライセンス | **500,000 JPY / 2 年・10 ユーザ** または **5,000,000 JPY / 永続・10 ユーザ** |
| CEJC | NINJAL Corpus Portal | free edition + 有償フル版 (商用条件は NINJAL に直接問い合わせ) |
| Switchboard NXT | LDC | 非メンバー ~$2,500 / メンバー無償 |
| HKUST/MTS | LDC | 非メンバー ~$2,500 / メンバー無償 |
| CALLHOME Mandarin | LDC | 非メンバー ~$1,500 / メンバー無償 |
| Datatang / SpeechOcean カスタムアノテ | quote | $0.5–$2 / 音声分 + $5,000–$20,000 セットアップ |

**CEJC を主、CSJ を補助** とする方針 (§2.2) は、CSJ 商用ライセンスの 50 万円 / 2 年というコストを Cochlis 商用化のタイミングまで遅延できる利点がある。

### 5.6 自前アノテーションのコスト見積とガイドライン

§3.2 / §3.3 で「自前アノテ前提」となるケース (主に ja / zh の Tier 2 self-correction span) のコスト見積:

**転記 + ラベリングの単価**:
- プロ転記 (verbatim、filler 残し): en $1.50–$3 / 分 + $0.25–$0.50 / 分追加、ja/zh は $3–$6 / 分
- 既存 transcript への span 注釈のみ: 0.3–1× real-time、訓練済アノテータで時給換算 ~¥3,000

**euhadra プロジェクト想定の budget**:

| タスク | 音声時間 | アノテータ | 概算コスト |
|---|---|---|---|
| ja filler F1 (ReazonSpeech サブセット) | 5h (~1,500 発話) | 2 名 (IAA 用) | ¥60–120k または社内 2 人週 |
| ja self-correction F1 | 5h | 2 名 | ¥80–150k (span 揺れで遅い) |
| zh filler F1 (MagicData-RAMC サブセット) | 5h | 2 名 | 同程度 (transcript 既存、span のみ) |
| zh self-correction F1 | 5h | 2 名 | ¥100–200k |

**Inter-annotator agreement の期待値** (文献ベース):
- Filler 同定 (closed class、binary content/filler): Cohen's κ **0.80–0.95** (1 時間のキャリブレーションあり)
- Reparandum span boundary: κ **0.55–0.75** ← **二重アノテ + 調停を前提に設計**

**再利用可能なアノテーションガイドライン**:
- Switchboard Disfluency Annotation Stylebook (Meteer et al. 1995) — reparandum / interregnum / repair 構造の定義、言語横断で流用可
- CSJ 転記マニュアル (NINJAL、無償公開) — ja の `(F)` / `(D)` / `(D2)` 慣例
- CEJC 転記ガイドライン (NINJAL、無償公開) — 日常会話寄り
- CS2W アノテーション規約 (CC-BY-SA、GitHub) — zh の spoken→written 規約

**推奨アプローチ**: Shriberg / Meteer の reparandum スキームを言語横断の構造的バックボーンとし、CSJ の `(F)` / `(D)` 閉集合を ja に、対応する zh 閉集合 (嗯 / 呃 / 啊 / 那个 / 这个 / 就是 等) を新規定義する。**音声に触れる前にガイドライン整備で半日確保**することが推奨。

#### 5.6.1 アノテーションスキーマと commit ポリシー

**入力**: 各コーパスの (utterance_id, text) ペア。アノテータは音声を録らず、転記もしない。**transcript の上にラベルを貼る** だけ。

**出力スキーマ** (JSON-Lines、1 発話 / 行):

Filler annotation 例 (zh):
```json
{
  "utterance_id": "magicdata_ramc_train_001234_seg_5",
  "text": "嗯 我觉得这个事情应该这样处理",
  "fillers": [
    {"start": 0, "end": 1, "label": "嗯"},
    {"start": 5, "end": 7, "label": "这个"}
  ]
}
```

Self-correction (reparandum/repair) 例 (zh):
```json
{
  "utterance_id": "magicdata_ramc_train_005678_seg_2",
  "text": "我想去波士顿啊丹佛工作",
  "repairs": [
    {
      "reparandum":  {"start": 3, "end": 6},
      "interregnum": {"start": 6, "end": 7},
      "repair":      {"start": 7, "end": 9},
      "type": "substitution"
    }
  ]
}
```

`repair.type` の closed set: `substitution` / `insertion` / `deletion` / `abandoned`。

**リポジトリ commit ポリシー**:

| 物件 | リポジトリ commit | 理由 |
|---|---|---|
| 元コーパスの音声 / transcript | **NG** | 元ライセンス + サイズ。`scripts/download_eval_data.sh` 経由で取得 |
| アノテーション JSON-Lines | **OK** | 自社制作物、軽量、`utterance_id` で元データを参照 |
| アノテーションガイドライン .md | **OK** | 自社制作物、再現性に必須 |
| キャリブレーション結果 (κ 値、調停ログ) | **OK** | 評価レポート用 |
| LLM 下書きの生出力 (検証前) | **NG (任意)** | 中間生成物、価値低い |

つまり Phase C-3 で `tests/evaluation/annotations/{ja_filler,ja_self_correction,zh_filler,zh_self_correction}.jsonl` + `tests/evaluation/annotations/guidelines/*.md` のみ commit する。

#### 5.6.2 LLM 補助による自動化 (hybrid パイプライン)

純人手アノテのコストを下げるため、LLM を **下書き生成器** として使う。**ただし純 LLM 出力を ground truth にしてはならない** — 評価対象 (filler 検出器・自己訂正検出器) と LLM が同じ判断を下すなら測定が循環する。

**評価循環の回避ルール**:
- パイプライン側で LLM refiner を使う場合、**アノテ用 LLM は別系統** (e.g., パイプライン Claude → アノテ Gemini / Qwen)
- LLM 単独で打ったラベルは **silver standard** であって gold ではない
- 必ず **gold subset (全体の 5–10%)** を純人手二重 + 調停で作り、LLM ↔ 人手 IAA を計算
- 系統的バイアス (e.g., LLM が `あの` を過剰検出) を毎ラウンドレポート

**Hybrid パイプライン (5 段階)**:

1. **LLM 下書き** — Claude / Gemini / Qwen で 100% 自動ラベリング。プロンプトに **closed class を必ず明示列挙**、出力は strict JSON schema 強制、span 確信度 (logprob 平均) も同時取得
2. **自動検証** — 閉集合外の文字列が `filler` ラベル、reparandum と repair が同一文字列、段落跨ぎ span などを reject
3. **人手スポットチェック** — ランダム 10–20% を抽出し人手レビュー、LLM ↔ 人手 IAA を計算
4. **昇格判定** — IAA κ ≥ 0.7 なら silver standard で進む、< 0.7 なら全件フル人手レビューに昇格
5. **gold subset 並行作成** — 最初から 5–10% を純人手二重 + 調停。LLM ↔ 人手の系統差分を継続監視

**コスト比較** (zh self-correction、5h 音声 ~1,500 発話想定):

| アプローチ | 人手作業量 | API コスト | 合計概算 |
|---|---|---|---|
| 純人手 (二重 + 調停) | ~30 人時 | 0 | ¥100–200k |
| **Hybrid (推奨)** | ~6–10 人時 | $5–15 | **¥30–50k** |
| 純 LLM (検証なし) | 0 | $5–10 | $5–10 — **ground truth として無効** |

Hybrid で 50–70% 削減。前提として gold subset の確保を必須とする。

**プロンプト設計のチェックリスト**:
- closed class を文字列リテラルで列挙 (LLM の暗黙判断に頼らない)
- reparandum/repair 境界規則を明文化 + 具体例 5–10 個 (Switchboard stylebook サマリ)
- 出力 JSON schema を厳格化 (生成失敗時はリトライ)
- `confidence` フィールドを必須にし、低確信度サンプルを優先的に人手レビューへ回せるようにする

**自動化の適性が低いタスク** (純人手 + 調停を維持):
- Reparandum span boundary の **微調整** (文字単位の境界決定) — LLM ↔ 人手 IAA がそもそも κ 0.6–0.7 程度に留まる
- ガイドライン境界ケース (e.g., `じゃない` が否定か言い直しか) — 文脈依存度が高く、LLM プロンプトでは網羅しきれない

---

## 6. 推奨テストスイート (2026 Q2 開始時点)

§3.0 の 2 軸設計に基づき、**直接評価 (F1)** と **Ablation 評価 (ΔWER)** を別フェーズで進める。優先度はデータ取得コストの低いものから。

### Phase A — L1 スモーク (即実装、CI)

§1.1 の **2 ジョブ並列構成** に沿って、段階的に実装する:

**Phase A-1 (本 PR で完了): `evaluate-asr` ジョブ — ASR live スモーク**

```
src/eval/
├── mod.rs
├── metrics.rs                       (WER / CER、Levenshtein、normalize)
├── latency.rs                       (Sample collector、p50 / p95)
└── baseline.rs                      (ci_baseline.json I/O、回帰判定)

examples/
└── eval_l1_smoke.rs                 (FLEURS subset 走らせる binary)

scripts/
├── download_fleurs_subset.py        (HuggingFace datasets 経由)
└── setup_whisper.sh                 (whisper.cpp ビルド + tiny モデル DL)

docs/benchmarks/
└── ci_baseline.json                 (各言語の WER/CER + latency baseline)
```

**Phase A-2 (後続 PR): `evaluate-fast` ジョブ — 層 ablation + per-layer latency**

各層を順に追加していく構造。Mock ASR で transcript 注入し、Tier 1+2 の各層を on/off で 200 件程度走らせて ΔWER と layer latency を取る。

```
src/eval/ablation.rs                  (層 on/off の差分計算ヘルパ)
examples/eval_ablation.rs             (or 各層別の binary)
tests/evaluation/fixtures/            (transcript fixture、CC-BY 4.0)
```

CI ジョブとしては A-1 と並列実行。両ジョブが揃って始めて L1 が完成する。

CI で毎 PR 実行。WER/CER + latency の閾値を回帰検出に使う (§3.8 参照)。Phase B 以降の **共通基盤** (`src/eval/metrics.rs` + データ download ヘルパ) を兼ねる。

### Phase B — L2 標準ベンチマーク (リリース毎、手動実行)

```
tests/evaluation/l2_benchmark/
├── librispeech_test.rs
├── reazonspeech_test.rs
├── aishell1_test.rs
└── l2_robust/
    ├── manifests/
    │   └── musan_snr_sweep.tsv     (ノイズ加算 manifest)
    ├── noise_aug.rs                 (MUSAN 加算)
    ├── reverb_aug.rs                (OpenSLR RIR convolution)
    └── snr_sweep.rs                 (4 SNR × 3 言語の sweep)
```

リリース前に手動実行、結果を `docs/benchmarks/YYYY-MM-DD.md` に記録。L2-Robust の SNR 別劣化曲線も同レポートに含める。

### Phase C-1 — L3 直接評価 (無償データのみ、Phase A 完了後)

無償または学術無償で取得できるデータのみで直接 F1 を測る。Cochlis 商用化以前から運用可能。

```
tests/evaluation/l3_direct/
├── filler_buckeye.rs               (en filler F1、Buckeye、研究無償)
├── filler_magicdata_ramc.rs        (zh filler F1、OpenSLR 123、無償)
├── filler_cs2w.rs                  (zh filler text-only F1、CC-BY-SA、CI 候補)
├── filler_cejc.rs                  (ja filler F1、CEJC free edition)
├── paragraph_tedlium.rs            (en 段落 F1、Pk / WindowDiff)
└── paragraph_cejc.rs               (ja 段落 F1)
```

CS2W は音声不要かつ軽量なため、**L1 スモークと同じ CI ジョブで回す候補**。

### Phase C-2 — L3 Ablation 評価 (E2E データ流用、Phase A 完了後)

§3.7 の E2E データに対して各層を on / off で走らせ ΔWER/CER を測る。F1 用追加 annotation 不要。

```
tests/evaluation/l3_ablation/
├── filter_ablation.rs              (filter on/off × en/ja/zh)
├── self_correction_ablation.rs     (detector on/off × en/ja/zh)
├── phoneme_ablation.rs             (with/without dict、§3.4 の主評価)
└── paragraph_proxy.rs              (段落あたり文数分布、§3.5 補助)
```

**Phase C-1 と並行可能**。F1 と ΔWER の両方が出揃うと §3.0 のパターン表で診断できるので、両方走らせるのが本来の運用形態。

### Phase C-3 — 自前アノテ + 自己訂正 F1 (中コスト)

§5.6 のコスト見積に従い、ja / zh の self-correction span を自前アノテ:

```
tests/evaluation/l3_direct/
├── self_correction_cejc.rs         (ja、自前 span アノテ後)
├── self_correction_magicdata.rs    (zh、自前 span アノテ後)
└── tools/annotation_guidelines.md  (Switchboard stylebook + CSJ + CS2W ベース)
```

5h × 2 名 × 言語数 で **¥200–400k** または社内 4–6 人週。商用化検討フェーズの直前で実施するのが ROI 上効率的。

### Phase D — 商用ライセンスデータ (任意、Cochlis 商用化フェーズ)

CSJ 商用ライセンス (50 万円 / 2 年)、Switchboard NXT (LDC)、HKUST (LDC) を購入して直接 F1 のカバレッジを拡大。Phase C で **無償データの F1 + ablation を出した上で**、商用化 ROI が明確になってから判断する。

talkadict 研究レポートの基礎データは Phase C-1 / C-2 / C-3 の出力をもって構成する。

---

## 7. 将来検討

### 7.1 AudioPreprocessor 評価 (新 trait 追加時)

VAD・ノイズ抑制・ステム分離を導入する場合、§4 の枠組みに以下を追加:
- VAD: trigger latency (発話開始検出の遅延)、clipping rate (発話冒頭欠損率)
- ノイズ抑制: PESQ, STOI, DNSMOS、AudioPreprocessor 適用前後の WER/CER 差分
- 評価データ: §4.4 の URGENT 2025 challenge データ (7 種の歪み × 多言語) を流用

### 7.2 パーソナル VAD / target speaker extraction

複数話者環境での dictation 精度評価。LExt / VoiceFilter 系を入れる場合に追加検討。WHAM! や DNS Challenge personalized track が候補。

### 7.3 Code-switching

en-ja / en-zh の code-switched 入力。CS-FLEURS (52 言語、113 ペア) が候補。Cochlis の海外展開時に重要となる。

### 7.4 Long-form ASR

数十分の長尺音声に対する pipeline 全体の安定性評価。GigaSpeech、Libriheavy が候補。

### 7.5 共通評価ハーネス

Buckeye のように euhadra と Speciphonorm の両方で使うデータセットについては、両プロジェクト共通の評価ランタイムを切り出すことを検討。`euhadra-eval` のような独立 crate にする可能性あり。

---

## 8. 参考リンク

**ASR データセット**:
- FLEURS: https://huggingface.co/datasets/google/fleurs
- Common Voice: https://commonvoice.mozilla.org
- ReazonSpeech: https://huggingface.co/datasets/reazon-research/reazonspeech
- LibriSpeech: https://www.openslr.org/12
- AISHELL-1: https://www.openslr.org/33
- AISHELL-4 (会議): https://www.openslr.org/111/
- WenetSpeech: https://wenet.org.cn/WenetSpeech/
- Buckeye: https://buckeyecorpus.osu.edu/
- TED-LIUM 3: https://www.openslr.org/51

**Disfluency / 自己訂正 アノテ付きコーパス**:
- CSJ (日本語話し言葉コーパス): https://clrd.ninjal.ac.jp/csj/en/
- CSJ 利用料金: https://clrd.ninjal.ac.jp/csj/en/fee.html
- CEJC (日常会話コーパス): https://www2.ninjal.ac.jp/conversation/cejc/
- CEJC LREC 2022 paper: https://aclanthology.org/2022.lrec-1.599/
- MagicData-RAMC (zh 会話): https://www.openslr.org/123/
- MagicData-RAMC paper: https://arxiv.org/abs/2203.16844
- CS2W (zh spoken→written): https://github.com/guozishan/CS2W
- CS2W paper (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.241.pdf
- Switchboard NXT (LDC2009T26): https://catalog.ldc.upenn.edu/LDC2009T26
- Switchboard Disfluency Annotation Stylebook (Meteer et al. 1995): https://www.cs.brandeis.edu/~cs140b/CS140b_docs/DysfluencyGuide.pdf

**ノイズ・RIR データセット**:
- MUSAN: https://www.openslr.org/17
- DEMAND: https://zenodo.org/records/1227121
- DNS Challenge: https://github.com/microsoft/DNS-Challenge
- OpenSLR RIRs and noises (SLR26): https://www.openslr.org/26
- OpenSLR Simulated RIRs (SLR28): https://www.openslr.org/28
- VoiceBank + DEMAND: https://datashare.ed.ac.uk/handle/10283/2791
- CHiME Challenge: https://www.chimechallenge.org/
- URGENT 2024: https://urgent-challenge.github.io/urgent2024/
- URGENT 2025: https://urgent-challenge.github.io/urgent2025/

**シミュレーションツール**:
- pyroomacoustics: https://github.com/LCAV/pyroomacoustics
- audiomentations: https://github.com/iver56/audiomentations
- torch_audiomentations: https://github.com/asteroid-team/torch-audiomentations
- ESPnet recipes: https://github.com/espnet/espnet
