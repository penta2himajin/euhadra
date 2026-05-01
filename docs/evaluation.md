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
   - FLEURS の en / ja / zh / es から各 10 発話を実音声で ASR に通す
     (zh: whisper-tiny multilingual、**en: parakeet-tdt-0.6b-v3** — PR #12 以降、
     **ja: parakeet-tdt_ctc-0.6b-ja** — PR #11 以降、
     **es: canary-180m-flash via istupakov ONNX** — PR #28 以降、`--canary-es-dir` で起動)
   - 計測値: **WER (en) / CER (ja, zh, es)**、**E2E ユーザー知覚レイテンシ p50/p95**、**ASR 段レイテンシ p50/p95**
   - 期待実行時間: 3–5 分 (whisper.cpp ビルド + tiny モデル + FLEURS subset を CI キャッシュ)
2. **`evaluate-fast` ジョブ — 層 ablation + 各層レイテンシ μ-benchmark**
   - `tests/evaluation/fixtures/{en,ja,zh}.jsonl` の text fixture を MockAsr で注入、Tier 1+2 の各層を on/off で走らせる
   - 計測値: 各 layer の **ΔWER/CER** (full vs without_X)、各 layer の **p50/p95 latency** (μs オーダー、warmup 10 + 100 iter × fixture)
   - 期待実行時間: 30 秒以内 (whisper / FLEURS 不要)

**データ**:
- FLEURS の en / ja / zh / es 各 10 発話程度のサブセット
- 軽量 (合計 ~40 MB のオーディオ)、CC-BY 4.0、HuggingFace `google/fleurs` から取得
- ダウンロードスクリプト: `scripts/download_fleurs_subset.py` (es は `es_419` 中南米変種をマップ)

**メトリクス**:
- ASR レベル: WER (en) / CER (ja, zh, es)
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
| es | MLS ES `test` (Canary model card と直接比較) または Common Voice ES `test` | OpenSLR (MLS) / HuggingFace (CV)、CC-BY 4.0 / CC0 |

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
| 直接 (F1) | filler | CIEMPIESS Test (CC-BY-SA 4.0、HF `ciempiess/ciempiess_test`、download-only) | es |
| 直接 (F1) | 段落分割 | TED-LIUM 3 (CC-BY-NC-ND、OSS のみ) | en |
| Ablation (ΔWER) | 各層の最終出力寄与 | §3.7 の E2E 推奨データを流用 | en / ja / zh / es |

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

### 2.5 スペイン語

| データセット | 内容 | ライセンス | 用途 |
|-------------|-----|-----------|------|
| FLEURS es | `es_419` (中南米変種)、読み上げ | CC-BY 4.0 | L1 |
| Common Voice ES | クラウドソース 2,000h+、validated splits | CC0 1.0 | L1 / L2、自前アノテの素材 |
| MLS ES | LibriVox オーディオブック朗読 ~900h、test split あり | CC-BY 4.0 | L2 標準ベースライン (Canary model card と直接比較可) |
| **CIEMPIESS Test** | UNAM、メキシコ Spanish ラジオ放送 8h・3,558 utt、自然発話、orthographic disfluency markup (`e`, `este`, `pus`, `del del`, `sie siempre` 等を綴り字で含む) | **CC-BY-SA 4.0、HuggingFace `ciempiess/ciempiess_test`** | **L3 Tier 1 直接 F1 の es 主データ** (download-only / 都度動的計算ポリシー、§5.1 参照) |
| CIEMPIESS LIGHT | 17h メキシコ Spanish ラジオ、CC-BY-SA | CC-BY-SA 4.0 | L3 拡張、自然発話 ablation |
| TEDx Spanish (OpenSLR 67) | 24h | CC-BY-NC-ND 4.0 | **非商用評価のみ**、参考 |
| ASR-SpCSC (MagicHub) | 5.56h Peninsular | CC-BY-NC-ND 4.0 | **非商用評価のみ** |
| PRESEEA | 10M words 目標、地理・社会的変異 | 非商用 | **不可** (citation only) |
| ESLORA (USC Galicia) | 80h、POS+lemma+morphology タグ付き | 非商用 | **不可** |
| C-ORAL-ROM ES | ~300k words、prosodic break + disfluency 解析 | ELRA 商用 DVD 配布 | **不可** (条件が厳しい) |
| Fisher Spanish (LDC2010T04) | 163h Caribbean / non-Caribbean tel | LDC 有償 | △ 商用ライセンス別売 |
| CALLHOME Spanish (LDC96T17) | 120 通話 | LDC 有償 | △ 同上 |
| DIMEx100 | 6h メキシコ読み上げ、phonetic transcription | "Free Open License" | 読み上げ中心、disfluency なし |

**スペイン語 Tier 1 F1 の状況**: en (Buckeye)、ja (CSJ・CEJC)、zh (MagicData-RAMC・CS2W) と異なり、**商用フレンドリー × disfluency 直接アノテーション** を同時に満たすコーパスは事実上 CIEMPIESS Test のみ。よって運用は:

- **直接 F1**: CIEMPIESS Test (CC-BY-SA 4.0) を **download-only / 都度動的計算** で使う。`scripts/build_es_filler_annotations.py` が transcripts の orthographic markup から構造化 JSONL を生成 (`fillers: [{start, end, label}]`)、`cargo run --example eval_l3 -- --task filler --lang es --input <jsonl>` で F1 を計測。生成 JSONL は `data/cache/` に置き git 追跡対象外、計測スコアのみ `docs/benchmarks/` にコミット (派生物配布回避、SA 制約への対応)。
- **Ablation (ΔWER/CER)**: MLS ES test (CC-BY 4.0) + FLEURS-es (CC-BY 4.0) を使う。読み上げ音声なので Tier 1 が「空打ち」になる前提で、ASR 段の精度回帰検出に主に使う。Tier 1 ablation の有意な ΔWER は CIEMPIESS LIGHT 等の自然発話で測る。

**CIEMPIESS の SA 影響**: download-only 運用で派生物を再配布しない限り SA は euhadra 本体に伝播しない。詳細は §5.1。F1 数値そのものは事実情報なので SA 対象外、`docs/benchmarks/` にコミット可。

### 2.5.1 Spanish ASR backend (Canary-180M-Flash) と FLEURS-es 実測

L1 で es を評価する際の ASR は **NVIDIA Canary-180M-Flash via [`istupakov/canary-180m-flash-onnx`](https://huggingface.co/istupakov/canary-180m-flash-onnx)** (CC-BY 4.0、INT8 ~213 MB / FP32 ~779 MB)。`scripts/setup_canary_es.sh` でダウンロード、`cargo run --features onnx --release --example eval_l1_smoke -- --canary-es-dir <path> --langs es` で起動。FP32 を使うには `CANARY_FP32=1` を setup script に渡す。アーキテクチャ詳細・代替の Parakeet-TDT-0.6B-v3-multi へのフォールバック計画は `docs/canary-integration.md`。

実測 (2026-04-30、`docs/canary-integration.md` の「End-to-end validation」節に詳細):

| Subset | Weights | Penalty | WER (mean) | WER (median) | RTF | Note |
|---|---|---|---|---|---|---|
| FLEURS-es 10 utt | INT8 | — | (CER 9.75%) | — | 0.091 | v1 初回実装確認、外れ値 2 件 |
| FLEURS-es 100 utt | FP32 | 1.0 (off) | 35.37 % | 8.33 % | 0.150 | v2 baseline、catastrophic outlier 2 件が mean を引き上げ |
| FLEURS-es 100 utt | INT8 | 1.0 (off) | ~40 % (σ≈27 %) | — | 0.060 | v2 INT8、3 回で 34.5 / 36.4 / 94.4 % の非決定性 |
| FLEURS-es 100 utt | FP32 | 1.8 | 14.38 % | 8.33 % | 0.165 | v3 repetition penalty 適用、loop 完全消滅 |
| FLEURS-es 100 utt | FP32 | 1.8 + min-len 0.2 | 13.63 % | 8.45 % | 0.117 | v4 min-length gate 追加、hard fail 完全消滅 |
| FLEURS-es 100 utt | FP32 | 1.8 + min-len 0.2 + eos-margin 2.0 | 13.21 % | 8.17 % | 0.101 | v5 EOS-confidence margin 追加、clean +1 |
| **FLEURS-es 100 utt (primary)** | **FP32** | **2.0 + min-len 0.2 + eos-margin 2.0** | **12.89 %** | **8.45 %** | 0.171 | **v6 frontend を Python `onnx-asr` に bit-align、ペナルティ 1.8→2.0 に再調整、clean +2** |

公式 model card 値は **MLS Spanish WER 3.17 %** / **MCV-16.1 ES WER 4.90 %** (FLEURS-es は 180M モデルでは未公開、Canary-1B-v2 で 2.90 %)。v2 → v6 で **mean WER 35.37 % → 12.89 % (Δ = -22.5 pp)** に改善、catastrophic failure mode (repetition loop + hard fail) は完全消滅。v6 で Python `onnx-asr` の `NumpyPreprocessor` と bit-level に揃え (encoder embeddings の max_rel: 13.7% → 3.5e-6)、これにより上流 reference 実装と同等性が証明された。残る ~10 pp gap は greedy decoding 由来 (Python 公式実装でも同じ loop / dropout が出る) で、beam search 等の大改修が要る。詳細は `docs/canary-integration.md` の "v6 — Python-aligned frontend + retune" 節と "Next investigation steps"。

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
- L1: FLEURS en/ja/zh/es
- L2: LibriSpeech test-other / ReazonSpeech-test / AISHELL-1 test / **MLS ES test**
- ノイズ耐性: §4 を参照

### 3.2 Tier 1 — TextFilter (filler removal)

**評価対象**: SimpleFillerFilter、JapaneseFillerFilter、ChineseFillerFilter、SpanishFillerFilter、OnnxEmbeddingFilter

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
- **CIEMPIESS Test: orthographic markup (`e`, `este`, `pus`, `del del`, `sie siempre` 等) を `scripts/build_es_filler_annotations.py` で構造化 JSONL に昇格 (es、CC-BY-SA 4.0、download-only)**
- ReazonSpeech / WenetSpeech-meeting: 自前で軽量アノテーション (100–500 発話、§5.6 でコスト見積)

**推奨データ**:
- en: Buckeye (free)、Switchboard NXT (LDC 有償、入手可能なら追加)
- ja: **CEJC (free academic edition)** または **CSJ (有償または学術無償)**、補助に ReazonSpeech サブセット + 自前アノテ
- zh: **MagicData-RAMC test (無償、OpenSLR 123)**、**CS2W (text-only、CI 単体テスト用)**、補助に WenetSpeech-meeting + 自前アノテ
- es: **CIEMPIESS Test (CC-BY-SA 4.0、HuggingFace `ciempiess/ciempiess_test`)** を `build_es_filler_annotations.py` 経由。Phase 1 (現状) はパイプラインで生成した gold に対する自己整合性確認。Phase 2 で `Common Voice ES` (CC0) の自然発話サブセットへの自前アノテに拡張 (§5.6)。

#### Ablation 評価 (ΔWER/CER)

**メトリクス**: filter on / off の最終 WER/CER 差分。Filter on 時の WER 改善 (ΔWER < 0) が期待値。改善幅が直接評価の F1 と一致しない場合は §3.0 のパターン表で診断。

**Ground truth**: ASR 用 reference transcript のみ。専用 annotation 不要。

**推奨データ**: §3.7 の E2E 推奨データから **自然発話を含むものを優先**。読み上げ音声 (FLEURS 等) では filler が存在しないため空打ちになる。
- en: TED-LIUM 3 (CC-BY-NC-ND、OSS 評価のみ)
- ja: ReazonSpeech-test (§30-4 用途制限)
- zh: WenetSpeech `test-meeting` (CC-BY 4.0)
- es: CIEMPIESS LIGHT (CC-BY-SA 4.0、自然発話 18h、download-only)

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
- スペイン語: **Switchboard NXT 様のコーパスは存在しない**。CIEMPIESS LIGHT のクリーンな transcript 上に span を貼る自前アノテが必須 (§5.6)。cue 集合は `no, perdón, digo, mejor, no es, mejor dicho, o sea` を起点に CSJ の `(D)` パターンに準拠。

**推奨データ**:
- en: Switchboard NXT (LDC 有償、入手可能なら)、または en TED-LIUM 3 への自前 span アノテ
- ja: CEJC (主)、CSJ (補助、長文・準フォーマル比較用)
- zh: **MagicData-RAMC サブセット + 自前 span アノテ** (~5h、§5.6 で ¥100–200k 見積)
- es: **CIEMPIESS LIGHT サブセット + 自前 span アノテ** (~5h、§5.6 で ¥100–200k 見積)

**Inter-annotator agreement の留意点**: Reparandum span boundary は低 IAA タスク (英 Switchboard でも κ ≈ 0.55–0.75)。ja / zh で自前アノテする場合は **二重アノテ + 調停** を前提に設計し、単一アノテ単独の F1 を信頼しすぎない。

#### Ablation 評価 (ΔWER/CER)

**メトリクス**: detector on / off の最終 WER/CER 差分。

**Ground truth**: ASR 用 reference transcript のみ。

**推奨データ**: §3.7 E2E データ。**自然発話必須** (読み上げには自己訂正がない):
- en: TED-LIUM 3 (講演中の言い直しを含む)
- ja: ReazonSpeech-test
- zh: WenetSpeech `test-meeting`
- es: CIEMPIESS LIGHT (CC-BY-SA 4.0、自然発話 18h)

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
- スペイン語: G2P が決定的なため CMUdict 相当の音素辞書は不要、ルールベース G2P で `PhonemeCorrector` の主要ロジックは流用可。FLEURS-es / Common Voice ES の人名・地名セグメントで with-dict / without-dict ΔWER を測る

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
- 軽量: FLEURS 4 言語サブセット (en / ja / zh / es)
- 自然発話: TED-LIUM 3 (en) + ReazonSpeech-test (ja) + WenetSpeech test-meeting (zh) + CIEMPIESS LIGHT (es)

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

**回帰検出ポリシー** (相対 / 絶対の 2 軸):

**(A) baseline 相対** (`docs/benchmarks/ci_baseline*.json` を ground truth):

| メトリクス | warning | hard fail |
|---|---|---|
| WER / CER (絶対値) | > baseline + 5% (絶対) | > baseline + 10% |
| WER / CER (相対) | regression > 20% | > 50% |
| **RTF (相対)** | **regression > 100% (2× RTF)** | **> 200% (3× RTF)** |
| 各層 latency p50 (μ-bench) | regression > 200% | > 400% (5× 遅い) |
| ASR latency p50 | regression > 100% (2×) | > 200% (3×) |
| E2E latency p50 | regression > 50% | > 150% |
| ASR live スモーク完走 | — | 例外で落ちたら即 fail |

**(B) 環境非依存の絶対閾値** (warning のみ、fail にはしない):

| メトリクス | warning |
|---|---|
| **RTF** | **≥ 1.0** (リアルタイムより遅い → streaming dictation 不可) |
| **ASR latency p50** | **≥ 1,000 ms** |
| **E2E latency p50** | **≥ 1,000 ms** |
| **各層 latency p50** | **≥ 1,000,000 μs (= 1 秒)** — sub-millisecond 前提への sanity floor |

**なぜ 2 軸か**: 開発者の手元 (M2 Mac、低 RTF)、CI runner (共有 Linux VM、中 RTF)、実環境 (Cochlis のオンデバイス)で速度が大きく異なる。相対閾値だけでは「もともと baseline が遅い → 多少遅くなっても通る」状態を許容してしまうので、ユーザー知覚品質に直結する **絶対上限** を warning として常時可視化する。

**RTF (Real-Time Factor)** = ASR 処理時間 / 音声長。`< 1.0` ならリアルタイムより速い (streaming dictation の前提)。ハードウェア正規化された指標なので発話長に依存しない latency として最も比較しやすい。L1 evaluate-asr と L2 の両方で記録。

絶対閾値は **L1 evaluate-asr** (`ci_baseline.json` の `tolerances` に保存)、**L1 evaluate-fast** (`ci_baseline_layers.json`)、**L2** (`examples/eval_l2.rs` 内のハードコード定数) すべてで同じ値を共有する。CI 上で warning が連続して出る場合、それは「baseline 自体が user-perceivable 性能を満たしていない」というシグナルなので、モデル軽量化なり host CPU 強化なりが要対応。

**ノイズ吸収のため**:
- 30 発話程度では WER stderr ~5%、絶対閾値判定は粗いため warning のみで運用
- latency は **rolling N コミット (例: 直近 5 PR) の中央値** を baseline とすると spike 耐性が上がる (本 PR ではまだ実装せず、将来検討)
- runner ノイズが疑われる失敗は再実行して再現確認

**baseline ファイル構成**:

`docs/benchmarks/` 配下に CI ジョブごとに 1 ファイル:

- `ci_baseline.json` — `evaluate-asr` 用 (Phase A-1)。各言語の WER/CER + ASR + E2E latency
- `ci_baseline_layers.json` — `evaluate-fast` 用 (Phase A-2)。各言語の layer ablation (full / without_X) + 各 layer の μ-benchmark latency (μs 単位)

`evaluate-asr` (live ASR) と `evaluate-fast` (post-ASR ablation) は **異なる正規化パス** を使う:

- `evaluate-asr` は **lenient** (`wer_lenient` / `cer_lenient`) — 大文字小文字、ASCII / 全角句読点、smart quote、ハイフン分割、漢数字 ↔ Arabic、英単語数値 ↔ Arabic を吸収。FLEURS reference と Whisper / Parakeet / Paraformer 出力の **書式ゆれ** を打ち消し、ASR モデル本体の認識性能のみを測る目的。
- `evaluate-fast` は **strict** (`wer` / `cer`) — 正規化なし (whitespace ハンドリングのみ tokenization の都合で残す)。各 post-ASR 層が挿入・変更した case / 句読点 / ハイフンが ablation delta にそのまま現れる。`BasicPunctuationRestorer` が末尾 `.` と先頭大文字を加える効果を測れるようにするため。

**`ci_baseline.json` の schema** (Phase A-1):

```json
{
  "schema_version": 1,
  "generated": "2026-04-25T...",
  "asr_model": "ggml-tiny.en.bin / ggml-tiny.bin",
  "languages": {
    "en": {
      "samples": 10,
      "wer": 0.08,
      "cer": null,
      "rtf": 0.16,
      "asr_latency_ms":  {"p50": 800, "p95": 1100},
      "e2e_latency_ms":  {"p50": 850, "p95": 1200}
    },
    "ja": { "samples": 10, "wer": null, "cer": 0.42, "rtf": 0.14, "...": "..." },
    "zh": { "samples": 10, "wer": null, "cer": 0.32, "rtf": 0.08, "...": "..." }
  },
  "tolerances": {
    "wer_absolute_warn":   0.05,
    "wer_absolute_fail":   0.10,
    "wer_relative_warn":   0.20,
    "wer_relative_fail":   0.50,
    "latency_p50_relative_warn": 1.00,
    "latency_p50_relative_fail": 2.00,
    "e2e_latency_p50_relative_warn": 0.50,
    "e2e_latency_p50_relative_fail": 1.50,
    "rtf_relative_warn": 1.00,
    "rtf_relative_fail": 2.00,
    "rtf_absolute_warn":            1.00,
    "asr_latency_absolute_warn_ms": 1000.0,
    "e2e_latency_absolute_warn_ms": 1000.0
  }
}
```

**`ci_baseline_layers.json` の schema** (Phase A-2):

```json
{
  "schema_version": 1,
  "generated": "2026-04-25T...",
  "languages": {
    "en": {
      "fixtures": 27,
      "ablation": {
        "full":                    0.04,
        "without_filler":          0.20,
        "without_self_correction": 0.20,
        "without_punctuation":     0.04
      },
      "layer_latency_us": {
        "filler":          {"p50": 3.7, "p95": 6.1},
        "self_correction": {"p50": 0.6, "p95": 1.6},
        "punctuation":     {"p50": 0.4, "p95": 0.5}
      }
    },
    "ja": { "fixtures": 23, "ablation": {"...": "..."}, "layer_latency_us": {"...": "..."} },
    "zh": { "fixtures": 15, "ablation": {"full": 0.0}, "layer_latency_us": {"...": "..."} }
  },
  "tolerances": {
    "ablation_absolute_warn":          0.02,
    "ablation_absolute_fail":          0.05,
    "layer_latency_p50_relative_warn": 2.00,
    "layer_latency_p50_relative_fail": 4.00,
    "layer_latency_absolute_warn_us":  1000000.0
  }
}
```

`zh` は Tier 1 filter (`ChineseFillerFilter`) が PR #10 で実装され、ablation も他言語と同じ 4 構成 (`full` / `without_filler` / `without_self_correction` / `without_punctuation`) を出力する。Tier 2 self-correction は中文ルール未対応のため `without_self_correction` の ΔWER は ~0 になる (機能追加時に拡張)。

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
| es filler F1 (Common Voice ES 自然発話 / CIEMPIESS LIGHT 補助) | 5h | 2 名 | ¥80–150k (transcript は CC0 / CC-BY-SA、span のみ) |
| es self-correction F1 (CIEMPIESS LIGHT サブセット) | 5h | 2 名 | ¥100–200k (`no, perdón, digo, mejor` cue 起点) |

**Inter-annotator agreement の期待値** (文献ベース):
- Filler 同定 (closed class、binary content/filler): Cohen's κ **0.80–0.95** (1 時間のキャリブレーションあり)
- Reparandum span boundary: κ **0.55–0.75** ← **二重アノテ + 調停を前提に設計**

**再利用可能なアノテーションガイドライン**:
- Switchboard Disfluency Annotation Stylebook (Meteer et al. 1995) — reparandum / interregnum / repair 構造の定義、言語横断で流用可
- CSJ 転記マニュアル (NINJAL、無償公開) — ja の `(F)` / `(D)` / `(D2)` 慣例
- CEJC 転記ガイドライン (NINJAL、無償公開) — 日常会話寄り
- CS2W アノテーション規約 (CC-BY-SA、GitHub) — zh の spoken→written 規約
- Val.Es.Co (UV) の disfluency 解析論文 — es の自然発話 disfluency 分布の参照 (アノテーションガイドラインそのものは公開されていないが、prevalence 数値が利用可能)

**推奨アプローチ**: Shriberg / Meteer の reparandum スキームを言語横断の構造的バックボーンとし、CSJ の `(F)` / `(D)` 閉集合を ja に、対応する zh 閉集合 (嗯 / 呃 / 啊 / 那个 / 这个 / 就是 等)、対応する es 閉集合 (e / eh / este / pus / o sea 等、`scripts/build_es_filler_annotations.py` の lexicon と整合) を新規定義する。**音声に触れる前にガイドライン整備で半日確保**することが推奨。

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
├── metrics.rs                       (WER / CER strict + lenient、Levenshtein、normalize_lenient)
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

**Phase A-2 (実装済み): `evaluate-fast` ジョブ — 層 ablation + per-layer latency**

Mock ASR で text fixture を注入し、Tier 1+2 の各層を on/off で走らせて ΔWER と layer latency を取る。Phase A-1 と並列実行で、両ジョブが揃って L1 が完成する。

```
src/eval/fixtures.rs                  (JSON-Lines fixture loader)
src/eval/baseline.rs                  (LayerBaseline schema、ci_baseline_layers.json I/O)
examples/eval_l1_fast.rs              (ablation 配置 + per-layer μ-benchmark binary)
tests/evaluation/fixtures/{en,ja,zh}.jsonl  (text fixture、自社制作物、project license)

docs/benchmarks/
└── ci_baseline_layers.json           (各言語の ablation + per-layer latency baseline)
```

各 fixture は `{id, category, reference, asr_hypothesis}` の JSON-Lines。`asr_hypothesis` は人手で書き起こした仮想 ASR 出力 (filler / 自己訂正 / clean カテゴリ)。`reference` が ground truth。Pipeline を `(filter, self_correction, punctuation)` の各 layer の on/off 4 通りで走らせて、フル構成 vs 各 layer-off 構成の WER/CER を比較する。

CI ジョブとしては A-1 と並列実行。両ジョブが揃って始めて L1 が完成する。

CI で毎 PR 実行。WER/CER + latency の閾値を回帰検出に使う (§3.8 参照)。Phase B 以降の **共通基盤** (`src/eval/metrics.rs` + データ download ヘルパ) を兼ねる。

### Phase B — L2 標準ベンチマーク (リリース毎、手動実行、実装済み)

CI には乗せず、リリース前にローカルマシン (GPU 推奨、CPU でも whisper-tiny なら可) で手動実行する。Setup と評価実行を統合した単一の Cargo エイリアス経由で起動する:

```bash
# baseline (LibriSpeech test-clean を未取得なら自動 download)
cargo eval-l2 -- --dataset librispeech-test-clean --condition baseline

# L2-Robust: ノイズ加算
cargo eval-l2 -- --dataset librispeech-test-clean --condition noise --snr-db 10

# L2-Robust: 残響加算
cargo eval-l2 -- --dataset librispeech-test-clean --condition reverb

# L2-Robust: ノイズ + 残響 (フル現実条件)
cargo eval-l2 -- --dataset librispeech-test-clean --condition noise-reverb --snr-db 5
```

実装ファイル:

```
examples/eval_l2.rs                  (cargo eval-l2 のエントリ。FFT 畳み込み +
                                       SNR スケール混合をインライン実装)
scripts/download_l2_data.sh          (LibriSpeech / AISHELL-1 / MUSAN / SLR26
                                       RIR を curl + 抽出)
scripts/download_l2_data.py          (ReazonSpeech を HuggingFace 経由で取得)
.cargo/config.toml                   ([alias] eval-l2)
```

サポートデータセット (`--dataset` 引数):

| 値 | 言語 | サイズ | ライセンス |
|---|---|---|---|
| `librispeech-test-clean` | en | ~350 MB | OpenSLR / CC-BY 4.0 |
| `librispeech-test-other` | en | ~330 MB | OpenSLR / CC-BY 4.0 |
| `aishell1-test` | zh | ~16 GB (full archive) | OpenSLR / Apache-2.0 |
| `reazonspeech-test` | ja | ~1 GB | §30-4 (機械学習用途のみ) |

Robust 用の補助データ:

| データ | サイズ | ライセンス | デフォルトパス |
|---|---|---|---|
| MUSAN noise | ~11 GB | CC-BY 4.0 | `data/l2/musan-noise` |
| OpenSLR SLR26 RIR | ~2.1 GB | Apache-2.0 系 | `data/l2/rir-slr26` |

メトリクス: WER (en) / CER (ja, zh) + **RTF** + ASR/E2E latency (p50/p95)。出力は標準出力 + `--output report.json` で JSON 保存可能。

**再現性 (§4.6 準拠)**: `--seed` フラグ (デフォルト 42) でノイズ・RIR の選択を決定論化。同じ seed × dataset × condition × snr_db は同一の混合音声を生成 (混合音声自体は commit せず、§5.2 のポリシーに従う)。

リリース前に手動実行、結果を `docs/benchmarks/YYYY-MM-DD.md` に記録。L2-Robust の SNR 別劣化曲線も同レポートに含める。

**スコープ外** (フォローアップ予定):
- Mix manifest (`tests/evaluation/manifests/l2_robust.tsv`) を commit する完全な再現性: 現状はランタイムで seed から生成。MUSAN/RIR pool のバージョン固定が前提となるため、最初のリリース benchmark を実走させた後に確定版を commit する
- `aishell1-test` setup script は archive 全体 (16 GB) を fetch する。test split のみの mirror が見つかれば script を更新する

### Phase C — L3 評価 (一部実装、CI 対象外、`cargo eval-l3` 経由)

L3 は **手動 / 研究レポート用途**。CI には乗せず、ローカルマシンでローカルアノテーション + ローカルデータに対して走らせる。Setup と評価実行を統合した単一の Cargo エイリアス経由で起動する:

```bash
# C-1: ja self-correction 直接 F1 (commit 済 annotation を使用)
cargo eval-l3 -- --task self-correction --lang ja \
    --input tests/evaluation/annotations/ja_self_correction.jsonl

# C-2: 自然発話 fixture に対する layer ablation
cargo eval-l3 -- --task ablation --lang ja \
    --input tests/evaluation/fixtures-natural/ja.jsonl
```

#### Phase C-1 — L3 直接評価 (F1)

| データセット | 言語 | 状態 | アクセス |
|---|---|---|---|
| **自社 ja self-correction** (35 発話) | ja | **✓ commit 済** (`tests/evaluation/annotations/ja_self_correction.jsonl`) | 自社制作、project license。Claude 下書き、人手レビュー予定 |
| **CIEMPIESS Test filler** | es | **✓ harness 実装済** (PR #21、`cargo run --example eval_l3 -- --task filler --lang es --input <jsonl>`)。JSONL は `scripts/build_es_filler_annotations.py` で都度生成、git 追跡対象外 (download-only / SA 制約への対応) | HF `ciempiess/ciempiess_test` (CC-BY-SA 4.0) |
| Buckeye filler | en | 未実装 | https://buckeyecorpus.osu.edu (要登録) |
| MagicData-RAMC filler | zh | 未実装 (annotation loader 未対応) — `ChineseFillerFilter` は実装済 (PR #10) なので detector 自体の評価は可能、loader 実装後に F1 を出す | OpenSLR 123 (`scripts/download_l3_data.sh magicdata-ramc-info` で取得手順) |
| CS2W filler text-only | zh | clone 済 (`scripts/download_l3_data.sh cs2w`)、loader 未実装 — `ChineseFillerFilter` 実装済のため text-only F1 を直接評価する手段が次の PR で追加される | https://github.com/guozishan/CS2W |
| CEJC filler / self-correction | ja | 未実装 | NINJAL Corpus Portal (要申請) |
| TED-LIUM 段落分割 F1 | en | 未実装 | OpenSLR 51 (~50 GB、`scripts/download_l3_data.sh tedlium3-test`) |

#### Phase C-2 — L3 Ablation (ΔWER/CER)

§3.7 の E2E データに対して各層を on / off で走らせ ΔWER/CER を測る。F1 用追加 annotation 不要。

| 言語 | fixtures | 状態 |
|---|---|---|
| en | `tests/evaluation/fixtures-natural/en.jsonl` (10 発話、FLEURS 由来 whisper-tiny output) | **✓ commit 済** |
| ja | `tests/evaluation/fixtures-natural/ja.jsonl` (10 発話、同上) | **✓ commit 済** |
| zh | `tests/evaluation/fixtures-natural/zh.jsonl` (10 発話、同上) | **✓ commit 済** |
| es | (未生成) — `build_l3_natural_fixtures.py` を Canary パスに対応させた後 FLEURS-es 10 発話 + CIEMPIESS LIGHT サブセットから生成 | 未実装 |

現状の natural-speech fixtures は FLEURS (読み上げ音声) に whisper-tiny を通した小規模サンプル。**Tier 1 / Tier 2 self-correction の効果はほぼ見えない**ことを前提とする。本格的な自然発話 ablation には ReazonSpeech / WenetSpeech-meeting / TED-LIUM へ拡張が必要 (フォローアップ; ReazonSpeech は HF gated repo、TED-LIUM は ~50 GB)。

`scripts/build_l3_natural_fixtures.py` で fixtures を再生成可能。`source=manifest --manifest <path>` を渡せば任意の FLEURS フォーマット manifest から fixtures を生成する。

#### Phase C-3 — 自前アノテ + 自己訂正 F1

| 言語 | annotation | 状態 |
|---|---|---|
| **ja self-correction** (35 発話) | `tests/evaluation/annotations/ja_self_correction.jsonl` | **✓ Claude 下書き済、人手レビュー待ち** |
| zh self-correction | — | 未実装 (§5.6 で ¥100–200k 見積) |
| **es self-correction** (rule-based detector) | — | **✓ 実装済 (PR #35)** — `SelfCorrectionDetector::detect_spanish` が cue 集合 `no, perdón, digo, mejor dicho, o sea, no es, quiero decir` で動作。`src/processor.rs` に 10 unit test、`tests/pipeline_e2e.rs::spanish_full_pipeline_no_llm` に E2E test (filter + detector + punctuation 連鎖)。F1 ベンチマーク用アノテはまだ未着手 (§5.6 で ¥100–200k 見積)。 |

ja annotation のスキーマと判定ルールは `tests/evaluation/annotations/guidelines.md` を参照。
PR レビューで以下を確認:
1. cue 語の使い方が自然か (機械的すぎないか)
2. span 境界の判定が guidelines §3.2 のルールに従っているか
3. edge case (`edge_001`, `edge_002`) の判定が妥当か
4. クリーン文に違和感がないか

ja annotation v0.2 (40 発話) で `SelfCorrectionDetector` を評価した結果:

| 指標 | 値 |
|---|---|
| utterance-level F1 | 1.000 (33/33 fire correct, 7/7 silence correct) |
| span-level F1 (IoU≥0.5) | **1.000** (33/33 boundary match) |
| span-level F1 (strict) | **1.000** |

v0.1 (35 発話) 当時は span-level F1 が 0.964 で、「鈴木課長、っていうか佐藤課長です」で `っていうか` cue が cue 一覧上の `ていうか` (substring) に取られて reparandum が残る現象を surface していた。**この PR で `SelfCorrectionDetector` 側を cue 長さ降順でマッチさせるよう修正**し、annotation も `じゃない` 正用例 + 長文文脈例で 5 件拡張した結果、現在の F1 = 1.000 に到達。L3 評価が detector の真のロジックエッジを surface し、修正後にも同じ評価で確認できる流れの最初の実例となった。

5h × 2 名 × 言語数 で **¥200–400k** または社内 4–6 人週 (フル運用時)。現状は v0.2 の 40 発話 (Claude 下書き) を起点とし、人手レビューで silver-standard へ昇格、規模を必要に応じて拡張する。

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
