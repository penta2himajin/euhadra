# euhadra — Technical Specification v0.1

## 1. Overview

### 1.1 What is euhadra

euhadra は、音声入力を汎用的なプログラマブル入力として扱うための OSS フレームワークである。ASR（自動音声認識）、LLM による後処理、OS コンテキスト取得、テキスト挿入を統合する非同期パイプラインを Rust で提供し、開発者が Aqua Voice や TYPELESS 相当の音声 dictation 体験を最小限のコードで構築できるようにする。

コアはプラットフォーム非依存の抽象層として設計され、OS 固有の機能は薄いネイティブシェル（Swift / Kotlin / C++ 等）を通じてプラグインとして注入される。

### 1.2 Naming

"euhadra" は日本列島を中心に種分化したカタツムリであるマイマイ属（*Euhadra*）の学名に由来する。

- 耳（ear / 聴覚）→ 蝸牛（cochlea / 内耳の器官）→ カタツムリ → *Euhadra*

音声入力という聴覚ドメインとの意味的連鎖に加え、日本固有の属であることから日本発の OSS プロジェクトとしてのアイデンティティを表現している。

### 1.3 Positioning

| 比較軸 | Aqua Voice / TYPELESS | euhadra |
|--------|----------------------|---------|
| 形態 | エンドユーザー向け完成品 | 開発者向け OSS フレームワーク |
| ASR / LLM | 固定（自社 or 非公開） | 任意のプロバイダーを差し替え可能（BYO-Model） |
| ホスティング | クラウド専用 | セルフホスト可能 / オンデバイス完結可能 |
| 拡張性 | なし | パイプライン全体がプログラマブル |
| ライセンス | プロプライエタリ | MIT or Apache 2.0 |

戦略的には、Aqua Voice / TYPELESS が販売しているレイヤーを OSS としてコモディティ化し、音声 dictation を「当たり前」にした上で、より上位の音声 UI/UX 抽象層へ進む。

### 1.4 Growth Path

- **Phase 1**: Dictation のコモディティ化 — ASR → LLM refinement → テキスト挿入のパイプラインを OSS で提供。10 行で Aqua Voice 相当が動く体験を実現する。
- **Phase 2**: 音声入力のプログラマブル化 — パイプラインの出力をテキスト挿入以外に拡張。構造化データ（JSON / コマンドオブジェクト）への変換、アプリケーションアクションへのディスパッチ、マルチモーダル入力イベントとしての抽象化。
- **Phase 3**: 音声 UI/UX レイヤーの抽象化 — フィードバック表現、会話的な訂正・取り消し・曖昧性解消のプロトコル、アプリが「音声対応」を宣言できるインターフェースの提供。

---

## 2. Architecture

### 2.1 High-Level Structure

```
┌──────────────────────────────────────────────┐
│              OS Shell (per-platform)          │
│  Swift (macOS/iOS) / Kotlin (Android) /      │
│  C++ (Windows) / etc.                        │
│                                              │
│  ┌─────────────┐ ┌──────────────┐            │
│  │ Mic Capture  │ │ Accessibility│            │
│  │ & Activation │ │ API Bridge   │            │
│  └──────┬──────┘ └──────┬───────┘            │
│         │               │                    │
│  ┌──────┴───────────────┴───────┐            │
│  │    C ABI / UniFFI Boundary   │            │
│  └──────────────┬───────────────┘            │
│                 │                            │
│  ┌──────────────▼───────────────────────┐    │
│  │         euhadra core (Rust)          │    │
│  │                                      │    │
│  │  ┌────────────────────────────────┐  │    │
│  │  │     Pipeline Runtime (tokio)   │  │    │
│  │  │                                │  │    │
│  │  │  Activation ──► ASR Adapter    │  │    │
│  │  │                    │           │  │    │
│  │  │              ┌─────▼──────┐    │  │    │
│  │  │              │  Context   │    │  │    │
│  │  │              │  Provider  │    │  │    │
│  │  │              └─────┬──────┘    │  │    │
│  │  │                    │           │  │    │
│  │  │              ┌─────▼──────┐    │  │    │
│  │  │              │    LLM     │    │  │    │
│  │  │              │ Refinement │    │  │    │
│  │  │              └─────┬──────┘    │  │    │
│  │  │                    │           │  │    │
│  │  │              ┌─────▼──────┐    │  │    │
│  │  │              │   Output   │    │  │    │
│  │  │              │  Emitter   │    │  │    │
│  │  │              └────────────┘    │  │    │
│  │  └────────────────────────────────┘  │    │
│  │                                      │    │
│  │  ┌──────────┐ ┌───────────┐          │    │
│  │  │  State   │ │  Channel  │          │    │
│  │  │ Machine  │ │  Manager  │          │    │
│  │  └──────────┘ └───────────┘          │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

### 2.2 Layer Responsibilities

**euhadra core (Rust)**
- パイプラインランタイム: tokio ベースの非同期実行エンジン
- Adapter trait 定義: ASR / LLM / Context Provider / Output Emitter の抽象インターフェース
- チャネル管理: bounded channel によるステージ間通信、backpressure 制御
- ステートマシン: activation → recording → processing → emitting → idle の状態遷移管理
- キャンセル伝播: CancellationToken による inflight リクエストの即時中断
- エラーハンドリング: ステージ単位の timeout / retry / fallback

**OS Shell (ネイティブ層) — euhadra の出荷物ではなく、利用側アプリの統合パターン**

当初この層には 4 つの責務を置いていたが、実装の結果 2 つはネイティブコードなしで解決した。

| 当初の責務 | 現状 |
|---|---|
| マイクキャプチャ | `cpal`（`mic` feature）で実装済み。**Rust・クロスプラットフォーム** |
| テキスト挿入（クリップボード） | `arboard`（`clipboard` feature）で実装済み。**Rust・クロスプラットフォーム** |
| Accessibility 経由のコンテキスト取得 | 未実装。真にネイティブが必要（#123） |
| OS 組み込み LLM 呼び出し | 未実装。真にネイティブが必要（#122） |

残るネイティブ必須項目は Accessibility Provider、OS 組み込み LLM、グローバルホットキーによる activation（§7.1）、IME 統合（§9.3 で Phase 2 Non-Goal）である。前 2 者は #122 / #123 で保留とした。

したがって **euhadra はこの層を出荷しない**。euhadra が提供するのは trait（接続点）であり、ネイティブ実装を注ぐかどうかは利用側アプリの判断となる。activation についても、mic / clipboard と同様に Rust のクロスプラットフォームクレートで済む可能性があり、ネイティブ前提と決めつけない。

**C ABI / UniFFI 境界 — 未着工**

構想は §10.3 の通りだが、現時点で `uniffi` / `extern "C"` / `#[no_mangle]` はリポジトリに存在しない。**消費者が現れるまで着工しない**方針を取る。理由は §10.3 に記す。

---

## 3. Core Abstractions (Trait Design)

### 3.1 ASR Adapter

音声バイナリを受け取り、テキストストリームを返す。ローカル / クラウドを問わず同一インターフェースで扱う。

```
trait AsrAdapter {
    /// 音声データのストリームを受け取り、認識結果のストリームを返す
    fn transcribe(audio: Stream<AudioChunk>) -> Stream<AsrResult>
}

struct AsrResult {
    text: String,           // 認識テキスト
    is_final: bool,         // 確定結果か partial か
    confidence: f32,        // 信頼度 (0.0 - 1.0)
    timestamp: Duration,    // 音声内のタイムスタンプ
}
```

**想定実装**:
- `WhisperLocalAdapter` — Whisper.cpp / MLX Whisper（オンデバイス）
- `WhisperCloudAdapter` — OpenAI Whisper API
- `DeepgramAdapter` — Deepgram Streaming API
- `ElevenLabsAdapter` — ElevenLabs Scribe API
- `AvalonnAdapter` — Aqua Voice Avalon API
- `AppleSpeechAdapter` — Apple Speech framework（iOS / macOS）
- `GoogleSttAdapter` — Google Cloud Speech-to-Text

### 3.2 Context Provider

現在のアプリケーションコンテキストを構造化データとして提供する。

```
trait ContextProvider {
    /// 現時点のコンテキストを取得
    fn get_context() -> ContextSnapshot
}

struct ContextSnapshot {
    app_name: Option<String>,         // フォーカス中のアプリ名
    app_bundle_id: Option<String>,    // アプリの識別子
    field_content: Option<String>,    // テキストフィールドの既存内容
    field_type: Option<FieldType>,    // テキストフィールドの種別
    custom_dictionary: Vec<String>,   // ユーザー辞書
    instructions: Option<String>,     // ユーザー定義のカスタム指示
    locale: Option<String>,           // 現在のロケール
}

enum FieldType {
    CodeEditor,
    EmailCompose,
    ChatMessage,
    Terminal,
    Document,
    SearchBar,
    Generic,
}
```

**想定実装**:
- `MacAccessibilityProvider` — macOS AXUIElement API
- `WindowsUiaProvider` — Windows UI Automation
- `LinuxAtSpiProvider` — Linux AT-SPI（D-Bus 経由）
- `OcrFallbackProvider` — OCR ベースのフォールバック
- `ManualProvider` — 手動でコンテキストを指定（テスト / CLI 用）

### 3.3 LLM Refinement

ASR の生テキストとコンテキストを受け取り、整形済みの出力を返す。

```
trait LlmRefiner {
    /// 生テキストとコンテキストから整形結果を生成
    fn refine(input: RefinementInput) -> Stream<RefinementOutput>
}

struct RefinementInput {
    raw_text: String,                 // ASR の生テキスト
    context: ContextSnapshot,         // アプリコンテキスト
    mode: RefinementMode,             // 処理モード
}

enum RefinementMode {
    Dictation,        // 通常のテキスト整形
    Command,          // コマンド解釈 (Phase 2)
    Structured,       // 構造化出力 (Phase 2)
}

enum RefinementOutput {
    /// Phase 1: 整形テキストをアプリに挿入
    TextInsertion {
        text: String,
        formatting: Option<FormattingHint>,
    },
    /// Phase 2: アクションとして解釈
    Command {
        action: String,
        parameters: HashMap<String, Value>,
    },
    /// Phase 2-3: 意図 + コンテキスト + テキストの複合
    StructuredInput {
        intent: String,
        text: Option<String>,
        metadata: HashMap<String, Value>,
    },
}
```

**想定実装 — クラウド**:
- `CerebrasRefiner` — Cerebras Inference API（超低レイテンシ）
- `GroqRefiner` — Groq API（低レイテンシ）
- `OpenAiRefiner` — OpenAI API（GPT-4o-mini 等）
- `AnthropicRefiner` — Anthropic API（Claude Haiku 等）

**想定実装 — オンデバイス（OS 組み込み API）**:
- `AppleFoundationRefiner` — Apple Foundation Models framework（~3B、Swift FFI 経由）。iOS 26 / macOS 26 以降。Guided generation・Tool calling 対応。LoRA アダプター訓練可能
- `GeminiNanoRefiner` — Gemini Nano（Android AICore API、Kotlin JNI 経由）。Android 14+、Pixel 8+ 等
- `PhiSilicaRefiner` — Phi Silica（Windows App SDK / Windows AI Foundry 経由、C++ FFI）。Copilot+ PC（40+ TOPS NPU）必須。LoRA ファインチューン対応（preview）

**想定実装 — オンデバイス（汎用フォールバック）**:
- `LlamaCppRefiner` — llama.cpp C API 直接リンク（libllama、インプロセス推論）。全プラットフォーム共通。ユーザーが任意の GGUF モデルを指定可能（Phi-3.5, Qwen2.5, Gemma 2, Llama 3.2 等）。OS 組み込み API が利用できない環境での標準フォールバック

### 3.4 Output Emitter

パイプラインの最終出力をシステムに反映する。

```
trait OutputEmitter {
    /// 出力を対象に挿入 / 実行
    fn emit(output: RefinementOutput) -> EmitResult

    /// 直前の出力を取り消し
    fn undo() -> EmitResult
}

struct EmitResult {
    success: bool,
    error: Option<EmitError>,
}
```

**想定実装**:
- `ClipboardEmitter` — クリップボード経由でペースト（最も互換性が高い）
- `KeyEmulationEmitter` — キーストロークエミュレーション
- `ImeEmitter` — IME 統合（日本語等の入力メソッド対応）
- `CallbackEmitter` — コールバック関数でアプリケーションに直接返す（ライブラリ利用時）
- `StdoutEmitter` — 標準出力に出力（CLI / パイプ連携用）

### 3.5 Text Filter（軽量前処理層）

ASR 出力に対してフィラー除去等の軽量な前処理を行う。LLM を使わず、埋め込みモデルやルールベースで動作する。パイプライン上では ASR の直後、TextProcessor や LLM Refiner の前に位置する。

```
trait TextFilter {
    /// テキストを変換し、フィルタ後のテキストと除去された部分を返す
    fn filter(text: &str) -> FilterResult
}

struct FilterResult {
    text: String,           // フィルタ後のテキスト
    removed: Vec<String>,   // 除去されたセグメント（診断 / undo 用）
}
```

**想定実装**:
- `FillerFilter::for_language(Language)` — **推奨エントリポイント**。言語から分かち書き方式（空白 / `、` / `，`）を型で選び、以下の具象フィルタへ委譲する。空白区切りの `SimpleFillerFilter` を日本語・中国語に手で組み合わせると、フィラーで始まる発話が丸ごと 1 トークンとして削除され、出力が空になる（エラーは出ない）。この組み合わせ事故を型で防ぐのが目的
- `SimpleFillerFilter` — 辞書照合ベースのフィラー除去（階層化: pure / contextual / multi-word）。空白区切り言語（en / es / ko）専用
- ~~`EmbeddingFillerFilter` / `OnnxEmbeddingFilter`~~ — 埋め込みコサイン類似度によるフィラー検出。**v0.2.0 で削除済み**。実測でルールベース実装が全言語で上回ったため（[`model-upgrade-candidates.md`](./model-upgrade-candidates.md) §3.2）、Tier 1 のフィラー除去はルールベースのみとする。較正用の `bench_embedder` も併せて削除した（被写体が無くなったため）。測定値そのものは同ドキュメントに記録として残る
- `JapaneseFillerFilter` — 読点区切り3パス検出 + ASR アーティファクト対応
- `ChineseFillerFilter` — 中文 `，` 区切り 3 パス (pure: 嗯 / 呃 / 哦, contextual: 那个 / 这个 / 就是 / 然后 / 怎么说)

各フィルタの品質は言語ごとに独立して測る必要がある。実装があることは測定されていることを意味しない（§11.4）。

**設計判断: 埋め込みベースのフィラー除去**

当初、フィラー除去は LLM refinement の一部として設計されていたが、実験の結果、埋め込みモデルによるコサイン類似度判定で十分な精度が得られることが判明した:

- 純粋フィラー（um, uh, er）: **フィラー辞書に一致するか、または**フィラー辞書との最大コサイン類似度が閾値を超えるものを除去する。辞書が既知形のリコールを担保し、埋め込みが辞書外の変種（ummm, uhh, ASR アーティファクト）に汎化する
- 文脈依存フィラー（so, well, basically）: 文頭位置の場合のみフィラーとして判定し、文中では内容語として保持
- マルチワードフィラー（you know, I mean）: バイグラム照合で事前除去
- 日本語フィラー: 読点区切りセグメントの独立性を判定基準とし、ASR アーティファクト（えーと→映像）も辞書に含める

**コサイン閾値はモデル固有の値であり、バックエンド間で移植できない。** 実測（`bench_embedder`、v0.2.0 で削除）では最適値が bge-small-en-v1.5 = 0.80、granite-embedding-97m-multilingual-r2 = 0.90、potion-multilingual-128M = 0.38 と大きく散る。bge-small 由来の 0.82 を granite に流用すると英語の非フィラー 143 語のうち 131 語が false positive になる。較正手順と全測定値は [`model-upgrade-candidates.md`](./model-upgrade-candidates.md) §3 を参照。同 §3.1 に、この分岐が当初 AND 条件で書かれていたため閾値が全レンジで無効化されていた件の記録がある。

この手法は LLM 呼び出しに対して以下の利点を持つ:
- レイテンシ: ミリ秒単位（LLM は数百ミリ秒〜数秒）
- コスト: ゼロ（API 呼び出し不要）
- プライバシー: 完全オンデバイス
- 依存: ONNX モデル 33MB のみ（さらに辞書照合版は依存ゼロ）

### 3.6 Text Processor（中間処理層）

ASR 出力に対して句読点挿入・大文字化・自己訂正検出などの構造的な補正を行う。LLM より大幅に軽量な専用モデル（数 MB〜数十 MB の ONNX モデル）で動作する。

```
trait TextProcessor {
    /// テキストに構造的な補正を適用する
    fn process(text: &str, context: &ContextSnapshot) -> ProcessResult
}

struct ProcessResult {
    text: String,                  // 補正後のテキスト
    corrections: Vec<Correction>,  // 適用された補正（診断用）
}

struct Correction {
    kind: CorrectionKind,
    original: String,
    replacement: String,
    position: usize,
}

enum CorrectionKind {
    PunctuationInserted,     // 句読点の挿入
    Capitalized,             // 大文字化
    SelfCorrectionRemoved,   // 自己訂正（言い直し）の検出・除去
    ListFormatted,           // リスト構造の検出・整形
    DictionaryMatch,         // 固有名詞辞書マッチによる補正
    EntityDetected,          // NER によるエンティティ検出（テキスト変更なし、メタ情報）
}
```

**想定実装**:
- `PunctuationRestorer` — CNN-BiLSTM ベースの句読点・大文字化モデル（ONNX, ~5MB）
- `DisfluencyDetector` — 自己訂正検出モデル（reparandum/repair パターン検出, ~50MB）
- `PhonemeCorrector` — 音素距離 + テキスト埋め込みによるカスタム辞書補正（CMUdict + G2P ONNX + bge-small）
- `ParagraphSplitter` — 隣接文の類似度列の**谷**（局所最小 + 深さ `depth(i) = (left_peak − sim(i)) + (right_peak − sim(i))`）で分割する。深さは差分のみで決まるためバックエンド間のオフセットに不変。旧実装は絶対コサイン閾値 0.5 で、granite は無関係な文字列でも 0.62–0.75 を返すため意味的分割が一度も発火しなかった（[`model-upgrade-candidates.md`](./model-upgrade-candidates.md) §5.3）

  **実測済みの能力と限界**（同 §6、Wikipedia 由来コーパス・WindowDiff）:
  - **話題転換の検出は 5 言語すべてで機能する**。合成連結タスクで WD 0.122 (en) / 0.154 (ja) / 0.178 (zh) / 0.194 (ko) / 0.244 (es)、対して「分割なし」0.43–0.47、等間隔分割 0.45–0.51。既定の深さ比 0.5 が全言語で最良
  - **著者の段落構成は再現できない**。実記事タスクでは en が 0.499 で「分割なし」の 0.515 と誤差の範囲。記事内の段落分けは文体・長さの都合が支配的で話題がほとんど動かないため。**この層は「話者が話題を変えた位置」を切るものであって、書き手の段落感覚を再現するものではない**
  - 測定は granite のみ。Wikipedia の散文は dictation より整っているため §6.1 は楽観側。`min_similarity_range` は未検証。CI には入れていない（third-party のネットワーク取得のため）
- `OnnxEntityRecognizer` — NER トークン分類モデルによる固有表現検出（PER / LOC / ORG / MISC）。`dslim/distilbert-NER` が参照実装。**upstream は量子化グラフを配布しておらず、実サイズは fp32 で 249MB**（本節が挙げていた ~65MB INT8 は upstream に存在しない。量子化は利用側の作業）
- `RuleBasedProcessor` — ルールベースの整形（リスト検出、数値フォーマット等、依存ゼロ）

**設計根拠: LLM を使わない構造的補正**

調査の結果、従来 LLM refinement の責務とされていた処理の大部分が、専用の軽量モデルで代替可能であることが判明した:

| 処理 | LLM 必要？ | 軽量代替 | モデルサイズ |
|------|-----------|---------|------------|
| フィラー除去 | 不要 | 埋め込み距離 + 辞書（TextFilter 層で対応済み） | 33MB / 0MB |
| 句読点挿入 | 不要 | CNN-BiLSTM（Transformer の 1/40 サイズ、2.5 倍高速、同等精度） | ~5MB |
| 大文字化 | 不要 | 句読点モデルと joint training | 上記に含む |
| 自己訂正検出 | 不要 | ACNN / BERT-tiny による sequence labeling | ~50MB |
| 固有名詞補正 | 不要 | 音素距離 + テキスト埋め込み + CMUdict + G2P ONNX | ~250MB |
| 段落分割 | 不要 | 隣接文の埋め込みコサイン類似度 | bge-small 共用 |
| エンティティ検出 (NER) | 不要 | DistilBERT-NER トークン分類（全プラットフォーム共通） | 249MB (fp32) |
| フォーマット整形 | 一部不要 | リスト検出はルールベースで可能 | 0MB |
| 数詞正規化 (ITN) | 不要 | WFST / 規則ベース（`text-processing-rs`、en/ja/zh 稼働。es/ko は upstream 対応待ち） | ~0MB |
| 口語縮約展開 | 不要 | 規則ベース辞書（`SpokenFormNormalizer`、en 稼働。他言語は拡張余地あり） | 0MB |
| 口語→書き言葉（言い換え） | 一部不要 | 縮約展開は規則ベースで対応済み。残る自由な言い換えは編集タガー（未実装）または LLM | — |
| トーン調整 | 一部不要 | ja 敬体↔常体は規則化可（形態素解析器が前提）。自由な文体改変は LLM | — |

この知見に基づき、euhadra のテキスト処理パイプラインは 3 層構造となる:

1. **TextFilter 層**: フィラー除去。LLM 不要。完全オンデバイス
2. **TextProcessor 層**: 句読点挿入、大文字化、自己訂正検出、固有名詞補正、エンティティ検出（NER）、段落分割、**数詞正規化 (ITN)**、**口語縮約展開**。小型専用モデル（ONNX）/ 規則ベース / WFST。LLM 不要
3. **LlmRefiner 層**: 自由な文体の言い換え、トーン調整、コンテキスト適応書き換え、コマンド解釈・構造化出力。**オプション**。なお調査の結果これらも編集タグ付け／軽量分類で多くが非LLM化可能と判明したが、形態素解析器の依存追加（ja トーン調整）や訓練済みモデル（en/es 編集タガー、intent+slot）が前提となるため段階的に移行する（issue #73 / #74 / #75 参照。現時点ではいずれも保留中）

TextFilter + TextProcessor だけで「句読点付き・フィラーなし・自己訂正済み・固有名詞補正済み・段落分割済み」のクリーンテキストが得られ、これは ASR 生テキストのみの競合（Superwhisper, VoiceInk）を大きく超える品質となる。LlmRefiner は「さらに磨く」ためのオプション層として位置づけ、商用版の差別化ポイントにもなる。

**NER の設計方針: 全プラットフォーム共通の Tier 2 処理**

エンティティ検出（NER）は OS 組み込み LLM ではなく ONNX NER モデル（DistilBERT-NER）で全プラットフォーム共通に処理する。理由:

- **レイテンシ**: NER 専用モデルは ~10ms。LLM に NER させると数百ms
- **LLM 非依存**: Tier 2 は LLM なしで動くのが euhadra の設計原則
- **一貫性**: 全プラットフォームで同じ NER モデルを使えばエンティティ認識の挙動が統一される
- **基盤再利用**: OnnxPunctuationRestorer と同じ BERT トークン分類アーキテクチャ

NER の検出結果は PhonemeCorrector の候補範囲絞り込みに使用し、false positive を削減する:

```
ASR 出力: "I deployed the app to cooper nets yesterday"

NER なし: 全単語 × カスタム辞書 → "the" や "yesterday" も比較対象
NER あり: "cooper nets" がエンティティ → ここだけ辞書マッチ → "Kubernetes"
```

**絞り込みに使うのは「エンティティか否か」であって、クラスではない。** 実測では
`dslim/distilbert-NER` は "Kubernetes" を LOC と分類する（ORG でも MISC でもない）。
固有名詞の誤認識に対して NER のクラス判定が正しい保証はなく、そこに依存すると
絞り込みが落ちる。必要なのは「この範囲は固有表現らしい」という一段弱い信号である。

### 3.7 Voice Activity Detection（無音の除去と発話分割）

音声から発話区間を検出し、無音を ASR に渡さない。**ASR アダプタの前段**に置かれた独立ステージであり、`mic` feature の中ではない。WAV 入力にも同じ問題があるためで、キャプチャ側に埋めるとファイル利用者が同じ穴に落ちる（#133 の配置案 B）。

```
trait VadBackend {
    fn frame_size() -> usize
    fn required_sample_rate() -> Option<u32>
    fn start() -> Box<dyn VadStream>       // 検出パス1回分
}

trait VadStream {
    fn speech_probability(frame: &[f32]) -> f32   // 0.0 - 1.0
}

struct SpeechSegment { start: usize, end: usize }  // サンプル索引、半開区間
```

**「どのフレームが音声か」と「どこで発話を切るか」を分離する。** 前者が `VadBackend`、後者が `Segmenter`。リスクは後者に集中しており、バックエンドを良いものに替えても `Segmenter` の設定が悪ければ品質は改善しない。

**実装**:

| 実装 | 方式 | 依存 | サンプルレート |
|---|---|---|---|
| `EarshotVad` [`vad`] | 組込み NN 40 KiB（`earshot` クレート） | 純 Rust 1 クレート、`no_std`、モデルファイル無し | 16 kHz 固定 |
| `EnergyVad` | RMS レベル vs 適応ノイズフロア | ゼロ | 任意 |

`EarshotVad` を推奨とする。`EnergyVad` は「部屋より大きいか」を判定しているのであって「音声か」ではなく、キーボード・ドア・音楽が通過し、静かな話者はノイズフロアの上昇に飲まれる。doc コメントに stopgap である旨を明記している。

**クレート選定の根拠**（2026-08 時点、crates.io 実測）: `earshot` 1.2.1 は直近 DL 97,941、MIT OR Apache-2.0、依存は optional な `libm` のみ。ONNX ランタイムもモデル取得も不要なため `docs/model-licenses.md` の同梱モデル審査が発生しない唯一の候補である。作者は euhadra が `onnx` feature で使う `ort` の作者（pykeio）。対抗馬は `voice_activity_detector`（Silero via `ort`、DL 58,376）と `webrtc-vad`（DL 193,955 だが 2019 年から更新なし・C FFI）。#133 に挙げた `silero-vad-rs` / `zuoer` / `ryu-vad` / `ten_vad` / `flexaudio-vad` はいずれも DL 2〜3 桁で、**候補リスト自体が不完全だった**。

#### 分割ポリシー（`Segmenter`）

| 設定 | 既定値 | 役割 |
|---|---|---|
| `threshold` | 0.5 | 音声とみなす確率 |
| `min_speech` | 120 ms | 発話を開くのに必要な連続音声。ドアの音などで開かないための下限 |
| `min_silence` | **700 ms** | 発話を閉じるのに必要な連続無音。**ヒステリシスの要** |
| `speech_pad` | 200 ms | 境界の前後に残す音声。語頭・語尾の欠けは無音より高くつく |
| `max_speech` | 30 s | 一切休まない話者への安全弁。唯一意図的に過分割する設定 |

**既定値は非対称に倒してある。** 過分割は破壊的で復旧不能——#134 の実測では、英語発話の 3.0 秒 prefix が「However, due to the slow communication.」という流暢で完結して見える誤答を生み、1〜2 秒では "Yeah." / 「あっ。」という幻覚が出た。一方の過小分割はレイテンシしか損なわない。Silero 自身の既定 `min_silence` 100 ms は endpointing レイテンシ向けのチューニングであり、文中のポーズを切るため採用しない。

#### 逐次出力（`Session::partials`）

閉じた発話単位で transcript を返す。**これは streaming ASR ではない。** 同梱アダプタに streaming API は無く、伸びていく prefix の再転写は #134 で実測のうえ棄却した（churn 182% (en) / 350% (ja)、warm RTF 1.54 で実時間より遅い）。dictation の用途では発話単位が実用的な粒度でもある。

チャネルは lossy（§4.3 と同じ方針）で、受信側が読まなくてもセッションは詰まらない。receiver を drop すると発話単位の転写自体が省かれる。

#### `FinalPass` — 最終 transcript の出所

| ポリシー | 最終 transcript | ASR パス数 |
|---|---|---|
| `SpeechOnly`（既定） | 検出された音声を連結し、**1 発話として**転写 | 2 |
| `WholeUtterance` | 録音そのまま（無音込み） | 2 |
| `JoinSegments` | 発話ごとの transcript を連結 | 1 |

**2 つの失敗モードは分離できる。** 無音を ASR に渡すと幻覚が出る／発話を途中で切ると断片の誤答が出る——この 2 つは独立しており、**無音を落とすのに発話を切る必要はない**。既定の `SpeechOnly` は前者だけを行う。`WholeUtterance` は最終テキストを一切変えないため、ポリシー間の ΔWER 測定の基準線に使う。`JoinSegments` は最も安く、かつ分割誤りをそのまま被る唯一のポリシーである。

#### 測定（実施済み）

全結果は [`benchmarks/vad_delta_wer.md`](./benchmarks/vad_delta_wer.md)。FLEURS の en / ja 各 30 発話に前後 5 秒の無音を付加し、euhadra が実際に出荷しているモデル（en = canary-180m-flash INT8、ja = parakeet-tdt_ctc-0.6b-ja）で測った。clean 値は `ci_baseline.json` と完全一致する。

| 条件 | en (WER) | Δ | ja (CER) | Δ |
|---|---|---|---|---|
| clean・検出器なし | 0.0762 | — | 0.0724 | — |
| 無音付加・**検出器なし** | **0.1875** | **+0.1114** | 0.1211 | +0.0487 |
| 無音付加・`SpeechOnly` | 0.0762 | **+0.0000** | 0.0759 | +0.0035 |
| 無音付加・`JoinSegments`（−45 dBFS） | **0.3940** | **+0.3178** | 0.1376 | +0.0652 |

**3 点が確定した。**

1. **無音は実害を出す。** en で WER が 2.5 倍。30 発話中 7 件が肥大した
2. **既定構成がそれを消す。** `SpeechOnly` で Δ+0.0000〜+0.0150
3. **`SpeechOnly` と `JoinSegments` の差は分割の質ではなくポリシーだけで生じる。** 同一の segmentation（過分割 2.20）から en −45 で 0.0855 対 0.3940。「無音を落とすのに発話を切る必要はない」が実測で裏打ちされた

無音単独を転写させたときの出力がアーキテクチャの違いを直接示す:

| モデル | 出力 |
|---|---|
| canary (en, AED) | `".S. Sometimes it's a long way, …"`（約 50 回反復）/ 流暢な作話パラグラフ |
| parakeet (ja, TDT-CTC) | `"心の声。"` / `"うん。"` |

**上限を測る指標である点は変わらない。** 合成無音は実録音の環境音とスペクトルが違い、この条件は energy VAD に有利に働く。zh / ko / es は未測定。

#### 判断ゲート: テキスト側の除去は不要

#133 の分岐は「VAD 導入後も幻覚が残るか」だった。**残らない。** よって言語別ブラックリストには踏み込まない——§11.4 帰結 3 でいう「中央で広げられない」領域に落ちる対処を避けられた。

ただし AED 特有の残課題がある。`src/canary/decoder.rs` の `min_token_to_frame_ratio` は出力長が `0.2 × T_sub` に達するまで EOS を `-inf` にするが、この `T_sub` は**録音全体**のエンコーダフレーム数である。無音込みの録音では発話が正当化する量を超えてトークン生成が強制される。clean な FLEURS-es で truncation を潰すために調整された値であり、無音が入る運用では向きが逆になる。`SpeechOnly` は無音をエンコーダに渡さないため実害を回避しているが、ガード自体は発話フレーム基準に直すべきで、上限側のノブも無い（#136）。


---

## 4. Pipeline Runtime

### 4.1 Data Flow

```
[Activation Signal]
       │
       ▼
[Mic Capture] ─── Stream<AudioChunk> ───►[VAD / Segmenter] ───►[ASR Adapter]
                                       (発話単位に分割・無音を除去)
                                              │
                                        Stream<AsrResult>
                                              │
                                              ▼
                                       [Text Filter]
                                     (フィラー除去)
                                              │
                                              ▼
                                      [Text Processor]
                                  (句読点・自己訂正・整形)
                                              │
                              ┌────────────────┤
                              │                │
                    [Context Provider]         │
                         get_context()         │
                              │                │
                              ▼                ▼
                        ContextSnapshot + Processed Text
                              │
                              ▼
                    [LLM Refiner] (optional)
                     (トーン調整・書き換え)
                              │
                     Stream<RefinementOutput>
                              │
                              ▼
                      [Output Emitter]
                              │
                              ▼
                    [Deactivation / Idle]
```

TextFilter と TextProcessor は LLM なしで動作し、LLM Refiner はオプション。TextFilter + TextProcessor だけでも実用的な dictation 品質が得られる。

VAD ステージは省略可能で、設定しない場合は録音がそのまま ASR に渡る（0.2.0 までの挙動）。配置が `mic` feature の中ではなく ASR アダプタの前段なのは、WAV 入力にも同じ問題があるためである（§3.7）。

### 4.2 Streaming Strategy

ASR は partial result を連続的に返す。LLM refinement への投入タイミングには 2 つの戦略がある:

- **Final-only 戦略**: `is_final=true` の結果のみ LLM に渡す。シンプルだがレイテンシが大きい。Phase 1 のデフォルト。
- **Speculative 戦略**: partial result を debounce して LLM に投機的に渡し、final result で確定する。レイテンシは低いが LLM 呼び出しコストが増える。Phase 1 ではオプション。

### 4.3 Channel Design

ステージ間は tokio の bounded channel で接続する。

```
mic_capture ──[bounded(32)]──► asr_adapter
asr_adapter ──[bounded(8)]───► refinement_scheduler
refinement  ──[bounded(4)]───► output_emitter
```

- bounded channel により自然な backpressure が発生
- ASR が高速に partial result を吐いても refinement が詰まらない
- チャネルが full の場合、古い partial result を drop して最新のみ保持（lossy モード）

### 4.4 State Machine

```
         ┌────────────────────────────────────┐
         │                                    │
         ▼                                    │
      [Idle] ──(hotkey/VAD)──► [Activating]   │
                                    │         │
                                    ▼         │
                              [Recording]     │
                                    │         │
                              (speech end     │
                               / hotkey       │
                               release)       │
                                    │         │
                                    ▼         │
                             [Processing]     │
                                    │         │
                              (output ready)  │
                                    │         │
                                    ▼         │
                              [Emitting]      │
                                    │         │
                              (complete)      │
                                    │         │
                                    ▼         │
                               [Idle] ────────┘

         ※ どの状態からでも [Cancelling] → [Idle] に遷移可能
         ※ [Cancelling] は inflight の ASR / LLM リクエストを abort し、
            pending の output を破棄する
```

### 4.5 Cancellation

- `CancellationToken` を各ステージに伝播
- ユーザーが hotkey を離す / ESC / VAD 無音検出で発火
- inflight 中の ASR streaming と LLM リクエストを即座に abort
- Output Emitter に pending の挿入があればロールバック
- 全リソースは Rust の drop semantics で確実に解放

### 4.6 Error Handling & Fallback

各ステージは独立して失敗しうる。エラー時の振る舞い:

| ステージ | エラー例 | デフォルト動作 |
|---------|---------|--------------|
| ASR | API タイムアウト、ネットワーク断 | fallback ASR に切替 / ユーザー通知 |
| Context Provider | Accessibility 権限なし | 空の ContextSnapshot で続行 |
| LLM Refinement | API エラー、レート制限 | 生テキストをそのまま出力（graceful degradation） |
| Output Emitter | クリップボード失敗 | 代替手段（key emulation）にフォールバック |

---

## 5. On-Device Model Integration

> **スコープ注記**: 本章の LLM 部分（§5.2 / §5.3、および §5.1 の LLM 列）は **Phase 1 スコープ外**であり、`llm` feature ごと #122 で保留としている。現時点で `impl LlmRefiner` は `MockRefiner` のみ。ASR 部分（§5.1 の ASR 列）は稼働中で、本注記の対象外。

### 5.1 Platform Matrix

| Platform | ASR (on-device) | LLM (on-device) | LLM 統合方式 | Context API | Notes |
|----------|----------------|-----------------|-------------|-------------|-------|
| macOS | Apple Speech / Whisper.cpp / Parakeet ONNX | Apple Foundation Models (~3B) | Swift FFI → FoundationModels framework | AXUIElement | iOS 26 / macOS 26 以降。Guided generation、Tool calling、LoRA 対応 |
| iOS | Apple Speech / Parakeet ONNX | Apple Foundation Models (~3B) | Swift FFI → FoundationModels framework | UIAccessibility | Apple Intelligence 対応デバイス必須（iPhone 15 Pro 以降） |
| Android | Google Speech | Gemini Nano (AICore) | Kotlin JNI → AICore API | AccessibilityService | Android 14+、Pixel 8+ 等 |
| Windows | Whisper.cpp (DirectML) / Parakeet ONNX | Phi Silica (~3.3B, Phi 系) | C++ FFI → Windows App SDK / Windows AI Foundry | UI Automation | Copilot+ PC（40+ TOPS NPU）必須。Intel / AMD / Snapdragon 各シリコン向けに最適化版が提供 |
| Linux | Whisper.cpp (CUDA / CPU) / Parakeet ONNX | llama.cpp (任意 GGUF モデル) | C FFI → libllama（インプロセス） | AT-SPI (D-Bus) | Wayland 環境で制約あり。GGUF モデルはユーザーが選択（Phi-3.5, Qwen2.5, Gemma 2 等） |
| 全共通 | — | llama.cpp (フォールバック) | C FFI → libllama | — | OS 組み込み API が利用不可の場合の汎用フォールバック |

### 5.2 LLM Refiner 3-Tier Strategy

LlmRefiner trait の実装は 3 つの優先度層で構成され、ランタイムで動的に選択可能:

```
LlmRefiner trait
    │
    ├── Tier 1: OS 組み込み API（最優先、レイテンシ最小）
    │   ├── macOS/iOS:  AppleFoundationRefiner (Swift FFI)
    │   ├── Android:    GeminiNanoRefiner (Kotlin JNI)
    │   └── Windows:    PhiSilicaRefiner (C++ FFI → Windows App SDK)
    │
    ├── Tier 2: llama.cpp インプロセス（OS API 非対応時のフォールバック）
    │   └── 全プラットフォーム: LlamaCppRefiner (C FFI → libllama)
    │       ユーザーが任意の GGUF モデルを指定
    │
    └── Tier 3: クラウド API（オプション、最高品質）
        ├── CerebrasRefiner / GroqRefiner（低レイテンシ）
        └── OpenAiRefiner / AnthropicRefiner（高品質）
```

- **デフォルト: OS 組み込み API 優先** — ネットワーク不要、レイテンシ最小、プライバシー最大、推論コスト無料
- **フォールバック: llama.cpp** — OS API が利用不可の場合（Linux、古い OS バージョン等）。インプロセス推論でデーモン不要
- **オプション: クラウド API** — 最高品質が必要な場合にユーザーが明示的に選択
- **LLM なし** — Tier 1-2 TextFilter + TextProcessor のみで「80 点の dictation 体験」が成立。LLM は「80 点 → 95 点」のオプション層

### 5.3 OS 組み込み LLM API Bridge

各プラットフォームの OS 組み込み LLM は同一パターンで Rust コアから呼び出す: OS Shell 層に薄いネイティブラッパーを置き、「テキストを受けてテキストを返す」C ABI 関数として公開する。プロンプト構築ロジックは Rust 側に持たせる。

**Apple Foundation Models (macOS / iOS)**:
```
euhadra core (Rust)
    │  C ABI: euhadra_refine(raw_text, context) -> refined_text
    ▼
Swift Bridge Layer
    │  import FoundationModels
    │  LanguageModelSession.respond(to:generating:)
    ▼
Apple On-Device LLM (~3B)
```

Foundation Models framework は iOS 26 / macOS 26（2025年9月）でリリース済み。26.4（2026年3月）で instruction-following・tool-calling が改善。Swift との深い統合（`@Generable` マクロ、guided generation）を持つが、euhadra からは「文字列を受けて文字列を返す」薄いラッパーとして使用し、プロンプト構築は Rust 側で行う。

**Phi Silica (Windows)**:
```
euhadra core (Rust)
    │  C ABI: euhadra_refine(raw_text, context) -> refined_text
    ▼
C++ Bridge Layer
    │  Windows App SDK → LanguageModel API
    ▼
Phi Silica (~3.3B, NPU 最適化)
```

Windows App SDK 1.8（2025年後半〜）で Phi Silica API が公開。OS 組み込みでモデルバンドル不要。Copilot+ PC（40+ TOPS NPU）が必須。Intel / AMD / Snapdragon 各シリコン向けに個別最適化される。LoRA ファインチューン対応（preview）。

**Gemini Nano (Android)**:
```
euhadra core (Rust)
    │  JNI: euhadra_refine(raw_text, context) -> refined_text
    ▼
Kotlin Bridge Layer
    │  AICore API → GenerativeModel
    ▼
Gemini Nano (on-device)
```

**llama.cpp (全プラットフォーム共通フォールバック)**:
```
euhadra core (Rust)
    │  C FFI: llama_decode() → llama_sampling_sample() loop
    ▼
libllama (C/C++, インプロセス)
    │  GGUF モデル（ユーザー指定）
    ▼
CPU / GPU / NPU
```

llama.cpp は OS 組み込み API が利用できない環境（Linux、古い macOS/Windows/iOS、NPU 非搭載機）での標準フォールバック。外部デーモン不要のインプロセス推論。GGUF 形式で公開されているほぼ全てのオープンモデルが使用可能。

---

## 6. Text Processing Pipeline Detail

### 6.1 Three-Tier Processing Architecture

テキスト処理は 3 つの独立した層で構成される。各層はオプションであり、必要に応じて有効化 / 無効化できる。Tier 1 + Tier 2 だけで LLM なしの実用的な dictation が成立する。

```
ASR Output (raw text)
    │
    ▼
[Tier 1: TextFilter]  ← LLM 不要、ミリ秒、0〜127MB
    │  フィラー除去（um, uh, えーと, 嗯, 呃...）
    │  - FillerFilter::for_language: 言語→実装の型付き選択 ✅ 実装済み
    │  - SimpleFillerFilter: ルールベース（en/es/ko） ✅ 実装済み
    │  - JapaneseFillerFilter: ルールベース（日本語） ✅ 実装済み
    │  - ChineseFillerFilter: ルールベース（中国語）  ✅ 実装済み
    │
    ▼
[Tier 2: TextProcessor]  ← LLM 不要、数十ミリ秒、5〜250MB ONNX
    │  自己訂正検出・除去                              ✅ 実装済み
    │  句読点挿入・大文字化（BERT ONNX）              ✅ 実装済み [onnx]
    │  句読点挿入（ルールベース）                      ✅ 実装済み
    │  固有名詞補正（音素距離 + テキスト埋め込み）    ✅ 実装済み [onnx]
    │  - CMUdict IPA 辞書（124K 語）による音素引き
    │  - G2P ONNX（DeepPhonemizer）による OOV 音素生成
    │  - 音素 Levenshtein 距離 + bge-small 埋め込み複合スコア
    │  エンティティ検出 NER（DistilBERT-NER ONNX）   ✅ 実装済み [onnx]
    │  - PER / LOC / ORG / MISC のトークン分類
    │  - PhonemeCorrector の候補範囲絞り込みに使用
    │  段落分割（意味的距離 + 最大文数制約）           ✅ 実装済み [onnx]
    │  - 隣接文の埋め込みコサイン類似度による分割（gold 未整備）
    │  数詞正規化 ITN（text-processing-rs）           ✅ 実装済み
    │  - en: normalize_sentence / ja・zh: normalize_with_lang
    │  - es・ko は upstream 対応待ち（patches/ に同梱）
    │  口語縮約展開（gonna→going to 等）              ✅ 実装済み
    │  - SpokenFormNormalizer: 規則ベース辞書（英語）
    │
    ▼
[Tier 3: LlmRefiner]  ← オプション、数百ミリ秒〜秒 / 具象実装なし（#122 保留）
    │  自由な文体の言い換え                            ☐ 未実装（issue #122 保留）
    │  トーン調整（app_name / field_type に基づく）   ☐ 未実装（issue #75 保留）
    │  コンテキスト適応書き換え（field_content 文脈） ☐ 未実装（issue #74 保留）
    │  コマンド解釈・構造化出力（Phase 2）            ☐ 未実装（issue #73 保留）
    │
    ▼
Refined Output
```

Tier 1 + Tier 2 のみで「句読点付き・フィラーなし・自己訂正済み・固有名詞補正済み・段落分割済み・数詞正規化済み・口語縮約展開済み」のクリーンテキストが得られる。Tier 3（LlmRefiner）は自由な文体の言い換えなど、意味理解が必要な処理のためのオプション層。なお調査の結果、Tier 3 の各タスク（トーン調整・コンテキスト適応・コマンド解釈）も編集タグ付け／軽量分類で多くが非LLM化可能と判明しており、形態素解析器の依存追加や訓練済みモデルの整備が済み次第 Tier 2 へ段階的に移行する（issue #73 / #74 / #75）。

**推論エンジンの使い分け**:

| 用途 | エンジン | 理由 |
|------|---------|------|
| ASR (Parakeet TDT) | ONNX Runtime | encoder-decoder、固定構造の推論 |
| 句読点/埋め込み/G2P (BERT, bge-small, DeepPhonemizer) | ONNX Runtime | 分類/埋め込み、固定構造の推論 |
| LLM refinement | llama.cpp / OS 組み込み API / クラウド API | 自己回帰テキスト生成。GGUF モデル選択肢の豊富さ |

ONNX Runtime は「固定構造の推論」（分類、エンコーダ-デコーダ、埋め込み）に最適。LLM の自己回帰テキスト生成には KV cache 管理が本業の llama.cpp または OS 組み込み API を使用する。

### 6.2 Prompt Architecture（Tier 3: LlmRefiner）

LlmRefiner を有効化した場合のプロンプトは以下の要素で構成される。Tier 1-2 で既にフィラー除去・句読点挿入・自己訂正検出が済んでいるため、LLM の責務はトーン調整とコンテキスト適応に限定される:

```
[System Instructions]
  - Tier 1-2 で処理済みのため、フィラー除去・句読点付与は不要
  - トーン / スタイル調整が主責務
  - フォーマットルール: リスト整形、段落分割（Tier 2 で未処理の高度なもの）

[Context Block]
  - app_name / field_type から推定されるトーン / スタイル
  - field_content（既存テキスト、継続入力の文脈として）
  - custom_dictionary（固有名詞、技術用語のヒント）
  - custom_instructions（ユーザー定義の出力ルール）

[Input]
  - Tier 1-2 処理済みテキスト（既にクリーン）

[Output Format]
  - Phase 1: plain text
  - Phase 2+: JSON (RefinementOutput に準拠)
```

### 6.3 App-Specific Tone Mapping（Tier 3: LlmRefiner）

ContextSnapshot の `app_name` / `field_type` に基づいて、refinement プロンプトにトーン指示を注入する:

| field_type | トーン | 例 |
|-----------|--------|-----|
| `CodeEditor` | 技術的、簡潔、コメント / docstring 書式 | `// TODO: implement error handling` |
| `EmailCompose` | フォーマル、完全な文章 | `Dear team, ...` |
| `ChatMessage` | カジュアル、短文、句読点省略可 | `sounds good, lets do it` |
| `Terminal` | コマンド形式、改行最小 | `git checkout -b feature/auth` |
| `Document` | フォーマル、段落構成 | 適切な見出し・箇条書き |
| `SearchBar` | キーワード的、簡潔 | `rust async channel backpressure` |

### 6.4 Custom Dictionary Integration

ユーザー辞書の固有名詞補正は、Tier 2 の PhonemeCorrector が LLM なしで処理する。音素距離ベースのマッチングにより、ASR の音声的な誤認識を正確に補正できる:

```
処理フロー:
  ASR 出力 "import tensor flow" の各単語
    ↓
  1. CMUdict IPA 辞書（124K 語）で音素引き
     "import" → "ɪmpɔrt"  (辞書ヒット)
     "tensor" → 辞書にない → G2P ONNX で生成 → "tɛnɝfloʊ"
     "flow"   → "floʊ"    (辞書ヒット)
    ↓
  2. 隣接語マージ: "tensor"+"flow" → 音素連結 "tɛnɝfloʊfloʊ"
    ↓
  3. カスタム辞書エントリとの距離計算:
     vs "TensorFlow"("tɛnsɝfloʊ") → phoneme_sim=0.82, text_sim=0.91
     composite(α=0.7) = 0.7×0.82 + 0.3×0.91 = 0.85 ≥ threshold
    ↓
  4. 置換: "import TensorFlow for machine learning"
```

**スコアリング方式**:
- 基本: IPA 音素列の Levenshtein 距離（正規化類似度）
- 拡張（`onnx` feature）: `α × phoneme_similarity + (1-α) × text_embedding_similarity`
  テキスト埋め込み（bge-small）との複合スコアで、音素的に曖昧な候補を意味的に判別

**text_sim はバックエンドの下限で正規化してから blend する。** 生のコサインは
モデルごとに零点が違い（無関係な文字列が bge-small-en-v1.5 では ≈0.45、
granite-embedding-97m-multilingual-r2 では ≈0.70）、正規化 Levenshtein である
phoneme_sim と尺度が揃わない。そのままだと同じ α が各バックエンドで違う重み付けになり、
上の α=0.7 は実測で bge-small では 19 件中 3 件の補正を取りこぼしていた
（F1 0.882、granite は 1.000）。`similarity::rescale` で
`(cos − floor)/(1 − floor)` に写すと α=0.7 は**両バックエンドで F1 1.000** になる。
`floor` は `EmbeddingBackend` が load 後の初回利用時に自己測定する。

ただし正規化はアフィン変換であり、零点は揃うが上側の分布形状までは揃わない。
α ≤ 0.60（意味項が支配的な領域）では依然としてバックエンド間で差が出る。
移植可能なのは「運用に値する領域」に限られる。

**受理閾値は経路ごとに分かれる。** 正規化で下駄（floor 0.70 なら全スコアに
0.3 × 0.70 = 0.21）が外れるため複合スコアは正当に下がる。実測での最適値は
複合が 0.65、phoneme-only が 0.85（複合側の 0.65 を phoneme-only に適用すると
false positive が 2 件出る）。これらは別の量に対する別の閾値だが、旧 α 表と違い
**バックエンドごとの値は不要**。測定は
[`model-upgrade-candidates.md`](./model-upgrade-candidates.md) §5.2 を参照。

**OOV 語の音素生成**:
- CMUdict に載っていない語（固有名詞、技術用語等）は G2P ONNX モデル（DeepPhonemizer、59MB）で音素列を自動生成
- カスタム辞書エントリの音素はユーザーが JSON で事前指定、または G2P で自動生成

Tier 3（LlmRefiner）を有効化した場合は、カスタム辞書を LLM プロンプトの context block にも `preferred_terms` として注入し、Tier 2 で漏れた補正を LLM がカバーする二重構造とする。

### 6.5 将来検討: T5 ベースのスタイル変換による Tier 3 軽量化

> **更新メモ**: 本節は seq2seq（T5）による Tier 3 軽量化を検討したものだが、その後の調査で、Tier 3 の各タスクは「生成」ではなく**編集ベースのタグ付け／軽量分類**に再定式化することで、より小型・低レイテンシかつ自己回帰生成を伴わない形で非LLM化できる見込みが立った。具体的には: 口語縮約展開は規則ベースで実装済み（`SpokenFormNormalizer`）、数詞正規化は WFST/規則ベースで実装済み（`text-processing-rs`）、残るトーン調整・コンテキスト適応・コマンド解釈は issue #75 / #74 / #73 として整理（いずれも形態素解析器の依存追加や訓練済みモデルの整備が前提のため現時点では保留）。T5 ベースの seq2seq は、編集ベース手法で品質が不足する「自由な言い換え」に限った将来オプションとして残す。

現在 Tier 3（LlmRefiner、3B+ LLM）に委ねている処理の一部が、ファインチューン済み T5-small/base（60M〜220M パラメータ）のエンコーダ-デコーダモデルで代替できる可能性がある。

**既存の公開モデル例**:

| モデル | ベース | タスク | サイズ |
|--------|--------|--------|--------|
| `rajistics/informal_formal_style_transfer` | T5-base | カジュアル→フォーマル変換（英語） | ~240MB |
| `s-nlp/t5-informal` | T5-base | フォーマル→カジュアル変換（GYAFC dataset） | ~240MB |
| `it5/mt5-small-formal-to-informal` | mT5-small | フォーマル→カジュアル（イタリア語） | ~120MB |
| `erfan226/persian-t5-formality-transfer` | T5 | 口語→書き言葉（ペルシャ語） | ~240MB |

実例（`rajistics/informal_formal_style_transfer`）:
```
[Casual]  "I am quitting my job"
[Formal]  "I will be stepping down from my job."

[Casual]  "What do guys do to show that they like a gal?"
[Formal]  "What do guys do to demonstrate their affinity for women?"
```

**レイテンシ見積もり（30 語入力 → 20 トークン出力）**:

| モデル | エンコーダ | デコーダ（自己回帰） | 合計 |
|--------|-----------|---------------------|------|
| T5-small (60M) | ~2ms | 20 tok × ~1ms | **~22ms** |
| T5-base (220M) | ~5ms | 20 tok × ~3ms | **~65ms** |
| LLM 3B | ~50ms | 20 tok × ~33ms | **~710ms** |

T5-small なら ~22ms で変換でき、LLM の 30 分の 1。Tier 2 の処理時間レンジ（~35ms）に収まる。

**T5 がデコーダ only LLM より高速な理由**:
- エンコーダが入力全体を 1 回の forward pass で符号化（LLM は prefill で全入力を処理するが、パラメータ数が桁違い）
- デコーダは 60M〜220M パラメータで、3B LLM のデコーダの 1/15〜1/50 のサイズ。トークン生成ごとのメモリ読み出し量が少ない
- ただしデコーダ部分は LLM と同様に 1 トークンずつ自己回帰生成するため、出力長に比例してレイテンシが伸びる点は同じ

**euhadra への適用可能性と課題**:

もし実現すれば、パイプラインの全処理を非 LLM モデルで完結でき、LLM を完全にオプション化できる:

```
Tier 1:   TextFilter          — ルールベース、~1ms
Tier 2:   TextProcessor       — BERT/埋め込み（エンコーダ only）、~35ms
Tier 2.5: StyleTransformer    — T5-small seq2seq、~20-60ms
Tier 3:   LlmRefiner          — LLM 3B+、~700ms、オプション（意味理解が必要な高度な処理のみ）
```

ただし現時点では以下の課題があり、Phase 2 以降の研究テーマとして位置づける:

1. **訓練データの構築**: ASR 口語テキスト→書き言葉の並列コーパスが必要。LLM で合成データを大量生成し T5 をファインチューンするパイプライン（LLM で蒸留→ T5 で推論）が現実的
2. **field_type 別の多様なスタイル**: 既存公開モデルは「カジュアル↔フォーマル」の 1 ペアのみ。euhadra が必要とする field_type 別 6 パターン（Email/Chat/Code/Terminal/Document/SearchBar）は自前ファインチューンが必要
3. **日本語対応**: mT5 で多言語対応可能だが、日本語の口語→書き言葉並列コーパスは英語より乏しい
4. **品質**: T5-small（60M）では複雑な言い換えの品質が落ちる。T5-base（220M）が実用下限の可能性

**Phase 2 アクションプラン**:
- 商用 LLM（Claude / GPT-4o 等）で ASR 口語→書き言葉の並列コーパスを 10 万ペア規模で合成生成
- T5-small / T5-base を英語・日本語でファインチューン
- 品質が Tier 3 LLM の 80% 以上に達すれば、Tier 2.5 として本採用。LLM は「意味理解が必要な処理」（コマンド解釈、指示語解決等）のみに限定

---

## 7. OS Shell Specifications

本章は euhadra が出荷する層の仕様ではなく、**利用側アプリが OS 統合を実装する際の設計指針**である（§2.2 参照）。このうち §7.2 の Clipboard + Paste は `ClipboardEmitter`（`clipboard` feature）として euhadra 側に実装済みで、ネイティブコードを要しない。

### 7.1 Activation Subsystem

| Method | Description | Implementation |
|--------|-------------|---------------|
| Hotkey | グローバルキーバインド（押下で開始、離すと終了） | OS 固有のグローバルキー監視 |
| Push-to-Talk | 明示的な開始 / 終了操作 | ボタン押下 / 離し |
| VAD | Voice Activity Detection による自動開始 / 終了 | **`vad` モジュールとして Rust 側に実装済み**（§3.7）。ネイティブ不要 |

### 7.2 Text Insertion Strategy

| Method | Pros | Cons | 適用場面 |
|--------|------|------|---------|
| Clipboard + Paste | 最も互換性が高い | ユーザーのクリップボードを上書き | デフォルト |
| Key Emulation | クリップボード非破壊 | アプリによって挙動が異なる / IME と競合しうる | オプション |
| IME Integration | 日本語等の入力メソッドとの親和性 | OS / IME ごとに実装が必要 | 将来対応 |
| Direct Callback | 最も高速・確実 | アプリ側の統合が必要 | ライブラリ利用時 |

Phase 1 では Clipboard + Paste をデフォルトとし、クリップボードの退避 / 復元を行う:

1. 現在のクリップボード内容を退避
2. 整形テキストをクリップボードに書き込み
3. Cmd+V / Ctrl+V をエミュレート
4. 元のクリップボード内容を復元

---

## 8. Commercial Offering

### 8.1 OSS vs Commercial Boundary

**OSS (MIT / Apache 2.0)**:
- euhadra core（Rust パイプラインランタイム全体）
- 全 adapter trait 定義
- 主要 adapter の参照実装（Whisper, OpenAI, Cerebras, Groq, Apple FM 等）
- OS Shell の参照実装（macOS 優先、段階的に拡大）
- CLI ツール
- ドキュメント・サンプルコード

**Commercial (Managed API)**:
- 低遅延で正確なクラウド文字起こし API — 開発者が API キー 1 つで「音声バイナリを送ったら整形済みテキストが返る」エンドポイント
  - 内部では euhadra コアと同一パイプライン（ASR → Context → LLM refinement）をサーバー側で実行
  - ASR / LLM の選定・プロンプトチューニング・辞書管理を最適化済み
  - セルフホスト版との差別化は「設定不要で高品質」と「運用の手間ゼロ」
- チューニング済み refinement プロンプト / モデル（高品質な箱出し体験）
- ダッシュボード（使用量、精度メトリクス、辞書管理、プロンプト管理）
- エンタープライズ機能（SSO、監査ログ、SLA、専用インスタンス）

### 8.2 Moat Strategy

MIT / Apache ライセンスのため、コード自体による参入障壁は意図的に設けない。商用版の競争優位は以下に依拠する:

- **品質の蓄積**: refinement プロンプトの最適化、言語別チューニング、ドメイン辞書。コードをコピーしても再現できない
- **運用ペインの吸収**: ASR / LLM の API キー管理、レイテンシモニタリング、fallback 切替、バージョン間互換性維持
- **エコシステム速度**: adapter 追加、OS 対応拡大、コミュニティ PR マージの速度。先行者として「公式」であるブランド
- **統合テストの維持**: ASR / LLM / OS という 3 方向の外部依存の互換性テスト

---

## 9. Phase 1 Scope (MVP)

### 9.1 Goal

「80 点の Aqua Voice 体験が 10 行のコードで動く」こと。

ただしこの表現は**アプリ**の言い方であり、Phase 1 が実際に出荷するのは**ライブラリ**である（`.asr()` のみ必須、CLI はデモ）。「10 行」が指すのは §9.4 の `PipelineBuilder` 構成であって、完成した dictation アプリではない。§1.3 の positioning（開発者向け OSS フレームワーク）が正であり、OS Shell を Phase 1 から外したのはこの整理に沿う（§2.2 / #123）。

### 9.2 MVP Feature Set

**コアパイプライン**:
- [x] Rust コアパイプライン（tokio ベース非同期ランタイム）
- [x] ステートマシン（Idle → Recording → Processing → Emitting → Idle）
- [x] キャンセル伝播（CancellationToken）

**ASR Adapter**:
- [x] ASR adapter trait 定義
- [x] WhisperLocal（whisper.cpp subprocess、現状の zh デフォルト)
- [x] ParakeetAdapter（Parakeet TDT 0.6B、Rust-native ONNX 推論）[onnx]
   - **en: `parakeet-tdt-0.6b-v3` (FastConformer-TDT、128-mel)** — `load(dir)` で読む。L1 の en で whisper-tiny.en を置き換え、WER 8.4% → 7.5%、warm RTF 0.10 → 0.05 (PR #12)
   - **ja: `nvidia/parakeet-tdt_ctc-0.6b-ja` (Hybrid TDT-CTC、80-mel)** — `load_with_feature_size(dir, 80)` で読む。L1 の ja で whisper-tiny を置き換え、CER 42% → 3.3%、RTF 0.14 → 0.05 (PR #11)
   - v2 (en、128-mel) も `load(dir)` で同様に動作 (内部的に v3 と同じパス)
   - 80-mel サポートのため `parakeet-rs` を `penta2himajin/parakeet-rs@feature-size-injection` に git pin (`from_pretrained_with_feature_size` を追加した patch)
- [ ] Whisper Cloud (OpenAI API)

**Tier 1: TextFilter**:
- [x] TextFilter trait 定義
- [x] FillerFilter::for_language（Language → 実装の型付きディスパッチ）
- [x] SimpleFillerFilter（英語、ルールベース）
- [x] JapaneseFillerFilter（日本語、ルールベース）
- [x] ChineseFillerFilter（中国語、ルールベース）
- [x] ~~OnnxEmbeddingFilter（bge-small 埋め込み距離）[onnx]~~ — **v0.2.0 で削除**（§3.5 参照）

**Tier 2: TextProcessor**:
- [x] TextProcessor trait 定義
- [x] SelfCorrectionDetector（自己訂正検出・除去）
- [x] BasicPunctuationRestorer（ルールベース句読点）
- [x] OnnxPunctuationRestorer（BERT ONNX、句読点+大文字化）[onnx]
- [x] PhonemeCorrector（音素距離辞書補正、CMUdict 124K 語 + G2P ONNX + bge-small 複合スコア）[onnx]
- [x] ParagraphSplitter（意味的距離 + 最大文数制約）[onnx]
- [x] OnnxEntityRecognizer（DistilBERT-NER ONNX、PER/LOC/ORG/MISC 検出）[onnx]
- [ ] NER による PhonemeCorrector の候補絞り込み配線 — 未実装。processor 間でメタ情報を渡す経路が無いため、`PhonemeCorrector` 側に recognizer を注入する形（`with_g2p` / `with_embedder` と同じ合成）で行う

**Tier 3: LlmRefiner**:
- [x] LlmRefiner trait 定義
- [x] MockRefiner（passthrough / uppercase、テスト用）
- 具象実装（`LlamaCppRefiner` / OS 組み込み / クラウド）は **Phase 1 スコープ外**。`llm` feature ごと #122 で保留

  trait は残す。§12 のプロダクトパターンは全てここに乗るため、euhadra をプログラマブルにしている接続点そのものである。一方 §3.6 の表は LLM 列からほぼ全項目を追い出しており、残る「自由な言い換え」「トーン調整」は *dictation* ではなく **編集**にあたる。接続点を提供するのはインフラの仕事だが、そこに何を挿すかは意見であり、実装を同梱した時点で euhadra が editorial な立場を取ることになる。この線引き自体が #122 の主論点。

**Context Provider**:
- [x] ContextProvider trait 定義
- [x] MockContextProvider（手動コンテキスト指定）
- Accessibility 系の具象実装は **Phase 1 スコープ外**（#123）

  `ContextSnapshot` を実際に読んでいるのは `ParagraphSplitter` の `field_type` 1 フィールドのみで、残る 6 フィールドは未消費である。この構造体は §6.2 の「LLM プロンプトの Context Block」向けに設計されており、主な受益者が Tier 3 である以上 #122 の決着に従属する。なお `ParagraphSplitter` は `field_type` が `None` のとき分割する既定動作を持つため、Provider 不在でも動作する。

**Output Emitter**:
- [x] OutputEmitter trait 定義
- [x] StdoutEmitter
- [x] ClipboardEmitter（arboard）

**Voice Activity Detection**:
- [x] VadBackend / VadStream trait 定義
- [x] Segmenter（ヒステリシス付き発話分割、依存ゼロ）
- [x] EnergyVad（レベル判定、依存ゼロ、stopgap）
- [x] EarshotVad（組込み NN 40 KiB、`earshot`）[vad]
- [x] 逐次出力（`Session::partials`）と `FinalPass` ポリシー
- [x] **ΔWER の実測**（en / ja）— [`benchmarks/vad_delta_wer.md`](../docs/benchmarks/vad_delta_wer.md)
- [x] **判断ゲート**: 幻覚は VAD で消える。テキスト側の除去は不要
- [x] CI 配線 — `test (vad feature)` ジョブ + `onnx-check` の `Check (onnx + vad)` ステップ
- [ ] zh / ko / es の ΔWER — モデルバンドル未取得
- [ ] ΔWER 測定の CI 配線 — モデル取得が要るため `evaluate (ASR live smoke)` と同じ扱いになる

**CLI / 入力**:
- [x] CLI ツール（dictate / transcribe / record）
- [x] マイク入力（cpal）
- [x] WAV ファイル入力

**ドキュメント**:
- [x] README.md + Getting Started ガイド
- [x] 技術仕様書（spec.md）

**OS Shell**: Phase 1 スコープ外（#123）。§2.2 の通り euhadra が出荷する層ではない。4 責務のうちマイクキャプチャとテキスト挿入はクロスプラットフォーム Rust で実装済み、残る Accessibility と OS 組み込み LLM は #123 / #122 で保留。

### 9.3 MVP Non-Goals (Phase 2+)

- OS Shell 対応全般（macOS 含む）— euhadra の出荷物ではない。§2.2 / #123
- LlmRefiner 具象実装と `llm` feature — #122
- Command / StructuredInput 出力モード
- Streaming（speculative）戦略 — **partial result を投機的に流す形は引き続き Non-Goal**。同梱アダプタに streaming API が無いことを #134 で確認済み。逐次出力は §3.7 の発話単位（VAD 経由）に一本化した
- IME 統合
- 商用 API サーバー

※ 当初 Non-Goals としていた以下は Phase 1 で実装済み:
- オンデバイス ASR 統合 → WhisperLocal + ParakeetAdapter（ONNX）で実現
- Tier 2 テキスト処理 → 自己訂正検出、句読点挿入（ルール+ONNX）、固有名詞補正（音素距離）、段落分割（埋め込み距離）
- オンデバイス LLM → 一度 Phase 1 残タスクとしたが、#122 で再び Phase 1 スコープ外に戻した（責務の線引き自体が未決のため）
- VAD → §7.1 でネイティブ層の責務としていたが、`vad` モジュールとして Rust 側で実装（§3.7 / #133）。mic・clipboard と同じく「ネイティブ前提と決めつけない」（§2.2）が正しかった例

### 9.4 Target User Experience

最小構成は `PipelineBuilder` の doctest として compile 検証されている（`src/pipeline.rs`）。ここに転記したものが実際に動くコードであり、ドキュメント側だけが古くなることはない。

```rust
use euhadra::prelude::*;
use euhadra::whisper_local::WhisperLocal;

// LLM なしでも実用的な dictation が動く最小構成。
// 必須は .asr() のみ。context / emitter / refiner は省略可。
let pipeline = PipelineBuilder::new()
    .asr(WhisperLocal::new("whisper-cli", "ggml-base.bin"))
    .filter(FillerFilter::for_language(Language::English))
    .processor(SelfCorrectionDetector::new())
    .processor(BasicPunctuationRestorer)
    .emitter(StdoutEmitter)
    .build()?;
```

ONNX モデルを使った高品質構成（LLM なし）。`onnx` feature が必要で、こちらも `src/onnx_processing.rs` の doctest として compile 検証されている。埋め込みモデルが `phoneme` モジュール側にある点に注意:

```rust
use euhadra::prelude::*;
use euhadra::onnx_processing::OnnxPunctuationRestorer;
use euhadra::parakeet::ParakeetAdapter;
use euhadra::paragraph::ParagraphSplitter;
use euhadra::phoneme::OnnxTextEmbedder;

let pipeline = PipelineBuilder::new()
    .asr(ParakeetAdapter::load("models/parakeet-tdt-0.6b-v3")?)
    .filter(FillerFilter::for_language(Language::English))
    .processor(SelfCorrectionDetector::new())
    .processor(OnnxPunctuationRestorer::load(
        "models/punct/model.onnx",
        "models/punct/tokenizer.json",
        OnnxPunctuationRestorer::default_labels(),
    )?)
    .processor(
        ParagraphSplitter::new()
            .with_embedder(OnnxTextEmbedder::load("models/bge-small-en")?),
    )
    .emitter(StdoutEmitter)
    .build()?;
```

`FillerFilter::for_language` は言語ごとの分かち書き方式（英・西・韓は空白区切り、日は `、`、中は `，`）を型で選ぶ。**この対応付けを手で行うと、フィラーで始まる発話で出力が空になる**（`SimpleFillerFilter` は空白区切りのため、日本語の発話全体が 1 トークンとして削除される）。この事故を防ぐのが `for_language` の役割で、`FillerFilter` 以外を直接構築する場合は言語との対応を自分で保証する必要がある。

Tier 3（LlmRefiner）と `MacAccessibilityProvider` は未実装のため、上記に含めていない。実装されたら `.refiner(...)` / `.context(...)` を追加する形になる（§5.2 / §9.2 参照）。

### 9.5 Target Platforms (Phase 1)

- macOS (Apple Silicon) — 最優先
- CLI（プラットフォーム非依存、Context Provider = Manual）

---

## 10. Technical Decisions

### 10.1 Language Choice

| Component | Language | Rationale |
|-----------|----------|-----------|
| Core pipeline | Rust | 非同期ランタイム、trait 抽象、メモリ安全性、FFI 起点として最適 |
| macOS Shell | Swift | Apple API（Accessibility, Foundation Models）のネイティブアクセス |
| Android Shell | Kotlin | Android API（AICore, AccessibilityService）のネイティブアクセス |
| Windows Shell | C++ / C# | UI Automation / DirectML のネイティブアクセス |
| CLI | Rust | コアと同一言語、追加依存なし |

### 10.2 Key Dependencies (Rust Core)

| Crate | Purpose | Feature gate |
|-------|---------|-------------|
| `tokio` | 非同期ランタイム | — |
| `tokio::sync::mpsc` | bounded channel（ステージ間通信） | — |
| `tokio_util::sync::CancellationToken` | キャンセル伝播 | — |
| `async-trait` | trait の非同期メソッド | — |
| `serde` / `serde_json` | 構造化データのシリアライズ（IPA 辞書、設定） | — |
| `cpal` | クロスプラットフォームオーディオキャプチャ | — |
| `arboard` | クリップボード操作（ClipboardEmitter） | — |
| `earshot` | Voice Activity Detection（`EarshotVad`、組込み NN 40 KiB） | `vad` |
| `tracing` | 構造化ログ / メトリクス | — |
| `clap` | CLI 引数パーサ | — |
| `ort` | ONNX Runtime バインディング（Parakeet ASR, BERT 句読点, bge-small 埋め込み, G2P） | `onnx` |
| `ndarray` | 多次元配列（ONNX モデル入出力） | `onnx` |
| `tokenizers` | HuggingFace トークナイザ（BERT, bge-small） | `onnx` |
| `rustfft` | FFT（Parakeet メルスペクトログラム計算） | `onnx` |
| `llama-cpp-2` (予定) | llama.cpp C API バインディング（LlamaCppRefiner） | `llm` (予定) |

**Feature gate 設計**: デフォルトビルドは ML 依存ゼロ（ルールベース処理のみ）。`onnx` フラグで ONNX モデル推論を有効化。`llm` フラグ（予定）で llama.cpp 統合を有効化。

### 10.3 FFI Strategy

**現状: 未着工。消費者が現れるまで着工しない。**

`uniffi` / `extern "C"` / `#[no_mangle]` はリポジトリに存在しない。これは意図的な保留であり、根拠は #119 の経験にある。`AsrAdapter` はチャネルベースで設計されていたが、実アダプタが 9 個揃った段階でバッチ API へ作り直す必要が生じた。設計が不注意だったのではなく、**実装が揃うまで誤りが見えなかった**。

`ContextProvider` と `LlmRefiner` は現在まさにその状態にある（モック実装のみ）。この 2 つをまたぐ C ABI を先に確定させると、同種の誤りを修正コストの一桁高い層に固定してしまう。境界の設計は、注入される実物が少なくとも 1 つ存在してから行う。

以下は着工時の方針:

- **UniFFI** を第一候補とする（Kotlin / Swift / Python バインディングを自動生成）
- Apple Foundation Models: Swift 側に薄い C ABI ブリッジを手書き（UniFFI で表現しにくい Apple 固有型のため）
- Phi Silica: C++ 側に薄い C ABI ブリッジを手書き（Windows App SDK 固有型のため）
- llama.cpp: C ABI 直接リンク（libllama の C API はシンプルで FFI 生成不要）
- 全 FFI 関数はエラーを Result 型で返し、パニックを OS Shell 側に伝播させない

---

## 11. Testing Strategy

### 11.1 Unit Tests

各 adapter trait に対する mock 実装を用いてパイプライン全体をテスト:

- `MockAsr` — 固定テキストを返す / 指定タイミングで partial → final を返す
- `MockRefiner` — 入力をそのまま返す / 固定変換を返す
- `MockContextProvider` — 固定の ContextSnapshot を返す
- `MockEmitter` — 出力をバッファに蓄積

### 11.2 Integration Tests

- ステートマシン遷移の網羅テスト（正常系 + キャンセル + エラー）
- Backpressure テスト（ASR が高速に結果を返す場合の挙動）
- Cancellation テスト（各ステージでのキャンセル伝播の確認）
- Fallback テスト（ASR / LLM エラー時の graceful degradation）

### 11.3 E2E Tests (per platform)

- macOS: マイク入力 → ASR → refinement → クリップボード挿入の全フロー
- レイテンシ計測（activation → テキスト挿入までの end-to-end）
- メモリリーク / リソースリークの検出

### 11.4 Language Support Policy

**測れない言語は出さない。**

euhadra は言語ごとに品質が変わる。フィラー辞書は言語ごとの手書き、句読点や自己訂正の挙動も言語に依存する。したがって「対応言語」を名乗れるのは、**その言語で品質を測る手段があり、退行を検出できる**場合に限る。

#### 現状の測定範囲

| 層 | 測定対象 | gold / baseline がある言語 | CI で実際にゲートしている範囲 |
|---|---|---|---|
| L1 | ASR の WER / CER | en / ja / zh / es / ko | **5 言語すべて**（FLEURS 各 30 サンプル） |
| L3 | フィラー除去の直接 F1 | en / ja / zh / ko（es は後述の理由でツリー外） | **es のみ**（download-only、`--min-f1 1.0`） |
| L3 | 自己訂正の F1 | en / ja / zh / es / ko | なし |
| L3 | 固有名詞補正の F1 | **en のみ** | en のみ（`--min-f1 1.0`） |
| L2 | 段落分割の WindowDiff | 実測は granite のみ、CI 非搭載 | なし |

gold が存在することと CI がゲートしていることは別である。上表のとおり L3 で常時ゲートされているのは固有名詞補正の en だけで、それ以外は手動実行に留まる。**「gold がある」を「守られている」と読み替えてはならない。**

#### gold 自体の出自（未検証である）

上表の gold は大半が **Claude による下書きで、ネイティブ話者の人手レビューは未実施**である（[`annotations/guidelines.md`](../tests/evaluation/annotations/guidelines.md) §5）。対象は en / ja / zh / ko のフィラー、5 言語すべての自己訂正、en の固有名詞補正——**現在 CI とベースラインが依拠しているものほぼ全部**にあたる。

ここに反転がある。**人手アノテーション由来の唯一の言語（es、CIEMPIESS の綴り字マークアップ）が、測定されていない言語である。** 測定されている 4 言語の gold は LLM 下書きで未レビューという状態にある。

したがって現時点の「測れている」は二段階の弱さを持つ:

1. gold があっても CI がゲートしていない（上記）
2. gold 自体がネイティブ検証を経ていない（本項）

`guidelines.md` §5 は PR レビューでのネイティブ確認を依頼しているが、その依頼は `tests/` 配下にあり貢献者の目に触れにくい。導線の整備は [`CONTRIBUTING.md`](../CONTRIBUTING.md) で扱う。

#### 帰結 1: ライセンスが測定範囲を決めることがある

es のフィラー gold は「未整備」ではない。生成器 `scripts/build_es_filler_annotations.py` は存在する。しかし原データの CIEMPIESS Test が **CC-BY-SA 4.0** であり、派生アノテーションをツリーにコミットすると ShareAlike が euhadra 本体（MIT / Apache 2.0）へ伝播する。そのため download-only / 都度動的計算とし、生成物は git 追跡外の `data/cache/` に置き、**事実情報である F1 スコアのみ** `docs/benchmarks/` にコミットする運用になっている（[`evaluation.md`](./evaluation.md) §2.5）。

つまり es は「gold が作れない」のではなく「**gold を同梱できない**」。これは 100 言語規模へ広げる際に効いてくる制約で、逐語書き起こしを持つ音声コーパスには CC-BY-SA や NC が多い。**測定範囲を決めるのはデータの有無だけでなくライセンスでもある。**

#### 帰結 2: 実装済みと測定済みは一致しない

上記の運用は設計されているが、**CI への配線は未了**である。その結果 es のフィラー除去は実質的に無検証のまま推移し、**フィラーの直後にカンマがあると何も除去しない**という欠陥が長期間残った（句読点なしの CIEMPIESS 書き起こしを前提に実装・検証していたため）。発覚したのは 5 言語を手で流したときである。

download-only の言語については、**gold が同梱されないぶん CI 配線が唯一の防波堤**になる。配線されるまでは「測定されていない」と扱う。

この配線は完了した（`evaluate (es filler F1)`）。ただし**測っているものを正確に述べる必要がある**:

- 生成器 `build_es_filler_annotations.py` の語彙は `SpanishFillerFilter` と**バイト一致**する。したがって F1 1.0 は「Rust 実装が Python 生成器と一致する」ことを意味し、人間の判断と一致することは意味しない。**ドリフト検出であって品質測定ではない**
- CIEMPIESS の書き起こしには**句読点が一切ない**（3558 発話中 0 件）。よって #121 の欠陥クラス（句読点付きフィラー）はこのゲートでは原理的に検出できない
- PURE 語彙 15 語のうち**実際に出現するのは `e` のみ**（1365 スパン）。残る 14 語（eh / ehm / mm / hmm / ah / oh 等）は 1 度も現れず、そこでのドリフトは不可視である。実質的に行使されているのは `e`、`o sea`（199）、繰り返し検出（rep 1218 / rep2 263）、部分語（66）
- ゲートが機能することは削除実験で確認済み: `e` を落とすと span F1 が 0.719 に下がり CI が落ちる。`eh` を落としても何も変わらない

すなわち **「CI が通っている」は「es のフィラー除去が正しい」ではなく、「実装が生成器と一致し、実際に行使される経路に退行がない」**である。この区別を曖昧にしないこと。

新しい言語のフィルタや処理を追加する際は、gold と baseline を同時に用意するか、**測定されていないことを明示する**。

#### 帰結 3: 層ごとに到達可能な言語数が違う

「euhadra は N 言語対応」という単一の数字は書けない。gold の作られ方が層によって異なり、スケールの仕方が根本的に違うためである。

| 層 | gold の作り方 | 現実的な上限 | 律速 |
|---|---|---|---|
| L1 ASR (WER / CER) | 既存ベンチをそのまま使う | **100+** | ほぼ無し（FLEURS 102 言語を既に使用） |
| Tier 2 句読点・大文字化・ITN・口語縮約 | **逆変換で合成**（整形済みテキストから句読点を剥がせば入力、元が正解） | **数十〜百** | 無し。テキストコーパスがあれば足りる |
| Tier 1 フィラー・自己訂正 | 逐語書き起こしコーパス、または人手アノテーション | **10 未満** | コーパスの存在 + **ライセンス** + ネイティブ話者 |
| Tier 3 言い換え・トーン調整 | 正解が存在しない | — | 品質は測れない（後述） |

**Tier 2 は中央で広げられる。** 破壊的変換の逆なので人手もアノテーションも LLM も要らない。自己教師ありの手法は 85 言語規模の実績がある。**euhadra の最大の伸びしろはここ**にある。ただし合成した「話し言葉」は実際の ASR 出力と分布が異なるため、**上限を測る指標であって実力の証明ではない**点は明示する。

**Tier 1 は中央で広げられない。** 何がフィラーかはネイティブ話者の判断であり、人工挿入では実際の言い淀みの分布を再現できない。既存の多言語 disfluency コーパス（DISCO の en/hi/de/fr 等）を合わせても 10 言語に届かない。したがって:

- **逐語コーパスが既にある言語** → euhadra 側で導出できる（es が実例。ライセンス運用は帰結 1）
- **無い言語** → ネイティブ話者の貢献が必須

コミュニティへ貢献を求める前に、対象言語について**逐語コーパスの棚卸しを先に行う**。既に導出可能な言語にまで人手を要求しないためである。また貢献を成立させるには、euhadra 側が「新言語の追加 = コード貢献（対象コーパス毎の生成スクリプト作成）」を「**データ貢献**」に変える必要がある。現在 `scripts/build_*_filler_annotations.py` は 5 本がそれぞれ個別実装になっている。

**Tier 3 は品質ではなく不変条件を測る。** 自由な言い換えに正解は無く、LLM 出力を gold に据えると「元の LLM への近さ」を測ることになる。研究側でも TST の自動評価は人間判断との相関が低いことが繰り返し報告されている。代わりに euhadra が守るべきは「壊していないか」であり、これは決定的に検査できる:

- 入力にあった固有表現が出力にも残っているか
- 数値（金額・日付）が保存されているか
- 否定が反転していないか
- 入力に無い固有名詞を追加していないか
- 長さが妥当な範囲か（切り落とし・要約が起きていないか）

品質メトリクスではなく安全網である。dictation の文脈では、語尾の硬さより**金額が変わることの方が深刻**であり、この優先順位は妥当と考える。

#### 帰結 4: モデルの対応言語数は選定基準として弱い

Tier 2 / Tier 3 にモデルを導入する検討では対応言語数が指標になりやすいが、**律速は gold データであってモデルではない**。201 言語のモデルを採用しても、euhadra が品質を主張できるのは測れる範囲だけである。

したがってモデル選定の優先順位は「対応言語数」より、**ライセンス（再配布可能か）** と **euhadra が現に測っている言語を含むか** に置く。言語数が効いてくるのは gold の整備が先行した後であり、それ以前は天井の高さでしかない。

この原則は #84（ASR を 100 言語超へ拡張するロードマップ）とも整合させる必要がある。ASR はアダプタ追加で言語が増えるが、**テキスト処理層は同じようにスケールしない**。両者の到達言語数が乖離すること自体は許容するが、乖離を隠さず記述する。

---

## 12. Product Patterns — 音声入力を起点に構築できるもの

euhadra のパイプラインは「音声 → ASR → LLM → 出力」の各ステージを adapter で差し替え可能な汎用構造であるため、dictation 以外にも多様なプロダクトパターンを同一フレームワーク上に実現できる。以下は代表的なパターンと、それぞれで差し替わるコンポーネントの対応関係を示す。

### 12.1 Voice Dictation（基本形）

Aqua Voice / TYPELESS / Wispr Flow 相当。Phase 1 MVP そのもの。

```
音声 → ASR → LLM(フィラー除去・文法補正・フォーマット整形) → テキスト挿入
```

- LLM Refiner: `RefinementMode::Dictation`
- Context Provider: アプリ別トーン調整
- Output Emitter: ClipboardEmitter / KeyEmulationEmitter

### 12.2 Real-time Translation（リアルタイム翻訳）

話した言語をリアルタイムで別言語に変換して出力する。LLM refiner のプロンプトを「翻訳+自然な文体への整形」に差し替えるだけで成立する。オンデバイス LLM で動けばネットワーク遅延なしで翻訳が完結する。

```
音声(言語A) → ASR(言語A) → LLM(翻訳 A→B + 整形) → テキスト出力(言語B)
```

- LLM Refiner: 翻訳プロンプト。source/target 言語ペアを設定
- Output Emitter: ClipboardEmitter / StdoutEmitter / 字幕表示UI

応用例:
- 多言語チャットでの即時翻訳入力（Slack で日本語で話して英語で投稿）
- 旅行中のリアルタイム会話翻訳
- 映像制作における多言語字幕のライブ生成

### 12.3 Voice Memo → Structured Notes（音声メモの構造化）

話しっぱなしの音声メモを、要約・構造化された Markdown / JSON に変換する。

```
音声(長時間) → ASR(streaming) → LLM(要約 + 構造化 + タグ付け) → ファイル出力
```

- LLM Refiner: 要約・構造化プロンプト。見出し生成、箇条書き化、アクションアイテム抽出
- Output Emitter: FileEmitter（Markdown / JSON / Notion API 等）

応用例:
- 散歩中のアイデアメモ → 帰宅後に構造化されたドキュメントが出来ている
- 1on1 の会話メモ → アクションアイテムと決定事項が自動抽出される
- ブレインストーミング → マインドマップ構造への変換

### 12.4 Voice → Command Execution（音声コマンド実行）

音声をアプリケーションのアクションとして解釈し実行する。Phase 2 の `RefinementOutput::Command` の実証。

```
音声 → ASR → LLM(意図解釈 → Command 構造化) → コマンド実行
```

- LLM Refiner: `RefinementMode::Command`。意図をアクション名+パラメータに分解
- Output Emitter: CommandEmitter（シェル実行 / API 呼び出し / OS オートメーション）

応用例:
- 「最新のログを tail して」→ `tail -f /var/log/app.log` 実行
- 「Alice にこの PR のレビューを依頼して」→ GitHub API 呼び出し
- 「明日の 10 時に歯医者の予約をカレンダーに入れて」→ カレンダー API 呼び出し

### 12.5 Accessibility Input Layer（アクセシビリティ入力層）

身体的制約によりキーボード/マウスの使用が困難なユーザー向けに、音声をあらゆるアプリケーションの入力として機能させる汎用入力層。

```
音声 → ASR → LLM(意図解釈: テキスト / ナビゲーション / 操作) → 適切な出力
```

- LLM Refiner: テキスト入力・UI ナビゲーション・アプリ操作を統合的に解釈
- Output Emitter: テキスト挿入 / キーボードショートカット発行 / アクセシビリティ API 操作

応用例:
- RSI（反復性ストレス障害）を持つ開発者のコーディング支援
- 視覚障碍者向けのスクリーンリーダー連携音声操作
- 高齢者向けのシンプルな音声 PC 操作インターフェース

### 12.6 Voice Journal / Logging（音声ジャーナル・ログ）

日常の記録や業務ログを音声で蓄積し、検索・分析可能な構造化データに変換する。

```
音声 → ASR → LLM(分類 + メタデータ付与 + 感情分析) → データストア
```

- LLM Refiner: カテゴリ分類、タイムスタンプ、感情タグ、キーワード抽出
- Output Emitter: DatabaseEmitter（SQLite / API / ローカルファイル）

応用例:
- 業務日報の音声入力 → 自動分類・集計
- 育児記録（授乳・睡眠・体調）を声で記録 → 時系列データ化
- フィールドワークの音声観察記録 → 検索可能なデータベース

### 12.7 Live Captioning / Subtitling（ライブ字幕生成）

リアルタイムの音声を字幕として表示する。翻訳と組み合わせれば多言語ライブ字幕になる。

```
音声(continuous) → ASR(streaming) → LLM(整形 + 句読点 + 改行制御 [+ 翻訳]) → 字幕表示
```

- LLM Refiner: 字幕向け整形（文字数制限、改行位置、表示タイミング）
- Output Emitter: SubtitleEmitter（WebSocket / OBS 連携 / SRT 出力）

応用例:
- 配信者のリアルタイム字幕（日本語話者に英語字幕を付与）
- 聴覚障碍者向けのリアルタイム文字表示
- 国際会議での多言語ライブ字幕

### 12.8 Voice-Driven Development（音声駆動開発）

開発ワークフロー全体を音声でドライブする。dictation + command + コンテキスト理解の統合。

```
音声 → ASR → Context(IDE状態 + コードベース) → LLM(コード生成 / 編集指示 / Git操作) → IDE操作
```

- Context Provider: IDE 拡張からファイル構造・カーソル位置・エラー情報を取得
- LLM Refiner: コード生成、リファクタリング指示、コミットメッセージ生成を状況に応じて切替
- Output Emitter: IDE API / LSP 連携 / Git CLI

応用例:
- 「この関数にエラーハンドリングを追加して」→ コード差分生成
- 「今の変更を commit して、メッセージは変更内容から自動生成」→ git commit
- ハンズフリーでのペアプログラミング（一人が音声、一人がレビュー）

### 12.9 パターン横断の設計含意

上記パターンが全て同一パイプラインの adapter 差し替えで実現できることが、euhadra の抽象設計の正しさを証明する。特に重要な含意:

- **LLM Refiner の出力型の拡張性**: `TextInsertion` / `Command` / `StructuredInput` の enum 設計が、dictation から音声コマンド、構造化データ出力まで自然にカバーする
- **Output Emitter の多様性**: クリップボード、ファイル、API、データベース、WebSocket、IDE と、出力先が根本的に異なるパターンを同一 trait で扱える
- **Context Provider の価値**: 同じ音声でも、コンテキスト（IDE / メール / チャット / ターミナル）によって出力が全く変わることが、Context Provider の独立した抽象としての正当性を示す

### 12.10 開発戦略

フレームワークの設計を実プロダクトが駆動する形で進める:

1. **Phase 1 と同時**: Voice Dictation（12.1）を MVP として実装。コアパイプラインの trait 境界を実証
2. **Phase 1 完了後**: Real-time Translation（12.2）と Voice Memo → Structured Notes（12.3）を追加実装。LLM Refiner と Output Emitter の差し替えが実際に機能することを検証
3. **Phase 2**: Voice → Command（12.4）と Voice-Driven Development（12.8）で `RefinementOutput::Command` と IDE 統合を実証

各パターンの実装過程で発見された「コアに必要な抽象」はコアに還元し、「プロダクト固有のロジック」は adapter 実装側に留める規律を維持する。

---

## 13. Appendix

### A. Glossary

| Term | Definition |
|------|-----------|
| ASR | Automatic Speech Recognition — 音声をテキストに変換する技術 |
| VAD | Voice Activity Detection — 音声区間を検出する技術 |
| Refinement | ASR の生テキストを文法・フォーマット的に整形する処理 |
| Partial result | ASR のストリーミング中に返される未確定の認識結果 |
| Final result | ASR が確定した認識結果 |
| Backpressure | 下流が処理しきれない場合に上流の速度を制御する仕組み |
| Activation | 音声入力セッションの開始トリガー |
| OS Shell | euhadra core の外側にある、OS 固有のネイティブ実装層 |

### B. Competitive Landscape

#### B.1 Tier 1: クラウドベース・AI 整形あり（euhadra の直接競合）

| Product | ASR | LLM Refinement | Context | Pricing | Revenue Model | Platforms | Funding / Notes |
|---------|-----|----------------|---------|---------|--------------|-----------|----------------|
| Aqua Voice | Avalon (proprietary) | Proprietary | Accessibility API | Free(1,000語) / Pro $8/mo(年額$96) / Team $12/mo | B2Cサブスク + B2D API (Avalon API, 従量制) | macOS, Windows | YC W24。Avalon API は OpenAI Whisper API 互換のドロップイン ASR API として開発者に提供。デスクトップアプリのサブスクとAPI従量課金の2本柱 |
| TYPELESS | Undisclosed | Undisclosed | Undisclosed | Free(週4,000語) / Pro $12/mo(年額) or $30/mo(月額) | B2Cサブスク | macOS, Windows, iOS, Android | 100言語対応、オンデバイス処理を謳う。API/SDK提供なし |
| Wispr Flow | Undisclosed (cloud) | Proprietary | Screen capture | Free(週2,000語) / Pro $12-15/mo | B2Cサブスク | macOS, Windows, iOS, Android | $81M調達。SOC2 Type II / HIPAA対応。Whisper Mode（ささやき認識）。クロスプラットフォーム最強 |

#### B.2 Tier 2: オフライン/ローカル重視

| Product | ASR | LLM Refinement | Pricing | Revenue Model | Platforms | Notes |
|---------|-----|----------------|---------|--------------|-----------|-------|
| Superwhisper | Whisper (on-device) | なし（生テキスト）/ Custom mode で AI 処理可 | $8.49/mo / $249 lifetime | サブスク + 買い切り | macOS, iOS, Windows(限定) | 高カスタマイズ性。手動クリーンアップが必要 |
| Voibe | Whisper (on-device) | なし | $4.90/mo / $99 lifetime | サブスク + 買い切り | macOS only | 100%オフライン。Developer Mode（VS Code/Cursor IDE統合）。英語のみ |
| Spokenly | Whisper (local) + BYOK cloud | AI text processing (BYOK) | 無料（ローカル+BYOK） | 無料（収益モデル不明） | macOS, iOS | MCP統合で Claude Code 等と連携。Agent mode。BYOK で自前 API キーによるクラウド利用も無料 |
| VoiceInk | Whisper (on-device) | なし | $39 one-time / OSS | 買い切り + OSS | macOS | オープンソース（ソースからビルド可能）。ASR 単体、LLM refinement なし |

#### B.3 Tier 3: ファイル/会議特化・レガシー（euhadra と直接競合しない）

MacWhisper（音声ファイル文字起こし）、Otter.ai（会議特化・話者識別）、Notta（会議文字起こし）、Dragon Professional（$500+、レガシー）

#### B.4 euhadra の独自ポジション

| Product | ASR | LLM Refinement | Context | Pricing | Open Source |
|---------|-----|----------------|---------|---------|-------------|
| **euhadra** | **Pluggable（任意プロバイダー）** | **Pluggable（任意LLM）** | **Pluggable（OS API/OCR/Manual）** | **Free (OSS) + Commercial API** | **Yes (MIT/Apache 2.0)** |

既存プレイヤーは全て「完成品アプリ」であり、パイプラインの各ステージを差し替えたり、自前のモデルを持ち込んだり、セルフホストしたりする手段を提供していない。Spokenly が BYOK + MCP 統合で最も近いが、Mac 向けアプリであってフレームワークではない。VoiceInk は OSS だが ASR 単体で LLM refinement パイプラインを持たない。

「ASR → LLM refinement の統合パイプラインをプログラマブルに組める OSS フレームワーク」というポジションは現時点で空白である。

### C. References

- Aqua Voice: https://aquavoice.com/
- Avalon Model Card: https://app.aquavoice.com/research/avalon-model-card.pdf
- Avalon API: https://aquavoice.com/avalon-api
- TYPELESS: https://www.typeless.com/
- Wispr Flow: https://wisprflow.ai/
- Superwhisper: https://superwhisper.com/
- Voibe: https://www.getvoibe.com/
- Spokenly: https://spokenly.app/
- VoiceInk: https://github.com/nicepkg/VoiceInk (OSS)
- Apple Foundation Models: https://developer.apple.com/documentation/FoundationModels
- Gemini Nano: https://developer.android.com/ai/aicore
- Whisper: https://github.com/openai/whisper
- UniFFI: https://github.com/mozilla/uniffi-rs
