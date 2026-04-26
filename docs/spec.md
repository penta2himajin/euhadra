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

**OS Shell (ネイティブ層)**
- マイクキャプチャと activation 制御（hotkey / VAD / push-to-talk）
- Accessibility API を通じたフォーカスアプリ情報・テキストフィールド内容の取得
- テキスト挿入（クリップボード / キーエミュレーション / IME 統合）
- OS 固有のオンデバイスモデル呼び出し（Apple Foundation Models 等）

**C ABI / UniFFI 境界**
- OS Shell と euhadra core の間のインターフェース
- OS Shell は euhadra core が定義した trait の具体実装を C ABI 経由で注入する
- euhadra core はプラットフォーム非依存のまま、OS 固有機能を利用可能

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
- `SimpleFillerFilter` — 辞書照合ベースのフィラー除去（階層化: pure / contextual / multi-word）
- `EmbeddingFillerFilter` — 埋め込みコサイン類似度によるフィラー検出（fastembed, ONNX, bge-small-en 33MB）
- `JapaneseFillerFilter` — 読点区切り3パス検出 + ASR アーティファクト対応
- `ChineseFillerFilter` — 中文 `，` 区切り 3 パス (pure: 嗯 / 呃 / 哦, contextual: 那个 / 这个 / 就是 / 然后 / 怎么说)

**設計判断: 埋め込みベースのフィラー除去**

当初、フィラー除去は LLM refinement の一部として設計されていたが、実験の結果、埋め込みモデルによるコサイン類似度判定で十分な精度が得られることが判明した:

- 純粋フィラー（um, uh, er）: フィラー辞書との最大コサイン類似度 > 0.82、通常語は < 0.76 で明確に分離
- 文脈依存フィラー（so, well, basically）: 文頭位置の場合のみフィラーとして判定し、文中では内容語として保持
- マルチワードフィラー（you know, I mean）: バイグラム照合で事前除去
- 日本語フィラー: 読点区切りセグメントの独立性を判定基準とし、ASR アーティファクト（えーと→映像）も辞書に含める

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
- `ParagraphSplitter` — 隣接文の埋め込みコサイン類似度による段落分割
- `EntityRecognizer` — NER トークン分類モデル（DistilBERT-NER ONNX, ~65MB INT8）による固有表現検出（PER / LOC / ORG / MISC）
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
| エンティティ検出 (NER) | 不要 | DistilBERT-NER トークン分類（全プラットフォーム共通） | ~65MB |
| フォーマット整形 | 一部不要 | リスト検出はルールベースで可能 | 0MB |
| トーン調整 | **必要** | 文全体の書き換えは LLM の領域 | — |
| 口語→書き言葉変換 | **必要** | パターンの幅が広く LLM が必要 | — |

この知見に基づき、euhadra のテキスト処理パイプラインは 3 層構造となる:

1. **TextFilter 層**: フィラー除去。LLM 不要。完全オンデバイス
2. **TextProcessor 層**: 句読点挿入、大文字化、自己訂正検出、固有名詞補正、エンティティ検出（NER）、段落分割。小型専用モデル（ONNX）。LLM 不要
3. **LlmRefiner 層**: トーン調整、コンテキスト適応書き換え、口語→書き言葉変換。LLM 必要。**オプション**

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
NER あり: "cooper nets" が ORG/MISC → ここだけ辞書マッチ → "Kubernetes"
```

---

## 4. Pipeline Runtime

### 4.1 Data Flow

```
[Activation Signal]
       │
       ▼
[Mic Capture] ─── Stream<AudioChunk> ───►[ASR Adapter]
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
    │  - SimpleFillerFilter: ルールベース（英語）     ✅ 実装済み
    │  - JapaneseFillerFilter: ルールベース（日本語） ✅ 実装済み
    │  - ChineseFillerFilter: ルールベース（中国語）  ✅ 実装済み
    │  - OnnxEmbeddingFilter: bge-small 埋め込み     ✅ 実装済み [onnx]
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
    │  エンティティ検出 NER（DistilBERT-NER ONNX）   ☐ 未実装 [onnx]
    │  - PER / LOC / ORG / MISC のトークン分類
    │  - PhonemeCorrector の候補範囲絞り込みに使用
    │  段落分割（意味的距離 + 最大文数制約）           ✅ 実装済み [onnx]
    │  - 隣接文の bge-small 埋め込みコサイン類似度による分割
    │
    ▼
[Tier 3: LlmRefiner]  ← LLM 必要、数百ミリ秒〜秒、オプション
    │  トーン調整（app_name / field_type に基づく）
    │  コンテキスト適応書き換え（field_content 文脈）
    │  口語→書き言葉変換
    │  自然言語→コマンド変換（Phase 2）
    │
    ▼
Refined Output
```

Tier 1 + Tier 2 のみで「句読点付き・フィラーなし・自己訂正済み・固有名詞補正済み・段落分割済み」のクリーンテキストが得られる。Tier 3（LlmRefiner）はトーン調整・文体変換・コマンド解釈など、意味理解が必要な処理のためのオプション層。

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

**OOV 語の音素生成**:
- CMUdict に載っていない語（固有名詞、技術用語等）は G2P ONNX モデル（DeepPhonemizer、59MB）で音素列を自動生成
- カスタム辞書エントリの音素はユーザーが JSON で事前指定、または G2P で自動生成

Tier 3（LlmRefiner）を有効化した場合は、カスタム辞書を LLM プロンプトの context block にも `preferred_terms` として注入し、Tier 2 で漏れた補正を LLM がカバーする二重構造とする。

### 6.5 将来検討: T5 ベースのスタイル変換による Tier 3 軽量化

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

### 7.1 Activation Subsystem

| Method | Description | Implementation |
|--------|-------------|---------------|
| Hotkey | グローバルキーバインド（押下で開始、離すと終了） | OS 固有のグローバルキー監視 |
| Push-to-Talk | 明示的な開始 / 終了操作 | ボタン押下 / 離し |
| VAD | Voice Activity Detection による自動開始 / 終了 | WebRTC VAD or Silero VAD |

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

### 9.2 MVP Feature Set

**コアパイプライン**:
- [x] Rust コアパイプライン（tokio ベース非同期ランタイム）
- [x] ステートマシン（Idle → Recording → Processing → Emitting → Idle）
- [x] キャンセル伝播（CancellationToken）

**ASR Adapter**:
- [x] ASR adapter trait 定義
- [x] WhisperLocal（whisper.cpp subprocess）
- [x] ParakeetAdapter（Parakeet TDT 0.6B v3、Rust-native ONNX 推論）[onnx]
- [ ] Whisper Cloud (OpenAI API)

**Tier 1: TextFilter**:
- [x] TextFilter trait 定義
- [x] SimpleFillerFilter（英語、ルールベース）
- [x] JapaneseFillerFilter（日本語、ルールベース）
- [x] ChineseFillerFilter（中国語、ルールベース）
- [x] OnnxEmbeddingFilter（bge-small 埋め込み距離）[onnx]

**Tier 2: TextProcessor**:
- [x] TextProcessor trait 定義
- [x] SelfCorrectionDetector（自己訂正検出・除去）
- [x] BasicPunctuationRestorer（ルールベース句読点）
- [x] OnnxPunctuationRestorer（BERT ONNX、句読点+大文字化）[onnx]
- [x] PhonemeCorrector（音素距離辞書補正、CMUdict 124K 語 + G2P ONNX + bge-small 複合スコア）[onnx]
- [x] ParagraphSplitter（意味的距離 + 最大文数制約）[onnx]
- [ ] OnnxEntityRecognizer（DistilBERT-NER ONNX、PER/LOC/ORG/MISC 検出 → PhonemeCorrector 候補絞り込み）[onnx]

**Tier 3: LlmRefiner**:
- [x] LlmRefiner trait 定義
- [x] MockRefiner（passthrough / uppercase、テスト用）
- [ ] LlamaCppRefiner（llama.cpp C FFI、汎用オンデバイス）
- [ ] クラウド refiner（OpenAI / Cerebras / Groq）

**Context Provider**:
- [x] ContextProvider trait 定義
- [x] MockContextProvider（手動コンテキスト指定）
- [ ] MacAccessibilityProvider（macOS AXUIElement API）

**Output Emitter**:
- [x] OutputEmitter trait 定義
- [x] StdoutEmitter
- [x] ClipboardEmitter（arboard）

**CLI / 入力**:
- [x] CLI ツール（dictate / transcribe / record）
- [x] マイク入力（cpal）
- [x] WAV ファイル入力

**ドキュメント**:
- [x] README.md + Getting Started ガイド
- [x] 技術仕様書（spec.md）

**OS Shell**:
- [ ] macOS 向け OS Shell（Accessibility API + Clipboard 挿入 + Apple Foundation Models）

### 9.3 MVP Non-Goals (Phase 2+)

- Windows / Linux / iOS / Android OS Shell 対応
- Command / StructuredInput 出力モード
- Streaming（speculative）戦略
- IME 統合
- 商用 API サーバー

※ 当初 Non-Goals としていた以下は Phase 1 で実装済み:
- オンデバイス ASR 統合 → WhisperLocal + ParakeetAdapter（ONNX）で実現
- Tier 2 テキスト処理 → 自己訂正検出、句読点挿入（ルール+ONNX）、固有名詞補正（音素距離）、段落分割（埋め込み距離）
- オンデバイス LLM → llama.cpp C FFI 統合を Phase 1 残タスクとして追加

### 9.4 Target User Experience

```rust
use euhadra::prelude::*;

// LLM なしでも実用的な dictation が動く最小構成
let pipeline = PipelineBuilder::new()
    .asr(WhisperLocal::new("whisper-cli", "ggml-base.bin"))
    .filter(SimpleFillerFilter::english())
    .processor(SelfCorrectionDetector::new())
    .processor(BasicPunctuationRestorer)
    .context(MockContextProvider::new())
    .emitter(StdoutEmitter)
    .build()
    .unwrap();

// ONNX モデルを使った高品質構成（LLM なし）
let pipeline_onnx = PipelineBuilder::new()
    .asr(ParakeetAdapter::load("parakeet-tdt-0.6b-v3-int8")?)
    .filter(OnnxEmbeddingFilter::load("bge-small-en")?)
    .processor(SelfCorrectionDetector::new())
    .processor(OnnxPunctuationRestorer::load("punct/model.onnx", ...)?)
    .processor(PhonemeCorrector::new(ipa_dict, custom_entries)
        .with_g2p(OnnxG2p::load("g2p")?)
        .with_embedder(OnnxTextEmbedder::load("bge-small-en")?, 0.7))
    .processor(ParagraphSplitter::new()
        .with_embedder(OnnxTextEmbedder::load("bge-small-en")?))
    .context(MockContextProvider::new())
    .emitter(ClipboardEmitter::new())
    .build()
    .unwrap();

// オプション: LLM を追加してトーン調整を有効化
let pipeline_with_llm = PipelineBuilder::new()
    .asr(WhisperLocal::new("whisper-cli", "ggml-base.bin"))
    .filter(SimpleFillerFilter::english())
    .processor(SelfCorrectionDetector::new())
    .processor(BasicPunctuationRestorer)
    .refiner(LlamaCppRefiner::new("phi-3.5-mini-q4.gguf")?)
    .context(MacAccessibilityProvider::new())
    .emitter(ClipboardEmitter::new())
    .build()
    .unwrap();
```

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
| `tracing` | 構造化ログ / メトリクス | — |
| `clap` | CLI 引数パーサ | — |
| `ort` | ONNX Runtime バインディング（Parakeet ASR, BERT 句読点, bge-small 埋め込み, G2P） | `onnx` |
| `ndarray` | 多次元配列（ONNX モデル入出力） | `onnx` |
| `tokenizers` | HuggingFace トークナイザ（BERT, bge-small） | `onnx` |
| `rustfft` | FFT（Parakeet メルスペクトログラム計算） | `onnx` |
| `llama-cpp-2` (予定) | llama.cpp C API バインディング（LlamaCppRefiner） | `llm` (予定) |

**Feature gate 設計**: デフォルトビルドは ML 依存ゼロ（ルールベース処理のみ）。`onnx` フラグで ONNX モデル推論を有効化。`llm` フラグ（予定）で llama.cpp 統合を有効化。

### 10.3 FFI Strategy

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
