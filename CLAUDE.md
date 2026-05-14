# euhadra

## Overview

euhadra は、音声入力を汎用的なプログラマブル入力として扱うための OSS フレームワーク。ASR・テキスト後処理・OS 統合を非同期パイプラインとして Rust で提供する。詳細仕様は @docs/spec.md を参照。

## Project Structure

```
src/                  # Rust ライブラリ + CLI 本体
  traits.rs             # 4つのコアアダプタトレイト定義
  pipeline.rs           # PipelineBuilder + async session ランタイム
  filter.rs             # TextFilter 実装（フィラー除去）
  processor.rs          # TextProcessor 実装（自己訂正・句読点・ITN・口語縮約）
  mock.rs               # テスト用モック実装
  eval/                 # 評価ハーネス（L1/L2/L3 メトリクス）
  canary/ paraformer/ sensevoice/ parakeet.rs   # ONNX ASR アダプタ（onnx feature 配下）
docs/                 # 仕様・評価・ベンチマーク（@docs/spec.md ほか）
tests/                # 統合テスト
examples/             # 評価ランナー（eval_l1_smoke 等）
scripts/              # モデル取得・データ準備スクリプト
models/               # Alloy 形式モデル（euhadra.als）
patches/              # 外部リポジトリ向けパッチ（text-processing-rs の ko/es ITN）
```

## Development Setup

```bash
# Rust toolchain（rustup 推奨）

# cpal が ALSA を要求するため、Debian/Ubuntu 系では以下が必要
sudo apt-get install -y pkg-config libasound2-dev

# Pre-push hook（fmt / clippy を push 前に実行）。テンプレートは git-hooks/pre-push
# 注: 現状リポジトリは rustfmt 未適用のため、フックを有効化する前に一度
#     `cargo fmt --all` を実行してフォーマットを揃えること
git config core.hooksPath git-hooks
```

## Build & Test

```bash
cargo build                   # 通常ビルド
cargo build --features onnx   # ONNX ASR アダプタ込みビルド
cargo test                    # テスト実行（unit + integration）
cargo clippy --all-targets    # lint
```

検証コマンドは前提セットアップなしで実行でき、Claude が自己検証に使えること。

## Development Principles

機能追加・修正は TDD（Red → Green → Refactor）のサイクルに従う:

1. **Red**: 失敗するテストを先に書く
2. **Green**: テストを通す最小限のコードを実装する
3. **Refactor**: テストがグリーンの状態でコードを整理する

テストが失敗する場合は、テストを削除・除外するのではなく、**プロダクションコードを修正してテストを通す**こと。

- 規則ベースの後処理実装（`BasicPunctuationRestorer`, `SpokenFormNormalizer` 等）は「専用 ONNX モデル導入までの stopgap」という位置づけで統一する。doc コメントにその旨を明記する。

## Architectural Boundaries

非同期パイプライン構成。各ステージはトレイトで抽象化（`traits.rs`）され、OS 固有機能は薄いネイティブシェル経由でプラグインとして注入される。

```
Mic/WAV → AsrBackend → TextFilter → TextProcessor → [LlmRefiner] → OutputEmitter
```

- **3 層のテキスト処理**: TextFilter（Tier 1）/ TextProcessor（Tier 2、非LLM）/ LlmRefiner（Tier 3、オプション）。Tier 1+2 のみで LLM 非依存のクリーンテキストが得られる設計。詳細は @docs/spec.md §3.5
- **ONNX ASR アダプタ**（`canary/` `paraformer/` `sensevoice/` `parakeet.rs`）は `onnx` feature gate 配下に置く。default build は lean に保つ
- `mock.rs` はテスト専用。プロダクションコードから依存しない

## Prohibitions

1. **テストの削除・無効化の禁止**: 既存テストを削除・スキップ・コメントアウトしてはならない
2. **CI 設定の無断変更禁止**: CI 設定ファイルをユーザーの明示的な指示なく変更してはならない
3. **テストを通すためのプロダクションコード劣化の禁止**: テストを通すことを目的としてプロダクションコードの品質を下げてはならない
4. **破壊的 API エンドポイントの追加禁止**: データを削除・破壊する API をユーザーの明示的な許可なく追加してはならない

## Git Conventions

- **Conventional Commits**: `feat:` `fix:` `docs:` `refactor:` `test:` `ci:` `chore:`。スコープ可（`feat(processor):` 等）
- **ブランチ名**: Claude 主導の作業は `claude/<topic>`
- **コミット前提**: テスト全パス + warning ゼロを確認してからコミットする
- **Trailer**: Claude が author のコミットには次を付与する。モデル名・セッション情報は埋めない（必要なら本文へ）:

  ```
  Co-Authored-By: Claude <noreply@anthropic.com>
  ```

- **Pull Request**: `.github/PULL_REQUEST_TEMPLATE.md` に従い、関連 issue を `Closes #N` でリンクする。ドラフトではなく ready 状態で開く

## Stream Idle Timeout Mitigation

Claude Code cloud session で `Stream idle timeout - partial response received` が発生する場合の予防策:

1. **段階的 Write/Edit**: 長いドキュメントやコードファイルは骨子（見出し・関数シグネチャ等）を先に Write し、各セクションを後続 Edit で埋める。一度に 200 行を超えるブロックを書こうとしない
2. **tool_use 直後の長文書き出し回避**: Read/Bash/Grep の直後に長文を生成するパターンで発火しやすい。ツール呼び出しと長文生成の間に短い説明を挟むか、ターンを分ける
3. **失敗時の挙動**: タイムアウトしてもファイル書き込み自体は完了している場合がある。再実行前に `git status` で状態を確認し、重複書き込みを避ける

## Session Handoff

複数セッションにまたがる長期作業は GitHub issue でコンテキストを引き継ぐ。詳細は [`templates/docs/handoff-protocol.md`](https://github.com/penta2himajin/templates/blob/main/docs/handoff-protocol.md)。

- ラベル: `session-handoff`
- 1 ワークストリーム 1 issue（セッション単位ではない）
- セッション開始時に該当 handoff issue を読み、**Next action** をユーザーに確認してから実行する
- issue body は現在状態のみ（毎セッション上書き）。履歴は pinned "Session log" コメントに追記

## Internationalisation

README は英語が正。翻訳は言語ディレクトリではなく suffix file（`README.ja.md`）で管理し、[`templates/docs/i18n-policy.md`](https://github.com/penta2himajin/templates/blob/main/docs/i18n-policy.md) に従う。`CLAUDE.md` や engineering docs は英語のみ（翻訳対象外）。現時点で追跡中の翻訳ペアはない。
