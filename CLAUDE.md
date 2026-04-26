# euhadra 開発ガイドライン

## ビルド・テスト

```bash
mise exec rust -- cargo build                  # 通常ビルド
mise exec rust -- cargo build --features onnx  # ONNX機能付きビルド
mise exec rust -- cargo test                   # テスト実行 (unit + integration)
mise exec rust -- cargo clippy                 # lint
```

## TDD (テスト駆動開発)

機能追加・修正は Red-Green-Refactoring のサイクルに従う:

1. **Red**: 失敗するテストを先に書く
2. **Green**: テストを通す最小限のコードを実装する
3. **Refactor**: テストがグリーンの状態でコードを整理する

## 禁止事項

1. **テストの削除・無効化の禁止**: 既存テストを削除・スキップ・コメントアウトしてはならない
2. **CI設定の無断変更禁止**: CI設定ファイルをユーザーの明示的な指示なく変更してはならない
3. **テストを通すためのプロダクションコードの劣化禁止**: テストを通すことを目的としてプロダクションコードの品質を下げてはならない
4. **破壊的APIエンドポイントの追加禁止**: データを削除・破壊するAPIをユーザーの明示的な許可なく追加してはならない

テストが失敗する場合は、テストを削除・除外するのではなく、**プロダクションコードを修正してテストを通す**こと。

## アーキテクチャ

非同期パイプライン構成。各ステージはトレイトで抽象化:

```
Mic/WAV → AsrBackend → TextFilter → TextProcessor → [LlmRefiner] → OutputEmitter
```

- `traits.rs` — 4つのコアアダプタトレイト定義
- `pipeline.rs` — PipelineBuilder + async sessionランタイム
- `mock.rs` — テスト用モック実装

## コミット規約

- テスト全パス + warning ゼロを確認してからコミット
- コミットメッセージ末尾に `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`

## 長文出力（Stream idle timeout 対策）

Claude Code cloud session で `Stream idle timeout - partial response received` が発生する場合の予防策:

1. **段階的 Write/Edit**: 長いドキュメントやコードファイルを生成する際は、骨子（見出し・関数シグネチャ等）を先に Write し、各セクションを後続 Edit で埋める。一度に 200 行を超えるブロックを書こうとしない
2. **tool_use 直後の長文書き出し回避**: Read/Bash/Grep の直後に長文を生成するパターンで発火しやすい。ツール呼び出しと長文生成の間に短い説明を挟むか、ターンを分ける
3. **失敗時の挙動**: タイムアウトしてもファイル書き込み自体は完了している場合がある（git status で確認）。再実行前に状態を確認し、重複書き込みを避ける
