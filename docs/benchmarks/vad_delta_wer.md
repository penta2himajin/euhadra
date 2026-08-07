# VAD ΔWER — 無音の除去は幻覚を止めるか

**測定日**: 2026-08-07 / **ランナー**: `examples/eval_vad.rs` / **生データ**: [`vad_delta_wer.json`](./vad_delta_wer.json)

#133 の受入基準は VAD の F1 ではなく **ΔWER**、およびその後の判断ゲート——**テキスト側の幻覚除去が必要か**——である。本ドキュメントはその両方に答える。

## 結論

1. **無音は実害を出す。** euhadra が `en` で出荷している Canary では、無音を足しただけで WER が **0.0762 → 0.1875（Δ+0.1114）**、30 発話中 7 件が肥大した。
2. **VAD が既定構成でそれを消す。** `SpeechOnly` で **Δ+0.0000 〜 +0.0150**。
3. **テキスト側の除去機構は不要。** よって #133 の判断ゲートは「不要」で閉じる。言語別ブラックリストには踏み込まない。
4. **`JoinSegments` は使ってはならない場面がある。** Canary + 環境ノイズで **Δ+0.3178**、既定の 4 倍以上悪い。

## 測定条件

FLEURS サブセット（`en` / `ja` 各 30 発話）の各発話に、前後 5 秒ずつ無音を付加する。比較基準は無加工の録音を検出器なしで転写したもの。

| 言語 | モデル | アーキテクチャ | clean | `ci_baseline.json` |
|---|---|---|---|---|
| en | canary-180m-flash INT8 | attention encoder-decoder | 0.0762 (WER) | 0.0762 |
| ja | parakeet-tdt_ctc-0.6b-ja | Hybrid TDT-CTC | 0.0724 (CER) | 0.0724 |

**clean 値が CI ベースラインと完全一致する。** ハーネスが既存の測定と同じものを測っている確認になる。

**モデルは euhadra が実際に出荷しているものを使う。** 当初 `en` を Parakeet-v3 で測っていたが、それは euhadra が `en` に採用していないモデルであり、しかも**幻覚しない側のアーキテクチャ**だった（無音単独で空文字を返す）。採用モデルで測らなければ答えたことにならない。

## 無音だけを転写させると何が返るか

検出器と無関係な、モデル単体の性質である。

| モデル | −100 dBFS（デジタル無音） | −45 dBFS（環境ノイズ相当） |
|---|---|---|
| **canary (en)** | `".S. Sometimes it's a long way, sometimes it's a long way, a long way, a long way, …"`（約 50 回反復） | `". The first person to be able to help her, but wouldn't have to recover the burger that she had and would have gone back to the stairs of the other day or so, I would have to go back to the world when I was around."` |
| parakeet (ja) | `"心の声。"` | `"うん。"` |
| parakeet-v3 (en, 非採用) | `""` | `""` |

**失敗の形がアーキテクチャで決まる。** Transducer / CTC はフレーム同期で blank を持つため出力長が音響フレーム数に縛られ、混入しても短い。attention encoder-decoder はデコーダが EOS を出すまで生成を続けるため上限が無く、**退行ループ**（−100）と**流暢な作話**（−45）の 2 モードが両方出る。後者は読んで気付けない種類の誤りである。

## ΔWER / ΔCER

`inflated` は転写が参照より 25% 以上長くなった発話数（30 中）。`segments` は 1 録音あたりの検出発話数で、FLEURS は 1 ファイル 1 発話なので **1.00 が理想**、超過は過分割。

### en — canary-180m-flash INT8（WER）

| ノイズ | 検出器 | ポリシー | WER | Δ | segments | inflated |
|---|---|---|---|---|---|---|
| — | なし | — | 0.0762 | — | — | — |
| −100 | **なし** | whole | **0.1875** | **+0.1114** | 0.00 | **7** |
| −100 | energy | speech-only | 0.0762 | **+0.0000** | 1.03 | 2 |
| −100 | earshot | speech-only | 0.0784 | +0.0022 | 1.00 | 1 |
| −100 | energy | join | 0.0762 | +0.0000 | 1.03 | 2 |
| −100 | earshot | join | 0.0784 | +0.0022 | 1.00 | 1 |
| −45 | **なし** | whole | **0.1855** | **+0.1093** | 0.00 | **7** |
| −45 | energy | speech-only | 0.0855 | +0.0093 | 2.20 | 1 |
| −45 | earshot | speech-only | 0.0912 | +0.0150 | 2.03 | 0 |
| −45 | energy | **join** | **0.3940** | **+0.3178** | 2.20 | **16** |
| −45 | earshot | **join** | **0.3226** | **+0.2464** | 2.03 | **14** |

### ja — parakeet-tdt_ctc-0.6b-ja（CER）

| ノイズ | 検出器 | ポリシー | CER | Δ | segments | inflated |
|---|---|---|---|---|---|---|
| — | なし | — | 0.0724 | — | — | — |
| −100 | なし | whole | 0.1211 | +0.0487 | 0.00 | 1 |
| −100 | energy | speech-only | 0.0759 | +0.0035 | 1.00 | 1 |
| −100 | earshot | speech-only | 0.0743 | +0.0019 | 1.27 | 1 |
| −100 | energy | join | 0.0759 | +0.0035 | 1.00 | 1 |
| −100 | earshot | join | 0.0873 | +0.0149 | 1.27 | 1 |
| −45 | なし | whole | 0.0998 | +0.0274 | 0.00 | 1 |
| −45 | energy | speech-only | 0.0502 | **−0.0222** | 2.00 | 1 |
| −45 | earshot | speech-only | 0.0498 | **−0.0226** | 2.20 | 1 |
| −45 | energy | join | 0.1376 | +0.0652 | 2.00 | 1 |
| −45 | earshot | join | 0.1246 | +0.0522 | 2.20 | 2 |

## 読み取り

### `SpeechOnly` が既定である理由が、設計論ではなく数値で立つ

同じ分割から出発しても、最終 transcript の作り方だけで結果が変わる。en −45 の行を並べると:

| ポリシー | WER | 同一の segmentation から |
|---|---|---|
| `SpeechOnly` | 0.0855 | 音声を連結し **1 発話として**転写 |
| `JoinSegments` | 0.3940 | 発話ごとの transcript を連結 |

**4.6 倍の差が、分割の質ではなくポリシーだけで生じている。** 過分割（segments 2.20）は同じなのに、`SpeechOnly` はそれを吸収し `JoinSegments` は被る。`inflated` が 1 → 16 に跳ねているのが機序で、短い断片を渡された Canary のデコーダが作話している。

「無音を落とすのに発話を切る必要はない」という [`spec.md` §3.7](../spec.md) の主張は、これで実測に裏打ちされた。

### `JoinSegments` は安いが、AED では危険

ASR パスが 1 回で済む唯一のポリシーだが、**分割誤りを緩衝なしで被る**。Transducer (ja) では Δ+0.05〜0.07 に収まるのに対し、AED (en) では Δ+0.25〜0.32。断片に対する脆弱性がアーキテクチャで違うため、**モデルを替えると危険度が変わる**。逐次出力を見せるだけなら `SpeechOnly` のまま `Session::partials` を読めばよい。

### 検出器の優劣は小さい

較正を直したあとでは energy と earshot の差は Δ で 0.002〜0.006 程度で、ja −45 では earshot がわずかに上。**ここに大きな差は無い**——差が出たのは閾値を間違えていた間だけで、それは検出器の性能ではなく設定の問題だった（[`vad_threshold_sweep`](#付録-閾値スイープ) 参照）。

推奨が `EarshotVad` のままなのは実測差ではなく性質による: `EnergyVad` は「部屋より大きいか」を判定しているのであって、キーボード・ドア・音楽は本測定の合成ノイズには現れない。**合成無音は energy VAD に有利な条件である。**

### 過分割は −45 で起きている

`segments` が −100 で 1.00〜1.27、−45 で 2.00〜2.20。ノイズがあると発話中のポーズが相対的に「無音でない」と判定されにくくなり、切れやすくなる。`SpeechOnly` がこれを吸収しているので実害は出ていないが、**`min_silence` の既定 700 ms がノイズ下でも十分かは別途詰める余地がある**。

## 付録: 閾値スイープ

較正がバックエンドの性質であることを示した測定。en は Parakeet-v3、ja は Parakeet、いずれも −45 dBFS、`SpeechOnly`。

| threshold | en ΔWER | ja ΔCER |
|---|---|---|
| 0.05 | — | +0.0274 |
| 0.1 | −0.0060 | +0.0290 |
| 0.15 | — | +0.0142 |
| **0.2** | **−0.0111** | **−0.0226** |
| 0.3 | −0.0037 | +0.0083 |
| 0.5 | +0.0529 | +0.1263 |
| 0.7 | +0.2515 | — |
| 0.9 | +0.5945 | — |

0.15 未満では全フレームが音声判定になり、**検出器なしのベースラインと数値が完全一致する**（検出器は動いているが何も決めていない）。0.5 では逆に発話を取りこぼす。使える窓が両側から狭い。

`EarshotVad::default_threshold()` はこれを根拠に 0.2 とした。`EnergyVad` は ramp を 0.5 で交差するよう作ってあるため 0.5 のままでよい。

## 限界

1. **合成無音は上限を測る指標。** デジタル無音も一様ノイズも実際の室内騒音とスペクトルが違う。実録音の環境音は energy VAD に厳しく、earshot に有利に働くと予想されるが、**予想であって測定ではない**。
2. **2 言語のみ。** `zh` / `ko` / `es` は `vendor/` にモデルが無く未測定。`es` は Canary なので `en` と同じ挙動が予想されるが、これも予想である。
3. **1 発話 1 ファイルの素材。** FLEURS は朗読音声で、発話中のポーズが実際の dictation より少ない。過分割の評価としては楽観側。
4. **`inflated` は転写長の比較であって幻覚の判定ではない。** 長くなった理由が幻覚とは限らない。
5. **CI に入っていない。** モデルバンドルの取得が要るため、`evaluate (ASR live smoke)` と同じ扱いにするなら別途配線が必要。

## 再現

```bash
scripts/setup_canary.sh                       # CANARY_DIR=vendor/canary_en
scripts/setup_parakeet_ja.sh                  # vendor/parakeet_ja

cargo run --release --features onnx,vad --example eval_vad -- \
    --canary-en-dir   vendor/canary_en \
    --parakeet-ja-dir vendor/parakeet_ja \
    --langs en,ja \
    --out docs/benchmarks/vad_delta_wer.json
```

閾値スイープは `--detectors earshot --thresholds 0.1,0.2,0.3,0.5`。空の `--thresholds`（既定）は各バックエンドの `default_threshold()` を使う。
