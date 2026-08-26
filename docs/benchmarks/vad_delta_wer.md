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

## 派生した別件

**Canary の EOS ガードは発話フレーム基準（#136）。** `min_token_to_frame_ratio` は `0.2 × T_speech` まで EOS を抑え、`max_token_to_frame_ratio`（既定 0.8）で上限を切る。`T_speech` はアダプタがエネルギー推定した発話分のエンコーダフレーム数。既定の `SpeechOnly` は無音をエンコーダに渡さないため実害をさらに小さくする。

## 限界

1. **合成無音は上限を測る指標。** デジタル無音も一様ノイズも実際の室内騒音とスペクトルが違う。実録音の環境音は energy VAD に厳しく、earshot に有利に働くと予想されるが、**予想であって測定ではない**。
2. **2 言語のみ。** `zh` / `ko` / `es` は `vendor/` にモデルが無く未測定。`es` は Canary なので `en` と同じ挙動が予想されるが、これも予想である。
3. **1 発話 1 ファイルの素材。** FLEURS は朗読音声で、発話中のポーズが実際の dictation より少ない。過分割の評価としては楽観側。
4. **`inflated` は転写長の比較であって幻覚の判定ではない。** 長くなった理由が幻覚とは限らない。
5. **CI が守るのは既定構成のみ。** `evaluate (VAD ΔWER)` ジョブ（#137）は en / ja × −45 dBFS × `EarshotVad` × `SpeechOnly` の 1 行ずつしか判定しない。本ドキュメントの他の行——`JoinSegments`、`EnergyVad`、−100 dBFS、閾値スイープ——は**測定であって保護対象ではない**。既定値を選ぶための材料と、退行を防ぐための番人は別物である。

## CI ゲート

`evaluate (VAD ΔWER)` ジョブが以下を守る（#137）:

```
--langs en,ja --noise-db=-45 --detectors none,earshot --policies speech-only
--max-delta 0.03 --max-segments 3.0
```

**絶対誤り率ではなく Δ を見る。** 絶対値は ASR モデルの更新で動き、それは `ci_baseline.json` が既に見ている。ここで守るのは「検出器を前段に置いても悪化しない」という、他のどこも見ていない性質である。

**固定閾値で、ベースラインファイルは持たない。** ベースラインを置くと `--update-baseline` の運用が要る割に、Δ 固定閾値と比べて追加で捕まえるものがない。

`--max-segments` が別に要るのは、**`SpeechOnly` が過分割を吸収して Δ に出さない**ため。実測は 2.03 (en) / 2.20 (ja) で、分割が壊れても誤り率が動かない領域がある。

### 検証

| 確認 | 結果 |
|---|---|
| 決定性 | 同一構成 2 回で**全数値が一致**（差は実行時間のみ） |
| 退行検出 | `EarshotVad::default_threshold` を 0.5 に戻すと Δ+0.0150 → **+0.0448** でゲートが落ちる |
| ゲートが空振りしたとき | 判定対象 0 行なら失敗扱い。キー名の変更や検出器の不発が「成功」に見えないようにする |

**この 3 つ目が要るのは前例があるため。** es のフィラー F1 では、ゲートが通ることと欠陥を検出できることが別だった（PURE 語彙 15 語中、実際に出現するのは 1 語のみ）。「通っている」が「守られている」を意味しない状態を作らない。

### クロスマシン決定性 — 言語で割れる

初回 CI 実行で実測できた。**同一コミット・同一入力での手元と CI の差**:

| 値 | 手元 | CI | 差 |
|---|---|---|---|
| **ja** clean CER | 0.0724 | 0.0724 | **0** |
| **ja** 検出器なし Δ | +0.0274 | +0.0274 | **0** |
| **ja** earshot Δ | −0.0226 | −0.0226 | **0** |
| **ja** segments | 2.20 | 2.20 | **0** |
| **en** clean WER | 0.0762 | 0.0782 | +0.0020 |
| **en** 検出器なし Δ | +0.1093 | +0.0954 | −0.0139 |
| **en** earshot Δ | +0.0150 | +0.0069 | −0.0081 |
| **en** segments | 2.03 | 2.03 | **0** |

**ja はビット単位で一致し、en は一致しない。** 上で述べたアーキテクチャの分岐がここにも出ている——Canary の自己回帰貪欲デコードは、浮動小数点リダクション順序のわずかな差が 1 トークンの選択を変え、そこから出力が分岐する。Parakeet の TDT はフレーム同期なので分岐しようがない。

無音単独の転写がそれを直接示す。同じモデル・同じ入力で:

| | 出力 |
|---|---|
| 手元 | `". The first person to be able to help her, … or so, I would have to go back to the world when I was around."` |
| CI | `". The first person to be able to help her, … or so in the morning, when she was able to see her mother again, as well as her current cell phone."` |

前半は一致し、途中から分かれている。

**`segments` は両言語・両環境で完全一致。** 検出は純 Rust の `earshot` で、分割は整数ロジックなので浮動小数点の影響を受けない。ゲートの 2 本の柱のうち、片方は環境非依存である。

**閾値 0.03 の妥当性。** 観測された en の Δ 変動は約 0.008、実測最悪値は +0.0150（手元）。マージンは 0.015 で変動の約 2 倍。ただし**変動が今回たまたま有利な向きだっただけ**で、逆向きに +0.008 振れれば +0.023 になる。0.03 は余裕がある値ではなく、**AED の非決定性を吸収できる最小限**と理解すべきである。CI がこの線に近づいたら、閾値を緩めるのではなく **en を transducer で測る**（`--parakeet-en-dir`）方が筋が良い。

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
