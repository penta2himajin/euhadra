//! Autoregressive greedy decoder for Canary-180M-Flash.
//!
//! Loads `decoder-model.onnx` from the
//! [`istupakov/canary-180m-flash-onnx`](https://huggingface.co/istupakov/canary-180m-flash-onnx)
//! bundle and runs the standard greedy loop from
//! [`onnx-asr`](https://github.com/istupakov/onnx-asr)'s
//! `NemoConformerAED._decoding`:
//!
//! ```text
//! prefix (10 tokens) ──┐
//! encoder_embeddings ──┼─▶ decoder ──▶ logits, decoder_hidden_states
//! encoder_mask        ─┤              ↓
//! decoder_mems (KV)  ──┘    argmax(logits[:, -1])
//!                            │
//!                            ├─ EOS? ─▶ break
//!                            │
//!                            ▼ append next token
//!                       (decoder_hidden_states becomes next decoder_mems)
//! ```
//!
//! ## Decoder I/O
//!
//! Inputs (matching upstream NeMo / onnx-asr exactly):
//!
//! - `input_ids` — `[B, T_in]` i64. Full prefix on call 0, just the
//!   latest single token thereafter.
//! - `encoder_embeddings` — `[B, T_sub, D]` f32, from `CanaryEncoder`.
//! - `encoder_mask` — `[B, T_sub]` i64, from `CanaryEncoder`.
//! - `decoder_mems` — `[L, B, T_acc, H]` f32. KV cache; empty time
//!   dim on call 0, grows by 1 each step.
//!
//! Outputs:
//!
//! - `logits` — `[B, T_in, V]` f32.
//! - `decoder_hidden_states` — `[L, B, T_acc', H]` f32. Replaces
//!   `decoder_mems` for the next step.
//!
//! ## Prefix layout (10 tokens)
//!
//! ```text
//! [ " ",                                  // slot 0 — last-occurrence ▁
//!   <|startofcontext|>,                   // slot 1
//!   <|startoftranscript|>,                // slot 2
//!   <|emo:undefined|>,                    // slot 3
//!   <|<source_lang>|>,                    // slot 4 — set per-call
//!   <|<target_lang>|>,                    // slot 5 — = source for ASR
//!   <|pnc|> | <|nopnc|>,                  // slot 6
//!   <|noitn|>,                            // slot 7
//!   <|notimestamp|>,                      // slot 8
//!   <|nodiarize|>                         // slot 9
//! ]
//! ```

use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, Axis};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

use crate::traits::AsrError;

use super::vocab::Vocab;

pub const DECODER_INPUT_IDS: &str = "input_ids";
pub const DECODER_INPUT_ENCODER_EMBEDDINGS: &str = "encoder_embeddings";
pub const DECODER_INPUT_ENCODER_MASK: &str = "encoder_mask";
pub const DECODER_INPUT_DECODER_MEMS: &str = "decoder_mems";

pub const DECODER_OUTPUT_LOGITS: &str = "logits";
pub const DECODER_OUTPUT_HIDDEN_STATES: &str = "decoder_hidden_states";

/// Length of the static decoder prefix. Pinned so accidental layout
/// changes blow up at compile time rather than at inference.
pub const PREFIX_LEN: usize = 10;

/// Default cap on the autoregressive sequence length, matching the
/// `max_sequence_length` field of the istupakov canary config.json.
pub const DEFAULT_MAX_SEQUENCE_LENGTH: usize = 1024;

/// Default greedy-decode repetition penalty. `1.0` is a no-op; values
/// above 1 down-weight already-emitted tokens. The HuggingFace
/// transformers / NeMo defaults sit in 1.1–1.3; we pick **1.8** based
/// on the FLEURS-es 100-utt sweep recorded in
/// `docs/canary-integration.md` "End-to-end validation v3":
///
/// | penalty | WER | hard fails | repetition loops |
/// |---|---|---|---|
/// | 1.0 (off) | 35.37 % | 1 | 2 |
/// | 1.2 | 23.03 % | 1 | 1 |
/// | 1.5 | 14.63 % | 1 | 0 |
/// | **1.8** | **14.38 %** | 1 | 0 |
/// | 2.0 | 18.90 % | 1 | 1 (over-penalty resurfaces a loop) |
///
/// 1.5–1.8 hit the sweet spot; 1.8 narrowly wins on mean WER without
/// regressing the 41 / 100 clean utterances that already sit at
/// WER < 5 %. Set to `1.0` to disable.
pub const DEFAULT_REPETITION_PENALTY: f32 = 1.8;

/// Default minimum output-token-to-encoder-frame ratio. Greedy
/// decoding for Canary-180M-Flash sometimes emits `<|endoftext|>`
/// in the middle of a long utterance, dropping multi-word chunks
/// (~22 / 99 utt at 20–50 % WER on the FLEURS-es 100-utt FP32 smoke
/// even after repetition penalty 1.8). The min-length gate
/// suppresses the EOS logit until the generated suffix length
/// reaches `ratio × T_sub`, where `T_sub` is the encoder's output
/// frame count.
///
/// Sweep on FLEURS-es 100-utt FP32 with `repetition_penalty=1.8`
/// (recorded in `docs/canary-integration.md` "End-to-end
/// validation v4"):
///
/// | ratio | mean WER | 0–5 % | hard fails | loops |
/// |---|---|---|---|---|
/// | 0.0 (off) | 14.38 % | 41 | **1** | 0 |
/// | **0.2** | **13.63 %** | **41** | **0** | **0** |
/// | 0.3 | 15.26 % | 42 | 0 | 0 |
/// | 0.4 | 62.33 % | 26 | 0 | 3 (over-suppression triggers loops) |
/// | 0.5 | 96.32 % | 0 | 0 | 11 (catastrophic) |
///
/// 0.2 hits the sweet spot: kills the last hard-fail and slightly
/// improves WER without forcing the decoder to keep emitting on
/// short clips. Higher ratios overshoot, forcing the decoder to
/// run past natural end-of-speech and triggering repetition loops
/// the penalty alone can't catch. Set to `0.0` to disable.
pub const DEFAULT_MIN_TOKEN_TO_FRAME_RATIO: f32 = 0.2;

/// Per-call decoding knobs. Defaults to Spanish ASR with
/// punctuation-and-capitalisation enabled.
#[derive(Debug, Clone)]
pub struct DecodeOptions {
    /// Source language — `"en" | "de" | "fr" | "es"` (Canary-180M-Flash
    /// supports these four for ASR). Long-form aliases like `"spanish"`
    /// are accepted via `Vocab::language_token`.
    pub source_language: String,
    /// Target language for AST. For pure ASR, set equal to
    /// `source_language` (this is the upstream default).
    pub target_language: String,
    /// Emit punctuation + capitalisation (`<|pnc|>`) vs. raw
    /// lowercase (`<|nopnc|>`).
    pub pnc: bool,
    /// Hard cap on `prefix_len + generated_tokens`.
    pub max_sequence_length: usize,
    /// Greedy-decode repetition penalty. `1.0` disables; values > 1
    /// discount logits of tokens already emitted in the current
    /// utterance. See `DEFAULT_REPETITION_PENALTY`.
    pub repetition_penalty: f32,
    /// Minimum number of generated tokens, expressed as a fraction
    /// of the encoder's output frame count. The greedy step
    /// suppresses `<|endoftext|>` (sets its logit to `-inf`) until
    /// the suffix has emitted at least `ceil(ratio × T_sub)` tokens.
    /// `0.0` disables. See `DEFAULT_MIN_TOKEN_TO_FRAME_RATIO`.
    pub min_token_to_frame_ratio: f32,
}

impl DecodeOptions {
    pub fn for_asr(language: impl Into<String>) -> Self {
        let lang = language.into();
        Self {
            source_language: lang.clone(),
            target_language: lang,
            pnc: true,
            max_sequence_length: DEFAULT_MAX_SEQUENCE_LENGTH,
            repetition_penalty: DEFAULT_REPETITION_PENALTY,
            min_token_to_frame_ratio: DEFAULT_MIN_TOKEN_TO_FRAME_RATIO,
        }
    }
}

/// Result of a single decode call.
#[derive(Debug, Clone)]
pub struct DecodeOutput {
    /// Generated token ids, with the prefix stripped and angle-pipe
    /// special tokens (`<|...|>`) filtered out. Suitable for direct
    /// hand-off to `Vocab::decode`.
    pub tokens: Vec<u32>,
    /// Per-token log-probabilities aligned with `tokens` after the
    /// EOS mask. `tokens.len() == logprobs.len()`.
    pub logprobs: Vec<f32>,
}

/// Build the 10-token decoder prefix (i64-typed for direct ort
/// hand-off). Pure function — returns a fresh Vec each call.
pub fn build_decoder_prefix(
    vocab: &Vocab,
    opts: &DecodeOptions,
) -> Result<Vec<i64>, AsrError> {
    let space = vocab.last_id("\u{2581}").ok_or_else(|| AsrError {
        message: "vocab missing ▁ (used for the literal-space slot in the \
                  decoder prefix; onnx-asr maps `_tokens[\" \"]` to the \
                  last-occurring ▁ token id)"
            .into(),
    })?;
    let soc = vocab.soc()?;
    let sot = vocab.sot()?;
    let emo = vocab.id("<|emo:undefined|>").ok_or_else(|| AsrError {
        message: "vocab missing <|emo:undefined|>".into(),
    })?;
    let src_lang = vocab
        .language_token(&opts.source_language)
        .ok_or_else(|| AsrError {
            message: format!(
                "vocab has no language token for source_language={:?}",
                opts.source_language
            ),
        })?;
    let tgt_lang = vocab
        .language_token(&opts.target_language)
        .ok_or_else(|| AsrError {
            message: format!(
                "vocab has no language token for target_language={:?}",
                opts.target_language
            ),
        })?;
    let pnc = if opts.pnc { vocab.pnc()? } else { vocab.nopnc()? };
    let noitn = vocab.id("<|noitn|>").ok_or_else(|| AsrError {
        message: "vocab missing <|noitn|>".into(),
    })?;
    let notimestamp = vocab.id("<|notimestamp|>").ok_or_else(|| AsrError {
        message: "vocab missing <|notimestamp|>".into(),
    })?;
    let nodiarize = vocab.id("<|nodiarize|>").ok_or_else(|| AsrError {
        message: "vocab missing <|nodiarize|>".into(),
    })?;

    let prefix = vec![
        space, soc, sot, emo, src_lang, tgt_lang, pnc, noitn, notimestamp, nodiarize,
    ];
    debug_assert_eq!(prefix.len(), PREFIX_LEN);
    Ok(prefix.into_iter().map(|x| x as i64).collect())
}

/// Greedy argmax over the **last time-position** of a `[B, T, V]`
/// logits tensor. Returns one token id per batch element.
pub fn argmax_last_position(logits: &Array3<f32>) -> Vec<u32> {
    let (batch, time, vocab_size) = logits.dim();
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut best_v = u32::MAX;
        let mut best_score = f32::NEG_INFINITY;
        for v in 0..vocab_size {
            // Last time-position only.
            let score = logits[[b, time - 1, v]];
            if score > best_score {
                best_score = score;
                best_v = v as u32;
            }
        }
        out.push(best_v);
    }
    out
}

/// Suppress `<|endoftext|>` at the last time-position when the
/// suffix is too short relative to the encoder frame count. Sets
/// `logits[B-1, last, eos] = -inf` (which the greedy argmax can
/// never pick) when `suffix_len < ceil(ratio × n_encoder_frames)`,
/// otherwise leaves logits untouched.
///
/// `ratio == 0.0` is a no-op.
///
/// This gate addresses the chunk-dropout / hard-fail failure mode
/// observed on FLEURS-es where the greedy decoder picks `<|eos|>`
/// before consuming the full audio (`docs/canary-integration.md`
/// "Failure-mode inventory"). It does not address the inverse
/// failure (decoder runs past the natural end), which is bounded
/// by `max_sequence_length` and the repetition penalty.
pub fn suppress_eos_until_min_length(
    logits: &mut Array3<f32>,
    eos_token_id: u32,
    suffix_len: usize,
    n_encoder_frames: usize,
    ratio: f32,
) {
    if ratio <= 0.0 || n_encoder_frames == 0 {
        return;
    }
    let min_len = (ratio * n_encoder_frames as f32).ceil() as usize;
    if suffix_len >= min_len {
        return;
    }
    let (batch, time, vocab_size) = logits.dim();
    if batch == 0 || time == 0 {
        return;
    }
    let v = eos_token_id as usize;
    if v >= vocab_size {
        return;
    }
    let last_t = time - 1;
    for b in 0..batch {
        logits[[b, last_t, v]] = f32::NEG_INFINITY;
    }
}

/// Down-weight logits at the last time-position for tokens that
/// already appear in `history`. Mirrors HuggingFace transformers'
/// `RepetitionPenaltyLogitsProcessor`:
///
/// ```text
/// new_score = if old_score < 0 { old_score * penalty }
///             else              { old_score / penalty }
/// ```
///
/// `penalty == 1.0` or empty `history` is a no-op (the function
/// returns without touching `logits`). Each unique token id in
/// `history` is penalised exactly once, even if emitted multiple
/// times — this matches the upstream HF behaviour and keeps the
/// adjustment stable as the history grows.
///
/// Operates on the last time-position of `logits` because the
/// greedy step only consumes that slice; earlier positions are
/// either prefix tokens (not yet sampled) or already-emitted tokens
/// (irrelevant to the next sampling step).
pub fn apply_repetition_penalty(
    logits: &mut Array3<f32>,
    history: &[i64],
    penalty: f32,
) {
    if penalty == 1.0 || history.is_empty() {
        return;
    }
    let (batch, time, vocab_size) = logits.dim();
    if batch == 0 || time == 0 {
        return;
    }
    let last_t = time - 1;
    // Dedup token ids so each gets penalised exactly once. Allocates
    // for cleanliness; the prefix is short (≤ ~1024 tokens) so this
    // is negligible vs the encoder/decoder ONNX work.
    let mut seen = std::collections::HashSet::<i64>::new();
    for &tok in history {
        if tok < 0 || !seen.insert(tok) {
            continue;
        }
        let v = tok as usize;
        if v >= vocab_size {
            continue;
        }
        for b in 0..batch {
            let score = logits[[b, last_t, v]];
            let new_score = if score < 0.0 {
                score * penalty
            } else {
                score / penalty
            };
            logits[[b, last_t, v]] = new_score;
        }
    }
}

/// Strip the static prefix from a full-token sequence and drop any
/// remaining angle-pipe special tokens (`<|...|>`). Mirrors
/// `[id for id in tokens if not self._vocab[id].startswith("<|")]`
/// in onnx-asr.
pub fn strip_prefix_and_specials(
    all_tokens: &[u32],
    prefix_len: usize,
    vocab: &Vocab,
) -> Vec<u32> {
    if all_tokens.len() <= prefix_len {
        return Vec::new();
    }
    all_tokens[prefix_len..]
        .iter()
        .filter(|&&id| match vocab.piece(id) {
            Some(p) => !(p.starts_with("<|") && p.ends_with("|>")),
            None => false,
        })
        .copied()
        .collect()
}

/// `decoder-model.onnx` session wrapper.
pub struct CanaryDecoder {
    session: Mutex<Session>,
    /// `decoder_mems` first-axis size (number of decoder layers).
    /// Read once at load time so we can allocate the empty cache
    /// without re-introspecting the session per call.
    mems_layers: usize,
    /// `decoder_mems` last-axis size (per-layer hidden dim).
    mems_hidden: usize,
}

impl CanaryDecoder {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AsrError> {
        let path = path.as_ref();
        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(path))
            .map_err(|e| AsrError {
                message: format!("load Canary decoder {}: {e}", path.display()),
            })?;
        validate_decoder_io(&session, path)?;
        let (mems_layers, mems_hidden) = read_mems_shape(&session, path)?;
        Ok(Self {
            session: Mutex::new(session),
            mems_layers,
            mems_hidden,
        })
    }

    /// Run greedy decoding over the encoder outputs. `encoder_embeddings`
    /// shape `[1, T_sub, D]`, `encoder_mask` shape `[1, T_sub]`.
    /// Currently only batch_size = 1 is supported (matching the
    /// AsrAdapter trait's per-utterance contract).
    pub fn decode(
        &self,
        encoder_embeddings: &Array3<f32>,
        encoder_mask: &Array2<i64>,
        vocab: &Vocab,
        opts: &DecodeOptions,
    ) -> Result<DecodeOutput, AsrError> {
        if encoder_embeddings.shape()[0] != 1 {
            return Err(AsrError {
                message: format!(
                    "decoder currently supports batch_size=1 (got {})",
                    encoder_embeddings.shape()[0]
                ),
            });
        }
        if encoder_mask.shape()[0] != 1 {
            return Err(AsrError {
                message: format!(
                    "encoder_mask batch must match encoder_embeddings (got {})",
                    encoder_mask.shape()[0]
                ),
            });
        }

        let prefix = build_decoder_prefix(vocab, opts)?;
        let prefix_len = prefix.len();
        let max_len = opts.max_sequence_length.max(prefix_len + 1);
        let eos = vocab.eos()?;

        // batch_tokens grows by one per step; we own it as a flat Vec
        // for cheap append, then re-wrap into ndarray for ort.
        let mut batch_tokens: Vec<i64> = prefix;
        let mut logprobs: Vec<f32> = Vec::new();

        // Empty KV cache at start: (L, 1, 0, H).
        let mut decoder_mems: Array4<f32> =
            Array4::zeros((self.mems_layers, 1, 0, self.mems_hidden));

        let mut session = self.session.lock().map_err(|e| AsrError {
            message: format!("decoder session lock poisoned: {e}"),
        })?;

        while batch_tokens.len() < max_len {
            // input_ids: full prefix on call 0 (decoder_mems empty),
            // just the latest token thereafter.
            let input_ids: Array2<i64> = if decoder_mems.shape()[2] == 0 {
                Array2::from_shape_vec((1, batch_tokens.len()), batch_tokens.clone()).map_err(
                    |e| AsrError {
                        message: format!("input_ids reshape (initial): {e}"),
                    },
                )?
            } else {
                let last = *batch_tokens.last().unwrap();
                Array2::from_shape_vec((1, 1), vec![last]).map_err(|e| AsrError {
                    message: format!("input_ids reshape (step): {e}"),
                })?
            };

            let input_ids_v = Value::from_array(input_ids).map_err(|e| AsrError {
                message: format!("input_ids Value: {e}"),
            })?;
            let enc_emb_v = Value::from_array(encoder_embeddings.clone()).map_err(|e| AsrError {
                message: format!("encoder_embeddings Value: {e}"),
            })?;
            let enc_mask_v = Value::from_array(encoder_mask.clone()).map_err(|e| AsrError {
                message: format!("encoder_mask Value: {e}"),
            })?;
            let mems_v = Value::from_array(decoder_mems.clone()).map_err(|e| AsrError {
                message: format!("decoder_mems Value: {e}"),
            })?;

            let outputs = session
                .run(vec![
                    (DECODER_INPUT_IDS, input_ids_v.into_dyn()),
                    (DECODER_INPUT_ENCODER_EMBEDDINGS, enc_emb_v.into_dyn()),
                    (DECODER_INPUT_ENCODER_MASK, enc_mask_v.into_dyn()),
                    (DECODER_INPUT_DECODER_MEMS, mems_v.into_dyn()),
                ])
                .map_err(|e| AsrError {
                    message: format!("Canary decoder run: {e}"),
                })?;

            let logits_idx = output_index(&outputs, DECODER_OUTPUT_LOGITS).ok_or_else(|| {
                AsrError {
                    message: format!("decoder missing output {DECODER_OUTPUT_LOGITS}"),
                }
            })?;
            let mems_idx = output_index(&outputs, DECODER_OUTPUT_HIDDEN_STATES).ok_or_else(
                || AsrError {
                    message: format!("decoder missing output {DECODER_OUTPUT_HIDDEN_STATES}"),
                },
            )?;

            let mut logits: Array3<f32> = outputs[logits_idx]
                .try_extract_array::<f32>()
                .map_err(|e| AsrError {
                    message: format!("extract {DECODER_OUTPUT_LOGITS}: {e}"),
                })?
                .to_owned()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| AsrError {
                    message: format!("{DECODER_OUTPUT_LOGITS} rank: {e}"),
                })?;
            let new_mems: Array4<f32> = outputs[mems_idx]
                .try_extract_array::<f32>()
                .map_err(|e| AsrError {
                    message: format!("extract {DECODER_OUTPUT_HIDDEN_STATES}: {e}"),
                })?
                .to_owned()
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| AsrError {
                    message: format!("{DECODER_OUTPUT_HIDDEN_STATES} rank: {e}"),
                })?;
            decoder_mems = new_mems;

            // Repetition-penalty applies to the SUFFIX of the
            // current sequence — penalising prefix tokens (sot /
            // language / pnc / nodiarize / etc.) would suppress
            // legitimate emissions whose surface forms also appear
            // in the prefix region. Use only the post-prefix slice.
            let suffix_len = batch_tokens.len().saturating_sub(prefix_len);
            if opts.repetition_penalty != 1.0 && suffix_len > 0 {
                let suffix = &batch_tokens[prefix_len..];
                apply_repetition_penalty(&mut logits, suffix, opts.repetition_penalty);
            }

            // Min-length gate: forbid `<|endoftext|>` until the
            // suffix has consumed `ratio × T_sub` tokens. Suppresses
            // the chunk-dropout failure mode where greedy argmax
            // exits before the encoder frames are exhausted.
            if opts.min_token_to_frame_ratio > 0.0 {
                let n_enc_frames = encoder_embeddings.shape()[1];
                suppress_eos_until_min_length(
                    &mut logits,
                    eos,
                    suffix_len,
                    n_enc_frames,
                    opts.min_token_to_frame_ratio,
                );
            }

            let next_tokens = argmax_last_position(&logits);
            let next = next_tokens[0];
            if next == eos {
                break;
            }

            let next_logprob = logprob_of_token(&logits, 0, next);
            batch_tokens.push(next as i64);
            logprobs.push(next_logprob);
        }

        // Postprocess: drop prefix + special tokens, and align logprobs.
        let all_ids: Vec<u32> = batch_tokens.iter().map(|&x| x as u32).collect();
        let kept = strip_prefix_and_specials(&all_ids, prefix_len, vocab);
        // Map kept ids back to logprob positions: logprobs[i] corresponds
        // to all_ids[prefix_len + i]. Filter logprobs by the same special-
        // token predicate so `kept.len() == kept_logprobs.len()`.
        let kept_logprobs: Vec<f32> = all_ids[prefix_len..]
            .iter()
            .zip(logprobs.iter())
            .filter(|(id, _)| match vocab.piece(**id) {
                Some(p) => !(p.starts_with("<|") && p.ends_with("|>")),
                None => false,
            })
            .map(|(_, lp)| *lp)
            .collect();

        Ok(DecodeOutput {
            tokens: kept,
            logprobs: kept_logprobs,
        })
    }
}

/// Numerically stable per-token log-probability:
/// `logprobs[v] = logits[v] - logsumexp(logits)`.
fn logprob_of_token(logits: &Array3<f32>, batch: usize, token: u32) -> f32 {
    let (_, time, vocab_size) = logits.dim();
    let row: ArrayView2<f32> = logits.index_axis(Axis(0), batch);
    let last = row.index_axis(Axis(0), time - 1);
    let max = last.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return f32::NEG_INFINITY;
    }
    let sum_exp: f32 = (0..vocab_size).map(|v| (last[v] - max).exp()).sum();
    let logsumexp = max + sum_exp.ln();
    last[token as usize] - logsumexp
}

fn validate_decoder_io(session: &Session, path: &Path) -> Result<(), AsrError> {
    let want_in = [
        DECODER_INPUT_IDS,
        DECODER_INPUT_ENCODER_EMBEDDINGS,
        DECODER_INPUT_ENCODER_MASK,
        DECODER_INPUT_DECODER_MEMS,
    ];
    let got_in: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
    for name in &want_in {
        if !got_in.iter().any(|n| n == name) {
            return Err(AsrError {
                message: format!(
                    "decoder {} missing input {} (have: {got_in:?})",
                    path.display(),
                    name
                ),
            });
        }
    }

    let want_out = [DECODER_OUTPUT_LOGITS, DECODER_OUTPUT_HIDDEN_STATES];
    let got_out: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();
    for name in &want_out {
        if !got_out.iter().any(|n| n == name) {
            return Err(AsrError {
                message: format!(
                    "decoder {} missing output {} (have: {got_out:?})",
                    path.display(),
                    name
                ),
            });
        }
    }
    Ok(())
}

fn read_mems_shape(session: &Session, path: &Path) -> Result<(usize, usize), AsrError> {
    // The decoder_mems input has shape (L, B, T, H); L and H are
    // static, T is dynamic. We need L and H to allocate the empty
    // initial cache. Look up by name for robustness against ONNX
    // re-exports that reorder inputs.
    let input = session
        .inputs()
        .iter()
        .find(|i| i.name() == DECODER_INPUT_DECODER_MEMS)
        .ok_or_else(|| AsrError {
            message: format!(
                "decoder {} missing input {DECODER_INPUT_DECODER_MEMS}",
                path.display()
            ),
        })?;
    let shape = input.dtype().tensor_shape().ok_or_else(|| AsrError {
        message: format!(
            "decoder {} input {DECODER_INPUT_DECODER_MEMS} is not a tensor",
            path.display()
        ),
    })?;
    if shape.len() != 4 {
        return Err(AsrError {
            message: format!(
                "decoder {} input {DECODER_INPUT_DECODER_MEMS} expected rank 4, got {}",
                path.display(),
                shape.len()
            ),
        });
    }
    let layers = shape[0];
    let hidden = shape[3];
    if layers <= 0 || hidden <= 0 {
        return Err(AsrError {
            message: format!(
                "decoder {} input {DECODER_INPUT_DECODER_MEMS} has non-static \
                 layers/hidden dims (L={layers}, H={hidden})",
                path.display(),
            ),
        });
    }
    Ok((layers as usize, hidden as usize))
}

fn output_index(outputs: &ort::session::SessionOutputs<'_>, name: &str) -> Option<usize> {
    outputs.keys().position(|k| k == name)
}

/// Suppress an unused-import warning when the file is built without
/// the `Array1` constructor (kept for future test additions that
/// hand-craft 1-D logit slices).
#[allow(dead_code)]
fn _unused_array1_marker() -> Array1<f32> {
    Array1::zeros(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mini synthetic vocab covering the 10 prefix slots plus the
    /// duplicate-▁ quirk that drives `last_id` for the space slot.
    fn mini_vocab_text() -> String {
        let entries: Vec<(&str, u32)> = vec![
            // Control / special tokens
            ("<unk>", 0),
            ("<|nospeech|>", 1),
            ("<pad>", 2),
            ("<|endoftext|>", 3),
            ("<|startoftranscript|>", 4),
            ("<|pnc|>", 5),
            ("<|nopnc|>", 6),
            ("<|startofcontext|>", 7),
            ("<|noitn|>", 8),
            ("<|nodiarize|>", 9),
            ("<|notimestamp|>", 10),
            ("<|emo:undefined|>", 11),
            // Languages
            ("<|en|>", 12),
            ("<|de|>", 13),
            ("<|fr|>", 14),
            ("<|es|>", 15),
            // First ▁ (control-region SentencePiece sentinel — analogous
            // to id 1151 in the real vocab)
            ("\u{2581}", 16),
            // SentencePiece-region pieces, ending with a second ▁ that
            // is what the decoder prefix slot 0 resolves to.
            ("hello", 17),
            ("world", 18),
            ("\u{2581}", 19),
        ];
        let mut s = String::new();
        for (piece, id) in entries {
            s.push_str(&format!("{piece} {id}\n"));
        }
        s
    }

    #[test]
    fn prefix_layout_for_spanish_asr() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        let opts = DecodeOptions::for_asr("es");
        let p = build_decoder_prefix(&v, &opts).unwrap();
        assert_eq!(p.len(), PREFIX_LEN);
        // Slot 0 = LAST ▁ id (19, not 16).
        assert_eq!(p[0], 19);
        // Slot 1 = <|startofcontext|>
        assert_eq!(p[1], 7);
        // Slot 2 = <|startoftranscript|>
        assert_eq!(p[2], 4);
        // Slot 3 = <|emo:undefined|>
        assert_eq!(p[3], 11);
        // Slot 4 = source language <|es|>
        assert_eq!(p[4], 15);
        // Slot 5 = target language (= source for ASR)
        assert_eq!(p[5], 15);
        // Slot 6 = <|pnc|> when pnc=true (default)
        assert_eq!(p[6], 5);
        // Slot 7 = <|noitn|>
        assert_eq!(p[7], 8);
        // Slot 8 = <|notimestamp|>
        assert_eq!(p[8], 10);
        // Slot 9 = <|nodiarize|>
        assert_eq!(p[9], 9);
    }

    #[test]
    fn prefix_uses_nopnc_when_disabled() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        let mut opts = DecodeOptions::for_asr("en");
        opts.pnc = false;
        let p = build_decoder_prefix(&v, &opts).unwrap();
        assert_eq!(p[6], 6); // <|nopnc|>
    }

    #[test]
    fn prefix_supports_all_canary_languages() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        for (lang, expected_id) in [("en", 12), ("de", 13), ("fr", 14), ("es", 15)] {
            let opts = DecodeOptions::for_asr(lang);
            let p = build_decoder_prefix(&v, &opts).unwrap();
            assert_eq!(p[4], expected_id, "lang={lang}");
            assert_eq!(p[5], expected_id, "lang={lang}");
        }
    }

    #[test]
    fn prefix_target_language_independent_of_source() {
        // AST mode: source en, target es.
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        let opts = DecodeOptions {
            source_language: "en".into(),
            target_language: "es".into(),
            pnc: true,
            max_sequence_length: 1024,
            repetition_penalty: 1.0,
            min_token_to_frame_ratio: 0.0,
        };
        let p = build_decoder_prefix(&v, &opts).unwrap();
        assert_eq!(p[4], 12, "source = en");
        assert_eq!(p[5], 15, "target = es");
    }

    #[test]
    fn prefix_rejects_unsupported_language() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        let opts = DecodeOptions::for_asr("ja"); // not in mini vocab
        let err = match build_decoder_prefix(&v, &opts) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(err.message.contains("ja"), "{}", err.message);
    }

    #[test]
    fn prefix_errors_when_required_token_missing() {
        // Drop <|noitn|> — prefix builder must surface a useful error.
        let mut text = String::new();
        text.push_str("<unk> 0\n");
        text.push_str("<|nospeech|> 1\n");
        text.push_str("<pad> 2\n");
        text.push_str("<|endoftext|> 3\n");
        text.push_str("<|startoftranscript|> 4\n");
        text.push_str("<|pnc|> 5\n");
        text.push_str("<|nopnc|> 6\n");
        text.push_str("<|startofcontext|> 7\n");
        text.push_str("<|notimestamp|> 8\n");
        text.push_str("<|nodiarize|> 9\n");
        text.push_str("<|emo:undefined|> 10\n");
        text.push_str("<|en|> 11\n");
        text.push_str("\u{2581} 12\n");
        let v = Vocab::from_text(&text).unwrap();
        let opts = DecodeOptions::for_asr("en");
        let err = match build_decoder_prefix(&v, &opts) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(err.message.contains("<|noitn|>"), "{}", err.message);
    }

    #[test]
    fn argmax_last_position_picks_max_at_final_time() {
        // Single-batch logits with 2 time steps × 4 vocab.
        let logits = Array3::from_shape_vec(
            (1, 2, 4),
            vec![
                // t=0 — chooses 0 if we used wrong time index
                10.0, 1.0, 1.0, 1.0,
                // t=1 — actual final position; max at idx 2
                0.0, 0.5, 9.9, 0.5,
            ],
        )
        .unwrap();
        assert_eq!(argmax_last_position(&logits), vec![2]);
    }

    #[test]
    fn argmax_last_position_handles_batch() {
        let logits = Array3::from_shape_vec(
            (2, 1, 3),
            vec![
                // batch 0, t=0: max at idx 1
                0.0, 5.0, 0.0,
                // batch 1, t=0: max at idx 2
                1.0, 1.0, 9.0,
            ],
        )
        .unwrap();
        assert_eq!(argmax_last_position(&logits), vec![1, 2]);
    }

    // --- apply_repetition_penalty ---

    fn make_logits_1xv(row: &[f32]) -> Array3<f32> {
        let v = row.len();
        Array3::from_shape_vec((1, 1, v), row.to_vec()).unwrap()
    }

    #[test]
    fn repetition_penalty_no_op_at_one() {
        let mut logits = make_logits_1xv(&[1.0, 2.0, 3.0]);
        let before = logits.clone();
        apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
        assert_eq!(logits, before);
    }

    #[test]
    fn repetition_penalty_no_op_for_empty_history() {
        let mut logits = make_logits_1xv(&[1.0, -2.0, 3.0]);
        let before = logits.clone();
        apply_repetition_penalty(&mut logits, &[], 1.5);
        assert_eq!(logits, before);
    }

    #[test]
    fn repetition_penalty_divides_positive_logits() {
        let mut logits = make_logits_1xv(&[2.0, 4.0, 6.0]);
        apply_repetition_penalty(&mut logits, &[1], 2.0);
        // Token 1 was emitted, positive logit divided by penalty.
        assert_eq!(logits[[0, 0, 0]], 2.0); // unchanged
        assert_eq!(logits[[0, 0, 1]], 2.0); // 4 / 2
        assert_eq!(logits[[0, 0, 2]], 6.0); // unchanged
    }

    #[test]
    fn repetition_penalty_multiplies_negative_logits() {
        let mut logits = make_logits_1xv(&[-1.0, -2.0, -3.0]);
        apply_repetition_penalty(&mut logits, &[2], 2.0);
        // Token 2 was emitted, negative logit multiplied by penalty.
        assert_eq!(logits[[0, 0, 0]], -1.0);
        assert_eq!(logits[[0, 0, 1]], -2.0);
        assert_eq!(logits[[0, 0, 2]], -6.0); // -3 * 2
    }

    #[test]
    fn repetition_penalty_dedups_history() {
        // Token 1 appears 5× in the history. Penalty should apply
        // exactly once, not 5×.
        let mut logits = make_logits_1xv(&[1.0, 4.0, 9.0]);
        apply_repetition_penalty(&mut logits, &[1, 1, 1, 1, 1], 2.0);
        // Single application: 4 / 2 = 2.
        assert_eq!(logits[[0, 0, 1]], 2.0);
    }

    #[test]
    fn repetition_penalty_skips_out_of_range_tokens() {
        let mut logits = make_logits_1xv(&[1.0, 2.0, 3.0]);
        // Token id 99 is past vocab_size=3 — must not panic.
        apply_repetition_penalty(&mut logits, &[99, -1, 2], 2.0);
        // Token 2 is valid → 3 / 2 = 1.5.
        assert_eq!(logits[[0, 0, 2]], 1.5);
        // Other tokens untouched.
        assert_eq!(logits[[0, 0, 0]], 1.0);
        assert_eq!(logits[[0, 0, 1]], 2.0);
    }

    #[test]
    fn repetition_penalty_only_touches_last_time_position() {
        // 1 batch × 3 time × 4 vocab. Only the last time-slice gets
        // the penalty — earlier positions are either prefix tokens
        // (not yet sampled) or already-emitted tokens.
        let mut logits = Array3::from_shape_vec(
            (1, 3, 4),
            vec![
                // t=0
                1.0, 2.0, 3.0, 4.0,
                // t=1
                5.0, 6.0, 7.0, 8.0,
                // t=2 (last)
                9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        apply_repetition_penalty(&mut logits, &[1], 2.0);
        // t=0 and t=1 untouched.
        assert_eq!(logits[[0, 0, 1]], 2.0);
        assert_eq!(logits[[0, 1, 1]], 6.0);
        // t=2 column 1 divided by penalty.
        assert_eq!(logits[[0, 2, 1]], 5.0); // 10 / 2
    }

    #[test]
    fn repetition_penalty_changes_argmax_when_penalty_active() {
        // Without penalty: token 0 wins (logit 5).
        // History contains token 0 → with penalty=2.0 its logit
        // becomes 2.5, so token 1 (logit 4) wins instead.
        let mut logits = make_logits_1xv(&[5.0, 4.0, 3.0]);
        assert_eq!(argmax_last_position(&logits), vec![0]);
        apply_repetition_penalty(&mut logits, &[0], 2.0);
        assert_eq!(argmax_last_position(&logits), vec![1]);
    }

    #[test]
    fn decode_options_default_repetition_penalty() {
        let o = DecodeOptions::for_asr("es");
        assert_eq!(o.repetition_penalty, DEFAULT_REPETITION_PENALTY);
        assert_eq!(o.repetition_penalty, 1.8);
    }

    // --- suppress_eos_until_min_length ---

    #[test]
    fn min_length_no_op_at_zero_ratio() {
        let mut logits = make_logits_1xv(&[1.0, 2.0, 3.0]);
        let before = logits.clone();
        suppress_eos_until_min_length(&mut logits, 1, 0, 100, 0.0);
        assert_eq!(logits, before);
    }

    #[test]
    fn min_length_no_op_when_suffix_long_enough() {
        // ratio=0.3, n_enc_frames=10 → min_len = ceil(3.0) = 3.
        // suffix_len=3 already meets the min, so EOS not suppressed.
        let mut logits = make_logits_1xv(&[1.0, 5.0, 3.0]); // EOS=1
        let before = logits.clone();
        suppress_eos_until_min_length(&mut logits, 1, 3, 10, 0.3);
        assert_eq!(logits, before);
    }

    #[test]
    fn min_length_suppresses_eos_when_below_min() {
        // ratio=0.5, n_enc_frames=10 → min_len = ceil(5.0) = 5.
        // suffix_len=2 below min → EOS becomes -inf.
        let mut logits = make_logits_1xv(&[1.0, 5.0, 3.0]);
        suppress_eos_until_min_length(&mut logits, 1, 2, 10, 0.5);
        assert_eq!(logits[[0, 0, 0]], 1.0); // unchanged
        assert!(logits[[0, 0, 1]].is_infinite() && logits[[0, 0, 1]] < 0.0);
        assert_eq!(logits[[0, 0, 2]], 3.0); // unchanged
    }

    #[test]
    fn min_length_changes_argmax_when_eos_was_winning() {
        // EOS (id 1) wins on raw logits with score 5.0. After
        // suppression, the next-best (id 2 with 3.0) wins.
        let mut logits = make_logits_1xv(&[1.0, 5.0, 3.0]);
        assert_eq!(argmax_last_position(&logits), vec![1]);
        suppress_eos_until_min_length(&mut logits, 1, 0, 10, 0.3);
        assert_eq!(argmax_last_position(&logits), vec![2]);
    }

    #[test]
    fn min_length_uses_ceil_for_fractional_min() {
        // ratio=0.25, n_enc_frames=10 → min_len = ceil(2.5) = 3.
        // suffix_len=2 → still below → suppress.
        // suffix_len=3 → at min → don't suppress.
        let mut a = make_logits_1xv(&[1.0, 5.0, 3.0]);
        suppress_eos_until_min_length(&mut a, 1, 2, 10, 0.25);
        assert!(a[[0, 0, 1]].is_infinite());

        let mut b = make_logits_1xv(&[1.0, 5.0, 3.0]);
        suppress_eos_until_min_length(&mut b, 1, 3, 10, 0.25);
        assert_eq!(b[[0, 0, 1]], 5.0);
    }

    #[test]
    fn min_length_skips_when_zero_encoder_frames() {
        let mut logits = make_logits_1xv(&[1.0, 5.0, 3.0]);
        let before = logits.clone();
        suppress_eos_until_min_length(&mut logits, 1, 0, 0, 0.5);
        assert_eq!(logits, before);
    }

    #[test]
    fn min_length_skips_out_of_range_eos() {
        let mut logits = make_logits_1xv(&[1.0, 2.0, 3.0]);
        let before = logits.clone();
        // EOS id 99 is past vocab_size=3 — must not panic.
        suppress_eos_until_min_length(&mut logits, 99, 0, 10, 0.5);
        assert_eq!(logits, before);
    }

    #[test]
    fn min_length_only_touches_last_time_position() {
        let mut logits = Array3::from_shape_vec(
            (1, 2, 3),
            vec![
                // t=0
                1.0, 2.0, 3.0,
                // t=1 (last)
                4.0, 5.0, 6.0,
            ],
        )
        .unwrap();
        suppress_eos_until_min_length(&mut logits, 1, 0, 10, 0.5);
        // t=0 untouched
        assert_eq!(logits[[0, 0, 1]], 2.0);
        // t=1 EOS column suppressed
        assert!(logits[[0, 1, 1]].is_infinite() && logits[[0, 1, 1]] < 0.0);
    }

    #[test]
    fn decode_options_default_min_token_ratio() {
        let o = DecodeOptions::for_asr("es");
        assert_eq!(o.min_token_to_frame_ratio, DEFAULT_MIN_TOKEN_TO_FRAME_RATIO);
        assert_eq!(o.min_token_to_frame_ratio, 0.2);
    }

    #[test]
    fn strip_prefix_and_specials_drops_specials() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        // [prefix×10, hello (17), <|endoftext|> (3), world (18)]
        let mut all = vec![0u32; PREFIX_LEN];
        all.extend_from_slice(&[17, 3, 18]);
        // PRefix-strip + drop <|endoftext|> (special) → [17, 18]
        let kept = strip_prefix_and_specials(&all, PREFIX_LEN, &v);
        assert_eq!(kept, vec![17, 18]);
    }

    #[test]
    fn strip_prefix_and_specials_handles_no_emissions() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        let only_prefix = vec![0u32; PREFIX_LEN];
        let kept = strip_prefix_and_specials(&only_prefix, PREFIX_LEN, &v);
        assert!(kept.is_empty());
    }

    #[test]
    fn strip_prefix_and_specials_handles_oversized_prefix() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        let kept = strip_prefix_and_specials(&[1, 2, 3], 10, &v);
        assert!(kept.is_empty());
    }

    #[test]
    fn load_nonexistent_decoder_returns_error() {
        match CanaryDecoder::load("/nonexistent/path/to/decoder.onnx") {
            Ok(_) => panic!("expected error"),
            Err(e) => assert!(
                e.message.contains("load Canary decoder"),
                "{}",
                e.message
            ),
        }
    }

    #[test]
    fn io_name_constants_match_onnx_asr_conventions() {
        assert_eq!(DECODER_INPUT_IDS, "input_ids");
        assert_eq!(DECODER_INPUT_ENCODER_EMBEDDINGS, "encoder_embeddings");
        assert_eq!(DECODER_INPUT_ENCODER_MASK, "encoder_mask");
        assert_eq!(DECODER_INPUT_DECODER_MEMS, "decoder_mems");
        assert_eq!(DECODER_OUTPUT_LOGITS, "logits");
        assert_eq!(DECODER_OUTPUT_HIDDEN_STATES, "decoder_hidden_states");
    }

    #[test]
    fn prefix_len_constant_matches_layout() {
        // Pin the layout: 10 slots, no more, no less.
        assert_eq!(PREFIX_LEN, 10);
    }
}
