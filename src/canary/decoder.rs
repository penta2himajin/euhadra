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

/// Layout of the decoder prefix tokens. The official NeMo
/// `Canary2PromptFormatter` template
///
/// ```text
/// {CANARY2_BOCTX}|decodercontext|{CANARY_BOS}|emotion||source_lang|
/// |target_lang||pnc||itn||timestamp||diarize|
/// ```
///
/// renders to **9 tokens** when `decodercontext` is the empty string
/// (the default for ASR without a context prompt). The
/// `istupakov/onnx-asr` Python reference instead emits a 10-token
/// prefix that puts a leading `▁` (last-occurrence of U+2581) before
/// `<|startofcontext|>`. The two layouts produce different decoder
/// behaviour for some inputs; we expose the choice as an enum so a
/// future PR can sweep both layouts and pick the one that actually
/// matches Canary's training distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixFormat {
    /// 10-token onnx-asr layout: `[▁, <|soc|>, <|sot|>, ..., <|nodiarize|>]`.
    /// What we shipped through PR #28-#36; matches the Python
    /// `onnx-asr` reference bit-for-bit.
    OnnxAsr,
    /// 9-token NeMo Canary2PromptFormatter layout (no leading ▁):
    /// `[<|soc|>, <|sot|>, <|emo:undefined|>, <|src|>, <|tgt|>,
    ///  <|pnc|>, <|noitn|>, <|notimestamp|>, <|nodiarize|>]`.
    NemoCanary2,
}

impl PrefixFormat {
    pub fn token_count(&self) -> usize {
        match self {
            PrefixFormat::OnnxAsr => 10,
            PrefixFormat::NemoCanary2 => 9,
        }
    }
}

/// Default prefix format. `OnnxAsr` until the per-format sweep in
/// `docs/canary-integration.md` "v8 — official prompt alignment"
/// confirms which one the model was actually trained against.
pub const DEFAULT_PREFIX_FORMAT: PrefixFormat = PrefixFormat::OnnxAsr;

/// Default cap on the autoregressive sequence length, matching the
/// `max_sequence_length` field of the istupakov canary config.json.
pub const DEFAULT_MAX_SEQUENCE_LENGTH: usize = 1024;

/// Default greedy-decode repetition penalty. `1.0` is a no-op; values
/// above 1 down-weight already-emitted tokens. We pick **2.0** based
/// on the FLEURS-es 100-utt sweep on the v6-aligned frontend
/// (recorded in `docs/canary-integration.md` "End-to-end validation
/// v6"):
///
/// | penalty | WER | hard fails | repetition loops |
/// |---|---|---|---|
/// | 1.0 (off) | 34.50 % | 1 | (≥1) |
/// | 1.2 | 34.17 % | 1 | 2 |
/// | 1.5 | 26.37 % | 1 | 1 |
/// | 1.8 | 26.43 % | 1 | 1 |
/// | **2.0** | **14.12 %** | 1 | 0 |
///
/// The sweet spot moved up from 1.8 (v3-v5, pre-alignment frontend)
/// to 2.0 here because the v6 frontend that bit-aligns with onnx-asr
/// produces sharper logits, so a stronger penalty is needed to push
/// repeat tokens below the most-likely non-repeat. Set to `1.0` to
/// disable.
pub const DEFAULT_REPETITION_PENALTY: f32 = 2.0;

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

/// Default required margin (in raw logit units) by which
/// `<|endoftext|>` must lead the next-best non-EOS token before
/// the greedy step accepts it. Addresses the **partial-dropout**
/// failure mode where EOS marginally wins the argmax mid-utterance,
/// producing a transcript that's plausible but missing later words.
/// Complementary to the positional `min_token_to_frame_ratio` gate
/// — that one forbids EOS too early; this one demands EOS be
/// confidently dominant when it does win.
///
/// Sweep on FLEURS-es 100-utt FP32 with `repetition_penalty=1.8`
/// and `min_token_to_frame_ratio=0.2` (recorded in
/// `docs/canary-integration.md` "End-to-end validation v5"):
///
/// | margin | mean WER | 0–5 % | hard fails | loops |
/// |---|---|---|---|---|
/// | 0.0 (off, v4) | 13.63 % | 41 | 0 | 0 |
/// | 1.0 | 13.21 % | 42 | 0 | 0 |
/// | **2.0** | **13.21 %** | **42** | **0** | **0** |
/// | 3.0 | 13.21 % | 22 | 0 | 0 |
/// | 5.0 | 13.21 % | 21 | 0 | 0 |
///
/// 1.0–2.0 hit a small-but-real improvement; 2.0 chosen as a
/// slightly stricter and more robust default. Above 2.0 the mean
/// stays the same (decoder keeps emitting, but the high-WER tail
/// barely improves while clean utterances get padded with extra
/// tokens — the per-utterance trade is symmetric but the clean
/// count drops). Set to `0.0` to disable.
pub const DEFAULT_EOS_CONFIDENCE_MARGIN: f32 = 2.0;

/// Default beam-search width. `1` means greedy (the v6 path); higher
/// values trade compute for accuracy by exploring multiple
/// hypotheses in parallel and keeping the top-`beam_size` by
/// length-normalised log-probability. The Canary model card numbers
/// (FLEURS-es 2.9 % WER on the 1B-v2 variant) are reported with a
/// beam search; the istupakov ONNX export's `onnx-asr` Python
/// reference is greedy-only and clocks 34.43 % WER on the same 100-utt
/// FLEURS-es subset — the gap motivates this option. Sweep recorded
/// in `docs/canary-integration.md` "End-to-end validation v7".
pub const DEFAULT_BEAM_SIZE: usize = 1;

/// Default length penalty α for beam-search final scoring. Score per
/// hypothesis is `log_prob / length^α`; α = 0 disables length
/// normalisation (longer sequences are penalised), α = 1 fully
/// length-normalises. The HF transformers default for ASR is around
/// 0.6 — slightly favours longer hypotheses without making them
/// arbitrarily long. Only used when `beam_size > 1`.
pub const DEFAULT_LENGTH_PENALTY: f32 = 0.6;

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
    /// Required logit-margin by which `<|endoftext|>` must lead the
    /// next-best non-EOS token before the greedy step accepts it.
    /// Demotes EOS to `-inf` when its logit doesn't lead by at
    /// least this margin, forcing the decoder to keep emitting on
    /// borderline cases. `0.0` disables. See
    /// `DEFAULT_EOS_CONFIDENCE_MARGIN`.
    pub eos_confidence_margin: f32,
    /// Beam-search width. `1` keeps the v6 greedy path. Values > 1
    /// switch to a batched beam-search loop where the decoder runs
    /// `beam_size` hypotheses in parallel and selects the top-
    /// `beam_size` by accumulated length-normalised log-probability
    /// at each step.
    pub beam_size: usize,
    /// Length penalty α for beam-search final scoring; ignored when
    /// `beam_size == 1`. See `DEFAULT_LENGTH_PENALTY`.
    pub length_penalty: f32,
    /// Decoder prefix layout. See `PrefixFormat`.
    pub prefix_format: PrefixFormat,
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
            eos_confidence_margin: DEFAULT_EOS_CONFIDENCE_MARGIN,
            beam_size: DEFAULT_BEAM_SIZE,
            length_penalty: DEFAULT_LENGTH_PENALTY,
            prefix_format: DEFAULT_PREFIX_FORMAT,
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

    let prefix: Vec<u32> = match opts.prefix_format {
        PrefixFormat::OnnxAsr => vec![
            space, soc, sot, emo, src_lang, tgt_lang, pnc, noitn, notimestamp, nodiarize,
        ],
        PrefixFormat::NemoCanary2 => vec![
            soc, sot, emo, src_lang, tgt_lang, pnc, noitn, notimestamp, nodiarize,
        ],
    };
    debug_assert_eq!(prefix.len(), opts.prefix_format.token_count());
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

/// Demote `<|endoftext|>` to `-inf` at the last time-position if
/// its logit doesn't lead the next-best non-EOS token by at least
/// `margin`. Forces the greedy step to keep emitting on borderline
/// EOS picks — addresses the partial-dropout failure mode where
/// EOS is just barely the argmax mid-utterance.
///
/// `margin <= 0.0` is a no-op. EOS logits already at `-inf` (e.g.
/// suppressed by `suppress_eos_until_min_length`) are left alone:
/// the comparison `-inf - X` underflows and naturally fails the
/// margin check.
///
/// Operates on the last time-position only — earlier positions
/// are prefix tokens or already-emitted tokens that the greedy
/// step won't sample again.
pub fn enforce_eos_confidence_margin(
    logits: &mut Array3<f32>,
    eos_token_id: u32,
    margin: f32,
) {
    if margin <= 0.0 {
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
        let eos_logit = logits[[b, last_t, v]];
        if !eos_logit.is_finite() {
            continue;
        }
        // Find the highest non-EOS logit at the last time-position.
        let mut max_other = f32::NEG_INFINITY;
        for vi in 0..vocab_size {
            if vi == v {
                continue;
            }
            let l = logits[[b, last_t, vi]];
            if l > max_other {
                max_other = l;
            }
        }
        if !max_other.is_finite() {
            // No competing token exists — accept EOS rather than
            // demote to -inf and stall the decoder.
            continue;
        }
        if eos_logit - max_other < margin {
            logits[[b, last_t, v]] = f32::NEG_INFINITY;
        }
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

        if opts.beam_size > 1 {
            return self.decode_beam(encoder_embeddings, encoder_mask, vocab, opts);
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

            // EOS-confidence margin: complementary to min-length.
            // Demotes EOS when it's only marginally winning the
            // argmax, addressing partial-dropout cases where the
            // decoder gives up just as the next content token was
            // also competitive.
            if opts.eos_confidence_margin > 0.0 {
                enforce_eos_confidence_margin(
                    &mut logits,
                    eos,
                    opts.eos_confidence_margin,
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

/// Compute the per-token log-softmax of a single `[V]` logit row,
/// returning the full normalised vector. Numerically stable
/// (subtracts max before exponentiating).
fn log_softmax_row(logits: &[f32]) -> Vec<f32> {
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    if !max.is_finite() {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
    let mut sum_exp = 0.0_f32;
    for &v in logits {
        sum_exp += (v - max).exp();
    }
    let logsumexp = max + sum_exp.ln();
    logits.iter().map(|&v| v - logsumexp).collect()
}

/// Find the top-`k` indices of `xs` by descending value. Stable
/// across ties (lower index wins). Returns at most `k` items even
/// if `xs.len() < k`.
fn topk_indices(xs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = xs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    indexed.truncate(k);
    indexed
}

/// Length-normalised hypothesis score: `cum_logprob / length^α`.
/// `length` is the number of generated (non-prefix, non-EOS) tokens.
/// `α = 0` reduces to the raw cumulative log-prob; `α = 1` fully
/// length-normalises. Used to compare beams of different lengths
/// at the final selection step.
fn length_normalised_score(cum_logprob: f32, length: usize, alpha: f32) -> f32 {
    if length == 0 {
        return cum_logprob;
    }
    if alpha == 0.0 {
        return cum_logprob;
    }
    cum_logprob / (length as f32).powf(alpha)
}

#[derive(Clone)]
struct ActiveBeam {
    /// Full token sequence, prefix included.
    tokens: Vec<i64>,
    /// Sum of log-probabilities over the generated suffix.
    cum_logprob: f32,
}

#[derive(Clone)]
struct FinishedBeam {
    tokens: Vec<i64>,
    cum_logprob: f32,
    /// Generated-suffix length (excludes prefix, excludes the
    /// trailing EOS).
    length: usize,
}

impl CanaryDecoder {
    /// Beam-search decoding loop. Mirrors `decode` but maintains
    /// `beam_size` parallel hypotheses, expands each by the top-
    /// `beam_size` next tokens at every step, and keeps the global
    /// top-`beam_size` after re-ranking.
    ///
    /// All gates from the greedy path (repetition penalty, min-length,
    /// EOS-confidence margin) apply per beam — each beam carries its
    /// own history for the repetition penalty, while the positional
    /// min-length and EOS-margin gates operate on each beam's own
    /// suffix length.
    ///
    /// KV cache (`decoder_mems`) is reordered per step using
    /// `ndarray::select` along the batch axis so each new beam starts
    /// from its parent's cache.
    fn decode_beam(
        &self,
        encoder_embeddings: &Array3<f32>,
        encoder_mask: &Array2<i64>,
        vocab: &Vocab,
        opts: &DecodeOptions,
    ) -> Result<DecodeOutput, AsrError> {
        let beam_size = opts.beam_size;
        if beam_size < 2 {
            return Err(AsrError {
                message: format!(
                    "decode_beam requires beam_size ≥ 2 (got {beam_size})"
                ),
            });
        }

        let prefix = build_decoder_prefix(vocab, opts)?;
        let prefix_len = prefix.len();
        let max_len = opts.max_sequence_length.max(prefix_len + 1);
        let eos = vocab.eos()?;

        let mut session = self.session.lock().map_err(|e| AsrError {
            message: format!("decoder session lock poisoned: {e}"),
        })?;

        // === Step 0: identical prefix across all B beams; run once
        // and seed the beam set from the top-B candidates. ===
        let initial_input: Array2<i64> =
            Array2::from_shape_vec((1, prefix_len), prefix.clone()).map_err(|e| AsrError {
                message: format!("input_ids reshape (initial): {e}"),
            })?;
        let initial_mems: Array4<f32> =
            Array4::zeros((self.mems_layers, 1, 0, self.mems_hidden));
        let (init_logits, init_hidden) = run_decoder_step(
            &mut session,
            initial_input,
            encoder_embeddings.clone(),
            encoder_mask.clone(),
            initial_mems,
        )?;

        // Apply min-length / eos-margin gates once at step 0 (no
        // suffix yet, so repetition penalty has nothing to do).
        let mut step0_logits = init_logits;
        if opts.min_token_to_frame_ratio > 0.0 {
            let n_enc_frames = encoder_embeddings.shape()[1];
            suppress_eos_until_min_length(
                &mut step0_logits,
                eos,
                0,
                n_enc_frames,
                opts.min_token_to_frame_ratio,
            );
        }
        if opts.eos_confidence_margin > 0.0 {
            enforce_eos_confidence_margin(&mut step0_logits, eos, opts.eos_confidence_margin);
        }

        let last_t = step0_logits.shape()[1] - 1;
        let vocab_size = step0_logits.shape()[2];
        let last_row: Vec<f32> = (0..vocab_size).map(|v| step0_logits[[0, last_t, v]]).collect();
        let log_probs0 = log_softmax_row(&last_row);
        let candidates0 = topk_indices(&log_probs0, beam_size);

        let mut active: Vec<ActiveBeam> = Vec::with_capacity(beam_size);
        for (vid, lp) in &candidates0 {
            let mut tokens = prefix.clone();
            tokens.push(*vid as i64);
            active.push(ActiveBeam {
                tokens,
                cum_logprob: *lp,
            });
        }

        // Broadcast the [L, 1, T_acc, H] step-0 hidden state to
        // [L, B, T_acc, H] by stacking `beam_size` copies along
        // axis 1 — they're all identical at this point because the
        // prefix is shared.
        let mut decoder_mems: Array4<f32> = stack_along_batch(&init_hidden, beam_size)?;

        let mut finished: Vec<FinishedBeam> = Vec::new();

        // === Step ≥ 1: each active beam is now distinct. Run them
        // in a single batched decoder call (input_ids [B, 1],
        // encoder repeated). ===
        while !active.is_empty()
            && active[0].tokens.len() < max_len
            && finished.len() < beam_size
        {
            let b = active.len();

            // Build input_ids = each beam's last token, shape [B, 1].
            let last_tokens: Vec<i64> = active.iter().map(|x| *x.tokens.last().unwrap()).collect();
            let input_ids: Array2<i64> = Array2::from_shape_vec((b, 1), last_tokens.clone())
                .map_err(|e| AsrError {
                    message: format!("input_ids reshape (step): {e}"),
                })?;

            // Replicate encoder I/O across the batch dim.
            let enc_emb_b: Array3<f32> = repeat_along_axis0(encoder_embeddings, b);
            let enc_mask_b: Array2<i64> = repeat_along_axis0_i64(encoder_mask, b);

            let (mut logits, hidden) = run_decoder_step(
                &mut session,
                input_ids,
                enc_emb_b,
                enc_mask_b,
                decoder_mems,
            )?;

            // Apply per-beam gates: repetition penalty uses each
            // beam's own history; min-length / eos-margin operate on
            // each beam's suffix length (all beams share the same
            // prefix_len + step count, but suffix == step count is
            // identical across beams at this point).
            let suffix_len = active[0].tokens.len() - prefix_len;
            let n_enc_frames = encoder_embeddings.shape()[1];
            for (bi, beam) in active.iter().enumerate() {
                if opts.repetition_penalty != 1.0 {
                    apply_repetition_penalty_one_batch(
                        &mut logits,
                        bi,
                        &beam.tokens[prefix_len..],
                        opts.repetition_penalty,
                    );
                }
                if opts.min_token_to_frame_ratio > 0.0 {
                    suppress_eos_until_min_length_one_batch(
                        &mut logits,
                        bi,
                        eos,
                        suffix_len,
                        n_enc_frames,
                        opts.min_token_to_frame_ratio,
                    );
                }
                if opts.eos_confidence_margin > 0.0 {
                    enforce_eos_confidence_margin_one_batch(
                        &mut logits,
                        bi,
                        eos,
                        opts.eos_confidence_margin,
                    );
                }
            }

            // For each beam, compute log-softmax of last position
            // and grab top-`beam_size` candidates. Score each
            // candidate as parent.cum_logprob + log_prob[v].
            let last_t = logits.shape()[1] - 1;
            let vocab_size = logits.shape()[2];
            let mut all_candidates: Vec<(usize, i64, f32)> =
                Vec::with_capacity(b * beam_size);
            for bi in 0..b {
                let row: Vec<f32> = (0..vocab_size).map(|v| logits[[bi, last_t, v]]).collect();
                let lp = log_softmax_row(&row);
                let topk = topk_indices(&lp, beam_size);
                for (vid, score) in topk {
                    all_candidates.push((bi, vid as i64, active[bi].cum_logprob + score));
                }
            }
            all_candidates.sort_by(|a, c| {
                c.2.partial_cmp(&a.2)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&c.0))
            });

            // Take up to `beam_size` candidates, classifying each as
            // finished (next == EOS) or active. EOS hypotheses
            // always finalise — they're scored by the cumulative
            // log-prob *up to but not including* the EOS token,
            // which is the convention onnx-asr uses for its
            // logprob-aligned output.
            let mut next_active: Vec<ActiveBeam> = Vec::new();
            let mut parent_idx_for_active: Vec<usize> = Vec::new();
            for (parent, vid, cum) in all_candidates {
                if next_active.len() + finished.len() >= beam_size + finished.len() {
                    // pruned
                }
                if next_active.len() >= beam_size {
                    break;
                }
                let mut tokens = active[parent].tokens.clone();
                if vid as u32 == eos {
                    finished.push(FinishedBeam {
                        tokens,
                        cum_logprob: active[parent].cum_logprob,
                        length: suffix_len,
                    });
                    if finished.len() >= beam_size {
                        break;
                    }
                } else {
                    tokens.push(vid);
                    next_active.push(ActiveBeam {
                        tokens,
                        cum_logprob: cum,
                    });
                    parent_idx_for_active.push(parent);
                }
            }

            if next_active.is_empty() {
                break;
            }

            // Reorder decoder_mems by parent_idx so each new beam
            // starts from its parent's KV cache. `hidden` has shape
            // [L, b, T_acc+1, H]; `select` along axis 1 picks the
            // parent rows in the new beam order.
            decoder_mems = hidden.select(Axis(1), &parent_idx_for_active);
            active = next_active;
        }

        // Finalise: collect candidates from finished + still-active
        // beams, score them by length-normalised log-prob, pick the
        // best.
        let alpha = opts.length_penalty;
        let mut best: Option<(Vec<i64>, Vec<f32>)> = None;
        let mut best_score = f32::NEG_INFINITY;

        for f in &finished {
            let score = length_normalised_score(f.cum_logprob, f.length, alpha);
            if score > best_score {
                best_score = score;
                let logprobs = vec![0.0_f32; f.length];
                best = Some((f.tokens.clone(), logprobs));
            }
        }
        for a in &active {
            let length = a.tokens.len() - prefix_len;
            let score = length_normalised_score(a.cum_logprob, length, alpha);
            if score > best_score {
                best_score = score;
                let logprobs = vec![0.0_f32; length];
                best = Some((a.tokens.clone(), logprobs));
            }
        }

        let (best_tokens, best_logprobs) = best.ok_or_else(|| AsrError {
            message: "beam search produced no candidates".into(),
        })?;

        let all_ids: Vec<u32> = best_tokens.iter().map(|&x| x as u32).collect();
        let kept = strip_prefix_and_specials(&all_ids, prefix_len, vocab);
        // The per-token logprobs vector is a placeholder (filled with
        // zeros) — beam search merges contributions across multiple
        // ancestor paths, so per-step log-probabilities aren't
        // unambiguously aligned with the surviving tokens. Caller
        // can ignore them; `kept` is the user-facing transcript.
        let kept_logprobs: Vec<f32> = vec![0.0_f32; kept.len()];
        let _ = best_logprobs;

        Ok(DecodeOutput {
            tokens: kept,
            logprobs: kept_logprobs,
        })
    }
}

/// Run a single decoder step from outside the greedy/beam loops.
/// Returns `(logits, decoder_hidden_states)`.
fn run_decoder_step(
    session: &mut Session,
    input_ids: Array2<i64>,
    encoder_embeddings: Array3<f32>,
    encoder_mask: Array2<i64>,
    decoder_mems: Array4<f32>,
) -> Result<(Array3<f32>, Array4<f32>), AsrError> {
    let input_ids_v = Value::from_array(input_ids).map_err(|e| AsrError {
        message: format!("input_ids Value: {e}"),
    })?;
    let enc_emb_v = Value::from_array(encoder_embeddings).map_err(|e| AsrError {
        message: format!("encoder_embeddings Value: {e}"),
    })?;
    let enc_mask_v = Value::from_array(encoder_mask).map_err(|e| AsrError {
        message: format!("encoder_mask Value: {e}"),
    })?;
    let mems_v = Value::from_array(decoder_mems).map_err(|e| AsrError {
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
    let logits_idx = output_index(&outputs, DECODER_OUTPUT_LOGITS).ok_or_else(|| AsrError {
        message: format!("decoder missing output {DECODER_OUTPUT_LOGITS}"),
    })?;
    let mems_idx = output_index(&outputs, DECODER_OUTPUT_HIDDEN_STATES).ok_or_else(|| AsrError {
        message: format!("decoder missing output {DECODER_OUTPUT_HIDDEN_STATES}"),
    })?;
    let logits: Array3<f32> = outputs[logits_idx]
        .try_extract_array::<f32>()
        .map_err(|e| AsrError {
            message: format!("extract logits: {e}"),
        })?
        .to_owned()
        .into_dimensionality()
        .map_err(|e| AsrError {
            message: format!("logits rank: {e}"),
        })?;
    let hidden: Array4<f32> = outputs[mems_idx]
        .try_extract_array::<f32>()
        .map_err(|e| AsrError {
            message: format!("extract hidden: {e}"),
        })?
        .to_owned()
        .into_dimensionality()
        .map_err(|e| AsrError {
            message: format!("hidden rank: {e}"),
        })?;
    Ok((logits, hidden))
}

/// Stack `n_copies` of the rank-4 `hidden` tensor (shape
/// `[L, 1, T, H]`) along the batch axis to produce
/// `[L, n_copies, T, H]`. Used by `decode_beam` to broadcast the
/// shared step-0 KV cache to each beam.
fn stack_along_batch(hidden: &Array4<f32>, n_copies: usize) -> Result<Array4<f32>, AsrError> {
    let (l, b, t, h) = hidden.dim();
    if b != 1 {
        return Err(AsrError {
            message: format!("stack_along_batch expected batch=1, got {b}"),
        });
    }
    let mut out = Array4::<f32>::zeros((l, n_copies, t, h));
    for k in 0..n_copies {
        out.slice_mut(ndarray::s![.., k..k + 1, .., ..])
            .assign(hidden);
    }
    Ok(out)
}

/// Repeat a `[1, ..]` tensor along axis 0 `n_copies` times. Used to
/// expand the encoder outputs across beam batch dim.
fn repeat_along_axis0(arr: &Array3<f32>, n_copies: usize) -> Array3<f32> {
    let shape = arr.shape();
    let new_shape = (n_copies, shape[1], shape[2]);
    let mut out = Array3::<f32>::zeros(new_shape);
    for k in 0..n_copies {
        out.slice_mut(ndarray::s![k..k + 1, .., ..]).assign(arr);
    }
    out
}

fn repeat_along_axis0_i64(arr: &Array2<i64>, n_copies: usize) -> Array2<i64> {
    let shape = arr.shape();
    let new_shape = (n_copies, shape[1]);
    let mut out = Array2::<i64>::zeros(new_shape);
    for k in 0..n_copies {
        out.slice_mut(ndarray::s![k..k + 1, ..]).assign(arr);
    }
    out
}

/// Per-batch variants of the gate helpers — the public versions
/// iterate over all batch elements; in beam search we want to
/// process one row at a time because each beam has its own history.
fn apply_repetition_penalty_one_batch(
    logits: &mut Array3<f32>,
    batch: usize,
    history: &[i64],
    penalty: f32,
) {
    if penalty == 1.0 || history.is_empty() {
        return;
    }
    let (_, time, vocab_size) = logits.dim();
    if time == 0 {
        return;
    }
    let last_t = time - 1;
    let mut seen = std::collections::HashSet::<i64>::new();
    for &tok in history {
        if tok < 0 || !seen.insert(tok) {
            continue;
        }
        let v = tok as usize;
        if v >= vocab_size {
            continue;
        }
        let score = logits[[batch, last_t, v]];
        let new_score = if score < 0.0 {
            score * penalty
        } else {
            score / penalty
        };
        logits[[batch, last_t, v]] = new_score;
    }
}

fn suppress_eos_until_min_length_one_batch(
    logits: &mut Array3<f32>,
    batch: usize,
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
    let (_, time, vocab_size) = logits.dim();
    if time == 0 {
        return;
    }
    let v = eos_token_id as usize;
    if v >= vocab_size {
        return;
    }
    let last_t = time - 1;
    logits[[batch, last_t, v]] = f32::NEG_INFINITY;
}

fn enforce_eos_confidence_margin_one_batch(
    logits: &mut Array3<f32>,
    batch: usize,
    eos_token_id: u32,
    margin: f32,
) {
    if margin <= 0.0 {
        return;
    }
    let (_, time, vocab_size) = logits.dim();
    if time == 0 {
        return;
    }
    let v = eos_token_id as usize;
    if v >= vocab_size {
        return;
    }
    let last_t = time - 1;
    let eos_logit = logits[[batch, last_t, v]];
    if !eos_logit.is_finite() {
        return;
    }
    let mut max_other = f32::NEG_INFINITY;
    for vi in 0..vocab_size {
        if vi == v {
            continue;
        }
        let l = logits[[batch, last_t, vi]];
        if l > max_other {
            max_other = l;
        }
    }
    if !max_other.is_finite() {
        return;
    }
    if eos_logit - max_other < margin {
        logits[[batch, last_t, v]] = f32::NEG_INFINITY;
    }
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
    fn prefix_nemo_canary2_layout_is_nine_tokens() {
        // The official NeMo `Canary2PromptFormatter` template
        // (NVIDIA/NeMo nemo/collections/common/prompts/canary2.py)
        // renders to 9 tokens for ASR with empty `decodercontext`:
        //   [<|soc|>, <|sot|>, <|emo:undefined|>, <|src|>, <|tgt|>,
        //    <|pnc|>, <|noitn|>, <|notimestamp|>, <|nodiarize|>]
        // i.e. the same as `OnnxAsr` minus the leading ▁.
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        let mut opts = DecodeOptions::for_asr("es");
        opts.prefix_format = PrefixFormat::NemoCanary2;
        let p = build_decoder_prefix(&v, &opts).unwrap();
        assert_eq!(p.len(), 9);
        assert_eq!(p.len(), PrefixFormat::NemoCanary2.token_count());
        // Slot 0 = <|startofcontext|> (no leading ▁ in NeMo template).
        assert_eq!(p[0], 7);
        assert_eq!(p[1], 4);  // <|startoftranscript|>
        assert_eq!(p[2], 11); // <|emo:undefined|>
        assert_eq!(p[3], 15); // source <|es|>
        assert_eq!(p[4], 15); // target <|es|>
        assert_eq!(p[5], 5);  // <|pnc|>
        assert_eq!(p[6], 8);  // <|noitn|>
        assert_eq!(p[7], 10); // <|notimestamp|>
        assert_eq!(p[8], 9);  // <|nodiarize|>
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
            eos_confidence_margin: 0.0,
            beam_size: 1,
            length_penalty: 0.0,
            prefix_format: PrefixFormat::OnnxAsr,
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
        assert_eq!(o.repetition_penalty, 2.0);
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

    // --- enforce_eos_confidence_margin ---

    #[test]
    fn eos_margin_no_op_at_zero() {
        let mut logits = make_logits_1xv(&[5.0, 1.0, 2.0]); // EOS=0
        let before = logits.clone();
        enforce_eos_confidence_margin(&mut logits, 0, 0.0);
        assert_eq!(logits, before);
    }

    #[test]
    fn eos_margin_keeps_eos_when_dominant() {
        // EOS lead = 5 - 2 = 3 ≥ margin 2 → keep.
        let mut logits = make_logits_1xv(&[5.0, 1.0, 2.0]);
        enforce_eos_confidence_margin(&mut logits, 0, 2.0);
        assert_eq!(logits[[0, 0, 0]], 5.0);
    }

    #[test]
    fn eos_margin_demotes_eos_when_marginal() {
        // EOS lead = 5 - 4 = 1 < margin 2 → demote to -inf.
        let mut logits = make_logits_1xv(&[5.0, 4.0, 2.0]);
        enforce_eos_confidence_margin(&mut logits, 0, 2.0);
        assert!(logits[[0, 0, 0]].is_infinite() && logits[[0, 0, 0]] < 0.0);
        assert_eq!(logits[[0, 0, 1]], 4.0);
        assert_eq!(logits[[0, 0, 2]], 2.0);
    }

    #[test]
    fn eos_margin_skips_already_suppressed_eos() {
        // EOS already at -inf (e.g. by the min-length gate). The
        // margin enforcement must not panic and must not somehow
        // resurrect EOS.
        let mut logits = make_logits_1xv(&[f32::NEG_INFINITY, 1.0, 2.0]);
        let before = logits.clone();
        enforce_eos_confidence_margin(&mut logits, 0, 2.0);
        assert_eq!(format!("{:?}", logits), format!("{:?}", before));
    }

    #[test]
    fn eos_margin_skips_out_of_range_id() {
        let mut logits = make_logits_1xv(&[1.0, 2.0, 3.0]);
        let before = logits.clone();
        enforce_eos_confidence_margin(&mut logits, 99, 2.0);
        assert_eq!(logits, before);
    }

    #[test]
    fn eos_margin_changes_argmax_when_demoting() {
        // Without margin: EOS (id 0) wins.
        // With margin 2 and lead 1: EOS demoted, token 1 wins.
        let mut logits = make_logits_1xv(&[5.0, 4.0, 2.0]);
        assert_eq!(argmax_last_position(&logits), vec![0]);
        enforce_eos_confidence_margin(&mut logits, 0, 2.0);
        assert_eq!(argmax_last_position(&logits), vec![1]);
    }

    #[test]
    fn eos_margin_only_touches_last_time_position() {
        let mut logits = Array3::from_shape_vec(
            (1, 2, 3),
            vec![
                // t=0: EOS (id 0) lead = 1 < margin
                5.0, 4.0, 2.0,
                // t=1 (last): EOS lead = 5 - 4 = 1 < margin
                5.0, 4.0, 3.0,
            ],
        )
        .unwrap();
        enforce_eos_confidence_margin(&mut logits, 0, 2.0);
        // t=0 untouched
        assert_eq!(logits[[0, 0, 0]], 5.0);
        // t=1 EOS demoted
        assert!(logits[[0, 1, 0]].is_infinite() && logits[[0, 1, 0]] < 0.0);
    }

    #[test]
    fn eos_margin_keeps_eos_when_no_competing_token() {
        // Single-token vocab: EOS has no competition. The gate must
        // accept EOS rather than stall the decoder.
        let mut logits = make_logits_1xv(&[3.0]);
        let before = logits.clone();
        enforce_eos_confidence_margin(&mut logits, 0, 2.0);
        assert_eq!(logits, before);
    }

    #[test]
    fn decode_options_default_eos_confidence_margin() {
        let o = DecodeOptions::for_asr("es");
        assert_eq!(o.eos_confidence_margin, DEFAULT_EOS_CONFIDENCE_MARGIN);
        assert_eq!(o.eos_confidence_margin, 2.0);
    }

    // --- beam search helpers ---

    #[test]
    fn log_softmax_row_sums_to_one_in_probability_space() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let lp = log_softmax_row(&logits);
        let sum_p: f32 = lp.iter().map(|&x| x.exp()).sum();
        assert!((sum_p - 1.0).abs() < 1e-5, "sum p = {sum_p}");
    }

    #[test]
    fn log_softmax_row_handles_neg_inf_inputs() {
        // A finite max with a -inf neighbour must not produce NaN.
        let logits = vec![1.0_f32, f32::NEG_INFINITY, 2.0, 3.0];
        let lp = log_softmax_row(&logits);
        for v in &lp {
            assert!(v.is_finite() || *v == f32::NEG_INFINITY, "v={v}");
        }
        // The masked entry maps to -inf in log-prob space.
        assert_eq!(lp[1], f32::NEG_INFINITY);
    }

    #[test]
    fn topk_indices_picks_largest_in_descending_order() {
        let xs = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        let top = topk_indices(&xs, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 1); // 5.0
        assert_eq!(top[1].0, 3); // 4.0
        assert_eq!(top[2].0, 4); // 3.0
    }

    #[test]
    fn topk_indices_breaks_ties_by_lower_index() {
        let xs = vec![3.0, 3.0, 3.0, 1.0];
        let top = topk_indices(&xs, 2);
        assert_eq!(top[0].0, 0);
        assert_eq!(top[1].0, 1);
    }

    #[test]
    fn topk_indices_caps_at_input_length() {
        let xs = vec![1.0, 2.0];
        let top = topk_indices(&xs, 5);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn length_normalised_score_alpha_zero_returns_raw_logprob() {
        assert_eq!(length_normalised_score(-3.0, 5, 0.0), -3.0);
    }

    #[test]
    fn length_normalised_score_alpha_one_divides_by_length() {
        let s = length_normalised_score(-6.0, 3, 1.0);
        assert!((s - (-2.0)).abs() < 1e-6, "s={s}");
    }

    #[test]
    fn length_normalised_score_zero_length_returns_raw() {
        // Avoid 0^0 numerical issues; a zero-length sequence is just
        // its raw log-prob (which is 0.0 in practice).
        assert_eq!(length_normalised_score(0.0, 0, 0.6), 0.0);
    }

    #[test]
    fn length_normalised_score_alpha_below_one_penalises_longer_seqs() {
        // `score = cum_logprob / length^α` for `α < 1`: cum_logprob
        // grows linearly with length but the divisor grows sub-
        // linearly, so the longer sequence ends up with a more
        // negative score. (HF transformers' length_penalty kwarg
        // uses the same convention; values > 1 are needed to
        // promote longer sequences.)
        let short = length_normalised_score(-5.0, 5, 0.6);
        let long = length_normalised_score(-10.0, 10, 0.6);
        assert!(short > long, "short={short} long={long}");
    }

    #[test]
    fn length_normalised_score_alpha_above_one_promotes_longer_seqs() {
        // Mirror of the previous test: `α > 1` makes the divisor
        // grow super-linearly so longer sequences end up with a
        // less negative score and win.
        let short = length_normalised_score(-5.0, 5, 1.5);
        let long = length_normalised_score(-10.0, 10, 1.5);
        assert!(long > short, "short={short} long={long}");
    }

    #[test]
    fn decode_options_default_beam_size_is_one() {
        let o = DecodeOptions::for_asr("es");
        assert_eq!(o.beam_size, DEFAULT_BEAM_SIZE);
        assert_eq!(o.beam_size, 1);
    }

    #[test]
    fn stack_along_batch_replicates_correctly() {
        let h: Array4<f32> =
            Array4::from_shape_vec((2, 1, 3, 2), (0..12).map(|x| x as f32).collect()).unwrap();
        let stacked = stack_along_batch(&h, 4).unwrap();
        assert_eq!(stacked.dim(), (2, 4, 3, 2));
        for k in 0..4 {
            for l in 0..2 {
                for t in 0..3 {
                    for c in 0..2 {
                        assert_eq!(stacked[[l, k, t, c]], h[[l, 0, t, c]]);
                    }
                }
            }
        }
    }

    #[test]
    fn stack_along_batch_rejects_non_unit_batch() {
        let h: Array4<f32> = Array4::zeros((2, 3, 1, 1));
        assert!(stack_along_batch(&h, 4).is_err());
    }

    #[test]
    fn repeat_along_axis0_f32_replicates() {
        let a: Array3<f32> =
            Array3::from_shape_vec((1, 2, 3), (0..6).map(|x| x as f32).collect()).unwrap();
        let r = repeat_along_axis0(&a, 3);
        assert_eq!(r.dim(), (3, 2, 3));
        for k in 0..3 {
            for i in 0..2 {
                for j in 0..3 {
                    assert_eq!(r[[k, i, j]], a[[0, i, j]]);
                }
            }
        }
    }

    #[test]
    fn repeat_along_axis0_i64_replicates() {
        let a: Array2<i64> = Array2::from_shape_vec((1, 4), vec![10, 20, 30, 40]).unwrap();
        let r = repeat_along_axis0_i64(&a, 2);
        assert_eq!(r.dim(), (2, 4));
        assert_eq!(r[[0, 2]], 30);
        assert_eq!(r[[1, 2]], 30);
    }

    #[test]
    fn apply_repetition_penalty_one_batch_only_touches_target_batch() {
        // Two-batch logits; penalty applies to batch=1 only. Batch
        // 0 must remain unchanged.
        let mut logits: Array3<f32> =
            Array3::from_shape_vec((2, 1, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
        apply_repetition_penalty_one_batch(&mut logits, 1, &[1], 2.0);
        assert_eq!(logits[[0, 0, 1]], 2.0); // unchanged
        assert_eq!(logits[[1, 0, 1]], 1.0); // 2 / 2
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
