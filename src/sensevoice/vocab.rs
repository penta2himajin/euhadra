//! Vocabulary + CTC postprocess for SenseVoice-Small.
//!
//! The official `FunAudioLLM/SenseVoice` distribution ships a
//! SentencePiece BPE model (`chn_jpn_yue_eng_ko_spectok.bpe.model`)
//! rather than a plain text token table. `scripts/setup_sensevoice.sh`
//! materialises it as a `tokens.txt` with one piece per line — line
//! number is the token id — so the Rust side can load the vocab
//! without depending on the SentencePiece runtime.
//!
//! Decoding pipeline (matches the upstream `model.py` behaviour
//! followed by a `rich_transcription_postprocess`-style strip):
//!
//! ```text
//! logits [T, V]
//!   → argmax over V                     ids [T]
//!   → unique_consecutive                ids [T']
//!   → drop blank_id                     ids [T'']
//!   → vocab lookup → strings
//!   → drop <|...|> rich-text markers
//!   → join, replace SP "▁" with " "
//!   → trim leading whitespace
//! ```

use std::path::Path;

use crate::traits::AsrError;

/// SentencePiece "word boundary" prefix. Appears at the start of a
/// piece that begins a new whitespace-separated word in the source
/// transcript. Decoding swaps it for an ASCII space.
pub const SP_WORD_BOUNDARY: char = '\u{2581}'; // ▁

/// Load a `tokens.txt` produced by `scripts/setup_sensevoice.sh`. One
/// piece per line, line index = token id. Trailing blank lines are
/// ignored; embedded whitespace inside a piece (rare but possible for
/// a SentencePiece export of an audio-tag piece) is preserved verbatim.
pub fn load_tokens_txt(path: &Path) -> Result<Vec<String>, AsrError> {
    let text = std::fs::read_to_string(path).map_err(|e| AsrError {
        message: format!("read tokens.txt {}: {e}", path.display()),
    })?;
    let mut tokens: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    // Drop trailing empty lines so the vocab length is predictable.
    while tokens.last().map(String::is_empty).unwrap_or(false) {
        tokens.pop();
    }
    if tokens.is_empty() {
        return Err(AsrError {
            message: format!("tokens.txt {} is empty", path.display()),
        });
    }
    Ok(tokens)
}

/// CTC greedy collapse: drop consecutive duplicates first, then drop
/// the blank id. Mirrors the reference
/// `unique_consecutive(yseq); yseq[yseq != blank_id]` from
/// `FunAudioLLM/SenseVoice/model.py`.
pub fn ctc_collapse(ids: &[u32], blank_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut prev: Option<u32> = None;
    for &id in ids {
        if Some(id) == prev {
            continue;
        }
        prev = Some(id);
        if id != blank_id {
            out.push(id);
        }
    }
    out
}

/// Map collapsed ids → tokens, dropping ids that fall outside the
/// vocab. A well-formed model+vocab pair never produces out-of-range
/// ids; we treat them defensively to avoid panicking the pipeline.
pub fn ids_to_tokens(ids: &[u32], vocab: &[String]) -> Vec<String> {
    ids.iter()
        .filter_map(|&id| vocab.get(id as usize).cloned())
        .collect()
}

/// Render a token sequence as a final transcript.
///
/// SenseVoice predicts a handful of `<|...|>` rich-text markers
/// (language id, emotion, audio events, ITN flag) interleaved with
/// the SentencePiece content pieces. The user-facing transcript drops
/// every `<|...|>` token, then follows the SentencePiece convention
/// where pieces beginning with `▁` start a new whitespace-separated
/// word.
///
/// Korean / Japanese / Chinese pieces typically don't carry a
/// leading `▁` (CJK has no inter-word spacing in the source), so they
/// concatenate naturally; English / European pieces produce
/// space-separated words; mixed transcripts work because the rule is
/// purely "▁ → space" without language-aware branching.
pub fn decode_tokens(tokens: &[String]) -> String {
    let mut buf = String::new();
    for tok in tokens {
        if is_rich_text_marker(tok) {
            continue;
        }
        for c in tok.chars() {
            if c == SP_WORD_BOUNDARY {
                buf.push(' ');
            } else {
                buf.push(c);
            }
        }
    }
    // The very first content piece is usually `▁word`, which leaves a
    // leading space. Trim it without disturbing internal whitespace.
    let trimmed = buf.trim_start();
    if trimmed.len() == buf.len() {
        buf
    } else {
        trimmed.to_string()
    }
}

/// `<|en|>`, `<|HAPPY|>`, `<|withitn|>`, `<|Speech|>`, `<|/Speech|>`,
/// etc. all match this shape. Anything that opens with `<|` and ends
/// with `|>` is treated as a marker — mirrors how the official
/// `rich_transcription_postprocess` peels them off.
fn is_rich_text_marker(tok: &str) -> bool {
    tok.starts_with("<|") && tok.ends_with("|>")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(toks: &[&str]) -> Vec<String> {
        toks.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn ctc_collapse_dedups_then_drops_blank() {
        // 0 = blank. Sequence: 0 0 7 7 7 0 8 8 9 0 9 9
        // → unique_consecutive: 0 7 0 8 9 0 9
        // → drop blank:         7 8 9 9
        let collapsed = ctc_collapse(&[0, 0, 7, 7, 7, 0, 8, 8, 9, 0, 9, 9], 0);
        assert_eq!(collapsed, vec![7, 8, 9, 9]);
    }

    #[test]
    fn ctc_collapse_handles_empty_input() {
        assert_eq!(ctc_collapse(&[], 0), Vec::<u32>::new());
    }

    #[test]
    fn ctc_collapse_preserves_non_blank_runs_after_dedup() {
        // The dedup pass collapses an entire run to one frame, so
        // genuinely repeated content tokens (which the model emits
        // separated by a blank) survive.
        let collapsed = ctc_collapse(&[5, 5, 0, 5, 5], 0);
        assert_eq!(collapsed, vec![5, 5]);
    }

    #[test]
    fn ids_to_tokens_skips_out_of_range() {
        let v = vocab(&["<blank>", "a", "b"]);
        assert_eq!(ids_to_tokens(&[1, 999, 2], &v), vec!["a", "b"]);
    }

    #[test]
    fn decode_tokens_strips_rich_markers() {
        let toks = vocab(&[
            "<|ko|>",
            "<|NEUTRAL|>",
            "<|Speech|>",
            "<|withitn|>",
            "\u{2581}안",
            "녕",
            "하",
            "세",
            "요",
        ]);
        assert_eq!(decode_tokens(&toks), "안녕하세요");
    }

    #[test]
    fn decode_tokens_replaces_word_boundary_with_space() {
        let toks = vocab(&["<|en|>", "\u{2581}hel", "lo", "\u{2581}wor", "ld"]);
        assert_eq!(decode_tokens(&toks), "hello world");
    }

    #[test]
    fn decode_tokens_trims_leading_space_only() {
        // Internal double space (rare but possible if the BPE happens
        // to emit "▁ ▁") must be preserved so downstream tooling can
        // see exactly what the model produced.
        let toks = vocab(&["\u{2581}a", "\u{2581}", "\u{2581}b"]);
        assert_eq!(decode_tokens(&toks), "a  b");
    }

    #[test]
    fn decode_tokens_handles_mixed_korean_english() {
        let toks = vocab(&["<|ko|>", "안", "녕", "\u{2581}hello", "\u{2581}세", "계"]);
        assert_eq!(decode_tokens(&toks), "안녕 hello 세계");
    }

    #[test]
    fn decode_tokens_empty_input_yields_empty_string() {
        assert_eq!(decode_tokens(&[]), "");
    }

    #[test]
    fn decode_tokens_only_markers_yields_empty_string() {
        let toks = vocab(&["<|nospeech|>", "<|woitn|>", "<|EVENT_UNK|>"]);
        assert_eq!(decode_tokens(&toks), "");
    }

    #[test]
    fn load_tokens_txt_round_trips() {
        let dir =
            std::env::temp_dir().join(format!("euhadra_sensevoice_tokens_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokens.txt");
        std::fs::write(&path, "<blank>\n<s>\n</s>\n<|zh|>\n<|en|>\n\n").unwrap();
        let v = load_tokens_txt(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(v.len(), 5);
        assert_eq!(v[0], "<blank>");
        assert_eq!(v[3], "<|zh|>");
        assert_eq!(v[4], "<|en|>");
    }

    #[test]
    fn load_tokens_txt_rejects_empty_file() {
        let dir = std::env::temp_dir().join(format!(
            "euhadra_sensevoice_tokens_empty_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokens.txt");
        std::fs::write(&path, "\n\n").unwrap();
        let res = load_tokens_txt(&path);
        std::fs::remove_dir_all(&dir).ok();
        assert!(res.is_err());
    }

    #[test]
    fn load_tokens_txt_missing_file_errors() {
        let res = load_tokens_txt(Path::new("/nonexistent/euhadra/tokens.txt"));
        assert!(res.is_err());
    }
}
