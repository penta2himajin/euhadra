//! Token vocabulary for Canary-180M-Flash, parsed from the
//! `vocab.txt` shipped with `istupakov/canary-180m-flash-onnx`.
//!
//! File format (one token per line, total 5248 entries):
//!
//! ```text
//! <unk> 0
//! <|nospeech|> 1
//! <pad> 2
//! <|endoftext|> 3
//! <|startoftranscript|> 4
//! ...
//! <|en|> 62
//! ...
//! <|es|> 169
//! ...
//! ▁ 1151
//! en 1153
//! ▁d 1154
//! ...
//! ```
//!
//! IDs 0..=1150 are reserved for control / language / special
//! tokens; IDs 1151+ are SentencePiece subword pieces, with the
//! `▁` (U+2581) prefix marking word-initial pieces.
//!
//! This module loads the vocab and exposes:
//!
//! - integer-id ↔ piece-string round-trip,
//! - named lookup for the special tokens the decoder prefix needs
//!   (sot / pnc / nopnc / soc / language / eos / etc.),
//! - SentencePiece-aware detokenisation.

use std::collections::HashMap;
use std::path::Path;

use crate::traits::AsrError;

/// Loaded vocabulary. Token ids are dense and contiguous from 0;
/// `id_to_piece[id]` is the surface form, mirroring the order in
/// `vocab.txt`.
#[derive(Debug, Clone)]
pub struct Vocab {
    id_to_piece: Vec<String>,
    piece_to_id: HashMap<String, u32>,
}

impl Vocab {
    /// Parse `vocab.txt` content. Each non-empty line must be
    /// `<piece><SP><id>` with `<id>` matching its line index.
    pub fn from_text(content: &str) -> Result<Self, AsrError> {
        let mut id_to_piece: Vec<String> = Vec::new();
        let mut piece_to_id: HashMap<String, u32> = HashMap::new();

        for (line_no, raw) in content.lines().enumerate() {
            if raw.trim().is_empty() {
                continue;
            }
            // Format is `<piece><SP><id>`; the piece itself can contain
            // spaces (e.g. " ▁") but the *last* whitespace-separated
            // token on the line is always the integer id. Splitting
            // on the rightmost ASCII space keeps both halves intact.
            let split_at = raw.rfind(' ').ok_or_else(|| AsrError {
                message: format!(
                    "vocab line {} missing id separator: {raw:?}",
                    line_no + 1
                ),
            })?;
            let piece = &raw[..split_at];
            let id_str = &raw[split_at + 1..];
            let id: u32 = id_str.parse().map_err(|_| AsrError {
                message: format!(
                    "vocab line {} id is not a u32: {id_str:?}",
                    line_no + 1
                ),
            })?;
            if id as usize != id_to_piece.len() {
                return Err(AsrError {
                    message: format!(
                        "vocab line {} id={} but expected {} (ids must be \
                         dense and ascending from 0)",
                        line_no + 1,
                        id,
                        id_to_piece.len()
                    ),
                });
            }
            id_to_piece.push(piece.to_string());
            // First-occurrence wins. The real Canary vocab repeats
            // `<unk>` at id 1152 (the SentencePiece byte-fallback
            // piece) in addition to the control token at id 0; we
            // want `id("<unk>")` to return the control id so a stray
            // lookup doesn't accidentally hit the piece form.
            piece_to_id.entry(piece.to_string()).or_insert(id);
        }

        if id_to_piece.is_empty() {
            return Err(AsrError {
                message: "vocab is empty".into(),
            });
        }

        Ok(Self {
            id_to_piece,
            piece_to_id,
        })
    }

    /// Read and parse `vocab.txt` from a path.
    pub fn from_file(path: &Path) -> Result<Self, AsrError> {
        let content = std::fs::read_to_string(path).map_err(|e| AsrError {
            message: format!("read vocab {}: {e}", path.display()),
        })?;
        Self::from_text(&content)
    }

    pub fn len(&self) -> usize {
        self.id_to_piece.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_piece.is_empty()
    }

    /// `id → piece` lookup. `None` for ids outside `0..len()`.
    pub fn piece(&self, id: u32) -> Option<&str> {
        self.id_to_piece.get(id as usize).map(String::as_str)
    }

    /// `piece → id` lookup. Used to resolve special tokens by name.
    pub fn id(&self, piece: &str) -> Option<u32> {
        self.piece_to_id.get(piece).copied()
    }

    /// Look up the language token `<|<lang>|>` for the four ASR
    /// languages Canary-180M-Flash supports (en / de / fr / es).
    /// Returns `None` for unsupported codes — caller decides whether
    /// to fall back or error.
    pub fn language_token(&self, lang: &str) -> Option<u32> {
        let normalised = match lang {
            "english" => "en",
            "spanish" => "es",
            "german" => "de",
            "french" => "fr",
            other => other,
        };
        self.id(&format!("<|{normalised}|>"))
    }

    /// `<|endoftext|>` — terminator for the autoregressive decoder.
    pub fn eos(&self) -> Result<u32, AsrError> {
        self.id("<|endoftext|>").ok_or_else(|| AsrError {
            message: "vocab missing <|endoftext|>".into(),
        })
    }

    /// `<|startoftranscript|>` — first prefix token for the decoder.
    pub fn sot(&self) -> Result<u32, AsrError> {
        self.id("<|startoftranscript|>").ok_or_else(|| AsrError {
            message: "vocab missing <|startoftranscript|>".into(),
        })
    }

    /// `<|startofcontext|>` — opens the context prompt slot. Canary
    /// always emits this even when no context is provided.
    pub fn soc(&self) -> Result<u32, AsrError> {
        self.id("<|startofcontext|>").ok_or_else(|| AsrError {
            message: "vocab missing <|startofcontext|>".into(),
        })
    }

    /// `<|pnc|>` — request punctuation + capitalisation in the output.
    pub fn pnc(&self) -> Result<u32, AsrError> {
        self.id("<|pnc|>").ok_or_else(|| AsrError {
            message: "vocab missing <|pnc|>".into(),
        })
    }

    /// `<|nopnc|>` — opposite of `<|pnc|>`.
    pub fn nopnc(&self) -> Result<u32, AsrError> {
        self.id("<|nopnc|>").ok_or_else(|| AsrError {
            message: "vocab missing <|nopnc|>".into(),
        })
    }

    /// Detokenise a sequence of token ids into a single text string,
    /// applying SentencePiece conventions:
    /// - `▁` (U+2581) marks a word boundary; replace with `' '`.
    /// - Non-`▁` pieces concatenate to the previous piece.
    /// - Special tokens (those matching `<|...|>` or being `<unk>` /
    ///   `<pad>`) are skipped silently.
    /// - Out-of-range ids are skipped (defensive — should never happen
    ///   under a well-formed model + vocab).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for id in ids {
            let piece = match self.piece(*id) {
                Some(p) => p,
                None => continue,
            };
            if is_special_token(piece) {
                continue;
            }
            for ch in piece.chars() {
                if ch == '\u{2581}' {
                    out.push(' ');
                } else {
                    out.push(ch);
                }
            }
        }
        // SentencePiece outputs typically lead with a space if the
        // first kept piece starts with `▁`. Strip the leading single
        // space so callers don't have to.
        if out.starts_with(' ') {
            out.remove(0);
        }
        out
    }
}

/// True iff `piece` is one of the structural / control tokens that
/// must never appear in the user-facing transcript. Canary uses both
/// the angle-bracket-pipe form (`<|...|>`) and a couple of bare
/// sentinels (`<unk>`, `<pad>`).
fn is_special_token(piece: &str) -> bool {
    if piece == "<unk>" || piece == "<pad>" {
        return true;
    }
    piece.starts_with("<|") && piece.ends_with("|>")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic mini-vocab with just enough structure to cover the
    /// special-token lookups + a handful of SentencePiece pieces.
    /// Ids match the real istupakov ordering for the tokens shown.
    fn mini_vocab_text() -> String {
        let entries: Vec<(&str, u32)> = vec![
            ("<unk>", 0),
            ("<|nospeech|>", 1),
            ("<pad>", 2),
            ("<|endoftext|>", 3),
            ("<|startoftranscript|>", 4),
            ("<|pnc|>", 5),
            ("<|nopnc|>", 6),
            ("<|startofcontext|>", 7),
            ("<|en|>", 8),
            ("<|es|>", 9),
            ("<|de|>", 10),
            ("<|fr|>", 11),
            // Two SentencePiece pieces for detokenisation tests.
            ("\u{2581}hello", 12),
            ("\u{2581}world", 13),
            ("\u{2581}fuera", 14),
            ("del", 15),
            ("\u{2581}aire", 16),
        ];
        let mut s = String::new();
        for (piece, id) in entries {
            s.push_str(&format!("{piece} {id}\n"));
        }
        s
    }

    #[test]
    fn parses_synthetic_mini_vocab() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        assert_eq!(v.len(), 17);
        assert_eq!(v.piece(0), Some("<unk>"));
        assert_eq!(v.piece(3), Some("<|endoftext|>"));
        assert_eq!(v.piece(15), Some("del"));
        assert_eq!(v.id("<|endoftext|>"), Some(3));
    }

    #[test]
    fn rejects_non_dense_ids() {
        let bad = "<unk> 0\n<pad> 2\n";
        let err = Vocab::from_text(bad).unwrap_err();
        assert!(err.message.contains("expected 1"), "{}", err.message);
    }

    #[test]
    fn rejects_missing_id() {
        let bad = "<unk>\n";
        let err = Vocab::from_text(bad).unwrap_err();
        assert!(err.message.contains("missing id separator"), "{}", err.message);
    }

    #[test]
    fn rejects_empty_vocab() {
        let err = Vocab::from_text("").unwrap_err();
        assert!(err.message.contains("empty"));
    }

    #[test]
    fn special_token_named_lookups() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        assert_eq!(v.eos().unwrap(), 3);
        assert_eq!(v.sot().unwrap(), 4);
        assert_eq!(v.pnc().unwrap(), 5);
        assert_eq!(v.nopnc().unwrap(), 6);
        assert_eq!(v.soc().unwrap(), 7);
    }

    #[test]
    fn special_token_missing_returns_error() {
        // Drop <|endoftext|> — eos() should error rather than panic.
        let mut text = String::new();
        text.push_str("<unk> 0\n");
        text.push_str("<pad> 1\n");
        let v = Vocab::from_text(&text).unwrap();
        let err = v.eos().unwrap_err();
        assert!(err.message.contains("<|endoftext|>"));
    }

    #[test]
    fn language_token_lookup_supported_languages() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        assert_eq!(v.language_token("en"), Some(8));
        assert_eq!(v.language_token("es"), Some(9));
        assert_eq!(v.language_token("de"), Some(10));
        assert_eq!(v.language_token("fr"), Some(11));
        // Long form aliases.
        assert_eq!(v.language_token("english"), Some(8));
        assert_eq!(v.language_token("spanish"), Some(9));
        assert_eq!(v.language_token("german"), Some(10));
        assert_eq!(v.language_token("french"), Some(11));
    }

    #[test]
    fn language_token_unsupported_returns_none() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        assert_eq!(v.language_token("ja"), None);
        assert_eq!(v.language_token("zh"), None);
    }

    #[test]
    fn decode_sentencepiece_pieces_concatenates_with_spaces() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        // "▁hello" + "▁world" → "hello world"
        assert_eq!(v.decode(&[12, 13]), "hello world");
    }

    #[test]
    fn decode_concatenates_non_word_initial_pieces() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        // "▁fuera" + "del" + "▁aire" → "fueradel aire" (the missing
        // ▁ on "del" *should* glue it onto the previous word; the
        // SentencePiece convention is exact and intentional even
        // when the resulting surface form is unusual).
        assert_eq!(v.decode(&[14, 15, 16]), "fueradel aire");
    }

    #[test]
    fn decode_skips_special_tokens() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        // [<|startoftranscript|>, <|en|>, <|pnc|>, ▁hello, ▁world,
        //  <|endoftext|>] → "hello world"
        assert_eq!(v.decode(&[4, 8, 5, 12, 13, 3]), "hello world");
    }

    #[test]
    fn decode_skips_unk_and_pad() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        // [<unk>, ▁hello, <pad>, ▁world] → "hello world"
        assert_eq!(v.decode(&[0, 12, 2, 13]), "hello world");
    }

    #[test]
    fn decode_skips_out_of_range_ids() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        // 99999 is past the vocab bound → silently skipped.
        assert_eq!(v.decode(&[12, 99_999, 13]), "hello world");
    }

    #[test]
    fn decode_empty_input_returns_empty_string() {
        let v = Vocab::from_text(&mini_vocab_text()).unwrap();
        assert_eq!(v.decode(&[]), "");
    }

    #[test]
    fn duplicate_piece_keeps_first_occurrence() {
        // Mirrors the real istupakov vocab where `<unk>` shows up
        // both as the id-0 control token and again at id 1152 as a
        // SentencePiece byte-fallback piece. The control id must win
        // for `id("<unk>")` so callers can rely on it.
        let mut text = String::new();
        text.push_str("<unk> 0\n");
        text.push_str("<pad> 1\n");
        text.push_str("a 2\n");
        text.push_str("<unk> 3\n");
        let v = Vocab::from_text(&text).unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(v.id("<unk>"), Some(0));
        // The reverse lookup still recovers either spelling.
        assert_eq!(v.piece(0), Some("<unk>"));
        assert_eq!(v.piece(3), Some("<unk>"));
    }

    #[test]
    fn is_special_token_recognises_canary_forms() {
        assert!(is_special_token("<|endoftext|>"));
        assert!(is_special_token("<|en|>"));
        assert!(is_special_token("<|startoftranscript|>"));
        assert!(is_special_token("<unk>"));
        assert!(is_special_token("<pad>"));

        assert!(!is_special_token("\u{2581}hello"));
        assert!(!is_special_token("hello"));
        assert!(!is_special_token("<not-special>"));
        assert!(!is_special_token("<|partial"));
    }
}
