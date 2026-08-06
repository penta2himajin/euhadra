//! Dolphin's `tokens.txt` and CTC output rendering.
//!
//! The file is the k2-fsa two-column form — `symbol<space>id`, one per
//! line — rather than the bare one-piece-per-line list
//! `sensevoice::vocab::load_tokens_txt` reads, so it gets its own
//! loader. The ids happen to be sequential from zero in every published
//! bundle, but the format does not promise that and a silent off-by-one
//! in the vocabulary produces fluent nonsense rather than an error, so
//! the column is parsed rather than inferred from position.

use std::collections::BTreeMap;
use std::path::Path;

use crate::traits::AsrError;

/// SentencePiece word-boundary prefix, same convention as SenseVoice.
pub const SP_WORD_BOUNDARY: char = '\u{2581}'; // ▁

/// CTC blank. Dolphin's vocabulary puts `<blank>` at id 0.
pub const BLANK_ID: u32 = 0;

/// Load `symbol<space>id` pairs into an id-indexed table.
///
/// A symbol may itself contain spaces in principle, so the id is taken
/// from the *last* whitespace-separated field and everything before it
/// is the symbol.
pub fn load_tokens(path: &Path) -> Result<Vec<String>, AsrError> {
    let raw = std::fs::read_to_string(path).map_err(|e| AsrError::ModelLoad(format!("read tokens.txt {}: {e}", path.display())))?;

    let mut by_id: BTreeMap<u32, String> = BTreeMap::new();
    for (lineno, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let Some(split) = line.rfind(char::is_whitespace) else {
            return Err(AsrError::Inference(format!(
                    "{}:{}: expected `symbol<space>id`, got {line:?}",
                    path.display(),
                    lineno + 1
                )));
        };
        let (symbol, id) = line.split_at(split);
        let id: u32 = id.trim().parse().map_err(|e| AsrError::Inference(format!(
                "{}:{}: token id {:?} is not a number: {e}",
                path.display(),
                lineno + 1,
                id.trim()
            )))?;
        if by_id.insert(id, symbol.to_string()).is_some() {
            return Err(AsrError::Inference(format!("{}:{}: duplicate token id {id}", path.display(), lineno + 1)));
        }
    }

    if by_id.is_empty() {
        return Err(AsrError::Inference(format!("{} contains no tokens", path.display())));
    }
    // A gap would shift every id above it, so refuse rather than pad.
    let expected = by_id.len() as u32;
    if *by_id.keys().next_back().unwrap() != expected - 1 || *by_id.keys().next().unwrap() != 0 {
        return Err(AsrError::Inference(format!(
                "{}: token ids are not contiguous from 0 (got {}..={} across {} entries)",
                path.display(),
                by_id.keys().next().unwrap(),
                by_id.keys().next_back().unwrap(),
                by_id.len()
            )));
    }

    Ok(by_id.into_values().collect())
}

/// Is this one of the model's control symbols rather than transcript
/// content?
///
/// Dolphin's vocabulary opens with `<blank>`, `<unk>`, task tags
/// (`<asr>`, `<itn>`, `<nopunc>`, `<nospeech>`, `<na>`) and one tag per
/// supported language (`<ko>`, `<ja>`, …), and closes with `<sos>`,
/// `<eos>`, `<sop>`. The CTC branch has no decoder prompt, so these are
/// simply frames the model may emit and the transcript must not show.
pub fn is_control(token: &str) -> bool {
    token.len() >= 2 && token.starts_with('<') && token.ends_with('>')
}

/// Render collapsed token ids as a transcript.
///
/// SentencePiece convention: a piece starting with `▁` opens a new
/// whitespace-separated word, anything else glues to what precedes it.
/// Korean, Japanese and Chinese pieces mostly carry no `▁` because the
/// source has no inter-word spacing there, so they concatenate.
pub fn decode(tokens: &[String]) -> String {
    let mut out = String::new();
    for token in tokens {
        if is_control(token) {
            continue;
        }
        match token.strip_prefix(SP_WORD_BOUNDARY) {
            Some(rest) => {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(rest);
            }
            None => out.push_str(token),
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Self-deleting scratch file — the repo has no `tempfile`
    /// dev-dependency, and `src/sensevoice/vocab.rs` writes under
    /// `temp_dir()` the same way.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(body: &str) -> Self {
            static N: AtomicU32 = AtomicU32::new(0);
            let path = std::env::temp_dir().join(format!(
                "euhadra_dolphin_tokens_{}_{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&path, body).unwrap();
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn write_tokens(body: &str) -> Scratch {
        Scratch::new(body)
    }

    #[test]
    fn loads_two_column_pairs_in_id_order() {
        let f = write_tokens("<blank> 0\n<unk> 1\n\u{2581}the 2\n한 3\n");
        let v = load_tokens(f.path()).unwrap();
        assert_eq!(v, vec!["<blank>", "<unk>", "\u{2581}the", "한"]);
    }

    #[test]
    fn ids_drive_the_order_not_the_line_number() {
        // A file that lists ids out of order must still index correctly
        // — reading position as id is the failure this guards.
        let f = write_tokens("b 1\na 0\nc 2\n");
        assert_eq!(load_tokens(f.path()).unwrap(), vec!["a", "b", "c"]);
    }

    #[test]
    fn a_symbol_containing_a_space_keeps_it() {
        let f = write_tokens("a 0\nx y 1\n");
        assert_eq!(load_tokens(f.path()).unwrap(), vec!["a", "x y"]);
    }

    #[test]
    fn gaps_and_duplicates_are_rejected() {
        // Both would silently shift every id above them.
        let gap = write_tokens("a 0\nb 2\n");
        assert!(load_tokens(gap.path()).is_err());
        let dup = write_tokens("a 0\nb 1\nc 1\n");
        assert!(load_tokens(dup.path()).is_err());
        let nonzero = write_tokens("a 1\nb 2\n");
        assert!(load_tokens(nonzero.path()).is_err());
    }

    #[test]
    fn malformed_lines_are_rejected_not_skipped() {
        assert!(load_tokens(write_tokens("a 0\nnoid\n").path()).is_err());
        assert!(load_tokens(write_tokens("a 0\nb xx\n").path()).is_err());
        assert!(load_tokens(write_tokens("\n\n").path()).is_err());
    }

    #[test]
    fn blank_lines_are_ignored() {
        let f = write_tokens("a 0\n\nb 1\n\n");
        assert_eq!(load_tokens(f.path()).unwrap(), vec!["a", "b"]);
    }

    #[test]
    fn control_tokens_are_recognised() {
        for t in ["<blank>", "<unk>", "<ko>", "<asr>", "<nospeech>", "<eos>"] {
            assert!(is_control(t), "{t}");
        }
        // Content that merely contains an angle bracket must survive.
        for t in ["<", "안", "\u{2581}the", "a<b", "<b", "b>"] {
            assert!(!is_control(t), "{t}");
        }
    }

    #[test]
    fn decode_drops_control_tokens() {
        let toks: Vec<String> = ["<ko>", "<asr>", "안", "녕", "<eos>"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(decode(&toks), "안녕");
    }

    #[test]
    fn decode_treats_the_boundary_marker_as_a_space() {
        let toks: Vec<String> = ["\u{2581}hello", "\u{2581}wor", "ld"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(decode(&toks), "hello world");
    }

    #[test]
    fn decode_concatenates_cjk_without_spacing() {
        // CJK pieces carry no boundary marker, so inserting spaces
        // between them would be wrong — and would also change CER
        // against a reference that has none.
        let toks: Vec<String> = ["주", "기", "율", "표"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(decode(&toks), "주기율표");
    }

    #[test]
    fn a_leading_boundary_does_not_open_with_a_space() {
        let toks = vec!["\u{2581}the".to_string()];
        assert_eq!(decode(&toks), "the");
    }

    #[test]
    fn decode_of_nothing_is_empty() {
        assert_eq!(decode(&[]), "");
        assert_eq!(decode(&["<blank>".to_string()]), "");
    }
}
