//! Token-id ↔ string conversion and FunASR-style sentence post-processing.
//!
//! Mirrors `funasr_onnx.utils.postprocess_utils.sentence_postprocess`:
//! pure-Chinese → concatenation, pure-alphabetic with BPE `@@` →
//! whitespace-joined words, mixed → words joined by single spaces.

use std::path::Path;

use crate::traits::AsrError;

/// Loads `tokens.json` (a JSON array of strings) into a `Vec<String>`
/// indexed by token id. This is the format used by every published
/// FunASR Paraformer ONNX export.
pub fn load_tokens_json(path: &Path) -> Result<Vec<String>, AsrError> {
    let bytes = std::fs::read(path).map_err(|e| AsrError {
        message: format!("read tokens.json {}: {e}", path.display()),
    })?;
    let tokens: Vec<String> = serde_json::from_slice(&bytes).map_err(|e| AsrError {
        message: format!("parse tokens.json {}: {e}", path.display()),
    })?;
    if tokens.is_empty() {
        return Err(AsrError {
            message: format!("tokens.json {} is empty", path.display()),
        });
    }
    Ok(tokens)
}

/// Convert a sequence of token ids into the matching strings, dropping
/// the blank (id 0) and EOS (id 2) symbols. Out-of-range ids are
/// silently skipped — they should never appear under a well-formed
/// model + vocab pair, but mismatch must not panic the pipeline.
pub fn ids_to_tokens(ids: &[u32], vocab: &[String]) -> Vec<String> {
    ids.iter()
        .filter(|&&id| id != 0 && id != 2)
        .filter_map(|&id| vocab.get(id as usize).cloned())
        .collect()
}

/// Reproduces `sentence_postprocess` from FunASR's
/// `funasr_onnx/utils/postprocess_utils.py` for the non-`en-bpe`
/// (default) language path.
///
/// Rules:
/// - `<s>`, `</s>`, `<unk>` are stripped first.
/// - All tokens Chinese → concatenate without spaces.
/// - All tokens alphanumeric (Latin script) → join `@@`-suffixed BPE
///   units into a word, then space-separate words.
/// - Mixed → Chinese chars become standalone words; consecutive Latin
///   tokens follow BPE rules; everything joined with single spaces.
pub fn sentence_postprocess(tokens: &[String]) -> String {
    let washed: Vec<&str> = tokens
        .iter()
        .map(String::as_str)
        .filter(|t| !matches!(*t, "<s>" | "</s>" | "<unk>"))
        .collect();

    if washed.is_empty() {
        return String::new();
    }

    if washed.iter().all(|t| is_all_chinese(t)) {
        // Pure CJK — strip any embedded spaces (rare but mirrors Python)
        // and concatenate.
        return washed
            .iter()
            .map(|t| t.replace(' ', ""))
            .collect::<Vec<_>>()
            .join("");
    }

    if washed.iter().all(|t| is_all_alpha(t)) {
        return join_alpha_with_bpe(&washed);
    }

    // Mixed: walk the stream, emitting Chinese chars as standalone
    // words and gluing BPE-prefixed Latin pieces.
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    for tok in &washed {
        if is_all_chinese(tok) {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            out.push((*tok).to_string());
        } else if let Some(stem) = tok.strip_suffix("@@") {
            buf.push_str(stem);
        } else if is_all_alpha(tok) {
            buf.push_str(tok);
            out.push(std::mem::take(&mut buf));
        } else {
            // Unknown class — just flush what we have and emit verbatim.
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            out.push((*tok).to_string());
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out.join(" ")
}

fn join_alpha_with_bpe(tokens: &[&str]) -> String {
    let mut words = Vec::new();
    let mut buf = String::new();
    for tok in tokens {
        if let Some(stem) = tok.strip_suffix("@@") {
            buf.push_str(stem);
        } else {
            buf.push_str(tok);
            words.push(std::mem::take(&mut buf));
        }
    }
    if !buf.is_empty() {
        words.push(buf);
    }
    words.join(" ")
}

fn is_all_chinese(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_cjk_char)
}

fn is_all_alpha(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let stem = s.strip_suffix("@@").unwrap_or(s);
    if stem.is_empty() {
        return false;
    }
    stem.chars().all(|c| c.is_ascii_alphanumeric())
}

fn is_cjk_char(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_to_tokens_strips_blank_and_eos() {
        let vocab: Vec<String> = ["<blank>", "<s>", "</s>", "我", "们", "好"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // 0 = blank → drop, 3 = 我, 0 → drop, 4 = 们, 2 = </s> → drop, 5 = 好
        let ids = vec![0u32, 3, 0, 4, 2, 5];
        assert_eq!(ids_to_tokens(&ids, &vocab), vec!["我", "们", "好"]);
    }

    #[test]
    fn ids_to_tokens_skips_out_of_range() {
        let vocab: Vec<String> = ["<blank>", "<s>", "</s>", "x"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let ids = vec![3u32, 999];
        assert_eq!(ids_to_tokens(&ids, &vocab), vec!["x"]);
    }

    #[test]
    fn sentence_postprocess_pure_chinese_concatenates() {
        let toks = ["我", "们", "好"].iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(sentence_postprocess(&toks), "我们好");
    }

    #[test]
    fn sentence_postprocess_strips_specials() {
        let toks = ["<s>", "我", "</s>"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        assert_eq!(sentence_postprocess(&toks), "我");
    }

    #[test]
    fn sentence_postprocess_pure_alpha_with_bpe() {
        let toks = ["he@@", "llo", "wor@@", "ld"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        assert_eq!(sentence_postprocess(&toks), "hello world");
    }

    #[test]
    fn sentence_postprocess_mixed_zh_en() {
        // "hello 我 new world"
        let toks = ["hel@@", "lo", "我", "new", "world"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        assert_eq!(sentence_postprocess(&toks), "hello 我 new world");
    }

    #[test]
    fn sentence_postprocess_empty_input() {
        assert_eq!(sentence_postprocess(&[]), "");
    }

    #[test]
    fn sentence_postprocess_only_specials() {
        let toks = ["<s>", "</s>", "<unk>"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        assert_eq!(sentence_postprocess(&toks), "");
    }

    #[test]
    fn load_tokens_json_round_trips(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!(
            "euhadra_paraformer_vocab_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("tokens.json");
        std::fs::write(&path, r#"["<blank>", "<s>", "</s>", "我", "们"]"#.as_bytes())?;
        let vocab = load_tokens_json(&path)?;
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(vocab.len(), 5);
        assert_eq!(vocab[3], "我");
        Ok(())
    }

    #[test]
    fn load_tokens_json_rejects_empty_array() {
        let dir = std::env::temp_dir().join(format!(
            "euhadra_paraformer_empty_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokens.json");
        std::fs::write(&path, b"[]").unwrap();
        let res = load_tokens_json(&path);
        std::fs::remove_dir_all(&dir).ok();
        assert!(res.is_err());
    }

    #[test]
    fn load_tokens_json_missing_file_errors() {
        let res = load_tokens_json(Path::new("/nonexistent/euhadra/tokens.json"));
        assert!(res.is_err());
    }
}
