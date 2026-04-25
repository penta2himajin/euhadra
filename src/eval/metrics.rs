//! Word-level (WER) and character-level (CER) error rates.
//!
//! Both reduce to a normalized Levenshtein edit distance over a token stream:
//! WER tokenises by whitespace, CER by characters (with whitespace stripped
//! so "今日は" and "今日 は" compare equal).
//!
//! Normalisation is intentionally light:
//! - lowercase ASCII
//! - strip ASCII / fullwidth punctuation that ASR engines insert / drop
//!   inconsistently
//! - collapse runs of whitespace
//!
//! This matches the convention used by Whisper / FLEURS evaluation scripts.

/// Word Error Rate. Whitespace-tokenised after normalisation. Returns
/// `f64::NAN` if the reference is empty (undefined).
pub fn wer(reference: &str, hypothesis: &str) -> f64 {
    let r_norm = normalize(reference);
    let h_norm = normalize(hypothesis);
    let r: Vec<&str> = r_norm.split_whitespace().collect();
    let h: Vec<&str> = h_norm.split_whitespace().collect();

    if r.is_empty() {
        return f64::NAN;
    }
    let dist = levenshtein(&r, &h);
    dist as f64 / r.len() as f64
}

/// Character Error Rate. Whitespace is stripped before character
/// comparison so spacing differences do not inflate the score (relevant
/// for ja/zh).
pub fn cer(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<char> = normalize(reference).chars().filter(|c| !c.is_whitespace()).collect();
    let h: Vec<char> = normalize(hypothesis).chars().filter(|c| !c.is_whitespace()).collect();

    if r.is_empty() {
        return f64::NAN;
    }
    let dist = levenshtein(&r, &h);
    dist as f64 / r.len() as f64
}

/// Light normalisation shared by WER and CER:
/// - lowercase ASCII letters (does not touch CJK)
/// - strip a fixed set of punctuation (ASCII + fullwidth)
/// - collapse runs of whitespace into single spaces
pub fn normalize(text: &str) -> String {
    const PUNCT: &[char] = &[
        '.', ',', '?', '!', ':', ';', '"', '\'', '(', ')', '[', ']', '{', '}',
        '。', '、', '?', '!', ':', ';', '「', '」', '『', '』', '(', ')', '・',
    ];
    let mut out = String::with_capacity(text.len());
    let mut last_space = true; // suppress leading whitespace
    for c in text.chars() {
        if PUNCT.contains(&c) {
            continue;
        }
        if c.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
            continue;
        }
        last_space = false;
        for lower in c.to_lowercase() {
            out.push(lower);
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Standard Levenshtein edit distance over arbitrary `Eq` tokens.
fn levenshtein<T: Eq>(a: &[T], b: &[T]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for (i, ai) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, bj) in b.iter().enumerate() {
            let cost = if ai == bj { 0 } else { 1 };
            curr[j + 1] = (curr[j] + 1)
                .min(prev[j + 1] + 1)
                .min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wer_identical_is_zero() {
        assert_eq!(wer("hello world", "hello world"), 0.0);
    }

    #[test]
    fn wer_one_substitution_in_three_words() {
        // "hello dark world" vs "hello world world" → 1 sub / 3 ref words
        assert!((wer("hello dark world", "hello world world") - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn wer_deletion() {
        assert!((wer("the quick brown fox", "the brown fox") - 1.0 / 4.0).abs() < 1e-9);
    }

    #[test]
    fn wer_normalises_punctuation_and_case() {
        assert_eq!(wer("Hello, world!", "hello world"), 0.0);
    }

    #[test]
    fn wer_empty_reference_is_nan() {
        assert!(wer("", "anything").is_nan());
    }

    #[test]
    fn cer_identical_japanese() {
        assert_eq!(cer("今日は天気がいい", "今日は天気がいい"), 0.0);
    }

    #[test]
    fn cer_ignores_whitespace_differences() {
        assert_eq!(cer("今日は", "今日 は"), 0.0);
    }

    #[test]
    fn cer_one_char_substitution_in_three() {
        // 3 chars after whitespace strip, 1 sub → 1/3
        assert!((cer("今日は", "今夜は") - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn cer_strips_japanese_punctuation() {
        assert_eq!(cer("今日は、天気がいい。", "今日は天気がいい"), 0.0);
    }

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize("  hello   world  "), "hello world");
    }

    #[test]
    fn normalize_lowercases_ascii_only() {
        assert_eq!(normalize("Hello CAFÉ"), "hello café");
    }
}
