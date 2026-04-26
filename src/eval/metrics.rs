//! Word-level (WER) and character-level (CER) error rates.
//!
//! Both reduce to a normalized Levenshtein edit distance over a token stream:
//! WER tokenises by whitespace, CER by characters (with whitespace stripped
//! so "今日は" and "今日 は" compare equal).
//!
//! Normalisation is intentionally light:
//! - lowercase ASCII (case-insensitive comparison)
//! - strip ASCII / fullwidth punctuation that ASR engines insert / drop
//!   inconsistently
//! - collapse runs of whitespace
//! - convert Chinese numerals (digit and positional forms) to Arabic
//!   digits — FLEURS-zh references use `15`, `2011` etc. while
//!   Paraformer-large emits `十五`, `二零一一`. Both are valid; left
//!   un-normalised they read as wholesale character substitutions and
//!   inflated zh CER by ~6 percentage points before this change.
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
/// - convert Chinese numerals to Arabic digits (digit-by-digit and
///   positional 十/百/千/万/亿 forms)
pub fn normalize(text: &str) -> String {
    const PUNCT: &[char] = &[
        '.', ',', '?', '!', ':', ';', '"', '\'', '(', ')', '[', ']', '{', '}',
        '。', '、', '?', '!', ':', ';', '「', '」', '『', '』', '(', ')', '・',
    ];

    // Numerals first so the punctuation/case/whitespace pass below
    // operates on the canonical digit form.
    let digits_normalised = normalize_chinese_numerals(text);

    let mut out = String::with_capacity(digits_normalised.len());
    let mut last_space = true; // suppress leading whitespace
    for c in digits_normalised.chars() {
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

/// Walks `text` and replaces every maximal run of Chinese numeral
/// characters with its Arabic-digit equivalent. Two grammars are
/// recognised:
///
/// - **Digit-by-digit** (no `十/百/千/万/亿` in the run): each Chinese
///   digit becomes a single Arabic digit. e.g. `二零一一` → `2011`,
///   `四零三` → `403`.
/// - **Positional** (run contains at least one `十/百/千/万/亿`):
///   parsed as a Chinese number with the usual scale rules.
///   e.g. `十五` → `15`, `七十` → `70`, `一百二十三` → `123`,
///   `一万二千三百` → `12300`.
///
/// Mixed-case runs (`一万二千三零零`) take the positional path and
/// treat stray digits as the in-progress unit value, matching how
/// these surface forms are pronounced.
fn normalize_chinese_numerals(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run = String::new();
    for c in text.chars() {
        if is_zh_numeric(c) {
            run.push(c);
        } else {
            if !run.is_empty() {
                out.push_str(&convert_zh_numeric_run(&run));
                run.clear();
            }
            out.push(c);
        }
    }
    if !run.is_empty() {
        out.push_str(&convert_zh_numeric_run(&run));
    }
    out
}

fn is_zh_numeric(c: char) -> bool {
    matches!(
        c,
        '零' | '〇' | '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九'
            | '十' | '百' | '千' | '万' | '亿'
    )
}

fn zh_digit_value(c: char) -> Option<u64> {
    match c {
        '零' | '〇' => Some(0),
        '一' => Some(1),
        '二' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    }
}

fn convert_zh_numeric_run(run: &str) -> String {
    let has_positional = run
        .chars()
        .any(|c| matches!(c, '十' | '百' | '千' | '万' | '亿'));
    if has_positional {
        parse_positional(run).to_string()
    } else {
        run.chars()
            .filter_map(zh_digit_value)
            .filter_map(|d| char::from_digit(d as u32, 10))
            .collect()
    }
}

fn parse_positional(s: &str) -> u64 {
    // Standard Chinese-number scale parser. `total` holds the value
    // accumulated across 万/亿 boundaries; `section` is the in-progress
    // sub-10000 segment; `last` is the most recent unit digit awaiting
    // a 十/百/千 multiplier.
    let mut total: u64 = 0;
    let mut section: u64 = 0;
    let mut last: u64 = 0;
    let mut have_last = false;
    for c in s.chars() {
        if let Some(d) = zh_digit_value(c) {
            last = d;
            have_last = true;
        } else {
            match c {
                '十' => {
                    section += if have_last { last } else { 1 } * 10;
                    last = 0;
                    have_last = false;
                }
                '百' => {
                    section += last * 100;
                    last = 0;
                    have_last = false;
                }
                '千' => {
                    section += last * 1000;
                    last = 0;
                    have_last = false;
                }
                '万' => {
                    total += (section + last) * 10_000;
                    section = 0;
                    last = 0;
                    have_last = false;
                }
                '亿' => {
                    total += (section + last) * 100_000_000;
                    section = 0;
                    last = 0;
                    have_last = false;
                }
                _ => {}
            }
        }
    }
    total + section + last
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

    // -----------------------------------------------------------------
    // Numeral normalisation — exercises the FLEURS-zh failure modes
    // we saw in CI (refs use Arabic digits, ASR emits Chinese ones).
    // -----------------------------------------------------------------

    #[test]
    fn zh_digit_by_digit_year_form() {
        assert_eq!(normalize("二零一一年"), "2011年");
        assert_eq!(normalize("一九六三年"), "1963年");
        assert_eq!(normalize("二零零二"), "2002");
    }

    #[test]
    fn zh_positional_simple() {
        assert_eq!(normalize("十五米"), "15米");
        assert_eq!(normalize("七十多颗"), "70多颗");
        assert_eq!(normalize("二十"), "20");
        assert_eq!(normalize("二十五"), "25");
    }

    #[test]
    fn zh_positional_with_hundreds_and_thousands() {
        assert_eq!(normalize("一百二十三"), "123");
        assert_eq!(normalize("一千二百"), "1200");
        assert_eq!(normalize("三千五百"), "3500");
    }

    #[test]
    fn zh_positional_scaling_with_wan_and_yi() {
        assert_eq!(normalize("一万二千三百"), "12300");
        assert_eq!(normalize("二亿"), "200000000");
    }

    #[test]
    fn zh_lone_ten_and_zero_variants() {
        assert_eq!(normalize("十"), "10");
        assert_eq!(normalize("〇"), "0");
        // Run of pure zeros stays as zeros (digit-by-digit).
        assert_eq!(normalize("〇〇"), "00");
    }

    #[test]
    fn cer_zero_after_numeral_normalisation_zh() {
        // Verbatim from CI [zh 3] — every error in this utterance was a
        // pure number-format mismatch; with normalisation the CER must
        // collapse to zero.
        let r = "桥下垂直净空15米该项目于2011年8月完工但直到2017年3月才开始通车";
        let h = "桥下垂直净空十五米该项目于二零一一年八月完工但直到二零一七年三月才开始通车";
        assert!(cer(r, h).abs() < 1e-9, "got {}", cer(r, h));
    }

    #[test]
    fn cer_isolates_real_errors_after_normalisation_zh() {
        // CI [zh 7]: numbers normalise to zero error; the only true
        // model error is `物→雾` (1 character substitution).
        let r = "1963年大坝建成后季节性洪水被控制住了沉积物不再冲散到河流里";
        let h = "一九六三年大坝建成后季节性洪水被控制住了沉积雾不再冲散到河流里";
        let c = cer(r, h);
        // 1 sub / total chars in r (after stripping punct/whitespace)
        let r_chars = normalize(r).chars().filter(|x| !x.is_whitespace()).count();
        let expected = 1.0 / r_chars as f64;
        assert!(
            (c - expected).abs() < 1e-9,
            "expected {expected} got {c}"
        );
    }

    #[test]
    fn cer_handles_mixed_pos_and_digit_forms_zh() {
        // `403` (digit-by-digit `四零三`) and `公共→公交` (real error).
        let r = "scotturb 403 路 公 共 汽 车";
        let h = "scotburb 四零三路 公交汽车";
        // After normalisation:
        //   ref → "scotturb 403 路 公 共 汽 车" → punct/space collapsed,
        //         CER strips whitespace → "scotturb403路公共汽车"
        //   hyp → "scotburb403路公交汽车"
        // Edits: scotturb→scotburb (1 sub), 公共→公交 (1 sub). Two real errors.
        // We only assert that numeral mismatches no longer contribute.
        let r_norm: String = normalize(r).chars().filter(|x| !x.is_whitespace()).collect();
        let h_norm: String = normalize(h).chars().filter(|x| !x.is_whitespace()).collect();
        assert!(r_norm.contains("403"), "ref should contain 403, got {r_norm}");
        assert!(h_norm.contains("403"), "hyp should contain 403, got {h_norm}");
    }

    // -----------------------------------------------------------------
    // Existing-behaviour regressions — pin case folding + space stripping
    // so future normalisation tweaks can't silently drop them.
    // -----------------------------------------------------------------

    #[test]
    fn normalize_case_fold_is_locked_in() {
        // Mixed-case input must compare equal to its lowercase form.
        assert_eq!(wer("Hello World", "hello world"), 0.0);
        assert_eq!(cer("Many People", "many people"), 0.0);
    }

    #[test]
    fn cer_strips_internal_whitespace_for_cjk() {
        // FLEURS-zh references are space-separated character lists —
        // the metric must collapse those without inflating CER.
        let r = "这 并 不 是 告 别";
        let h = "这并不是告别";
        assert_eq!(cer(r, h), 0.0);
    }
}
