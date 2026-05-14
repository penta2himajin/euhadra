//! Word-level (WER) and character-level (CER) error rates.
//!
//! Two flavours, reflecting two distinct measurement intents:
//!
//! - **Strict** (`wer`, `cer`) — no surface-form normalisation. Used by
//!   the post-ASR ablation harness (`examples/eval_l1_fast.rs`), which
//!   replays a fixed JSONL hypothesis through configurations of the
//!   pipeline and measures each layer's contribution. Strict must
//!   preserve case, punctuation, hyphens and numerals so that a layer
//!   that inserts a `.` or capitalises a word produces a visible delta
//!   between `full` and `without_<layer>` configurations. Whitespace
//!   handling is intrinsic to the metric (WER tokenises by whitespace,
//!   CER strips it) and stays.
//! - **Lenient** (`wer_lenient`, `cer_lenient`) — full normalisation
//!   pass. Used by the live-ASR runners (`eval_l1_smoke`,
//!   `eval_l2`, `eval_l3`) where the goal is to compare model output
//!   against a gold transcript and absorb format-only differences:
//!   case folding, ASCII + fullwidth punctuation, smart quotes,
//!   ellipsis, hyphen splitting, Chinese numerals (`十五` → `15`,
//!   `二零一一` → `2011`), and English number words (`twelve` → `12`,
//!   `twentieth` → `20th`). This matches the convention used by
//!   Whisper / FLEURS evaluation scripts.
//!
//! ### Why the split exists
//!
//! Pre-split, both ablation and live runners called the same
//! aggressively-normalised `wer`/`cer`. The punctuation layer's
//! contribution was therefore invisible in the ablation deltas
//! (because both ref and hyp were stripped of punctuation before
//! comparison) and the baseline showed `full == without_punctuation`
//! for all three languages. Splitting the metric into strict / lenient
//! lets the ablation expose layer effects while still giving the live
//! runners a fair comparison against ground truth.

/// Word Error Rate, strict. Whitespace-tokenised; **no** other
/// normalisation. Use this for ablation comparisons where each
/// post-ASR layer's surface-form contribution must show up in the
/// score. Returns `f64::NAN` if the reference is empty (undefined).
pub fn wer(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<&str> = reference.split_whitespace().collect();
    let h: Vec<&str> = hypothesis.split_whitespace().collect();

    if r.is_empty() {
        return f64::NAN;
    }
    let dist = levenshtein(&r, &h);
    dist as f64 / r.len() as f64
}

/// Character Error Rate, strict. Whitespace is stripped before character
/// comparison so spacing differences do not inflate the score (relevant
/// for ja/zh); no other normalisation. Use this for ablation
/// comparisons where each post-ASR layer's surface-form contribution
/// must show up in the score.
pub fn cer(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<char> = reference.chars().filter(|c| !c.is_whitespace()).collect();
    let h: Vec<char> = hypothesis.chars().filter(|c| !c.is_whitespace()).collect();

    if r.is_empty() {
        return f64::NAN;
    }
    let dist = levenshtein(&r, &h);
    dist as f64 / r.len() as f64
}

/// Word Error Rate, lenient. Applies [`normalize_lenient`] to both
/// sides before tokenisation — case-folded, punctuation-stripped,
/// hyphen-split, numeral-canonicalised. Use this when comparing
/// raw ASR output to a reference where surface formatting differences
/// (`twelve` vs `12`, `whole-number` vs `whole number`,
/// `Hello, World!` vs `hello world`) are noise rather than signal.
pub fn wer_lenient(reference: &str, hypothesis: &str) -> f64 {
    let r_norm = normalize_lenient(reference);
    let h_norm = normalize_lenient(hypothesis);
    let r: Vec<&str> = r_norm.split_whitespace().collect();
    let h: Vec<&str> = h_norm.split_whitespace().collect();

    if r.is_empty() {
        return f64::NAN;
    }
    let dist = levenshtein(&r, &h);
    dist as f64 / r.len() as f64
}

/// Character Error Rate, lenient. Applies [`normalize_lenient`] to
/// both sides before character comparison.
pub fn cer_lenient(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<char> = normalize_lenient(reference)
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let h: Vec<char> = normalize_lenient(hypothesis)
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();

    if r.is_empty() {
        return f64::NAN;
    }
    let dist = levenshtein(&r, &h);
    dist as f64 / r.len() as f64
}

/// Lenient normalisation pass applied by [`wer_lenient`] / [`cer_lenient`]:
/// - lowercase ASCII letters (does not touch CJK)
/// - strip a fixed set of punctuation (ASCII + fullwidth)
/// - replace ASCII hyphens / fullwidth dashes with whitespace so
///   `whole-number` matches `whole number` and `25-30` aligns with
///   `25 30` (FLEURS-en references frequently spell hyphenated
///   compounds with a space, ASR output frequently glues them)
/// - collapse runs of whitespace into single spaces
/// - convert Chinese numerals to Arabic digits (digit-by-digit and
///   positional 十/百/千/万/亿 forms)
/// - convert Sino-Korean numerals to Arabic digits (positional
///   십/백/천/만/억/조 forms; mixed Arabic + 만 like `15만` → 150000)
/// - replace English number words with their Arabic-numeral form
///   (`twelve` → `12`, `twentieth` → `20th`) so ASR output that
///   transcribes digits doesn't lose against word-form references
pub fn normalize_lenient(text: &str) -> String {
    // Coverage notes:
    //
    // - ASCII: every standard sentence terminator + clause separator
    //   + bracket/quote pair.
    // - CJK: 。 、 「 」 『 』 ・ (Japanese), 《 》 〈 〉 〝 〟 (Chinese
    //   alternative quotes/angles), and the fullwidth ASCII clones
    //   ？ ！ ： ； （ ） ， ． that are *distinct codepoints* from
    //   their halfwidth counterparts (FF0C ≠ 002C). The pre-PR table
    //   listed `?` / `!` / `:` / `;` twice, but both occurrences were
    //   the U+003F/U+0021/... ASCII forms; the FF-block versions were
    //   missing entirely, leaving Chinese commas (`，`, FF0C) and
    //   Japanese fullwidth question marks unstripped.
    // - Smart quotes: ' ' " " (U+2018 / U+2019 / U+201C / U+201D) —
    //   Whisper / Paraformer output sometimes uses these instead of
    //   straight ASCII quotes.
    // - U+2026 horizontal ellipsis is treated as drop-punct so a
    //   trailing `…` on a hypothesis doesn't fragment the last word.
    const DROP_PUNCT: &[char] = &[
        // ASCII
        '.', ',', '?', '!', ':', ';', '"', '\'', '(', ')', '[', ']', '{', '}',
        // Japanese / CJK punctuation
        '。', '、', '「', '」', '『', '』', '・', // Fullwidth ASCII clones (FF block)
        '？', '！', '：', '；', '（', '）', '，', '．',
        // Chinese alternative brackets / quotes
        '《', '》', '〈', '〉', '〝', '〟', // Misc
        '…', '\u{2018}', '\u{2019}', // single smart quotes ' '
        '\u{201C}', '\u{201D}', // double smart quotes " "
    ];
    // Hyphens / dashes always behave as word separators, never as
    // skip-and-glue. ASCII `-`, en-dash `–`, em-dash `—`, fullwidth
    // minus `−`, and the horizontal-bar `―` (sometimes used as a
    // sentence dash in Japanese) all map the same way.
    const SPLIT_PUNCT: &[char] = &['-', '–', '—', '−', '―'];

    // Numerals first so the punctuation/case/whitespace pass below
    // operates on the canonical digit form. Order doesn't matter
    // between the Chinese and Korean passes — they operate on
    // disjoint Unicode codepoint sets (Chinese 一二三 / 十百千万亿
    // vs Korean 일이삼 / 십백천만억조).
    let digits_normalised = normalize_chinese_numerals(text);
    let digits_normalised = normalize_korean_numerals(&digits_normalised);

    let mut out = String::with_capacity(digits_normalised.len());
    let mut last_space = true; // suppress leading whitespace
    for c in digits_normalised.chars() {
        if DROP_PUNCT.contains(&c) {
            continue;
        }
        if SPLIT_PUNCT.contains(&c) || c.is_whitespace() {
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

    // English number-word replacement runs last so it sees lower-cased,
    // punctuation-free tokens — `Twelfth,` after the strip pass becomes
    // `twelfth`, which the table maps to `12th`.
    normalize_english_number_words(&out)
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
        '零' | '〇'
            | '一'
            | '二'
            | '三'
            | '四'
            | '五'
            | '六'
            | '七'
            | '八'
            | '九'
            | '十'
            | '百'
            | '千'
            | '万'
            | '亿'
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

/// Walks `text` and replaces every maximal run of Sino-Korean numeral
/// characters (with optional internal whitespace and Arabic digits)
/// with its Arabic-digit equivalent.
///
/// Run grammar:
/// - **Digit chars**: 영 / 공 (0), 일 (1), 이 (2), 삼 (3), 사 (4),
///   오 (5), 육 (6), 칠 (7), 팔 (8), 구 (9)
/// - **Unit chars**: 십 (10), 백 (100), 천 (1000), 만 (10⁴),
///   억 (10⁸), 조 (10¹²)
/// - **Arabic digits**: 0-9 (combine with units in mixed forms like
///   `15만` → 150000)
/// - **Internal whitespace**: a single space is allowed between
///   token chars, since FunASR / SenseVoice frequently emit
///   `이천 십 일` for 2011 (CTC framing artefact). The space is
///   absorbed into the run; the parser strips it before positional
///   parse.
///
/// Two normalisation paths:
/// 1. **Multi-token positional** — runs that contain **both at
///    least one digit char and at least one unit char** are
///    parsed with the standard Sino-Korean scale rules. This is
///    the high-confidence case that catches `이천 십 일 년` →
///    2011 년.
/// 2. **Single-digit + classifier** — if the run is exactly one
///    Korean digit char (no unit) and the immediately following
///    non-whitespace char is a Sino-Korean date / time
///    classifier (년 / 월 / 일 / 시 / 분 / 초), the digit is
///    normalised. This covers `팔 월` → `8 월` and `삼 월` →
///    `3 월`. The classifier list is intentionally narrow —
///    these six chars combine with Sino-Korean numerals in
///    essentially all common contexts, so the homograph risk
///    (e.g. 이 = "this" demonstrative) is dominated by the
///    Hanja-classifier disambiguation.
///
/// Everything else (single-token runs without a classifier
/// follower, runs of unit chars without digits like `천천히`)
/// passes through verbatim.
fn normalize_korean_numerals(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run = String::new();
    // Tracks whether the most recent character pushed into `run`
    // was a token (digit/unit), so a trailing whitespace is treated
    // as run-internal rather than run-terminating only when the run
    // already has content.
    let mut last_was_token = false;

    for (idx, c) in text.char_indices() {
        if is_kr_token(c) || c.is_ascii_digit() {
            run.push(c);
            last_was_token = true;
        } else if c.is_whitespace() && last_was_token {
            run.push(c);
            last_was_token = false;
        } else {
            // The "follow-up" is the unconsumed suffix starting at
            // c. We pass a string slice rather than a single char so
            // the classifier check can match multi-char tokens like
            // `세트` (English loanword "set", used as a unit
            // classifier in scoring contexts).
            flush_kr_run(&mut out, &run, Some(&text[idx..]));
            run.clear();
            last_was_token = false;
            out.push(c);
        }
    }
    flush_kr_run(&mut out, &run, None);

    out
}

fn flush_kr_run(out: &mut String, run: &str, follow_up: Option<&str>) {
    if run.is_empty() {
        return;
    }
    let trimmed = run.trim_end();
    let trailing_ws = &run[trimmed.len()..];
    if trimmed.is_empty() {
        out.push_str(run);
        return;
    }

    let cleaned: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();

    // Split the cleaned run at any "Korean-digit-after-Korean-digit"
    // boundary. Sino-Korean numbers have at most one digit per slot
    // between units, so a Korean digit immediately following another
    // Korean digit signals the end of a numeric expression — usually
    // the start of a Hanja classifier (especially `일` = day vs the
    // homographic digit 1, where positional parsing would otherwise
    // overwrite the value: `십오일` → 11 instead of "15日"). Arabic
    // digits don't trigger the split since they're naturally
    // multi-digit (`15만`, `2025년`).
    let segments = split_kr_run_at_korean_digit_boundaries(&cleaned);
    let n_segs = segments.len();
    for (i, seg) in segments.iter().enumerate() {
        let seg_follow_up: Option<&str> = if i + 1 < n_segs {
            // For inter-segment classifier disambiguation, the
            // next segment IS the follow-up — a single Korean
            // digit char like `일`, which is fine for the
            // single-char classifier check.
            Some(segments[i + 1].as_str())
        } else {
            follow_up
        };
        flush_kr_segment(out, seg, seg_follow_up);
        if i + 1 < n_segs {
            // Synthetic separator between split segments. CER
            // strips whitespace before comparison, so this doesn't
            // change scoring; it does keep the WER-side output
            // tokenisable.
            out.push(' ');
        }
    }
    out.push_str(trailing_ws);
}

fn split_kr_run_at_korean_digit_boundaries(cleaned: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut prev_was_korean_digit = false;
    for c in cleaned.chars() {
        let is_kr_digit = kr_digit_value(c).is_some();
        if is_kr_digit && prev_was_korean_digit && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(c);
        prev_was_korean_digit = is_kr_digit;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn flush_kr_segment(out: &mut String, segment: &str, follow_up: Option<&str>) {
    if segment.is_empty() {
        return;
    }
    let has_digit = segment
        .chars()
        .any(|c| kr_digit_value(c).is_some() || c.is_ascii_digit());
    let has_unit = segment.chars().any(is_kr_unit);

    if has_digit && has_unit {
        out.push_str(&parse_kr_positional(segment).to_string());
    } else if segment.chars().count() == 1
        && follow_up.map(starts_with_kr_classifier).unwrap_or(false)
    {
        // Single-token segment followed by a Korean classifier —
        // either a single-char Sino-Korean date / time classifier
        // (년 / 월 / 일 / 시 / 분 / 초) or a multi-char loanword
        // classifier (`세트`, used in tennis-style score contexts
        // like `이 세트` ↔ `2세트`). Normalise the digit when it
        // is a Korean digit char; Arabic / unit / homograph cases
        // fall through to verbatim.
        let single = segment.chars().next().unwrap();
        if let Some(d) = kr_digit_value(single) {
            out.push_str(&d.to_string());
        } else {
            out.push_str(segment);
        }
    } else {
        // Either the segment is a single token without a classifier
        // follower (`팔`, `만`, `15`) or a homograph candidate
        // (`천천` after trimming the trailing non-numeric char).
        // Leave the original segment text intact.
        out.push_str(segment);
    }
}

fn starts_with_kr_classifier(s: &str) -> bool {
    // Single-char date / time classifiers: 년 / 월 / 일 / 시 / 분 / 초.
    if let Some(c) = s.chars().next() {
        if is_kr_date_time_classifier(c) {
            return true;
        }
    }
    // Multi-char classifiers — only `세트` (English "set") so far,
    // motivated by the FLEURS-ko tennis utterance fragment
    // `이 세트` ↔ `2세트` / `2세트에서`. Korean particles after
    // `세트` (가, 를, 에서, 의, …) are NOT excluded — they don't
    // disambiguate "n sets" from anything else in practice. The
    // residual false-positive surface (e.g. `이 세트장` =
    // "this set venue") is small enough to accept; the homograph
    // is rare in FLEURS-ko reference text and CER is bounded by
    // utterance length.
    s.starts_with("세트")
}

fn is_kr_date_time_classifier(c: char) -> bool {
    matches!(c, '년' | '월' | '일' | '시' | '분' | '초')
}

fn is_kr_token(c: char) -> bool {
    kr_digit_value(c).is_some() || is_kr_unit(c)
}

fn is_kr_unit(c: char) -> bool {
    matches!(c, '십' | '백' | '천' | '만' | '억' | '조')
}

fn kr_digit_value(c: char) -> Option<u64> {
    match c {
        '영' | '공' => Some(0),
        '일' => Some(1),
        '이' => Some(2),
        '삼' => Some(3),
        '사' => Some(4),
        '오' => Some(5),
        '육' => Some(6),
        '칠' => Some(7),
        '팔' => Some(8),
        '구' => Some(9),
        _ => None,
    }
}

fn parse_kr_positional(s: &str) -> u64 {
    // Sino-Korean number scale parser. `total` accumulates across
    // 만 / 억 / 조 boundaries; `section` is the in-progress
    // sub-10000 segment; `current` is the most recent base value
    // (single Korean digit OR multi-digit Arabic block) awaiting a
    // 십/백/천 multiplier. Implicit-1 applies to every unit char so
    // bare `만 년` → 10000 and `천 명` → 1000.
    let mut total: u64 = 0;
    let mut section: u64 = 0;
    let mut current: u64 = 0;
    let mut have_current = false;

    for c in s.chars() {
        if let Some(d) = c.to_digit(10) {
            current = current * 10 + d as u64;
            have_current = true;
        } else if let Some(d) = kr_digit_value(c) {
            current = d;
            have_current = true;
        } else {
            match c {
                '십' => {
                    let v = if have_current { current } else { 1 };
                    section += v * 10;
                    current = 0;
                    have_current = false;
                }
                '백' => {
                    let v = if have_current { current } else { 1 };
                    section += v * 100;
                    current = 0;
                    have_current = false;
                }
                '천' => {
                    let v = if have_current { current } else { 1 };
                    section += v * 1000;
                    current = 0;
                    have_current = false;
                }
                '만' => {
                    let v = section + current;
                    let v = if v == 0 { 1 } else { v };
                    total += v * 10_000;
                    section = 0;
                    current = 0;
                    have_current = false;
                }
                '억' => {
                    let v = section + current;
                    let v = if v == 0 { 1 } else { v };
                    total += v * 100_000_000;
                    section = 0;
                    current = 0;
                    have_current = false;
                }
                '조' => {
                    let v = section + current;
                    let v = if v == 0 { 1 } else { v };
                    total += v * 1_000_000_000_000;
                    section = 0;
                    current = 0;
                    have_current = false;
                }
                _ => {}
            }
        }
    }
    total + section + current
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
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Token-level replacement of common English cardinals and ordinals
/// with their Arabic-numeral form. FLEURS-en references freely mix
/// `twelve`/`12`, `twentieth`/`20th`; without this pass each pair
/// reads as an unrelated substitution and inflates WER.
///
/// The table is intentionally limited to single-token forms (0-19,
/// the 20–90 multiples of ten, and `hundred`/`thousand`/`million`
/// plus their ordinal counterparts). Compound numbers like
/// `twenty-three` are split by the SPLIT_PUNCT pass above into two
/// tokens and each is normalised independently — imperfect for
/// pure compound handling but ample for the FLEURS smoke set.
fn normalize_english_number_words(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    text.split(' ')
        .map(|tok| english_number_replacement(tok).unwrap_or(tok))
        .collect::<Vec<_>>()
        .join(" ")
}

fn english_number_replacement(tok: &str) -> Option<&'static str> {
    match tok {
        // Cardinals
        "zero" => Some("0"),
        "one" => Some("1"),
        "two" => Some("2"),
        "three" => Some("3"),
        "four" => Some("4"),
        "five" => Some("5"),
        "six" => Some("6"),
        "seven" => Some("7"),
        "eight" => Some("8"),
        "nine" => Some("9"),
        "ten" => Some("10"),
        "eleven" => Some("11"),
        "twelve" => Some("12"),
        "thirteen" => Some("13"),
        "fourteen" => Some("14"),
        "fifteen" => Some("15"),
        "sixteen" => Some("16"),
        "seventeen" => Some("17"),
        "eighteen" => Some("18"),
        "nineteen" => Some("19"),
        "twenty" => Some("20"),
        "thirty" => Some("30"),
        "forty" => Some("40"),
        "fifty" => Some("50"),
        "sixty" => Some("60"),
        "seventy" => Some("70"),
        "eighty" => Some("80"),
        "ninety" => Some("90"),
        "hundred" => Some("100"),
        "thousand" => Some("1000"),
        "million" => Some("1000000"),
        // Ordinals — preserve the suffix so `20th` (Arabic) and
        // `twentieth` (word) collapse to a common form.
        "first" => Some("1st"),
        "second" => Some("2nd"),
        "third" => Some("3rd"),
        "fourth" => Some("4th"),
        "fifth" => Some("5th"),
        "sixth" => Some("6th"),
        "seventh" => Some("7th"),
        "eighth" => Some("8th"),
        "ninth" => Some("9th"),
        "tenth" => Some("10th"),
        "eleventh" => Some("11th"),
        "twelfth" => Some("12th"),
        "thirteenth" => Some("13th"),
        "fourteenth" => Some("14th"),
        "fifteenth" => Some("15th"),
        "sixteenth" => Some("16th"),
        "seventeenth" => Some("17th"),
        "eighteenth" => Some("18th"),
        "nineteenth" => Some("19th"),
        "twentieth" => Some("20th"),
        "thirtieth" => Some("30th"),
        "fortieth" => Some("40th"),
        "fiftieth" => Some("50th"),
        "sixtieth" => Some("60th"),
        "seventieth" => Some("70th"),
        "eightieth" => Some("80th"),
        "ninetieth" => Some("90th"),
        "hundredth" => Some("100th"),
        "thousandth" => Some("1000th"),
        "millionth" => Some("1000000th"),
        _ => None,
    }
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
    fn wer_lenient_normalises_punctuation_and_case() {
        assert_eq!(wer_lenient("Hello, world!", "hello world"), 0.0);
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
    fn cer_lenient_strips_japanese_punctuation() {
        assert_eq!(cer_lenient("今日は、天気がいい。", "今日は天気がいい"), 0.0);
    }

    #[test]
    fn normalize_lenient_collapses_whitespace() {
        assert_eq!(normalize_lenient("  hello   world  "), "hello world");
    }

    #[test]
    fn normalize_lenient_lowercases_ascii_only() {
        assert_eq!(normalize_lenient("Hello CAFÉ"), "hello café");
    }

    // -----------------------------------------------------------------
    // Numeral normalisation — exercises the FLEURS-zh failure modes
    // we saw in CI (refs use Arabic digits, ASR emits Chinese ones).
    // -----------------------------------------------------------------

    #[test]
    fn zh_digit_by_digit_year_form() {
        assert_eq!(normalize_lenient("二零一一年"), "2011年");
        assert_eq!(normalize_lenient("一九六三年"), "1963年");
        assert_eq!(normalize_lenient("二零零二"), "2002");
    }

    #[test]
    fn zh_positional_simple() {
        assert_eq!(normalize_lenient("十五米"), "15米");
        assert_eq!(normalize_lenient("七十多颗"), "70多颗");
        assert_eq!(normalize_lenient("二十"), "20");
        assert_eq!(normalize_lenient("二十五"), "25");
    }

    #[test]
    fn zh_positional_with_hundreds_and_thousands() {
        assert_eq!(normalize_lenient("一百二十三"), "123");
        assert_eq!(normalize_lenient("一千二百"), "1200");
        assert_eq!(normalize_lenient("三千五百"), "3500");
    }

    #[test]
    fn zh_positional_scaling_with_wan_and_yi() {
        assert_eq!(normalize_lenient("一万二千三百"), "12300");
        assert_eq!(normalize_lenient("二亿"), "200000000");
    }

    #[test]
    fn zh_lone_ten_and_zero_variants() {
        assert_eq!(normalize_lenient("十"), "10");
        assert_eq!(normalize_lenient("〇"), "0");
        // Run of pure zeros stays as zeros (digit-by-digit).
        assert_eq!(normalize_lenient("〇〇"), "00");
    }

    #[test]
    fn cer_lenient_zero_after_numeral_normalisation_zh() {
        // Verbatim from CI [zh 3] — every error in this utterance was a
        // pure number-format mismatch; with normalisation the CER must
        // collapse to zero.
        let r = "桥下垂直净空15米该项目于2011年8月完工但直到2017年3月才开始通车";
        let h = "桥下垂直净空十五米该项目于二零一一年八月完工但直到二零一七年三月才开始通车";
        assert!(cer_lenient(r, h).abs() < 1e-9, "got {}", cer_lenient(r, h));
    }

    #[test]
    fn cer_lenient_isolates_real_errors_after_normalisation_zh() {
        // CI [zh 7]: numbers normalise to zero error; the only true
        // model error is `物→雾` (1 character substitution).
        let r = "1963年大坝建成后季节性洪水被控制住了沉积物不再冲散到河流里";
        let h = "一九六三年大坝建成后季节性洪水被控制住了沉积雾不再冲散到河流里";
        let c = cer_lenient(r, h);
        // 1 sub / total chars in r (after stripping punct/whitespace)
        let r_chars = normalize_lenient(r)
            .chars()
            .filter(|x| !x.is_whitespace())
            .count();
        let expected = 1.0 / r_chars as f64;
        assert!((c - expected).abs() < 1e-9, "expected {expected} got {c}");
    }

    #[test]
    fn cer_lenient_handles_mixed_pos_and_digit_forms_zh() {
        // `403` (digit-by-digit `四零三`) and `公共→公交` (real error).
        let r = "scotturb 403 路 公 共 汽 车";
        let h = "scotburb 四零三路 公交汽车";
        // After normalisation:
        //   ref → "scotturb 403 路 公 共 汽 车" → punct/space collapsed,
        //         CER strips whitespace → "scotturb403路公共汽车"
        //   hyp → "scotburb403路公交汽车"
        // Edits: scotturb→scotburb (1 sub), 公共→公交 (1 sub). Two real errors.
        // We only assert that numeral mismatches no longer contribute.
        let r_norm: String = normalize_lenient(r)
            .chars()
            .filter(|x| !x.is_whitespace())
            .collect();
        let h_norm: String = normalize_lenient(h)
            .chars()
            .filter(|x| !x.is_whitespace())
            .collect();
        assert!(
            r_norm.contains("403"),
            "ref should contain 403, got {r_norm}"
        );
        assert!(
            h_norm.contains("403"),
            "hyp should contain 403, got {h_norm}"
        );
    }

    // -----------------------------------------------------------------
    // Existing-behaviour regressions — pin case folding + space stripping
    // so future normalisation tweaks can't silently drop them.
    // -----------------------------------------------------------------

    #[test]
    fn normalize_lenient_case_fold_is_locked_in() {
        // Mixed-case input must compare equal to its lowercase form
        // under the lenient pass.
        assert_eq!(wer_lenient("Hello World", "hello world"), 0.0);
        assert_eq!(cer_lenient("Many People", "many people"), 0.0);
    }

    #[test]
    fn cer_strips_internal_whitespace_for_cjk() {
        // FLEURS-zh references are space-separated character lists —
        // the metric must collapse those without inflating CER.
        let r = "这 并 不 是 告 别";
        let h = "这并不是告别";
        assert_eq!(cer(r, h), 0.0);
    }

    // -----------------------------------------------------------------
    // English numeral + hyphen normalisation — covers the FLEURS-en
    // pairs the smoke run flagged: `twentieth`/`20th`, `twelve`/`12`,
    // `whole-number`/`whole number`, `25-30`/`25 30`.
    // -----------------------------------------------------------------

    #[test]
    fn en_lenient_cardinals_round_trip_to_arabic() {
        assert_eq!(wer_lenient("twelve", "12"), 0.0);
        assert_eq!(wer_lenient("twenty", "20"), 0.0);
        assert_eq!(wer_lenient("hundred", "100"), 0.0);
        assert_eq!(wer_lenient("thousand", "1000"), 0.0);
    }

    #[test]
    fn en_lenient_ordinals_round_trip_to_th_form() {
        assert_eq!(wer_lenient("twentieth century", "20th century"), 0.0);
        assert_eq!(wer_lenient("first place", "1st place"), 0.0);
        assert_eq!(wer_lenient("twelfth night", "12th night"), 0.0);
        assert_eq!(
            wer_lenient("third time's the charm", "3rd time's the charm"),
            0.0
        );
    }

    #[test]
    fn en_lenient_hyphen_splits_compound_words() {
        // FLEURS-en [en 9] failure: `whole-number ratio` vs
        // `whole number ratio` should be a no-op.
        assert_eq!(wer_lenient("whole-number ratio", "whole number ratio"), 0.0);
    }

    #[test]
    fn en_lenient_hyphen_splits_numeric_ranges() {
        // [en 1] partial fix: `25-30` and `25 30` align after the
        // hyphen pass. The remaining `to` between is a real model-vs-
        // reference difference and stays as a single edit.
        let r = "25 to 30 years";
        let h = "25-30 years";
        // r tokens: 25 to 30 years
        // h tokens (after hyphen→space, lowercase): 25 30 years
        // edit distance = 1 (delete `to`) → WER 1/4 = 0.25
        let w = wer_lenient(r, h);
        assert!((w - 0.25).abs() < 1e-9, "expected 0.25, got {w}");
    }

    #[test]
    fn en_lenient_dump_regressions_align_after_normalisation() {
        // [en 8] verbatim from the CI dump.
        let r = "twentieth century research has shown that there are two pools of genetic variation hidden and expressed";
        let h = "20th century research has shown that there are two pools of genetic variation hidden and expressed";
        // After numeral normalisation both sides start with `20th
        // century`, so the only remaining edits are zero.
        assert_eq!(wer_lenient(r, h), 0.0);
    }

    #[test]
    fn en_lenient_compound_separator_does_not_glue_tokens() {
        // Hyphens must produce a word boundary, not delete-and-glue.
        // `25-30` would otherwise normalise to `2530` (single token)
        // and never align against `25 30` (two tokens).
        let normalized = normalize_lenient("25-30");
        assert_eq!(normalized, "25 30");
    }

    #[test]
    fn en_lenient_em_and_en_dash_treated_as_split() {
        assert_eq!(normalize_lenient("alpha\u{2014}beta"), "alpha beta"); // em-dash
        assert_eq!(normalize_lenient("alpha\u{2013}beta"), "alpha beta"); // en-dash
    }

    #[test]
    fn en_lenient_number_words_only_replace_whole_tokens() {
        // `oneness` must NOT become `1ness`; the replacement is
        // token-level, not substring-level.
        assert_eq!(normalize_lenient("oneness"), "oneness");
        assert_eq!(normalize_lenient("twenties"), "twenties");
        assert_eq!(normalize_lenient("hundreds"), "hundreds");
    }

    #[test]
    fn en_lenient_normalisation_preserves_non_number_tokens() {
        // Spot-check that we don't accidentally trip over English
        // function words that look number-adjacent.
        assert_eq!(
            normalize_lenient("the quick brown fox"),
            "the quick brown fox"
        );
    }

    // -----------------------------------------------------------------
    // Punctuation coverage — fullwidth FF-block forms are *distinct
    // codepoints* from their ASCII counterparts, so each must be in
    // the table. The pre-existing `'?', '!', ':', ';', '(', ')'` on
    // the second row of the table were ASCII duplicates, not the
    // fullwidth versions; these tests would have caught that.
    // -----------------------------------------------------------------

    #[test]
    fn lenient_drop_punct_covers_fullwidth_ff_block() {
        // U+FF0C 「，」 (Chinese fullwidth comma) is the most common
        // miss — it's what the Paraformer / Whisper-zh adapters emit
        // between clauses.
        assert_eq!(cer_lenient("你好，世界", "你好世界"), 0.0);
        assert_eq!(cer_lenient("你好世界", "你好，世界"), 0.0);
        // Symmetric coverage of the rest of the FF block.
        for p in &['？', '！', '：', '；', '（', '）', '．'] {
            let with_punct = format!("hello{p}world");
            let without = "helloworld";
            assert_eq!(
                cer_lenient(&with_punct, without),
                0.0,
                "punct {p:?} not stripped"
            );
        }
    }

    #[test]
    fn lenient_drop_punct_covers_japanese_brackets_and_dot() {
        // 「 」 『 』 ・ — all already there pre-PR; pinned here so
        // future cleanups don't drop them.
        assert_eq!(cer_lenient("「今日は」、晴れだ。", "今日は晴れだ"), 0.0);
        assert_eq!(cer_lenient("『良』『書』", "良書"), 0.0);
        assert_eq!(cer_lenient("カタログ・データ", "カタログデータ"), 0.0);
    }

    #[test]
    fn lenient_drop_punct_covers_smart_quotes_and_ellipsis() {
        // U+2018 / U+2019 / U+201C / U+201D — Whisper output sometimes
        // produces these instead of ASCII quotes.
        assert_eq!(wer_lenient("\u{2018}hello\u{2019}", "hello"), 0.0);
        assert_eq!(wer_lenient("\u{201C}hello\u{201D}", "hello"), 0.0);
        // Horizontal ellipsis must not fragment the trailing word.
        assert_eq!(cer_lenient("そうです…", "そうです"), 0.0);
    }

    // -----------------------------------------------------------------
    // Symmetry — `normalize_lenient()` must be applied to BOTH
    // reference and hypothesis. If either side were left raw, the
    // whole point of normalisation collapses (one side carries
    // punctuation/case the other doesn't, and Levenshtein over those
    // is essentially no-better-than-no-normalisation).
    // -----------------------------------------------------------------

    #[test]
    fn lenient_normalisation_is_symmetric_across_ref_and_hyp() {
        // ref dirty, hyp clean.
        assert_eq!(wer_lenient("Hello, World!", "hello world"), 0.0);
        // hyp dirty, ref clean.
        assert_eq!(wer_lenient("hello world", "Hello, World!"), 0.0);
        // ref carries fullwidth, hyp ASCII.
        assert_eq!(cer_lenient("你好，世界", "你好世界"), 0.0);
        // hyp carries fullwidth, ref clean.
        assert_eq!(cer_lenient("你好世界", "你好，世界"), 0.0);
        // ref carries Chinese numerals, hyp Arabic.
        assert_eq!(cer_lenient("二零二一年", "2021年"), 0.0);
        assert_eq!(cer_lenient("2021年", "二零二一年"), 0.0);
        // ref word form, hyp Arabic (and the reverse).
        assert_eq!(wer_lenient("twentieth century", "20th century"), 0.0);
        assert_eq!(wer_lenient("20th century", "twentieth century"), 0.0);
    }

    // -----------------------------------------------------------------
    // Whitespace coverage — `char::is_whitespace` follows Unicode's
    // White_Space property, which includes U+3000 IDEOGRAPHIC SPACE
    // (full-width 「　」). FLEURS-ja references occasionally use it
    // as a clause separator instead of `、`, and Paraformer output
    // never does, so failure to strip it would inflate ja CER.
    // -----------------------------------------------------------------

    #[test]
    fn fullwidth_space_is_collapsed_for_wer() {
        // U+3000 between two ASCII tokens must read as a word boundary,
        // not a substitution.
        assert_eq!(wer("hello\u{3000}world", "hello world"), 0.0);
    }

    #[test]
    fn fullwidth_space_is_stripped_for_cer() {
        // U+3000 between CJK characters must collapse the same way
        // ASCII spaces do (the CJK CER path strips ALL whitespace).
        let with = "今日\u{3000}は\u{3000}晴れだ";
        let without = "今日は晴れだ";
        assert_eq!(cer(with, without), 0.0);
    }

    #[test]
    fn mixed_whitespace_kinds_collapse_uniformly() {
        // ASCII space, tab, NBSP (U+00A0), and ideographic space
        // (U+3000) all need to behave identically as separators.
        assert_eq!(wer("a\tb", "a b"), 0.0);
        assert_eq!(wer("a\u{00A0}b", "a b"), 0.0);
        assert_eq!(wer("a\u{3000}b", "a b"), 0.0);
    }

    // -----------------------------------------------------------------
    // Strict vs lenient split. `wer`/`cer` (strict) are the metric used
    // by the ablation harness (eval_l1_fast): they preserve case,
    // punctuation, hyphens, and numerals so each post-ASR layer's
    // contribution is visible in the deltas. `wer_lenient`/`cer_lenient`
    // apply the full surface-form normalisation pass and are used by
    // the live-ASR runners (eval_l1_smoke / eval_l2 / eval_l3) where
    // the goal is to absorb format-only differences between model
    // output and gold transcript.
    //
    // The contract these tests pin: strict must NOT silently fold
    // anything the punctuation layer might insert/change. Otherwise
    // `full` and `without_punctuation` collapse to identical numbers
    // and the ablation reports zero contribution — which is exactly
    // the bug the previous baseline suffered from
    // (full == without_punctuation across all three languages).
    // -----------------------------------------------------------------

    #[test]
    fn strict_wer_preserves_punctuation_difference() {
        // BasicPunctuationRestorer's headline behaviour is appending
        // a terminal `.` to Latin-script utterances. Strict WER must
        // count that as one inserted token vs the bare reference.
        let r = "hello world";
        let h = "hello world.";
        // Tokens (split_whitespace): r=[hello, world], h=[hello, world.]
        // 1 substitution → 1/2 = 0.5
        assert!(
            (wer(r, h) - 0.5).abs() < 1e-9,
            "strict wer should see `.` token, got {}",
            wer(r, h)
        );
    }

    #[test]
    fn strict_cer_preserves_punctuation_difference() {
        let r = "hello world";
        let h = "hello world.";
        // Chars after whitespace strip: r="helloworld" (10), h="helloworld." (11)
        // 1 insertion / 10 ref chars
        let c = cer(r, h);
        assert!(
            (c - 0.1).abs() < 1e-9,
            "strict cer should see `.` char, got {c}"
        );
    }

    #[test]
    fn strict_wer_preserves_capitalisation_difference() {
        // Punct restorer capitalises first word + after `.`. Strict
        // WER must register that as substitution.
        let r = "hello world";
        let h = "Hello world";
        // Tokens differ on first → 1/2 = 0.5
        assert!((wer(r, h) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn strict_cer_preserves_capitalisation_difference() {
        let r = "hello world";
        let h = "Hello world";
        // 1 char sub / 10 chars (whitespace stripped)
        let c = cer(r, h);
        assert!((c - 0.1).abs() < 1e-9, "got {c}");
    }

    #[test]
    fn strict_cer_preserves_chinese_comma() {
        // Under strict, Paraformer's clause-comma surfaces as a real
        // edit against a punctuation-free FLEURS reference. That is
        // the desired ablation signal — the punctuation layer's
        // contribution comes from REMOVING the difference (or, in the
        // BasicPunctuationRestorer case, from preserving the gap).
        let r = "你好世界";
        let h = "你好，世界";
        // 1 insertion / 4 ref chars = 0.25
        let c = cer(r, h);
        assert!(
            (c - 0.25).abs() < 1e-9,
            "strict cer should see fullwidth comma, got {c}"
        );
    }

    #[test]
    fn strict_metrics_still_collapse_whitespace() {
        // Whitespace is intrinsic to tokenisation (WER) and char
        // streaming (CER), so it stays normalised even in strict mode.
        // Otherwise `hello  world` vs `hello world` would inflate WER
        // for a difference no layer would ever produce.
        assert_eq!(wer("hello  world", "hello world"), 0.0);
        assert_eq!(cer("今日は", "今日 は"), 0.0);
    }

    #[test]
    fn lenient_wer_strips_punctuation_and_case() {
        // Lenient retains the full pre-split semantics — used by the
        // live-ASR runners where ref/hyp surface-form differences
        // (case, punct, numeral spelling) are noise, not signal.
        assert_eq!(wer_lenient("Hello, World!", "hello world"), 0.0);
        assert_eq!(wer_lenient("hello world", "Hello, World!"), 0.0);
    }

    #[test]
    fn lenient_cer_strips_chinese_comma_and_normalises_numerals() {
        assert_eq!(cer_lenient("你好，世界", "你好世界"), 0.0);
        assert_eq!(cer_lenient("二零二一年", "2021年"), 0.0);
    }

    #[test]
    fn lenient_wer_normalises_english_number_words_and_hyphens() {
        assert_eq!(wer_lenient("twentieth century", "20th century"), 0.0);
        assert_eq!(wer_lenient("whole-number ratio", "whole number ratio"), 0.0);
    }

    #[test]
    fn normalize_lenient_pipeline_matches_legacy_normalize_contract() {
        // Spot-check that normalize_lenient does what the old
        // normalize() did across all five passes (case, punct, hyphen,
        // zh numerals, en numerals, smart quotes, ellipsis, fullwidth
        // space). If any of these regresses, the smoke runner's CER
        // jumps without warning.
        assert_eq!(normalize_lenient("Hello CAFÉ"), "hello café");
        assert_eq!(normalize_lenient("25-30"), "25 30");
        assert_eq!(normalize_lenient("十五米"), "15米");
        assert_eq!(normalize_lenient("twelve"), "12");
        assert_eq!(normalize_lenient("\u{2018}hello\u{2019}"), "hello");
        assert_eq!(normalize_lenient("そうです…"), "そうです");
        assert_eq!(normalize_lenient("hello\u{3000}world"), "hello world");
    }

    // -----------------------------------------------------------------
    // Korean (Sino-Korean) numeral normalisation — issue #47 part 2 fix.
    // -----------------------------------------------------------------

    #[test]
    fn normalize_korean_numerals_two_token_run() {
        // FLEURS-ko utterance 1 fragment: "15미터" vs SenseVoice's
        // "십 오 미터". The numeral run must collapse to 15.
        assert_eq!(normalize_korean_numerals("십 오 미터"), "15 미터");
    }

    #[test]
    fn normalize_korean_numerals_year_with_unit_chain() {
        // FLEURS-ko utterance 1: "2011년" vs "이천 십 일 년".
        assert_eq!(normalize_korean_numerals("이천 십 일 년"), "2011 년");
        assert_eq!(normalize_korean_numerals("이천 십 칠 년"), "2017 년");
    }

    #[test]
    fn normalize_korean_numerals_full_thousand_chain() {
        // FLEURS-ko utterance 7: "1940년" vs "천 구백 사십 년".
        assert_eq!(normalize_korean_numerals("천 구백 사십 년"), "1940 년");
    }

    #[test]
    fn normalize_korean_numerals_man_compound() {
        // "일만" → 10000.
        assert_eq!(normalize_korean_numerals("일만 년"), "10000 년");
        assert_eq!(normalize_korean_numerals("일 만 년"), "10000 년");
    }

    #[test]
    fn normalize_korean_numerals_mixed_arabic_with_man() {
        // "15만" — Arabic base + Korean unit. ASR / dictation forms
        // commonly mix these, the parser must combine them as
        // 15 × 10000 = 150000.
        assert_eq!(normalize_korean_numerals("15만"), "150000");
        assert_eq!(normalize_korean_numerals("12만 명"), "120000 명");
    }

    #[test]
    fn normalize_korean_numerals_eok_billion() {
        // 일억 = 100,000,000.
        assert_eq!(normalize_korean_numerals("일억 원"), "100000000 원");
        // 삼억 오천만 = 350,000,000.
        assert_eq!(normalize_korean_numerals("삼억 오천만 원"), "350000000 원");
    }

    #[test]
    fn normalize_korean_numerals_skips_homographs_without_unit() {
        // Single digit chars in non-numeric words: "이것은" (this
        // is...), "구매" (purchase), "사람" (person). None contains
        // a unit char so the conservative pass leaves them alone.
        assert_eq!(normalize_korean_numerals("이것은"), "이것은");
        assert_eq!(normalize_korean_numerals("구매"), "구매");
        assert_eq!(normalize_korean_numerals("사람"), "사람");
    }

    #[test]
    fn normalize_korean_numerals_skips_unit_only_homographs() {
        // "천천히" (slowly) — two unit chars 천천 followed by 히.
        // Has unit but no digit char, so leave verbatim.
        assert_eq!(normalize_korean_numerals("천천히"), "천천히");
        // "만나다" (to meet) — one unit char + non-digit.
        assert_eq!(normalize_korean_numerals("만나다"), "만나다");
    }

    #[test]
    fn normalize_korean_numerals_single_digit_alone_kept() {
        // "팔" alone could be 8 or "arm". Without surrounding
        // numeric context (no classifier, no other digit) we keep
        // it verbatim.
        assert_eq!(normalize_korean_numerals("팔"), "팔");
    }

    #[test]
    fn normalize_korean_numerals_single_digit_with_date_classifier() {
        // FLEURS-ko utterance 1 / 7 fragments: `팔 월` and `삼 월`
        // remain as residual mismatches after the multi-token
        // positional pass. The single-digit + classifier branch
        // catches them — the classifier (월) is unambiguously
        // Sino-Korean, so 팔 disambiguates to the numeral 8.
        assert_eq!(normalize_korean_numerals("팔 월"), "8 월");
        assert_eq!(normalize_korean_numerals("삼 월에"), "3 월에");
        assert_eq!(normalize_korean_numerals("일 일"), "1 일");
        assert_eq!(normalize_korean_numerals("오 시"), "5 시");
        assert_eq!(normalize_korean_numerals("구 분"), "9 분");
        assert_eq!(normalize_korean_numerals("육 초"), "6 초");
        assert_eq!(normalize_korean_numerals("이 년"), "2 년");
    }

    #[test]
    fn normalize_korean_numerals_classifier_only_widens_for_dates() {
        // 명 / 개 / 대 / 호 etc are Korean classifiers but they
        // aren't in the conservative date-time list. Keep them
        // verbatim to avoid widening the false-positive surface.
        assert_eq!(normalize_korean_numerals("팔 명"), "팔 명");
        assert_eq!(normalize_korean_numerals("이 개"), "이 개");
        assert_eq!(normalize_korean_numerals("삼 대"), "삼 대");
    }

    #[test]
    fn normalize_korean_numerals_run_followed_by_classifier_unaffected() {
        // Multi-token positional runs already win their digit case.
        // A trailing classifier doesn't change the outcome.
        assert_eq!(normalize_korean_numerals("이천 십 일 년"), "2011 년");
    }

    #[test]
    fn normalize_korean_numerals_split_at_consecutive_korean_digits() {
        // FLEURS-ko utterance 7 fragment: `십 오 일` literally means
        // "15 + day", but a naïve positional parse would compute
        // 0 + 10 + 1 = 11 because the trailing 일 (digit 1)
        // overwrites 오 (digit 5) inside the same digit slot.
        // Sino-Korean disallows two digits in a row without an
        // intervening unit, so the split heuristic isolates `십 오`
        // (→ 15) from `일` (→ classifier-tagged "day").
        assert_eq!(normalize_korean_numerals("십 오 일"), "15 일");
        assert_eq!(normalize_korean_numerals("십오일"), "15 일");
    }

    #[test]
    fn normalize_korean_numerals_split_doesnt_break_normal_positional() {
        // The split must NOT trigger when consecutive digits are
        // separated by a unit char — `이천 십 일` is the year 2011,
        // not 2010 + day.
        assert_eq!(normalize_korean_numerals("이천 십 일"), "2011");
        // And mixed Arabic + Korean unit forms must not split since
        // `15` is one multi-digit Arabic block, not two Korean
        // digits.
        assert_eq!(normalize_korean_numerals("15만"), "150000");
    }

    #[test]
    fn normalize_korean_numerals_split_with_classifier_followup() {
        // `십오일년` (CTC-glued "15-day-year"?) is incoherent, but
        // the run-split heuristic decomposes it into `15` + `일`,
        // and the trailing `년` then provides the classifier
        // disambiguation for the second segment. The 일 is treated
        // as digit 1 because 년 is a date classifier following it.
        assert_eq!(normalize_korean_numerals("십오일년"), "15 1년");
    }

    #[test]
    fn normalize_korean_numerals_multi_char_classifier_set() {
        // FLEURS-ko utterance 2 fragment: `이 세트` ↔ `2세트`.
        // `세트` is an English loanword used as a unit classifier
        // in scoring contexts (tennis sets, etc.). The follow-up
        // check matches the multi-char prefix even though `세`
        // alone is not in the single-char classifier whitelist.
        assert_eq!(normalize_korean_numerals("이 세트"), "2 세트");
        // Korean particles after `세트` don't disrupt the match.
        assert_eq!(normalize_korean_numerals("이 세트에서"), "2 세트에서");
        assert_eq!(
            normalize_korean_numerals("이 세트가 끝나고"),
            "2 세트가 끝나고"
        );
    }

    #[test]
    fn normalize_korean_numerals_bare_se_is_not_classifier() {
        // `세` alone (without `트`) does NOT trigger the classifier
        // path — `세` overlaps with the native Korean cardinal "3"
        // and the noun 世 (world / age), so we don't want to
        // normalise digits before it. Same goes for any Hangul
        // syllable that isn't on the explicit allow-list.
        assert_eq!(normalize_korean_numerals("이 세"), "이 세");
        assert_eq!(normalize_korean_numerals("이 세상"), "이 세상");
    }

    #[test]
    fn normalize_korean_numerals_classifier_skips_arabic_digit() {
        // Arabic digit + classifier should stay verbatim — the
        // ref form `2세트` already lacks the run-internal space
        // and CER's whitespace strip collapses any output spacing
        // anyway.
        assert_eq!(normalize_korean_numerals("2 세트"), "2 세트");
        assert_eq!(normalize_korean_numerals("2세트"), "2세트");
    }

    #[test]
    fn normalize_korean_numerals_single_internal_space_absorbed() {
        // A single space inside the run is absorbed into the run
        // (FunASR / SenseVoice routinely emit `이천 십 일` for 2011).
        assert_eq!(normalize_korean_numerals("이천 십"), "2010");
    }

    #[test]
    fn normalize_korean_numerals_double_space_terminates_run() {
        // Two consecutive spaces fall back to "end of run" semantics —
        // the second whitespace is a run-terminator because the
        // previous char was already whitespace, not a token. The
        // first sub-run normalises; the second is single-token so it
        // stays verbatim.
        assert_eq!(normalize_korean_numerals("이천  십"), "2000  십");
    }

    #[test]
    fn normalize_korean_numerals_passes_through_non_numeric() {
        // Hangul without numerals must not be touched.
        assert_eq!(
            normalize_korean_numerals("다리 밑 수직 간격은 미터이며"),
            "다리 밑 수직 간격은 미터이며"
        );
    }

    #[test]
    fn lenient_cer_normalises_korean_numerals_against_arabic_reference() {
        // FLEURS-ko utterance 1 mini-fragment. Strict CER would
        // flag every digit-form mismatch as a substitution; lenient
        // CER must collapse them to zero after both sides normalise.
        let reference = "공사는 2011년 8월에 마무리되었으며";
        // "8월" → "팔 월" stays unmatched (single-digit run) so we
        // pick a fragment without a single-digit segment to keep
        // this test focused on the multi-token positional fix.
        let hypothesis = "공사는 이천 십 일 년 8월에 마무리되었으며";
        let cer = cer_lenient(reference, hypothesis);
        assert!(
            cer < 0.05,
            "expected cer < 0.05 after kr numeral norm, got {cer}"
        );
    }
}
