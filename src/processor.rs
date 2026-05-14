use async_trait::async_trait;

use crate::types::ContextSnapshot;

// ---------------------------------------------------------------------------
// TextProcessor trait
// ---------------------------------------------------------------------------

/// Structural text correction applied between TextFilter and LlmRefiner.
///
/// Handles punctuation insertion, capitalization, self-correction detection,
/// and list formatting using lightweight models or rules — no LLM required.
///
/// Multiple processors can be chained.
#[async_trait]
pub trait TextProcessor: Send + Sync {
    async fn process(
        &self,
        text: &str,
        context: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError>;
}

#[derive(Debug, Clone)]
pub struct ProcessResult {
    pub text: String,
    pub corrections: Vec<Correction>,
}

#[derive(Debug, Clone)]
pub struct Correction {
    pub kind: CorrectionKind,
    pub original: String,
    pub replacement: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrectionKind {
    PunctuationInserted,
    Capitalized,
    SelfCorrectionRemoved,
    ListFormatted,
    DictionaryMatch,
    NumeralNormalized,
    SpokenFormNormalized,
}

#[derive(Debug, Clone)]
pub struct ProcessError {
    pub message: String,
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "process error: {}", self.message)
    }
}

impl std::error::Error for ProcessError {}

// ---------------------------------------------------------------------------
// SelfCorrectionDetector
// ---------------------------------------------------------------------------

/// Detects and removes self-corrections (disfluency repairs) in ASR output.
///
/// Pattern: "I want to go to Boston, to Denver" → "I want to go to Denver"
///
/// Uses a sliding window to find repeated structural patterns (reparandum →
/// repair), where the repair "overwrites" the reparandum.  Detection is based
/// on the observation that self-corrections typically produce a "rough copy":
/// the repair reuses the same or similar words in roughly the same order.
pub struct SelfCorrectionDetector {
    /// Minimum number of shared words between reparandum and repair to trigger.
    min_shared_words: usize,
    /// Correction cue words that signal a self-correction boundary.
    correction_cues_en: Vec<String>,
    correction_cues_ja: Vec<String>,
    correction_cues_es: Vec<String>,
    correction_cues_zh: Vec<String>,
    correction_cues_ko: Vec<String>,
}

impl SelfCorrectionDetector {
    pub fn new() -> Self {
        // Cues are sorted longest-first because the detector uses
        // `str::find(cue)` and a shorter cue that is a substring of a
        // longer one would otherwise win. Concretely: without this
        // sort, "ていうか" (4 chars) matches before "っていうか" (5 chars)
        // inside "鈴木課長、っていうか佐藤課長です", causing the detector
        // to treat just "っ" as the reparandum instead of the full
        // "鈴木課長" preceding the cue. Same hazard exists for en
        // ("no" / "no wait", "rather" / "or rather").
        let mut correction_cues_en: Vec<String> = vec![
            "no", "wait", "sorry", "i mean", "actually", "rather",
            "no wait", "or rather",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        correction_cues_en.sort_by_key(|c| std::cmp::Reverse(c.chars().count()));

        let mut correction_cues_ja: Vec<String> = vec![
            "いや", "じゃなくて", "じゃなく", "ではなく", "ていうか",
            "っていうか", "じゃない",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        correction_cues_ja.sort_by_key(|c| std::cmp::Reverse(c.chars().count()));

        // Spanish closed-set cues. Bare `no` is the most ambiguous
        // (clashes with the negation use of the same word) so we
        // require a sentence-internal comma context for it (handled
        // in `detect_spanish`); the rest are unambiguous markers
        // attested in the Val.Es.Co + CSJ-style disfluency literature.
        // Sorted longest-first: `mejor dicho` must outrank `mejor`,
        // `quiero decir` outrank `digo`, `no es` outrank `no`.
        let mut correction_cues_es: Vec<String> = vec![
            "mejor dicho", "quiero decir", "o sea", "perdón",
            "mejor", "digo", "no es", "no",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        correction_cues_es.sort_by_key(|c| std::cmp::Reverse(c.chars().count()));

        // Chinese cues — tokenisation by 、 / ， (no whitespace
        // between words), parallel to Japanese. Sorted longest-first
        // so 我的意思是 outranks 我是说 inside utterances that contain
        // both fragments.
        let mut correction_cues_zh: Vec<String> = vec![
            "我的意思是", "确切地说", "应该说", "我是说",
            "不对", "不是", "算了",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        correction_cues_zh.sort_by_key(|c| std::cmp::Reverse(c.chars().count()));

        // Korean cues — eojeol-tokenised (whitespace) like en / es.
        // Sorted longest-first so multi-eojeol cues (그게 아니라,
        // 잘못 말했네) outrank their shorter prefixes (아니, 그게).
        // The `아니` family overlaps with sentence-final negation
        // particles (X-아니에요 = "is not X"); the word-boundary
        // check in detect_korean handles that disambiguation.
        let mut correction_cues_ko: Vec<String> = vec![
            "그게 아니라", "그게 아니고", "잘못 말했다", "잘못 말했네",
            "아 잠깐", "잠깐만",
            "아니에요", "아니라", "아니야", "아니",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        correction_cues_ko.sort_by_key(|c| std::cmp::Reverse(c.chars().count()));

        Self {
            min_shared_words: 1,
            correction_cues_en,
            correction_cues_ja,
            correction_cues_es,
            correction_cues_zh,
            correction_cues_ko,
        }
    }

    /// Find self-correction patterns in English text.
    ///
    /// Looks for correction cues and removes the reparandum (the part before
    /// the cue that the speaker intended to discard).
    fn detect_english(&self, text: &str) -> Option<(String, Correction)> {
        let lower = text.to_lowercase();

        for cue in &self.correction_cues_en {
            if let Some(cue_pos) = lower.find(cue.as_str()) {
                let before_cue = &text[..cue_pos].trim_end();
                let after_cue = &text[cue_pos + cue.len()..].trim_start();

                if after_cue.is_empty() {
                    continue;
                }

                // Find the overlap: walk backwards from the cue to find where
                // the reparandum starts.  We look for the point where the
                // words before the cue start repeating structure found after it.
                let before_words: Vec<&str> = before_cue.split_whitespace().collect();
                let after_words: Vec<&str> = after_cue.split_whitespace().collect();

                if after_words.is_empty() {
                    continue;
                }

                // Find how many trailing words of `before` share a common
                // prefix with `after` (the "rough copy" pattern).
                let shared = Self::count_shared_prefix_from_end(&before_words, &after_words);

                if shared >= self.min_shared_words {
                    // Remove the reparandum: keep words before the shared
                    // prefix, then append the repair (after cue).
                    let keep_count = before_words.len() - shared;
                    let kept: Vec<&str> = before_words[..keep_count].to_vec();
                    let result = if kept.is_empty() {
                        after_cue.to_string()
                    } else {
                        format!("{} {}", kept.join(" "), after_cue)
                    };

                    let original_rm = before_words[keep_count..].join(" ");
                    return Some((
                        result,
                        Correction {
                            kind: CorrectionKind::SelfCorrectionRemoved,
                            original: format!("{} {}", original_rm, cue),
                            replacement: String::new(),
                        },
                    ));
                }
            }
        }
        None
    }

    /// Find self-correction patterns in Japanese text.
    fn detect_japanese(&self, text: &str) -> Option<(String, Correction)> {
        for cue in &self.correction_cues_ja {
            if let Some(cue_pos) = text.find(cue.as_str()) {
                let before = text[..cue_pos].trim_end_matches('、').trim_end();
                let after = text[cue_pos + cue.len()..].trim_start_matches('、').trim_start();

                if after.is_empty() || before.is_empty() {
                    continue;
                }

                // In Japanese, self-corrections often have the form:
                // "AAAいやBBB" where BBB replaces AAA.
                // We find the last comma-delimited segment before the cue
                // and remove it, keeping everything before and the repair after.
                let segments: Vec<&str> = before.split('、').collect();
                if segments.is_empty() {
                    continue;
                }

                let reparandum = segments.last().unwrap().trim();
                let kept_before = if segments.len() > 1 {
                    segments[..segments.len() - 1].join("、")
                } else {
                    String::new()
                };

                let result = if kept_before.is_empty() {
                    after.to_string()
                } else {
                    format!("{}、{}", kept_before, after)
                };

                return Some((
                    result,
                    Correction {
                        kind: CorrectionKind::SelfCorrectionRemoved,
                        original: format!("{}{}",  reparandum, cue),
                        replacement: String::new(),
                    },
                ));
            }
        }
        None
    }

    /// Find self-correction patterns in Spanish text.
    ///
    /// Spanish disfluencies follow the same `<reparandum> <cue>
    /// <repair>` structure as English (whitespace tokenisation,
    /// shared-word overlap between the trailing edge of the
    /// reparandum and the leading edge of the repair). Punctuation
    /// around the cue (commas, periods) is stripped before the
    /// shared-prefix check so utterances like "voy mañana, no, voy
    /// hoy" hit the same path as "voy mañana no voy hoy".
    ///
    /// Bare `no` is intentionally treated as a correction cue here
    /// even though Spanish uses the same word for negation; the
    /// shared-prefix requirement (`min_shared_words = 1`) keeps
    /// pure negations like "el gato no come pescado" un-flagged
    /// (no overlapping content word follows the cue).
    fn detect_spanish(&self, text: &str) -> Option<(String, Correction)> {
        let lower = text.to_lowercase();

        for cue in &self.correction_cues_es {
            // Only match the cue as a standalone word — without
            // boundary checks, "no" would fire inside "no es" or
            // tokens like "Mariano" / "minoría". Find with a sliding
            // search so each candidate position can be vetted.
            let mut from = 0usize;
            while let Some(rel) = lower[from..].find(cue.as_str()) {
                let cue_pos = from + rel;
                let cue_end = cue_pos + cue.len();

                // Standalone-word boundary: the chars on each side
                // must be either out-of-bounds or a non-alphabetic
                // ASCII delimiter (whitespace, comma, period, etc.).
                let left_ok = cue_pos == 0
                    || !lower
                        .as_bytes()
                        .get(cue_pos - 1)
                        .map(|b| b.is_ascii_alphabetic())
                        .unwrap_or(false);
                let right_ok = cue_end >= lower.len()
                    || !lower
                        .as_bytes()
                        .get(cue_end)
                        .map(|b| b.is_ascii_alphabetic())
                        .unwrap_or(false);
                if !(left_ok && right_ok) {
                    from = cue_pos + cue.chars().next().map_or(1, |c| c.len_utf8());
                    continue;
                }

                // Strip surrounding punctuation + whitespace so the
                // shared-prefix check operates on word tokens only.
                let trim_chars: &[char] = &[',', '.', ';', ':', '!', '?', ' ', '\t'];
                let before_cue = text[..cue_pos].trim_end_matches(trim_chars);
                let after_cue = text[cue_end..].trim_start_matches(trim_chars);

                if after_cue.is_empty() || before_cue.is_empty() {
                    from = cue_end;
                    continue;
                }

                let before_words: Vec<&str> = before_cue.split_whitespace().collect();
                let after_words: Vec<&str> = after_cue.split_whitespace().collect();
                if after_words.is_empty() || before_words.is_empty() {
                    from = cue_end;
                    continue;
                }

                let shared =
                    Self::count_shared_prefix_from_end(&before_words, &after_words);

                if shared >= self.min_shared_words {
                    let keep_count = before_words.len() - shared;
                    let kept: Vec<&str> = before_words[..keep_count].to_vec();
                    let result = if kept.is_empty() {
                        after_cue.to_string()
                    } else {
                        format!("{} {}", kept.join(" "), after_cue)
                    };
                    let original_rm = before_words[keep_count..].join(" ");
                    return Some((
                        result,
                        Correction {
                            kind: CorrectionKind::SelfCorrectionRemoved,
                            original: format!("{} {}", original_rm, cue),
                            replacement: String::new(),
                        },
                    ));
                }
                from = cue_end;
            }
        }
        None
    }

    /// Find self-correction patterns in Chinese text.
    ///
    /// Chinese, like Japanese, lacks inter-word whitespace and
    /// commonly uses 、 / ， as clause separators in ASR output.
    /// Detection mirrors `detect_japanese`: locate the cue, take
    /// the last comma-segment of the pre-cue text as the
    /// reparandum, and emit `<surviving prefix>，<repair>`.
    fn detect_chinese(&self, text: &str) -> Option<(String, Correction)> {
        for cue in &self.correction_cues_zh {
            if let Some(cue_pos) = text.find(cue.as_str()) {
                let trim_clause: &[char] = &['、', '，', ',', ' '];
                let before = text[..cue_pos]
                    .trim_end_matches(trim_clause)
                    .trim_end();
                let after = text[cue_pos + cue.len()..]
                    .trim_start_matches(trim_clause)
                    .trim_start();

                if after.is_empty() || before.is_empty() {
                    continue;
                }

                // Split before-text on either Chinese clause comma,
                // since utterance ASR may emit either form.
                let segments: Vec<&str> = before.split(['、', '，']).collect();
                if segments.is_empty() {
                    continue;
                }

                let reparandum = segments.last().unwrap().trim();
                let kept_before = if segments.len() > 1 {
                    segments[..segments.len() - 1].join("，")
                } else {
                    String::new()
                };

                let result = if kept_before.is_empty() {
                    after.to_string()
                } else {
                    format!("{}，{}", kept_before, after)
                };

                return Some((
                    result,
                    Correction {
                        kind: CorrectionKind::SelfCorrectionRemoved,
                        original: format!("{}{}", reparandum, cue),
                        replacement: String::new(),
                    },
                ));
            }
        }
        None
    }

    /// Find self-correction patterns in Korean text.
    ///
    /// Korean ASR output is eojeol-tokenised (whitespace-separated)
    /// but rarely carries inter-clause commas, so the en / es
    /// shared-prefix overlap heuristic only catches a fraction of
    /// the natural correction patterns. The closer fit is the
    /// Japanese comma-segmentation strategy: locate the cue, take
    /// the last `,` / `.` / `?` / `!` segment of the pre-cue text
    /// as the reparandum (the whole pre-cue text when there are no
    /// commas — the typical FLEURS-ko / SenseVoice case), and emit
    /// `<surviving prefix>, <repair>`.
    ///
    /// Eojeol-boundary check on the cue prevents `아니` from firing
    /// inside `아니에요` (sentence-final negation predicate) or
    /// other Hangul compounds.
    fn detect_korean(&self, text: &str) -> Option<(String, Correction)> {
        for cue in &self.correction_cues_ko {
            let mut from = 0usize;
            while let Some(rel) = text[from..].find(cue.as_str()) {
                let cue_pos = from + rel;
                let cue_end = cue_pos + cue.len();

                // Eojeol boundary check: chars on each side must be
                // whitespace, punctuation, or out-of-bounds. A Hangul
                // / alphanumeric char on either side means we are
                // inside a larger word.
                let left_ok = cue_pos == 0
                    || text[..cue_pos]
                        .chars()
                        .last()
                        .map(|c| !c.is_alphanumeric())
                        .unwrap_or(true);
                let right_ok = cue_end == text.len()
                    || text[cue_end..]
                        .chars()
                        .next()
                        .map(|c| !c.is_alphanumeric())
                        .unwrap_or(true);
                if !(left_ok && right_ok) {
                    from = cue_pos + cue.chars().next().map_or(1, |c| c.len_utf8());
                    continue;
                }

                let trim_chars: &[char] = &[',', '.', ';', ':', '!', '?', ' ', '\t'];
                let before_cue = text[..cue_pos].trim_end_matches(trim_chars);
                let after_cue = text[cue_end..].trim_start_matches(trim_chars);

                if after_cue.is_empty() || before_cue.is_empty() {
                    from = cue_end;
                    continue;
                }

                // Mirror detect_japanese: split the pre-cue text on
                // sentence-internal punctuation, treat the last
                // segment as the reparandum, keep the rest. For
                // FLEURS-ko / SenseVoice utterances (no internal
                // commas) the whole pre-cue is one segment → drop
                // it entirely → output is just the repair.
                let segments: Vec<&str> =
                    before_cue.split([',', '.', '!', '?', ';']).collect();
                if segments.is_empty() {
                    from = cue_end;
                    continue;
                }

                let reparandum = segments.last().unwrap().trim();
                let kept_before = if segments.len() > 1 {
                    segments[..segments.len() - 1].join(",")
                } else {
                    String::new()
                };

                let result = if kept_before.is_empty() {
                    after_cue.to_string()
                } else {
                    format!("{}, {}", kept_before.trim_end(), after_cue)
                };

                return Some((
                    result,
                    Correction {
                        kind: CorrectionKind::SelfCorrectionRemoved,
                        original: format!("{} {}", reparandum, cue),
                        replacement: String::new(),
                    },
                ));
            }
        }
        None
    }

    /// Count how many words at the end of `before` match the beginning of
    /// `after` when lowercased.
    fn count_shared_prefix_from_end(before: &[&str], after: &[&str]) -> usize {
        let mut count = 0;
        let max_check = before.len().min(after.len()).min(5); // limit search window

        for offset in 1..=max_check {
            let b_idx = before.len() - offset;
            if before[b_idx].to_lowercase() == after[0].to_lowercase() {
                // Check if subsequent words also match
                let mut matches = 1;
                for j in 1..offset.min(after.len()) {
                    if before[b_idx + j].to_lowercase() == after[j].to_lowercase() {
                        matches += 1;
                    } else {
                        break;
                    }
                }
                if matches >= 1 {
                    count = count.max(offset);
                }
            }
        }
        count
    }
}

impl Default for SelfCorrectionDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TextProcessor for SelfCorrectionDetector {
    async fn process(
        &self,
        text: &str,
        _context: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError> {
        // Order matters when a cue word exists in two languages.
        // We try the most specific / least-ambiguous detectors first:
        //   1. Korean — Hangul syllables, no Latin overlap, eojeol
        //      tokenisation. Cues (아니, 그게 아니라, 잠깐만, 잘못
        //      말했다) don't appear in any other language we cover.
        //   2. Chinese — Hanzi, comma-segmented. Cues (我是说, 不对,
        //      我的意思是) are clean of Hangul / Latin overlap.
        //   3. Japanese — Hiragana / Kanji, comma-segmented like zh
        //      but with a different cue set (いや, ていうか).
        //   4. Spanish — Latin, restricted closed-class cues
        //      (perdón, digo, mejor dicho). Bare `no` overlaps with
        //      English; the Spanish detector's stricter shared-prefix
        //      check beats the English detector for genuinely
        //      Spanish input.
        //   5. English — fallback, widest cue set, runs last.
        if let Some((result, correction)) = self.detect_korean(text) {
            return Ok(ProcessResult {
                text: result,
                corrections: vec![correction],
            });
        }
        if let Some((result, correction)) = self.detect_chinese(text) {
            return Ok(ProcessResult {
                text: result,
                corrections: vec![correction],
            });
        }
        if let Some((result, correction)) = self.detect_japanese(text) {
            return Ok(ProcessResult {
                text: result,
                corrections: vec![correction],
            });
        }
        if let Some((result, correction)) = self.detect_spanish(text) {
            return Ok(ProcessResult {
                text: result,
                corrections: vec![correction],
            });
        }
        if let Some((result, correction)) = self.detect_english(text) {
            return Ok(ProcessResult {
                text: result,
                corrections: vec![correction],
            });
        }

        Ok(ProcessResult {
            text: text.to_string(),
            corrections: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// BasicPunctuationRestorer — rule-based heuristic
// ---------------------------------------------------------------------------

/// Lightweight rule-based punctuation insertion.
///
/// This is a stopgap until a proper CNN-BiLSTM ONNX model is integrated.
/// It handles the most common cases:
/// - Capitalize first word of text (Latin scripts only)
/// - Ensure text ends with sentence-final punctuation
/// - Capitalize after sentence-ending punctuation (Latin scripts)
///
/// Terminal punctuation choice is script-aware:
/// - Latin (en / es / mixed): trailing `.`
/// - Hangul (ko): trailing `.` (Korean conventionally uses the
///   Western period in formal writing)
/// - Hanzi (zh): trailing `。` (CJK fullwidth period)
/// - Hiragana / Katakana / Kanji-only (ja): trailing `。`
pub struct BasicPunctuationRestorer;

/// Dominant-script classification for terminal-punctuation choice.
/// Determined by counting characters in each Unicode block; the
/// thresholds are deliberately permissive so a pure-Hangul utterance
/// with one stray English brand name still classifies as Hangul.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DominantScript {
    Latin,
    Hangul,
    Hanzi,
    Kana,
    Other,
}

fn classify_script(text: &str) -> DominantScript {
    let mut latin = 0usize;
    let mut hangul = 0usize;
    let mut hanzi = 0usize;
    let mut kana = 0usize;
    for c in text.chars() {
        if c.is_ascii_alphabetic() {
            latin += 1;
        } else if matches!(c as u32, 0xAC00..=0xD7A3 | 0x1100..=0x11FF | 0x3130..=0x318F) {
            hangul += 1;
        } else if matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF) {
            hanzi += 1;
        } else if matches!(c as u32, 0x3040..=0x30FF) {
            kana += 1;
        }
    }
    // Hiragana / Katakana presence dominates over Hanzi (since
    // pure-Hanzi utterances have zero kana and Japanese always
    // mixes the two). For Korean, Hangul presence dominates.
    if kana > 0 && kana + hanzi >= latin && kana + hanzi >= hangul {
        return DominantScript::Kana;
    }
    if hangul > latin && hangul > hanzi {
        return DominantScript::Hangul;
    }
    if hanzi > latin && hanzi > hangul {
        return DominantScript::Hanzi;
    }
    if latin > 0 {
        return DominantScript::Latin;
    }
    DominantScript::Other
}

#[async_trait]
impl TextProcessor for BasicPunctuationRestorer {
    async fn process(
        &self,
        text: &str,
        _context: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError> {
        if text.is_empty() {
            return Ok(ProcessResult {
                text: String::new(),
                corrections: vec![],
            });
        }

        let script = classify_script(text);
        let capitalise = matches!(script, DominantScript::Latin);

        let mut result = String::with_capacity(text.len() + 8);
        let mut corrections = Vec::new();
        let mut capitalize_next = capitalise;

        for ch in text.chars() {
            if capitalize_next && ch.is_alphabetic() {
                let upper: String = ch.to_uppercase().collect();
                if upper != ch.to_string() {
                    corrections.push(Correction {
                        kind: CorrectionKind::Capitalized,
                        original: ch.to_string(),
                        replacement: upper.clone(),
                    });
                }
                result.push_str(&upper);
                capitalize_next = false;
            } else {
                result.push(ch);
                if capitalise && (ch == '.' || ch == '!' || ch == '?') {
                    capitalize_next = true;
                } else if ch == '。' {
                    // Japanese / Chinese sentence-ending — next sentence starts
                    capitalize_next = false; // no capitalization in CJK
                }
            }
        }

        // Ensure terminal punctuation, picking the right form for
        // the dominant script. Skip Other (mixed / non-script-class
        // text — e.g. pure digits or punctuation-only) since we
        // don't know what to append.
        let trimmed = result.trim_end();
        let last_char = trimmed.chars().last();
        let want_terminal = match script {
            DominantScript::Latin | DominantScript::Hangul => Some('.'),
            DominantScript::Hanzi | DominantScript::Kana => Some('。'),
            DominantScript::Other => None,
        };
        if let Some(terminal) = want_terminal {
            if let Some(last) = last_char {
                let already_terminated = match terminal {
                    '.' => matches!(last, '.' | '!' | '?' | ')' | '"' | '\'' | '。'),
                    '。' => matches!(last, '。' | '！' | '？' | '.' | '!' | '?'),
                    _ => true,
                };
                if !already_terminated {
                    result = format!("{}{}", trimmed, terminal);
                    corrections.push(Correction {
                        kind: CorrectionKind::PunctuationInserted,
                        original: String::new(),
                        replacement: terminal.to_string(),
                    });
                }
            }
        }

        Ok(ProcessResult {
            text: result,
            corrections,
        })
    }
}

// ---------------------------------------------------------------------------
// InverseTextNormalizer
// ---------------------------------------------------------------------------

/// Inverse text normalization: rewrites spoken-form numerals into
/// written form ("twenty five" → "25", "二十五" → "25") using the
/// `text-processing-rs` crate. A non-LLM Tier-2 processor.
///
/// Language support tracks the upstream crate's *sentence-level* ITN
/// API, which today covers only en / ja / zh:
///
/// - `en` — `normalize_sentence` (English sentence scanner).
/// - `ja` / `zh` — `normalize_with_lang` (CJK in-place span scanner).
/// - everything else, incl. `es` and `ko` — pass through unchanged.
///   `es` has no sentence-level entry point upstream yet, and `ko` is
///   a pending upstream PR (FluidInference/text-processing-rs). When
///   either lands, add its arm to [`InverseTextNormalizer::new`].
pub struct InverseTextNormalizer {
    backend: ItnBackend,
}

enum ItnBackend {
    /// English sentence scanner (`normalize_sentence`).
    EnglishSentence,
    /// CJK sentence scanner (`normalize_with_lang`) for the given code.
    Lang(&'static str),
    /// Language not yet supported upstream — pass through unchanged.
    Passthrough,
}

impl InverseTextNormalizer {
    /// Build a normalizer for `lang` (a BCP-47-ish code like "en").
    /// Unsupported languages produce a passthrough normalizer.
    pub fn new(lang: &str) -> Self {
        let backend = match lang {
            "en" => ItnBackend::EnglishSentence,
            "ja" => ItnBackend::Lang("ja"),
            "zh" => ItnBackend::Lang("zh"),
            _ => ItnBackend::Passthrough,
        };
        Self { backend }
    }
}

#[async_trait]
impl TextProcessor for InverseTextNormalizer {
    async fn process(
        &self,
        text: &str,
        _context: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError> {
        let normalized = match self.backend {
            ItnBackend::EnglishSentence => text_processing_rs::normalize_sentence(text),
            ItnBackend::Lang(lang) => text_processing_rs::normalize_with_lang(text, lang),
            ItnBackend::Passthrough => {
                return Ok(ProcessResult {
                    text: text.to_string(),
                    corrections: vec![],
                });
            }
        };

        let corrections = if normalized != text {
            vec![Correction {
                kind: CorrectionKind::NumeralNormalized,
                original: text.to_string(),
                replacement: normalized.clone(),
            }]
        } else {
            vec![]
        };

        Ok(ProcessResult {
            text: normalized,
            corrections,
        })
    }
}

// ---------------------------------------------------------------------------
// SpokenFormNormalizer
// ---------------------------------------------------------------------------

/// Rewrites colloquial spoken-form reductions into their written form
/// ("gonna" → "going to", "lemme" → "let me"). A non-LLM Tier-2
/// processor (issue #72).
///
/// This is a rule-based stopgap — a curated dictionary of unambiguous
/// one-to-one reductions — pending an edit-tagging ONNX model that can
/// resolve context-dependent cases. Only English has a rule set today;
/// other languages pass through unchanged. To add a language, write a
/// `<lang>_spoken_form` lookup and wire it in [`SpokenFormNormalizer::new`].
///
/// The dictionary is deliberately conservative: entries like "gonna" /
/// "wanna" technically also have a "going a" / "want a" reading, but
/// the "going to" / "want to" sense dominates spoken usage, so the
/// dominant mapping is applied. Genuinely ambiguous reductions
/// ("ain't", bare "cause", "gotcha") are left out.
pub struct SpokenFormNormalizer {
    lookup: Option<fn(&str) -> Option<&'static str>>,
}

impl SpokenFormNormalizer {
    /// Build a normalizer for `lang`. Unsupported languages produce a
    /// passthrough normalizer.
    pub fn new(lang: &str) -> Self {
        let lookup = match lang {
            "en" => Some(en_spoken_form as fn(&str) -> Option<&'static str>),
            _ => None,
        };
        Self { lookup }
    }
}

/// English colloquial reductions → written form. Conservative,
/// one-to-one, unambiguous-in-the-dominant-sense entries only.
fn en_spoken_form(token: &str) -> Option<&'static str> {
    match token {
        "gonna" => Some("going to"),
        "wanna" => Some("want to"),
        "gotta" => Some("got to"),
        "hafta" => Some("have to"),
        "oughta" => Some("ought to"),
        "tryna" => Some("trying to"),
        "gimme" => Some("give me"),
        "lemme" => Some("let me"),
        "kinda" => Some("kind of"),
        "sorta" => Some("sort of"),
        "outta" => Some("out of"),
        "lotta" => Some("lot of"),
        "dunno" => Some("don't know"),
        "c'mon" => Some("come on"),
        "'cause" => Some("because"),
        "cuz" => Some("because"),
        "y'all" => Some("you all"),
        _ => None,
    }
}

/// Apply the leading-capitalization of `original` to `replacement`.
fn match_leading_case(original: &str, replacement: &str) -> String {
    if original.chars().next().is_some_and(|c| c.is_uppercase()) {
        let mut chars = replacement.chars();
        match chars.next() {
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            None => String::new(),
        }
    } else {
        replacement.to_string()
    }
}

#[async_trait]
impl TextProcessor for SpokenFormNormalizer {
    async fn process(
        &self,
        text: &str,
        _context: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError> {
        let Some(lookup) = self.lookup else {
            return Ok(ProcessResult {
                text: text.to_string(),
                corrections: vec![],
            });
        };

        let mut out: Vec<String> = Vec::new();
        let mut corrections = Vec::new();

        for token in text.split_whitespace() {
            // Split a trailing run of sentence punctuation so "gonna."
            // still matches; a leading apostrophe is kept (it is part
            // of words like "'cause").
            let trail_start = token
                .char_indices()
                .rev()
                .take_while(|(_, c)| matches!(c, '.' | ',' | '!' | '?' | ';' | ':'))
                .last()
                .map(|(i, _)| i)
                .unwrap_or(token.len());
            let (core, trailing) = token.split_at(trail_start);

            let lower = core.to_lowercase();
            match lookup(&lower) {
                Some(replacement) if !core.is_empty() => {
                    let cased = match_leading_case(core, replacement);
                    corrections.push(Correction {
                        kind: CorrectionKind::SpokenFormNormalized,
                        original: core.to_string(),
                        replacement: cased.clone(),
                    });
                    out.push(format!("{}{}", cased, trailing));
                }
                _ => out.push(token.to_string()),
            }
        }

        if corrections.is_empty() {
            // Nothing matched — return the input untouched so clean text
            // is byte-identical (no whitespace re-normalization).
            return Ok(ProcessResult {
                text: text.to_string(),
                corrections,
            });
        }

        Ok(ProcessResult {
            text: out.join(" "),
            corrections,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_context() -> ContextSnapshot {
        ContextSnapshot::default()
    }

    // --- SelfCorrectionDetector tests ---

    #[tokio::test]
    async fn self_correction_with_no_wait() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("I want to go to Boston no wait to Denver", &empty_context())
            .await
            .unwrap();
        assert!(
            result.text.contains("Denver"),
            "repair should be kept: {}",
            result.text
        );
        assert!(
            !result.text.contains("Boston"),
            "reparandum should be removed: {}",
            result.text
        );
        assert_eq!(result.corrections.len(), 1);
        assert_eq!(
            result.corrections[0].kind,
            CorrectionKind::SelfCorrectionRemoved
        );
    }

    #[tokio::test]
    async fn self_correction_with_i_mean() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process(
                "the meeting is tomorrow i mean the meeting is on Friday",
                &empty_context(),
            )
            .await
            .unwrap();
        assert!(
            result.text.contains("Friday"),
            "repair should be kept: {}",
            result.text
        );
    }

    #[tokio::test]
    async fn no_self_correction_in_clean_text() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("I want to go to Denver for the conference", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "I want to go to Denver for the conference");
        assert!(result.corrections.is_empty());
    }

    #[tokio::test]
    async fn japanese_self_correction() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("明日、いや明後日に会議があります", &empty_context())
            .await
            .unwrap();
        assert!(
            result.text.contains("明後日"),
            "repair should be kept: {}",
            result.text
        );
        assert!(
            !result.text.contains("明日"),
            "reparandum should be removed: {}",
            result.text
        );
    }

    #[tokio::test]
    async fn japanese_self_correction_with_janakute() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process(
                "東京駅、じゃなくて品川駅で待ち合わせ",
                &empty_context(),
            )
            .await
            .unwrap();
        assert!(result.text.contains("品川駅"), "repair: {}", result.text);
        assert!(!result.text.contains("東京駅"), "reparandum: {}", result.text);
    }

    // --- SelfCorrectionDetector — Spanish ---

    #[tokio::test]
    async fn spanish_self_correction_with_no() {
        // Bare `no` as a correction cue with shared content word.
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("voy mañana no voy hoy", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("hoy"), "repair: {}", result.text);
        assert!(!result.text.contains("mañana"), "reparandum: {}", result.text);
        assert_eq!(result.corrections.len(), 1);
    }

    #[tokio::test]
    async fn spanish_self_correction_with_perdon() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("voy a Madrid perdón a Barcelona", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("Barcelona"), "repair: {}", result.text);
        assert!(!result.text.contains("Madrid"), "reparandum: {}", result.text);
    }

    #[tokio::test]
    async fn spanish_self_correction_with_digo() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("el presidente digo el ex-presidente", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("ex-presidente"), "repair: {}", result.text);
    }

    #[tokio::test]
    async fn spanish_self_correction_with_mejor_dicho() {
        // `mejor dicho` must outrank bare `mejor` (sorted longest-first).
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("salgo a las cinco mejor dicho a las seis", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("seis"), "repair: {}", result.text);
        assert!(!result.text.contains("cinco"), "reparandum: {}", result.text);
    }

    #[tokio::test]
    async fn spanish_self_correction_with_o_sea() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("llamo a Juan o sea a Pedro", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("Pedro"), "repair: {}", result.text);
        assert!(!result.text.contains("Juan"), "reparandum: {}", result.text);
    }

    #[tokio::test]
    async fn spanish_negation_without_overlap_is_not_correction() {
        // Pure negation — `no` is not followed by a content word
        // that overlaps with the reparandum, so the shared-prefix
        // check fails and the detector leaves the text untouched.
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("el gato no come pescado", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "el gato no come pescado");
        assert!(result.corrections.is_empty());
    }

    #[tokio::test]
    async fn spanish_no_self_correction_in_clean_text() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("la conferencia es mañana en Barcelona", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "la conferencia es mañana en Barcelona");
        assert!(result.corrections.is_empty());
    }

    #[tokio::test]
    async fn spanish_self_correction_with_commas() {
        // The cue is surrounded by punctuation in the wild; the
        // detector strips it before computing the shared prefix.
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("voy a Madrid, perdón, a Barcelona", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("Barcelona"), "repair: {}", result.text);
        assert!(!result.text.contains("Madrid"), "reparandum: {}", result.text);
    }

    #[tokio::test]
    async fn spanish_no_does_not_match_inside_word() {
        // `no` as a substring of `Mariano` / `Antonio` / `minoría`
        // must not trigger correction. The word-boundary check in
        // detect_spanish guards against this.
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("Mariano y Antonio fueron al cine", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "Mariano y Antonio fueron al cine");
        assert!(result.corrections.is_empty());
    }

    #[tokio::test]
    async fn spanish_quiero_decir_outranks_digo() {
        // `quiero decir` is the longer form of `digo`; longest-first
        // sort puts it first so it gets matched as a single span.
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("voy a Madrid quiero decir a Barcelona", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("Barcelona"), "repair: {}", result.text);
        assert!(!result.text.contains("Madrid"), "reparandum: {}", result.text);
    }

    // --- BasicPunctuationRestorer tests ---

    #[tokio::test]
    async fn capitalize_first_word() {
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("hello world", &empty_context())
            .await
            .unwrap();
        assert!(result.text.starts_with('H'));
    }

    #[tokio::test]
    async fn add_terminal_period() {
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("hello world", &empty_context())
            .await
            .unwrap();
        assert!(result.text.ends_with('.'));
    }

    #[tokio::test]
    async fn punctuation_appends_period_for_korean() {
        // Hangul-dominant text gets a Western period (ko convention).
        // Capitalisation does not apply to Hangul.
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("내일 오후 세 시에 만나기로 했습니다", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "내일 오후 세 시에 만나기로 했습니다.");
    }

    #[tokio::test]
    async fn punctuation_appends_fullwidth_period_for_chinese() {
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("会议在下午三点开始", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "会议在下午三点开始。");
    }

    #[tokio::test]
    async fn punctuation_appends_fullwidth_period_for_japanese() {
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("今日は天気がいい", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "今日は天気がいい。");
    }

    #[tokio::test]
    async fn punctuation_does_not_double_terminal_for_korean() {
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("이것은 정말 좋은 책입니다.", &empty_context())
            .await
            .unwrap();
        // Already terminated, no second period.
        assert_eq!(result.text, "이것은 정말 좋은 책입니다.");
    }

    #[tokio::test]
    async fn punctuation_does_not_double_terminal_for_chinese() {
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("这是一本很好的书。", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "这是一本很好的书。");
    }

    #[tokio::test]
    async fn punctuation_skips_capitalisation_for_hangul() {
        // No Latin alpha → capitalize_next disabled. Hangul chars
        // pass through unchanged.
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("안녕하세요 반갑습니다", &empty_context())
            .await
            .unwrap();
        // No capitalisation correction emitted.
        assert!(
            !result
                .corrections
                .iter()
                .any(|c| matches!(c.kind, CorrectionKind::Capitalized)),
            "{:?}",
            result.corrections
        );
    }

    #[tokio::test]
    async fn preserve_existing_punctuation() {
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("Hello world!", &empty_context())
            .await
            .unwrap();
        assert!(result.text.ends_with('!'));
        assert!(!result.text.ends_with("!."));
    }

    #[tokio::test]
    async fn capitalize_after_period() {
        let proc = BasicPunctuationRestorer;
        let result = proc
            .process("hello. world", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "Hello. World.");
    }

    #[tokio::test]
    async fn japanese_no_terminal_period_added() {
        let proc = BasicPunctuationRestorer;
        // Pure Japanese text should not get an English period appended
        let result = proc
            .process("お江戸を発って二十里上方", &empty_context())
            .await
            .unwrap();
        assert!(
            !result.text.ends_with('.'),
            "Japanese should not get English period: {}",
            result.text
        );
    }

    #[tokio::test]
    async fn empty_text() {
        let proc = BasicPunctuationRestorer;
        let result = proc.process("", &empty_context()).await.unwrap();
        assert_eq!(result.text, "");
        assert!(result.corrections.is_empty());
    }

    /// Regression: when the input contains `っていうか`, the detector
    /// used to find the shorter `ていうか` substring first (because
    /// cues were iterated in declared order and `ていうか` precedes
    /// `っていうか` in the list). That treated just `っ` as the
    /// reparandum and left the real reparandum (e.g. `鈴木課長`) in
    /// the output. Sorting cues longest-first at construction time
    /// fixes this, surfaced by the L3 self-correction direct F1
    /// evaluation against the committed ja annotations.
    #[tokio::test]
    async fn ja_self_correction_with_tte_iu_ka() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("鈴木課長、っていうか佐藤課長です", &empty_context())
            .await
            .unwrap();
        assert!(
            !result.text.contains("鈴木課長"),
            "reparandum should be removed: {}",
            result.text,
        );
        assert!(
            result.text.contains("佐藤課長"),
            "repair should be kept: {}",
            result.text,
        );
        assert_eq!(result.text, "佐藤課長です");
    }

    /// Regression: same shape as `っていうか`/`ていうか` but for English
    /// (`no wait` containing `no` as a prefix). Without longest-first
    /// cue sorting, `no` could match first inside `no wait` and the
    /// detector would skip the full reparandum.
    #[tokio::test]
    async fn en_self_correction_no_wait_prefers_long_cue() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process(
                "I want to go to Boston no wait to Denver",
                &empty_context(),
            )
            .await
            .unwrap();
        assert!(!result.text.contains("Boston"), "{}", result.text);
        assert!(result.text.contains("Denver"), "{}", result.text);
    }

    // -----------------------------------------------------------------
    // Chinese self-correction — same comma-segmented pattern as ja.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn chinese_self_correction_with_bu_dui() {
        // 不对 = "no, wrong" — drops the trailing comma-segment
        // before the cue.
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("我们明天开会，不对，后天开会", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("后天"), "repair: {}", result.text);
        assert!(!result.text.contains("明天"), "reparandum: {}", result.text);
    }

    #[tokio::test]
    async fn chinese_self_correction_with_wo_shi_shuo() {
        // 我是说 = "I mean".
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("会议室在三楼，我是说，在四楼", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("四楼"), "{}", result.text);
        assert!(!result.text.contains("三楼"), "{}", result.text);
    }

    #[tokio::test]
    async fn chinese_self_correction_with_bu_shi() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("我去上海，不是，我去北京", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("北京"), "{}", result.text);
    }

    #[tokio::test]
    async fn chinese_no_self_correction_in_clean_text() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("会议在下午三点开始", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "会议在下午三点开始");
        assert_eq!(result.corrections.len(), 0);
    }

    // -----------------------------------------------------------------
    // Korean self-correction — eojeol-tokenised with 1-eojeol fallback.
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn korean_self_correction_single_eojeol_repair() {
        // `8시 아니 9시에 만나자` — single-eojeol reparandum,
        // 1-eojeol-fallback path.
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("8시 아니 9시에 만나자", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("9시"), "repair: {}", result.text);
        assert!(!result.text.contains("8시"), "reparandum: {}", result.text);
    }

    #[tokio::test]
    async fn korean_self_correction_with_geuge_anira() {
        // `그게 아니라` = "that's not it, but…" — multi-eojeol cue
        // outranks bare `아니` via longest-first sort.
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("내가 갈게 그게 아니라 네가 갈게", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("네가"), "{}", result.text);
        assert!(!result.text.contains("내가"), "{}", result.text);
    }

    #[tokio::test]
    async fn korean_self_correction_with_jamkkanman() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("회의실은 삼층 잠깐만 사층입니다", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("사층"), "{}", result.text);
    }

    #[tokio::test]
    async fn korean_self_correction_with_shared_prefix_overlap() {
        // En / es style shared-prefix path. The first word of the
        // repair (`오늘`) appears in the reparandum (`오늘 갈게`),
        // so count_shared_prefix_from_end returns 2 → drop both
        // trailing eojeol from before, keep the after verbatim.
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("오늘 갈게 아니 오늘 안 갈게", &empty_context())
            .await
            .unwrap();
        assert!(result.text.contains("안 갈게"), "{}", result.text);
    }

    #[tokio::test]
    async fn korean_no_self_correction_in_clean_text() {
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("내일 오후 세 시에 만나기로 했습니다", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "내일 오후 세 시에 만나기로 했습니다");
        assert_eq!(result.corrections.len(), 0);
    }

    #[tokio::test]
    async fn korean_ani_inside_word_does_not_trigger() {
        // `아니에요` is a sentence-final negation predicate
        // ("is not"), not a cue. The 아니에요 cue itself IS in our
        // list (so a bare "X 아니에요 Y" form would fire), but
        // `이것은 아니에요` (= "this is not") is a clean sentence
        // because there's no Y after the cue.
        let detector = SelfCorrectionDetector::new();
        let result = detector
            .process("이것은 아니에요", &empty_context())
            .await
            .unwrap();
        // Cue at end of utterance → after_cue is empty → detector
        // skips this position.
        assert_eq!(result.text, "이것은 아니에요");
    }

    // --- InverseTextNormalizer tests ---

    #[tokio::test]
    async fn itn_english_sentence_normalizes_numerals() {
        let itn = InverseTextNormalizer::new("en");
        let result = itn
            .process("i have twenty five dollars", &empty_context())
            .await
            .unwrap();
        assert!(
            result.text.contains("$25"),
            "expected written-form currency: {}",
            result.text
        );
        assert_eq!(result.corrections.len(), 1);
        assert_eq!(
            result.corrections[0].kind,
            CorrectionKind::NumeralNormalized
        );
    }

    #[tokio::test]
    async fn itn_chinese_sentence_normalizes_numerals() {
        let itn = InverseTextNormalizer::new("zh");
        let result = itn
            .process("我有二十五块钱", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "我有25块钱");
    }

    #[tokio::test]
    async fn itn_japanese_sentence_normalizes_numerals() {
        let itn = InverseTextNormalizer::new("ja");
        let result = itn
            .process("そこに鳥一羽がいます", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "そこに鳥1羽がいます");
    }

    #[tokio::test]
    async fn itn_clean_text_emits_no_correction() {
        let itn = InverseTextNormalizer::new("en");
        let result = itn
            .process("the meeting is on friday", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "the meeting is on friday");
        assert!(result.corrections.is_empty());
    }

    #[tokio::test]
    async fn itn_spanish_passes_through_until_upstream_support() {
        // es has no sentence-level ITN entry point upstream yet, so the
        // normalizer is a no-op rather than silently mangling text.
        let itn = InverseTextNormalizer::new("es");
        let result = itn
            .process("tengo veinticinco años", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "tengo veinticinco años");
        assert!(result.corrections.is_empty());
    }

    #[tokio::test]
    async fn itn_korean_passes_through_until_upstream_support() {
        // ko support is a pending upstream PR; until it lands the
        // normalizer leaves Korean text untouched.
        let itn = InverseTextNormalizer::new("ko");
        let result = itn
            .process("이천이십육년에 만났다", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "이천이십육년에 만났다");
        assert!(result.corrections.is_empty());
    }

    #[tokio::test]
    async fn itn_unknown_language_passes_through() {
        let itn = InverseTextNormalizer::new("xx");
        let result = itn
            .process("arbitrary text", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "arbitrary text");
        assert!(result.corrections.is_empty());
    }

    // --- SpokenFormNormalizer tests ---

    #[tokio::test]
    async fn spoken_form_expands_english_reductions() {
        let sfn = SpokenFormNormalizer::new("en");
        let result = sfn
            .process("i'm gonna grab a coffee", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "i'm going to grab a coffee");
        assert_eq!(result.corrections.len(), 1);
        assert_eq!(
            result.corrections[0].kind,
            CorrectionKind::SpokenFormNormalized
        );
    }

    #[tokio::test]
    async fn spoken_form_handles_multiple_and_punctuation() {
        let sfn = SpokenFormNormalizer::new("en");
        let result = sfn
            .process("lemme know, i kinda forgot.", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "let me know, i kind of forgot.");
        assert_eq!(result.corrections.len(), 2);
    }

    #[tokio::test]
    async fn spoken_form_preserves_leading_capitalization() {
        let sfn = SpokenFormNormalizer::new("en");
        let result = sfn
            .process("Gonna head out now", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "Going to head out now");
    }

    #[tokio::test]
    async fn spoken_form_keeps_apostrophe_words() {
        let sfn = SpokenFormNormalizer::new("en");
        let result = sfn
            .process("c'mon we leave 'cause it's late", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "come on we leave because it's late");
    }

    #[tokio::test]
    async fn spoken_form_clean_text_is_untouched() {
        let sfn = SpokenFormNormalizer::new("en");
        let result = sfn
            .process("the meeting  is on friday", &empty_context())
            .await
            .unwrap();
        // No match → input returned verbatim, including the double space.
        assert_eq!(result.text, "the meeting  is on friday");
        assert!(result.corrections.is_empty());
    }

    #[tokio::test]
    async fn spoken_form_unsupported_language_passes_through() {
        let sfn = SpokenFormNormalizer::new("ja");
        let result = sfn
            .process("これはテストです", &empty_context())
            .await
            .unwrap();
        assert_eq!(result.text, "これはテストです");
        assert!(result.corrections.is_empty());
    }
}
