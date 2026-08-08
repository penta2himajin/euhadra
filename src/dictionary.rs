//! User-supplied term substitution — the dictionary a speaker maintains
//! so their own vocabulary comes out right.
//!
//! ASR gets 「タイプライター」 correct. It is what was said. But the
//! speaker wanted `typwrtr`, and no amount of acoustic modelling will
//! produce a coined spelling nobody has trained on. This is not error
//! correction, it is **terminology substitution**, and the only thing
//! that can resolve it is the user saying what they meant.
//!
//! # What lives here and what does not
//!
//! euhadra owns the *behaviour* — what counts as a match, and which
//! normalisations are safe in which language. It does not own the
//! dictionary. There is no bundled term list, no file format, and no
//! `load(path)`: [`TermEntry`] derives [`serde::Deserialize`] so a
//! consuming application reads its own settings in its own format and
//! hands over the entries.
//!
//! That split is deliberate. A dictionary shipped by euhadra would be an
//! editorial opinion about what a speaker meant; a dictionary supplied by
//! the speaker is a fact about their vocabulary. euhadra only mechanises
//! the second.
//!
//! # Why it is a pipeline stage rather than the caller's own find-replace
//!
//! Substitution is thirty lines. Ordering is not. A replacement applied
//! after [`BasicPunctuationRestorer`](crate::processor::BasicPunctuationRestorer)
//! meets text whose sentence starts have already been capitalised;
//! applied after [`InverseTextNormalizer`](crate::processor::InverseTextNormalizer)
//! it meets rewritten numerals. Only something inside the pipeline can
//! choose where it sits, which is the argument for it living here.
//!
//! ```
//! use euhadra::prelude::*;
//! use euhadra::dictionary::{MatchPolicy, TermDictionary, TermEntry};
//!
//! # async fn f() -> Result<(), Box<dyn std::error::Error>> {
//! let dictionary = TermDictionary::new(
//!     [TermEntry {
//!         term: "typwrtr".into(),
//!         aliases: vec!["タイプライター".into(), "typewriter".into()],
//!     }],
//!     MatchPolicy::for_language(Language::Japanese),
//! )?;
//!
//! let result = dictionary.process("タイプライターで書く", &ContextSnapshot::default()).await?;
//! assert_eq!(result.text, "typwrtrで書く");
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use std::collections::HashMap;

use crate::processor::{Correction, CorrectionKind, ProcessError, ProcessResult, TextProcessor};
use crate::types::{ContextSnapshot, Language, Span};

/// The shortest an alias may be once folded.
///
/// One character is indefensible in any script — a stray `e` or `あ`
/// would rewrite half the transcript. Two is where this stops being
/// obviously wrong and starts being a judgement call: `IT` passes, and
/// an English speaker who registers it will find every `it` rewritten.
///
/// **Length is a proxy for the real hazard, which is frequency.** `Qz`
/// is two characters and harmless; `IT` is two characters and not,
/// because `it` is common. Warning on frequency needs a per-language
/// word-frequency list — the kind of per-language data `docs/spec.md`
/// §11.4 classifies as not centrally scalable — so it is not here.
const MIN_ALIAS_CHARS: usize = 2;

// ---------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------

/// One term and the ASR outputs that should become it.
///
/// Aliases are written out rather than derived from readings. That costs
/// the user some typing and buys three things: no morphological analyser
/// or G2P dependency, behaviour identical in every language, and a
/// result that can be checked by inspection instead of tuned by
/// threshold.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct TermEntry {
    /// What to write. Emitted verbatim — folding never touches it.
    pub term: String,
    /// What the ASR produces instead. Matched after folding.
    pub aliases: Vec<String>,
}

// ---------------------------------------------------------------------------
// Match policy
// ---------------------------------------------------------------------------

/// How much of the text an alias has to line up with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchScope {
    /// The match must not sit inside a longer alphanumeric run, so
    /// `cat` does not fire inside `concatenate`.
    WordBoundary,
    /// Any position. The only option for scripts without word
    /// separators.
    Substring,
}

/// Which normalisations to apply before matching, and how much context a
/// match needs.
///
/// **Built from a [`Language`], not assembled by hand.** Whether a
/// normalisation is safe is language knowledge, and getting it wrong is
/// silent: stripping Spanish accents makes `año` (year) and `ano`
/// (anus) the same string. euhadra takes the same position here as
/// [`FillerFilter::for_language`](crate::filter::FillerFilter::for_language),
/// where hand-pairing a filter with the wrong script empties the output
/// without erroring.
///
/// The rule the table below follows: **a fold that loses information is
/// not included.** Hiragana and katakana spellings of a word are the same
/// word; `タイプライタ` and `タイプライター` are not necessarily (a product
/// name and a common noun can differ by exactly that mark), so the long
/// vowel is left alone.
///
/// | Language | Scope | case | full-width | kana |
/// |---|---|---|---|---|
/// | English / Spanish / Korean | word boundary | ✓ | ✓ | — |
/// | Japanese | substring | ✓ | ✓ | ✓ |
/// | Chinese | substring | ✓ | ✓ | — |
/// | [`none`](Self::none) | substring | — | — | — |
///
/// Not included for any language, and why: Spanish accent stripping
/// (loses meaning, above); Chinese simplified↔traditional (needs an
/// OpenCC-scale mapping table, not a Unicode operation); Korean jamo
/// normalisation (no evidence either side varies in practice).
/// Everything absent here is covered by writing another alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchPolicy {
    scope: MatchScope,
    fold_case: bool,
    fold_fullwidth: bool,
    fold_kana: bool,
    fold_whitespace: bool,
}

impl MatchPolicy {
    /// The normalisations measured safe for `lang`.
    pub fn for_language(lang: Language) -> Self {
        let scope = match lang {
            Language::Japanese | Language::Chinese => MatchScope::Substring,
            _ => MatchScope::WordBoundary,
        };
        Self {
            scope,
            fold_case: true,
            fold_fullwidth: true,
            fold_kana: matches!(lang, Language::Japanese),
            fold_whitespace: true,
        }
    }

    /// Exact substring matching with no normalisation at all.
    ///
    /// What to use for a language euhadra has no [`Language`] variant
    /// for — which, as ASR adapters multiply past the five text-processing
    /// languages, will be most of them. Deliberately not a silent
    /// fallback inside [`for_language`](Self::for_language): a
    /// wrong-language normalisation is worse than none, the same reason
    /// [`Language::from_bcp47`] returns `None` rather than guessing.
    pub fn none() -> Self {
        Self {
            scope: MatchScope::Substring,
            fold_case: false,
            fold_fullwidth: false,
            fold_kana: false,
            fold_whitespace: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Something wrong with one entry, located well enough to highlight.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Problem {
    /// The replacement text is empty or whitespace only.
    EmptyTerm { entry: usize },
    /// An alias is empty or whitespace only.
    EmptyAlias { entry: usize, alias: usize },
    /// An alias is identical to its own term, so matching it would do
    /// nothing.
    AliasEqualsTerm {
        entry: usize,
        alias: usize,
        term: String,
    },
    /// An alias is too short to be safe. See [`MIN_ALIAS_CHARS`].
    AliasTooShort {
        entry: usize,
        alias: usize,
        text: String,
        folded_chars: usize,
    },
    /// Two different terms claim the same alias, so which one wins would
    /// be arbitrary.
    ConflictingAlias {
        alias: String,
        first_term: String,
        second_term: String,
    },
}

/// Everything wrong with a dictionary, reported together.
///
/// All problems at once rather than the first one: a dictionary grows
/// entry by entry through an application's registration flow, and
/// fix-one-rerun-find-the-next turns a single review into a dozen. The
/// check runs once at construction and costs nothing to complete.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("dictionary has {} problem(s): {problems:?}", problems.len())]
pub struct DictionaryError {
    pub problems: Vec<Problem>,
}

// ---------------------------------------------------------------------------
// Folding
// ---------------------------------------------------------------------------

/// Text with each character mapped to the position it came from.
///
/// `origin[i]` is the index, in the original `char` sequence, of the
/// character that produced `chars[i]`. One original character can
/// produce several folded ones (`ß` lowercases to `ss`) or none
/// (a run of spaces collapses), so the two vectors are the only
/// reliable way back to a span in the caller's text.
struct Folded {
    chars: Vec<char>,
    origin: Vec<usize>,
}

fn fold(text: &str, policy: &MatchPolicy) -> Folded {
    let mut chars = Vec::new();
    let mut origin = Vec::new();

    for (index, c) in text.chars().enumerate() {
        // Full-width ASCII and the ideographic space, which is what an
        // IME produces and an ASR happily echoes.
        let c = if policy.fold_fullwidth {
            match c {
                '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
                '\u{3000}' => ' ',
                other => other,
            }
        } else {
            c
        };

        // Hiragana into katakana. Same word, different spelling.
        let c = if policy.fold_kana && ('\u{3041}'..='\u{3096}').contains(&c) {
            char::from_u32(c as u32 + 0x60).unwrap_or(c)
        } else {
            c
        };

        if policy.fold_whitespace && c.is_whitespace() {
            // Collapse: a second space contributes nothing to match on.
            if chars.last() == Some(&' ') {
                continue;
            }
            chars.push(' ');
            origin.push(index);
            continue;
        }

        if policy.fold_case {
            for lowered in c.to_lowercase() {
                chars.push(lowered);
                origin.push(index);
            }
        } else {
            chars.push(c);
            origin.push(index);
        }
    }

    Folded { chars, origin }
}

// ---------------------------------------------------------------------------
// Dictionary
// ---------------------------------------------------------------------------

/// An alias compiled for matching, paired with what it becomes.
#[derive(Debug)]
struct CompiledAlias {
    folded: Vec<char>,
    term: String,
}

/// A user's term dictionary, ready to run as a [`TextProcessor`].
///
/// Immutable once built: every check that can fail happens in
/// [`new`](Self::new), so `process` cannot report a configuration
/// problem at a point where nothing can be done about it. Rebuild it
/// when the user edits their dictionary — it holds no models and
/// costs nothing to construct.
///
/// # What matching guarantees
///
/// One pass, left to right, longest alias first at each position, and
/// **replaced text is never rescanned**. Rescanning would let a term
/// that is also another entry's alias cascade, which is either a loop or
/// a result that depends on entry order; a single pass is predictable
/// and its cost is bounded.
///
/// A match always replaces. There is no confidence score and no context
/// check that might decline — if a speaker registers `タイプライター` as
/// `typwrtr`, the day they mean an actual typewriter it will still be
/// rewritten. That is the deliberate trade: a caller can undo a
/// [`Correction`] it was told about, but cannot recover a substitution
/// that silently did not happen.
#[derive(Debug)]
pub struct TermDictionary {
    /// Sorted by folded length, descending, so the first match at a
    /// position is the longest one.
    aliases: Vec<CompiledAlias>,
    policy: MatchPolicy,
}

impl TermDictionary {
    /// Compile `entries` under `policy`, or report everything wrong with
    /// them.
    ///
    /// Entries sharing a term are merged — an application whose
    /// registration flow appends "I saw X, I wanted Y" one correction at
    /// a time will naturally produce several entries for the same term.
    pub fn new(
        entries: impl IntoIterator<Item = TermEntry>,
        policy: MatchPolicy,
    ) -> Result<Self, DictionaryError> {
        let entries: Vec<TermEntry> = entries.into_iter().collect();
        let mut problems = Vec::new();
        // Folded alias -> (term, the alias as written) for conflict
        // detection. Folded, because two aliases that fold together are
        // the same alias as far as matching is concerned.
        let mut claimed: HashMap<String, (String, String)> = HashMap::new();
        let mut aliases: Vec<CompiledAlias> = Vec::new();

        for (entry_index, entry) in entries.iter().enumerate() {
            if entry.term.trim().is_empty() {
                problems.push(Problem::EmptyTerm { entry: entry_index });
                continue;
            }

            for (alias_index, alias) in entry.aliases.iter().enumerate() {
                if alias.trim().is_empty() {
                    problems.push(Problem::EmptyAlias {
                        entry: entry_index,
                        alias: alias_index,
                    });
                    continue;
                }
                // Compared as written, not folded: `typwrtr` -> `Typwrtr`
                // is a real correction even though the two fold together.
                if *alias == entry.term {
                    problems.push(Problem::AliasEqualsTerm {
                        entry: entry_index,
                        alias: alias_index,
                        term: entry.term.clone(),
                    });
                    continue;
                }

                let folded = fold(alias, &policy).chars;
                if folded.len() < MIN_ALIAS_CHARS {
                    problems.push(Problem::AliasTooShort {
                        entry: entry_index,
                        alias: alias_index,
                        text: alias.clone(),
                        folded_chars: folded.len(),
                    });
                    continue;
                }

                let key: String = folded.iter().collect();
                match claimed.get(&key) {
                    // The same term claiming an alias twice is an
                    // append-only registration flow doing its job, not a
                    // mistake.
                    Some((term, _)) if *term == entry.term => continue,
                    Some((term, written)) => {
                        problems.push(Problem::ConflictingAlias {
                            alias: written.clone(),
                            first_term: term.clone(),
                            second_term: entry.term.clone(),
                        });
                        continue;
                    }
                    None => {}
                }
                claimed.insert(key, (entry.term.clone(), alias.clone()));
                aliases.push(CompiledAlias {
                    folded,
                    term: entry.term.clone(),
                });
            }
        }

        if !problems.is_empty() {
            return Err(DictionaryError { problems });
        }

        // Longest first: with a linear scan per position, that makes the
        // first hit the longest hit.
        aliases.sort_by_key(|alias| std::cmp::Reverse(alias.folded.len()));
        Ok(Self { aliases, policy })
    }

    /// How many aliases are compiled. Zero is a valid dictionary that
    /// changes nothing.
    pub fn len(&self) -> usize {
        self.aliases.len()
    }

    pub fn is_empty(&self) -> bool {
        self.aliases.is_empty()
    }

    /// Apply the dictionary, returning the new text and what changed.
    ///
    /// The synchronous half of [`TextProcessor::process`], for callers
    /// outside a pipeline.
    pub fn apply(&self, text: &str) -> (String, Vec<Correction>) {
        let original: Vec<char> = text.chars().collect();
        let folded = fold(text, &self.policy);

        let mut corrections = Vec::new();
        // Replacements as (original char range, term), in text order.
        let mut plan: Vec<(usize, usize, &str)> = Vec::new();

        let mut i = 0usize;
        while i < folded.chars.len() {
            let hit = self
                .aliases
                .iter()
                .find(|alias| self.matches_at(&folded.chars, i, alias));

            match hit {
                Some(alias) => {
                    let end = i + alias.folded.len();
                    let from = folded.origin[i];
                    // `end - 1` is the last folded character of the
                    // match; the original character it came from is the
                    // last one the match covers.
                    let to = folded.origin[end - 1] + 1;
                    corrections.push(Correction {
                        kind: CorrectionKind::DictionaryMatch,
                        original: original[from..to].iter().collect(),
                        replacement: alias.term.clone(),
                        span: Some(Span {
                            start: from,
                            end: to,
                        }),
                    });
                    plan.push((from, to, &alias.term));
                    i = end;
                }
                None => i += 1,
            }
        }

        if plan.is_empty() {
            return (text.to_string(), corrections);
        }

        let mut out = String::with_capacity(text.len());
        let mut cursor = 0usize;
        for (from, to, term) in plan {
            out.extend(original[cursor..from].iter());
            out.push_str(term);
            cursor = to;
        }
        out.extend(original[cursor..].iter());
        (out, corrections)
    }

    fn matches_at(&self, haystack: &[char], at: usize, alias: &CompiledAlias) -> bool {
        let end = at + alias.folded.len();
        if end > haystack.len() || haystack[at..end] != alias.folded[..] {
            return false;
        }
        match self.policy.scope {
            MatchScope::Substring => true,
            MatchScope::WordBoundary => {
                let before_ok = at == 0 || !haystack[at - 1].is_alphanumeric();
                let after_ok = end == haystack.len() || !haystack[end].is_alphanumeric();
                before_ok && after_ok
            }
        }
    }
}

#[async_trait]
impl TextProcessor for TermDictionary {
    async fn process(
        &self,
        text: &str,
        _context: &ContextSnapshot,
    ) -> Result<ProcessResult, ProcessError> {
        let (text, corrections) = self.apply(text);
        Ok(ProcessResult { text, corrections })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(term: &str, aliases: &[&str]) -> TermEntry {
        TermEntry {
            term: term.into(),
            aliases: aliases.iter().map(|a| (*a).into()).collect(),
        }
    }

    fn ja(entries: Vec<TermEntry>) -> TermDictionary {
        TermDictionary::new(entries, MatchPolicy::for_language(Language::Japanese))
            .expect("valid dictionary")
    }

    fn en(entries: Vec<TermEntry>) -> TermDictionary {
        TermDictionary::new(entries, MatchPolicy::for_language(Language::English))
            .expect("valid dictionary")
    }

    // ── The case this exists for ────────────────────────────────────────

    /// The ASR is not wrong — 「タイプライター」 is what was said. The
    /// speaker wants a different spelling, and only they can say so.
    #[test]
    fn a_registered_term_replaces_its_alias() {
        let dict = ja(vec![entry("typwrtr", &["タイプライター"])]);
        let (text, corrections) = dict.apply("タイプライターで書いている");
        assert_eq!(text, "typwrtrで書いている");
        assert_eq!(corrections.len(), 1);
        assert_eq!(corrections[0].kind, CorrectionKind::DictionaryMatch);
        assert_eq!(corrections[0].original, "タイプライター");
        assert_eq!(corrections[0].replacement, "typwrtr");
    }

    #[test]
    fn text_with_no_match_is_returned_unchanged() {
        let dict = ja(vec![entry("typwrtr", &["タイプライター"])]);
        let (text, corrections) = dict.apply("今日はいい天気だ");
        assert_eq!(text, "今日はいい天気だ");
        assert!(corrections.is_empty());
    }

    #[test]
    fn every_occurrence_is_replaced() {
        let dict = ja(vec![entry("typwrtr", &["タイプライター"])]);
        let (text, corrections) = dict.apply("タイプライターとタイプライター");
        assert_eq!(text, "typwrtrとtypwrtr");
        assert_eq!(corrections.len(), 2);
    }

    // ── Spans ───────────────────────────────────────────────────────────

    /// Codepoints, not bytes. A caller offering undo needs to find the
    /// replacement again, and every other span in this crate is measured
    /// the same way.
    #[test]
    fn the_span_locates_the_match_in_codepoints() {
        let dict = ja(vec![entry("typwrtr", &["タイプライター"])]);
        let (_, corrections) = dict.apply("私はタイプライターを使う");
        let span = corrections[0].span.expect("a substitution knows where it was");
        assert_eq!(span, Span { start: 2, end: 9 });

        // Byte offsets would have been 6..27 — the assertion above is
        // what distinguishes the two.
        let chars: String = "私はタイプライターを使う"
            .chars()
            .skip(span.start)
            .take(span.len())
            .collect();
        assert_eq!(chars, "タイプライター");
    }

    #[test]
    fn spans_of_several_matches_are_in_text_order() {
        let dict = ja(vec![entry("typwrtr", &["タイプライター"])]);
        let (_, corrections) = dict.apply("タイプライターとタイプライター");
        let spans: Vec<Span> = corrections.iter().filter_map(|c| c.span).collect();
        assert_eq!(spans[0], Span { start: 0, end: 7 });
        assert_eq!(spans[1], Span { start: 8, end: 15 });
    }

    // ── Match scope ─────────────────────────────────────────────────────

    /// English has word boundaries, so use them. Substring matching here
    /// would rewrite `concatenate` for an alias of `cat`.
    #[test]
    fn an_english_alias_does_not_fire_inside_a_longer_word() {
        let dict = en(vec![entry("Category", &["cat"])]);
        let (text, corrections) = dict.apply("concatenate the cat");
        assert_eq!(text, "concatenate the Category");
        assert_eq!(corrections.len(), 1);
    }

    /// Japanese has none, so substring matching is the only option.
    #[test]
    fn a_japanese_alias_fires_mid_string() {
        let dict = ja(vec![entry("typwrtr", &["タイプライター"])]);
        let (text, _) = dict.apply("これはタイプライターだ");
        assert_eq!(text, "これはtypwrtrだ");
    }

    #[test]
    fn punctuation_still_bounds_an_english_match() {
        let dict = en(vec![entry("typwrtr", &["typewriter"])]);
        let (text, _) = dict.apply("a typewriter, and a typewriter.");
        assert_eq!(text, "a typwrtr, and a typwrtr.");
    }

    // ── Folding ─────────────────────────────────────────────────────────

    #[test]
    fn japanese_folds_hiragana_and_katakana_together() {
        let dict = ja(vec![entry("typwrtr", &["タイプライター"])]);
        let (text, _) = dict.apply("たいぷらいたーを使う");
        assert_eq!(text, "typwrtrを使う");
    }

    /// The long vowel is *not* folded — a product name and a common noun
    /// can differ by exactly that mark, so folding it would lose
    /// information the speaker may have meant.
    #[test]
    fn japanese_does_not_fold_the_long_vowel_mark() {
        let dict = ja(vec![entry("typwrtr", &["タイプライター"])]);
        let (text, corrections) = dict.apply("タイプライタを使う");
        assert_eq!(text, "タイプライタを使う");
        assert!(corrections.is_empty(), "got {corrections:?}");
    }

    #[test]
    fn full_width_ascii_folds_to_half_width() {
        let dict = ja(vec![entry("euhadra", &["ユーハドラ"])]);
        let (text, _) = dict.apply("ＧitHub のユーハドラ");
        assert_eq!(text, "ＧitHub のeuhadra");

        // And the other direction: a full-width alias matches half-width
        // input.
        let dict = ja(vec![entry("OK", &["ＯＫです"])]);
        let (text, _) = dict.apply("OKですね");
        assert_eq!(text, "OKね");
    }

    #[test]
    fn case_is_folded_for_matching_but_never_for_the_replacement() {
        let dict = en(vec![entry("TensorFlow", &["tensor flow"])]);
        let (text, _) = dict.apply("Import Tensor Flow now");
        assert_eq!(
            text, "Import TensorFlow now",
            "the term is emitted verbatim; only the match side folds"
        );
    }

    /// The capitalisation a punctuation restorer adds must not stop a
    /// match — that is what makes processor order less load-bearing than
    /// it would otherwise be.
    #[test]
    fn a_sentence_initial_capital_still_matches() {
        let dict = en(vec![entry("typwrtr", &["typewriter"])]);
        let (text, _) = dict.apply("Typewriter is the tool.");
        assert_eq!(text, "typwrtr is the tool.");
    }

    #[test]
    fn repeated_whitespace_collapses_for_matching() {
        let dict = en(vec![entry("TensorFlow", &["tensor flow"])]);
        let (text, corrections) = dict.apply("use tensor   flow here");
        assert_eq!(text, "use TensorFlow here");
        assert_eq!(
            corrections[0].original, "tensor   flow",
            "the correction reports what was actually in the text"
        );
    }

    #[test]
    fn a_policy_of_none_folds_nothing() {
        let dict = TermDictionary::new(
            vec![entry("typwrtr", &["typewriter"])],
            MatchPolicy::none(),
        )
        .unwrap();
        let (text, _) = dict.apply("Typewriter and typewriter");
        assert_eq!(
            text, "Typewriter and typwrtr",
            "exact matching only; the capitalised one is a different string"
        );
    }

    // ── Longest match, single pass ──────────────────────────────────────

    #[test]
    fn the_longest_alias_wins_at_a_position() {
        let dict = ja(vec![
            entry("Type", &["タイプ"]),
            entry("typwrtr", &["タイプライター"]),
        ]);
        let (text, _) = dict.apply("タイプライターとタイプ");
        assert_eq!(text, "typwrtrとType");
    }

    /// Replaced text is never rescanned. Without that, a term that is
    /// also another entry's alias would cascade, and the result would
    /// depend on entry order.
    #[test]
    fn a_replacement_is_not_rescanned() {
        let dict = ja(vec![
            entry("beta", &["アルファ"]),
            entry("gamma", &["beta"]),
        ]);
        let (text, corrections) = dict.apply("アルファ");
        assert_eq!(text, "beta", "one pass: `beta` is output, not re-matched");
        assert_eq!(corrections.len(), 1);
    }

    // ── Construction-time validation ────────────────────────────────────

    #[test]
    fn every_problem_is_reported_at_once() {
        let err = TermDictionary::new(
            vec![
                entry("", &["something"]),
                entry("ok", &[""]),
                entry("dup", &["e"]),
            ],
            MatchPolicy::for_language(Language::English),
        )
        .unwrap_err();
        assert_eq!(
            err.problems.len(),
            3,
            "a caller fixing a dictionary should see all of it, got {:?}",
            err.problems
        );
    }

    #[test]
    fn an_alias_shorter_than_the_minimum_is_refused() {
        let err = TermDictionary::new(
            vec![entry("euhadra", &["e"])],
            MatchPolicy::for_language(Language::English),
        )
        .unwrap_err();
        assert!(
            matches!(
                &err.problems[0],
                Problem::AliasTooShort { folded_chars: 1, .. }
            ),
            "got {:?}",
            err.problems
        );
    }

    /// Two characters is where the rule stops being obviously right.
    /// Pinned so that the limit is a known property rather than an
    /// accident — `IT` passing is a documented hazard, not an oversight.
    #[test]
    fn a_two_character_alias_is_allowed() {
        let dict = en(vec![entry("Information Technology", &["IT"])]);
        let (text, _) = dict.apply("the IT department");
        assert_eq!(text, "the Information Technology department");
    }

    #[test]
    fn two_terms_cannot_claim_the_same_alias() {
        let err = TermDictionary::new(
            vec![
                entry("typwrtr", &["タイプライター"]),
                entry("Typewriter Co.", &["タイプライター"]),
            ],
            MatchPolicy::for_language(Language::Japanese),
        )
        .unwrap_err();
        assert!(
            matches!(&err.problems[0], Problem::ConflictingAlias { .. }),
            "got {:?}",
            err.problems
        );
    }

    /// Two aliases that fold together are the same alias as far as
    /// matching is concerned, so the conflict is real even though the
    /// strings differ.
    #[test]
    fn aliases_that_fold_together_conflict() {
        let err = TermDictionary::new(
            vec![
                entry("typwrtr", &["タイプライター"]),
                entry("Typewriter Co.", &["たいぷらいたー"]),
            ],
            MatchPolicy::for_language(Language::Japanese),
        )
        .unwrap_err();
        assert!(
            matches!(&err.problems[0], Problem::ConflictingAlias { .. }),
            "got {:?}",
            err.problems
        );
    }

    /// An append-only registration flow produces one entry per
    /// correction, so the same term shows up repeatedly. That is the
    /// tool working, not a mistake.
    #[test]
    fn entries_sharing_a_term_are_merged() {
        let dict = ja(vec![
            entry("typwrtr", &["タイプライター"]),
            entry("typwrtr", &["タイプライター", "typewriter"]),
        ]);
        assert_eq!(dict.len(), 2, "the duplicate alias is folded away");
        let (text, _) = dict.apply("タイプライターと typewriter");
        assert_eq!(text, "typwrtrと typwrtr");
    }

    #[test]
    fn an_alias_identical_to_its_term_is_refused() {
        let err = TermDictionary::new(
            vec![entry("typwrtr", &["typwrtr"])],
            MatchPolicy::for_language(Language::English),
        )
        .unwrap_err();
        assert!(
            matches!(&err.problems[0], Problem::AliasEqualsTerm { .. }),
            "got {:?}",
            err.problems
        );
    }

    /// Differing only by case is a real correction, not a no-op, so it
    /// must survive the alias-equals-term check.
    #[test]
    fn an_alias_differing_only_in_case_is_allowed() {
        let dict = en(vec![entry("typwrtr", &["Typwrtr"])]);
        let (text, _) = dict.apply("Typwrtr here");
        assert_eq!(text, "typwrtr here");
    }

    /// One alias containing another is legitimate — `タイプ` → `Type`
    /// and `タイプライター` → `typwrtr` are both things a user might
    /// want. Longest-match resolves it deterministically.
    #[test]
    fn a_substring_relation_between_aliases_is_allowed() {
        let dict = ja(vec![
            entry("Type", &["タイプ"]),
            entry("typwrtr", &["タイプライター"]),
        ]);
        assert_eq!(dict.len(), 2);
    }

    #[test]
    fn an_empty_dictionary_is_valid_and_changes_nothing() {
        let dict = TermDictionary::new(
            Vec::<TermEntry>::new(),
            MatchPolicy::for_language(Language::English),
        )
        .unwrap();
        assert!(dict.is_empty());
        let (text, corrections) = dict.apply("nothing happens here");
        assert_eq!(text, "nothing happens here");
        assert!(corrections.is_empty());
    }

    // ── Pipeline integration ────────────────────────────────────────────

    #[tokio::test]
    async fn it_runs_as_a_text_processor() {
        let dict = ja(vec![entry("typwrtr", &["タイプライター"])]);
        let result = dict
            .process("タイプライターを使う", &ContextSnapshot::default())
            .await
            .unwrap();
        assert_eq!(result.text, "typwrtrを使う");
        assert_eq!(result.corrections.len(), 1);
    }
}
