//! The term dictionary inside a running pipeline.
//!
//! The unit tests in `src/dictionary.rs` decide what matches. These
//! decide that a pipeline carrying one behaves — that the substitution
//! reaches the output, that its `Correction` survives into
//! `Diagnostics`, and that stage order does what the module docs claim.

use euhadra::prelude::*;

fn dictionary(lang: Language, term: &str, aliases: &[&str]) -> TermDictionary {
    TermDictionary::new(
        [TermEntry {
            term: term.into(),
            aliases: aliases.iter().map(|a| (*a).into()).collect(),
        }],
        MatchPolicy::for_language(lang),
    )
    .expect("valid dictionary")
}

fn audio() -> Vec<AudioChunk> {
    vec![AudioChunk {
        samples: vec![0.0; 160],
        sample_rate: 16_000,
        channels: 1,
    }]
}

/// The case the module exists for, end to end: the ASR transcribes
/// correctly and the speaker still wants different text.
#[tokio::test]
async fn a_registered_term_reaches_the_output() {
    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("タイプライターで書いている"))
        .processor(dictionary(Language::Japanese, "typwrtr", &["タイプライター"]))
        .build()
        .unwrap();

    let result = pipeline.transcribe(&audio()).await.unwrap();
    assert_eq!(result.text(), "typwrtrで書いている");
    assert_eq!(
        result.raw_text, "タイプライターで書いている",
        "the raw ASR text is preserved for comparison"
    );
}

/// A caller offering undo needs to know what was swapped and where.
#[tokio::test]
async fn the_substitution_is_reported_in_diagnostics() {
    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("私はタイプライターを使う"))
        .processor(dictionary(Language::Japanese, "typwrtr", &["タイプライター"]))
        .build()
        .unwrap();

    let result = pipeline.transcribe(&audio()).await.unwrap();
    let correction = result
        .diagnostics
        .corrections
        .iter()
        .find(|c| c.kind == CorrectionKind::DictionaryMatch)
        .expect("the substitution must be reported, not applied silently");

    assert_eq!(correction.original, "タイプライター");
    assert_eq!(correction.replacement, "typwrtr");
    assert_eq!(
        correction.span,
        Some(Span { start: 2, end: 9 }),
        "codepoint offsets into the text the processor was given"
    );
}

/// The ordering argument from the module docs, exercised rather than
/// asserted: case folding means a capital added upstream does not stop a
/// match, which is what keeps stage order from being load-bearing.
#[tokio::test]
async fn a_capital_added_upstream_does_not_prevent_a_match() {
    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("typewriter is the tool"))
        // Capitalises the sentence and adds a period before the
        // dictionary ever sees the text.
        .processor(BasicPunctuationRestorer)
        .processor(dictionary(Language::English, "typwrtr", &["typewriter"]))
        .build()
        .unwrap();

    let result = pipeline.transcribe(&audio()).await.unwrap();
    assert_eq!(result.text(), "typwrtr is the tool.");
}

/// Tier 1 removes the filler, Tier 2 substitutes the term. Neither
/// stage knows about the other.
#[tokio::test]
async fn it_composes_with_the_filler_filter() {
    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("um the typewriter works"))
        .filter(FillerFilter::for_language(Language::English))
        .processor(dictionary(Language::English, "typwrtr", &["typewriter"]))
        .build()
        .unwrap();

    let result = pipeline.transcribe(&audio()).await.unwrap();
    assert_eq!(result.text(), "the typwrtr works");
    assert!(result.diagnostics.removed.iter().any(|r| r.contains("um")));
}

/// A dictionary with nothing to say must not disturb the text — the
/// state a pipeline is in before the user has registered anything.
#[tokio::test]
async fn an_empty_dictionary_is_transparent() {
    let empty = TermDictionary::new(
        Vec::<TermEntry>::new(),
        MatchPolicy::for_language(Language::English),
    )
    .unwrap();

    let pipeline = Pipeline::builder()
        .asr(MockAsr::new("nothing to substitute here"))
        .processor(empty)
        .build()
        .unwrap();

    let result = pipeline.transcribe(&audio()).await.unwrap();
    assert_eq!(result.text(), "nothing to substitute here");
    assert!(result.diagnostics.corrections.is_empty());
}

/// A bad dictionary fails where it can be fixed — at construction, in
/// the application's settings flow — rather than mid-session where
/// nothing can be done about it.
#[test]
fn a_conflicting_dictionary_fails_before_a_pipeline_is_built() {
    let err = TermDictionary::new(
        [
            TermEntry {
                term: "typwrtr".into(),
                aliases: vec!["タイプライター".into()],
            },
            TermEntry {
                term: "Typewriter Co.".into(),
                aliases: vec!["タイプライター".into()],
            },
        ],
        MatchPolicy::for_language(Language::Japanese),
    )
    .unwrap_err();

    assert_eq!(err.problems.len(), 1, "got {:?}", err.problems);
}
