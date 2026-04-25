//! Annotation schema for L3 direct evaluation (`§5.6.1`).
//!
//! JSON-Lines, one utterance per line, two flavours:
//!
//! - **Filler annotation** (`fillers` field): `[{start, end, label}]`
//!   spans within `text` that mark closed-class fillers (え, えーと,
//!   嗯, 那个, etc.). Used as gold for `TextFilter` F1.
//! - **Self-correction annotation** (`repairs` field): `[{reparandum,
//!   interregnum, repair, type}]` reparandum/interregnum/repair span
//!   triples. Used as gold for `SelfCorrectionDetector` F1.
//!
//! A single utterance can carry both `fillers` and `repairs`; either
//! may be empty.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::eval::f1::Span;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub utterance_id: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fillers: Vec<FillerSpan>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repairs: Vec<RepairAnnotation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillerSpan {
    pub start: usize,
    pub end: usize,
    pub label: String,
}

impl FillerSpan {
    pub fn span(&self) -> Span {
        Span {
            start: self.start,
            end: self.end,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairAnnotation {
    pub reparandum: SpanField,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interregnum: Option<SpanField>,
    pub repair: SpanField,
    #[serde(default = "default_repair_type")]
    pub r#type: String,
}

fn default_repair_type() -> String {
    "substitution".to_string()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpanField {
    pub start: usize,
    pub end: usize,
}

impl SpanField {
    pub fn span(&self) -> Span {
        Span {
            start: self.start,
            end: self.end,
        }
    }
}

pub fn load_jsonl(path: &Path) -> std::io::Result<Vec<Annotation>> {
    let raw = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let a: Annotation = serde_json::from_str(line).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}:{}: {e}", path.display(), i + 1),
            )
        })?;
        out.push(a);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tempfile(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "euhadra-eval-anno-{}-{}",
            std::process::id(),
            name
        ));
        p
    }

    #[test]
    fn loads_filler_only_entry() {
        let path = tempfile("f.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"utterance_id":"a","text":"um hi","fillers":[{{"start":0,"end":2,"label":"um"}}]}}"#
        )
        .unwrap();
        let anno = load_jsonl(&path).unwrap();
        assert_eq!(anno.len(), 1);
        assert_eq!(anno[0].fillers.len(), 1);
        assert_eq!(anno[0].fillers[0].label, "um");
        assert!(anno[0].repairs.is_empty());
    }

    #[test]
    fn loads_repair_only_entry() {
        let path = tempfile("r.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"utterance_id":"b","text":"明日いや明後日","repairs":[{{"reparandum":{{"start":0,"end":2}},"interregnum":{{"start":2,"end":4}},"repair":{{"start":4,"end":7}},"type":"substitution"}}]}}"#
        )
        .unwrap();
        let anno = load_jsonl(&path).unwrap();
        assert_eq!(anno[0].repairs.len(), 1);
        assert_eq!(anno[0].repairs[0].reparandum.start, 0);
        assert_eq!(anno[0].repairs[0].r#type, "substitution");
    }

    #[test]
    fn round_trip_with_both() {
        let a = Annotation {
            utterance_id: "x".into(),
            text: "um I think no wait yes".into(),
            fillers: vec![FillerSpan {
                start: 0,
                end: 2,
                label: "um".into(),
            }],
            repairs: vec![RepairAnnotation {
                reparandum: SpanField { start: 9, end: 12 },
                interregnum: Some(SpanField { start: 13, end: 17 }),
                repair: SpanField { start: 18, end: 21 },
                r#type: "substitution".into(),
            }],
        };
        let json = serde_json::to_string(&a).unwrap();
        let parsed: Annotation = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.fillers[0].label, "um");
        assert_eq!(parsed.repairs[0].reparandum.start, 9);
    }

    #[test]
    fn missing_optional_interregnum_loads() {
        let path = tempfile("nointer.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"utterance_id":"c","text":"a no b","repairs":[{{"reparandum":{{"start":0,"end":1}},"repair":{{"start":5,"end":6}}}}]}}"#
        )
        .unwrap();
        let anno = load_jsonl(&path).unwrap();
        assert!(anno[0].repairs[0].interregnum.is_none());
        assert_eq!(anno[0].repairs[0].r#type, "substitution");
    }
}
