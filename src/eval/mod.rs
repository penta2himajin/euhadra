//! Evaluation primitives shared by the L1 smoke runner and any future
//! benchmark harness — WER/CER metrics, latency sample collection, and
//! `ci_baseline.json` regression gating.

pub mod baseline;
pub mod latency;
pub mod metrics;
