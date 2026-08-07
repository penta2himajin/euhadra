//! The names needed to build and run a pipeline.
//!
//! Glob-importing this should be enough for ordinary use:
//!
//! ```
//! use euhadra::prelude::*;
//! ```
//!
//! Concrete adapters that carry a dependency (ONNX backends, ASR runtimes)
//! are not here — import those from their own modules, so that what a
//! dependent pulls in stays visible at the use site.

pub use crate::filter::{
    ChineseFillerFilter, FillerFilter, FilterError, FilterResult, JapaneseFillerFilter,
    SimpleFillerFilter, SpanishFillerFilter, TextFilter,
};
pub use crate::pipeline::{
    Diagnostics, FinalPass, Partial, Pipeline, PipelineBuilder, PipelineError, Session,
    SessionResult, Stage, StageFailure,
};
pub use crate::processor::{
    BasicPunctuationRestorer, ProcessResult, SelfCorrectionDetector, TextProcessor,
};
pub use crate::router::{AdapterRequest, AsrRouter, AsrRuntimeFactory, ModelSource, RouterError};
pub use crate::traits::*;
pub use crate::types::*;

pub use crate::emitters::StdoutEmitter;

#[cfg(any(test, feature = "testing"))]
pub use crate::mock::{MockAsr, MockContextProvider, MockEmitter, MockRefiner, RecordingAsr};

#[cfg(feature = "clipboard")]
pub use crate::emitters::ClipboardEmitter;
#[cfg(feature = "mic")]
pub use crate::mic::{MicConfig, MicStopGuard};
