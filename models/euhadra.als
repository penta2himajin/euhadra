-- euhadra.als
-- Core domain model for the euhadra voice input framework.
-- Single source of truth for types shared across Rust core, Swift shell, Kotlin shell.

-------------------------------------------------------------------------------
-- ASR Layer
-------------------------------------------------------------------------------

sig AudioChunk {
  sampleRate: one AudioChunk,   -- placeholder for integer (samples/sec)
  channels:   one AudioChunk    -- placeholder for integer (channel count)
}

sig AsrResult {
  text:       one AsrResult,    -- recognized text content
  isFinal:    one Finality,     -- whether this is a final or partial result
  confidence: one AsrResult,    -- confidence score (0.0–1.0)
  timestamp:  one AsrResult     -- timestamp within audio stream
}

abstract sig Finality {}
one sig Final   extends Finality {}
one sig Partial extends Finality {}

-------------------------------------------------------------------------------
-- Context Layer
-------------------------------------------------------------------------------

abstract sig FieldType {}
one sig CodeEditor   extends FieldType {}
one sig EmailCompose extends FieldType {}
one sig ChatMessage  extends FieldType {}
one sig Terminal     extends FieldType {}
one sig Document     extends FieldType {}
one sig SearchBar    extends FieldType {}
one sig Generic      extends FieldType {}

sig ContextSnapshot {
  appName:          lone ContextSnapshot,   -- focused app name (optional)
  appBundleId:      lone ContextSnapshot,   -- app identifier (optional)
  fieldContent:     lone ContextSnapshot,   -- existing text field content (optional)
  fieldType:        lone FieldType,         -- text field kind (optional)
  customDictionary: set ContextSnapshot,    -- user dictionary terms
  instructions:     lone ContextSnapshot,   -- user-defined custom instructions (optional)
  locale:           lone ContextSnapshot    -- current locale (optional)
}

-------------------------------------------------------------------------------
-- LLM Refinement Layer
-------------------------------------------------------------------------------

abstract sig RefinementMode {}
one sig Dictation  extends RefinementMode {}
one sig Command    extends RefinementMode {}
one sig Structured extends RefinementMode {}

sig RefinementInput {
  rawText: one RefinementInput,       -- ASR raw text
  context: one ContextSnapshot,       -- app context
  mode:    one RefinementMode         -- processing mode
}

abstract sig RefinementOutput {}

sig TextInsertion extends RefinementOutput {
  insertionText: one TextInsertion    -- formatted text to insert
}

sig CommandOutput extends RefinementOutput {
  action:     one CommandOutput       -- action name
  -- parameters would be a map, simplified here
}

sig StructuredInput extends RefinementOutput {
  intent:       one StructuredInput,  -- detected intent
  intentText:   lone StructuredInput  -- optional text payload
}

-------------------------------------------------------------------------------
-- Pipeline State Machine
-------------------------------------------------------------------------------

abstract sig PipelineState {}
one sig Idle       extends PipelineState {}
one sig Activating extends PipelineState {}
one sig Recording  extends PipelineState {}
one sig Processing extends PipelineState {}
one sig Emitting   extends PipelineState {}
one sig Cancelling extends PipelineState {}

sig PipelineSession {
  currentState: one PipelineState,
  lastResult:   lone AsrResult,
  lastOutput:   lone RefinementOutput
}

-------------------------------------------------------------------------------
-- Output Layer
-------------------------------------------------------------------------------

abstract sig EmitStatus {}
one sig EmitSuccess extends EmitStatus {}
one sig EmitFailure extends EmitStatus {}

sig EmitResult {
  status: one EmitStatus,
  error:  lone EmitResult     -- error description (optional)
}

-------------------------------------------------------------------------------
-- Activation Layer
-------------------------------------------------------------------------------

abstract sig ActivationMethod {}
one sig Hotkey      extends ActivationMethod {}
one sig PushToTalk  extends ActivationMethod {}
one sig VoiceActivityDetection extends ActivationMethod {}

sig ActivationConfig {
  method: one ActivationMethod
}

-------------------------------------------------------------------------------
-- Domain Facts (Invariants)
-------------------------------------------------------------------------------

fact RefinementInputRequiresContext {
  all r: RefinementInput | some r.context
}

fact SessionStateConsistency {
  all s: PipelineSession |
    s.currentState = Idle implies no s.lastResult
}

fact EmitFailureRequiresError {
  all e: EmitResult | e.status = EmitFailure implies some e.error
}

fact EmitSuccessNoError {
  all e: EmitResult | e.status = EmitSuccess implies no e.error
}

-------------------------------------------------------------------------------
-- State Transition Predicates
-------------------------------------------------------------------------------

pred activate[s: one PipelineSession] {
  s.currentState = Idle or s.currentState = Activating
}

pred startRecording[s: one PipelineSession] {
  s.currentState = Activating
}

pred finishRecording[s: one PipelineSession] {
  s.currentState = Recording
}

pred emitOutput[s: one PipelineSession] {
  s.currentState = Processing
}

pred returnToIdle[s: one PipelineSession] {
  s.currentState = Emitting or s.currentState = Cancelling
}

pred cancel[s: one PipelineSession] {
  s.currentState != Idle
}

-------------------------------------------------------------------------------
-- Safety Assertions
-------------------------------------------------------------------------------

assert SessionAlwaysHasState {
  all s: PipelineSession | some s.currentState
}

assert EmitResultConsistency {
  all e: EmitResult |
    (e.status = EmitSuccess implies no e.error) and
    (e.status = EmitFailure implies some e.error)
}

check SessionAlwaysHasState   for 6
check EmitResultConsistency   for 6
