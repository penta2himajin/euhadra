use crate::types::PipelineState;

/// Manages the lifecycle state of a single dictation session.
///
/// All transitions are validated — invalid transitions return `Err` with the
/// current state.  Cancellation is allowed from any non-Idle state.
#[derive(Debug)]
pub struct StateMachine {
    state: PipelineState,
}

#[derive(Debug, Clone)]
pub struct TransitionError {
    pub from: PipelineState,
    pub to: PipelineState,
}

impl std::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid transition: {:?} → {:?}", self.from, self.to)
    }
}

impl std::error::Error for TransitionError {}

impl StateMachine {
    pub fn new() -> Self {
        Self {
            state: PipelineState::Idle,
        }
    }

    pub fn state(&self) -> PipelineState {
        self.state
    }

    /// Attempt a state transition.  Returns Ok(new_state) on success.
    pub fn transition(&mut self, to: PipelineState) -> Result<PipelineState, TransitionError> {
        if self.is_valid_transition(to) {
            tracing::debug!(from = ?self.state, to = ?to, "state transition");
            self.state = to;
            Ok(to)
        } else {
            Err(TransitionError {
                from: self.state,
                to,
            })
        }
    }

    /// Cancel from any non-Idle state → Cancelling → Idle.
    pub fn cancel(&mut self) -> Result<PipelineState, TransitionError> {
        if self.state == PipelineState::Idle {
            return Err(TransitionError {
                from: self.state,
                to: PipelineState::Cancelling,
            });
        }
        tracing::debug!(from = ?self.state, "cancelling");
        self.state = PipelineState::Cancelling;
        Ok(PipelineState::Cancelling)
    }

    /// Reset to Idle (called after cancellation cleanup completes).
    pub fn reset(&mut self) {
        tracing::debug!("reset to idle");
        self.state = PipelineState::Idle;
    }

    fn is_valid_transition(&self, to: PipelineState) -> bool {
        use PipelineState::*;
        matches!(
            (self.state, to),
            (Idle, Activating)
                | (Activating, Recording)
                | (Recording, Processing)
                | (Processing, Emitting)
                | (Emitting, Idle)
                | (Cancelling, Idle)
        )
    }
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PipelineState::*;

    #[test]
    fn happy_path() {
        let mut sm = StateMachine::new();
        assert_eq!(sm.state(), Idle);
        assert!(sm.transition(Activating).is_ok());
        assert!(sm.transition(Recording).is_ok());
        assert!(sm.transition(Processing).is_ok());
        assert!(sm.transition(Emitting).is_ok());
        assert!(sm.transition(Idle).is_ok());
    }

    #[test]
    fn cancel_from_recording() {
        let mut sm = StateMachine::new();
        sm.transition(Activating).unwrap();
        sm.transition(Recording).unwrap();
        assert!(sm.cancel().is_ok());
        assert_eq!(sm.state(), Cancelling);
        sm.reset();
        assert_eq!(sm.state(), Idle);
    }

    #[test]
    fn cancel_from_idle_fails() {
        let mut sm = StateMachine::new();
        assert!(sm.cancel().is_err());
    }

    #[test]
    fn invalid_transition_fails() {
        let mut sm = StateMachine::new();
        assert!(sm.transition(Recording).is_err()); // Idle → Recording is invalid
        assert!(sm.transition(Processing).is_err());
    }

    #[test]
    fn cancel_from_every_active_state() {
        for start in [Activating, Recording, Processing, Emitting] {
            let mut sm = StateMachine::new();
            sm.transition(Activating).unwrap();
            // Walk to the target state
            let path: &[PipelineState] = match start {
                Activating => &[],
                Recording => &[Recording],
                Processing => &[Recording, Processing],
                Emitting => &[Recording, Processing, Emitting],
                _ => unreachable!(),
            };
            for &s in path {
                sm.transition(s).unwrap();
            }
            assert!(sm.cancel().is_ok(), "cancel should work from {:?}", start);
        }
    }
}
