//! Exact presence-based conflict coefficients and bidirectional route state.
//!
//! Events identify original variables, independently of physical arc visits.
//! A graph adapter must emit an event on every arc representing that variable.
//! This module does not register a separator or modify labeling automatically.
use crate::{invalid, Result};
use std::sync::Arc;
mod pricing;
pub use pricing::{PresencePricing, PricingInput};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Each stored pair contributes one when both events occur. Duplicates count.
    PairCount,
    /// All stored pairs together contribute at most one.
    AnyPair,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PresenceRow {
    events: usize,
    pairs: Vec<(usize, usize)>,
    mode: Mode,
}

/// Route event incidence. Equality is a sufficient state test for dominance;
/// coefficient equality alone is insufficient because unseen partners differ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PresenceState {
    events: usize,
    words: Vec<u64>,
}

/// Optional per-row route incidence carried by extended labeling records.
/// Labels share immutable incidence until an arc adds a previously unseen event.
pub type LabelState = Option<Arc<[PresenceState]>>;

impl PresenceRow {
    pub fn new(events: usize, pairs: Vec<(usize, usize)>, mode: Mode) -> Result<Self> {
        if pairs.iter().any(|&(a, b)| a >= events || b >= events) {
            return Err(invalid("soft-conflict event outside domain"));
        }
        Ok(Self {
            events,
            pairs,
            mode,
        })
    }

    /// Upper bound over every possible route incidence, independent of graph
    /// reachability. Repeated stored pairs count separately in pair-count mode.
    pub fn maximum_coefficient(&self) -> usize {
        match self.mode {
            Mode::PairCount => self.pairs.len(),
            Mode::AnyPair => usize::from(!self.pairs.is_empty()),
        }
    }

    #[cfg(feature = "clp")]
    pub(crate) fn mode(&self) -> Mode {
        self.mode
    }

    #[cfg(feature = "clp")]
    pub(crate) fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    pub fn empty_state(&self) -> PresenceState {
        PresenceState {
            events: self.events,
            words: vec![0; self.events / 64 + usize::from(self.events % 64 != 0)],
        }
    }

    fn check(&self, state: &PresenceState) -> Result<()> {
        if state.events != self.events {
            return Err(invalid("soft-conflict state domain mismatch"));
        }
        Ok(())
    }

    /// Exact coefficient of the route incidence, including repeated pairs.
    pub fn coefficient(&self, state: &PresenceState) -> Result<usize> {
        self.check(state)?;
        let present = |event: usize| state.words[event / 64] & (1 << (event % 64)) != 0;
        let matching = self
            .pairs
            .iter()
            .filter(|&&(a, b)| present(a) && present(b));
        Ok(match self.mode {
            Mode::PairCount => matching.count(),
            Mode::AnyPair => usize::from(matching.take(1).count() != 0),
        })
    }

    /// Mark one original variable present. Returns the coefficient increment;
    /// emitting an already-present event changes neither state nor cost.
    pub fn visit(&self, state: &mut PresenceState, event: usize) -> Result<usize> {
        self.check(state)?;
        if event >= self.events {
            return Err(invalid("soft-conflict event outside domain"));
        }
        let bit = 1 << (event % 64);
        if state.words[event / 64] & bit != 0 {
            return Ok(0);
        }
        let before = self.coefficient(state)?;
        state.words[event / 64] |= bit;
        Ok(self.coefficient(state)? - before)
    }

    /// Concatenate forward/backward incidence by union, including an overlap
    /// at the meeting vertex. Pair completion across halves is counted once.
    pub fn join(&self, a: &PresenceState, b: &PresenceState) -> Result<PresenceState> {
        self.check(a)?;
        self.check(b)?;
        Ok(PresenceState {
            events: self.events,
            words: a.words.iter().zip(&b.words).map(|(a, b)| a | b).collect(),
        })
    }

    /// Correction to the sum of the two half coefficients. Negative values
    /// remove double counting; positive values complete cross-half pairs.
    /// Multiply by minus the row dual when joining reduced costs.
    pub fn join_correction(&self, a: &PresenceState, b: &PresenceState) -> Result<i64> {
        let convert =
            |n| i64::try_from(n).map_err(|_| invalid("soft-conflict coefficient overflow"));
        let joined = convert(self.coefficient(&self.join(a, b)?)?)?;
        let a = convert(self.coefficient(a)?)?;
        let b = convert(self.coefficient(b)?)?;
        joined
            .checked_sub(a)
            .and_then(|n| n.checked_sub(b))
            .ok_or_else(|| invalid("soft-conflict join overflow"))
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/cuts/soft_conflicts/tests.rs"]
mod tests;
