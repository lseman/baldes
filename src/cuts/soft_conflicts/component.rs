//! Whole-column classes over sparse integer additive coordinates.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
pub enum ComponentSense {
    GreaterEqual,
    LessEqual,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
pub struct ComponentBound {
    pub coordinate: usize,
    pub threshold: f64,
    pub sense: ComponentSense,
}

#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
pub struct ComponentSequence {
    pub bounds: Vec<ComponentBound>,
}

impl ComponentSequence {
    /// Complement only the last predicate, preserving its inherited prefix.
    pub fn sibling(&self) -> Result<Self> {
        let mut next = self.clone();
        let last = next
            .bounds
            .last_mut()
            .ok_or_else(|| invalid("empty component sequence uses multiplicity branching"))?;
        if !last.threshold.is_finite() || last.threshold.fract() != 0. {
            return Err(invalid("component siblings require an integral threshold"));
        }
        let previous = last.threshold;
        match last.sense {
            ComponentSense::GreaterEqual => {
                last.threshold -= 1.;
                last.sense = ComponentSense::LessEqual;
            }
            ComponentSense::LessEqual => {
                last.threshold += 1.;
                last.sense = ComponentSense::GreaterEqual;
            }
        }
        if !last.threshold.is_finite() || (last.threshold - previous).abs() != 1. {
            return Err(invalid(
                "component sibling threshold must change by exactly one",
            ));
        }
        Ok(next)
    }

    pub fn satisfied_by(&self, mut value: impl FnMut(usize) -> Option<f64>) -> bool {
        self.bounds.iter().all(|b| match value(b.coordinate) {
            None => b.sense == ComponentSense::LessEqual,
            Some(x) if x.is_finite() && b.threshold.is_finite() => {
                let tolerance = x.abs().max(b.threshold.abs()) * 1e-10 + 1e-6;
                match b.sense {
                    ComponentSense::GreaterEqual => !(x < b.threshold - tolerance),
                    ComponentSense::LessEqual => !(b.threshold < x - tolerance),
                }
            }
            _ => false,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Class {
    pub coordinates: usize,
    sequence: ComponentSequence,
    /// Last event activates the class only once a physical terminal arc occurs.
    emissions: Vec<(usize, i64)>,
}

// All validated thresholds are finite, so structural equality is reflexive.
impl Eq for Class {}

impl PresenceRow {
    /// Replay an ordered physical-event stream without assuming monotonicity.
    #[cfg(feature = "clp")]
    pub(crate) fn coefficient_for_events(
        &self,
        events: impl IntoIterator<Item = usize>,
    ) -> Result<usize> {
        let mut state = self.empty_state();
        for event in events {
            if event >= self.events {
                return Err(invalid("component event outside domain"));
            }
            if let Some(class) = &self.component {
                class.advance(&mut state, event)?;
            } else {
                self.advance(&mut state, event)?;
            }
        }
        self.coefficient(&state)
    }

    #[cfg(feature = "clp")]
    pub(crate) fn replay_work(&self, events: usize) -> Option<usize> {
        if let Some(class) = &self.component {
            events
                .checked_add(class.coordinates)?
                .checked_add(
                    class
                        .emissions
                        .len()
                        .checked_mul(class.sequence.bounds.len())?,
                )?
                .checked_add(self.events)
        } else {
            events
                .checked_mul(self.pairs.len().checked_mul(2)?.checked_add(3)?)?
                .checked_add(self.events)
        }
    }

    /// Component events add signed integer values; the final event marks the
    /// sink arc. A complete path receives the conjunction's unit indicator.
    pub fn component_class(
        coordinates: usize,
        sequence: ComponentSequence,
        emissions: Vec<(usize, i64)>,
    ) -> Result<Self> {
        if sequence
            .bounds
            .iter()
            .any(|b| b.coordinate >= coordinates || !b.threshold.is_finite())
            || emissions
                .iter()
                .any(|&(c, v)| c >= coordinates || v.unsigned_abs() > (1_u64 << 53))
        {
            return Err(invalid("component class coordinate or threshold"));
        }
        Ok(Self {
            events: emissions.len() + 1,
            pairs: vec![],
            mode: Mode::AnyPair,
            component: Some(Class {
                coordinates,
                sequence,
                emissions,
            }),
        })
    }

    pub fn is_component_class(&self) -> bool {
        self.component.is_some()
    }
}

impl Class {
    pub(super) fn coefficient(&self, state: &PresenceState) -> Result<usize> {
        if state.values.len() != self.coordinates {
            return Err(invalid("component state domain"));
        }
        let present = |event: usize| state.words[event / 64] & (1 << (event % 64)) != 0;
        if !present(self.emissions.len()) {
            return Ok(0);
        }
        Ok(usize::from(self.sequence.satisfied_by(|coordinate| {
            self.emissions
                .iter()
                .enumerate()
                .any(|(event, &(c, _))| c == coordinate && present(event))
                .then_some(state.values[coordinate] as f64)
        })))
    }

    pub(super) fn advance(&self, state: &mut PresenceState, event: usize) -> Result<()> {
        if let Some(&(coordinate, value)) = self.emissions.get(event) {
            let next = state.values[coordinate]
                .checked_add(value)
                .filter(|v| v.unsigned_abs() <= (1_u64 << 53))
                .ok_or_else(|| invalid("component coordinate overflow"))?;
            state.values[coordinate] = next;
        }
        state.words[event / 64] |= 1 << (event % 64);
        Ok(())
    }

    pub(super) fn join(&self, a: &PresenceState, b: &PresenceState) -> Result<PresenceState> {
        let values = a
            .values
            .iter()
            .zip(&b.values)
            .map(|(&a, &b)| {
                a.checked_add(b)
                    .filter(|v| v.unsigned_abs() <= (1_u64 << 53))
                    .ok_or_else(|| invalid("component join overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PresenceState {
            events: a.events,
            words: a.words.iter().zip(&b.words).map(|(a, b)| a | b).collect(),
            values,
        })
    }
}
