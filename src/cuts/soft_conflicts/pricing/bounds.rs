//! Directed envelopes for future rewards and bidirectional corrections.
use super::*;
use crate::core::rounding::{add_down, multiply_down};

fn product(a: f64, coefficient: usize) -> Option<f64> {
    let integer = u64::try_from(coefficient).ok()?;
    if integer > (1_u64 << 53) {
        return None;
    }
    multiply_down(a, integer as f64)
}

impl PresencePricing {
    /// Lower bound on additional presence cost over any continuation.
    /// Positive duals can earn at most the as-yet-unearned coefficient;
    /// negative duals cannot improve cost by adding more events.
    /// `None` means no finite numerical envelope was established.
    pub fn remaining_cost_lower_bound(&self, state: &LabelState) -> Result<Option<f64>> {
        let states = self.states(state)?;
        let mut bound = 0.;
        for (input, state) in self.rows.iter().zip(states) {
            if input.dual <= 0. {
                continue;
            }
            let remaining = input.row.maximum_coefficient() - input.row.coefficient(state)?;
            let Some(term) = product(-input.dual, remaining) else {
                return Ok(None);
            };
            let Some(sum) = add_down(bound, term) else {
                return Ok(None);
            };
            bound = sum;
        }
        Ok(Some(bound))
    }

    /// Lower bound on the correction to two already-priced halves, knowing
    /// only this half's incidence. This differs from a remaining-cost bound:
    /// a negative dual can improve the join by removing double counting.
    /// For current coefficient c and opposite coefficient d, the union is
    /// between max(c,d) and M. Thus correction is in [-c, M-c].
    pub fn join_correction_lower_bound(&self, state: &LabelState) -> Result<Option<f64>> {
        let states = self.states(state)?;
        let mut bound = 0.;
        for (input, state) in self.rows.iter().zip(states) {
            let current = input.row.coefficient(state)?;
            let (factor, coefficient) = if input.dual < 0. {
                (input.dual, current)
            } else {
                (-input.dual, input.row.maximum_coefficient() - current)
            };
            let Some(term) = product(factor, coefficient) else {
                return Ok(None);
            };
            let Some(sum) = add_down(bound, term) else {
                return Ok(None);
            };
            bound = sum;
        }
        Ok(Some(bound))
    }
}
