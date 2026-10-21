//! Physical-arc adapter for exact presence transitions and reduced costs.
use super::*;

/// One nonlinear row's original-variable events on physical arc indices.
/// Several arcs may emit the same event, and one arc may emit several events.
#[derive(Clone, Debug, PartialEq)]
pub struct PricingInput {
    pub row: PresenceRow,
    pub arc_events: Vec<Vec<usize>>,
    pub dual: f64,
}

#[derive(Clone, Debug, Default)]
pub struct PresencePricing {
    arcs: usize,
    rows: Vec<PricingInput>,
}

impl PresencePricing {
    /// Validate the whole mapping before publishing any pricing state.
    pub fn new(arcs: usize, rows: Vec<PricingInput>) -> Result<Self> {
        for input in &rows {
            if input.arc_events.len() != arcs
                || !input.dual.is_finite()
                || input
                    .arc_events
                    .iter()
                    .flatten()
                    .any(|&e| e >= input.row.events)
            {
                return Err(invalid("soft-conflict pricing mapping or dual"));
            }
        }
        Ok(Self { arcs, rows })
    }

    pub(crate) fn validate_engine(&self, arcs: usize) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        if self.arcs != arcs {
            return Err(invalid("presence pricing graph arc count"));
        }
        let mut total = 0.;
        for input in &self.rows {
            let maximum = u64::try_from(input.row.maximum_coefficient())
                .map_err(|_| invalid("presence pricing coefficient range"))?;
            if maximum > (1_u64 << 53) {
                return Err(invalid("presence pricing coefficient range"));
            }
            let term = crate::core::rounding::multiply_down(-input.dual.abs(), maximum as f64)
                .map(|n| -n)
                .ok_or_else(|| invalid("presence pricing numerical range"))?;
            total = crate::core::rounding::add_up(total, term)
                .ok_or_else(|| invalid("presence pricing numerical range"))?;
        }
        // A join can combine two halves and their overlap correction. Reserve
        // headroom for these finite operations before accepting engine state.
        if total > f64::MAX / 8. {
            return Err(invalid("presence pricing numerical range"));
        }
        Ok(())
    }

    pub(crate) fn inputs(&self) -> &[PricingInput] {
        &self.rows
    }

    pub fn active(&self) -> bool {
        !self.rows.is_empty()
    }

    pub fn empty_state(&self) -> LabelState {
        self.active()
            .then(|| self.rows.iter().map(|r| r.row.empty_state()).collect())
    }

    fn states<'a>(&self, state: &'a LabelState) -> Result<&'a [PresenceState]> {
        let states = state.as_deref().unwrap_or(&[]);
        if states.len() != self.rows.len() {
            return Err(invalid("soft-conflict pricing state row count"));
        }
        for (input, state) in self.rows.iter().zip(states) {
            input.row.check(state)?;
        }
        Ok(states)
    }

    fn cost(dual: f64, coefficient: i64) -> Result<f64> {
        // Coefficient conversion must be exact before applying the supplied
        // binary64 dual. This is numerical pricing, not a directed certificate.
        if coefficient.unsigned_abs() > (1_u64 << 53) {
            return Err(invalid("soft-conflict pricing coefficient overflow"));
        }
        let cost = -dual * coefficient as f64;
        if !cost.is_finite() {
            return Err(invalid("soft-conflict pricing cost overflow"));
        }
        Ok(cost)
    }

    /// Arc traversal is direction independent: use the physical arc's own
    /// events in either traversal direction. The input is never mutated.
    pub fn extend(&self, state: &LabelState, arc: usize) -> Result<(LabelState, f64)> {
        let states = self.states(state)?;
        if arc >= self.arcs {
            return Err(invalid("soft-conflict physical arc outside mapping"));
        }
        if !self.active() {
            return Ok((None, 0.));
        }
        let mut next = states.to_vec();
        let mut cost = 0.;
        for (input, state) in self.rows.iter().zip(&mut next) {
            let before = input.row.coefficient(state)?;
            for &event in &input.arc_events[arc] {
                input.row.visit(state, event)?;
            }
            let delta = input.row.coefficient(state)? - before;
            let delta =
                i64::try_from(delta).map_err(|_| invalid("soft-conflict coefficient overflow"))?;
            cost += Self::cost(input.dual, delta)?;
            if !cost.is_finite() {
                return Err(invalid("soft-conflict pricing cost overflow"));
            }
        }
        Ok((Some(next.into_boxed_slice()), cost))
    }

    /// Adjustment to already accumulated forward/backward reduced costs.
    pub fn join_cost(&self, a: &LabelState, b: &LabelState) -> Result<f64> {
        let a = self.states(a)?;
        let b = self.states(b)?;
        let mut cost = 0.;
        for ((input, a), b) in self.rows.iter().zip(a).zip(b) {
            cost += Self::cost(input.dual, input.row.join_correction(a, b)?)?;
            if !cost.is_finite() {
                return Err(invalid("soft-conflict pricing join overflow"));
            }
        }
        Ok(cost)
    }

    /// Numerical presence cost replay for retained physical paths.
    pub fn route_cost(&self, arcs: &[usize]) -> Result<f64> {
        if !self.active() {
            return Ok(0.);
        }
        let mut total = 0.;
        for (input, coefficient) in self.rows.iter().zip(self.route_coefficients(arcs)?) {
            let coefficient =
                i64::try_from(coefficient).map_err(|_| invalid("presence coefficient overflow"))?;
            total += Self::cost(input.dual, coefficient)?;
            if !total.is_finite() {
                return Err(invalid("presence route cost overflow"));
            }
        }
        Ok(total)
    }

    /// Physical-route master coefficients, using the same event mapping as
    /// incremental pricing. Traversals repeated in the route remain presence.
    pub fn route_coefficients(&self, arcs: &[usize]) -> Result<Vec<usize>> {
        let mut state = self.empty_state();
        for &arc in arcs {
            if arc >= self.arcs {
                return Err(invalid("soft-conflict physical arc outside mapping"));
            }
            if let Some(states) = &mut state {
                for (input, state) in self.rows.iter().zip(states.iter_mut()) {
                    for &event in &input.arc_events[arc] {
                        input.row.visit(state, event)?;
                    }
                }
            }
        }
        self.rows
            .iter()
            .zip(self.states(&state)?)
            .map(|(r, s)| r.row.coefficient(s))
            .collect()
    }
}

mod bounds;
