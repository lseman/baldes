//! Variable-cost caching and route bound correction.
use super::*;
impl Solver {
    pub fn cache_variable_costs(&mut self, costs: &[f64]) -> Result<()> {
        if costs.iter().any(|v| !v.is_finite())
            || self.maximum_variable_id.is_some_and(|id| id >= costs.len())
        {
            return Err(invalid("variable cost domain"));
        }
        let mut next = vec![];
        for arc in &self.graph.arcs {
            let mut value = arc.cost;
            for &(id, a) in &arc.variables {
                let raw = *costs
                    .get(id)
                    .ok_or_else(|| invalid("variable cost array is too short"))?;
                value += (raw * 1e8 + 0.5).floor() / 1e8 * a;
            }
            if !value.is_finite() {
                return Err(invalid("cached arc cost overflow"));
            }
            next.push(value);
        }
        for (arc, cost) in self.graph.arcs.iter_mut().zip(next) {
            arc.reduced_cost = cost;
        }
        self.variable_costs = costs.to_vec();
        Ok(())
    }
    /// Price the raw variable-cost problem for the functor interface. Rounding
    /// can change dominance and discard the raw optimum, so correcting only a
    /// returned cached route cannot establish a global raw pricing bound.
    pub fn price_functor_phase(&mut self, index: usize, maximum: usize) -> Result<PricingResult> {
        if self.variable_costs.is_empty()
            && self.graph.arcs.iter().any(|arc| !arc.variables.is_empty())
        {
            return Err(invalid("cache variable costs before functor pricing"));
        }
        let started = std::time::Instant::now();
        let budget = self.pricing_time_limit()?;
        let cached = self.price_phase(index, maximum, false)?;
        if self.variable_costs.is_empty() {
            return Ok(cached);
        }
        // Bucket fixing and retained paths may omit the raw optimum. Physical
        // forbidden arcs and Ryan-Foster constraints remain in the fork.
        let mut raw = self.fork_pricing(false);
        for arc in &mut raw.graph.arcs {
            let mut cost = arc.cost;
            for &(id, coefficient) in &arc.variables {
                cost += self
                    .variable_costs
                    .get(id)
                    .ok_or_else(|| invalid("cached variable cost domain"))?
                    * coefficient;
            }
            if !cost.is_finite() {
                return Err(invalid("raw variable arc cost overflow"));
            }
            arc.reduced_cost = cost;
        }
        let remaining = budget.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            self.requires_rollback = self.rollback_point_saved;
            return Err(incomplete("raw variable pricing time limit"));
        }
        raw.controls.time_limit = remaining;
        let mut answer = raw.price_phase(index, maximum, false);
        self.round = raw.round;
        self.requires_rollback = raw.requires_rollback;
        // Report the work of both sweeps while keeping the cached labels and
        // arc costs associated with the original solver's pricing domain.
        let mut statistics = cached.statistics;
        if let Some(engine) = &raw.engine {
            statistics.absorb(&engine.statistics);
            statistics.forward_time += engine.statistics.forward_time;
            statistics.backward_time += engine.statistics.backward_time;
            statistics.concatenation_time += engine.statistics.concatenation_time;
        }
        statistics.elapsed = started.elapsed().as_secs_f64();
        if let Some(engine) = &mut self.engine {
            engine.statistics = statistics.clone();
        }
        if let Ok(result) = &mut answer {
            result.statistics = statistics;
        }
        answer
    }

    /// Correct one cached route's value. This is a route-specific correction,
    /// not a lower bound on routes discarded by rounded pricing.
    pub fn corrected_bound(&self, mut bound: f64, route: &crate::Route) -> Result<f64> {
        if self.variable_costs.is_empty() {
            return Ok(bound);
        }
        let mut values = std::collections::BTreeMap::new();
        for &id in &route.arcs {
            let arc = self
                .graph
                .arcs
                .iter()
                .find(|a| a.id == id)
                .ok_or_else(|| invalid("unknown functor route arc"))?;
            for &(id, a) in &arc.variables {
                *values.entry(id).or_insert(0.) += a;
            }
        }
        for (id, value) in values {
            let raw = *self
                .variable_costs
                .get(id)
                .ok_or_else(|| invalid("cached variable cost domain"))?;
            bound += (raw - (raw * 1e8 + 0.5).floor() / 1e8) * value;
        }
        Ok(bound)
    }
}
