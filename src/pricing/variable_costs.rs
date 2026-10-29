//! Variable-cost caching and route bound correction.
use super::*;
impl<const D: usize> Solver<D> {
    /// Install a cached arc objective while retaining the actual raw arc
    /// objective. Base graph costs and model variable associations are intact.
    pub(crate) fn cache_arc_costs(&mut self, raw: &[f64], cached: &[f64]) -> Result<()> {
        if raw.len() != self.graph.arcs.len()
            || cached.len() != raw.len()
            || raw.iter().chain(cached).any(|v| !v.is_finite())
        {
            return Err(invalid("arc pricing cache domain"));
        }
        for (arc, &cost) in self.graph.arcs.iter_mut().zip(cached) {
            arc.reduced_cost = cost;
        }
        self.variable_costs.clear();
        let raw_costs = self.raw_arc_costs.get_or_insert_with(Vec::new);
        raw_costs.clear();
        raw_costs.extend_from_slice(raw);
        Ok(())
    }
    /// MP retains a one-to-one raw physical variable representation, while its
    /// cached heuristic objective may be rounded in original-variable space.
    pub(crate) fn cache_physical_costs(&mut self, raw: &[f64], cached: &[f64]) -> Result<()> {
        if cached.len() != self.graph.arcs.len()
            || cached.iter().any(|c| !c.is_finite())
            || self
                .graph
                .arcs
                .iter()
                .enumerate()
                .any(|(p, arc)| arc.cost != 0. || arc.variables.as_slice() != [(p, 1.)])
        {
            return Err(invalid("physical pricing cache domain"));
        }
        self.cache_variable_costs(raw)?;
        for (arc, &cost) in self.graph.arcs.iter_mut().zip(cached) {
            arc.reduced_cost = cost;
        }
        Ok(())
    }
    pub fn cache_variable_costs(&mut self, costs: &[f64]) -> Result<()> {
        if costs.iter().any(|v| !v.is_finite())
            || self.maximum_variable_id.is_some_and(|id| id >= costs.len())
        {
            return Err(invalid("variable cost domain"));
        }
        let next = &mut self.arc_cost_scratch;
        next.clear();
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
        for (arc, cost) in self.graph.arcs.iter_mut().zip(next.iter()) {
            arc.reduced_cost = *cost;
        }
        self.variable_costs.clear();
        self.variable_costs.extend_from_slice(costs);
        self.raw_arc_costs = None;
        Ok(())
    }
    /// Price the raw variable-cost problem for the functor interface. Rounding
    /// can change dominance and discard the raw optimum, so correcting only a
    /// returned cached route cannot establish a global raw pricing bound.
    pub fn price_functor_phase(&mut self, index: usize, maximum: usize) -> Result<PricingResult> {
        if self
            .enumeration
            .as_ref()
            .is_some_and(|paths| paths.completed)
        {
            let phase = self
                .phases
                .get(index)
                .ok_or_else(|| invalid("pricing phase out of range"))?;
            if !phase.exact() {
                return self.price_phase(index, maximum, false);
            }
        }
        if self.variable_costs.is_empty()
            && self.raw_arc_costs.is_none()
            && self.graph.arcs.iter().any(|arc| !arc.variables.is_empty())
        {
            return Err(invalid("cache variable costs before functor pricing"));
        }
        let started = std::time::Instant::now();
        let budget = self.pricing_time_limit()?;
        let exact_phase = self
            .phases
            .get(index)
            .ok_or_else(|| invalid("pricing phase out of range"))?
            .exact();
        if self.variable_costs.is_empty()
            && self.raw_arc_costs.is_none()
            && (!exact_phase || (self.cuts.cuts.is_empty() && self.cuts.knapsack_cuts.is_empty()))
        {
            return self.price_phase(index, maximum, false);
        }
        // The returned columns and bound belong to the raw objective. Running
        // the cached objective first discards its columns and advances the
        // pricing round twice. Refresh coefficient state without a search.
        self.refresh_cuts()?;
        // Bucket fixing and retained paths may omit the raw optimum. Physical
        // forbidden arcs and Ryan-Foster constraints remain in the fork.
        let phase = self
            .phases
            .get(index)
            .ok_or_else(|| invalid("pricing phase out of range"))?;
        let controls = self.engine_controls(phase.exact());
        let previous = self.raw_pricing.take().filter(|cache| {
            cache.controls == controls
                && cache.solver.graph.same_topology(&self.graph)
                && cache.solver.forbidden_arcs == self.forbidden_arcs
                && cache.solver.ryan_foster == self.ryan_foster
                && cache.solver.presence_pricing.inputs() == self.presence_pricing.inputs()
        });
        let reused = previous.is_some();
        let mut raw =
            previous.map_or_else(|| Box::new(self.fork_pricing(false)), |cache| cache.solver);
        if reused {
            // Only graph geometry and allocation capacity persist. Objectives,
            // cut inputs and the controller's logical round start at the same
            // parent state as a fresh raw fork. solve rebuilds all labels.
            raw.controls = self.controls.clone();
            raw.options.clone_from(&self.options);
            raw.phases.clone_from(&self.phases);
            raw.cuts.clone_from(&self.cuts);
            raw.round.clone_from(&self.round);
            raw.bucket_steps = self.bucket_steps;
            raw.cutoff = self.cutoff;
            raw.rollback_point_saved = self.rollback_point_saved;
            raw.maximum_variable_id = self.maximum_variable_id;
            raw.dominance_mode = self.dominance_mode;
            raw.knapsack_resource_indices
                .clone_from(&self.knapsack_resource_indices);
            raw.knapsack_exact_coefficients = self.knapsack_exact_coefficients;
        }
        // A raw-objective bound must also use true row prices and exact
        // knapsack coefficients. Filtering or rounding can discard the route
        // attaining the bound. refresh_cuts detects these mode changes.
        raw.exact_cut_duals = self.exact_cut_duals || exact_phase;
        raw.knapsack_exact_coefficients = self.knapsack_exact_coefficients || exact_phase;
        debug_assert!(raw.enumeration.is_none() && raw.raw_pricing.is_none());
        if self
            .raw_arc_costs
            .as_ref()
            .is_some_and(|v| v.len() != raw.graph.arcs.len())
        {
            return Err(invalid("raw arc cost domain"));
        }
        for (position, arc) in raw.graph.arcs.iter_mut().enumerate() {
            let mut cost = self.raw_arc_costs.as_ref().map_or_else(
                || {
                    if self.variable_costs.is_empty() {
                        self.graph.arcs[position].reduced_cost
                    } else {
                        arc.cost
                    }
                },
                |v| v[position],
            );
            if self.raw_arc_costs.is_none() {
                for &(id, coefficient) in &arc.variables {
                    cost += self
                        .variable_costs
                        .get(id)
                        .ok_or_else(|| invalid("cached variable cost domain"))?
                        * coefficient;
                }
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
        self.round = std::mem::take(&mut raw.round);
        self.requires_rollback = raw.requires_rollback;
        // Preserve the cached objective and its labels; only the raw sweep
        // contributes statistics to this logical pricing call.
        let mut statistics = raw
            .engine
            .as_ref()
            .map_or_else(Default::default, |e| e.statistics.clone());
        statistics.elapsed = started.elapsed().as_secs_f64();
        if let Some(engine) = &mut self.engine {
            engine.statistics = statistics.clone();
        }
        if let Ok(result) = &mut answer {
            result.statistics = statistics;
        }
        if let Some(engine) = &mut raw.engine {
            engine.release_raw_labels();
        }
        // A failed solve may have partial work, but the next solve discards
        // its labels. No raw bucket arc is ever fixed using an old objective.
        self.raw_pricing = Some(RawPricing {
            solver: raw,
            controls,
        });
        answer
    }

    /// Correct one cached route's value. This is a route-specific correction,
    /// not a lower bound on routes discarded by rounded pricing.
    pub fn corrected_bound(&self, mut bound: f64, route: &crate::Route) -> Result<f64> {
        if self.variable_costs.is_empty() && self.raw_arc_costs.is_none() {
            return Ok(bound);
        }
        for &id in &route.arcs {
            let arc = self
                .graph
                .arcs
                .iter()
                .enumerate()
                .find(|(_, a)| a.id == id)
                .ok_or_else(|| invalid("unknown functor route arc"))?;
            let (position, arc) = arc;
            let mut raw = self
                .raw_arc_costs
                .as_ref()
                .map_or(arc.cost, |v| v[position]);
            if self.raw_arc_costs.is_none() {
                for &(id, a) in &arc.variables {
                    raw += self
                        .variable_costs
                        .get(id)
                        .ok_or_else(|| invalid("cached variable cost domain"))?
                        * a;
                }
            }
            bound += raw - arc.reduced_cost;
        }
        Ok(bound)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/pricing/variable_costs_tests.rs"]
mod tests;
