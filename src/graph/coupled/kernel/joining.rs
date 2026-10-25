//! Forward/backward label joins and joined resource objectives.
use super::*;

impl CoupledKernel {
    pub fn join(&self, f: &CoupledLabelState, b: &CoupledLabelState) -> Result<Option<f64>> {
        self.check_state(f)?;
        self.check_state(b)?;
        if f.suffix.is_some() || b.suffix.is_none() {
            return Err(invalid("coupled join direction mismatch"));
        }
        if self.has_slope_fronts() {
            let Some(total) = self.join_resource_objective(f, b)? else {
                return Ok(None);
            };
            return Ok(Some(finite(
                finite(total - f.resource_objective)? - b.resource_objective,
            )?));
        }
        if f.selection
            .iter()
            .zip(&b.selection)
            .any(|(a, b)| a & b == 0)
        {
            return Ok(None);
        }
        for ((&left, &right), &capacity) in f.loads.iter().zip(&b.loads).zip(&self.load_capacities)
        {
            let load = combine_load(left, right)?;
            if finite(load[0] + load[2])? > capacity {
                return Ok(None);
            }
        }
        if !self.budgets_join(f, b)? {
            return Ok(None);
        }
        let mut front_correction = 0.;
        for (left, right) in f.fronts.iter().zip(&b.fronts) {
            let Some(joined) = left.joined_minimum(right)? else {
                return Ok(None);
            };
            front_correction = finite(
                front_correction + finite(finite(joined - left.minimum())? - right.minimum())?,
            )?;
        }
        let mut workspace = CoupledWorkspace::default();
        for ((left, right), prepared) in f.relations.iter().zip(&b.relations).zip(&self.paired) {
            if !paired::domains_may_intersect(left, right, self.front_bucket_policy) {
                return Ok(None);
            }
            let joined = match prepared.cached_join(left, right) {
                Some(Some(value)) => value,
                Some(None) => return Ok(None),
                None => {
                    let Some(value) = prepared.replay_suffix(
                        left,
                        &b.suffix.as_ref().unwrap().program,
                        &mut workspace.relation,
                    )?
                    else {
                        return Ok(None);
                    };
                    value
                }
            };
            if self.cost_scale != 0. {
                let baseline = left
                    .forward()
                    .ok_or_else(|| invalid("paired forward join expected"))?
                    .minimum();
                // Feasibility-only paired relations have no objective.
                if prepared.has_costs() {
                    front_correction = finite(
                        front_correction + finite(self.cost_scale * finite(joined - baseline)?)?,
                    )?;
                }
            }
        }
        let suffix = b.suffix.as_ref().unwrap();
        for (front, plugin) in f.plugins.iter().zip(&self.plugins) {
            if !plugin.0.join(front, &mut suffix.program.iter())? {
                return Ok(None);
            }
        }
        let mut state_cost = 0.;
        if self.transitions.deferred {
            let mut numeric = f.numeric.clone();

            for arc in suffix.program.iter() {
                for operation in &self.transitions.arcs[arc].operations {
                    let scale =
                        if matches!(operation, transitions::NumericOperation::StateCost { .. }) {
                            self.cost_scale
                        } else {
                            0.
                        };
                    if !operation.apply(&mut numeric, scale, &mut state_cost)?
                        || !operation.feasible(&numeric)
                    {
                        return Ok(None);
                    }
                }
            }
        }
        for bound in &suffix.bounds {
            let value = finite(bound.offset + dot(&bound.coefficients, &f.numeric)?)?;
            if value < bound.bounds.lower || value > bound.bounds.upper {
                return Ok(None);
            }
        }
        let entry_cost = dot(&suffix.coefficients, &f.numeric)?;
        let baseline = dot(&suffix.coefficients, &self.initial)?;
        Ok(Some(finite(
            finite(finite(entry_cost - baseline)? + state_cost)? + front_correction,
        )?))
    }
    /// Physical suffix arcs in forward traversal order. Charged resources
    /// retain this chain even when their numeric transition is affine.
    pub(crate) fn suffix_arcs<'a>(
        &self,
        b: &'a CoupledLabelState,
    ) -> impl Iterator<Item = usize> + 'a {
        b.suffix
            .as_ref()
            .into_iter()
            .flat_map(|suffix| suffix.program.iter())
    }
    /// Return the complete joined resource objective directly. Reconstructing
    /// it from two rounded minima plus a correction loses scalar information.
    pub(crate) fn join_resource_objective(
        &self,
        f: &CoupledLabelState,
        b: &CoupledLabelState,
    ) -> Result<Option<f64>> {
        self.check_state(f)?;
        self.check_state(b)?;
        if f.suffix.is_some() || b.suffix.is_none() {
            return Err(invalid("coupled join direction mismatch"));
        }
        if f.selection
            .iter()
            .zip(&b.selection)
            .any(|(a, b)| a & b == 0)
        {
            return Ok(None);
        }
        for ((&left, &right), &capacity) in f.loads.iter().zip(&b.loads).zip(&self.load_capacities)
        {
            let load = combine_load(left, right)?;
            if finite(load[0] + load[2])? > capacity {
                return Ok(None);
            }
        }
        if !self.budgets_join(f, b)? {
            return Ok(None);
        }
        // Numeric-only canonical joins retain one state vector and literal
        // cost accumulator. Rebuilding a boxed physical label per suffix arc
        // would only copy already-checked selection and load summaries.
        if self.front_indices.is_empty()
            && !self.has_slope_fronts()
            && self.relation_indices.is_empty()
            && self.plugins.is_empty()
        {
            let mut numeric = f.numeric.clone();
            let mut cost = f.numeric_cost;
            for arc in self.suffix_arcs(b) {
                let prepared = &self.transitions.arcs[arc];
                if prepared.impossible {
                    return Ok(None);
                }
                for operation in &prepared.operations {
                    if !operation.apply(&mut numeric, self.cost_scale, &mut cost)?
                        || !operation.feasible(&numeric)
                    {
                        return Ok(None);
                    }
                }
            }
            return Ok(Some(cost));
        }
        // In the restricted exact bounded-cost specialization the cached
        // final dependent minimum is already canonical; avoid suffix replay.
        if self.front_indices.is_empty()
            && !self.has_slope_fronts()
            && self.initial.is_empty()
            && !self.paired.is_empty()
            && self.plugins.is_empty()
        {
            let mut total = 0.;
            let mut known = true;
            for ((left, right), prepared) in f.relations.iter().zip(&b.relations).zip(&self.paired)
            {
                match prepared.cached_join(left, right) {
                    Some(Some(final_minimum)) => {
                        total = finite(
                            total + prepared.final_objective(final_minimum, self.cost_scale)?,
                        )?
                    }
                    Some(None) => return Ok(None),
                    None => {
                        known = false;
                        break;
                    }
                }
            }
            if known {
                return Ok(Some(total));
            }
        }
        let mut workspace = CoupledWorkspace::default();
        let mut current: Option<Box<CoupledLabelState>> = None;
        for arc in self.suffix_arcs(b) {
            let input = current.as_deref().unwrap_or(f);
            let Some((next, _)) =
                self.extend_internal(input, arc, true, false, false, &mut workspace)?
            else {
                return Ok(None);
            };
            current = Some(next);
        }
        Ok(Some(current.as_deref().unwrap_or(f).resource_objective))
    }
}
