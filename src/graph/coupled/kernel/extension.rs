//! Directional label creation, physical-arc extension, and path replay.
use super::*;

impl CoupledKernel {
    pub fn root(&self, forward: bool) -> Box<CoupledLabelState> {
        Box::new(CoupledLabelState {
            selection: StateValues::from_slice(&self.selection_initial),
            loads: StateValues::filled(self.load_capacities.len(), [0.; 3]),
            budgets: budgeted::PreparedBudgeted::root(&self.budgeted, forward),
            reversal_peaks: None,
            numeric: StateValues::from_slice(&self.initial),
            slope_fronts: self
                .slope_roots
                .iter()
                .map(|root| {
                    if forward {
                        root.clone()
                    } else {
                        slope_interval::SlopeLabel {
                            front: None,
                            minimum: 0.,
                        }
                    }
                })
                .collect(),
            slope_relations: self
                .slope_paired
                .iter()
                .map(|paired| paired.root(forward))
                .collect(),
            numeric_cost: 0.,
            resource_objective: 0.,
            accounting: Some(accounting::ScalarCosts::default()),
            relations: self.paired.iter().map(|p| p.root(forward)).collect(),
            charging: self.charging.iter().map(|p| p.root(forward)).collect(),
            fronts: self
                .front_indices
                .iter()
                .map(|&i| match &self.definitions[i] {
                    CoupledResource::IntervalCost {
                        initial, bounds, ..
                    } => front::CostFront::zero(if forward { *initial } else { *bounds }),
                    _ => unreachable!(),
                })
                .collect(),
            suffix: (!forward).then(|| {
                Box::new(Suffix {
                    bounds: vec![],
                    coefficients: StateValues::filled(self.initial.len(), 0.),
                    constant: 0.,
                    program: Default::default(),
                    state_costs: Default::default(),
                })
            }),
        })
    }
    /// Zero-compensation dominance with directional load ordering and equal
    /// numeric entry conditions. Use `dominance_penalty` for cost tradeoffs.
    pub fn dominates(a: Option<&CoupledLabelState>, b: Option<&CoupledLabelState>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                let account_safe = a.numeric_cost <= b.numeric_cost
                    && match (a.accounting, b.accounting) {
                        (Some(a_cost), Some(b_cost)) => a_cost.reduced <= b_cost.reduced,
                        (None, None) => true,
                        _ => false,
                    };
                account_safe
                    && dominance::loads_dominate(a, b)
                    // Entry kinds are unknown here; equality is order-free.
                    && a.budgets == b.budgets
                    && a.fronts == b.fronts
                    && a.slope_fronts == b.slope_fronts
                    && a.slope_relations == b.slope_relations
                    && a.relations == b.relations
                    && a.charging == b.charging
                    && a.numeric == b.numeric
                    && a.suffix == b.suffix
                    && a.selection.len() == b.selection.len()
                    && a.selection
                        .iter()
                        .zip(&b.selection)
                        .all(|(a, b)| a & b == *b)
            }
            _ => false,
        }
    }
    /// Extend a physical arc using the necessary complete-route filter. The
    /// returned increment belongs to both reduced and original objectives.
    /// Arithmetic failure is distinct from infeasibility. Standalone partial
    /// paths can be evaluated separately with `replay` or `replay_state`.
    pub fn extend(
        &self,
        state: &CoupledLabelState,
        arc: usize,
        forward: bool,
    ) -> Result<Option<(Box<CoupledLabelState>, f64)>> {
        self.extend_with_workspace(state, arc, forward, &mut CoupledWorkspace::default())
    }
    pub fn extend_with_workspace(
        &self,
        state: &CoupledLabelState,
        arc: usize,
        forward: bool,
        workspace: &mut CoupledWorkspace,
    ) -> Result<Option<(Box<CoupledLabelState>, f64)>> {
        self.extend_internal(state, arc, forward, true, false, workspace)
    }
    pub(super) fn extend_internal(
        &self,
        state: &CoupledLabelState,
        arc: usize,
        forward: bool,
        route_filter: bool,
        reversal_union: bool,
        workspace: &mut CoupledWorkspace,
    ) -> Result<Option<(Box<CoupledLabelState>, f64)>> {
        if arc >= self.arc_count {
            return Err(invalid("unknown coupled resource arc"));
        }
        self.check_state(state)?;
        if forward == state.suffix.is_some() {
            return Err(invalid("coupled label direction mismatch"));
        }
        if self.transitions.arcs[arc].impossible
            || (route_filter && !self.route_domains.possible[arc])
        {
            return Ok(None);
        }
        // The old suffix is read below while its replacement is composed.
        // Cloning it here would copy every old entry condition only to discard it.
        let mut next = CoupledLabelState {
            selection: state.selection.clone(),
            loads: state.loads.clone(),
            budgets: state.budgets.clone(),
            reversal_peaks: state.reversal_peaks.clone(),
            numeric: state.numeric.clone(),
            suffix: None,
            fronts: Vec::with_capacity(state.fronts.len()),
            slope_fronts: Vec::with_capacity(state.slope_fronts.len()),
            slope_relations: Vec::with_capacity(state.slope_relations.len()),
            relations: Vec::with_capacity(state.relations.len()),
            charging: Vec::with_capacity(state.charging.len()),
            numeric_cost: 0.,
            resource_objective: 0.,
            accounting: None,
        };
        let prepared = &self.transitions.arcs[arc];
        for (domain, &allowed) in next.selection.iter_mut().zip(&prepared.selection) {
            *domain &= allowed;
            if *domain == 0 {
                return Ok(None);
            }
        }
        for (index, ((load, &step), &capacity)) in next
            .loads
            .iter_mut()
            .zip(&prepared.loads)
            .zip(&self.load_capacities)
            .enumerate()
        {
            if let Some(peaks) = &mut next.reversal_peaks {
                peaks[index] = finite(peaks[index] + step[1])?.max(0.);
            }
            *load = if forward {
                combine_load(*load, step)?
            } else {
                combine_load(step, *load)?
            };
            if finite(load[0] + load[2])? > capacity
                && !(reversal_union
                    && forward
                    && next
                        .reversal_peaks
                        .as_ref()
                        .is_some_and(|peaks| load[0] + peaks[index] <= capacity))
            {
                return Ok(None);
            }
        }
        if !self.budgets_extend(&mut next.budgets, arc, forward)? {
            return Ok(None);
        }
        for (old, &i) in state.fronts.iter().zip(&self.front_indices) {
            let CoupledResource::IntervalCost {
                bounds,
                increments,
                functions,
                ..
            } = &self.definitions[i]
            else {
                unreachable!()
            };
            let Some(front) = old.extend(
                increments[arc],
                &functions[arc],
                *bounds,
                forward,
                self.cost_scale,
            )?
            else {
                return Ok(None);
            };
            next.fronts.push(front);
        }
        for ((old, &index), charges) in state
            .slope_fronts
            .iter()
            .zip(&self.slope_indices)
            .zip(&self.slope_charges)
        {
            let Some(front) = old.extend(
                &self.definitions[index],
                &charges[arc],
                arc,
                forward,
                self.cost_scale,
            )?
            else {
                return Ok(None);
            };
            next.slope_fronts.push(front);
        }
        for (old, prepared) in state.slope_relations.iter().zip(&self.slope_paired) {
            let Some(front) = prepared.extend(old, arc, forward, self.cost_scale)? else {
                return Ok(None);
            };
            next.slope_relations.push(front);
        }
        let first_backward = state
            .suffix
            .as_ref()
            .is_some_and(|s| s.program.iter().next().is_none());
        for (old, prepared) in state.relations.iter().zip(&self.paired) {
            let Some(label) =
                prepared.extend(old, arc, forward, first_backward, &mut workspace.relation)?
            else {
                return Ok(None);
            };
            next.relations.push(label);
        }
        for (old, prepared) in state.charging.iter().zip(&self.charging) {
            let Some(label) = prepared.extend(old, arc, forward)? else {
                return Ok(None);
            };
            next.charging.push(label);
        }
        let cost = if forward {
            let mut cost = state.numeric_cost;
            for operation in &self.transitions.arcs[arc].operations {
                if !operation.apply(&mut next.numeric, self.cost_scale, &mut cost)?
                    || !operation.feasible(&next.numeric)
                {
                    return Ok(None);
                }
            }
            cost
        } else {
            let old = state.suffix.as_ref().unwrap();
            let n = self.initial.len();
            let transition = self.transitions.arcs[arc]
                .backward
                .as_ref()
                .ok_or_else(|| invalid("nonfinite coupled resource arithmetic"))?;
            let matrix = &transition.matrix;
            let offsets = &transition.offsets;
            let mut coefficients = StateValues::<f64, 4>::from_slice(&transition.coefficients);
            let mut constant = transition.constant;
            let mut bounds_out = transition.bounds.clone();
            // Prepend this arc to the suffix: transform every entry condition
            // and its cost function through the arc's triangular affine map.
            for bound in &old.bounds {
                let mut row = StateValues::<f64, 4>::filled(n, 0.);
                for (j, value) in row.iter_mut().enumerate() {
                    for (i, entry) in matrix.iter().enumerate() {
                        *value = finite(*value + finite(bound.coefficients[i] * entry[j])?)?;
                    }
                }
                bounds_out.push(AffineBound {
                    coefficients: row,
                    offset: finite(bound.offset + dot(&bound.coefficients, offsets)?)?,
                    bounds: bound.bounds,
                });
            }
            if self.transitions.deferred {
                for (value, &previous) in coefficients.iter_mut().zip(&old.coefficients) {
                    *value = finite(*value + previous)?;
                }
            } else {
                for (j, value) in coefficients.iter_mut().enumerate() {
                    for (i, entry) in matrix.iter().enumerate() {
                        *value = finite(*value + finite(old.coefficients[i] * entry[j])?)?;
                    }
                }
            }
            constant = finite(constant + finite(old.constant + dot(&old.coefficients, offsets)?)?)?;
            let new_cost = finite(constant + dot(&coefficients, &self.initial)?)?;
            let state_costs = self.transitions.arcs[arc]
                .state_costs
                .combine(&old.state_costs);
            next.suffix = Some(Box::new(Suffix {
                bounds: bounds_out,
                state_costs,
                coefficients,
                constant,
                program: if self.transitions.deferred
                    || !self.relation_indices.is_empty()
                    || self.cost_resource
                    || !self.slope_paired.is_empty()
                    || !self.charging.is_empty()
                {
                    old.program.prepend(arc)
                } else {
                    Default::default()
                },
            }));
            new_cost
        };
        next.numeric_cost = cost;
        let mut objective = next.numeric_cost;
        for front in &next.fronts {
            objective = finite(objective + front.minimum())?;
        }
        for front in &next.slope_fronts {
            objective = finite(objective + front.minimum)?;
        }
        for front in &next.slope_relations {
            objective = finite(objective + front.minimum)?;
        }
        for (prepared, label) in self.paired.iter().zip(&next.relations) {
            objective = finite(objective + prepared.objective(label, self.cost_scale)?)?;
        }
        next.resource_objective = objective;
        Ok(Some((
            Box::new(next),
            finite(objective - state.resource_objective)?,
        )))
    }
    /// Join at one physical vertex. All arc services are accounted for once.
    pub fn replay(&self, arcs: &[usize]) -> Result<Option<f64>> {
        Ok(self.replay_state(arcs)?.map(|(_, cost)| cost))
    }
    /// Replay a physical arc-position path and return its final coupled state.
    pub fn replay_state(&self, arcs: &[usize]) -> Result<Option<(Box<CoupledLabelState>, f64)>> {
        let mut state = self.root(true);
        let mut workspace = CoupledWorkspace::default();
        for &arc in arcs {
            let Some((next, _)) =
                self.extend_internal(&state, arc, true, false, false, &mut workspace)?
            else {
                return Ok(None);
            };
            state = next;
        }
        let cost = state.resource_objective;
        Ok(Some((state, cost)))
    }
}
