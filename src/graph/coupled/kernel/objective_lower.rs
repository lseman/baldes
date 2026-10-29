//! Polynomial directed objective relaxation over bounded physical walks.
//! Correlations between state coordinates are forgotten; no feasible walk is
//! removed. This is independent of optimized resource-front witnesses.
use super::*;
use bounds::Range;
use std::time::Instant;
use transitions::NumericOperation;

#[derive(Clone)]
struct State {
    numeric: Vec<Range>,
    cost: Range,
    fronts: Vec<Range>,
    paired: Vec<Range>,
}

fn take(work: &mut usize, count: usize, deadline: Instant) -> Option<()> {
    if Instant::now() >= deadline {
        return None;
    }
    *work = work.checked_sub(count)?;
    Some(())
}
fn finite(range: Range) -> Option<Range> {
    range.finite().then_some(range)
}
fn clip(range: Range, window: Window) -> Option<Range> {
    let result = Range {
        lo: range.lo.max(window.lower),
        hi: range.hi.min(window.upper),
    };
    (result.lo <= result.hi).then_some(result)
}
fn hull(a: &mut Range, b: Range) {
    a.lo = a.lo.min(b.lo);
    a.hi = a.hi.max(b.hi);
}
impl State {
    fn same_bits(&self, other: &Self) -> bool {
        let same = |a: &Range, b: &Range| {
            a.lo.to_bits() == b.lo.to_bits() && a.hi.to_bits() == b.hi.to_bits()
        };
        same(&self.cost, &other.cost)
            && [&self.numeric, &self.fronts, &self.paired]
                .into_iter()
                .zip([&other.numeric, &other.fronts, &other.paired])
                .all(|(a, b)| a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(a, b)))
    }
    fn merge(&mut self, other: Self) {
        for (a, b) in self.numeric.iter_mut().zip(other.numeric) {
            hull(a, b);
        }
        hull(&mut self.cost, other.cost);
        for (a, b) in self.fronts.iter_mut().zip(other.fronts) {
            hull(a, b);
        }
        for (a, b) in self.paired.iter_mut().zip(other.paired) {
            hull(a, b);
        }
    }
    fn minimum(&self, paired_parameters: &[(Range, f64)]) -> Option<f64> {
        let mut objective = self.cost;
        for &front in &self.fronts {
            objective = finite(objective.add(front))?;
        }
        for (&paired, &(baseline, scale)) in self.paired.iter().zip(paired_parameters) {
            let charge = finite(finite(paired.subtract(baseline))?.multiply(Range::point(scale)))?;
            objective = finite(objective.add(charge))?;
        }
        Some(objective.lo)
    }
}
impl CoupledKernel {
    /// Directed pricing-resource bound on source/sink walks of at most
    /// `max_arcs` arcs. A larger original walk domain needs its own justified
    /// hop bound; this method does not establish one.
    pub fn pricing_objective_walk_lower_bound<const D: usize>(
        &self,
        graph: &Graph<D>,
        max_arcs: usize,
        work: &mut usize,
        deadline: Instant,
    ) -> Option<f64> {
        let mut lower = self.objective_walk_lower_bound(graph, max_arcs, work, deadline)?;
        for activity in self.pricing_activities.iter() {
            lower = crate::core::rounding::add_down(
                lower,
                activity.objective_walk_lower_bound(graph, max_arcs, work, deadline)?,
            )?;
        }
        Some(lower)
    }
    /// Lower bound for resource objective on every source/sink walk of at
    /// most `max_arcs` arcs. The caller must justify that hop limit for its
    /// original domain. Plugins/feasibility restrictions are safely dropped.
    pub(crate) fn objective_walk_lower_bound<const D: usize>(
        &self,
        graph: &Graph<D>,
        max_arcs: usize,
        work: &mut usize,
        deadline: Instant,
    ) -> Option<f64> {
        self.objective_walk_bound(graph, max_arcs, work, deadline)?
    }

    /// Outer None is unknown; Some(None) proves the complete bounded walk
    /// relaxation empty. A finite lower bound need not be attained. All
    /// rejected interval states have directed feasibility proofs, while an
    /// unavailable arithmetic operation aborts the entire proof as unknown.
    pub(crate) fn objective_walk_bound<const D: usize>(
        &self,
        graph: &Graph<D>,
        max_arcs: usize,
        work: &mut usize,
        deadline: Instant,
    ) -> Option<Option<f64>> {
        take(
            work,
            graph.vertices.len().checked_add(self.initial.len())?,
            deadline,
        )?;
        if self.cost_scale == 0. || !self.has_costs() {
            return Some(Some(0.));
        }
        if max_arcs == 0 {
            return Some((graph.source == graph.sink).then_some(0.));
        }
        // Numerical slope-front operators need their own directed semantics.
        // Do not reuse their computed minima as original-domain bounds.
        if self.has_machine_cost_contracts() || self.plugins.iter().any(|p| p.0.has_costs()) {
            return None;
        }
        // Keep the same family order as final_resource_objective: raw pairs
        // precede knot pairs. Each nonempty walk's paired charge is its final
        // value, normalized once, rather than a sum of per-arc minima.
        let mut paired_resources = Vec::new();
        take(work, self.definitions.len(), deadline)?;
        for (resource, definition) in self.definitions.iter().enumerate() {
            if let CoupledResource::SlopePaired { mode, policy, .. } = definition {
                if *mode != PairedResourceMode::Feasibility
                    && self.objective_coefficients[resource] != 0.
                {
                    // Front-algebra arithmetic needs its own directed proof.
                    if *policy != FrontOperationPolicy::StrictDomain {
                        return None;
                    }
                    paired_resources.push(resource);
                }
            }
        }
        for &resource in &self.relation_indices {
            if self.objective_coefficients[resource] != 0.
                && matches!(self.definitions[resource],
                    CoupledResource::PairedPiecewiseLinear { mode, .. }
                    if mode != PairedResourceMode::Feasibility)
            {
                paired_resources.push(resource);
            }
        }
        let cache_size = if paired_resources.is_empty() {
            0
        } else {
            graph.arcs.len()
        };
        take(
            work,
            paired_resources.len().checked_add(cache_size)?,
            deadline,
        )?;
        let mut paired_parameters = Vec::with_capacity(paired_resources.len());
        let mut paired_initial = Vec::with_capacity(paired_resources.len());
        for &resource in &paired_resources {
            let normalization = match &self.definitions[resource] {
                CoupledResource::SlopePaired { normalization, .. }
                | CoupledResource::PairedPiecewiseLinear { normalization, .. } => *normalization,
                _ => return None,
            };
            let baseline = if normalization == PairedCostNormalization::FinalMinusInitialMinimum {
                let [lo, hi] = self.paired_initial_minimum(resource, work, Some(deadline))?;
                finite(Range { lo, hi })?
            } else {
                Range::point(0.)
            };
            let scale = self.cost_scale * self.objective_coefficients[resource];
            if !scale.is_finite() {
                return None;
            }
            paired_parameters.push((baseline, scale));
            let [lo, hi] = self.paired_terminal_bounds(resource, &[], work, Some(deadline))?;
            paired_initial.push(finite(Range { lo, hi })?);
        }
        // Transfer bounds depend only on the physical arc, not the hop count.
        // Populate lazily so unreachable physical arcs consume no function
        // scans, then share the same enclosure across all cyclic revisits.
        let mut paired_by_arc: Vec<Option<Vec<certificate::PairedArcBounds>>> =
            vec![None; cache_size];
        let root = State {
            numeric: self.initial.iter().copied().map(Range::point).collect(),
            cost: Range::point(0.),
            fronts: vec![Range::point(0.); self.front_indices.len()],
            paired: paired_initial,
        };
        let mut previous = vec![None; graph.vertices.len()];
        previous[graph.source] = Some(root);
        let mut minimum = (graph.source == graph.sink).then_some(0.);
        for step in 0..max_arcs {
            take(work, graph.vertices.len(), deadline)?;
            let mut next: Vec<Option<State>> = vec![None; graph.vertices.len()];
            for (index, arc) in graph.arcs.iter().enumerate() {
                take(work, 1, deadline)?;
                let Some(old) = &previous[arc.tail] else {
                    continue;
                };
                take(
                    work,
                    old.numeric
                        .len()
                        .checked_add(old.fronts.len())?
                        .checked_add(old.paired.len())?
                        .checked_add(1)?,
                    deadline,
                )?;
                let mut state = old.clone();
                let mut possible = true;
                for operation in &self.transitions.arcs.get(index)?.operations {
                    take(work, 1, deadline)?;
                    match operation {
                        NumericOperation::Cumulative {
                            slot,
                            increment,
                            weight,
                            cost_coefficient,
                        } => {
                            state.numeric[*slot] =
                                finite(state.numeric[*slot].add(Range::point(*increment)))?;
                            if *cost_coefficient != 0. {
                                let charge =
                                    finite(state.numeric[*slot].multiply(Range::point(*weight)))?;
                                state.cost = finite(
                                    state.cost.add(finite(
                                        charge
                                            .multiply(Range::point(*cost_coefficient))
                                            .multiply(Range::point(self.cost_scale)),
                                    )?),
                                )?;
                            }
                        }
                        NumericOperation::Dependent {
                            slot,
                            parent,
                            offset,
                            factor,
                            bounds,
                        } => {
                            let delta = finite(Range::point(*offset).add(finite(
                                state.numeric[*parent].multiply(Range::point(*factor)),
                            )?))?;
                            let Some(value) =
                                clip(finite(state.numeric[*slot].add(delta))?, *bounds)
                            else {
                                possible = false;
                                break;
                            };
                            state.numeric[*slot] = value;
                        }
                        NumericOperation::BaseMap {
                            slot,
                            bounds,
                            function,
                        }
                        | NumericOperation::PiecewiseLinear {
                            slot,
                            parent: _,
                            bounds,
                            function,
                        } => {
                            let parent = match operation {
                                NumericOperation::PiecewiseLinear { parent, .. } => *parent,
                                _ => *slot,
                            };
                            // None is unknown, not a proof of an empty arc.
                            let image = certificate::encoded_image(
                                function,
                                state.numeric[parent],
                                work,
                                Some(deadline),
                            )?;
                            let image = finite(image)?;
                            let value = if matches!(operation, NumericOperation::BaseMap { .. }) {
                                image
                            } else {
                                finite(state.numeric[*slot].add(image))?
                            };
                            let Some(value) = clip(value, *bounds) else {
                                possible = false;
                                break;
                            };
                            state.numeric[*slot] = value;
                        }
                        NumericOperation::StateCost { parent, function } => {
                            let image = certificate::encoded_image(
                                function,
                                state.numeric[*parent],
                                work,
                                Some(deadline),
                            )?;
                            let charge =
                                finite(finite(image)?.multiply(Range::point(self.cost_scale)))?;
                            state.cost = finite(state.cost.add(charge))?;
                        }
                    }
                }
                if !possible {
                    continue;
                }
                for (front, &definition) in state.fronts.iter_mut().zip(&self.front_indices) {
                    let CoupledResource::IntervalCost { functions, .. } =
                        &self.definitions[definition]
                    else {
                        return None;
                    };
                    let function = functions.get(index)?;
                    take(work, function.points.len(), deadline)?;
                    let domain = function.prepared_domain();
                    let charge = finite(function.range(Range {
                        lo: domain.lower,
                        hi: domain.upper,
                    })?)?;
                    *front =
                        finite(front.add(finite(charge.multiply(Range::point(self.cost_scale)))?))?;
                }
                if !paired_resources.is_empty() {
                    if paired_by_arc[index].is_none() {
                        take(work, paired_resources.len(), deadline)?;
                        let mut transfers = Vec::with_capacity(paired_resources.len());
                        for &resource in &paired_resources {
                            transfers.push(self.paired_arc_source_bounds(
                                resource,
                                index,
                                work,
                                Some(deadline),
                            )?);
                        }
                        paired_by_arc[index] = Some(transfers);
                    }
                    take(work, paired_resources.len(), deadline)?;
                    for (value, transfer) in
                        state.paired.iter_mut().zip(paired_by_arc[index].as_ref()?)
                    {
                        match transfer.apply(*value)? {
                            Some(next) => *value = next,
                            None => {
                                possible = false;
                                break;
                            }
                        }
                    }
                }
                if !possible {
                    continue;
                }
                if let Some(old) = &mut next[arc.head] {
                    old.merge(state);
                } else {
                    next[arc.head] = Some(state);
                }
            }
            if let Some(state) = &next[graph.sink] {
                take(work, paired_resources.len(), deadline)?;
                let lower = state.minimum(&paired_parameters)?;
                minimum = Some(minimum.map_or(lower, |old: f64| old.min(lower)));
            }
            if next.iter().all(Option::is_none) {
                break;
            }
            if step + 1 < max_arcs {
                let mut fixed = true;
                for (a, b) in previous.iter().zip(&next) {
                    take(work, 1, deadline)?;
                    match (a, b) {
                        (None, None) => {}
                        (Some(a), Some(b)) => {
                            take(
                                work,
                                a.numeric
                                    .len()
                                    .checked_add(a.fronts.len())?
                                    .checked_add(a.paired.len())?
                                    .checked_add(1)?,
                                deadline,
                            )?;
                            if !a.same_bits(b) {
                                fixed = false;
                                break;
                            }
                        }
                        _ => {
                            fixed = false;
                            break;
                        }
                    }
                }
                // The deterministic next-row operation has no hop-dependent
                // terms. Identical directed states therefore prove all later
                // rows identical too, including their terminal objective.
                if fixed {
                    break;
                }
            }
            previous = next;
        }
        take(work, 0, deadline)?;
        Some(minimum)
    }
}
#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/objective_lower_tests.rs"]
mod tests;
