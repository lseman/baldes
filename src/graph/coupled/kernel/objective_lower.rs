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
    fn merge(&mut self, other: Self) {
        for (a, b) in self.numeric.iter_mut().zip(other.numeric) {
            hull(a, b);
        }
        hull(&mut self.cost, other.cost);
        for (a, b) in self.fronts.iter_mut().zip(other.fronts) {
            hull(a, b);
        }
    }
    fn minimum(&self) -> Option<f64> {
        let mut objective = self.cost;
        for &front in &self.fronts {
            objective = finite(objective.add(front))?;
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
        take(
            work,
            graph.vertices.len().checked_add(self.initial.len())?,
            deadline,
        )?;
        if self.cost_scale == 0. || !self.has_costs() {
            return Some(0.);
        }
        // Numerical slope-front operators need their own directed semantics.
        // Do not reuse their computed minima as original-domain bounds.
        if self.has_machine_cost_contracts()
            || self.plugins.iter().any(|p| p.0.has_costs())
            || self.paired.iter().any(|p| p.has_costs())
            || self.slope_paired.iter().any(|p| p.has_costs())
        {
            return None;
        }
        let root = State {
            numeric: self.initial.iter().copied().map(Range::point).collect(),
            cost: Range::point(0.),
            fronts: vec![Range::point(0.); self.front_indices.len()],
        };
        let mut previous = vec![None; graph.vertices.len()];
        previous[graph.source] = Some(root);
        let mut minimum = (graph.source == graph.sink).then_some(0.);
        for _ in 0..max_arcs {
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
                if let Some(old) = &mut next[arc.head] {
                    old.merge(state);
                } else {
                    next[arc.head] = Some(state);
                }
            }
            if let Some(state) = &next[graph.sink] {
                let lower = state.minimum()?;
                minimum = Some(minimum.map_or(lower, |old: f64| old.min(lower)));
            }
            if next.iter().all(Option::is_none) {
                break;
            }
            previous = next;
        }
        take(work, 0, deadline)?;
        minimum
    }
}
#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/objective_lower_tests.rs"]
mod tests;
