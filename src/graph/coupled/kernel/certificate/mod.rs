//! Outward interval replay for integer witnesses, distinct from tolerant pricing.
use super::*;
mod encoded;
mod fronts;
mod machine;
mod operators;
mod paired;
use crate::core::rounding::{add_down, add_up, multiply_down};
use std::time::Instant;

const DEFAULT_CERTIFICATE_WORK: usize = 1_000_000;

fn debit(work: &mut usize, amount: usize, deadline: Option<Instant>) -> Option<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return None;
    }
    *work = work.checked_sub(amount)?;
    Some(())
}

/// Outward image of the accepted encoded arguments, sharing the caller's proof
/// budget. `None` is unknown or unavailable, never a global emptiness proof.
/// Route feasibility callers must separately prove full argument acceptance.
pub(super) fn encoded_image(
    function: &crate::graph::coupled::EncodedFunction,
    argument: bounds::Range,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<bounds::Range> {
    encoded::image(function, argument, work, deadline)
}

/// Ordered source-encoding image over the strict-domain portion of `argument`.
/// This does not certify arbitrary raw front-algebra operators.
pub(super) fn slope_function_range(
    function: &crate::graph::coupled::PreparedSlopeFunction,
    argument: bounds::Range,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<bounds::Range> {
    paired::slope_range(function, argument, work, deadline)
}

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}
impl Interval {
    fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    fn add(self, b: Self) -> Option<Self> {
        Some(Self {
            lo: add_down(self.lo, b.lo)?,
            hi: add_up(self.hi, b.hi)?,
        })
    }
    fn multiply(self, b: f64) -> Option<Self> {
        let (a, c) = if b >= 0. {
            (self.lo, self.hi)
        } else {
            (self.hi, self.lo)
        };
        Some(Self {
            lo: multiply_down(a, b)?,
            hi: -multiply_down(-c, b)?,
        })
    }
    fn max(self, b: Self) -> Self {
        Self {
            lo: self.lo.max(b.lo),
            hi: self.hi.max(b.hi),
        }
    }
    /// Some(true) when every value is at most `upper`, Some(false) when none is.
    fn at_most(self, upper: f64) -> Option<bool> {
        if self.hi <= upper {
            Some(true)
        } else if self.lo > upper {
            Some(false)
        } else {
            None
        }
    }
    fn within(self, b: Window) -> Option<bool> {
        if self.hi < b.lower || self.lo > b.upper {
            Some(false)
        } else if self.lo >= b.lower && self.hi <= b.upper {
            Some(true)
        } else {
            None
        }
    }
}
impl CoupledKernel {
    /// Independently check the alternate native machine-front model without
    /// changing this kernel's original-domain choice. This result cannot be
    /// substituted for a default graph's source-control feasibility proof.
    /// Use the explicit graph model setter to select that different domain.
    /// This checks resource operations; callers still validate physical path
    /// continuity, terminals and original master-row activities separately.
    pub fn certify_front_program(
        &self,
        arcs: &[usize],
        work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<super::FrontProgramCertificate> {
        if !self.has_machine_front_contracts() || arcs.iter().any(|&arc| arc >= self.arc_count) {
            return None;
        }
        debit(work, 1, deadline)?;
        let mut alternate = self.clone();
        alternate.front_program_domain = true;
        let (feasible, objective) = alternate.certify_with_budget(arcs, work, deadline)?;
        Some(super::FrontProgramCertificate {
            feasible,
            objective,
        })
    }

    /// A rigorously feasible/infeasible path and an outward objective interval.
    /// Arithmetic failure or a boundary-straddling interval remains unknown.
    pub(crate) fn certify(&self, arcs: &[usize]) -> Option<(bool, [f64; 2])> {
        let mut work = DEFAULT_CERTIFICATE_WORK;
        self.certify_with_budget(arcs, &mut work, None)
    }

    /// Share logical proof work and a cooperative deadline with a caller that
    /// certifies several routes. Exhaustion never produces an infeasibility or
    /// objective result. Failed attempts retain their consumed work charge.
    pub(crate) fn certify_with_budget(
        &self,
        arcs: &[usize],
        work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<(bool, [f64; 2])> {
        self.debit_replay(arcs, work, deadline)?;
        if self.cost_scale != 0.
            && self.definitions.iter().enumerate().any(|(r, definition)| {
                self.objective_coefficients[r] < 0.
                    && matches!(definition, CoupledResource::PairedFront { .. })
            })
        {
            // An unspecified minimum-Gamma choice is not a lower objective
            // bound when the model coefficient is negative.
            return None;
        }
        let result = self.certify_inner(arcs, work, deadline);
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            None
        } else {
            result
        }
    }

    /// Prove feasibility and enclose one custom/cumulative activity separately
    /// from the combined objective, regardless of model-cost suppression or
    /// pricing adjustments. Other descriptor families return Unknown.
    /// The complete coupled-resource domain is checked first; numerical replay is never
    /// promoted to a proof. Uses and respects the caller's shared work budget.
    pub fn certify_accumulated_activity(
        &self,
        resource: usize,
        arcs: &[usize],
        work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<(bool, [f64; 2])> {
        let (initial, increments, weights) = match self.definitions.get(resource)? {
            CoupledResource::CustomResource {
                increments,
                weights,
                ..
            } => (0., increments, weights),
            CoupledResource::CumulativeCost {
                initial,
                increments,
                weights,
                ..
            } => (*initial, increments, weights),
            _ => return None,
        };
        let mut feasibility;
        let proof = if self.has_negative_paired_objectives() && self.cost_scale != 0. {
            feasibility = self.clone();
            feasibility.suppress_costs();
            &feasibility
        } else {
            self
        };
        let (feasible, _) = proof.certify_with_budget(arcs, work, deadline)?;
        if !feasible {
            return Some((false, [0.; 2]));
        }
        debit(work, arcs.len().checked_mul(4)?.checked_add(1)?, deadline)?;
        let mut level = Interval::point(initial);
        let mut cost = Interval::point(0.);
        for &arc in arcs {
            level = level.add(Interval::point(*increments.get(arc)?))?;
            cost = cost.add(level.multiply(*weights.get(arc)?)?)?;
        }
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            None
        } else {
            Some((true, [cost.lo, cost.hi]))
        }
    }

    fn debit_replay(
        &self,
        arcs: &[usize],
        work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<()> {
        let base = arcs
            .len()
            .checked_add(1)?
            .checked_mul(self.definitions.len().checked_add(1)?)?;
        debit(work, base, deadline)?;
        for definition in self.definitions.iter() {
            let extra = match definition {
                CoupledResource::BudgetedLoad { budget, .. }
                | CoupledResource::BudgetedTime { budget, .. } => {
                    arcs.len().checked_mul(budget.checked_add(1)?)?
                }
                CoupledResource::SimultaneousPickupDelivery { .. } => arcs.len(),
                CoupledResource::PiecewiseLinearResource { functions, .. }
                | CoupledResource::PiecewiseLinearDependent { functions, .. }
                | CoupledResource::PiecewiseLinearCost { functions, .. } => {
                    arcs.iter().try_fold(0usize, |n, &arc| {
                        n.checked_add(functions.get(arc)?.points.len())
                    })?
                }
                CoupledResource::IntervalCost { functions, .. } => {
                    let points = arcs.iter().try_fold(0usize, |n, &arc| {
                        n.checked_add(functions.get(arc)?.points.len())
                    })?;
                    // Every retained knot and interval can evaluate each arc
                    // function. Sorting and translations are charged as well.
                    let knots = points.checked_add(4)?;
                    knots.checked_mul(points.checked_mul(2)?.checked_add(usize::BITS as usize)?)?
                }
                _ => 0,
            };
            debit(work, extra, deadline)?;
        }
        Some(())
    }

    fn certify_inner(
        &self,
        arcs: &[usize],
        work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<(bool, [f64; 2])> {
        // A feasibility-only plugin proof must never erase an objective.
        if self.plugins.iter().any(|p| p.0.has_costs()) {
            return None;
        }
        for plugin in &self.plugins {
            debit(work, arcs.len().checked_add(1)?, deadline)?;
            let before = *work;
            let result = plugin.0.certify(arcs, work, deadline);
            if *work > before {
                *work = before;
                return None;
            }
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                return None;
            }
            if !result? {
                return Some((false, [0.; 2]));
            }
        }
        if !self.selection_proper(arcs) {
            return Some((false, [0.; 2]));
        }
        let mut numeric: Vec<_> = self.initial.iter().copied().map(Interval::point).collect();
        let (valid, front_cost) = self.certify_fronts(arcs)?;
        if !valid {
            return Some((false, [0.; 2]));
        }
        let mut objective = Interval {
            lo: front_cost[0],
            hi: front_cost[1],
        };
        let (valid, slope_cost) = self.certify_slope_fronts(arcs, work, deadline)?;
        if !valid {
            return Some((false, [0.; 2]));
        }
        objective = objective.add(Interval {
            lo: slope_cost[0],
            hi: slope_cost[1],
        })?;
        let (valid, paired_cost) = self.certify_relations(arcs, work, deadline)?;
        if !valid {
            return Some((false, [0.; 2]));
        }
        objective = objective.add(Interval {
            lo: paired_cost[0],
            hi: paired_cost[1],
        })?;
        for r in self.definitions.iter() {
            match r {
                CoupledResource::Selection { .. } => {}
                CoupledResource::BudgetedLoad {
                    capacity,
                    budget,
                    demands,
                    deviations,
                } => {
                    let mut load = vec![Interval::point(0.); budget + 1];
                    for &arc in arcs {
                        let q = Interval::point(*demands.get(arc)?);
                        let worst = q.add(Interval::point(deviations[arc]))?;
                        for g in (1..load.len()).rev() {
                            load[g] = load[g].add(q)?.max(load[g - 1].add(worst)?);
                        }
                        load[0] = load[0].add(q)?;
                        if !load[*budget].at_most(*capacity)? {
                            return Some((false, [0.; 2]));
                        }
                    }
                }
                CoupledResource::BudgetedTime {
                    initial,
                    budget,
                    durations,
                    deviations,
                    windows,
                } => {
                    let mut start = vec![Interval::point(*initial); budget + 1];
                    for &arc in arcs {
                        let d = Interval::point(*durations.get(arc)?);
                        let worst = d.add(Interval::point(deviations[arc]))?;
                        let open = Interval::point(windows[arc].lower);
                        for g in (1..start.len()).rev() {
                            start[g] = start[g].add(d)?.max(start[g - 1].add(worst)?).max(open);
                        }
                        start[0] = start[0].add(d)?.max(open);
                        if !start[*budget].at_most(windows[arc].upper)? {
                            return Some((false, [0.; 2]));
                        }
                    }
                }
                CoupledResource::SimultaneousPickupDelivery {
                    capacity,
                    tolerance,
                    deliveries,
                    pickups,
                } => {
                    let mut load = Interval::point(0.);
                    for &arc in arcs {
                        load = load.add(Interval::point(*deliveries.get(arc)?))?;
                    }
                    let capacity = Window {
                        lower: 0.,
                        upper: capacity + tolerance,
                    };
                    if !load.within(capacity)? {
                        return Some((false, [0.; 2]));
                    }
                    for &arc in arcs {
                        load = load
                            .add(Interval::point(-deliveries[arc]))?
                            .add(Interval::point(pickups[arc]))?;
                        if !load.within(capacity)? {
                            return Some((false, [0.; 2]));
                        }
                    }
                }
                _ => {}
            }
        }
        for &arc in arcs {
            for (r, definition) in self.definitions.iter().enumerate() {
                match definition {
                    CoupledResource::EncodedPiecewiseLinearResource {
                        bounds, functions, ..
                    } => {
                        let i = self.slots[r]?;
                        let function = functions.get(arc)?;
                        if !numeric[i].within(function.prepared_domain())? {
                            return Some((false, [0.; 2]));
                        }
                        let value = encoded::image(
                            function,
                            bounds::Range {
                                lo: numeric[i].lo,
                                hi: numeric[i].hi,
                            },
                            work,
                            deadline,
                        )?;
                        numeric[i] = Interval {
                            lo: value.lo,
                            hi: value.hi,
                        };
                        if !numeric[i].within(*bounds)? {
                            return Some((false, [0.; 2]));
                        }
                    }
                    CoupledResource::EncodedPiecewiseLinearDependent {
                        parent,
                        bounds,
                        functions,
                        ..
                    } => {
                        let i = self.slots[r]?;
                        let p = self.slots[*parent]?;
                        let function = functions.get(arc)?;
                        if !numeric[p].within(function.prepared_domain())? {
                            return Some((false, [0.; 2]));
                        }
                        let value = encoded::image(
                            function,
                            bounds::Range {
                                lo: numeric[p].lo,
                                hi: numeric[p].hi,
                            },
                            work,
                            deadline,
                        )?;
                        numeric[i] = numeric[i].add(Interval {
                            lo: value.lo,
                            hi: value.hi,
                        })?;
                        if !numeric[i].within(*bounds)? {
                            return Some((false, [0.; 2]));
                        }
                    }
                    CoupledResource::EncodedPiecewiseLinearCost { parent, functions } => {
                        let p = self.slots[*parent]?;
                        let function = functions.get(arc)?;
                        if !numeric[p].within(function.prepared_domain())? {
                            return Some((false, [0.; 2]));
                        }
                        let value = encoded::image(
                            function,
                            bounds::Range {
                                lo: numeric[p].lo,
                                hi: numeric[p].hi,
                            },
                            work,
                            deadline,
                        )?;
                        if self.cost_scale != 0. {
                            objective = objective.add(
                                Interval {
                                    lo: value.lo,
                                    hi: value.hi,
                                }
                                .multiply(self.cost_scale)?,
                            )?;
                        }
                    }
                    CoupledResource::PiecewiseLinearResource {
                        bounds, functions, ..
                    } => {
                        let i = self.slots[r]?;
                        let function = functions.get(arc)?;
                        if !numeric[i].within(function.prepared_domain())? {
                            return Some((false, [0.; 2]));
                        }
                        let value = function.range(bounds::Range {
                            lo: numeric[i].lo,
                            hi: numeric[i].hi,
                        })?;
                        numeric[i] = Interval {
                            lo: value.lo,
                            hi: value.hi,
                        };
                        if !numeric[i].within(*bounds)? {
                            return Some((false, [0.; 2]));
                        }
                    }
                    CoupledResource::PiecewiseLinearCost { parent, functions } => {
                        let p = self.slots[*parent]?;
                        let function = functions.get(arc)?;
                        if !numeric[p].within(function.prepared_domain())? {
                            return Some((false, [0.; 2]));
                        }
                        if self.cost_scale != 0. {
                            let value = function.range(bounds::Range {
                                lo: numeric[p].lo,
                                hi: numeric[p].hi,
                            })?;
                            objective = objective.add(
                                Interval {
                                    lo: value.lo,
                                    hi: value.hi,
                                }
                                .multiply(self.cost_scale)?,
                            )?;
                        }
                    }
                    CoupledResource::CumulativeCost {
                        increments,
                        weights,
                        ..
                    }
                    | CoupledResource::CustomResource {
                        increments,
                        weights,
                        ..
                    } => {
                        let i = self.slots[r]?;
                        numeric[i] = numeric[i].add(Interval::point(*increments.get(arc)?))?;
                        if self.cost_scale != 0. && self.objective_coefficients[r] != 0. {
                            objective = objective.add(
                                numeric[i]
                                    .multiply(weights[arc])?
                                    .multiply(self.objective_coefficients[r])?
                                    .multiply(self.cost_scale)?,
                            )?;
                        }
                    }
                    CoupledResource::PiecewiseLinearDependent {
                        parent,
                        bounds,
                        functions,
                        ..
                    } => {
                        let i = self.slots[r]?;
                        let p = self.slots[*parent]?;
                        let function = functions.get(arc)?;
                        if !numeric[p].within(function.prepared_domain())? {
                            return Some((false, [0.; 2]));
                        }
                        let delta = function.range(bounds::Range {
                            lo: numeric[p].lo,
                            hi: numeric[p].hi,
                        })?;
                        numeric[i] = numeric[i].add(Interval {
                            lo: delta.lo,
                            hi: delta.hi,
                        })?;
                        if !numeric[i].within(*bounds)? {
                            return Some((false, [0.; 2]));
                        }
                    }
                    CoupledResource::Dependent {
                        parent,
                        bounds,
                        offsets,
                        factors,
                        ..
                    } => {
                        let i = self.slots[r]?;
                        let p = self.slots[*parent]?;
                        numeric[i] = numeric[i].add(
                            Interval::point(*offsets.get(arc)?)
                                .add(numeric[p].multiply(factors[arc])?)?,
                        )?;
                        if !numeric[i].within(*bounds)? {
                            return Some((false, [0.; 2]));
                        }
                    }
                    _ => {}
                }
            }
        }
        Some((true, [objective.lo, objective.hi]))
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/remaining_certificate_tests.rs"]
mod remaining_tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/operator_certificate_oracle_tests.rs"]
mod operator_oracle_tests;
