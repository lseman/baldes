//! Outward interval replay for integer witnesses, distinct from tolerant pricing.
use super::*;
mod fronts;
use crate::core::rounding::{add_down, add_up, multiply_down};
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
    /// A rigorously feasible/infeasible path and an outward objective interval.
    /// Arithmetic failure or a boundary-straddling interval remains unknown.
    pub(crate) fn certify(&self, arcs: &[usize]) -> Option<(bool, [f64; 2])> {
        // Literal slope descriptors do not yet provide directed interval
        // replay. Numerical execution must not become a certificate.
        if self.has_slope_fronts() {
            return None;
        }
        if self.definitions.iter().any(|r| {
            matches!(
                r,
                CoupledResource::EncodedPiecewiseLinearResource { .. }
                    | CoupledResource::EncodedPiecewiseLinearDependent { .. }
                    | CoupledResource::EncodedPiecewiseLinearCost { .. }
            )
        }) {
            return None;
        }
        // General relation optimization uses numerical intersection/waiting
        // envelopes; a numerical replay must not masquerade as a directed
        // integer feasibility or objective certificate, even in phase I.
        if !self.relation_indices.is_empty() || !self.plugins.is_empty() {
            return None;
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
        for r in self.definitions.iter() {
            match r {
                CoupledResource::Selection {
                    initial, allowed, ..
                } => {
                    let mut domain = *initial;
                    for &arc in arcs {
                        domain &= *allowed.get(arc)?;
                    }
                    if domain == 0 {
                        return Some((false, [0.; 2]));
                    }
                }
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
                    } => {
                        let i = self.slots[r]?;
                        numeric[i] = numeric[i].add(Interval::point(*increments.get(arc)?))?;
                        if self.cost_scale != 0. {
                            objective = objective.add(
                                numeric[i]
                                    .multiply(weights[arc])?
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
