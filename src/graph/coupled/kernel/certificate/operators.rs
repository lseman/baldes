//! Exact operator certificates for dyadic-lattice raw slope geometry.
//!
//! This does not reinterpret a general numerical front as source replay.
//! Bounded integer lattice coordinates/values and lattice slopes make endpoint
//! operations exact in binary64. Native lattices additionally have spacing
//! greater than its absolute tolerance; the scaled magnitude bound keeps the
//! relative tolerance below one lattice step.
use super::*;
use crate::graph::coupled::descriptor::PreparedSlopeFunction;

const LIMIT: f64 = 1_000_000.;

fn small(x: f64) -> bool {
    x.is_finite() && x.fract() == 0. && x.abs() <= LIMIT
}
fn lattice(x: f64, unit: f64) -> bool {
    small(x / unit)
}

fn coordinate_unit(
    initial: Window,
    bounds: Window,
    path: &[usize],
    increments: &[f64],
    functions: &[PreparedSlopeFunction],
    native: bool,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<f64> {
    // Power-of-two scaling is exact in this bounded normal range. A native
    // step >=2^-13 exceeds1e-4; even translated endpoint comparisons have
    // relative bands below0.001 steps.
    let limit = if native { 13 } else { 30 };
    let mut unit = 1.;
    for _ in 0..=limit {
        debit(work, 4, deadline)?;
        let mut accepted = [initial.lower, initial.upper, bounds.lower, bounds.upper]
            .into_iter()
            .all(|x| lattice(x, unit));
        for &arc in path {
            debit(work, 2, deadline)?;
            let function = functions.get(arc)?;
            debit(work, function.pairs().len(), deadline)?;
            accepted &= lattice(*increments.get(arc)?, unit)
                && function.pairs().iter().all(|p| lattice(p.1, unit));
        }
        if accepted {
            return Some(unit);
        }
        unit *= 0.5;
    }
    None
}

fn value_unit(
    path: &[usize],
    functions: &[PreparedSlopeFunction],
    coordinate_unit: f64,
    charged: bool,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<f64> {
    if !charged {
        return Some(coordinate_unit);
    }
    let mut unit = 1.;
    for _ in 0..=60 {
        let mut accepted = true;
        for &arc in path {
            let function = functions.get(arc)?;
            debit(work, function.pairs().len(), deadline)?;
            accepted &= lattice(function.pairs()[0].0, unit)
                && function.pairs()[1..]
                    .iter()
                    .all(|p| lattice(p.0, unit / coordinate_unit));
        }
        if accepted {
            return Some(unit);
        }
        unit *= 0.5;
    }
    None
}
fn exact(x: f64) -> bool {
    x.is_finite() && x.fract() == 0. && x.abs() <= 4_503_599_627_370_496.
}

fn shape(
    function: &PreparedSlopeFunction,
    charged: bool,
    coordinate_unit: f64,
    value_unit: f64,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<(f64, f64)> {
    debit(work, function.pairs().len(), deadline)?;
    if !function
        .pairs()
        .iter()
        .all(|p| lattice(p.1, coordinate_unit))
    {
        return None;
    }
    if !charged {
        return Some((0., 0.));
    }
    let (mut value, mut previous) = function.pairs()[0];
    if !lattice(value, value_unit) {
        return None;
    }
    let (mut magnitude, mut slope_magnitude) = (value.abs(), 0f64);
    for &(slope, end) in &function.pairs()[1..] {
        if !lattice(slope, value_unit / coordinate_unit) {
            return None;
        }
        let product = slope * (end - previous);
        let next = value + product;
        if !exact(product / value_unit) || !lattice(next, value_unit) {
            return None;
        }
        value = next;
        previous = end;
        magnitude = magnitude.max(value.abs());
        slope_magnitude = slope_magnitude.max(slope.abs());
    }
    Some((magnitude, slope_magnitude))
}

impl CoupledKernel {
    pub(super) fn certify_slope_fronts(
        &self,
        path: &[usize],
        work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<(bool, [f64; 2])> {
        let mut total = Interval::point(0.);
        for &index in &self.slope_indices {
            let CoupledResource::SlopeIntervalCost {
                initial,
                bounds,
                increments,
                functions,
                policy,
            } = &self.definitions[index]
            else {
                return None;
            };
            if *policy == descriptor::FrontOperationPolicy::FrontAlgebra {
                // The native program also merges near-equal summed slopes.
                // Coordinate-lattice spacing alone does not prove source
                // equivalence after that merge. Check the declared machine
                // program independently instead of reusing its optimized min.
                if !self.front_program_domain {
                    match super::machine::source_feasibility(
                        *initial, *bounds, increments, functions, path, work, deadline,
                    )? {
                        false => return Some((false, [0.; 2])),
                        true => {}
                    }
                }
                let (feasible, cost) = super::machine::raw(
                    *initial,
                    *bounds,
                    increments,
                    functions,
                    path,
                    self.cost_scale,
                    work,
                    deadline,
                )?;
                if !feasible {
                    return self.front_program_domain.then_some((false, [0.; 2]));
                }
                total = total.add(Interval {
                    lo: cost[0],
                    hi: cost[1],
                })?;
                continue;
            }
            if !(self.cost_scale == 0. || self.cost_scale == 1.) {
                return None;
            }
            let unit = coordinate_unit(
                *initial,
                *bounds,
                path,
                increments,
                functions,
                *policy == descriptor::FrontOperationPolicy::FrontAlgebra,
                work,
                deadline,
            )?;
            let value_unit =
                value_unit(path, functions, unit, self.cost_scale != 0., work, deadline)?;
            let (mut lo, mut hi, mut shift) = (initial.lower, initial.upper, 0.);
            let mut knots = vec![lo, hi];
            let mut shifts = Vec::with_capacity(path.len());
            let (mut magnitude, mut slopes) = (0., 0.);
            for &arc in path {
                debit(work, 1, deadline)?;
                let increment = *increments.get(arc)?;
                if !lattice(increment, unit) {
                    return None;
                }
                shift += increment;
                if !lattice(shift, unit) {
                    return None;
                }
                let function = functions.get(arc)?;
                let (m, s) = shape(
                    function,
                    self.cost_scale != 0.,
                    unit,
                    value_unit,
                    work,
                    deadline,
                )?;
                magnitude += m;
                slopes += s;
                // In lattice units every intermediate anchor and summed slope
                // is bounded. Coordinates <=2*LIMIT units keep all segment
                // products below2^53, including dyadic physical coordinates.
                if !lattice(magnitude, value_unit) || !lattice(slopes, value_unit / unit) {
                    return None;
                }
                let (left, right) = function.domain();
                lo = lo.max(bounds.lower - shift).max(left - shift);
                hi = hi.min(bounds.upper - shift).min(right - shift);
                shifts.push(shift);
                knots.extend(function.pairs().iter().map(|p| p.1 - shift));
            }
            if lo > hi {
                return Some((false, [0.; 2]));
            }
            knots.extend([lo, hi]);
            debit(
                work,
                knots.len().checked_mul(usize::BITS as usize)?,
                deadline,
            )?;
            knots.retain(|x| *x >= lo && *x <= hi);
            knots.sort_by(f64::total_cmp);
            knots.dedup();
            let mut minimum = f64::INFINITY;
            for x in knots {
                let mut cost = Interval::point(0.);
                for (&arc, &shift) in path.iter().zip(&shifts) {
                    if self.cost_scale == 0. {
                        continue;
                    }
                    let value = paired::slope_range(
                        &functions[arc],
                        bounds::Range::point(x + shift),
                        work,
                        deadline,
                    )?;
                    if value.lo != value.hi || !lattice(value.lo, value_unit) {
                        return None;
                    }
                    cost = cost.add(Interval::point(value.lo))?;
                }
                if cost.lo != cost.hi || !lattice(cost.lo, value_unit) {
                    return None;
                }
                minimum = minimum.min(cost.lo);
            }
            if !minimum.is_finite() {
                return None;
            }
            total = total.add(Interval::point(minimum))?;
        }
        Some((true, [total.lo, total.hi]))
    }
}
