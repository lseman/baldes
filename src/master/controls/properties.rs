//! Properties of source controls, independent of master rows and objectives.
use super::affine::Interval;
use crate::graph::coupled::{CoupledResource, PiecewiseLinearFunction, PreparedSlopeFunction};
use crate::graph::Graph;

fn slope_constant(function: &PreparedSlopeFunction) -> Option<f64> {
    let pairs = function.pairs();
    pairs
        .iter()
        .skip(1)
        .all(|pair| pair.0 == 0.)
        .then_some(pairs.first()?.0)
}
fn point_constant(function: &PiecewiseLinearFunction) -> Option<f64> {
    let points = &function.points;
    let first = points.first()?[1];
    points.iter().all(|p| p[1] == first).then_some(first)
}

/// With zero waiting, constant integer increments and lower clamps preserve
/// integer Gamma for every base choice, not just extremal controls. Upper
/// functions and physical/resource restrictions can only remove controls.
pub(crate) fn integer_program_range<const D: usize>(
    graph: &Graph<D>,
    resource: usize,
) -> Option<Interval> {
    if graph.front_program_domain() {
        return None;
    }
    let hops = graph.max_route_arcs().or_else(|| {
        graph
            .is_acyclic()
            .then(|| graph.vertices.len().saturating_sub(1))
    });
    let (initial, increments, lower, upper): (f64, Vec<f64>, Vec<f64>, Option<Vec<f64>>) =
        match graph.coupled_kernel()?.definitions().get(resource)? {
            CoupledResource::SlopePaired { initial, arcs, .. } => {
                if arcs.iter().any(|arc| arc.waiting != 0) {
                    return None;
                }
                (
                    slope_constant(initial)?,
                    arcs.iter()
                        .map(|a| slope_constant(&a.increment))
                        .collect::<Option<_>>()?,
                    arcs.iter()
                        .map(|a| slope_constant(&a.lower))
                        .collect::<Option<_>>()?,
                    arcs.iter().map(|a| slope_constant(&a.upper)).collect(),
                )
            }
            CoupledResource::PairedPiecewiseLinear { initial, arcs, .. } => {
                if arcs.iter().any(|arc| arc.waiting != 0) {
                    return None;
                }
                (
                    point_constant(initial)?,
                    arcs.iter()
                        .map(|a| point_constant(&a.increment))
                        .collect::<Option<_>>()?,
                    arcs.iter()
                        .map(|a| point_constant(&a.lower))
                        .collect::<Option<_>>()?,
                    arcs.iter().map(|a| point_constant(&a.upper)).collect(),
                )
            }
            _ => return None,
        };
    const EXACT: u128 = 1 << 53;
    let magnitude = |v: f64| {
        (v.is_finite() && v.fract() == 0. && v.abs() <= EXACT as f64).then_some(v.abs() as u128)
    };
    let nonnegative =
        initial >= 0. && (increments.iter().all(|&v| v >= 0.) || lower.iter().all(|&v| v >= 0.));
    let nonpositive =
        initial <= 0. && increments.iter().all(|&v| v <= 0.) && lower.iter().all(|&v| v <= 0.);
    let initial = magnitude(initial)?;
    let increment = increments
        .iter()
        .try_fold(0u128, |old, &v| Some(old.max(magnitude(v)?)))?;
    let clamp = lower
        .iter()
        .try_fold(0u128, |old, &v| Some(old.max(magnitude(v)?)))?;
    let hop_bound = hops
        .and_then(|hops| {
            initial
                .max(clamp)
                .checked_add((hops as u128).checked_mul(increment)?)
        })
        .filter(|&bound| bound <= EXACT);
    // A uniform terminal window supplies an inductive bound even on cycles.
    // Bound the pre-clamp addition as well as every retained Gamma by 2^53.
    let window_bound = upper.and_then(|values| {
        let upper = values.iter().try_fold(0u128, |old, &v| {
            (v.is_finite() && v.abs().ceil() <= EXACT as f64)
                .then_some(old.max(v.abs().ceil() as u128))
        })?;
        let bound = initial.max(clamp).max(upper);
        (bound.checked_add(increment)? <= EXACT).then_some(bound)
    });
    let bound = match (hop_bound, window_bound) {
        (Some(a), Some(b)) => a.min(b),
        (a, b) => a.or(b)?,
    };
    Some(Interval {
        lo: if nonnegative { 0. } else { -(bound as f64) },
        hi: if nonpositive { 0. } else { bound as f64 },
    })
}

/// A constant source recurrence gives one exact Gamma for every accepted base
/// control on this path. Unknown arithmetic is not a rounded singleton proof.
pub(crate) fn invariant_value<const D: usize>(
    graph: &Graph<D>,
    resource: usize,
    path: &[usize],
    work: &mut usize,
    deadline: Option<std::time::Instant>,
) -> Option<f64> {
    if graph.front_program_domain() || deadline.is_some_and(|end| std::time::Instant::now() >= end)
    {
        return None;
    }
    let definition = graph.coupled_kernel()?.definitions().get(resource)?;
    let (mut value, steps): (f64, Vec<(f64, f64)>) = match definition {
        CoupledResource::SlopePaired { initial, arcs, .. } => {
            let count = path.iter().try_fold(initial.pairs().len(), |n, &p| {
                let arc = arcs.get(p)?;
                n.checked_add(arc.increment.pairs().len())?
                    .checked_add(arc.lower.pairs().len())
            })?;
            *work = work.checked_sub(count)?;
            (
                slope_constant(initial)?,
                path.iter()
                    .map(|&p| {
                        let arc = arcs.get(p)?;
                        (arc.waiting == 0).then_some(())?;
                        Some((slope_constant(&arc.increment)?, slope_constant(&arc.lower)?))
                    })
                    .collect::<Option<_>>()?,
            )
        }
        CoupledResource::PairedPiecewiseLinear { initial, arcs, .. } => {
            let count = path.iter().try_fold(initial.points.len(), |n, &p| {
                let arc = arcs.get(p)?;
                n.checked_add(arc.increment.points.len())?
                    .checked_add(arc.lower.points.len())
            })?;
            *work = work.checked_sub(count)?;
            (
                point_constant(initial)?,
                path.iter()
                    .map(|&p| {
                        let arc = arcs.get(p)?;
                        (arc.waiting == 0).then_some(())?;
                        Some((point_constant(&arc.increment)?, point_constant(&arc.lower)?))
                    })
                    .collect::<Option<_>>()?,
            )
        }
        _ => return None,
    };
    for (increment, lower) in steps {
        let lo = crate::core::rounding::add_down(value, increment)?;
        if lo != crate::core::rounding::add_up(value, increment)? {
            return None;
        }
        value = lo.max(lower);
    }
    if deadline.is_some_and(|end| std::time::Instant::now() >= end) {
        return None;
    }
    value.is_finite().then_some(value)
}
