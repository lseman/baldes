//! Exact additive schedules on a bounded binary resource grid.
use super::*;

// Resource dominance uses an absolute EPSILON interval. Distinct grid points
// must not become interchangeable, including rounding of an interval endpoint.
const MIN_EXPONENT: i32 = -18;
const MIN_QUANTUM: f64 = 1. / 262_144.;
const _: () = assert!(MIN_QUANTUM > 2. * crate::labeling::EPSILON);
const MAX_UNITS: f64 = 4_503_599_627_370_496.; // 2^52

fn values<const D: usize>(g: &Graph<D>, r: usize) -> impl Iterator<Item = f64> + '_ {
    std::iter::once(g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower))
        .chain(
            g.vertices
                .iter()
                .flat_map(move |v| [v.bounds[r].lower, v.bounds[r].upper]),
        )
        .chain(g.arcs.iter().flat_map(move |a| {
            std::iter::once(a.consumption[r])
                .chain(a.bounds[r].into_iter().flat_map(|w| [w.lower, w.upper]))
        }))
}

/// Smallest power-of-two quantum needed by this nonnegative binary64 value.
fn exponent(value: f64) -> i32 {
    let bits = value.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if biased == 0 {
        -1074 + fraction.trailing_zeros() as i32
    } else {
        biased - 1023 - 52 + (fraction | (1_u64 << 52)).trailing_zeros() as i32
    }
}

/// All inputs are integer multiples of one quantum in [2^-18, 1], with
/// at most 2^52 units each. A feasible prefix is bounded by a vertex window;
/// adding an arc therefore uses at most 2^53 units and is exact, even when
/// the resulting candidate is infeasible. Waiting selects an exact endpoint.
pub(super) fn validate<const D: usize>(g: &Graph<D>, r: usize) -> Result<()> {
    let mut unit_exponent = 0;
    let inputs = || values(g, r).chain(g.time_window_endpoints(r));
    for value in inputs() {
        if !value.is_finite() || value < 0. {
            return Err(invalid(
                "resource knapsack requires nonnegative finite loads and windows",
            ));
        }
        if value != 0. {
            unit_exponent = unit_exponent.min(exponent(value));
        }
    }
    if unit_exponent < MIN_EXPONENT {
        return Err(invalid(
            "resource knapsack binary grid is finer than the pricing tolerance permits",
        ));
    }
    let quantum = f64::from_bits(((unit_exponent + 1023) as u64) << 52);
    if inputs().any(|value| value / quantum > MAX_UNITS) {
        return Err(invalid(
            "resource knapsack binary grid exceeds the exact additive range",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/multi_pricing/resource_rows/lattice/tests.rs"]
mod tests;
