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
mod tests {
    use super::*;

    fn graph(value: f64, upper: f64) -> Graph<2> {
        let mut g = Graph::<2>::new(3, 1).unwrap();
        g.add_arc(0, 1, 0.).unwrap();
        g.arcs[0].consumption[0] = value;
        for v in &mut g.vertices {
            v.bounds[0] = Window::new(0., upper).unwrap();
        }
        g
    }

    #[test]
    fn binary_resource_grid_accepts_exact_fractions_and_rejects_ambiguous_scales() {
        for value in [0., 1., 1.5, 0.125, MIN_QUANTUM] {
            assert!(validate(&graph(value, 1_000_000.), 0).is_ok(), "{value}");
        }
        for value in [0.1, MIN_QUANTUM / 2., f64::from_bits(1)] {
            assert!(validate(&graph(value, 1.), 0).is_err(), "{value}");
        }
        assert!(validate(&graph(0.5, MAX_UNITS / 2.), 0).is_ok());
        assert!(validate(&graph(0.5, MAX_UNITS), 0).is_err());
        assert!(validate(&graph(-0.5, 1.), 0).is_err());
        let mut g = graph(1., 10.);
        g.initial[0] = Some(f64::NAN);
        assert!(validate(&g, 0).is_err());
        let mut g = graph(1., 10.);
        g.arcs[0].bounds[0] = Some(Window::new(0.1, 1.).unwrap());
        assert!(validate(&g, 0).is_err());
    }

    #[cfg(feature = "multi-time-windows")]
    #[test]
    fn resource_grid_checks_interior_window_endpoints() {
        let mut g = graph(1., 10.);
        g.set_time_windows(
            1,
            0,
            &[Window::new(0., 0.1).unwrap(), Window::new(2., 10.).unwrap()],
        )
        .unwrap();
        assert!(validate(&g, 0).is_err());
        g.set_time_windows(
            1,
            0,
            &[Window::new(0., 0.5).unwrap(), Window::new(2., 10.).unwrap()],
        )
        .unwrap();
        assert!(validate(&g, 0).is_ok());
    }

    #[test]
    fn accepted_grid_additions_and_dominance_keep_distinct_resource_points() {
        let mut seed = 73_u64;
        for quantum in [MIN_QUANTUM, 0.125, 0.5, 1.] {
            for _ in 0..10_000 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let left = seed & ((1_u64 << 52) - 1);
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let right = seed & ((1_u64 << 52) - 1);
                let a = left as f64 * quantum;
                let b = right as f64 * quantum;
                // Integer sum below 2^53 is an independent exact oracle.
                assert_eq!(a + b, (left + right) as f64 * quantum);
                let next = (left + 1) as f64 * quantum;
                assert!(next > a + crate::labeling::EPSILON);
                assert!(a < next - crate::labeling::EPSILON);
            }
        }
    }
}
