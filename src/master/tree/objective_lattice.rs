//! Integer-route objective bounds for decimal arc costs, with binary64 error.
use super::*;
use crate::core::rounding::{add_down, add_up, multiply_down};

pub(super) struct ObjectiveLattice {
    scale: f64,
    error: f64,
}
impl ObjectiveLattice {
    pub(super) fn prepare<const D: usize>(problem: &RootProblem<D>) -> Option<Self> {
        let graph = &problem.graph;
        // Unit customer coverage and nonempty elementary routes imply at most
        // 2n arcs in an integer solution. Resource-dependent costs have no such
        // decimal objective contract.
        if graph.coupled_kernel().is_some_and(|k| k.has_costs())
            || graph
                .arcs
                .iter()
                .any(|a| a.tail == graph.source && a.head == graph.sink)
        {
            return None;
        }
        for scale in [1., 10., 100., 1_000., 10_000., 100_000., 1_000_000.] {
            let mut error = 0f64;
            let mut valid = true;
            for arc in &graph.arcs {
                let lo = multiply_down(arc.cost, scale)?;
                let hi = -multiply_down(-arc.cost, scale)?;
                let integer = (arc.cost * scale).round();
                if integer.abs() > 4_503_599_627_370_496. {
                    valid = false;
                    break;
                }
                let distance = add_up(integer, -lo)?.max(add_up(hi, -integer)?);
                if distance > 1e-7 {
                    valid = false;
                    break;
                }
                error = error.max(distance);
            }
            if valid {
                let error = -multiply_down(-error, 2. * problem.customers.len() as f64)?;
                return Some(Self { scale, error });
            }
        }
        None
    }
    pub(super) fn strengthen(&self, bound: f64) -> f64 {
        let tightened = (|| {
            // scale*objective lies within `error` of an integer. Use outward
            // operations before ceil, including at exact lattice boundaries.
            let integer = add_down(multiply_down(bound, self.scale)?, -self.error)?.ceil();
            let numerator = add_down(integer, -self.error)?;
            let value = numerator / self.scale;
            if !value.is_finite() {
                return None;
            }
            // One downward ulp encloses the correctly rounded division.
            let lower = if value == 0. {
                -f64::from_bits(1)
            } else {
                f64::from_bits(if value > 0. {
                    value.to_bits() - 1
                } else {
                    value.to_bits() + 1
                })
            };
            Some(lower)
        })();
        tightened.map_or(bound, |b| bound.max(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn problem(cost: f64) -> RootProblem<2> {
        let mut graph = Graph::new(3, 1).unwrap();
        graph.add_arc(0, 1, cost).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        RootProblem::set_partitioning(graph, 1.).unwrap()
    }
    use crate::Graph;
    #[test]
    fn decimal_lattice_closes_small_gaps_without_exceeding_arc_costs() {
        let problem = problem(220.4);
        let lattice = ObjectiveLattice::prepare(&problem).unwrap();
        for bound in [220.39999796, 220.4, 220.30000001, 220.3, 220.29999999] {
            let tightened = lattice.strengthen(bound);
            assert!(tightened >= bound && tightened <= 220.4);
        }
        assert!(lattice.strengthen(220.39999796) > 220.4 - 1e-9);
        assert!(lattice.strengthen(220.3) <= 220.3);
    }
    #[test]
    fn arbitrary_costs_and_empty_routes_decline_lattice_strengthening() {
        assert!(ObjectiveLattice::prepare(&problem(2f64.sqrt())).is_none());
        let mut problem = problem(0.1);
        problem.graph.add_arc(0, 2, 0.).unwrap();
        assert!(ObjectiveLattice::prepare(&problem).is_none());
    }

    #[test]
    fn nearby_binary_costs_keep_their_error_in_the_strengthened_bound() {
        for cost in [0.1 - 1e-10, 0.1, 0.1 + 1e-10, 220.4 - 1e-10, 220.4 + 1e-10] {
            let lattice = ObjectiveLattice::prepare(&problem(cost)).unwrap();
            for bound in [cost, cost - 1e-6, cost - 0.1] {
                assert!(lattice.strengthen(bound) <= cost);
            }
        }
    }
}
