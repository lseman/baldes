//! Validate a rounded integer witness before publishing its objective.
use super::*;

#[derive(Default)]
struct ExactSum(Vec<f64>);
impl ExactSum {
    fn add(&mut self, mut value: f64) -> Option<()> {
        if !value.is_finite() {
            return None;
        }
        let mut terms = Vec::new();
        for &term in &self.0 {
            let sum = value + term;
            let z = sum - value;
            let error = (value - (sum - z)) + (term - z);
            if !sum.is_finite() || !error.is_finite() {
                return None;
            }
            if error != 0. {
                terms.push(error);
            }
            value = sum;
        }
        if value != 0. {
            terms.push(value);
        }
        self.0 = terms;
        Some(())
    }
    fn product(&mut self, value: f64, integer: f64) -> Option<()> {
        let product = value * integer;
        // Integer multipliers preserve the subnormal lattice, so an FMA
        // residual cannot hide a fractional subnormal product error.
        let error = value.mul_add(integer, -product);
        self.add(error)?;
        self.add(product)
    }
    fn sign(&self) -> f64 {
        self.0.last().copied().unwrap_or(0.)
    }
    fn upper(&self) -> Option<f64> {
        let mut value = 0.;
        for &term in &self.0 {
            let sum = value + term;
            let z = sum - value;
            let error = (value - (sum - z)) + (term - z);
            if !sum.is_finite() || !error.is_finite() {
                return None;
            }
            value = if error > 0. {
                if sum == 0. {
                    f64::from_bits(1)
                } else {
                    f64::from_bits(if sum > 0. {
                        sum.to_bits() + 1
                    } else {
                        sum.to_bits() - 1
                    })
                }
            } else {
                sum
            };
        }
        value.is_finite().then_some(value)
    }
}

pub(super) fn validated_upper(
    problem: &RootProblem,
    routes: &[Route],
    tags: &[i32],
    values: &[f64],
) -> Option<f64> {
    if tags.len() != values.len() {
        return None;
    }
    let mut rows: Vec<_> = problem.rows.iter().map(|_| ExactSum::default()).collect();
    let mut objective = ExactSum::default();
    for (&tag, &value) in tags.iter().zip(values) {
        let quantity = value.round();
        if !value.is_finite()
            || (value - quantity).abs() > 1e-6
            || quantity < 0.
            || quantity > 9_007_199_254_740_992.
        {
            return None;
        }
        if quantity == 0. {
            continue;
        }
        let route = routes.get(usize::try_from(tag).ok()?)?;
        problem.elementary_route(&route.arcs)?;
        for &(row, coefficient) in &problem.route_entries {
            rows.get_mut(row)?.product(coefficient, quantity)?;
        }
        for &id in &route.arcs {
            let arc = problem.arc_position(id)?;
            objective.product(problem.graph.arcs[arc].cost, quantity)?;
            for &(row, coefficient) in &problem.arc_entries[arc] {
                rows.get_mut(row)?.product(coefficient, quantity)?;
            }
        }
        for (row, constraint) in &problem.branch_constraints {
            if matches!(constraint, BranchConstraint::RyanFoster(_)) {
                let coefficient = constraint.coefficient(problem, route);
                if coefficient != 0. {
                    return None;
                }
                rows.get_mut(*row)?.product(coefficient, quantity)?;
            }
        }
    }
    for (mut activity, row) in rows.into_iter().zip(&problem.rows) {
        activity.add(-row.rhs)?;
        let sign = activity.sign();
        if match row.sense {
            RowSense::Equal => sign != 0.,
            RowSense::LessEqual => sign > 0.,
            RowSense::GreaterEqual => sign < 0.,
        } {
            return None;
        }
    }
    for (route, quantity) in &problem.fixed_routes {
        for &id in &route.arcs {
            objective.product(
                problem.graph.arcs[problem.arc_position(id)?].cost,
                f64::from(*quantity),
            )?;
        }
    }
    objective.upper()
}

impl RootProblem {
    pub(crate) fn unit_route_incumbent(&self, routes: &[Route]) -> Option<f64> {
        let tags = (0..routes.len())
            .map(i32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .ok()?;
        validated_upper(self, routes, &tags, &vec![1.; routes.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounded_witness_uses_original_cost_and_exact_rows() {
        let mut graph = Graph::new(3, 1).unwrap();
        for v in 0..3 {
            graph.set_bounds(v, 0, 0., 10.).unwrap();
        }
        graph.add_arc(0, 1, 1e6).unwrap();
        graph.add_arc(1, 2, 0.25).unwrap();
        let mut problem = RootProblem::set_partitioning(graph, 1.).unwrap();
        let ids = problem.graph.arcs.iter().map(|a| a.id).collect::<Vec<_>>();
        let route = problem.elementary_route(&ids).unwrap();
        assert_eq!(
            validated_upper(&problem, &[route.clone()], &[0], &[1. - 1e-12]),
            Some(1_000_000.25)
        );
        assert!(validated_upper(&problem, &[route.clone()], &[0], &[2.]).is_none());
        problem.rows[1].rhs = 1. + 1e-12;
        assert!(validated_upper(&problem, &[route], &[0], &[1.]).is_none());
    }
    #[test]
    fn exact_sum_preserves_cancellation_and_rounds_objective_up() {
        let mut sum = ExactSum::default();
        for x in [1e16, 1., -1e16] {
            sum.add(x).unwrap();
        }
        assert_eq!(sum.upper(), Some(1.));
        sum.add(-1.).unwrap();
        assert_eq!(sum.sign(), 0.);
        sum.add(1.).unwrap();
        sum.add(f64::EPSILON / 4.).unwrap();
        assert_eq!(sum.upper(), Some(f64::from_bits(1f64.to_bits() + 1)));
    }
}
