//! Validate a rounded integer witness before publishing its objective.
use super::*;

#[derive(Clone, Default)]
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
    fn compare(&self, value: f64) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        if !value.is_finite() {
            return None;
        }
        let sign = self.sign();
        if sign == 0. {
            return 0f64.partial_cmp(&value);
        }
        // Opposite signs can be compared without overflowing the difference.
        if sign > 0. && value <= 0. {
            return Some(Ordering::Greater);
        }
        if sign < 0. && value >= 0. {
            return Some(Ordering::Less);
        }
        let mut difference = self.clone();
        difference.add(-value)?;
        difference.sign().partial_cmp(&0.)
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

/// Exact-real scalar feasibility of a nonempty physically elementary path.
/// Stored floating resource profiles are not used as feasibility proofs.
pub(crate) fn exact_route_feasible(problem: &RootProblem, route: &Route) -> Option<()> {
    use crate::graph::ResourceKind;
    use std::cmp::Ordering::{Greater, Less};
    let g = &problem.graph;
    if g.resources.len() > 2
        || g.source == g.sink
        || g.sink >= g.vertices.len()
        || route.arcs.len() >= g.vertices.len()
        || route.nodes.len() != route.arcs.len().checked_add(1)?
        || route.nodes.first().copied() != Some(i32::try_from(g.source).ok()?)
    {
        return None;
    }
    let source = g.vertices.get(g.source)?;
    let mut resources = [ExactSum::default(), ExactSum::default()];
    let valid_window =
        |w: crate::graph::Window| w.lower.is_finite() && w.upper.is_finite() && w.lower <= w.upper;
    for (r, value) in resources.iter_mut().enumerate().take(g.resources.len()) {
        let bounds = source.bounds[r];
        if !valid_window(bounds) {
            return None;
        }
        value.add(g.initial[r].unwrap_or(bounds.lower))?;
        if value.compare(bounds.lower)? == Less || value.compare(bounds.upper)? == Greater {
            return None;
        }
    }
    let mut at = g.source;
    let mut visited = HashSet::new();
    for (position, &id) in route.arcs.iter().enumerate() {
        let arc = g.arcs.get(problem.arc_position(id)?)?;
        if arc.id != id
            || arc.tail != at
            || at == g.sink
            || arc.head == g.source
            || route.nodes[position + 1] != i32::try_from(arc.head).ok()?
        {
            return None;
        }
        let head = g.vertices.get(arc.head)?;
        if arc.head != g.sink && !visited.insert(arc.head) {
            return None;
        }
        for (r, value) in resources.iter_mut().enumerate().take(g.resources.len()) {
            let mut bounds = head.bounds[r];
            if !valid_window(bounds) {
                return None;
            }
            if let Some(window) = arc.bounds[r] {
                if !valid_window(window) {
                    return None;
                }
                bounds.lower = bounds.lower.max(window.lower);
                bounds.upper = bounds.upper.min(window.upper);
            }
            let consumption = arc.consumption[r];
            if g.resources[r] == ResourceKind::Disposable && consumption < 0. {
                return None;
            }
            value.add(consumption)?;
            if value.compare(bounds.lower)? == Less {
                if g.resources[r] == ResourceKind::Equality {
                    return None;
                }
                value.0.clear();
                value.add(bounds.lower)?;
            }
            if value.compare(bounds.upper)? == Greater {
                return None;
            }
        }
        at = arc.head;
    }
    (at == g.sink && !visited.is_empty()).then_some(())
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
    // Most LP solutions are fractional. Reject the whole vector before
    // allocating exact row sums or validating any of its integral routes.
    if values.iter().any(|&value| {
        let quantity = value.round();
        !value.is_finite()
            || (value - quantity).abs() > 1e-6
            || quantity < 0.
            || quantity > 9_007_199_254_740_992.
    }) {
        return None;
    }
    let mut rows: Vec<_> = problem.rows.iter().map(|_| ExactSum::default()).collect();
    let mut objective = ExactSum::default();
    for (&tag, &value) in tags.iter().zip(values) {
        let quantity = value.round();
        if quantity == 0. {
            continue;
        }
        let route = routes.get(usize::try_from(tag).ok()?)?;
        exact_route_feasible(problem, route)?;
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
            if constraint.nonlinear() {
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
        if *quantity == 0 {
            return None;
        }
        exact_route_feasible(problem, route)?;
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
    fn chain(consumption: &[f64], upper: f64) -> (RootProblem, Route) {
        let n = consumption.len() + 1;
        let mut graph = Graph::new(n, 1).unwrap();
        graph.initial[0] = Some(0.);
        for v in 0..n {
            graph.set_bounds(v, 0, 0., upper).unwrap();
        }
        for (v, &c) in consumption.iter().enumerate() {
            let id = graph.add_arc(v, v + 1, 1.).unwrap();
            graph.arc_mut(id).unwrap().consumption[0] = c;
        }
        let problem = RootProblem::set_partitioning(graph, 1.).unwrap();
        // Profiles and rounded costs are untrusted payloads. Feasibility and
        // objective are reconstructed from physical arcs and original inputs.
        let route = Route {
            nodes: (0..n as i32).collect(),
            arcs: (0..n as i32 - 1).collect(),
            cost: 0.,
            original_cost: 0.,
            resources: vec![0.],
            resource_profile: vec![vec![0.]; n],
        };
        (problem, route)
    }
    #[test]
    fn exact_resources_reject_false_feasibility_and_recover_false_rejection() {
        let (p, r) = chain(&[1., f64::EPSILON / 2.], 1.);
        assert!(p.elementary_route(&r.arcs).is_some());
        assert!(validated_upper(&p, &[r], &[0], &[1.]).is_none());
        let delta = 3. * 2f64.powi(-53);
        let upper = f64::from_bits(1f64.to_bits() + 3);
        let (p, r) = chain(&[1., delta, delta], upper);
        assert!(p.elementary_route(&r.arcs).is_none());
        assert_eq!(validated_upper(&p, &[r], &[0], &[1.]), Some(3.));
    }
    #[test]
    fn exact_resource_waiting_windows_and_equality_are_distinct() {
        let (mut p, r) = chain(&[0., 0.], f64::MAX);
        p.graph.initial[0] = Some(-f64::MAX);
        p.graph.set_bounds(0, 0, -f64::MAX, -f64::MAX).unwrap();
        p.graph.set_bounds(1, 0, f64::MAX, f64::MAX).unwrap();
        p.graph.set_bounds(2, 0, f64::MAX, f64::MAX).unwrap();
        assert!(exact_route_feasible(&p, &r).is_some());
        p.graph.resources[0] = crate::graph::ResourceKind::Equality;
        assert!(exact_route_feasible(&p, &r).is_none());
        let (mut p, r) = chain(&[0.7, 0.3], 2.);
        p.graph.arcs[0].bounds[0] = Some(crate::Window::new(1., 1.).unwrap());
        assert!(exact_route_feasible(&p, &r).is_some());
        p.graph.set_bounds(2, 0, 0., 1.).unwrap();
        assert!(exact_route_feasible(&p, &r).is_none());
    }
    #[test]
    fn fixed_resource_paths_and_recorded_node_chains_are_revalidated() {
        let (mut p, r) = chain(&[1., f64::EPSILON / 2.], 1.);
        p.fixed_routes.push((r, 1));
        for row in &mut p.rows {
            row.rhs = 0.;
        }
        assert!(validated_upper(&p, &[], &[], &[]).is_none());
        let (mut p, r) = chain(&[0.5, 0.5], 1.);
        p.fixed_routes.push((r, 1));
        for row in &mut p.rows {
            row.rhs = 0.;
        }
        assert_eq!(validated_upper(&p, &[], &[], &[]), Some(2.));
        let (mut p, mut r) = chain(&[0., 0., 0.], 1.);
        p.branch(BranchConstraint::RyanFoster(
            crate::pricing::restrictions::RyanFosterConstraint {
                first: 1,
                second: 2,
                together: false,
            },
        ))
        .unwrap();
        r.nodes[2] = 3; // Cannot hide the second endpoint from the branch row.
        assert!(validated_upper(&p, &[r], &[0], &[1.]).is_none());
    }
}
