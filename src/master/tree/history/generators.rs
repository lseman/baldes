//! Refresh evaluated generators only after their family passes admission.
use super::*;

impl BranchEvaluationHistory {
    pub(in crate::master::tree) fn refresh_generators<const D: usize>(
        &self,
        problem: &RoutingContext<D>,
        routes: &[(&Route, f64)],
        ancestors: &[BranchConstraint],
        options: &BranchingOptions,
        cutoff: f64,
        supplied: &mut Vec<BranchCandidate>,
    ) {
        self.refresh_resource_generators_admitted(
            problem,
            routes,
            ancestors,
            Some(options),
            cutoff,
            supplied,
        );
        let mut present: HashSet<_> = supplied.iter().map(|c| key(&c.down)).collect();
        for (identity, template) in &self.generators {
            use BranchConstraint::*;
            let enabled = match &template.down {
                AccumulatedResourceGroup(_) => continue,
                AccumulatedResource(_) => options.accumulated_resource,
                ArcFlow { .. } => true,
                AggregateFlow { .. } => options.aggregate_flow(),
                WindowFlow { .. } => options.window_arcs.is_some(),
                NumberOfVehicles { .. } => options.number_of_vehicles,
                PackingSetAssignment { .. } => options.packing_set_assignment,
                RyanFoster(_) => options.ryan_foster,
                ClusterBoundary { .. } | ClusterPair { .. } => options.clusters.is_some(),
            };
            if !enabled
                || !admission::admitted(options.priorities.priority(&template.down), cutoff)
                || present.contains(identity)
                || template.down.validate(problem).is_err()
            {
                continue;
            }
            if matches!(
                template.down,
                AccumulatedResource(_) | PackingSetAssignment { .. } | RyanFoster(_)
            ) && ancestors.iter().any(|a| key(a) == *identity)
            {
                continue;
            }
            if matches!(template.down, RyanFoster(_))
                && ancestors
                    .iter()
                    .filter_map(|a| {
                        if let RyanFoster(c) = a {
                            Some(if c.together { 3 } else { 1 })
                        } else {
                            None
                        }
                    })
                    .sum::<usize>()
                    + 3
                    > 512
            {
                continue;
            }
            let mut value = 0.;
            let mut valid = true;
            for &(route, weight) in routes {
                if !weight.is_finite() || weight < 0. {
                    valid = false;
                    break;
                }
                if weight == 0. {
                    continue;
                }
                let coefficient = match &template.down {
                    RyanFoster(c) => {
                        (route.nodes.contains(&(c.first as i32))
                            && route.nodes.contains(&(c.second as i32)))
                            as u8 as f64
                    }
                    AccumulatedResource(c) => {
                        let mut mass = 0.;
                        for (i, &arc) in route.arcs.iter().enumerate() {
                            if arc != c.arc {
                                continue;
                            }
                            let Some(&q) = route
                                .resource_profile
                                .get(i + 1)
                                .and_then(|p| p.get(c.resource))
                                .filter(|q| q.is_finite())
                            else {
                                valid = false;
                                break;
                            };
                            mass += (q < c.boundary()) as u8 as f64;
                        }
                        mass
                    }
                    _ => template.down.coefficient(problem, route),
                };
                value += coefficient * weight;
            }
            if !valid
                || !value.is_finite()
                || value.abs() >= 9_007_199_254_740_992.
                || value - value.floor() <= 1e-6
                || value - value.floor() >= 1. - 1e-6
            {
                continue;
            }
            if matches!(
                template.down,
                AccumulatedResource(_) | PackingSetAssignment { .. } | RyanFoster(_)
            ) && !(value > 1e-6 && value < 1. - 1e-6)
            {
                continue;
            }
            let mut candidate = template.clone();
            candidate.value = value;
            for (constraint, rhs) in [
                (&mut candidate.down, value.floor()),
                (&mut candidate.up, value.ceil()),
            ] {
                match constraint {
                    ArcFlow { rhs: r, .. }
                    | AggregateFlow { rhs: r, .. }
                    | WindowFlow { rhs: r, .. }
                    | NumberOfVehicles { rhs: r, .. }
                    | ClusterBoundary { rhs: r, .. }
                    | ClusterPair { rhs: r, .. } => *r = rhs,
                    _ => {}
                }
            }
            if ancestors.contains(&candidate.down) || ancestors.contains(&candidate.up) {
                continue;
            }
            present.insert(identity.clone());
            supplied.push(candidate);
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/history/generators/tests.rs"]
mod tests;
