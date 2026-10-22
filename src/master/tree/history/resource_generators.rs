//! Refresh retained resource generators independently of fresh separation.
use super::*;

impl BranchEvaluationHistory {
    #[cfg(test)]
    pub(crate) fn refresh_resource_generators<const D: usize>(
        &self,
        problem: &RootProblem<D>,
        routes: &[(&Route, f64)],
        ancestors: &[BranchConstraint],
        supplied: &mut Vec<BranchCandidate>,
    ) {
        self.refresh_resource_generators_admitted(
            problem,
            routes,
            ancestors,
            None,
            f64::NEG_INFINITY,
            supplied,
        );
    }
    pub(super) fn refresh_resource_generators_admitted<const D: usize>(
        &self,
        problem: &RootProblem<D>,
        routes: &[(&Route, f64)],
        ancestors: &[BranchConstraint],
        options: Option<&BranchingOptions>,
        cutoff: f64,
        supplied: &mut Vec<BranchCandidate>,
    ) {
        let mut present: HashSet<_> = supplied.iter().map(|c| key(&c.down)).collect();
        present.extend(ancestors.iter().map(key));
        for (identity, template) in &self.resource_generators {
            if options.is_some_and(|o| {
                !o.accumulated_resource
                    || !super::admission::admitted(o.priorities.priority(&template.down), cutoff)
            }) {
                continue;
            }
            if present.contains(identity) {
                continue;
            }
            let BranchConstraint::AccumulatedResourceGroup(group) = &template.down else {
                continue;
            };
            if group.validate(problem).is_err() {
                continue;
            }
            let mut value = 0.;
            let mut total = 0.;
            let mut valid = true;
            for &(route, weight) in routes {
                if !weight.is_finite() || weight < 0. {
                    valid = false;
                    break;
                }
                if weight == 0. {
                    continue;
                }
                for (position, arc) in route.arcs.iter().enumerate() {
                    if group.arcs().binary_search(arc).is_err() {
                        continue;
                    }
                    let Some(&quantity) = route
                        .resource_profile
                        .get(position + 1)
                        .and_then(|q| q.get(group.resource()))
                        .filter(|q| q.is_finite())
                    else {
                        valid = false;
                        break;
                    };
                    total += weight;
                    if quantity < group.boundary() {
                        value += weight;
                    }
                }
                if !valid {
                    break;
                }
            }
            if valid && total <= 1. + 1e-6 && value > 1e-6 && value < 1. - 1e-6 {
                supplied.push(BranchCandidate {
                    value,
                    ..template.clone()
                });
                present.insert(identity.clone());
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/history/resource_generators/tests.rs"]
mod tests;
