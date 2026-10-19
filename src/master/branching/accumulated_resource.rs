//! Master-side accumulated-resource validation and schedule violations.
use crate::{
    master::root::RootProblem, pricing::restrictions::AccumulatedResourceBranch, Result, Route,
};

pub(super) fn validate(problem: &RootProblem, rule: AccumulatedResourceBranch) -> Result<()> {
    rule.window(&problem.graph)?;
    Ok(())
}

pub(super) fn coefficient(rule: AccumulatedResourceBranch, route: &Route) -> f64 {
    route
        .arcs
        .iter()
        .enumerate()
        .filter(|&(i, &id)| {
            if id != rule.arc {
                return false;
            }
            route
                .resource_profile
                .get(i + 1)
                .and_then(|q| q.get(rule.resource))
                .map_or(true, |&q| {
                    !q.is_finite()
                        || if rule.upper {
                            q < rule.threshold
                        } else {
                            q > rule.threshold
                        }
                })
        })
        .count() as f64
}
