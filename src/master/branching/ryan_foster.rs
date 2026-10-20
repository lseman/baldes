//! Master-side Ryan–Foster validation and route violation coefficients.
use crate::{
    invalid, master::root::RootProblem, pricing::restrictions::RyanFosterConstraint, Result, Route,
};

pub(super) fn validate<const D: usize>(
    problem: &RootProblem<D>,
    rule: RyanFosterConstraint,
) -> Result<()> {
    if rule.first == rule.second
        || !problem.customers.contains(&rule.first)
        || !problem.customers.contains(&rule.second)
    {
        return Err(invalid(
            "Ryan–Foster requires distinct customer packing sets",
        ));
    }
    Ok(())
}

pub(super) fn coefficient(rule: RyanFosterConstraint, route: &Route) -> f64 {
    (!rule.satisfied_by(route.nodes.iter().map(|&v| v as usize))) as u8 as f64
}
