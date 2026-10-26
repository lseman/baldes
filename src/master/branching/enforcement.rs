//! Where routing branches are enforced: in pricing, or only by master rows.
//!
//! Fixed per build. The Cargo default matches the modern reference: Ryan–Foster
//! and accumulated-resource branches are enforced through master rows, while
//! pricing ignores those branches and their row duals. Violating columns reach
//! the master, where the row keeps them at zero; numerical node closure follows
//! the reference policy below. Certified mode still requires independent
//! original-domain proof. `--no-default-features` restores pricing-side
//! enforcement for both families unless their individual master features are
//! selected.
use super::BranchConstraint;
use crate::master::{
    cg::{self, BoundMode, FeasibilityStatus},
    routing::RoutingContext,
};
use crate::{pricing::restrictions::RyanFosterConstraint, Route};

/// Whether pricing enforces Ryan–Foster branches.
pub const RYAN_FOSTER_IN_PRICING: bool = !cfg!(feature = "ryan-foster-master");
/// Whether pricing enforces accumulated-resource branches.
pub const ACCUMULATED_RESOURCE_IN_PRICING: bool = !cfg!(feature = "accumulated-resource-master");

impl BranchConstraint {
    /// Whether this build enforces the branch only through its master row.
    pub(crate) fn master_only(&self) -> bool {
        match self {
            Self::RyanFoster(_) => !RYAN_FOSTER_IN_PRICING,
            Self::AccumulatedResource(_) | Self::AccumulatedResourceGroup(_) => {
                !ACCUMULATED_RESOURCE_IN_PRICING
            }
            _ => false,
        }
    }
}

/// Whether the node carries a branch that pricing does not see.
pub(crate) fn has_master_only<const D: usize>(problem: &RoutingContext<D>) -> bool {
    problem.branches().any(BranchConstraint::master_only)
}

/// The Ryan–Foster rules pricing enforces with binary resources.
pub(crate) fn pricing_ryan_foster<const D: usize>(
    problem: &RoutingContext<D>,
) -> Vec<RyanFosterConstraint> {
    problem
        .branches()
        .filter_map(|branch| match branch {
            BranchConstraint::RyanFoster(rule) if RYAN_FOSTER_IN_PRICING => Some(*rule),
            _ => None,
        })
        .collect()
}

/// Whether appending this branch tightens pricing's arc windows.
pub(crate) fn installs_in_pricing(constraint: &BranchConstraint) -> bool {
    match constraint {
        BranchConstraint::AccumulatedResource(_)
        | BranchConstraint::AccumulatedResourceGroup(_) => ACCUMULATED_RESOURCE_IN_PRICING,
        _ => false,
    }
}

/// Pricing-side Ryan–Foster binary resources also make each pair endpoint
/// elementary, so a retained column with an NG cycle through one is retired.
pub(crate) fn retires_column(constraint: &BranchConstraint, route: &Route) -> bool {
    matches!(constraint, BranchConstraint::RyanFoster(rule)
        if RYAN_FOSTER_IN_PRICING
            && !rule.elementary_on_pair(route.nodes.iter().map(|&v| v as usize)))
}

/// Reference master-only closure: a completed column-generation evaluation
/// with no artificial left closes at its master objective, since no new
/// master-improving column remains and the branch duals never reach pricing.
pub(crate) fn close_at_master_objective<const D: usize>(
    problem: &RoutingContext<D>,
    phase_floor: usize,
    values: &[f64],
    output: &mut cg::Output,
) {
    if phase_floor != 0
        || output.status != cg::Status::Optimal
        || output.pruned_by_bound
        || !has_master_only(problem)
        || output.artificials.iter().any(|&c| values[c] > 1e-6)
    {
        return;
    }
    output.numerical_bound = output.numerical_bound.max(output.lp);
    if output.bound_mode == BoundMode::Numerical {
        output.bound = output.numerical_bound;
    }
    output.warm_start.numerical_bound = output.numerical_bound;
    output.warm_start.bound = output.bound;
}

/// Reference pure phase I: an exact pass that inserts no improving column
/// marks the node infeasible. Certified mode asks `certify` (the independent
/// original-domain proof) instead. Returns the phase-I status and whether it
/// is certified, or None when the policy does not apply.
pub(crate) fn phase_one_stall<const D: usize>(
    problem: &RoutingContext<D>,
    exact: bool,
    bound_mode: BoundMode,
    certify: impl FnOnce() -> bool,
) -> Option<(FeasibilityStatus, bool)> {
    if !exact || !has_master_only(problem) {
        return None;
    }
    if bound_mode == BoundMode::Certified {
        let certified = certify();
        let status = if certified {
            FeasibilityStatus::Infeasible
        } else {
            FeasibilityStatus::Stalled
        };
        return Some((status, certified));
    }
    Some((FeasibilityStatus::Infeasible, false))
}
