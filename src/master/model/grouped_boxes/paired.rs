//! Source paired envelopes and proofs of integer terminal activities.
use super::*;
use crate::master::controls::{terminal_domain, ControlSearchLimits};

pub(super) fn integer_program_range<const D: usize>(
    graph: &Graph<D>,
    resource: usize,
) -> Option<Bounds> {
    let range = crate::master::controls::integer_program_range(graph, resource)?;
    Some(Bounds {
        lower: range.lo,
        upper: range.hi,
    })
}

/// Cover all base controls on every path of the common scalar relaxation.
/// An integer proof requires a singleton integer enclosure on each path;
/// integer endpoints of a nontrivial terminal interval prove nothing here.
pub(super) fn range<const D: usize>(
    graph: &Graph<D>,
    resource: usize,
    integer: bool,
) -> Option<Option<Bounds>> {
    graph.validate().ok()?;
    if !activity_rows::paired(graph, resource) || graph.front_program_domain() {
        return None;
    }
    let deadline = Instant::now().checked_add(std::time::Duration::from_secs(1))?;
    let mut work = 1_000_000;
    let paths = super::relaxed_activity_paths(graph, &mut work, deadline)?;
    let mut limits = ControlSearchLimits {
        max_cells: usize::MAX,
        max_work: work,
        max_lp_iterations: 100_000,
        deadline: Some(deadline),
    };
    let mut range: Option<Bounds> = None;
    for path in paths {
        let Some(domain) = terminal_domain(graph, &path, resource, &mut limits)? else {
            continue;
        };
        let (lower, upper) = (domain.lower, domain.upper);
        if integer
            && (lower != upper || lower.fract() != 0. || lower.abs() > 9_007_199_254_740_992.)
        {
            return None;
        }
        match &mut range {
            Some(old) => {
                old.lower = old.lower.min(lower);
                old.upper = old.upper.max(upper);
            }
            None => range = Some(Bounds { lower, upper }),
        }
    }
    (Instant::now() < deadline).then_some(range)
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/paired_activity_box_tests.rs"]
mod tests;
