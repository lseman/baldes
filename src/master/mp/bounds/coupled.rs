//! Directed coupled-objective relaxation over a complete scalar path superset.
//!
//! The scalar walk bound cannot simply omit a signed resource objective. This
//! bounded fallback derives one route-wide lower bound independently of arc
//! costs and duals. Every candidate is covered or the entire proof is unknown.
use super::*;
use crate::enumeration::{CompleteResourceRelaxation, PathDomainLimits};

pub(super) fn minimum<const D: usize>(
    graph: &Graph<D>,
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    let Some(kernel) = graph.coupled_kernel().filter(|k| k.has_costs()) else {
        // Dropping feasibility-only coupled restrictions enlarges the domain.
        return Some(Some(0.));
    };
    if *work == 0 || Instant::now() >= deadline {
        return None;
    }
    // The integer route contract is physically vertex-simple. The directed
    // relaxation includes all walks up to n-1 arcs, so exponentially many
    // simple routes can share polynomially many abstract states.
    let allowance = (*work / 4).min(250_000);
    let mut remaining = allowance;
    let relaxed_minimum = kernel.objective_walk_lower_bound(
        graph,
        graph.vertices.len().saturating_sub(1),
        &mut remaining,
        deadline,
    );
    *work -= allowance - remaining;
    let fallback = || {
        (Instant::now() < deadline)
            .then_some(relaxed_minimum)
            .flatten()
            .map(Some)
    };
    let allowance = (*work / 4).min(100_000);
    if allowance == 0 {
        return fallback();
    }
    let mut relaxed = graph.clone();
    relaxed
        .resources
        .fill(crate::graph::ResourceKind::Disposable);
    relaxed.set_coupled_resources(&[]).ok()?;
    let collected = match CompleteResourceRelaxation::collect(
        &relaxed,
        PathDomainLimits {
            max_paths: 4096,
            max_work: allowance,
            max_stored_arcs: allowance,
            deadline: Some(deadline),
        },
    ) {
        Ok(paths) => paths,
        Err(_) => {
            // Failed collectors expose no partial universe or work counter.
            *work -= allowance;
            return fallback();
        }
    };
    *work = work.checked_sub(collected.work_used())?;
    let mut minimum: Option<f64> = None;
    for path in collected.paths() {
        if Instant::now() >= deadline {
            return None;
        }
        let Some(left) = path
            .arc_positions()
            .len()
            .checked_add(1)
            .and_then(|amount| work.checked_sub(amount))
        else {
            return fallback();
        };
        *work = left;
        let Some((feasible, objective)) =
            kernel.certify_with_budget(path.arc_positions(), work, Some(deadline))
        else {
            return fallback();
        };
        if feasible {
            if !objective[0].is_finite() {
                return fallback();
            }
            minimum = Some(minimum.map_or(objective[0], |old| old.min(objective[0])));
        }
    }
    (Instant::now() < deadline).then_some(minimum)
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/mp/coupled_walk_bounds_tests.rs"]
mod tests;
