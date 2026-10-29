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
    // Cover n-1 arcs for physical-simple routes or the explicit finite walk
    // bound. An unproved n-1 truncation could omit cheaper repeated walks,
    // particularly when resource objectives have negative costs.
    let allowance = *work / 4;
    let mut remaining = allowance;
    let relaxed_minimum =
        kernel.objective_walk_bound(graph, graph.original_max_arcs(), &mut remaining, deadline);
    *work -= allowance - remaining;
    // This caller needs a universal lower enclosure, not an attained minimum.
    // Keep a successful polynomial proof instead of enumerating the same
    // cyclic path domain merely to tighten it before pricing can begin.
    if let Some(bound) = relaxed_minimum {
        return (bound.is_none_or(f64::is_finite) && Instant::now() < deadline).then_some(bound);
    }
    let allowance = *work / 4;
    if allowance == 0 {
        return None;
    }
    let mut relaxed = graph.clone();
    relaxed
        .resources
        .fill(crate::graph::ResourceKind::Disposable);
    relaxed.set_coupled_resources(&[]).ok()?;
    // Reserve half of this phase's allowance for certification while the
    // finite path universe is visited incrementally. The previous collector
    // retained every physical path and imposed a separate 4,096-path ceiling.
    let mut certificate_work = work.checked_sub(allowance)?;
    let mut minimum: Option<f64> = None;
    let visited = match CompleteResourceRelaxation::visit(
        &relaxed,
        PathDomainLimits {
            max_paths: allowance,
            max_work: allowance,
            max_stored_arcs: allowance,
            deadline: Some(deadline),
        },
        |path| {
            if Instant::now() >= deadline {
                return Err(crate::enumeration::PathDomainFailure::TimeLimit);
            }
            let Some(left) = path
                .len()
                .checked_add(1)
                .and_then(|amount| certificate_work.checked_sub(amount))
            else {
                return Err(crate::enumeration::PathDomainFailure::WorkLimit);
            };
            certificate_work = left;
            let Some((feasible, objective)) =
                kernel.certify_with_budget(path, &mut certificate_work, Some(deadline))
            else {
                return Err(crate::enumeration::PathDomainFailure::WorkLimit);
            };
            if feasible {
                if !objective[0].is_finite() {
                    return Err(crate::enumeration::PathDomainFailure::Arithmetic);
                }
                minimum = Some(minimum.map_or(objective[0], |old| old.min(objective[0])));
            }
            Ok(())
        },
    ) {
        Ok(stats) => stats,
        Err(_) => {
            // A failed visitor exposes no partial traversal counter, so charge
            // its entire reserved allowance. The polynomial proof above was
            // unavailable, so a partial visitor cannot supply a bound.
            *work = certificate_work;
            return None;
        }
    };
    *work = certificate_work.checked_add(allowance.checked_sub(visited.work_used)?)?;
    (Instant::now() < deadline).then_some(minimum)
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/coupled_walk_bounds_tests.rs"]
mod tests;
