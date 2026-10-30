//! Recover genuine original columns without treating relaxation paths as witnesses.
use super::*;
use crate::enumeration::PathDomainLimits;
use std::time::{Duration, Instant};

pub(super) fn columns<const D: usize>(
    problem: &RoutingContext<D>,
    state: &mut State<D>,
    duals: &[f64],
) -> Result<Vec<Column>> {
    let now = Instant::now();
    let Some(local) = now.checked_add(Duration::from_millis(250)) else {
        return Ok(vec![]);
    };
    let deadline = state.deadline.map_or(local, |end| end.min(local));
    let graph = &problem.model.networks[0].graph;
    // Arc-element descriptors require their own witness replay. Conservatively
    // decline this optional recovery rather than interpreting them as vertices.
    if graph.arcs.iter().any(|a| !a.elements.is_empty())
        || graph
            .vertices
            .iter()
            .flat_map(|v| &v.elements)
            .any(|&e| e >= graph.element_count)
    {
        return Ok(vec![]);
    }
    let limits = PathDomainLimits {
        max_paths: 10_000,
        max_stored_arcs: 1_000_000,
        max_work: 10_000_000,
        deadline: Some(deadline),
    };
    let Ok(domain) = ConservativeCoverRelaxation::collect(problem, limits) else {
        return Ok(vec![]);
    };
    let mut work = limits.max_work - domain.work_used();
    let mut storage = 1_000_000usize;
    let mut pending = Vec::new();
    for path in domain.paths().paths() {
        if Instant::now() >= deadline {
            return Ok(vec![]);
        }
        let raw_entries = path.arc_positions().iter().try_fold(
            problem.model.networks[0].route_entries.len(),
            |sum, &arc| sum.checked_add(problem.model.networks[0].arc_entries[arc].len()),
        );
        let charge = path
            .arc_positions()
            .len()
            .checked_mul(
                problem.model.rows.len()
                    + state.cut.rows.len()
                    + state.cut.capacity_cuts.len()
                    + state.solver.cuts.knapsack_cuts.len()
                    + graph.element_count
                    + 1,
            )
            .and_then(|n| raw_entries.and_then(|entries| n.checked_add(entries)));
        let Some(remaining) = charge.and_then(|n| work.checked_sub(n)) else {
            return Ok(vec![]);
        };
        work = remaining;
        let ids = path
            .arc_positions()
            .iter()
            .map(|&a| graph.arcs[a].id)
            .collect::<Vec<_>>();
        let Some(route) = problem.elementary_route_checked(&ids)? else {
            continue;
        };
        if !route.cost.is_finite()
            || super::incumbent::exact_route_feasible(problem, &route).is_none()
        {
            continue;
        }
        if domain.customer_rows().iter().any(|&(vertex, row)| {
            problem.model.rows[row].rhs == 0. && route.nodes.contains(&(vertex as i32))
        }) {
            continue;
        }
        let mut elements = vec![false; graph.element_count];
        let mut elementary = true;
        for &vertex in &route.nodes {
            for &element in &graph.vertices[vertex as usize].elements {
                if elements[element] {
                    elementary = false;
                    break;
                }
                elements[element] = true;
            }
            if !elementary {
                break;
            }
        }
        if !elementary {
            continue;
        }
        let column = state.column(problem, &route, duals, 0)?;
        if !column.cost.is_finite()
            || !column.reduced_cost.is_finite()
            || column.entries.iter().any(|&(_, value)| !value.is_finite())
        {
            continue;
        }
        let Some(remaining) = storage.checked_sub(route.arcs.len() + column.entries.len()) else {
            return Ok(vec![]);
        };
        storage = remaining;
        pending.push((route, column));
    }
    if Instant::now() >= deadline
        || state
            .routes
            .len()
            .checked_add(pending.len())
            .map_or(true, |n| n > i32::MAX as usize)
    {
        return Ok(vec![]);
    }
    // No route/cache mutation occurs before the bounded validation completes.
    pending
        .into_iter()
        .map(|(route, mut column)| {
            column.tag = register(state, &route)?;
            // Validation used a provisional tag without mutating the cache.
            // Publish the registered identity only after that transaction.
            column.identity = format!("route:{}", column.tag);
            Ok(column)
        })
        .collect()
}
