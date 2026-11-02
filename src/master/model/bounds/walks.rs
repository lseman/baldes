//! Joint scalar-window and registered-row proof over bounded repeated walks.
use super::*;

/// Enumerate a conservative walk relaxation, retaining directed scalar lower
/// states with each path. Repeated vertices remain allowed. A proved resource
/// horizon (or DAG topology) bounds depth; incomplete coverage returns unknown.
pub(super) fn minimum<const D: usize>(
    oracle: &Oracle<D>,
    graph: &Graph<D>,
    weights: &[f64],
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    *work = work.checked_sub(graph.vertices.len().checked_add(graph.arcs.len())?)?;
    if weights.len() != graph.arcs.len() || weights.iter().any(|w| !w.is_finite()) {
        return None;
    }
    let hops = if graph.is_acyclic() {
        graph.vertices.len().checked_sub(1)?
    } else {
        resource::hop_limit(graph, *work)?
    };
    let mut outgoing = vec![vec![]; graph.vertices.len()];
    for (a, arc) in graph.arcs.iter().enumerate() {
        if !oracle
            .proved_zero_arcs
            .get(oracle.index[&arc.id])
            .copied()
            .unwrap_or(false)
        {
            outgoing[arc.tail].push(a);
        }
    }
    let mut initial = [0.; D];
    for (r, value) in initial.iter_mut().enumerate().take(graph.resources.len()) {
        *value = graph.initial[r].unwrap_or(graph.vertices[graph.source].bounds[r].lower);
    }
    let mut stack = vec![(graph.source, initial, 0., Vec::new())];
    let mut minimum: Option<f64> = None;
    while let Some((vertex, prefix, cost, path)) = stack.pop() {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        if vertex == graph.sink {
            let value = add(cost, penalty(oracle, &path, offset, duals, work, deadline)?)?;
            minimum = Some(minimum.map_or(value, |old| old.min(value)));
            continue;
        }
        if path.len() == hops {
            continue;
        }
        for &a in &outgoing[vertex] {
            *work = work.checked_sub(
                path.len()
                    .checked_add(graph.resources.len())?
                    .checked_add(2)?,
            )?;
            let arc = &graph.arcs[a];
            let mut next = prefix;
            let mut allowed = true;
            for r in 0..graph.resources.len() {
                let vertex = graph.vertices[arc.head].bounds[r];
                let window = arc.bounds[r].unwrap_or(vertex);
                // Waiting on equality axes is deliberately allowed too. Lower
                // transitions therefore retain every original feasible walk.
                next[r] = add(prefix[r], arc.consumption[r])?
                    .max(vertex.lower)
                    .max(window.lower);
                if next[r] > vertex.upper.min(window.upper) {
                    allowed = false;
                    break;
                }
            }
            if allowed {
                let mut path = path.clone();
                path.push((arc.tail, arc.head, arc.id));
                stack.push((arc.head, next, add(cost, weights[a])?, path));
            }
        }
    }
    (Instant::now() < deadline).then_some(minimum)
}

fn penalty<const D: usize>(
    oracle: &Oracle<D>,
    path: &[(usize, usize, i32)],
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<f64> {
    let mut value = 0.;
    for &(cut, row) in &oracle.nonlinear_rows {
        if Instant::now() >= deadline {
            return None;
        }
        let dual = duals[offset + row];
        if dual >= 0. {
            // Positive duals already have their conservative arc envelope.
            continue;
        }
        *work = work.checked_sub(path.len().checked_mul(4)?.checked_add(1)?)?;
        let input = &oracle.solver.cuts.cuts[cut];
        // Registry replay uses i32 carry/count arithmetic. Bound every prefix
        // before replay, rather than risk an overflowing adversarial walk.
        let graph = oracle.solver.graph.graph_id.max(0);
        let mut magnitude = i64::from(input.denominator).abs();
        for &(_, head, id) in path {
            magnitude =
                magnitude.checked_add(i64::from(input.weight_for_graph(graph, head)).abs())?;
            magnitude =
                magnitude.checked_add(i64::from(input.arc_weight_for_graph(graph, id)).abs())?;
        }
        if magnitude > i64::from(i32::MAX) || path.len() > i32::MAX as usize {
            return None;
        }
        let limited = crate::cuts::coefficient(input, graph, path, false).ok()?;
        let full = crate::cuts::coefficient(input, graph, path, true).ok()?;
        // Both ordinary and previously enumerated columns can be present.
        // Taking the smaller nonnegative penalty covers either encoding.
        value = super::subtract_product(value, dual, f64::from(limited.min(full)))?;
    }
    if !oracle.presence_rows.iter().any(|&(input, _)| {
        oracle.solver.presence_rows()[input]
            .row
            .is_component_class()
    }) {
        return Some(value);
    }
    for &(input, row) in &oracle.presence_rows {
        if Instant::now() >= deadline {
            return None;
        }
        let dual = duals[offset + row];
        if dual == 0. {
            continue;
        }
        let input = &oracle.solver.presence_rows()[input];
        let arcs = path
            .iter()
            .map(|&(_, _, id)| oracle.index.get(&id).copied())
            .collect::<Option<Vec<_>>>()?;
        let count = arcs
            .iter()
            .try_fold(0usize, |n, &a| n.checked_add(input.arc_events[a].len()))?;
        *work = work.checked_sub(input.row.replay_work(count)?.checked_add(arcs.len())?)?;
        let coefficient = input
            .row
            .coefficient_for_events(
                arcs.iter()
                    .flat_map(|&a| input.arc_events[a].iter().copied()),
            )
            .ok()?;
        if coefficient as u128 > (1_u128 << 53) {
            return None;
        }
        value = super::subtract_product(value, dual, coefficient as f64)?;
    }
    Some(value)
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/bounds/walks_tests.rs"]
mod tests;
