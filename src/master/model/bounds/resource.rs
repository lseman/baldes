//! Resource-bounded walk relaxation, without elementarity assumptions.
use crate::{core::rounding::add_down, graph::Graph};
use std::time::Instant;

/// Prove scalar-resource infeasibility on a DAG. Coordinate minima may come
/// from different paths, so reachability is only an overapproximation. Waiting
/// is also allowed on equality axes here, preserving a relaxation of all paths.
pub(super) fn infeasible<const D: usize>(
    graph: &Graph<D>,
    work: &mut usize,
    deadline: Instant,
) -> Option<bool> {
    if graph.has_coupled_resources() {
        return None;
    }
    let n = graph.vertices.len();
    let charge = n.checked_mul(D.checked_add(2)?)?.checked_add(
        graph
            .arcs
            .len()
            .checked_mul(graph.resources.len().checked_add(2)?)?,
    )?;
    *work = work.checked_sub(charge)?;
    let mut outgoing = vec![vec![]; n];
    let mut degree = vec![0usize; n];
    for (index, arc) in graph.arcs.iter().enumerate() {
        outgoing[arc.tail].push(index);
        degree[arc.head] += 1;
    }
    let mut queue: std::collections::VecDeque<_> = (0..n).filter(|&v| degree[v] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(vertex) = queue.pop_front() {
        if Instant::now() >= deadline {
            return None;
        }
        order.push(vertex);
        for &index in &outgoing[vertex] {
            let head = graph.arcs[index].head;
            degree[head] -= 1;
            if degree[head] == 0 {
                queue.push_back(head);
            }
        }
    }
    if order.len() != n {
        return None;
    }
    let mut initial = [0.; D];
    for (r, value) in initial.iter_mut().enumerate().take(graph.resources.len()) {
        *value = graph.initial[r].unwrap_or(graph.vertices[graph.source].bounds[r].lower);
    }
    let mut lower = vec![None; n];
    lower[graph.source] = Some(initial);
    for vertex in order {
        if Instant::now() >= deadline {
            return None;
        }
        let Some(prefix) = lower[vertex] else {
            continue;
        };
        for &index in &outgoing[vertex] {
            let arc = &graph.arcs[index];
            let mut next = prefix;
            let mut allowed = true;
            for r in 0..graph.resources.len() {
                let bounds = graph.vertices[arc.head].bounds[r];
                next[r] = add_down(prefix[r], arc.consumption[r])?.max(bounds.lower);
                if next[r] > bounds.upper {
                    allowed = false;
                    break;
                }
            }
            if allowed {
                match &mut lower[arc.head] {
                    Some(previous) => {
                        for r in 0..graph.resources.len() {
                            previous[r] = previous[r].min(next[r]);
                        }
                    }
                    slot @ None => *slot = Some(next),
                }
            }
        }
    }
    Some(lower[graph.sink].is_none())
}

/// Resource-aware lower bound over a complete DAG path relaxation. Keep each
/// path's cost and scalar lower states together: coordinate minima from other
/// paths cannot justify the cost of this one. Directed transitions and waiting
/// on equality axes deliberately overapproximate the original feasible paths.
/// An independent coupled-resource certificate may reject a complete path;
/// unknown certificates keep it in the relaxation. Coupled objective costs are
/// bounded separately by the caller and are never added to these arc weights.
/// Exhaustion returns unknown, never the minimum of a partially visited domain.
pub(super) fn dag_minimum<const D: usize>(
    graph: &Graph<D>,
    weights: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    let coupled = graph.coupled_kernel();
    if (graph.resources.is_empty() && coupled.is_none())
        || weights.len() != graph.arcs.len()
        || weights.iter().any(|c| !c.is_finite())
    {
        return None;
    }
    *work = work.checked_sub(graph.vertices.len().checked_add(graph.arcs.len())?)?;
    if Instant::now() >= deadline || !graph.is_acyclic() {
        return None;
    }
    let mut outgoing = vec![Vec::new(); graph.vertices.len()];
    for (p, arc) in graph.arcs.iter().enumerate() {
        outgoing[arc.tail].push(p);
    }
    let mut initial = [0.; D];
    for (r, value) in initial.iter_mut().enumerate().take(graph.resources.len()) {
        *value = graph.initial[r].unwrap_or(graph.vertices[graph.source].bounds[r].lower);
    }
    let mut stack = vec![(graph.source, initial, 0., coupled.map(|_| Vec::new()))];
    let mut minimum: Option<f64> = None;
    while let Some((vertex, prefix, cost, path)) = stack.pop() {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        if vertex == graph.sink {
            if let (Some(kernel), Some(path)) = (coupled, path.as_deref()) {
                let certificate = kernel.certify_with_budget(path, work, Some(deadline));
                if Instant::now() >= deadline || (certificate.is_none() && *work == 0) {
                    return None;
                }
                if certificate.is_some_and(|(feasible, _)| !feasible) {
                    continue;
                }
            }
            minimum = Some(minimum.map_or(cost, |old| old.min(cost)));
            continue;
        }
        for &p in &outgoing[vertex] {
            let trace_work = path
                .as_ref()
                .map_or(Some(0), |path| path.len().checked_add(1))?;
            *work = work.checked_sub(
                graph
                    .resources
                    .len()
                    .checked_add(2)?
                    .checked_add(trace_work)?,
            )?;
            let arc = &graph.arcs[p];
            let mut next = prefix;
            let mut allowed = true;
            for r in 0..graph.resources.len() {
                let bounds = graph.vertices[arc.head].bounds[r];
                next[r] = add_down(prefix[r], arc.consumption[r])?.max(bounds.lower);
                if next[r] > bounds.upper {
                    allowed = false;
                    break;
                }
            }
            if allowed {
                let next_path = path.as_ref().map(|path| {
                    let mut next = path.clone();
                    next.push(p);
                    next
                });
                stack.push((arc.head, next, add_down(cost, weights[p])?, next_path));
            }
        }
    }
    Some(minimum)
}

/// A nonnegative scalar bounds its positive-consumption arc occurrences. The
/// sum of such per-axis counts also bounds arcs positive on any selected axis.
/// If arcs zero on all selected axes form a DAG, each zero block has at most
/// n-1 edges, giving (k+1)*n-1 total arcs. All resource-feasible repeated walks
/// remain in the relaxation; unsupported or unaffordable proofs stay unknown.
pub(super) fn hop_limit<const D: usize>(graph: &Graph<D>, work: usize) -> Option<usize> {
    let n = graph.vertices.len();
    let per_pass = graph.arcs.len().checked_add(n.checked_mul(2)?)?;
    if per_pass == 0 {
        return None;
    }
    let mut best = graph
        .max_route_arcs()
        .filter(|&hops| hops <= work / per_pass);
    let mut joint_axes = [false; D];
    let mut joint_positive = Some(0usize);
    let mut axis_count = 0;
    for r in 0..graph.resources.len() {
        if graph
            .arcs
            .iter()
            .any(|a| !a.consumption[r].is_finite() || a.consumption[r] < 0.)
        {
            continue;
        }
        let delta = graph
            .arcs
            .iter()
            .map(|a| a.consumption[r])
            .filter(|&v| v > 0.)
            .fold(f64::INFINITY, f64::min);
        if !delta.is_finite() {
            continue;
        }
        let initial = graph.initial[r].unwrap_or(graph.vertices[graph.source].bounds[r].lower);
        let upper = graph
            .vertices
            .iter()
            .map(|v| v.bounds[r].upper)
            .fold(f64::NEG_INFINITY, f64::max);
        // Negation turns a downward sum into an upper bound on the budget.
        let Some(budget) = add_down(initial, -upper).map(|v| -v) else {
            continue;
        };
        let ratio = budget / delta;
        // Below this cap division's rounding error is less than one unit.
        // An additional positive arc therefore rounds the integer limit out.
        if !ratio.is_finite() || ratio < 0. || ratio > 1_000_000. {
            continue;
        }
        let positive = ratio.floor() as usize + 1;
        joint_axes[r] = true;
        joint_positive = joint_positive.and_then(|count| count.checked_add(positive));
        axis_count += 1;
        let Some(hops) = walk_length(n, positive) else {
            continue;
        };
        if hops > work / per_pass || best.is_some_and(|old| hops >= old) {
            continue;
        }
        let mut axes = [false; D];
        axes[r] = true;
        if zero_arcs_are_acyclic(graph, &axes) {
            best = Some(hops);
        }
    }
    // Each axis may have a zero-consumption cycle, while their intersection
    // has none. Count occurrences per axis rather than summing floating-point
    // consumptions with possibly unrelated scales or units.
    if axis_count > 1 {
        if let Some(hops) = joint_positive.and_then(|positive| walk_length(n, positive)) {
            if hops <= work / per_pass
                && best.map_or(true, |old| hops < old)
                && zero_arcs_are_acyclic(graph, &joint_axes)
            {
                best = Some(hops);
            }
        }
    }
    best
}

fn walk_length(vertices: usize, positive: usize) -> Option<usize> {
    positive
        .checked_add(1)?
        .checked_mul(vertices)?
        .checked_sub(1)
}

fn zero_arcs_are_acyclic<const D: usize>(graph: &Graph<D>, axes: &[bool; D]) -> bool {
    let n = graph.vertices.len();
    let mut outgoing = vec![Vec::new(); n];
    let mut indegree = vec![0usize; n];
    for arc in &graph.arcs {
        if axes
            .iter()
            .enumerate()
            .take(graph.resources.len())
            .all(|(r, &selected)| !selected || arc.consumption[r] == 0.)
        {
            outgoing[arc.tail].push(arc.head);
            indegree[arc.head] += 1;
        }
    }
    let mut queue: Vec<_> = (0..n).filter(|&v| indegree[v] == 0).collect();
    let mut processed = 0;
    while processed < queue.len() {
        let vertex = queue[processed];
        processed += 1;
        for &head in &outgoing[vertex] {
            indegree[head] -= 1;
            if indegree[head] == 0 {
                queue.push(head);
            }
        }
    }
    processed == n
}

pub(super) fn minimum<const D: usize>(
    graph: &Graph<D>,
    weights: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    let hops = hop_limit(graph, *work)?;
    let mut previous = vec![None; graph.vertices.len()];
    previous[graph.source] = Some(0.);
    for _ in 0..hops {
        if Instant::now() >= deadline {
            return None;
        }
        *work = work.checked_sub(graph.arcs.len() + 2 * graph.vertices.len())?;
        // Synchronous relaxation: each pass adds at most one arc, making the
        // resource-derived walk-length limit explicit.
        let mut next = previous.clone();
        let mut changed = false;
        for (arc, &weight) in graph.arcs.iter().zip(weights) {
            if let Some(tail) = previous[arc.tail] {
                let value = add_down(tail, weight)?;
                if next[arc.head].map_or(true, |old| value < old) {
                    next[arc.head] = Some(value);
                    changed = true;
                }
            }
        }
        previous = next;
        if !changed {
            break;
        }
    }
    Some(previous[graph.sink])
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/bounds/resource/tests.rs"]
mod tests;
