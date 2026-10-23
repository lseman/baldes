//! Resource-bounded walk relaxation, without elementarity assumptions.
use crate::{core::rounding::add_down, graph::Graph};
use std::time::Instant;

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
    let mut best = None;
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
#[path = "../../../../tests/unit/master/mp/bounds/resource/tests.rs"]
mod tests;
