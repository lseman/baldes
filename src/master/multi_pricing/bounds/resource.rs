//! Resource-bounded walk relaxation, without elementarity assumptions.
use crate::{graph::Graph, master::rounding::add_down};
use std::time::Instant;

/// A nonnegative scalar bounds the number of positive-consumption arcs. If its
/// zero-consumption subgraph is acyclic, each block of zero arcs has at most
/// n-1 edges, so k positive arcs give at most (k+1)*n-1 total arcs. Extra walks
/// remain in the relaxation; every resource-feasible walk is still included.
pub(super) fn hop_limit<const D: usize>(graph: &Graph<D>, work: usize) -> Option<usize> {
    let n = graph.vertices.len();
    let per_pass = graph.arcs.len().checked_add(2 * n)?;
    if per_pass == 0 {
        return None;
    }
    let mut best = None;
    for r in 0..graph.resources.len() {
        if graph.arcs.iter().any(|a| a.consumption[r] < 0.) {
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
        let Some(hops) = (positive + 1).checked_mul(n).and_then(|v| v.checked_sub(1)) else {
            continue;
        };
        if hops > work / per_pass || best.is_some_and(|old| hops >= old) {
            continue;
        }
        let mut outgoing = vec![Vec::new(); n];
        let mut indegree = vec![0usize; n];
        for arc in &graph.arcs {
            if arc.consumption[r] == 0. {
                outgoing[arc.tail].push(arc.head);
                indegree[arc.head] += 1;
            }
        }
        let mut queue: Vec<_> = (0..n).filter(|&v| indegree[v] == 0).collect();
        let mut processed = 0;
        while processed < queue.len() {
            let v = queue[processed];
            processed += 1;
            for &head in &outgoing[v] {
                indegree[head] -= 1;
                if indegree[head] == 0 {
                    queue.push(head);
                }
            }
        }
        if processed == n {
            best = Some(hops);
        }
    }
    best
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
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn bounded_walks_cover_repeated_vertices_and_budget_failures_stay_unknown() {
        let mut g = Graph::<5>::new(4, 5).unwrap();
        for v in 0..4 {
            g.set_bounds(v, 4, 0., 4.).unwrap();
        }
        for (u, v, c, q) in [
            (0, 1, 0., 1.),
            (1, 2, -1., 1.),
            (2, 1, 0., 1.),
            (1, 3, 0., 0.),
        ] {
            let id = g.add_arc(u, v, c).unwrap();
            g.arc_mut(id).unwrap().consumption[4] = q;
        }
        let deadline = Instant::now() + Duration::from_secs(1);
        let weights: Vec<_> = g.arcs.iter().map(|a| a.cost).collect();
        let mut work = 100_000;
        let value = minimum(&g, &weights, &mut work, deadline).unwrap().unwrap();
        // 0->1->2->1->3 is resource-feasible with cost -1. Excluding repeated
        // vertices would produce 0, an invalid bound on this walk domain.
        assert!(value <= -1.);
        assert!(value.is_finite());
        assert!(minimum(&g, &weights, &mut 0, deadline).is_none());
        assert!(minimum(&g, &weights, &mut 100_000, Instant::now()).is_none());
        // No resource bounds this cycle now. A finite hop cutoff is unjustified.
        g.arc_mut(0).unwrap().consumption[4] = 0.;
        g.arc_mut(1).unwrap().consumption[4] = 0.;
        g.arc_mut(2).unwrap().consumption[4] = 0.;
        assert!(minimum(&g, &weights, &mut 100_000, deadline).is_none());
        // A positive source arc alone cannot bound zero-consumption cycles.
        g.arc_mut(0).unwrap().consumption[4] = 1.;
        assert!(minimum(&g, &weights, &mut 100_000, deadline).is_none());
        g.arc_mut(1).unwrap().consumption[4] = -1.;
        assert!(minimum(&g, &weights, &mut 100_000, deadline).is_none());
    }

    #[test]
    fn resource_hop_limits_round_outward_at_integer_consumption_boundaries() {
        for upper in [
            f64::from_bits(0.375f64.to_bits() - 1),
            0.375,
            f64::from_bits(0.375f64.to_bits() + 1),
        ] {
            let mut g = Graph::<2>::new(4, 1).unwrap();
            for v in 0..4 {
                g.set_bounds(v, 0, 0., upper).unwrap();
            }
            for (u, v, c, q) in [
                (0, 1, 0., 0.),
                (1, 2, -1., 0.125),
                (2, 1, 0., 0.),
                (1, 3, 0., 0.),
            ] {
                let id = g.add_arc(u, v, c).unwrap();
                g.arc_mut(id).unwrap().consumption[0] = q;
            }
            let weights: Vec<_> = g.arcs.iter().map(|a| a.cost).collect();
            // All these quantities are exact binary fractions. The first
            // window permits two cycles; the other two permit three.
            let cycles = if upper < 0.375 { 2. } else { 3. };
            let value = minimum(
                &g,
                &weights,
                &mut 100_000,
                Instant::now() + Duration::from_secs(1),
            )
            .unwrap()
            .unwrap();
            assert!(value.is_finite() && value <= -cycles);
        }
    }
}
