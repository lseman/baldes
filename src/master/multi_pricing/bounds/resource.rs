//! Resource-bounded walk relaxation, without elementarity assumptions.
use crate::{graph::Graph, master::rounding::add_down};
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
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn joint_axes_bound_cycles_that_no_individual_axis_bounds() {
        // Every pair of dyadic scales and budgets has an independently known
        // integer-walk optimum: exhaust each of two independent loop budgets.
        for left in 0..=4 {
            for right in 0..=4 {
                for (left_scale, right_scale) in [(0.125, 2.), (1., 0.25), (8., 0.5)] {
                    let mut graph = Graph::<7>::new(4, 7).unwrap();
                    for vertex in 0..4 {
                        graph
                            .set_bounds(vertex, 3, 0., left as f64 * left_scale)
                            .unwrap();
                        graph
                            .set_bounds(vertex, 6, 0., right as f64 * right_scale)
                            .unwrap();
                    }
                    graph.add_arc(0, 1, 0.).unwrap();
                    let a = graph.add_arc(1, 1, -2.).unwrap();
                    graph.arc_mut(a).unwrap().consumption[3] = left_scale;
                    graph.add_arc(1, 2, 0.).unwrap();
                    let b = graph.add_arc(2, 2, -3.).unwrap();
                    graph.arc_mut(b).unwrap().consumption[6] = right_scale;
                    graph.add_arc(2, 3, 0.).unwrap();
                    let mut axis = [false; 7];
                    axis[3] = true;
                    assert!(!zero_arcs_are_acyclic(&graph, &axis));
                    axis = [false; 7];
                    axis[6] = true;
                    assert!(!zero_arcs_are_acyclic(&graph, &axis));
                    axis[3] = true;
                    assert!(zero_arcs_are_acyclic(&graph, &axis));
                    let hops = hop_limit(&graph, 100_000).unwrap();
                    let exact_hops = left + right + 3;
                    assert!(hops >= exact_hops);
                    let weights: Vec<_> = graph.arcs.iter().map(|a| a.cost).collect();
                    let bound = minimum(
                        &graph,
                        &weights,
                        &mut 100_000,
                        Instant::now() + Duration::from_secs(1),
                    )
                    .unwrap()
                    .unwrap();
                    let exact_cost = -2. * left as f64 - 3. * right as f64;
                    assert!(bound.is_finite() && bound <= exact_cost);
                    assert!(hop_limit(&graph, 1).is_none());
                    // Removing either resource bound leaves an adverse loop
                    // with zero consumption on every usable coordinate.
                    graph.arc_mut(a).unwrap().consumption[3] = 0.;
                    assert!(hop_limit(&graph, 100_000).is_none());
                    graph.arc_mut(a).unwrap().consumption[3] = -left_scale;
                    assert!(hop_limit(&graph, 100_000).is_none());
                }
            }
        }
    }

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
