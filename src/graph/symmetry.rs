//! Native split-depot reversal and standard-resource symmetry predicates.
use super::buckets::BucketGraph;
use super::{Graph, ResourceKind};
pub fn reverse_arcs<const D: usize>(g: &Graph<D>) -> Option<Vec<usize>> {
    let mut reverse = vec![];
    for a in &g.arcs {
        if g.arcs
            .iter()
            .filter(|b| b.tail == a.tail && b.head == a.head)
            .count()
            != 1
        {
            return None;
        }
        let mut tail = a.tail;
        let mut head = a.head;
        if tail == g.source && head != g.source {
            tail = g.sink;
        } else if tail != g.sink && head == g.sink {
            head = g.source;
        }
        let b = g
            .arcs
            .iter()
            .position(|b| b.tail == head && b.head == tail)?;
        if a.cost != g.arcs[b].cost {
            return None;
        }
        reverse.push(b);
    }
    Some(reverse)
}
pub fn consumption_symmetric<const D: usize>(
    g: &Graph<D>,
    grid: &BucketGraph,
    reverse: &[usize],
) -> bool {
    if !depot_elements_symmetric(g) || reversal_costs(g, reverse).is_none() {
        return false;
    }
    if g.resources.iter().any(|k| *k != ResourceKind::Disposable) {
        return false;
    }
    for (i, a) in g.arcs.iter().enumerate() {
        let b = &g.arcs[reverse[i]];
        if a.elements != b.elements {
            return false;
        }
        for r in 0..g.resources.len() {
            let source = g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower);
            let limit = g.vertices[g.sink].bounds[r].upper;
            let w = a.bounds[r].unwrap_or(g.vertices[a.head].bounds[r]);
            if a.consumption[r] != b.consumption[r] || w.lower != source || w.upper != limit {
                return false;
            }
        }
    }
    for (r, &slot) in grid.resource_slots.iter().enumerate() {
        let source = g.initial[slot].unwrap_or(g.vertices[g.source].bounds[slot].lower);
        let limit = g.vertices[g.sink].bounds[slot].upper;
        for v in 0..grid.order.len() {
            if grid.lower[v][r] != source {
                return false;
            }
            if v > 0 && v + 1 < grid.order.len() && grid.upper[v][r] != limit {
                return false;
            }
        }
    }
    true
}

/// Forward-prefix reuse also reverses its arc costs. Resource symmetry alone
/// does not justify reusing a prefix at arbitrary master duals.
pub fn reduced_costs_symmetric<const D: usize>(g: &Graph<D>, reverse: &[usize]) -> bool {
    reverse.len() == g.arcs.len()
        && g.arcs.iter().zip(reverse).all(|(a, &r)| {
            g.arcs
                .get(r)
                .is_some_and(|b| a.reduced_cost == b.reduced_cost)
        })
}

/// Terminal elements can be exchanged only when they have no internal
/// membership and the source and sink describe disjoint element sets.
pub fn depot_elements_symmetric<const D: usize>(g: &Graph<D>) -> bool {
    let source = &g.vertices[g.source].elements;
    let sink = &g.vertices[g.sink].elements;
    !source.iter().any(|e| sink.contains(e))
        && g.vertices.iter().enumerate().all(|(v, data)| {
            v == g.source
                || v == g.sink
                || data
                    .elements
                    .iter()
                    .all(|e| !source.contains(e) && !sink.contains(e))
        })
        && g.arcs.iter().all(|a| {
            a.elements
                .iter()
                .all(|e| !source.contains(e) && !sink.contains(e))
        })
}

/// Find a vertex potential making opposite arcs have equal reduced cost.
/// Source and sink share the depot potential, so all complete route costs are
/// preserved. Nonconservative directional costs cannot use prefix reversal.
pub fn reversal_costs<const D: usize>(g: &Graph<D>, reverse: &[usize]) -> Option<Vec<f64>> {
    if reverse.len() != g.arcs.len() {
        return None;
    }
    let vertex = |v| if v == g.sink { g.source } else { v };
    let mut adjacent = vec![vec![]; g.vertices.len()];
    for (a, &r) in g.arcs.iter().zip(reverse) {
        let b = g.arcs.get(r)?;
        let delta = (a.reduced_cost - b.reduced_cost) * 0.5;
        if !delta.is_finite() {
            return None;
        }
        let (u, v) = (vertex(a.tail), vertex(a.head));
        adjacent[u].push((v, delta));
        adjacent[v].push((u, -delta));
    }
    let mut potential = vec![None; g.vertices.len()];
    for root in 0..g.vertices.len() {
        if potential[root].is_some() {
            continue;
        }
        potential[root] = Some(0.);
        let mut work = vec![root];
        while let Some(u) = work.pop() {
            for &(v, delta) in &adjacent[u] {
                let next: f64 = potential[u]? + delta;
                if !next.is_finite() {
                    return None;
                }
                if let Some(old) = potential[v] {
                    if (old - next).abs() > 1e-10 {
                        return None;
                    }
                } else {
                    potential[v] = Some(next);
                    work.push(v);
                }
            }
        }
    }
    let mut costs = vec![0.; g.arcs.len()];
    for (i, a) in g.arcs.iter().enumerate() {
        let r = reverse[i];
        if r < i {
            costs[i] = costs[r];
        } else {
            costs[i] = a.reduced_cost + potential[vertex(a.tail)]? - potential[vertex(a.head)]?;
        }
    }
    Some(costs)
}

pub fn depot_mask<const D: usize>(g: &Graph<D>) -> [u64; 16] {
    let mut mask = [0; 16];
    for &e in g.vertices[g.source]
        .elements
        .iter()
        .chain(&g.vertices[g.sink].elements)
    {
        mask[e / 64] |= 1 << (e % 64);
    }
    mask
}
