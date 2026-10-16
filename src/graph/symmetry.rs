//! Native split-depot reversal and standard-resource symmetry predicates.
use super::buckets::BucketGraph;
use super::{Graph, ResourceKind};
pub fn reverse_arcs(g: &Graph) -> Option<Vec<usize>> {
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
pub fn consumption_symmetric(g: &Graph, grid: &BucketGraph, reverse: &[usize]) -> bool {
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
