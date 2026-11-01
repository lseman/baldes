//! Ryan–Foster-aware walk minimum for the directed pricing certificate.
//!
//! A node's Ryan–Foster rows exclude every path that serves exactly one of a
//! `together` pair, or both of an `apart` pair. Pricing respects them; the plain
//! shortest walk does not. This refinement runs the same directed walk on the
//! product of the graph with the set of Ryan–Foster vertices already visited,
//! accepting only states that satisfy every row, so it is a valid lower bound
//! over the node's domain that can only tighten the unrestricted walk.
use super::add;
use crate::pricing::restrictions::RyanFosterConstraint;
use crate::Graph;
use std::time::Instant;

/// Product states are capped so the refinement stays a bounded side proof.
const MAX_TRACKED_VERTICES: usize = 6;

/// `Some(Some(v))`: every permitted walk costs at least `v`. `Some(None)`: no
/// permitted source/sink walk exists. `None`: unavailable (no restriction,
/// too many tracked vertices, a negative cycle, budget or deadline), in which
/// case the caller keeps its unrestricted bound.
pub(super) fn minimum<const D: usize>(
    graph: &Graph<D>,
    weights: &[f64],
    constraints: &[RyanFosterConstraint],
    excluded: impl Fn(usize) -> bool,
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    if constraints.is_empty() {
        return None;
    }
    let mut tracked: Vec<usize> = vec![];
    for c in constraints {
        for v in [c.first, c.second] {
            if !tracked.contains(&v) {
                tracked.push(v);
            }
        }
    }
    if tracked.len() > MAX_TRACKED_VERTICES || tracked.iter().any(|&v| v >= graph.vertices.len()) {
        return None;
    }
    let bit = |v: usize| tracked.iter().position(|&t| t == v).map(|i| 1usize << i);
    let states = 1usize << tracked.len();
    let accepted = |mask: usize| {
        constraints.iter().all(|c| {
            let a = bit(c.first).is_some_and(|b| mask & b != 0);
            let b = bit(c.second).is_some_and(|b| mask & b != 0);
            if c.together {
                a == b
            } else {
                !(a && b)
            }
        })
    };
    let vertices = graph.vertices.len();
    let mut distances: Vec<Option<f64>> = vec![None; vertices * states];
    let start = bit(graph.source).unwrap_or(0);
    distances[graph.source * states + start] = Some(0.);
    // Bellman–Ford over the product; convergence within |V|·|S| passes is
    // required, otherwise a negative cycle leaves the refinement unknown.
    let passes = vertices * states;
    let mut converged = false;
    for _ in 0..=passes {
        if Instant::now() >= deadline {
            return None;
        }
        let cost = graph.arcs.len().checked_mul(states)?;
        *work = work.checked_sub(cost)?;
        let mut changed = false;
        for (a, (arc, &weight)) in graph.arcs.iter().zip(weights).enumerate() {
            if excluded(a) {
                continue;
            }
            let entered = bit(arc.head).unwrap_or(0);
            for mask in 0..states {
                let Some(tail) = distances[arc.tail * states + mask] else {
                    continue;
                };
                let candidate = add(tail, weight)?;
                let slot = &mut distances[arc.head * states + (mask | entered)];
                if slot.map_or(true, |previous| candidate < previous) {
                    *slot = Some(candidate);
                    changed = true;
                }
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    if !converged {
        return None;
    }
    Some(
        (0..states)
            .filter(|&mask| accepted(mask))
            .filter_map(|mask| distances[graph.sink * states + mask])
            .reduce(f64::min),
    )
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/bounds/ryan_foster_tests.rs"]
mod tests;
