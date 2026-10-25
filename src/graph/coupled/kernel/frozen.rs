//! Prepared proof that an interval front's remaining forward operations carry
//! its minimum unchanged. This is deliberately stronger than mathematical
//! zero cost: every reachable arc must translate by literal zero, every domain
//! must contain the complete current front, and every charge must be zero (or
//! suppressed). Existing knots then survive every merge, interpolation clamps
//! new knots above the same minimum, and the minimum delta is exactly zero.
use super::*;
use std::collections::VecDeque;

/// Bound preprocessing work independently of graph size and cycle structure.
/// Exhaustion declines the optimization for that resource; it proves nothing
/// about infeasibility and never truncates the physical graph.
const MAX_PROPAGATIONS: usize = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Continuation {
    zero_increment: bool,
    zero_charge: bool,
    lower: f64,
    upper: f64,
}

impl Continuation {
    const IDENTITY: Self = Self {
        zero_increment: true,
        zero_charge: true,
        lower: f64::NEG_INFINITY,
        upper: f64::INFINITY,
    };
    const UNKNOWN: Self = Self {
        zero_increment: false,
        zero_charge: false,
        lower: f64::INFINITY,
        upper: f64::NEG_INFINITY,
    };

    fn meet(self, other: Self) -> Self {
        Self {
            zero_increment: self.zero_increment && other.zero_increment,
            zero_charge: self.zero_charge && other.zero_charge,
            lower: self.lower.max(other.lower),
            upper: self.upper.min(other.upper),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct PreparedFrozen {
    // Index is the interval-front slot, not the heterogeneous definition slot.
    // Prepared data is immutable and shared across graph/phase-I snapshots.
    fronts: std::sync::Arc<[Vec<Continuation>]>,
}

impl PreparedFrozen {
    pub(super) fn new<const D: usize>(
        graph: &Graph<D>,
        definitions: &[CoupledResource],
        front_indices: &[usize],
    ) -> Self {
        if front_indices.is_empty() {
            return Self {
                fronts: Vec::new().into(),
            };
        }
        let mut predecessors = vec![Vec::new(); graph.vertices.len()];
        for arc in &graph.arcs {
            predecessors[arc.head].push(arc.tail);
        }
        let fronts = front_indices
            .iter()
            .map(|&index| {
                let CoupledResource::IntervalCost {
                    bounds,
                    increments,
                    functions,
                    ..
                } = &definitions[index]
                else {
                    return vec![Continuation::UNKNOWN; graph.vertices.len()];
                };
                let mut summaries = vec![Continuation::IDENTITY; graph.vertices.len()];
                for (arc_index, arc) in graph.arcs.iter().enumerate() {
                    let charge = &functions[arc_index];
                    let domain = charge.prepared_domain();
                    let direct = Continuation {
                        zero_increment: increments[arc_index] == 0.,
                        zero_charge: charge.points.iter().all(|point| point[1] == 0.),
                        lower: bounds.lower.max(domain.lower),
                        upper: bounds.upper.min(domain.upper),
                    };
                    summaries[arc.tail] = summaries[arc.tail].meet(direct);
                }
                let mut queue: VecDeque<_> = (0..graph.vertices.len()).collect();
                let mut queued = vec![true; graph.vertices.len()];
                let mut work = 0usize;
                while let Some(vertex) = queue.pop_front() {
                    queued[vertex] = false;
                    let continuation = summaries[vertex];
                    for &predecessor in &predecessors[vertex] {
                        if work == MAX_PROPAGATIONS {
                            return vec![Continuation::UNKNOWN; graph.vertices.len()];
                        }
                        work += 1;
                        let next = summaries[predecessor].meet(continuation);
                        if next != summaries[predecessor] {
                            summaries[predecessor] = next;
                            if !queued[predecessor] {
                                queued[predecessor] = true;
                                queue.push_back(predecessor);
                            }
                        }
                    }
                }
                summaries
            })
            .collect::<Vec<_>>()
            .into();
        Self { fronts }
    }

    /// True only when every possible physical continuation leaves this
    /// complete domain feasible and the front's minimum unchanged. Terminal
    /// vertices with no outgoing arcs satisfy this vacuously. Cycles and
    /// disconnected successors are included, never assumed unreachable.
    pub(super) fn contains(
        &self,
        front_slot: usize,
        vertex: usize,
        domain: Window,
        cost_scale: f64,
    ) -> bool {
        let Some(summary) = self.fronts.get(front_slot).and_then(|row| row.get(vertex)) else {
            return false;
        };
        summary.zero_increment
            && (cost_scale == 0. || summary.zero_charge)
            && domain.lower.is_finite()
            && domain.upper.is_finite()
            && domain.lower <= domain.upper
            && summary.lower <= domain.lower
            && domain.upper <= summary.upper
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/frozen/tests.rs"]
mod tests;
