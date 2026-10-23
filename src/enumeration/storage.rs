//! Partitioned enumeration label storage and dominance.
use super::*;
use crate::labeling::equality_resource_dominates;
#[derive(Clone, Debug)]
pub struct Storage<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub partitions: Vec<Vec<usize>>,
    pub arena: Vec<Label<D>>,
    pub live: Vec<bool>,
    cursor: usize,
    pub pending: usize,
    pub checks: u64,
    pub(super) preserve_arc_paths: bool,
}

impl<const D: usize> Storage<D> {
    pub fn new(partitions: usize) -> Result<Self> {
        if partitions == 0 {
            return Err(invalid("positive enumeration partitions required"));
        }
        Ok(Self {
            partitions: vec![vec![]; partitions],
            arena: vec![],
            live: vec![],
            cursor: 0,
            pending: 0,
            checks: 0,
            preserve_arc_paths: false,
        })
    }
    fn dominates(g: &Graph<D>, a: &Label<D>, b: &Label<D>, forward: bool) -> bool {
        let Some(penalty) = g.coupled_dominance_penalty(
            a.coupled_state.as_deref(),
            b.coupled_state.as_deref(),
            forward,
            a.vertex,
        ) else {
            return false;
        };
        if g.coupled_kernel()
            .map_or(false, |k| k.retains_cost_components())
        {
            if a.cut_cost > b.cut_cost || a.cut_words != b.cut_words {
                return false;
            }
            // Enumeration dominance orders original objectives. Reduced-cost
            // component order alone does not prove original scalar order.
            let (Some(a_base), Some(b_base)) = (
                a.coupled_state
                    .as_deref()
                    .and_then(|s| s.scalar_original_cost()),
                b.coupled_state
                    .as_deref()
                    .and_then(|s| s.scalar_original_cost()),
            ) else {
                return false;
            };
            if a_base > b_base {
                return false;
            }
        }
        let cost = if penalty == 0. {
            a.original_cost
        } else {
            crate::core::rounding::add_up(a.original_cost, penalty).unwrap_or(f64::INFINITY)
        };
        a.presence_state == b.presence_state
            && a.vertex == b.vertex
            && a.visited == b.visited
            && cost - b.original_cost < EPSILON
            && g.resources.iter().enumerate().all(|(r, k)| match k {
                ResourceKind::Disposable => {
                    if forward {
                        a.resources[r] <= b.resources[r] + EPSILON
                    } else {
                        b.resources[r] - EPSILON <= a.resources[r]
                    }
                }
                ResourceKind::Equality => {
                    equality_resource_dominates(a.resources[r], b.resources[r])
                }
            })
    }
    pub fn insert(&mut self, g: &Graph<D>, mut label: Label<D>, forward: bool) -> bool {
        let partition = label.element_hash as usize % self.partitions.len();
        if self.preserve_arc_paths {
            let id = self.arena.len();
            label.id = id;
            self.arena.push(label);
            self.live.push(true);
            self.partitions[partition].push(id);
            self.pending += 1;
            return true;
        }
        for &id in &self.partitions[partition] {
            let previous = &self.arena[id];
            if label.original_cost + EPSILON < previous.original_cost {
                break;
            }
            self.checks += 1;
            if Self::dominates(g, previous, &label, forward) {
                return false;
            }
        }
        let records = &mut self.partitions[partition];
        records.retain(|&id| {
            self.checks += 1;
            if Self::dominates(g, &label, &self.arena[id], forward) {
                self.live[id] = false;
                if id >= self.cursor {
                    self.pending -= 1;
                }
                false
            } else {
                true
            }
        });
        let at = records
            .partition_point(|&id| !(label.original_cost + EPSILON < self.arena[id].original_cost));
        let id = self.arena.len();
        label.id = id;
        self.arena.push(label);
        self.live.push(true);
        records.insert(at, id);
        self.pending += 1;
        true
    }
    /// Take the next live identity, leaving its immutable predecessor record
    /// in the arena. Reborrow it between insertions, which may grow the arena.
    pub fn pop_id(&mut self) -> Option<usize> {
        while self.cursor < self.arena.len() {
            let id = self.cursor;
            self.cursor += 1;
            if self.live[id] {
                self.pending -= 1;
                return Some(id);
            }
        }
        None
    }
    /// Return an owned label for callers that need one. Enumeration engines
    /// use `pop_id` to avoid copying coupled state on every expansion.
    pub fn pop(&mut self) -> Option<Label<D>> {
        self.pop_id().map(|id| self.arena[id].clone())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/enumeration/storage/tests.rs"]
mod tests;
