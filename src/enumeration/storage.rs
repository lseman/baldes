//! Partitioned enumeration label storage and dominance.
use super::*;
#[derive(Clone, Debug)]
pub struct Storage {
    pub partitions: Vec<Vec<usize>>,
    pub arena: Vec<Label>,
    pub live: Vec<bool>,
    cursor: usize,
    pub pending: usize,
    pub checks: u64,
    pub(super) preserve_arc_paths: bool,
}
impl Storage {
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
    fn dominates(g: &Graph, a: &Label, b: &Label, forward: bool) -> bool {
        a.vertex == b.vertex
            && a.visited == b.visited
            && a.original_cost - b.original_cost < EPSILON
            && g.resources.iter().enumerate().all(|(r, k)| match k {
                ResourceKind::Disposable => {
                    if forward {
                        a.resources[r] <= b.resources[r] + EPSILON
                    } else {
                        b.resources[r] <= a.resources[r] + EPSILON
                    }
                }
                ResourceKind::Equality => (a.resources[r] - b.resources[r]).abs() <= EPSILON,
            })
    }
    pub fn insert(&mut self, g: &Graph, mut label: Label, forward: bool) -> bool {
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
        let old = self.partitions[partition].clone();
        for &id in &old {
            let previous = &self.arena[id];
            if label.original_cost + EPSILON < previous.original_cost {
                break;
            }
            self.checks += 1;
            if Self::dominates(g, previous, &label, forward) {
                return false;
            }
        }
        let mut records = vec![];
        for id in old {
            self.checks += 1;
            if Self::dominates(g, &label, &self.arena[id], forward) {
                self.live[id] = false;
                if id >= self.cursor {
                    self.pending -= 1;
                }
            } else {
                records.push(id);
            }
        }
        let at = records
            .partition_point(|&id| !(label.original_cost + EPSILON < self.arena[id].original_cost));
        let id = self.arena.len();
        label.id = id;
        self.arena.push(label);
        self.live.push(true);
        records.insert(at, id);
        self.partitions[partition] = records;
        self.pending += 1;
        true
    }
    pub fn pop(&mut self) -> Option<Label> {
        while self.cursor < self.arena.len() {
            let id = self.cursor;
            self.cursor += 1;
            if self.live[id] {
                self.pending -= 1;
                return Some(self.arena[id].clone());
            }
        }
        None
    }
}
