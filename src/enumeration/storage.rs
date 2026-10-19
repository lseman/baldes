//! Partitioned enumeration label storage and dominance.
use super::*;
use crate::labeling::equality_resource_dominates;
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
                        b.resources[r] - EPSILON <= a.resources[r]
                    }
                }
                ResourceKind::Equality => {
                    equality_resource_dominates(a.resources[r], b.resources[r])
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_dominance_preserves_binary_boundaries_during_enumeration() {
        let mut graph = Graph::new(3, 1).unwrap();
        graph.resources[0] = ResourceKind::Equality;
        let make = |resource, cost| Label {
            id: 0,
            vertex: 1,
            resources: [resource, 0.],
            reduced_cost: cost,
            original_cost: cost,
            visited: [0; 16],
            element_hash: 0,
            parent: None,
            arc: None,
            hops: 1,
            cut_words: [0; FULL_WORDS],
            cut_cost: 0.,
        };
        for forward in [true, false] {
            for (a, b) in [
                (1., 1. - EPSILON),
                (1e6, 1e6 - EPSILON),
                (1e6, 1e6 + EPSILON),
            ] {
                let mut storage = Storage::new(1).unwrap();
                assert!(storage.insert(&graph, make(a, -1.), forward));
                assert!(!storage.insert(&graph, make(b, 0.), forward));
                assert_eq!(storage.pending, 1);
                let mut storage = Storage::new(1).unwrap();
                assert!(storage.insert(&graph, make(b, 0.), forward));
                assert!(storage.insert(&graph, make(a, -1.), forward));
                assert_eq!(storage.pending, 1);
                assert_eq!(storage.live, [false, true]);
            }
            // The immediately adjacent value outside the upper endpoint
            // must survive: tolerance must not be enlarged by the fix.
            let edge: f64 = 1. + EPSILON;
            let outside = f64::from_bits(edge.to_bits() + 1);
            let mut storage = Storage::new(1).unwrap();
            assert!(storage.insert(&graph, make(1., -1.), forward));
            assert!(storage.insert(&graph, make(outside, 0.), forward));
            assert_eq!(storage.pending, 2);
        }
        graph.resources[0] = ResourceKind::Disposable;
        let mut storage = Storage::new(1).unwrap();
        assert!(storage.insert(&graph, make(-EPSILON, -1.), false));
        assert!(!storage.insert(&graph, make(f64::from_bits(1), 0.), false));
        assert_eq!(storage.pending, 1);
    }
}
