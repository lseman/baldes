//! Pareto frontiers of pairing buckets used by reduced-cost bucket-arc fixing.
use crate::graph::buckets::{BucketGraph, BucketId};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Default)]
pub struct PairingCache {
    pub associations: BTreeMap<(BucketId, i32), Vec<usize>>,
}
pub fn contains(grid: &BucketGraph, frontier: &[usize], vertex: usize, bucket: usize) -> bool {
    if grid.dimensions == 1 {
        return frontier.first().is_some_and(|&other| other <= bucket);
    }
    let stride = grid.stride[vertex][1];
    frontier
        .iter()
        .any(|&other| other % stride <= bucket % stride && other / stride <= bucket / stride)
}
pub fn update(grid: &BucketGraph, frontier: &mut Vec<usize>, vertex: usize, bucket: usize) {
    if grid.dimensions == 1 {
        frontier.clear();
        frontier.push(bucket);
        return;
    }
    let stride = grid.stride[vertex][1];
    let row = bucket / stride;
    let col = bucket % stride;
    let position = frontier.partition_point(|&other| other % stride < col);
    frontier.insert(position, bucket);
    let mut i = position + 1;
    while i < frontier.len() {
        if frontier[i] / stride >= row {
            frontier.remove(i);
        } else {
            i += 1;
        }
    }
}
impl PairingCache {
    pub fn initialize(
        &mut self,
        grid: &BucketGraph,
        forward: bool,
        source: BucketId,
        arc: i32,
        target: usize,
    ) -> Vec<usize> {
        let mut frontier = self
            .associations
            .get(&(source, arc))
            .cloned()
            .unwrap_or_default();
        for p in &grid.direction(forward).buckets[source.vertex][source.bucket].predecessors {
            if let Some(previous) = self.associations.get(&(p.bucket, arc)) {
                for &bucket in previous {
                    if !contains(grid, &frontier, target, bucket) {
                        update(grid, &mut frontier, target, bucket);
                    }
                }
            }
        }
        frontier
    }
}
#[derive(Clone, Debug, Default)]
pub struct FixingReport {
    pub checked: usize,
    pub eliminated_forward: usize,
    pub eliminated_backward: usize,
}
#[derive(Clone, Debug, Default)]
pub struct FixingResult {
    pub eliminated: usize,
    pub state: i32,
    pub routes: usize,
}
