//! Per-bucket native insertion with immutable ancestry in an arena. Removing a
//! dominated active label never destroys a predecessor needed for reconstruction.
use super::arena::ArenaRecord;
pub use super::arena::LabelArena;
/// Keep the plain borrowed path a pointer-sized argument, while moving owned
/// labels with heap state into the arena after acceptance.
pub(crate) trait LabelInput {
    fn label(&self) -> &StoredLabel;
    fn into_label(self) -> StoredLabel;
}
impl LabelInput for StoredLabel {
    #[inline(always)]
    fn label(&self) -> &StoredLabel {
        self
    }
    #[inline(always)]
    fn into_label(self) -> StoredLabel {
        self
    }
}
impl LabelInput for &StoredLabel {
    #[inline(always)]
    fn label(&self) -> &StoredLabel {
        self
    }
    #[inline(always)]
    fn into_label(self) -> StoredLabel {
        self.clone()
    }
}
use super::{equality_resource_dominates, Statistics, EPSILON};
use crate::cuts::state::{packed_dominate, DominanceDescriptor, LIMITED_WORDS};
use crate::graph::buckets::BucketId;
use crate::graph::{Graph, ResourceKind};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertionTest {
    Full,
    ResourcesAndCost,
    ResourcesAndNg,
    Resources,
}
#[derive(Clone, Debug)]
pub struct StoredLabel {
    pub id: usize,
    pub local_id: u64,
    pub bucket: BucketId,
    pub vertex: usize,
    pub cost: f64,
    pub original_cost: f64,
    pub resources: [f64; 2],
    pub ng: u16,
    pub branch_state: crate::pricing::restrictions::BranchState,
    pub parent: Option<usize>,
    pub arc: Option<usize>,
    pub hops: usize,
    pub expanded: bool,
    /// Packed limited-memory cut state and its accumulated cost.
    pub cut_words: [u32; LIMITED_WORDS],
    pub additional_cost: f64,
    pub half_potential: f64,
}
impl StoredLabel {
    /// Bucket ordering key, (0.5 potential + additional) + (0 + reduced);
    /// the rank-one potential is zero. Equals the standard key without cuts.
    pub fn key(&self) -> f64 {
        (self.half_potential + self.additional_cost) + (0.0 + self.cost)
    }
}
/// The fields dominance reads before the cut words, packed so bucket scans
/// avoid loading the full record. The scalar cache is 56 bytes on 64-bit targets.
#[derive(Clone, Copy, Debug)]
pub struct HotLabel {
    pub key: f64,
    pub cost: f64,
    pub additional_cost: f64,
    pub resources: [f64; 2],
    pub hops: usize,
    pub ng: u16,
}
impl HotLabel {
    #[inline]
    pub fn of(l: &StoredLabel) -> Self {
        Self {
            key: l.key(),
            cost: l.cost,
            additional_cost: l.additional_cost,
            resources: l.resources,
            hops: l.hops,
            ng: l.ng,
        }
    }
}
/// `HotLabel`s as columns, so a bucket's prefix scan compares eight labels
/// at a time.
#[derive(Clone, Debug, Default)]
pub struct HotColumns {
    pub key: Vec<f64>,
    pub cost: Vec<f64>,
    pub additional_cost: Vec<f64>,
    pub resources: [Vec<f64>; 2],
    pub hops: Vec<usize>,
    pub ng: Vec<u16>,
}

#[derive(Clone, Debug, Default)]
pub struct LabelBucket {
    pub active: Vec<usize>,
    /// The `HotLabel` of each `active` entry, in the same order.
    pub hot: HotColumns,
    pub archived: Vec<usize>,
    pub next_id: u64,
    pub lower_cost: f64,
    /// Completion bound including native resource-knapsack dual costs.
    pub knapsack_lower_cost: f64,
}
impl LabelBucket {
    pub fn completion_bound(&self, knapsack: bool) -> f64 {
        if knapsack {
            self.knapsack_lower_cost
        } else {
            self.lower_cost
        }
    }
}
#[derive(Clone, Debug)]
pub struct LabelStorage {
    pub labels: LabelArena,
    /// `HotLabel::of(&labels[i])`, kept contiguous so bucket scans stay in cache.
    pub hot: Vec<HotLabel>,
    pub buckets: Vec<Vec<LabelBucket>>,
    pub crossing: Vec<Vec<usize>>,
    pub root: Option<usize>,
}

#[cfg(test)]
mod tests;

mod dominance;
mod insertion;
mod management;

mod hot_columns;
