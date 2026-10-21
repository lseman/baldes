//! Component-ordered forward/backward pricing with typed label storage.
use super::arena::{
    ArenaRecord, BranchedLabel, ExtendedLabel, LabelData, LabelRef, OrdinaryLabel, RecordKind,
};
use super::pool::SolutionPool;

use super::storage::{HotLabel, InsertionTest, LabelInput, LabelStorage, StoredLabel};
use super::{extend_resources, Controls, PricingResult, Route, Statistics, EPSILON, NO_ROUTE_COST};
use crate::cuts::state::{
    self, CutResourceInfo, FullPreparation, LimitedInput, LimitedKPathInput, LimitedPreparation,
    LIMITED_WORDS,
};
use crate::graph::buckets::{BucketArc, BucketGraph, BucketId};
use crate::graph::ng::PreparedNg;
use crate::graph::{Graph, ResourceKind};
use crate::pricing::phases::Phase;
use crate::{incomplete, invalid, Result};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
pub(crate) const RESOURCE_REPLAY_MISMATCH: &str = "reconstructed route violates resource windows";
#[derive(Clone, Debug)]
pub struct EngineControls<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub bidirectional: bool,
    pub bidirectional_mode: i32,
    pub dominance_mode: i32,
    pub steps: crate::graph::ResourceValues<D>,
    pub target_buckets: i32,
    pub group_limit: i32,
    pub filter_other_buckets: bool,
    pub direction_flags: u8,
    pub sink_concatenation: bool,
    pub use_symmetric_execution: bool,
}
impl<const D: usize> Default for EngineControls<D> {
    fn default() -> Self {
        Self {
            bidirectional: true,
            bidirectional_mode: 2,
            dominance_mode: 0,
            steps: [0.; D].into(),
            target_buckets: 100,
            group_limit: 0,
            filter_other_buckets: true,
            direction_flags: 1,
            sink_concatenation: true,
            use_symmetric_execution: true,
        }
    }
}
/// Limited-memory cut state of the extended labels. Empty inputs select the
/// standard labels: no words, zero additional cost, no descriptors.
#[derive(Clone, Debug, Default)]
pub struct EngineCuts {
    pub inputs: Vec<LimitedInput>,
    pub kpath: Vec<LimitedKPathInput>,
    pub knapsack: crate::cuts::knapsack::KnapsackPricing,
    /// Presence rows; use `load_presence` to invalidate cached labels safely.
    pub presence: crate::cuts::soft_conflicts::PresencePricing,
    pub info: CutResourceInfo,
    pub limited: LimitedPreparation,
    /// Full-memory enumeration tables, prepared before an enumeration.
    pub full: Option<FullPreparation>,
}
#[derive(Debug)]
pub struct LabelingEngine<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub grid: BucketGraph,
    pub forward: LabelStorage<D>,
    pub backward: LabelStorage<D>,
    pub statistics: Statistics,
    pub controls: EngineControls<D>,
    pub phase: Phase,
    pub ng: PreparedNg,
    pub branching: crate::pricing::restrictions::RyanFosterResources,
    pub(crate) forbidden_arcs: Vec<bool>,
    pub half_ready: bool,
    pub full_ready: bool,
    last_costs: Vec<f64>,
    pending_middle: Option<bool>,
    pub consumption_joins: bool,
    symmetric_costs: Vec<f64>,
    pub reverse: Vec<usize>,
    extension_checks: Vec<i32>,
    filter_checks: Vec<i32>,
    pub cuts: EngineCuts,
}
/// Replays a sweep's terminal candidates into the pool, in recorded order.
fn insert_terminals(
    pool: &mut SolutionPool<(Option<usize>, Option<usize>)>,
    forward: bool,
    terminals: &[(usize, f64, usize)],
) {
    for &(group, cost, target) in terminals {
        pool.insert(
            group,
            cost,
            0.,
            if forward {
                (Some(target), None)
            } else {
                (None, Some(target))
            },
        );
    }
}

mod completion;
mod context;
mod cuts;
mod fixing;
mod preparation;
mod presence;
mod reconstruction;
mod solve;
mod sweep;
use context::Context;

mod directions;
