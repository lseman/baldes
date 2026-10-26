//! Stateful pricing, cut preparation, and retained elementary route universe.
mod completion;
mod configuration;
mod construction;
mod cuts;
mod fixing_workflow;
mod lifecycle;
mod presence;
mod relaxation;
pub mod restrictions;
mod solve;
mod universe;
mod variable_costs;
/// Compatibility alias; new callers should use `pricing::restrictions`.
#[doc(hidden)]
pub use restrictions as branching;
pub mod controller;
pub mod fixing;
pub mod phases;
use crate::core::options::Options;
use crate::cuts::state::{self, FullInput, FullKPathInput, LimitedInput, LimitedKPathInput};
use crate::cuts::CutRegistry;
use crate::enumeration::{self, Enumeration};
use crate::labeling::engine::{EngineControls, LabelingEngine};
use crate::{incomplete, invalid, Controls, Graph, PricingResult, Result};
use std::sync::Arc;
use std::time::Duration;
#[derive(Debug)]
struct RawPricing<const D: usize> {
    solver: Box<Solver<D>>,
    controls: EngineControls<D>,
}
#[derive(Debug)]
pub struct Solver<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub graph: Graph<D>,
    pub(crate) forbidden_arcs: Vec<bool>,
    pub(crate) ryan_foster: Vec<restrictions::RyanFosterConstraint>,
    pub controls: Controls,
    pub options: Options,
    pub phases: Vec<phases::Phase>,
    pub engine: Option<LabelingEngine<D>>,
    pub enumeration: Option<Enumeration<D>>,
    /// Preserve arc orderings needed by current or future arc-flow branches.
    pub enumeration_preserve_arc_paths: bool,
    pub cutoff: f64,
    prepared: Option<Arc<Graph<D>>>,
    pub bucket_steps: crate::graph::ResourceValues<D>,
    pub rollback_point_saved: bool,
    pub requires_rollback: bool,
    pub round: controller::RoundState,
    pub maximum_variable_id: Option<usize>,
    pub dominance_mode: i32,
    accumulated_branches: Vec<restrictions::AccumulatedResourceBranch>,
    accumulated_original_bounds: Vec<(i32, usize, Option<crate::graph::Window>, bool)>,
    ng_baseline: Option<(Vec<Vec<usize>>, Vec<Vec<usize>>)>,
    pub variable_costs: Vec<f64>,
    raw_arc_costs: Option<Vec<f64>>,
    arc_cost_scratch: Vec<f64>,
    // Private full-domain workspace: it never receives reduced-cost fixing or
    // enumeration, and is never copied into node/transaction forks.
    raw_pricing: Option<RawPricing<D>>,
    enumeration_snapshot: Option<Arc<Graph<D>>>,
    /// Registered rank-one cuts and their separator.
    pub cuts: CutRegistry,
    /// Original knapsack resource ID -> solver coordinate. Empty is identity.
    pub knapsack_resource_indices: std::collections::BTreeMap<usize, usize>,
    /// Use graph-scaled knapsack coefficients and unrounded nonpositive duals.
    /// This does not by itself provide a directed-rounding certificate.
    pub knapsack_exact_coefficients: bool,
    /// Keep current row duals without compatibility quantization in phase I.
    pub(crate) exact_cut_duals: bool,
    /// Cut inputs the engine was last prepared with.
    knapsack_pricing: crate::cuts::knapsack::KnapsackPricing,
    /// Owned physical-arc presence rows survive engine replacement and forks.
    presence_pricing: crate::cuts::soft_conflicts::PresencePricing,
    applied_cuts: (
        Vec<LimitedInput>,
        Vec<LimitedKPathInput>,
        Vec<FullInput>,
        Vec<FullKPathInput>,
    ),
    cut_input_source: Option<Arc<state::CutInputSource>>,
}
