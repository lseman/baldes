//! Column generation with cut rounds and fixing/enumeration master
//! transitions, from the C++ reference `master/ColumnGeneration.h`.
//!
//! Generic over any `LpMaster` backend (Clp, HiGHS).  The controller trait
//! receives a mutable reference to the generic master so it can inspect
//! primal/dual values and add columns.
pub use super::lp_master::LpMaster;
use super::{
    stabilization::{DualPriceSmoothing, Feedback, Parameters},
    LpSolveOutcome, RowSense,
};
use crate::pricing::phases::{self, Phase};
use crate::{invalid, Result};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Column {
    pub cost: f64,
    pub reduced_cost: f64,
    pub entries: Vec<(usize, f64)>,
    pub acceptable: bool,
    pub tag: i32,
    pub identity: String,
}

/// Exact membership key. Formatting floating-point values is not injective:
/// nearby costs or coefficients must never suppress an improving column.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ColumnKey {
    Identity(String),
    Structural {
        tag: i32,
        cost: u64,
        entries: Vec<(usize, u64)>,
    },
}

impl Column {
    fn signature(&self) -> ColumnKey {
        if !self.identity.is_empty() {
            return ColumnKey::Identity(self.identity.clone());
        }
        ColumnKey::Structural {
            // Tags may carry different semantic upper bounds even for equal
            // algebraic columns, so the fallback must preserve them as well.
            tag: self.tag,
            // Signed zeros denote the same objective/zero coefficient.
            cost: if self.cost == 0. {
                0
            } else {
                self.cost.to_bits()
            },
            entries: self
                .entries
                .iter()
                .filter(|(_, value)| *value != 0.)
                .map(|&(row, value)| (row, value.to_bits()))
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Answer {
    pub columns: Vec<Column>,
    pub minimum: f64,
    pub exact: bool,
    pub bound_minimum: Option<f64>,
}

/// Outcome of a zero-real-objective feasibility pricing round. This is
/// independent of convergence or bounds for the original objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeasibilityStatus {
    Feasible,
    Infeasible,
    Stalled,
    TimeLimit,
    IterationLimit,
}
impl FeasibilityStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Feasible => "feasible",
            Self::Infeasible => "infeasible",
            Self::Stalled => "stalled",
            Self::TimeLimit => "time_limit",
            Self::IterationLimit => "iteration_limit",
        }
    }
}

#[derive(Clone, Debug)]
pub struct FeasibilityReport {
    pub status: FeasibilityStatus,
    pub iterations: usize,
    pub pricing_calls: usize,
    pub simplex_iterations: usize,
    pub ipm_iterations: usize,
    pub seconds: f64,
    pub artificial_activity: Option<f64>,
    /// A bound for artificial activity, never an original-objective bound.
    pub lower_bound: Option<f64>,
    /// True only when original-domain directed arithmetic justifies infeasibility.
    pub certified_infeasible: bool,
}

/// Original-objective columns recovered on a private phase-I model. Tags and
/// coefficients must remain valid when the caller installs these columns.
pub struct FeasibilityRecovery {
    pub columns: Vec<Column>,
    pub report: FeasibilityReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
    TimeLimit,
    Stalled,
}

#[derive(Clone, Debug)]
pub struct Iteration {
    pub iteration: usize,
    pub lp: f64,
    pub minimum: f64,
    pub new_columns: usize,
    pub columns: usize,
    pub exact: bool,
    pub alpha: f64,
    pub effective_alpha: f64,
    pub angle: f64,
    pub misprices: i32,
    pub phase: usize,
    pub pricing_calls: usize,
    /// Gap supplied to stabilization before this iteration's LP is accepted.
    pub stabilization_gap: f64,
    /// Requested IPM relative optimality tolerance; None for accurate solves.
    pub master_gap_tolerance: Option<f64>,
}

/// Which lower bound authorizes pruning and gap closure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoundMode {
    /// Exact pricing with floating-point costs and configured tolerances.
    #[default]
    Numerical,
    /// Only a controller-validated bound; failed certification stays unknown.
    Certified,
}
impl BoundMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Numerical => "numerical",
            Self::Certified => "certified",
        }
    }
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "numerical" => Ok(Self::Numerical),
            "certified" => Ok(Self::Certified),
            _ => Err(invalid("bound mode must be numerical or certified")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct WarmStart {
    /// Selected lower bound inherited by restrictions of this node.
    /// Negative infinity denotes an unknown bound.
    pub bound: f64,
    pub bound_mode: BoundMode,
    pub numerical_bound: f64,
    pub certified_bound: f64,
    /// Accepted parent center; node schedules restart independently.
    pub stabilization: Option<DualPriceSmoothing>,
    /// Generated columns in creation order, including those outside the
    /// restricted master and duplicates created in a single oracle batch.
    pub generated_pool: Vec<Column>,
    pub columns: Vec<Option<Column>>,
    pub artificials: Vec<usize>,
    pub cut_artificials: Vec<(usize, usize)>,
    pub initial_rows: usize,
}

impl WarmStart {
    pub(crate) fn empty(bound_mode: BoundMode) -> Self {
        Self {
            bound: f64::NEG_INFINITY,
            bound_mode,
            numerical_bound: f64::NEG_INFINITY,
            certified_bound: f64::NEG_INFINITY,
            stabilization: None,
            generated_pool: vec![],
            columns: vec![],
            artificials: vec![],
            cut_artificials: vec![],
            initial_rows: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub bound_mode: BoundMode,
    /// Progress table on stderr every N iterations; None keeps CG silent.
    pub log_interval: Option<usize>,
    /// Metadata of an already populated, solved master. No duplicate base
    /// artificials are added when resuming a node.
    pub warm_start: Option<WarmStart>,
    pub smoothing: Parameters,
    pub stabilize: bool,
    pub tolerance: f64,
    pub multiplicity: f64,
    pub max_iterations: usize,
    pub time_limit: Duration,
    pub known: Vec<Column>,
    pub artificial_cost: f64,
    pub penalty_factor: f64,
    pub max_penalty_updates: usize,
    pub insert_non_negative: bool,
    pub gap_absolute: f64,
    pub gap_relative: f64,
    pub enumerated_columns_added: usize,
    pub max_cut_rounds: usize,
    /// Stop after this many rounds whose gain
    /// is below the threshold times 0.1|LP|.
    pub tailing_off_threshold: f64,
    pub tailing_off_counter: usize,
    /// Remove cut rows whose slack is basic before separating again.
    pub remove_basic_cuts: bool,
    /// Cost of the local artificials of separated rows.
    pub cut_artificial_cost: f64,
    /// cleanupRestrictedMastColumns: with more generated columns than this
    /// (0 disables), keep `column_cleanup_ratio` of them before pricing.
    pub column_cleanup_threshold: f32,
    pub column_cleanup_ratio: f32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            bound_mode: BoundMode::Numerical,
            log_interval: None,
            warm_start: None,
            smoothing: Parameters::default(),
            stabilize: true,
            tolerance: 1e-6,
            multiplicity: 1.,
            max_iterations: 100000,
            time_limit: Duration::from_secs(3600000),
            known: vec![],
            artificial_cost: 0.,
            penalty_factor: 2.,
            max_penalty_updates: 5,
            insert_non_negative: true,
            gap_absolute: 1e-6,
            gap_relative: 1e-9,
            enumerated_columns_added: 0,
            max_cut_rounds: 50,
            tailing_off_threshold: 0.02,
            tailing_off_counter: 3,
            remove_basic_cuts: false,
            cut_artificial_cost: 0.,
            column_cleanup_threshold: 0.,
            column_cleanup_ratio: 0.66,
        }
    }
}

impl Options {
    /// Retain reusable controls while moving the owned continuation into a run.
    pub(crate) fn take_continuation(&mut self) -> Self {
        let warm_start = self.warm_start.take();
        let known = std::mem::take(&mut self.known);
        let mut next = self.clone();
        next.warm_start = warm_start;
        next.known = known;
        next
    }
}

#[derive(Clone, Debug)]
pub struct Output {
    pub status: Status,
    /// Auxiliary feasibility rounds; their values never enter objective history.
    pub feasibility: Vec<FeasibilityReport>,
    /// The controller requested a resumable handoff, rather than termination.
    /// `status` remains a limit status; this flag never certifies enumeration.
    pub enumeration_requested: bool,
    /// Most recent declined in-loop transfer error; ordinary pricing remains usable.
    pub enumeration_failure: Option<String>,
    pub lp: f64,
    /// Selected bound; its guarantee is identified by `bound_mode`.
    pub bound: f64,
    pub bound_mode: BoundMode,
    /// Numerical estimate from exact pricing, never an original-domain proof.
    pub numerical_bound: f64,
    /// Controller-validated bound, or negative infinity when unknown. An
    /// integer-domain certificate may exceed the pricing LP's objective.
    pub certified_bound: f64,
    pub iterations: usize,
    pub pricing_calls: usize,
    pub columns: usize,
    pub history: Vec<Iteration>,
    pub column_tags: Vec<i32>,
    pub inserted_tags: Vec<i32>,
    pub artificials: Vec<usize>,
    pub penalty_updates: usize,
    pub pruned_by_bound: bool,
    pub cut_rounds: usize,
    pub warm_start: WarmStart,
}

mod controller;
pub use controller::{Controller, NoController};

mod driver;
mod setup;
pub use driver::optimize;
mod enumeration;
mod feasibility;

mod bounds;
pub use bounds::gap_smaller_than_tolerance;
use bounds::{lagrangian, master_reduced_cost, optimality_gap, publish_certificate};

mod cleanup;
use cleanup::cleanup_columns;
mod pool;

#[cfg(all(test, feature = "clp"))]
#[path = "../../../tests/unit/master/cg/identity_pool_tests.rs"]
mod identity_pool_tests;
