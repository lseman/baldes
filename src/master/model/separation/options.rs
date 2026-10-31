//! Cut-family configuration of automatic separation.
use crate::cuts::capacity::{CapacityParameters, TimeWindows};
use crate::cuts::knapsack::{KnapsackParameters, KnapsackRow};
use crate::cuts::kpath::{GeneralKPathSeparator, KPathGraph, KPathParameters, KPathSeparator};

/// Rank-one cut controls of the root driver.
#[derive(Clone, Debug)]
pub struct CutOptions {
    pub enabled: bool,
    pub tailing_off_threshold: f64,
    pub tailing_off_counter: usize,
    pub min_rounds_before_stop_by_sp: usize,
}
impl Default for CutOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            tailing_off_threshold: 0.02,
            tailing_off_counter: 3,
            min_rounds_before_stop_by_sp: 1,
        }
    }
}
#[derive(Clone, Debug)]
pub struct CapacityCutConfig {
    /// Depot at index 0, customers at indices 1..n; sink is n+1.
    pub demands: Vec<f64>,
    pub capacity: f64,
    pub parameters: CapacityParameters,
    pub time_windows: Option<TimeWindows>,
}
impl CapacityCutConfig {
    /// Attach the time resource used by the Kohl 2-path separator.
    /// These cuts share the capacity family's separation priority and pool.
    pub fn with_time_windows(mut self, windows: TimeWindows) -> Self {
        self.parameters.use_two_path = true;
        self.time_windows = Some(windows);
        self
    }
    pub fn new(demands: Vec<f64>, capacity: f64) -> Self {
        let mut parameters = CapacityParameters::default();
        // GenericRCSPCapacityCutConstr's packing form, no Kohl resource.
        parameters.complement_form = true;
        parameters.use_two_path = false;
        Self {
            demands,
            capacity,
            parameters,
            time_windows: None,
        }
    }
}
#[derive(Clone, Debug)]
pub struct KPathCutConfig {
    /// Depot at index 0, customers at indices 1..n; sink is n+1.
    pub demands: Vec<f64>,
    pub capacity: f64,
    pub parameters: KPathParameters,
    pub time_windows: Option<TimeWindows>,
}
impl KPathCutConfig {
    pub fn new(demands: Vec<f64>, capacity: f64) -> Self {
        Self {
            demands,
            capacity,
            parameters: KPathParameters::default(),
            time_windows: None,
        }
    }
    pub fn with_time_windows(mut self, windows: TimeWindows) -> Self {
        self.time_windows = Some(windows);
        self
    }
}
/// General native strong k-path separation on original graph/set IDs.
#[derive(Clone, Debug)]
pub struct GeneralKPathCutConfig {
    pub graphs: Vec<KPathGraph>,
    /// Original zero-based packing/covering set order; no depot entry.
    pub demands: Vec<f64>,
    pub capacity: f64,
    pub use_packing_sets: bool,
    pub two_path_resource: i32,
    pub parameters: KPathParameters,
}
impl GeneralKPathCutConfig {
    pub fn new(
        graphs: Vec<KPathGraph>,
        demands: Vec<f64>,
        capacity: f64,
        use_packing_sets: bool,
    ) -> Self {
        Self {
            graphs,
            demands,
            capacity,
            use_packing_sets,
            two_path_resource: -1,
            parameters: Default::default(),
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) enum KPathSeparation {
    Standalone(KPathSeparator),
    General {
        separator: GeneralKPathSeparator,
        graphs: Vec<KPathGraph>,
    },
}
/// Original integer resource rows used by RouteLoadKnapsackCutSeparation.
#[derive(Clone, Debug, Default)]
pub struct KnapsackCutConfig {
    /// Original resource ID -> current pricing graph's solver coordinate.
    /// Empty retains the identity mapping used by the standalone API.
    pub resource_indices: std::collections::BTreeMap<usize, usize>,
    pub rows: Vec<KnapsackRow>,
    pub parameters: KnapsackParameters,
}
impl KnapsackCutConfig {
    /// Opt in to independently validated Fenchel load-table cuts. The original
    /// rows must hold for every feasible integer master solution.
    pub fn with_fenchel(mut self, parameters: crate::cuts::knapsack::FenchelParameters) -> Self {
        self.parameters.fenchel = Some(parameters);
        self
    }
    pub fn new(rows: Vec<KnapsackRow>) -> Self {
        Self {
            rows,
            parameters: Default::default(),
            resource_indices: Default::default(),
        }
    }
    pub fn with_resource_indices(
        mut self,
        indices: std::collections::BTreeMap<usize, usize>,
    ) -> Self {
        self.resource_indices = indices;
        self
    }
}
