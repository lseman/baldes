//! Root driver and cut-family configuration.
use super::*;
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
    /// Attach the time resource used by Kohl 2-path separator.
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
pub struct NativeKPathCutConfig {
    pub graphs: Vec<KPathGraph>,
    /// Original zero-based packing/covering set order; no depot entry.
    pub demands: Vec<f64>,
    pub capacity: f64,
    pub use_packing_sets: bool,
    pub two_path_resource: i32,
    pub parameters: KPathParameters,
}
impl NativeKPathCutConfig {
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
pub(super) enum RootKPathSeparator {
    Standalone(KPathSeparator),
    Native {
        separator: NativeKPathSeparator,
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
#[derive(Clone, Debug)]
pub struct RootOptions {
    /// Retain an independently forkable model and active cuts for descendants.
    pub capture_state: bool,
    /// Node depth: capacity/Kohl priority is 3 at root and 1 below it,
    /// as in generic capacity-cut registration.
    pub node_depth: usize,
    /// Lowest pricing phase evaluated (0 for a certified node solve).
    /// Strong-branching heuristic trials may stop above the exact phase.
    pub pricing_phase_floor: usize,
    pub cg: Options,
    pub pricing_options: Vec<(String, f64)>,
    pub initial_routes: Vec<Route>,
    pub incumbent: f64,
    pub enumerate: bool,
    pub round_reduced_costs: bool,
    pub cuts: CutOptions,
    pub capacity_cuts: Option<CapacityCutConfig>,
    pub kpath_cuts: Option<KPathCutConfig>,
    pub native_kpath_cuts: Option<NativeKPathCutConfig>,
    pub knapsack_cuts: Option<KnapsackCutConfig>,
    /// ColGenSpRelaxationImprovementPriority: separator priority of dynamic
    /// NG strengthening (0 disables it, default). Its mode and caps
    /// are the pricing options RCSPdynamicNGmode, RCSPmaxNGneighbourhoodSize,
    /// RCSPmaxNGaverNeighbourhoodSize, RCSPmaxNumOfColumnsInDynamicNG and
    /// RCSPmaxCycleSizeInDynamicNG.
    pub relaxation_improvement_priority: f32,
}
impl Default for RootOptions {
    fn default() -> Self {
        let mut cg = Options::default();
        cg.artificial_cost = 2.76;
        // master column clean-up: threshold 10000, keep ratio 0.66.
        cg.column_cleanup_threshold = 10000.;
        cg.column_cleanup_ratio = 0.66;
        Self {
            capture_state: false,
            node_depth: 0,
            pricing_phase_floor: 0,
            cg,
            pricing_options: vec![],
            initial_routes: vec![],
            incumbent: f64::INFINITY,
            enumerate: true,
            round_reduced_costs: true,
            cuts: CutOptions::default(),
            capacity_cuts: None,
            kpath_cuts: None,
            native_kpath_cuts: None,
            knapsack_cuts: None,
            relaxation_improvement_priority: 0.,
        }
    }
}
impl RootOptions {
    pub fn with_capacity_cuts(mut self, config: CapacityCutConfig) -> Self {
        self.capacity_cuts = Some(config);
        self.cg.remove_basic_cuts = true;
        self.cg.cut_artificial_cost = if self.cg.artificial_cost > 0. {
            self.cg.artificial_cost / 1.2
        } else {
            0.
        };
        self.cg.tailing_off_counter = 1_000_000;
        self
    }
    pub fn with_kpath_cuts(mut self, config: KPathCutConfig) -> Self {
        self.kpath_cuts = Some(config);
        self.native_kpath_cuts = None;
        self.cg.remove_basic_cuts = true;
        self.cg.cut_artificial_cost = if self.cg.artificial_cost > 0. {
            self.cg.artificial_cost / 1.2
        } else {
            0.
        };
        self.cg.tailing_off_counter = 1_000_000;
        self
    }
    pub fn with_native_kpath_cuts(mut self, config: NativeKPathCutConfig) -> Self {
        self.native_kpath_cuts = Some(config);
        self.kpath_cuts = None;
        self.cg.remove_basic_cuts = true;
        self.cg.cut_artificial_cost = if self.cg.artificial_cost > 0. {
            self.cg.artificial_cost / 1.2
        } else {
            0.
        };
        self.cg.tailing_off_counter = 1_000_000;
        self
    }
    pub fn with_knapsack_cuts(mut self, config: KnapsackCutConfig) -> Self {
        self.knapsack_cuts = Some(config);
        self.cg.remove_basic_cuts = true;
        self.cg.cut_artificial_cost = if self.cg.artificial_cost > 0. {
            self.cg.artificial_cost / 1.2
        } else {
            0.
        };
        self.cg.tailing_off_counter = 1_000_000;
        self
    }
    /// The C++ driver's master options with rank-one separation: basic cut
    /// rows are removed, cut artificials cost artificial_cost / 1.2, and the
    /// separator owns tailing off.
    pub fn with_rank_one_cuts(mut self) -> Self {
        self.cuts.enabled = true;
        self.cg.remove_basic_cuts = true;
        self.cg.cut_artificial_cost = if self.cg.artificial_cost > 0. {
            self.cg.artificial_cost / 1.2
        } else {
            0.
        };
        self.cg.tailing_off_counter = 1_000_000;
        self
    }
}
