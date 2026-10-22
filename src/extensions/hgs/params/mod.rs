//! Problem parameters, instance data, and penalty coefficients for HGS.
//!
//! Mirrors `Params.h` / `Params.cpp` from the original HGS-CVRP codebase.

use std::cell::Cell;
use std::fmt;
use std::time::Instant;

use crate::inputs::Instance;

use super::circle_sector::CircleSector;
use super::matrix::Matrix;

/// Precision used to avoid numerical instabilities (HGS MY_EPSILON).
pub const MY_EPSILON: f64 = 1e-5;

/// Latest time used for clients without a time window. Time-window merges
/// use a finite horizon to keep unconstrained-window merges well defined.
pub const HORIZON: f64 = 1_000_000_000.;

/// Problem type identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProblemType {
    Vrptw, // Vehicle Routing Problem with Time Windows
    Evrp,  // Electric Vehicle Routing Problem
    Cvrp,  // Capacitated Vehicle Routing Problem
}

impl Default for ProblemType {
    fn default() -> Self {
        Self::Vrptw
    }
}

/// A single time window.
#[derive(Clone, Copy, Debug)]
pub struct TimeWindow {
    pub earliest: f64,
    pub latest: f64,
}

impl TimeWindow {
    pub fn new(earliest: impl Into<f64>, latest: impl Into<f64>) -> Self {
        Self {
            earliest: earliest.into(),
            latest: latest.into(),
        }
    }
}

/// A client (customer) in the VRP instance.
/// Index 0 is the depot; indices 1..n are customers.
#[derive(Clone, Debug)]
pub struct Client {
    /// Customer number (0 = depot).
    pub cust_num: i32,
    /// X coordinate (scaled ×10).
    pub coord_x: f64,
    /// Y coordinate (scaled ×10).
    pub coord_y: f64,
    /// Service duration (scaled ×10).
    pub service_duration: f64,
    /// Demand.
    pub demand: i32,
    /// Earliest arrival time (scaled ×10).
    pub earliest_arrival: f64,
    /// Latest arrival time (scaled ×10).
    pub latest_arrival: f64,
    /// Release time (route cannot depart before this).
    pub release_time: f64,
    /// Polar angle around depot in [0, 65535].
    pub polar_angle: u16,
    /// Multiple time windows (empty for standard VRPTW).
    pub time_windows: Vec<TimeWindow>,
    /// Number of time windows.
    pub n_tw: i32,
}

impl Default for Client {
    fn default() -> Self {
        Self {
            cust_num: 0,
            coord_x: 0.0,
            coord_y: 0.0,
            service_duration: 0.,
            demand: 0,
            earliest_arrival: 0.,
            latest_arrival: HORIZON,
            release_time: 0.,
            polar_angle: 0,
            time_windows: Vec::new(),
            n_tw: 0,
        }
    }
}

/// Clarke & Wright savings structure.
#[derive(Clone, Debug)]
pub struct Savings {
    /// Client 1.
    pub c1: i32,
    /// Client 2.
    pub c2: i32,
    /// Cost savings if two routes are concatenated through edge (c1, c2).
    pub value: f64,
}

/// Configuration parameters for the HGS algorithm.
#[derive(Clone, Debug)]
pub struct Config {
    /// Iterations without improvement until termination.
    pub nb_iter: i32,
    /// CPU time limit in seconds (0 = infinity).
    pub time_limit: f64,
    /// Use wall-clock time instead of CPU time.
    pub use_wall_clock_time: bool,
    /// Path to Best Known Solution file.
    pub path_bks: String,

    // ── Construction heuristic fractions ─────────────────────────────
    pub fraction_generated_nearest: f64,
    pub fraction_generated_furthest: f64,
    pub fraction_generated_sweep: f64,
    pub fraction_generated_randomly: f64,
    pub fraction_generated_mdm: f64,
    pub min_sweep_fill_percentage: i32,
    pub max_tolerated_capacity_violation: i32,
    pub max_tolerated_time_warp: i32,
    pub initial_time_warp_penalty: f64,
    pub penalty_booster: f64,

    // ── Genetic algorithm parameters ─────────────────────────────────
    pub minimum_population_size: i32,
    pub generation_size: i32,
    pub nb_elite: i32,
    pub nb_close: i32,
    pub target_feasible: f64,
    pub repair_probability: i32,
    pub grow_nb_granular_after_non_improvement_iterations: i32,
    pub grow_nb_granular_after_iterations: i32,
    pub grow_nb_granular_size: i32,
    pub grow_population_after_non_improvement_iterations: i32,
    pub grow_population_after_iterations: i32,
    pub grow_population_size: i32,
    pub diversity_weight: f64,
    pub initial_solution: String,

    // ── Other parameters ─────────────────────────────────────────────
    pub nb_veh: i32,
    pub log_pool_interval: i32,
    pub is_dimacs_run: bool,
    pub use_dynamic_parameters: bool,
    pub path_solution: String,
    pub nb_granular: i32,
    pub intensification_probability_ls: i32,
    pub use_swap_star_tw: bool,
    pub skip_swap_star_dist: bool,
    pub circle_sector_overlap_tolerance_degrees: i32,
    pub min_circle_sector_size_degrees: i32,
    pub seed: i32,
    pub path_instance: String,
    pub use_symmetric_correlated_vertices: bool,
    pub do_repeat_until_time_limit: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            nb_iter: 20000,
            time_limit: 0.0, // 0 = infinity (checked separately)
            use_wall_clock_time: false,
            path_bks: String::new(),

            fraction_generated_nearest: 0.05,
            fraction_generated_furthest: 0.05,
            fraction_generated_sweep: 0.05,
            fraction_generated_randomly: 0.60,
            fraction_generated_mdm: 0.25,
            min_sweep_fill_percentage: 60,
            max_tolerated_capacity_violation: 50,
            max_tolerated_time_warp: 100,
            initial_time_warp_penalty: 1.0,
            penalty_booster: 2.0,

            minimum_population_size: 25,
            generation_size: 40,
            nb_elite: 4,
            nb_close: 5,
            target_feasible: 0.2,
            repair_probability: 50,
            grow_nb_granular_after_non_improvement_iterations: 5000,
            grow_nb_granular_after_iterations: 0,
            grow_nb_granular_size: 0,
            grow_population_after_non_improvement_iterations: 5000,
            grow_population_after_iterations: 0,
            grow_population_size: 0,
            diversity_weight: 0.0,
            initial_solution: String::new(),

            nb_veh: i32::MAX,
            log_pool_interval: 0,
            is_dimacs_run: false,
            use_dynamic_parameters: false,
            path_solution: String::new(),
            nb_granular: 40,
            intensification_probability_ls: 15,
            use_swap_star_tw: true,
            skip_swap_star_dist: false,
            circle_sector_overlap_tolerance_degrees: 0,
            min_circle_sector_size_degrees: 15,
            seed: 0,
            path_instance: String::new(),
            use_symmetric_correlated_vertices: false,
            do_repeat_until_time_limit: true,
        }
    }
}

// Preserve the original HGS import path.
pub use crate::rng::Xoroshiro128Plus;

/// All parameters and instance data needed by the HGS algorithm.
pub struct Params {
    /// Algorithm configuration.
    pub config: Config,
    /// Random number generator (interior mutability for shared access).
    pub rng: std::cell::RefCell<Xoroshiro128Plus>,
    /// Start time for elapsed-time tracking.
    start_time: Instant,

    // ── Adaptive penalty coefficients ────────────────────────────────
    pub penalty_capacity: Cell<f64>,
    pub penalty_wait_time: f64,
    pub penalty_time_warp: Cell<f64>,

    // ── Proximity weights ────────────────────────────────────────────
    pub proximity_weight_wait_time: f64,
    pub proximity_weight_time_warp: f64,

    /// Problem type.
    pub problem_type: ProblemType,

    // ── Instance data ────────────────────────────────────────────────
    pub instance_name: String,
    pub is_duration_constrained: bool,
    pub is_time_window_constrained: bool,
    pub is_explicit_distance_matrix: bool,
    pub nb_clients: i32, // excluding depot
    pub nb_vehicles: i32,
    pub duration_limit: f64,
    pub vehicle_capacity: i32,
    pub total_demand: i32,
    pub max_demand: i32,
    pub max_dist: f64,
    /// Clients including depot at index 0.
    pub cli: Vec<Client>,
    /// Distance/time matrix (including depot).
    pub time_cost: Matrix,
    /// For each client, other clients sorted by proximity.
    pub order_proximities: Vec<Vec<(f64, i32)>>,
    /// Correlated vertices (neighbourhood restrictions for granular search).
    pub correlated_vertices: Vec<Vec<i32>>,
    /// Circle sector overlap tolerance in [0, 65535].
    pub circle_sector_overlap_tolerance: u16,
    /// Minimum circle sector size in [0, 65535].
    pub min_circle_sector_size: u16,

    /// Clarke & Wright savings list (sorted by value descending).
    pub savings_list: Vec<Savings>,

    // ── Iterative CG integration fields ──────────────────────────────
    /// Node duals from master LP (customer numbers are 0-based indices into this vec).
    pub duals_ptr: Option<Vec<f64>>,
    /// Forbidden edges for split algorithm (legacy, stored as customer number pairs).
    pub forbidden_edges_set: Option<std::collections::HashSet<u64>>,

    /// Whether to log verbose output.
    pub verbose: bool,
}

/// Data extracted from a BALDES `Instance` for HGS construction.
#[derive(Clone, Debug)]
pub struct InstanceData {
    pub n_v: i32, // number of vehicles
    pub q: i32,   // vehicle capacity
    pub problem_type: ProblemType,
    pub x_coord: Vec<f64>,
    pub y_coord: Vec<f64>,
    pub demand: Vec<i32>,
    pub window_open: Vec<i32>,
    pub window_close: Vec<i32>,
    pub service_time: Vec<i32>,
    pub n_tw: Vec<i32>,
    pub time_windows: Vec<Vec<(i32, i32)>>,
    pub distances: Option<Vec<Vec<f64>>>,
}

impl fmt::Debug for Params {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Params")
            .field("config", &self.config)
            .field("nb_clients", &self.nb_clients)
            .field("nb_vehicles", &self.nb_vehicles)
            .field("vehicle_capacity", &self.vehicle_capacity)
            .field("total_demand", &self.total_demand)
            .field("penalty_capacity", &self.penalty_capacity)
            .field("penalty_time_warp", &self.penalty_time_warp)
            .field("problem_type", &self.problem_type)
            .finish()
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/extensions/hgs/params/tests.rs"]
mod tests;

mod construction;
mod runtime;
