//! Problem parameters, instance data, and penalty coefficients for HGS.
//!
//! Mirrors `Params.h` / `Params.cpp` from the original HGS-CVRP codebase.

use std::cell::Cell;
use std::fmt;
use std::time::Instant;

use crate::inputs::{Instance, Rounding};

use super::circle_sector::CircleSector;
use super::matrix::Matrix;

/// Precision used to avoid numerical instabilities (HGS MY_EPSILON).
pub const MY_EPSILON: f64 = 1e-5;

/// Latest time used for clients without a time window. Time-window merges
/// add and subtract several times, so i32::MAX would overflow.
pub const HORIZON: i32 = 1_000_000_000;

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
    pub earliest: i32,
    pub latest: i32,
}

impl TimeWindow {
    pub fn new(earliest: i32, latest: i32) -> Self {
        Self { earliest, latest }
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
    pub service_duration: i32,
    /// Demand.
    pub demand: i32,
    /// Earliest arrival time (scaled ×10).
    pub earliest_arrival: i32,
    /// Latest arrival time (scaled ×10).
    pub latest_arrival: i32,
    /// Release time (route cannot depart before this).
    pub release_time: i32,
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
            service_duration: 0,
            demand: 0,
            earliest_arrival: 0,
            latest_arrival: HORIZON,
            release_time: 0,
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

/// SplitMix64 step; advances `x` like the C++ reference parameter.
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Xoroshiro128+ with SplitMix64 seeding, as BALDES's C++ `math/RNG.h`.
#[derive(Clone, Debug)]
pub struct Xoroshiro128Plus {
    s0: u64,
    s1: u64,
}

impl Xoroshiro128Plus {
    /// Seed as the C++ constructor, which converts the int seed to uint64.
    pub fn new(seed: i32) -> Self {
        let mut x = seed as i64 as u64;
        let mut s0 = splitmix64(&mut x);
        // SplitMix64 takes its argument by reference, so seeding the second
        // word also advances the first one.
        let s1 = splitmix64(&mut s0);
        Self { s0, s1 }
    }

    /// Generate the next random u64.
    #[inline]
    pub fn next(&mut self) -> u64 {
        let s0 = self.s0;
        let mut s1 = self.s1;
        let result = s0.wrapping_add(s1);
        s1 ^= s0;
        self.s0 = s0.rotate_left(55) ^ s1 ^ (s1 << 14);
        self.s1 = s1.rotate_left(36);
        result
    }

    /// `rng() % max`, as HGS draws bounded values (no rejection sampling).
    #[inline]
    pub fn next_u64(&mut self, max: u64) -> u64 {
        self.next() % max
    }

    /// Random f64 in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Pick a random element from a slice.
    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        if items.is_empty() {
            panic!("cannot choose from empty slice");
        }
        let idx = self.next_u64(items.len() as u64) as usize;
        &items[idx]
    }

    /// Uniform value in [0, range) as libstdc++'s uniform_int_distribution
    /// with a 64-bit generator (Lemire's nearly divisionless method).
    pub fn uniform_below(&mut self, range: u64) -> u64 {
        let mut product = self.next() as u128 * range as u128;
        if (product as u64) < range {
            let threshold = range.wrapping_neg() % range;
            while (product as u64) < threshold {
                product = self.next() as u128 * range as u128;
            }
        }
        (product >> 64) as u64
    }

    /// Shuffle a slice in place as libstdc++'s std::shuffle: two swap
    /// positions per draw when the generator range allows it.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        let n = items.len() as u64;
        if n == 0 {
            return;
        }
        if u64::MAX / n >= n {
            let mut i = 1;
            if n % 2 == 0 {
                items.swap(1, self.uniform_below(2) as usize);
                i = 2;
            }
            while i < n {
                let swap_range = i + 1;
                let x = self.uniform_below(swap_range * (swap_range + 1));
                items.swap(i as usize, (x / (swap_range + 1)) as usize);
                items.swap(i as usize + 1, (x % (swap_range + 1)) as usize);
                i += 2;
            }
        } else {
            for i in 1..n {
                items.swap(i as usize, self.uniform_below(i + 1) as usize);
            }
        }
    }
}

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
    pub nb_clients: i32,          // excluding depot
    pub nb_vehicles: i32,
    pub duration_limit: i32,
    pub vehicle_capacity: i32,
    pub total_demand: i32,
    pub max_demand: i32,
    pub max_dist: i32,
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
    pub n_v: i32,          // number of vehicles
    pub q: i32,            // vehicle capacity
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

impl Params {
    /// Create Params from a BALDES Instance. Coordinates, times and explicit
    /// distances are scaled by 10 and truncated, as in HGS-VRPTW.
    pub fn from_instance(instance: &Instance) -> Self {
        let scale = |value: f64| (value * 10.0) as i32;
        let mut has_time_windows = false;
        let cli = instance
            .customers
            .iter()
            .enumerate()
            .map(|(i, c)| {
                has_time_windows |= c.due.is_finite();
                Client {
                    cust_num: i as i32,
                    coord_x: c.x * 10.0,
                    coord_y: c.y * 10.0,
                    service_duration: scale(c.service),
                    demand: c.demand as i32,
                    earliest_arrival: scale(c.ready),
                    latest_arrival: if c.due.is_finite() { scale(c.due).min(HORIZON) } else { HORIZON },
                    ..Client::default()
                }
            })
            .collect();
        // Explicit or rounded distances (CVRPLIB) are used as given; unrounded
        // Solomon distances are truncated to tenths from the coordinates.
        let n = instance.customers.len();
        let distances = (instance.distances.is_some() || instance.rounding != Rounding::None).then(|| {
            (0..n)
                .map(|i| (0..n).map(|j| (instance.distance(i, j, instance.rounding) * 10.0).round()).collect())
                .collect::<Vec<Vec<f64>>>()
        });
        let problem_type = if has_time_windows { ProblemType::Vrptw } else { ProblemType::Cvrp };
        let mut params = Self::build(
            cli,
            instance.vehicles as i32,
            instance.capacity as i32,
            problem_type,
            distances.as_ref(),
        );
        params.instance_name = instance.name.clone();
        params
    }

    /// Create Params from raw instance data. Coordinates are scaled by 10;
    /// times are taken as given.
    pub fn from_instance_data(data: &InstanceData) -> Self {
        let cli = (0..data.x_coord.len())
            .map(|i| Client {
                cust_num: i as i32,
                coord_x: data.x_coord[i] * 10.0,
                coord_y: data.y_coord[i] * 10.0,
                service_duration: data.service_time[i],
                demand: data.demand[i],
                earliest_arrival: data.window_open[i],
                latest_arrival: data.window_close[i].min(HORIZON),
                time_windows: if data.n_tw[i] > 0 {
                    data.time_windows[i].iter().map(|&(e, l)| TimeWindow::new(e, l)).collect()
                } else {
                    Vec::new()
                },
                n_tw: data.n_tw[i],
                ..Client::default()
            })
            .collect();
        Self::build(cli, data.n_v, data.q, data.problem_type, data.distances.as_ref())
    }

    /// Shared part of the constructors (HGS Params::Params). `nb_vehicles` 0
    /// means the default fleet ceil(1.3 * demand / capacity) + 3.
    fn build(
        mut cli: Vec<Client>,
        nb_vehicles: i32,
        vehicle_capacity: i32,
        problem_type: ProblemType,
        distances: Option<&Vec<Vec<f64>>>,
    ) -> Self {
        let config = Config::default();
        let n = cli.len();
        let nb_clients = n as i32 - 1;

        let (depot_x, depot_y) = (cli[0].coord_x, cli[0].coord_y);
        let mut total_demand = 0;
        let mut max_demand = 0;
        for client in &mut cli {
            let angle = 32768. * (client.coord_y - depot_y).atan2(client.coord_x - depot_x) / std::f64::consts::PI;
            client.polar_angle = CircleSector::positive_mod(angle as i32);
            max_demand = max_demand.max(client.demand);
            total_demand += client.demand;
        }
        let nb_vehicles = if nb_vehicles <= 0 {
            ((1.3 * total_demand as f64 / vehicle_capacity as f64).ceil() + 3.) as i32
        } else {
            nb_vehicles
        };

        let mut time_cost = Matrix::new(n);
        let mut max_dist = 0;
        for i in 0..n {
            for j in 0..n {
                let cost = if i == j {
                    0
                } else if let Some(distances) = distances {
                    distances[i][j] as i32
                } else {
                    let (dx, dy) = (cli[i].coord_x - cli[j].coord_x, cli[i].coord_y - cli[j].coord_y);
                    (dx * dx + dy * dy).sqrt() as i32
                };
                time_cost.set(i, j, cost);
                max_dist = max_dist.max(cost);
            }
        }

        // Neighbors by distance plus the cheaper of the two waiting/time-warp
        // orientations (Vidal et al. 2013).
        let (weight_wait, weight_warp) = (0.2, 1.0);
        let mut order_proximities = vec![Vec::new(); n];
        for i in 1..n {
            for j in (1..n).filter(|&j| j != i) {
                let t = time_cost.get(i, j) as i64;
                let (a, b) = (&cli[i], &cli[j]);
                let orient = |a: &Client, b: &Client| {
                    let wait = (b.earliest_arrival as i64 - t - a.service_duration as i64 - a.latest_arrival as i64).max(0);
                    let warp = (a.earliest_arrival as i64 + a.service_duration as i64 + t - b.latest_arrival as i64).max(0);
                    weight_wait * wait as f64 + weight_warp * warp as f64
                };
                order_proximities[i].push((t as f64 + orient(a, b).min(orient(b, a)), j as i32));
            }
            order_proximities[i].sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        }

        let mut savings_list = Vec::with_capacity(n * n.saturating_sub(1) / 2);
        for i in 1..n {
            for j in 1..i {
                let value = (time_cost.get(0, i) + time_cost.get(0, j) - time_cost.get(i, j)) as f64;
                savings_list.push(Savings { c1: i as i32, c2: j as i32, value });
            }
        }
        savings_list.sort_by(|a, b| b.value.total_cmp(&a.value));

        let penalty_capacity = (max_dist as f64 / max_demand.max(1) as f64).clamp(0.1, 1000.);
        let penalty_time_warp = match problem_type {
            ProblemType::Vrptw | ProblemType::Evrp => config.initial_time_warp_penalty,
            ProblemType::Cvrp => 0.,
        };
        let degrees_to_ring = |degrees: i32| (degrees as f64 / 360. * 65536.) as u16;
        let mut params = Params {
            rng: std::cell::RefCell::new(Xoroshiro128Plus::new(config.seed)),
            start_time: Instant::now(),
            penalty_capacity: Cell::new(penalty_capacity),
            penalty_wait_time: 0.0,
            penalty_time_warp: Cell::new(penalty_time_warp),
            proximity_weight_wait_time: weight_wait,
            proximity_weight_time_warp: weight_warp,
            problem_type,
            instance_name: String::new(),
            is_duration_constrained: false,
            is_time_window_constrained: problem_type != ProblemType::Cvrp,
            is_explicit_distance_matrix: distances.is_some(),
            nb_clients,
            nb_vehicles,
            duration_limit: i32::MAX,
            vehicle_capacity,
            total_demand,
            max_demand,
            max_dist,
            cli,
            time_cost,
            order_proximities,
            correlated_vertices: Vec::new(),
            circle_sector_overlap_tolerance: degrees_to_ring(config.circle_sector_overlap_tolerance_degrees),
            min_circle_sector_size: degrees_to_ring(config.min_circle_sector_size_degrees),
            savings_list,
            duals_ptr: None,
            forbidden_edges_set: None,
            verbose: true,
            config,
        };
        params.correlated_vertices = params.correlated_vertices_for(params.config.nb_granular);
        params
    }

    /// The `nb_granular` nearest neighbors of each client by proximity, in
    /// increasing index order (HGS SetCorrelatedVertices).
    pub fn correlated_vertices_for(&self, nb_granular: i32) -> Vec<Vec<i32>> {
        let n = self.nb_clients as usize;
        let count = (nb_granular.max(0) as usize).min(n.saturating_sub(1));
        let mut sets = vec![std::collections::BTreeSet::new(); n + 1];
        for i in 1..=n {
            for &(_, j) in &self.order_proximities[i][..count] {
                sets[i].insert(j);
                if self.config.use_symmetric_correlated_vertices {
                    sets[j as usize].insert(i as i32);
                }
            }
        }
        sets.into_iter().map(|set| set.into_iter().collect()).collect()
    }

    /// Set the seed of the random number generator.
    pub fn set_seed(&mut self, seed: i32) {
        self.config.seed = seed;
        self.rng = std::cell::RefCell::new(Xoroshiro128Plus::new(seed));
    }

    /// Restart the clock used by the time limit.
    pub fn reset_clock(&mut self) {
        self.start_time = Instant::now();
    }

    /// Elapsed seconds since creation.
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Whether the time limit has been exceeded.
    pub fn is_time_limit_exceeded(&self) -> bool {
        if self.config.time_limit <= 0.0 {
            return false;
        }
        self.elapsed_secs() >= self.config.time_limit
    }

    /// Update adaptive penalty coefficients based on solution feasibility history.
    /// 
    /// This implements the penalty adaptation scheme from Vidal 2012.
    pub fn update_penalties(&mut self, num_feasible: usize, num_infeasible: usize, 
                           best_capacity_excess: i32, best_time_warp: i32) {
        let total = num_feasible + num_infeasible;
        if total == 0 {
            return;
        }

        let ratio_feasible = num_feasible as f64 / total as f64;
        
        // Adapt capacity penalty
        if best_capacity_excess > 0 {
            self.penalty_capacity.set(self.penalty_capacity.get() * 1.5);
        } else if best_capacity_excess == 0 && ratio_feasible > self.config.target_feasible {
            self.penalty_capacity.set(self.penalty_capacity.get() / 1.5);
        }

        // Adapt time warp penalty
        if best_time_warp > 0 {
            self.penalty_time_warp.set(self.penalty_time_warp.get() * 1.5);
        } else if best_time_warp == 0 && ratio_feasible > self.config.target_feasible {
            self.penalty_time_warp.set(self.penalty_time_warp.get() / 1.5);
        }

        // Ensure penalties stay in reasonable bounds
        self.penalty_capacity.set(self.penalty_capacity.get().max(0.1).min(1000.0));
        self.penalty_time_warp.set(self.penalty_time_warp.get().max(0.1).min(1000.0));

        // Penalty booster: if no feasible solutions, increase penalties faster
        if num_feasible == 0 && self.config.penalty_booster > 0.0 {
            self.penalty_capacity.set(self.penalty_capacity.get() * self.config.penalty_booster);
            self.penalty_time_warp.set(self.penalty_time_warp.get() * self.config.penalty_booster);
        }
    }

    /// Format as a human-readable string for logging.
    pub fn summary(&self) -> String {
        format!(
            "Params{{{} clients, {} vehicles, capacity={}, total_demand={}, \
             penalty_cap={:.2}, penalty_tw={:.2}}}",
            self.nb_clients,
            self.nb_vehicles,
            self.vehicle_capacity,
            self.total_demand,
            self.penalty_capacity.get(),
            self.penalty_time_warp.get(),
        )
    }
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
mod tests {
    use super::*;
    use crate::inputs::Customer;

    fn make_test_instance() -> Instance {
        // Simple 5-customer instance (index 0 = depot)
        Instance {
            name: "test-5".to_string(),
            vehicles: 3,
            capacity: 100.0,
            customers: vec![
                Customer { x: 0.0, y: 0.0, demand: 0.0, ready: 0.0, due: i32::MAX as f64, service: 0.0 }, // depot
                Customer { x: 10.0, y: 0.0, demand: 10.0, ready: 0.0, due: 1000.0, service: 5.0 },
                Customer { x: 20.0, y: 0.0, demand: 20.0, ready: 0.0, due: 1000.0, service: 10.0 },
                Customer { x: 30.0, y: 0.0, demand: 15.0, ready: 0.0, due: 1000.0, service: 5.0 },
                Customer { x: 40.0, y: 0.0, demand: 25.0, ready: 0.0, due: 1000.0, service: 10.0 },
            ],
            rounding: crate::inputs::Rounding::None,
            distances: None,
        }
    }

    #[test]
    fn create_params_from_instance() {
        let instance = make_test_instance();
        let params = Params::from_instance(&instance);
        
        assert_eq!(params.nb_clients, 4);
        assert_eq!(params.nb_vehicles, 3);
        assert_eq!(params.vehicle_capacity, 100);
        assert_eq!(params.cli.len(), 5); // including depot
        assert_eq!(params.cli[0].cust_num, 0); // depot
        assert_eq!(params.time_cost.size(), 5);
    }

    #[test]
    fn params_time_limit() {
        let instance = make_test_instance();
        let mut params = Params::from_instance(&instance);
        
        // No time limit → never exceeded
        assert!(!params.is_time_limit_exceeded());
        
        // Zero time limit → treated as infinity
        params.config.time_limit = 0.0;
        assert!(!params.is_time_limit_exceeded());
    }

    #[test]
    fn penalty_update_increases_on_violation() {
        let instance = make_test_instance();
        let mut params = Params::from_instance(&instance);
        
        let old_cap = params.penalty_capacity.get();
        params.update_penalties(0, 10, 5, 0); // infeasible due to capacity
        
        assert!(params.penalty_capacity.get() > old_cap, 
                "capacity penalty should increase when there's excess load");
    }

    #[test]
    fn penalty_update_decreases_when_feasible() {
        let instance = make_test_instance();
        let mut params = Params::from_instance(&instance);
        
        let old_cap = params.penalty_capacity.get();
        // All feasible, above target ratio → decrease penalty
        params.update_penalties(10, 0, 0, 0);
        
        assert!(params.penalty_capacity.get() < old_cap, 
                "capacity penalty should decrease when all solutions are feasible");
    }

    #[test]
    fn savings_list_sorted_descending() {
        let instance = make_test_instance();
        let params = Params::from_instance(&instance);
        
        for i in 1..params.savings_list.len() {
            assert!(params.savings_list[i].value <= params.savings_list[i-1].value,
                    "savings list should be sorted descending");
        }
    }

    #[test]
    fn rng_matches_cpp_reference() {
        // Values from BALDES's C++ math/RNG.h and libstdc++ 16 std::shuffle.
        let mut rng = Xoroshiro128Plus::new(0);
        assert_eq!(
            [rng.next(), rng.next(), rng.next()],
            [2836985337983649843, 17383111449422238058, 15852657771511707500]
        );
        assert_eq!(rng.next_u64(100), 86);
        let mut shuffled: Vec<Vec<usize>> = [1, 2, 10].iter().map(|&n| (0..n).collect()).collect();
        for v in &mut shuffled {
            rng.shuffle(v);
        }
        assert_eq!(shuffled, [vec![0], vec![1, 0], vec![7, 5, 9, 2, 3, 4, 1, 0, 8, 6]]);
    }

    #[test]
    fn polar_angles_computed() {
        let instance = make_test_instance();
        let params = Params::from_instance(&instance);
        
        // Depot angle should be 0
        assert_eq!(params.cli[0].polar_angle, 0);
        
        // Customer angles should be in [0, 65535]
        for client in params.cli.iter().skip(1) {
            assert!(client.polar_angle <= 65535);
        }
    }
}
