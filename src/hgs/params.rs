//! Problem parameters, instance data, and penalty coefficients for HGS.
//!
//! Mirrors `Params.h` / `Params.cpp` from the original HGS-CVRP codebase.

use std::fmt;
use std::time::Instant;

use crate::vrptw::Instance;

use super::circle_sector::CircleSector;
use super::matrix::Matrix;

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
            latest_arrival: i32::MAX,
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

/// Simple Xoroshiro128+ PRNG (fast, good enough for HGS).
#[derive(Clone, Debug)]
pub struct Xoroshiro128Plus {
    s0: u64,
    s1: u64,
}

impl Xoroshiro128Plus {
    /// Create a new generator seeded from `seed`.
    pub fn new(seed: i32) -> Self {
        // Simple seed mixing to avoid all-zero state
        let s0 = (seed as u64).wrapping_add(0x9E3779B97F4A7C15);
        let s1 = if seed == 0 { 1 } else { (seed as u64).wrapping_mul(6364136223846793005) };
        Self { s0, s1 }
    }

    /// Generate the next random u64.
    #[inline]
    pub fn next(&mut self) -> u64 {
        let s0 = self.s0;
        let mut s1 = self.s1;
        let result = s0.wrapping_add(s1);
        s1 ^= s0;
        self.s0 = ((s0 << 55).wrapping_add(s0)) ^ s1 ^ (s1 << 14);
        self.s1 = (s1 << 36).wrapping_add(s1);
        result
    }

    /// Random u64 in [0, max).
    #[inline]
    pub fn next_u64(&mut self, max: u64) -> u64 {
        let threshold = ((u64::MAX - max) % max).wrapping_add(1);
        loop {
            let r = self.next();
            if r >= threshold {
                return r % max;
            }
        }
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

    /// Shuffle a slice in place (Fisher-Yates).
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.next_u64((i + 1) as u64) as usize;
            items.swap(i, j);
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
    pub penalty_capacity: f64,
    pub penalty_wait_time: f64,
    pub penalty_time_warp: f64,

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
    /// Create params from a BALDES `Instance`.
    pub fn from_instance(instance: &Instance) -> Self {
        let n_customers = instance.customers.len();
        // Instance has depot at index 0 (first customer in the list is actually the depot)
        // In BALDES, customers[0] is the depot
        
        let mut cli = Vec::with_capacity(n_customers);
        
        for (i, c) in instance.customers.iter().enumerate() {
            let client = Client {
                cust_num: i as i32,
                coord_x: c.x * 10.0,
                coord_y: c.y * 10.0,
                service_duration: (c.service * 10.0) as i32,
                demand: c.demand as i32,
                earliest_arrival: (c.ready * 10.0) as i32,
                latest_arrival: (c.due * 10.0) as i32,
                release_time: 0,
                polar_angle: 0, // computed below
                time_windows: Vec::new(),
                n_tw: 0,
            };
            cli.push(client);
        }

        // Compute polar angles and track demand totals
        let mut total_demand = 0i32;
        let mut max_demand = 0i32;
        let depot_x = cli[0].coord_x;
        let depot_y = cli[0].coord_y;
        
        for client in cli.iter_mut().skip(1) { // skip depot
            let dx = client.coord_x - depot_x;
            let dy = client.coord_y - depot_y;
            let angle_degrees = (32768.0 * dy.atan2(dx) / std::f64::consts::PI).round() as i32;
            client.polar_angle = CircleSector::positive_mod(angle_degrees);
            
            if client.demand > max_demand {
                max_demand = client.demand;
            }
            total_demand += client.demand;
        }

        let nb_clients = (n_customers - 1) as i32; // exclude depot
        
        // Set default fleet size if not specified
        let nb_vehicles = if instance.vehicles == 0 {
            ((1.3 * total_demand as f64 / instance.capacity as f64).ceil() + 3.0) as i32
        } else {
            instance.vehicles as i32
        };

        // Build distance matrix
        let is_explicit = instance.distances.is_some();
        let mut time_cost = Matrix::new(n_customers);
        let mut max_dist = 0i32;
        
        if !is_explicit {
            for i in 0..n_customers {
                time_cost.set(i, i, 0);
                for j in (i + 1)..n_customers {
                    let dx = cli[i].coord_x - cli[j].coord_x;
                    let dy = cli[i].coord_y - cli[j].coord_y;
                    let d = (dx * dx + dy * dy).sqrt();
                    let cost = d as i32;
                    time_cost.set(i, j, cost);
                    time_cost.set(j, i, cost);
                    if cost > max_dist {
                        max_dist = cost;
                    }
                }
            }
        } else if let Some(ref dists) = instance.distances {
            for i in 0..n_customers {
                time_cost.set(i, i, 0);
                for j in 0..n_customers {
                    if i != j {
                        let cost = dists[i][j] as i32;
                        time_cost.set(i, j, cost);
                        if cost > max_dist {
                            max_dist = cost;
                        }
                    }
                }
            }
        }

        // Compute order proximities
        let mut order_proximities = vec![Vec::new(); n_customers];
        let pw_wait = 0.2;
        let pw_twarp = 1.0;
        
        for i in 1..n_customers {
            for j in 1..n_customers {
                if i != j {
                    let time_ij = time_cost.get(i, j);
                    let c_j = &cli[j];
                    let c_i = &cli[i];
                    
                    let prox = time_ij as f64
                        + (pw_wait * ((c_j.earliest_arrival - time_ij - c_i.service_duration - c_i.latest_arrival).max(0) as f64)
                            + pw_twarp * ((c_i.earliest_arrival + c_i.service_duration + time_ij - c_j.latest_arrival).max(0) as f64))
                        .min(
                            pw_wait * ((c_i.earliest_arrival - time_ij - c_j.service_duration - c_j.latest_arrival).max(0) as f64)
                                + pw_twarp * ((c_j.earliest_arrival + c_j.service_duration + time_ij - c_i.latest_arrival).max(0) as f64),
                        );
                    
                    order_proximities[i].push((prox, j as i32));
                }
            }
            order_proximities[i].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }

        // Compute correlated vertices (granular neighbourhood)
        let mut correlated_vertices = vec![Vec::new(); n_customers];
        for i in 1..n_customers {
            let limit = Config::default().nb_granular as usize;
            for j in 0..limit.min(order_proximities[i].len()) {
                if let Some(&(_, cust_j)) = order_proximities[i].get(j) {
                    correlated_vertices[i].push(cust_j);
                }
            }
        }

        // Build savings list (Clarke & Wright)
        let mut savings_list = Vec::new();
        for i in 1..n_customers {
            for j in 1..i {
                let value = time_cost.get(0, i) as f64
                    + time_cost.get(0, j) as f64
                    - time_cost.get(i, j) as f64;
                savings_list.push(Savings {
                    c1: i as i32,
                    c2: j as i32,
                    value,
                });
            }
        }
        savings_list.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        // Initial penalty values
        let penalty_capacity = (max_dist as f64 / max_demand.max(1) as f64).max(0.1).min(1000.0);
        let penalty_time_warp = 1.0; // VRPTW default

        Params {
            config: Config::default(),
            rng: std::cell::RefCell::new(Xoroshiro128Plus::new(0)),
            start_time: Instant::now(),
            penalty_capacity,
            penalty_wait_time: 0.0,
            penalty_time_warp,
            proximity_weight_wait_time: pw_wait,
            proximity_weight_time_warp: pw_twarp,
            problem_type: ProblemType::Vrptw,
            instance_name: instance.name.clone(),
            is_duration_constrained: false,
            is_time_window_constrained: true,
            is_explicit_distance_matrix: is_explicit,
            nb_clients,
            nb_vehicles,
            duration_limit: i32::MAX,
            vehicle_capacity: instance.capacity as i32,
            total_demand,
            max_demand,
            max_dist,
            cli,
            time_cost,
            order_proximities,
            correlated_vertices,
            circle_sector_overlap_tolerance: 0,
            min_circle_sector_size: (15i32 as f64 / 360.0 * 65536.0) as u16,
            savings_list,
            duals_ptr: None,
            forbidden_edges_set: None,
            verbose: true,
        }
    }

    /// Create params from raw instance data.
    pub fn from_instance_data(data: &InstanceData) -> Self {
        let n_customers = data.x_coord.len();
        
        let mut cli = Vec::with_capacity(n_customers);
        
        for i in 0..n_customers {
            let client = Client {
                cust_num: i as i32,
                coord_x: data.x_coord[i] * 10.0,
                coord_y: data.y_coord[i] * 10.0,
                service_duration: data.service_time[i],
                demand: data.demand[i],
                earliest_arrival: data.window_open[i],
                latest_arrival: data.window_close[i],
                release_time: 0,
                polar_angle: 0,
                time_windows: if data.n_tw[i] > 0 {
                    data.time_windows[i].iter().map(|&(e, l)| TimeWindow::new(e, l)).collect()
                } else {
                    Vec::new()
                },
                n_tw: data.n_tw[i],
            };
            cli.push(client);
        }

        // Compute polar angles and track demand totals
        let mut total_demand = 0i32;
        let mut max_demand = 0i32;
        let depot_x = cli[0].coord_x;
        let depot_y = cli[0].coord_y;
        
        for client in cli.iter_mut().skip(1) {
            let dx = client.coord_x - depot_x;
            let dy = client.coord_y - depot_y;
            let angle_degrees = (32768.0 * dy.atan2(dx) / std::f64::consts::PI).round() as i32;
            client.polar_angle = CircleSector::positive_mod(angle_degrees);
            
            if client.demand > max_demand {
                max_demand = client.demand;
            }
            total_demand += client.demand;
        }

        let nb_clients = (n_customers - 1) as i32;
        
        let nb_vehicles = if data.n_v == 0 {
            ((1.3 * total_demand as f64 / data.q as f64).ceil() + 3.0) as i32
        } else {
            data.n_v
        };

        // Build distance matrix
        let is_explicit = data.distances.is_some();
        let mut time_cost = Matrix::new(n_customers);
        let mut max_dist = 0i32;
        
        if !is_explicit {
            for i in 0..n_customers {
                time_cost.set(i, i, 0);
                for j in (i + 1)..n_customers {
                    let dx = cli[i].coord_x - cli[j].coord_x;
                    let dy = cli[i].coord_y - cli[j].coord_y;
                    let d = (dx * dx + dy * dy).sqrt();
                    let cost = d as i32;
                    time_cost.set(i, j, cost);
                    time_cost.set(j, i, cost);
                    if cost > max_dist {
                        max_dist = cost;
                    }
                }
            }
        } else if let Some(ref dists) = data.distances {
            for i in 0..n_customers {
                time_cost.set(i, i, 0);
                for j in 0..n_customers {
                    if i != j {
                        let cost = dists[i][j] as i32;
                        time_cost.set(i, j, cost);
                        if cost > max_dist {
                            max_dist = cost;
                        }
                    }
                }
            }
        }

        // Compute order proximities
        let mut order_proximities = vec![Vec::new(); n_customers];
        let pw_wait = 0.2;
        let pw_twarp = 1.0;
        
        for i in 1..n_customers {
            for j in 1..n_customers {
                if i != j {
                    let time_ij = time_cost.get(i, j);
                    let c_j = &cli[j];
                    let c_i = &cli[i];
                    
                    let prox = time_ij as f64
                        + (pw_wait * ((c_j.earliest_arrival - time_ij - c_i.service_duration - c_i.latest_arrival).max(0) as f64)
                            + pw_twarp * ((c_i.earliest_arrival + c_i.service_duration + time_ij - c_j.latest_arrival).max(0) as f64))
                        .min(
                            pw_wait * ((c_i.earliest_arrival - time_ij - c_j.service_duration - c_j.latest_arrival).max(0) as f64)
                                + pw_twarp * ((c_j.earliest_arrival + c_j.service_duration + time_ij - c_i.latest_arrival).max(0) as f64),
                        );
                    
                    order_proximities[i].push((prox, j as i32));
                }
            }
            order_proximities[i].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }

        // Compute correlated vertices
        let mut correlated_vertices = vec![Vec::new(); n_customers];
        for i in 1..n_customers {
            let limit = (data.n_v.max(0) as usize).min(n_customers - 1);
            for j in 0..limit.min(order_proximities[i].len()) {
                if let Some(&(_, cust_j)) = order_proximities[i].get(j) {
                    correlated_vertices[i].push(cust_j);
                }
            }
        }

        // Build savings list
        let mut savings_list = Vec::new();
        for i in 1..n_customers {
            for j in 1..i {
                let value = time_cost.get(0, i as usize) as f64
                    + time_cost.get(0, j as usize) as f64
                    - time_cost.get(i as usize, j as usize) as f64;
                savings_list.push(Savings {
                    c1: i as i32,
                    c2: j as i32,
                    value,
                });
            }
        }
        savings_list.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        let penalty_capacity = (max_dist as f64 / max_demand.max(1) as f64).max(0.1).min(1000.0);
        let penalty_time_warp = if matches!(data.problem_type, ProblemType::Vrptw | ProblemType::Evrp) {
            1.0
        } else {
            0.0
        };

        let circle_sector_overlap_tolerance = 0u16; // default
        let min_circle_sector_size = (15i32 as f64 / 360.0 * 65536.0) as u16;

        Params {
            config: Config::default(),
            rng: std::cell::RefCell::new(Xoroshiro128Plus::new(0)),
            start_time: Instant::now(),
            penalty_capacity,
            penalty_wait_time: 0.0,
            penalty_time_warp,
            proximity_weight_wait_time: pw_wait,
            proximity_weight_time_warp: pw_twarp,
            problem_type: data.problem_type,
            instance_name: String::new(),
            is_duration_constrained: false,
            is_time_window_constrained: true,
            is_explicit_distance_matrix: is_explicit,
            nb_clients,
            nb_vehicles,
            duration_limit: i32::MAX,
            vehicle_capacity: data.q,
            total_demand,
            max_demand,
            max_dist,
            cli,
            time_cost,
            order_proximities,
            correlated_vertices,
            circle_sector_overlap_tolerance,
            min_circle_sector_size,
            savings_list,
            duals_ptr: None,
            forbidden_edges_set: None,
            verbose: true,
        }
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
            self.penalty_capacity *= 1.5;
        } else if best_capacity_excess == 0 && ratio_feasible > self.config.target_feasible {
            self.penalty_capacity /= 1.5;
        }

        // Adapt time warp penalty
        if best_time_warp > 0 {
            self.penalty_time_warp *= 1.5;
        } else if best_time_warp == 0 && ratio_feasible > self.config.target_feasible {
            self.penalty_time_warp /= 1.5;
        }

        // Ensure penalties stay in reasonable bounds
        self.penalty_capacity = self.penalty_capacity.max(0.1).min(1000.0);
        self.penalty_time_warp = self.penalty_time_warp.max(0.1).min(1000.0);

        // Penalty booster: if no feasible solutions, increase penalties faster
        if num_feasible == 0 && self.config.penalty_booster > 0.0 {
            self.penalty_capacity *= self.config.penalty_booster;
            self.penalty_time_warp *= self.config.penalty_booster;
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
            self.penalty_capacity,
            self.penalty_time_warp,
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
    use crate::vrptw::Customer;

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
            rounding: crate::vrptw::Rounding::None,
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
        
        let old_cap = params.penalty_capacity;
        params.update_penalties(0, 10, 5, 0); // infeasible due to capacity
        
        assert!(params.penalty_capacity > old_cap, 
                "capacity penalty should increase when there's excess load");
    }

    #[test]
    fn penalty_update_decreases_when_feasible() {
        let instance = make_test_instance();
        let mut params = Params::from_instance(&instance);
        
        let old_cap = params.penalty_capacity;
        // All feasible, above target ratio → decrease penalty
        params.update_penalties(10, 0, 0, 0);
        
        assert!(params.penalty_capacity < old_cap, 
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
