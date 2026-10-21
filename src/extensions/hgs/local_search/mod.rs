//! Local search operators for HGS-CVRP/VRPTW.
//!
//! Implements the local search moves and constructive heuristics used by Hybrid Genetic Search:
//! - **RI (Relocate Individual)**: Remove a client (or two) and insert elsewhere
//! - **SWAP**: Swap clients between or within routes
//! - **2-opt / 2-opt***: Reverse segments within or between routes
//! - **Constructive heuristics**: Seed-based and sweep construction

use std::cmp::max;

use crate::extensions::hgs::circle_sector::CircleSector;
use crate::extensions::hgs::individual::SortableF64;
use crate::extensions::hgs::{Individual, Params, ProblemType};

use crate::extensions::hgs::params::MY_EPSILON;

/// Time window data for a sequence of nodes (Vidal 2012).
#[derive(Clone, Debug, Default)]
pub struct TimeWindowData {
    pub first_node_index: i32,
    pub last_node_index: i32,
    pub duration: f64,
    pub time_warp: f64,
    pub earliest_arrival: f64,
    pub latest_arrival: f64,
    pub latest_release_time: f64,
}

impl TimeWindowData {
    /// Create TimeWindowData for a single node.
    pub fn single(node_idx: i32, params: &Params) -> Self {
        let cli = &params.cli[node_idx as usize];
        TimeWindowData {
            first_node_index: node_idx,
            last_node_index: node_idx,
            duration: cli.service_duration,
            time_warp: 0.,
            earliest_arrival: cli.earliest_arrival,
            latest_arrival: cli.latest_arrival,
            latest_release_time: cli.release_time,
        }
    }

    /// Merge two consecutive time window data segments (Vidal 2012).
    pub fn merge(tw1: &TimeWindowData, tw2: &TimeWindowData, params: &Params) -> TimeWindowData {
        let delta_cost = params
            .time_cost
            .get(tw1.last_node_index as usize, tw2.first_node_index as usize);
        let delta_duration = delta_cost;
        let delta = tw1.duration - tw1.time_warp + delta_duration;
        let delta_wait_time = (tw2.earliest_arrival - delta - tw1.latest_arrival).max(0.);
        let delta_time_warp = (tw1.earliest_arrival + delta - tw2.latest_arrival).max(0.);

        TimeWindowData {
            first_node_index: tw1.first_node_index,
            last_node_index: tw2.last_node_index,
            duration: tw1.duration + tw2.duration + delta_duration + delta_wait_time,
            time_warp: tw1.time_warp + tw2.time_warp + delta_time_warp,
            earliest_arrival: (tw2.earliest_arrival - delta).max(tw1.earliest_arrival)
                - delta_wait_time,
            latest_arrival: (tw2.latest_arrival - delta).min(tw1.latest_arrival) + delta_time_warp,
            latest_release_time: tw1.latest_release_time.max(tw2.latest_release_time),
        }
    }
}

/// A route in the local search representation.
#[derive(Clone, Debug, Default)]
pub struct HGSRoute {
    pub cour: i32,
    pub nb_customers: i32,
    pub when_last_modified: i32,
    pub when_last_tested_large_nb: i32,
    pub is_delta_removel_tw_outdated: bool,
    pub duration: f64,
    pub load: i32,
    pub reversal_distance: f64,
    pub tw_data: TimeWindowData,
    pub penalty: f64,
    pub polar_angle_barycenter: f64,
    pub sector: CircleSector,
}

/// A node in the local search representation.
#[derive(Clone, Debug, Default)]
pub struct Node {
    pub is_depot: bool,
    pub cour: i32,
    pub position: i32,
    pub when_last_tested_ri: i32,
    pub next: i32,
    pub prev: i32,
    pub route_idx: i32,
    pub cumulated_load: i32,
    pub cumulated_time: f64,
    pub cumulated_reversal_distance: f64,
    pub delta_removal: f64,
    pub delta_removal_tw: f64,
    pub tw_data: TimeWindowData,
    pub prefix_tw_data: TimeWindowData,
    pub postfix_tw_data: TimeWindowData,
    /// Whether `next_seed` is set: the node four positions later.
    pub is_seed: bool,
    /// Time window data of the nodes after this one up to `next_seed`.
    pub to_next_seed_tw_data: TimeWindowData,
    pub next_seed: i32,
}

/// Helper struct for construction heuristics.
#[derive(Clone, Debug)]
pub struct NodeToInsert {
    pub client_idx: i32,
    pub load: i32,
    pub service_duration: f64,
    pub angle_from_depot: f64,
    pub tw_data: TimeWindowData,
}

/// Three best insertion positions of a client in a route.
#[derive(Clone, Debug, Default)]
pub struct ThreeBestInsert {
    pub when_last_calculated: i32,
    pub best_cost: [f64; 3],
    pub best_location: [i32; 3],
}

impl ThreeBestInsert {
    pub fn new() -> Self {
        Self {
            when_last_calculated: -1,
            best_cost: [f64::INFINITY; 3],
            best_location: [-1, -1, -1],
        }
    }

    /// Forget the stored positions, keeping `when_last_calculated`.
    pub fn reset(&mut self) {
        self.best_cost = [f64::INFINITY; 3];
        self.best_location = [-1; 3];
    }

    /// Keep `location` if its cost is among the three best.
    pub fn compare_and_add(&mut self, cost: f64, location: i32) {
        if cost >= self.best_cost[2] {
            return;
        }
        let rank = if cost >= self.best_cost[1] {
            2
        } else if cost >= self.best_cost[0] {
            1
        } else {
            0
        };
        for k in (rank + 1..3).rev() {
            self.best_cost[k] = self.best_cost[k - 1];
            self.best_location[k] = self.best_location[k - 1];
        }
        self.best_cost[rank] = cost;
        self.best_location[rank] = location;
    }
}

/// Best SWAP* move found between two routes.
#[derive(Clone, Debug)]
struct SwapStarElement {
    move_cost: f64,
    load_pen_u: f64,
    load_pen_v: f64,
    u: usize,
    best_position_u: Option<usize>,
    v: usize,
    best_position_v: Option<usize>,
}

/// Main Local Search engine for HGS.
pub struct HGSLocalSearch<'p> {
    params: &'p Params,
    pub search_completed: bool,
    pub nb_moves: i32,
    pub order_nodes: Vec<usize>,
    pub order_routes: Vec<usize>,
    pub empty_routes: std::collections::BTreeSet<usize>,
    pub loop_id: i32,

    /// All nodes:
    /// - 0: unused sentinel
    /// - 1..=nb_clients: clients
    /// - nb_clients + 1 + r: start depot for route r
    /// - nb_clients + 1 + nb_vehicles + r: end depot for route r
    pub nodes: Vec<Node>,
    pub routes: Vec<HGSRoute>,

    pub best_insert_initialized_for_route: Vec<bool>,
    pub best_insert_client: Vec<Vec<ThreeBestInsert>>,
    pub best_insert_client_tw: Vec<Vec<ThreeBestInsert>>,

    pub penalty_capacity_ls: f64,
    pub penalty_time_warp_ls: f64,

    /// Granular neighborhoods, reshuffled during the search. Owned here
    /// because the search mutates them and `params` is shared.
    correlated_vertices: Vec<Vec<usize>>,
    nb_granular: i32,
}

mod construction;
mod relocate;
mod routes;
mod search;
mod star;
mod swap;
mod two_opt;
