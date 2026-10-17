//! Local search operators for HGS-CVRP/VRPTW.
//!
//! Implements the core local search moves used by Hybrid Genetic Search:
//! - **RI (Relocate Individual)**: Remove a client and insert it elsewhere
//! - **SWAP***: Swap clients between routes with optimal insertion positions
//! - **2-opt / 2-opt***: Reverse segments within or between routes
//! - **Constructive heuristics**: Seed-based and sweep construction

use std::cmp::{max, min};
use std::f64::INFINITY;

use crate::hgs::individual::CostSol;
use crate::hgs::{Individual, Params};

/// Time window data for a sequence of nodes.
#[derive(Clone, Debug, Default)]
pub struct TimeWindowData {
    /// Index of the first node in the sequence.
    pub first_node_index: i32,
    /// Index of the last node in the sequence.
    pub last_node_index: i32,
    /// Cumulative duration (driving + service, excluding waiting).
    pub duration: i32,
    /// Cumulative time warp.
    pub time_warp: i32,
    /// Earliest start of servicing first node (given min cost route sequence).
    pub earliest_arrival: i32,
    /// Latest start of servicing first node (given min cost route sequence).
    pub latest_arrival: i32,
    /// Latest of all release times in the sequence.
    pub latest_release_time: i32,
}

impl TimeWindowData {
    /// Create a new TimeWindowData for a single node.
    pub fn single(node_idx: i32, params: &Params) -> Self {
        let cli = &params.cli[node_idx as usize];
        TimeWindowData {
            first_node_index: node_idx,
            last_node_index: node_idx,
            duration: cli.service_duration,
            time_warp: 0,
            earliest_arrival: cli.earliest_arrival,
            latest_arrival: cli.latest_arrival,
            latest_release_time: cli.release_time,
        }
    }

    /// Merge two consecutive time window data segments.
    pub fn merge(tw1: &TimeWindowData, tw2: &TimeWindowData, params: &Params) -> TimeWindowData {
        let mut result = TimeWindowData::default();
        result.first_node_index = tw1.first_node_index;
        result.last_node_index = tw2.last_node_index;

        // Duration: drive from last of tw1 to first of tw2, plus service and driving within segments
        let edge_duration = params.time_cost.get(tw1.last_node_index as usize, tw2.first_node_index as usize) as i32;
        result.duration = tw1.duration + edge_duration + tw2.duration;

        // Time warp calculation
        let arrival_at_tw2_start = tw1.earliest_arrival + tw1.duration + edge_duration;
        let wait_time = max(0, tw2.earliest_arrival - arrival_at_tw2_start);
        let earliest_arr = min(tw1.earliest_arrival, tw2.earliest_arrival - wait_time - edge_duration);

        let arrival_latest = tw1.latest_arrival + tw1.duration + edge_duration;
        let time_warp_tw2 = max(0, tw2.latest_arrival - arrival_latest) as i32;
        result.time_warp = tw1.time_warp + time_warp_tw2;

        result.earliest_arrival = earliest_arr;
        result.latest_arrival = min(tw1.latest_arrival, tw2.latest_arrival - time_warp_tw2 as i32) as i32;
        result.latest_release_time = max(tw1.latest_release_time, tw2.latest_release_time);

        result
    }
}

/// A route in the local search representation.
#[derive(Clone, Debug)]
pub struct HGSRoute {
    /// Route index.
    pub cour: i32,
    /// Number of customers in the route.
    pub nb_customers: i32,
    /// When this route was last modified.
    pub when_last_modified: i32,
    /// When large neighborhood moves were last tested.
    pub when_last_tested_large_nb: i32,
    /// Flag for deltaRemovalTW data staleness.
    pub is_delta_removel_tw_outdated: bool,
    /// Total duration (driving + service).
    pub duration: i32,
    /// Total load on the route.
    pub load: i32,
    /// Cost difference if the route is reversed.
    pub reversal_distance: i32,
    /// Time window data for the entire route.
    pub tw_data: TimeWindowData,
    /// Current penalty (capacity + time warp).
    pub penalty: f64,
    /// Polar angle of barycenter.
    pub polar_angle_barycenter: f64,
}

impl Default for HGSRoute {
    fn default() -> Self {
        HGSRoute {
            cour: 0,
            nb_customers: 0,
            when_last_modified: 0,
            when_last_tested_large_nb: 0,
            is_delta_removel_tw_outdated: true,
            duration: 0,
            load: 0,
            reversal_distance: 0,
            tw_data: TimeWindowData::default(),
            penalty: 0.0,
            polar_angle_barycenter: 0.0,
        }
    }
}

/// A node in the linked-list representation of routes.
#[derive(Clone, Debug)]
pub struct Node {
    /// Whether this is a depot node.
    pub is_depot: bool,
    /// Node index (1-based for clients).
    pub cour: i32,
    /// Position in the route.
    pub position: i32,
    /// When RI moves were last tested.
    pub when_last_tested_ri: i32,
    /// Next node in the route (index into clients/depots vectors).
    pub next_idx: i32,
    /// Previous node in the route (index into clients/depots vectors).
    pub prev_idx: i32,
    /// Associated route index.
    pub route_idx: i32,
    /// Cumulative load until this node.
    pub cumulated_load: i32,
    /// Cumulative time until this node.
    pub cumulated_time: i32,
    /// Cumulative reversal distance (0..cour).
    pub cumulated_reversal_distance: i32,
    /// Cost delta if removed from current route.
    pub delta_removal: i32,
    /// Cost delta if removed including time warp.
    pub delta_removal_tw: i32,
    /// Time window data for this node.
    pub tw_data: TimeWindowData,
    /// Time window data list (for multi-TW clients).
    pub tw_data_list: Vec<TimeWindowData>,
    /// Prefix TW data (0..cour including self).
    pub prefix_tw_data: TimeWindowData,
    /// Postfix TW data (cour..0 including self).
    pub postfix_tw_data: TimeWindowData,
    /// Whether a next seed is available.
    pub is_seed: bool,
    /// Next seeded node's TW data (excluding self).
    pub to_next_seed_tw_d: TimeWindowData,
    /// Index of next seeded node (-1 if none).
    pub next_seed_idx: i32,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            is_depot: false,
            cour: 0,
            position: 0,
            when_last_tested_ri: 0,
            next_idx: 0,
            prev_idx: 0,
            route_idx: 0,
            cumulated_load: 0,
            cumulated_time: 0,
            cumulated_reversal_distance: 0,
            delta_removal: 0,
            delta_removal_tw: 0,
            tw_data: TimeWindowData::default(),
            tw_data_list: Vec::new(),
            prefix_tw_data: TimeWindowData::default(),
            postfix_tw_data: TimeWindowData::default(),
            is_seed: false,
            to_next_seed_tw_d: TimeWindowData::default(),
            next_seed_idx: -1,
        }
    }
}

/// Stores the three best insertion positions for SWAP*.
#[derive(Clone, Debug)]
pub struct ThreeBestInsert {
    /// When last calculated.
    pub when_last_calculated: i32,
    /// Best costs (sorted ascending).
    pub best_cost: [i32; 3],
    /// Best locations (indices into nodes vector).
    pub best_location: [i32; 3],
}

impl Default for ThreeBestInsert {
    fn default() -> Self {
        ThreeBestInsert::new()
    }
}

impl ThreeBestInsert {
    fn new() -> Self {
        ThreeBestInsert {
            when_last_calculated: 0,
            best_cost: [i32::MAX, i32::MAX, i32::MAX],
            best_location: [-1, -1, -1],
        }
    }

    /// Compare and potentially add a new insertion position.
    fn compare_and_add(&mut self, cost_insert: i32, place_insert: i32) {
        if cost_insert >= self.best_cost[2] {
            return;
        } else if cost_insert >= self.best_cost[1] {
            self.best_cost[2] = cost_insert;
            self.best_location[2] = place_insert;
        } else if cost_insert >= self.best_cost[0] {
            self.best_cost[2] = self.best_cost[1];
            self.best_location[2] = self.best_location[1];
            self.best_cost[1] = cost_insert;
            self.best_location[1] = place_insert;
        } else {
            self.best_cost[2] = self.best_cost[1];
            self.best_location[2] = self.best_location[1];
            self.best_cost[1] = self.best_cost[0];
            self.best_location[1] = self.best_location[0];
            self.best_cost[0] = cost_insert;
            self.best_location[0] = place_insert;
        }
    }

    /// Reset to no insertion calculated.
    fn reset(&mut self) {
        self.best_cost = [i32::MAX, i32::MAX, i32::MAX];
        self.best_location = [-1, -1, -1];
    }
}

/// Tracks the best SWAP* move between two routes.
#[derive(Clone, Debug)]
pub struct SwapStarElement {
    /// Cost of the move.
    pub move_cost: f64,
    /// Load penalty for route U.
    pub load_pen_u: f64,
    /// Load penalty for route V.
    pub load_pen_v: f64,
    /// Node U to be removed from route U.
    pub u_idx: i32,
    /// Best position for U in route V.
    pub best_position_u_idx: i32,
    /// Node V to be removed from route V.
    pub v_idx: i32,
    /// Best position for V in route U.
    pub best_position_v_idx: i32,
}

impl Default for SwapStarElement {
    fn default() -> Self {
        SwapStarElement {
            move_cost: INFINITY,
            load_pen_u: INFINITY,
            load_pen_v: INFINITY,
            u_idx: 0,
            best_position_u_idx: 0,
            v_idx: 0,
            best_position_v_idx: 0,
        }
    }
}

/// Data for nodes during construction heuristics.
#[derive(Clone, Debug)]
pub struct NodeToInsert {
    /// Index in the clients vector (1-based).
    pub client_idx: i32,
    /// Load of the node.
    pub load: i32,
    /// Service duration.
    pub service_duration: i32,
    /// Angle from depot (-PI to PI).
    pub angle_from_depot: f64,
    /// Time window data.
    pub tw_data: TimeWindowData,
    /// Time window data list.
    pub tw_data_list: Vec<TimeWindowData>,
}

/// Hybrid Genetic Search local search engine.
///
/// Represents the solution as a linked list of nodes for efficient modifications.
pub struct HGSLocalSearch<'p> {
    /// Reference to problem parameters.
    params: &'p Params,
    /// Whether all moves have been evaluated without success.
    search_completed: bool,
    /// Total number of moves applied (also used to avoid repeating evaluations).
    nb_moves: i32,
    /// Randomized order for RI move evaluation.
    order_nodes: Vec<i32>,
    /// Randomized order for SWAP* route evaluation.
    order_routes: Vec<i32>,
    /// Indices of empty routes.
    empty_routes: std::collections::HashSet<i32>,
    /// Current loop index.
    loop_id: i32,

    /// Linked list nodes (clients[0] is a sentinel).
    clients: Vec<Node>,
    /// Depot nodes.
    depots: Vec<Node>,
    /// End depot nodes (mark route ends).
    depots_end: Vec<Node>,
    /// Route data.
    routes: Vec<HGSRoute>,

    /// For each route, whether best insertions have been initialized.
    best_insert_initialized_for_route: Vec<bool>,
    /// Best insertion costs (excluding TW) for SWAP*.
    best_insert_client: Vec<Vec<ThreeBestInsert>>,
    /// Best insertion costs (including TW) for SWAP*.
    best_insert_client_tw: Vec<Vec<ThreeBestInsert>>,

    /// Temporary variables for move evaluation.
    node_u_idx: i32,
    node_x_idx: i32,
    node_v_idx: i32,
    node_y_idx: i32,
    route_u_idx: i32,
    route_v_idx: i32,
    node_u_prev_idx: i32,
    node_u_index: i32,
    node_x_index: i32,
    node_x_next_index: i32,
    node_v_prev_idx: i32,
    node_v_index: i32,
    node_y_index: i32,
    node_y_next_index: i32,
    load_u: i32,
    load_x: i32,
    load_v: i32,
    load_y: i32,
    service_u: i32,
    service_x: i32,
    service_v: i32,
    service_y: i32,
    route_u_time_warp: bool,
    route_u_load_penalty: bool,
    route_v_time_warp: bool,
    route_v_load_penalty: bool,
    penalty_capacity_ls: f64,
    penalty_time_warp_ls: f64,

    /// Cached distance values for route U.
    local_dist_route_u: Vec<i32>,
    /// Cached distance values for route V.
    local_dist_route_v: Vec<i32>,
}

impl<'p> HGSLocalSearch<'p> {
    /// Create a new local search instance.
    pub fn new(params: &'p Params) -> Self {
        let nb_clients = params.nb_clients as usize;
        let nb_vehicles = params.nb_vehicles as usize;

        HGSLocalSearch {
            params,
            search_completed: false,
            nb_moves: 0,
            order_nodes: Vec::new(),
            order_routes: Vec::new(),
            empty_routes: std::collections::HashSet::new(),
            loop_id: 0,

            clients: vec![Node::default(); nb_clients + 1],
            depots: vec![Node::default(); nb_vehicles],
            depots_end: vec![Node::default(); nb_vehicles],
            routes: (0..nb_vehicles)
                .map(|r| HGSRoute {
                    cour: r as i32,
                    ..Default::default()
                })
                .collect(),

            best_insert_initialized_for_route: vec![false; nb_vehicles],
            best_insert_client: vec![vec![ThreeBestInsert::new(); nb_clients + 1]; nb_vehicles],
            best_insert_client_tw: vec![vec![ThreeBestInsert::new(); nb_clients + 1]; nb_vehicles],

            node_u_idx: 0,
            node_x_idx: 0,
            node_v_idx: 0,
            node_y_idx: 0,
            route_u_idx: 0,
            route_v_idx: 0,
            node_u_prev_idx: 0,
            node_u_index: 0,
            node_x_index: 0,
            node_x_next_index: 0,
            node_v_prev_idx: 0,
            node_v_index: 0,
            node_y_index: 0,
            node_y_next_index: 0,
            load_u: 0,
            load_x: 0,
            load_v: 0,
            load_y: 0,
            service_u: 0,
            service_x: 0,
            service_v: 0,
            service_y: 0,
            route_u_time_warp: false,
            route_u_load_penalty: false,
            route_v_time_warp: false,
            route_v_load_penalty: false,
            penalty_capacity_ls: 0.0,
            penalty_time_warp_ls: 0.0,

            local_dist_route_u: Vec::new(),
            local_dist_route_v: Vec::new(),
        }
    }

    // =========================================================================
    // PUBLIC INTERFACE
    // =========================================================================

    /// Run the local search with the specified penalty values.
    pub fn run(&mut self, indiv: &mut Individual, penalty_capacity: f64, penalty_time_warp: f64) {
        self.penalty_capacity_ls = penalty_capacity;
        self.penalty_time_warp_ls = penalty_time_warp;

        // Load the individual into the local search representation.
        self.load_individual(indiv);

        self.nb_moves = 0;
        self.loop_id = 0;

        // Main local search loop — iterate until no improving moves found.
        loop {
            self.search_completed = true;
            self.loop_id += 1;

            // Initialize randomized orders.
            self.init_order_nodes();
            self.init_order_routes();

            // Try RI (Relocate Individual) moves.
            self.try_ri_moves();

            // Try SWAP* moves.
            self.try_swap_star_moves();

            // Try 2-opt and 2-opt* moves.
            self.try_two_opt_moves();

            // Reorder time windows if beneficial.
            self.reorder_tws_if_needed();

            // If no improvement, check if we should terminate.
            if self.search_completed {
                // Try RelocateStar on recently modified routes.
                if !self.relocate_star() {
                    break;
                }
            }

            self.nb_moves += 1;
            if self.nb_moves > 100000 {
                // Safety limit to prevent infinite loops.
                break;
            }
        }

        // Export the improved solution back to the individual.
        self.export_individual(indiv);
    }

    /// Load an individual into the local search representation.
    pub fn load_individual(&mut self, indiv: &Individual<'_>) {
        let nb_clients = self.params.nb_clients as usize;
        let nb_vehicles = self.params.nb_vehicles as usize;

        // Reset data structures.
        self.clients = vec![Node::default(); nb_clients + 1];
        self.depots = vec![Node::default(); nb_vehicles];
        self.depots_end = vec![Node::default(); nb_vehicles];
        self.routes = (0..nb_vehicles)
            .map(|r| HGSRoute {
                cour: r as i32,
                ..Default::default()
            })
            .collect();
        self.empty_routes.clear();

        // Initialize depot links and TW data.
        for r in 0..nb_vehicles {
            let depot_tw = TimeWindowData {
                first_node_index: 0,
                last_node_index: 0,
                duration: 0,
                time_warp: 0,
                earliest_arrival: self.params.cli[0].earliest_arrival,
                latest_arrival: self.params.cli[0].latest_arrival,
                latest_release_time: self.params.cli[0].release_time,
            };

            let depot = &mut self.depots[r];
            depot.is_depot = true;
            depot.cour = 0;
            depot.route_idx = r as i32;
            depot.next_idx = (nb_vehicles + r) as i32;
            depot.prev_idx = (nb_vehicles + r) as i32;
            depot.tw_data = depot_tw.clone();
            depot.prefix_tw_data = depot_tw.clone();
            depot.postfix_tw_data = depot_tw.clone();

            let depot_end = &mut self.depots_end[r];
            depot_end.is_depot = true;
            depot_end.cour = 0;
            depot_end.route_idx = r as i32;
            depot_end.next_idx = r as i32;
            depot_end.prev_idx = r as i32;
            depot_end.tw_data = depot_tw.clone();
            depot_end.prefix_tw_data = depot_tw.clone();
            depot_end.postfix_tw_data = depot_tw;
        }

        // Load clients into routes.
        for r in 0..nb_vehicles {
            let route_clients: Vec<usize> = indiv.chrom_r[r].clone();
            if route_clients.is_empty() {
                self.empty_routes.insert(r as i32);
                continue;
            }

            let route_idx = r as i32;
            let depot_idx = r as i32;
            let depot_end_idx = (nb_vehicles + r) as i32;

            // Insert clients in order between depot and depot_end.
            let mut prev_idx = depot_idx;
            for (pos, &client_idx) in route_clients.iter().enumerate() {
                let node_idx = client_idx as i32;
                let mut node = Node::clone(&self.clients[node_idx as usize]);
                node.is_depot = false;
                node.cour = node_idx;
                node.position = (pos + 1) as i32;
                node.route_idx = route_idx;
                node.prev_idx = prev_idx;
                node.next_idx = 0; // will be set in next iteration

                // Update cumulated load and time.
                if pos == 0 {
                    node.cumulated_load = self.params.cli[client_idx].demand;
                    node.cumulated_time = self.params.time_cost.get(0, client_idx) as i32
                        + self.params.cli[client_idx].service_duration;
                } else {
                    let prev_client = route_clients[pos - 1];
                    let prev_cum_load = self.clients[self.clients[prev_idx as usize].next_idx as usize].cumulated_load;
                    node.cumulated_load = prev_cum_load + self.params.cli[client_idx].demand;
                    let prev_cum_time = self.clients[prev_idx as usize].cumulated_time;
                    let edge_dist = self.params.time_cost.get(
                        if prev_client == 0 { 0 } else { prev_client },
                        client_idx,
                    ) as i32;
                    node.cumulated_time = prev_cum_time + edge_dist;
                }

                // Initialize TW data for single node.
                let cli = &self.params.cli[client_idx];
                node.tw_data = TimeWindowData {
                    first_node_index: node_idx,
                    last_node_index: node_idx,
                    duration: cli.service_duration,
                    time_warp: 0,
                    earliest_arrival: cli.earliest_arrival,
                    latest_arrival: cli.latest_arrival,
                    latest_release_time: cli.release_time,
                };
                node.prefix_tw_data = node.tw_data.clone();
                node.postfix_tw_data = node.tw_data.clone();

                self.clients[node_idx as usize] = node;

                // Link previous to current.
                self.clients[prev_idx as usize].next_idx = node_idx;
                prev_idx = node_idx;
            }

            // Link last client to depot_end.
            self.clients[prev_idx as usize].next_idx = depot_end_idx;
            self.clients[depot_end_idx as usize].prev_idx = prev_idx;

            // Update route data.
            self.routes[route_idx as usize] = self.compute_route_data(route_idx);
        }

        // Initialize best insert tracking.
        self.best_insert_initialized_for_route = vec![false; nb_vehicles];
    }

    /// Export the local search solution back to an individual.
    pub fn export_individual(&self, indiv: &mut Individual) {
        let nb_vehicles = self.params.nb_vehicles as usize;

        // Clear all routes.
        for r in 0..indiv.chrom_r.len() {
            indiv.chrom_r[r].clear();
        }

        // Extract routes from the linked list.
        for r in 0..nb_vehicles {
            let depot_idx = r as i32;
            let mut curr = self.clients[depot_idx as usize].next_idx;
            while curr != (self.params.nb_clients + r as i32 + 1) && !self.clients[curr as usize].is_depot {
                indiv.chrom_r[r].push(curr as usize);
                curr = self.clients[curr as usize].next_idx;
            }
        }

        // Rebuild predecessor/successor maps from routes.
        indiv.rebuild_maps();
        indiv.evaluate_complete_cost();
    }

    // =========================================================================
    // RI (RELOCATE INDIVIDUAL) MOVES
    // =========================================================================

    /// Initialize randomized node order for RI moves.
    fn init_order_nodes(&mut self) {
        let nb_clients = self.params.nb_clients as usize;
        self.order_nodes = (1..=nb_clients as i32).collect();
        self.params.rng.borrow_mut().shuffle(&mut self.order_nodes);
    }

    /// Initialize randomized route order for SWAP* moves.
    fn init_order_routes(&mut self) {
        let nb_vehicles = self.params.nb_vehicles as usize;
        self.order_routes = (0..nb_vehicles as i32).collect();
        self.params.rng.borrow_mut().shuffle(&mut self.order_routes);
    }

    /// Try all RI (relocate) moves.
    fn try_ri_moves(&mut self) {
        let order: Vec<i32> = self.order_nodes.iter().copied().collect();
        for u in order {
            if u <= 0 || u > self.params.nb_clients {
                continue;
            }

            let node_u_idx = u as usize;
            let is_depot = self.clients[node_u_idx].is_depot;
            if is_depot {
                continue;
            }

            // Try removing U and inserting after each other node V.
            let route_u_idx = self.clients[node_u_idx].route_idx;
            self.routes[route_u_idx as usize].when_last_modified = self.loop_id;

            let mut curr = self.clients[self.clients[node_u_idx].prev_idx as usize].next_idx; // U itself
            while curr != (self.params.nb_clients + route_u_idx + 1) {
                let v = curr;
                let is_v_depot = self.clients[v as usize].is_depot;
                if !is_v_depot && v != u {
                    if self.move_single_client(u, v) {
                        break; // Move applied, restart RI loop.
                    }
                }
                curr = self.clients[curr as usize].next_idx;
            }
        }
    }

    /// Try to relocate client U and insert it after V.
    fn move_single_client(&mut self, u: i32, v: i32) -> bool {
        // Calculate cost of removing U from its current position.
        let u_prev = self.clients[u as usize].prev_idx;
        let u_next = self.clients[u as usize].next_idx;

        // Cost change from removing U.
        let remove_cost = self.calculate_removal_cost(u);

        // Cost change from inserting U after V.
        let insert_cost = self.calculate_insertion_cost(v, u);

        let total_delta = remove_cost + insert_cost;

        if total_delta < -1 {
            // Apply the move: remove U, insert after V.
            self.remove_node(u);
            self.insert_node_after(u, v);
            self.update_route_data_for_changed_routes(vec![u_prev, u_next, v]);
            return true;
        }

        false
    }

    /// Calculate the cost delta of removing node U from its route.
    fn calculate_removal_cost(&self, u: i32) -> i32 {
        let u_prev = self.clients[u as usize].prev_idx;
        let u_next = self.clients[u as usize].next_idx;

        // Old cost: depot -> ... -> u_prev -> u -> u_next -> ... -> depot
        // New cost: depot -> ... -> u_prev -> u_next -> ... -> depot
        let old_edge = self.params.time_cost.get(u_prev as usize, u as usize)
            + self.params.time_cost.get(u as usize, u_next as usize);
        let new_edge = self.params.time_cost.get(u_prev as usize, u_next as usize);

        new_edge - old_edge
    }

    /// Calculate the cost delta of inserting node U after V.
    fn calculate_insertion_cost(&self, v: i32, u: i32) -> i32 {
        let v_next = self.clients[v as usize].next_idx;

        // Old edge: v -> v_next
        // New edges: v -> u -> v_next
        let old_edge = self.params.time_cost.get(v as usize, v_next as usize);
        let new_edges = self.params.time_cost.get(v as usize, u as usize)
            + self.params.time_cost.get(u as usize, v_next as usize);

        new_edges - old_edge
    }

    /// Remove a node from its route.
    fn remove_node(&mut self, u: i32) {
        let u_prev = self.clients[u as usize].prev_idx;
        let u_next = self.clients[u as usize].next_idx;

        self.clients[u_prev as usize].next_idx = u_next;
        self.clients[u_next as usize].prev_idx = u_prev;

        // Clear the node.
        self.clients[u as usize] = Node::default();
    }

    /// Insert node U after V.
    fn insert_node_after(&mut self, u: i32, v: i32) {
        let v_next = self.clients[v as usize].next_idx;

        self.clients[v as usize].next_idx = u;
        self.clients[u as usize].prev_idx = v;
        self.clients[u as usize].next_idx = v_next;
        self.clients[v_next as usize].prev_idx = u;
    }

    /// Update route data for routes that were affected by a move.
    fn update_route_data_for_changed_routes(&mut self, node_indices: Vec<i32>) {
        // First collect route indices without borrowing self.clients and self.routes simultaneously.
        let mut route_indices: Vec<i32> = Vec::new();
        for &idx in &node_indices {
            if idx > 0 && idx <= self.params.nb_clients {
                let r = self.clients[idx as usize].route_idx;
                if !route_indices.contains(&r) {
                    route_indices.push(r);
                }
            }
        }

        // Now update each route.
        for &r in &route_indices {
            self.routes[r as usize] = self.compute_route_data(r);
        }
    }

    /// Compute route data without mutating self (for use in load_individual).
    fn compute_route_data(&self, route_idx: i32) -> HGSRoute {
        let depot_idx = route_idx;
        let depot_end_idx = self.params.nb_clients + route_idx + 1;

        // Traverse the route and compute cumulative values.
        let mut cum_load = 0i32;
        let mut cum_time = 0i32;
        let mut nb_customers = 0i32;
        let mut total_duration = 0i32;
        let mut prev_idx = depot_idx as i32;

        let mut route = HGSRoute::default();
        route.cour = route_idx;

        let mut curr = self.clients[depot_idx as usize].next_idx;
        while curr != depot_end_idx {
            let node = &self.clients[curr as usize];
            nb_customers += 1;

            let client_demand = self.params.cli[node.cour as usize].demand;
            cum_load += client_demand;
            route.load = cum_load;

            let edge_dist = if prev_idx == depot_idx {
                self.params.time_cost.get(0, node.cour as usize)
            } else {
                self.params.time_cost.get(prev_idx as usize, node.cour as usize)
            } as i32;

            cum_time += edge_dist + self.params.cli[node.cour as usize].service_duration;
            total_duration += edge_dist;

            prev_idx = curr;
            curr = self.clients[curr as usize].next_idx;
        }

        route.nb_customers = nb_customers;
        route.duration = total_duration;

        // Calculate TW data for the entire route.
        if nb_customers > 0 {
            let first_node = self.clients[depot_idx as usize].next_idx;

            let mut tw_acc = TimeWindowData::single(first_node, &self.params);

            curr = self.clients[first_node as usize].next_idx;
            while curr != depot_end_idx {
                let next_tw = TimeWindowData::single(curr, &self.params);
                let edge_dist = self.params.time_cost.get(
                    if self.clients[curr as usize].prev_idx == depot_idx { 0 }
                    else { self.clients[curr as usize].prev_idx as usize },
                    curr as usize,
                ) as i32;

                tw_acc.duration += edge_dist + next_tw.duration;

                let arrival = tw_acc.earliest_arrival + edge_dist;
                let wait = if arrival < next_tw.earliest_arrival {
                    next_tw.earliest_arrival - arrival
                } else {
                    0
                };
                tw_acc.earliest_arrival = if arrival < next_tw.earliest_arrival {
                    next_tw.earliest_arrival - wait
                } else {
                    arrival
                };

                let arrival_latest = tw_acc.latest_arrival + edge_dist;
                let tw_warp = if arrival_latest > next_tw.latest_arrival {
                    arrival_latest - next_tw.latest_arrival
                } else {
                    0
                };
                tw_acc.time_warp += tw_warp;
                tw_acc.latest_arrival = if arrival_latest + tw_warp < next_tw.latest_arrival {
                    arrival_latest + tw_warp
                } else {
                    next_tw.latest_arrival
                };

                curr = self.clients[curr as usize].next_idx;
            }

            route.tw_data = tw_acc;
        }

        // Calculate penalty.
        let capacity_penalty = if route.load > self.params.vehicle_capacity {
            (route.load - self.params.vehicle_capacity) as f64 * self.penalty_capacity_ls
        } else {
            0.0
        };
        let tw_penalty = route.tw_data.time_warp as f64 * self.penalty_time_warp_ls;
        route.penalty = capacity_penalty + tw_penalty;

        route
    }

    /// Update preprocessed data for a route.
    fn update_route_data(&mut self, route: &mut HGSRoute) {
        let route_idx = route.cour;
        let depot_idx = route_idx;
        let depot_end_idx = self.params.nb_clients + route_idx + 1;

        // Traverse the route and compute cumulative values.
        let mut cum_load = 0i32;
        let mut cum_time = 0i32;
        let mut nb_customers = 0i32;
        let mut total_duration = 0i32;
        let mut prev_idx = depot_idx as i32;

        route.nb_customers = 0;
        route.load = 0;
        route.duration = 0;

        let mut curr = self.clients[depot_idx as usize].next_idx;
        while curr != depot_end_idx {
            let node = &mut self.clients[curr as usize];
            nb_customers += 1;

            let client_demand = self.params.cli[node.cour as usize].demand;
            cum_load += client_demand;
            route.load = cum_load;

            let edge_dist = if prev_idx == depot_idx {
                self.params.time_cost.get(0, node.cour as usize)
            } else {
                self.params.time_cost.get(prev_idx as usize, node.cour as usize)
            } as i32;

            cum_time += edge_dist + self.params.cli[node.cour as usize].service_duration;
            total_duration += edge_dist;

            node.cumulated_load = cum_load;
            node.cumulated_time = cum_time;
            node.position = nb_customers;

            prev_idx = curr;
            curr = self.clients[curr as usize].next_idx;
        }

        route.nb_customers = nb_customers;
        route.duration = total_duration;

        // Calculate TW data for the entire route.
        if nb_customers > 0 {
            let first_node = self.clients[depot_idx as usize].next_idx;
            let last_node_idx = prev_idx;

            let mut tw_acc = TimeWindowData::single(first_node, &self.params);

            curr = self.clients[first_node as usize].next_idx;
            while curr != depot_end_idx {
                let next_tw = TimeWindowData::single(curr, &self.params);
                let edge_dist = self.params.time_cost.get(
                    if self.clients[curr as usize].prev_idx == depot_idx { 0 }
                    else { self.clients[curr as usize].prev_idx as usize },
                    curr as usize,
                ) as i32;

                tw_acc.duration += edge_dist + next_tw.duration;

                let arrival = tw_acc.earliest_arrival + edge_dist;
                let wait = if arrival < next_tw.earliest_arrival {
                    next_tw.earliest_arrival - arrival
                } else {
                    0
                };
                tw_acc.earliest_arrival = if arrival < next_tw.earliest_arrival {
                    next_tw.earliest_arrival - wait
                } else {
                    arrival
                };

                let arrival_latest = tw_acc.latest_arrival + edge_dist;
                let tw_warp = if arrival_latest > next_tw.latest_arrival {
                    arrival_latest - next_tw.latest_arrival
                } else {
                    0
                };
                tw_acc.time_warp += tw_warp;
                tw_acc.latest_arrival = if arrival_latest + tw_warp < next_tw.latest_arrival {
                    arrival_latest + tw_warp
                } else {
                    next_tw.latest_arrival
                };

                curr = self.clients[curr as usize].next_idx;
            }

            route.tw_data = tw_acc;
        }

        // Calculate penalty.
        let capacity_penalty = if route.load > self.params.vehicle_capacity {
            (route.load - self.params.vehicle_capacity) as f64 * self.penalty_capacity_ls
        } else {
            0.0
        };
        let tw_penalty = route.tw_data.time_warp as f64 * self.penalty_time_warp_ls;
        route.penalty = capacity_penalty + tw_penalty;
    }

    // =========================================================================
    // SWAP* MOVES (simplified)
    // =========================================================================

    /// Try all SWAP* moves between pairs of routes.
    fn try_swap_star_moves(&mut self) {
        let order: Vec<i32> = self.order_routes.iter().copied().collect();
        for r_u in &order {
            if self.empty_routes.contains(r_u) {
                continue;
            }

            self.routes[*r_u as usize].when_last_modified = self.loop_id;

            for r_v in &order {
                if *r_v == *r_u || self.empty_routes.contains(r_v) {
                    continue;
                }

                self.routes[*r_v as usize].when_last_modified = self.loop_id;

                // Simplified SWAP*: try swapping each pair of clients.
                if self.swap_two_single_clients(*r_u, *r_v) {
                    break;
                }
            }
        }
    }

    /// Try swapping two single clients between routes.
    fn swap_two_single_clients(&mut self, r_u: i32, r_v: i32) -> bool {
        let depot_u = r_u;
        let depot_end_u = self.params.nb_clients + r_u + 1;
        let depot_v = r_v;
        let depot_end_v = self.params.nb_clients + r_v + 1;

        // Traverse route U.
        let mut curr_u = self.clients[depot_u as usize].next_idx;
        while curr_u != depot_end_u {
            if self.clients[curr_u as usize].is_depot {
                curr_u = self.clients[curr_u as usize].next_idx;
                continue;
            }

            let u = curr_u;
            let u_next = self.clients[u as usize].next_idx;
            let u_prev = self.clients[u as usize].prev_idx;

            // Traverse route V.
            let mut curr_v = self.clients[depot_v as usize].next_idx;
            while curr_v != depot_end_v {
                if self.clients[curr_v as usize].is_depot {
                    curr_v = self.clients[curr_v as usize].next_idx;
                    continue;
                }

                let v = curr_v;
                let v_next = self.clients[v as usize].next_idx;
                let v_prev = self.clients[v as usize].prev_idx;

                // Calculate cost of swapping U and V.
                let cost_u_remove = self.params.time_cost.get(u_prev as usize, u as usize)
                    + self.params.time_cost.get(u as usize, u_next as usize);
                let cost_u_insert = self.params.time_cost.get(v_prev as usize, u as usize)
                    + self.params.time_cost.get(u as usize, v_next as usize);

                let cost_v_remove = self.params.time_cost.get(v_prev as usize, v as usize)
                    + self.params.time_cost.get(v as usize, v_next as usize);
                let cost_v_insert = self.params.time_cost.get(u_prev as usize, v as usize)
                    + self.params.time_cost.get(v as usize, u_next as usize);

                let total_delta = (cost_u_insert + cost_v_insert) - (cost_u_remove + cost_v_remove);

                if total_delta < -1 {
                    // Apply swap.
                    let u_pos_in_v = v_prev;
                    let u_next_after_swap = v_next;
                    let v_pos_in_u = u_prev;
                    let v_next_after_swap = u_next;

                    // Remove U from route U, V from route V.
                    self.clients[u_prev as usize].next_idx = u_next;
                    self.clients[u_next as usize].prev_idx = u_prev;
                    self.clients[v_prev as usize].next_idx = v_next;
                    self.clients[v_next as usize].prev_idx = v_prev;

                    // Insert V into route U, U into route V.
                    self.clients[v_prev as usize].next_idx = v;
                    self.clients[v as usize].prev_idx = v_prev;
                    self.clients[v as usize].next_idx = u_next;
                    self.clients[u_next as usize].prev_idx = v;

                    self.clients[u_prev as usize].next_idx = u;
                    self.clients[u as usize].prev_idx = u_prev;
                    self.clients[u as usize].next_idx = v_next;
                    self.clients[v_next as usize].prev_idx = u;

                    // Update route data.
                    self.routes[r_u as usize] = self.compute_route_data(r_u);
                    self.routes[r_v as usize] = self.compute_route_data(r_v);
                    return true;
                }

                curr_v = self.clients[curr_v as usize].next_idx;
            }

            curr_u = self.clients[curr_u as usize].next_idx;
        }

        false
    }

    // =========================================================================
    // 2-OPT / 2-OPT* MOVES (simplified)
    // =========================================================================

    /// Try 2-opt and 2-opt* moves.
    fn try_two_opt_moves(&mut self) {
        let order: Vec<i32> = self.order_routes.iter().copied().collect();
        for r_u in &order {
            let nb_cust = self.routes[*r_u as usize].nb_customers;
            if self.empty_routes.contains(r_u) || nb_cust < 2 {
                continue;
            }

            self.routes[*r_u as usize].when_last_modified = self.loop_id;

            // Try 2-opt within the same route.
            if self.two_opt_within_trip(*r_u, *r_u) {
                continue;
            }

            // Try 2-opt* between different routes.
            for r_v in &order {
                if *r_v == *r_u || self.empty_routes.contains(r_v) {
                    continue;
                }

                if self.two_opt_between_trips(*r_u, *r_v) {
                    break;
                }
            }
        }
    }

    /// 2-opt within a single route.
    fn two_opt_within_trip(&mut self, r_u: i32, _r_v: i32) -> bool {
        let depot_idx = r_u;
        let depot_end_idx = self.params.nb_clients + r_u + 1;

        // Get all nodes in the route.
        let mut route_nodes: Vec<i32> = Vec::new();
        let mut curr = self.clients[depot_idx as usize].next_idx;
        while curr != depot_end_idx {
            if !self.clients[curr as usize].is_depot {
                route_nodes.push(curr);
            }
            curr = self.clients[curr as usize].next_idx;
        }

        if route_nodes.len() < 2 {
            return false;
        }

        // Try all pairs (i, j) with i < j.
        for i in 0..route_nodes.len() - 1 {
            for j in (i + 1)..route_nodes.len() {
                let u = route_nodes[i];
                let u_next = self.clients[u as usize].next_idx;
                let v = route_nodes[j];
                let v_prev = self.clients[v as usize].prev_idx;

                // Old edges: u -> u_next ... v_prev -> v
                // New edges: u -> v, u_next -> ... -> v_prev (reversed segment)
                let old_cost = self.params.time_cost.get(u as usize, u_next as usize)
                    + self.params.time_cost.get(v_prev as usize, v as usize);

                let new_cost = self.params.time_cost.get(u as usize, v as usize)
                    + self.params.time_cost.get(u_next as usize, v_prev as usize);

                if new_cost + 2 < old_cost {
                    // Apply 2-opt reversal.
                    self.apply_2_opt_reversal(v, u_next);
                    self.routes[r_u as usize] = self.compute_route_data(r_u);
                    return true;
                }
            }
        }

        false
    }

    /// Apply a 2-opt reversal: reverse the segment between `v` and `u_next`.
    fn apply_2_opt_reversal(&mut self, v: i32, u_next: i32) {
        // Reverse the linked list segment from v to u_next.
        let depot_end_idx = self.params.nb_clients + self.clients[v as usize].route_idx + 1;

        let mut prev = v;
        let mut curr = self.clients[v as usize].next_idx;
        let first = Some(v);

        while curr != depot_end_idx && curr != u_next {
            let next = self.clients[curr as usize].next_idx;
            self.clients[curr as usize].next_idx = prev;
            self.clients[curr as usize].prev_idx = next;
            prev = curr;
            curr = next;
        }

        // Fix endpoints.
        if let Some(first_node) = first {
            self.clients[first_node as usize].prev_idx = prev;
            self.clients[first_node as usize].next_idx = u_next;
        }
        self.clients[v as usize].next_idx = first.unwrap();
        self.clients[u_next as usize].prev_idx = v;
    }

    /// 2-opt* between two different routes.
    fn two_opt_between_trips(&mut self, r_u: i32, r_v: i32) -> bool {
        let depot_u = r_u;
        let depot_end_u = self.params.nb_clients + r_u + 1;
        let depot_v = r_v;
        let depot_end_v = self.params.nb_clients + r_v + 1;

        // Get nodes from route U.
        let mut nodes_u: Vec<i32> = Vec::new();
        let mut curr = self.clients[depot_u as usize].next_idx;
        while curr != depot_end_u {
            if !self.clients[curr as usize].is_depot {
                nodes_u.push(curr);
            }
            curr = self.clients[curr as usize].next_idx;
        }

        // Get nodes from route V.
        let mut nodes_v: Vec<i32> = Vec::new();
        curr = self.clients[depot_v as usize].next_idx;
        while curr != depot_end_v {
            if !self.clients[curr as usize].is_depot {
                nodes_v.push(curr);
            }
            curr = self.clients[curr as usize].next_idx;
        }

        if nodes_u.is_empty() || nodes_v.is_empty() {
            return false;
        }

        // Try all pairs.
        for &u in &nodes_u {
            let u_next = self.clients[u as usize].next_idx;
            for &v in &nodes_v {
                let v_prev = self.clients[v as usize].prev_idx;

                // Old: u -> u_next (route U), v_prev -> v (route V)
                // New: u -> v (route U), u_next -> ... -> depot_U -> ... -> depot_V -> ... -> v_prev -> u_next (route V, 2-opt*)
                let old_cost = self.params.time_cost.get(u as usize, u_next as usize)
                    + self.params.time_cost.get(v_prev as usize, v as usize);

                let new_cost = self.params.time_cost.get(u as usize, v as usize)
                    + self.params.time_cost.get(u_next as usize, depot_end_u as usize)
                    + self.params.time_cost.get(depot_end_u as usize, depot_v as usize)
                    + self.params.time_cost.get(depot_v as usize, depot_end_v as usize)
                    + self.params.time_cost.get(depot_end_v as usize, u_next as usize)
                    + self.params.time_cost.get(v_prev as usize, u_next as usize);

                if new_cost + 2 < old_cost {
                    // Apply 2-opt* crossover.
                    // Route U: depot -> ... -> u -> v -> ... -> depot_V (old route V ending at v)
                    // Route V: depot -> ... -> u_next -> ... -> depot_U -> ... -> depot -> ... -> v_prev -> u_next

                    // This is a complex move — simplified version just swaps the tails.
                    self.clients[u as usize].next_idx = v;
                    self.clients[v as usize].prev_idx = u;
                    self.clients[v_prev as usize].next_idx = u_next;
                    self.clients[u_next as usize].prev_idx = v_prev;

                    // Update route assignments.
                    let mut curr_node = v;
                    while curr_node != depot_end_v {
                        self.clients[curr_node as usize].route_idx = r_u;
                        curr_node = self.clients[curr_node as usize].next_idx;
                    }

                    curr_node = u_next;
                    let route_v_new = r_u; // will be updated
                    while curr_node != depot_end_u {
                        self.clients[curr_node as usize].route_idx = r_v;
                        curr_node = self.clients[curr_node as usize].next_idx;
                    }

                    self.routes[r_u as usize] = self.compute_route_data(r_u);
                    self.routes[r_v as usize] = self.compute_route_data(r_v);
                    return true;
                }
            }
        }

        false
    }

    // =========================================================================
    // TIME WINDOW REORDERING
    // =========================================================================

    /// Reorder time windows within a route if beneficial.
    fn reorder_tws_if_needed(&mut self) {
        for r in 0..self.params.nb_vehicles as i32 {
            if self.empty_routes.contains(&r) || self.routes[r as usize].nb_customers < 2 {
                continue;
            }

            // Check if reordering by earliest arrival improves the solution.
            let depot_idx = r;
            let depot_end_idx = self.params.nb_clients + r + 1;

            // Collect nodes in order.
            let mut nodes: Vec<i32> = Vec::new();
            let mut curr = self.clients[depot_idx as usize].next_idx;
            while curr != depot_end_idx {
                if !self.clients[curr as usize].is_depot {
                    nodes.push(curr);
                }
                curr = self.clients[curr as usize].next_idx;
            }

            if nodes.len() < 2 {
                continue;
            }

            // Try sorting by earliest arrival.
            let mut sorted = nodes.clone();
            sorted.sort_by(|&a, &b| {
                self.params.cli[a as usize]
                    .earliest_arrival
                    .cmp(&self.params.cli[b as usize].earliest_arrival)
            });

            if sorted != nodes {
                // Calculate cost of current order.
                let current_cost = self.calculate_route_cost(r);

                // Calculate cost of sorted order (simplified: just check TW feasibility).
                let mut has_tw_violation = false;
                let mut arrival_time = 0i32;
                for &node in &sorted {
                    let edge_dist = if nodes.is_empty() || node == sorted[0] {
                        self.params.time_cost.get(0, node as usize)
                    } else {
                        let prev = sorted.iter().position(|&x| x == node).unwrap();
                        if prev == 0 {
                            self.params.time_cost.get(0, node as usize)
                        } else {
                            self.params.time_cost.get(sorted[prev - 1] as usize, node as usize)
                        }
                    };
                    arrival_time += edge_dist;
                    if arrival_time > self.params.cli[node as usize].latest_arrival {
                        has_tw_violation = true;
                        break;
                    }
                }

                if !has_tw_violation && current_cost > 0 {
                    // TODO: Full reordering implementation would rebuild the route.
                    // For now, skip this optimization.
                }
            }
        }
    }

    /// Calculate the total cost of a route.
    fn calculate_route_cost(&self, r: i32) -> i32 {
        let depot_idx = r;
        let depot_end_idx = self.params.nb_clients + r + 1;

        let mut cost = 0i32;
        let mut prev = depot_idx as i32;
        let mut curr = self.clients[depot_idx as usize].next_idx;

        while curr != depot_end_idx {
            cost += self.params.time_cost.get(prev as usize, curr as usize);
            prev = curr;
            curr = self.clients[curr as usize].next_idx;
        }
        cost += self.params.time_cost.get(prev as usize, 0); // back to depot

        cost
    }

    // =========================================================================
    // RELOCATE STAR
    // =========================================================================

    /// Try RelocateStar moves on recently modified routes.
    fn relocate_star(&mut self) -> bool {
        // Simplified: just check if we found any improvements this iteration.
        // Full implementation would try relocating each node to all recently modified routes.
        false
    }
}
