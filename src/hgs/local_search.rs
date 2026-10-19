//! Local search operators for HGS-CVRP/VRPTW.
//!
//! Implements the local search moves and constructive heuristics used by Hybrid Genetic Search:
//! - **RI (Relocate Individual)**: Remove a client (or two) and insert elsewhere
//! - **SWAP**: Swap clients between or within routes
//! - **2-opt / 2-opt***: Reverse segments within or between routes
//! - **Constructive heuristics**: Seed-based and sweep construction

use std::cmp::max;

use crate::hgs::circle_sector::CircleSector;
use crate::hgs::individual::SortableF64;
use crate::hgs::{Individual, Params, ProblemType};

use crate::hgs::params::MY_EPSILON;

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
            earliest_arrival: (tw2.earliest_arrival - delta).max(tw1.earliest_arrival) - delta_wait_time,
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

impl<'p> HGSLocalSearch<'p> {
    /// Create a new local search engine.
    pub fn new(params: &'p Params) -> Self {
        let nb_clients = params.nb_clients as usize;
        let nb_vehicles = params.nb_vehicles as usize;
        let total_nodes = nb_clients + 2 * nb_vehicles + 1;

        let mut nodes = vec![Node::default(); total_nodes];
        for i in 1..=nb_clients {
            nodes[i].cour = i as i32;
            nodes[i].is_depot = false;
        }

        let routes = (0..nb_vehicles)
            .map(|r| HGSRoute {
                cour: r as i32,
                ..Default::default()
            })
            .collect();

        let order_nodes: Vec<usize> = (1..=nb_clients).collect();
        let order_routes: Vec<usize> = (0..nb_vehicles).collect();

        Self {
            params,
            search_completed: false,
            nb_moves: 0,
            order_nodes,
            order_routes,
            empty_routes: std::collections::BTreeSet::new(),
            loop_id: 0,
            nodes,
            routes,
            best_insert_initialized_for_route: vec![false; nb_vehicles],
            best_insert_client: vec![vec![ThreeBestInsert::new(); nb_clients + 1]; nb_vehicles],
            best_insert_client_tw: vec![vec![ThreeBestInsert::new(); nb_clients + 1]; nb_vehicles],
            penalty_capacity_ls: 0.0,
            penalty_time_warp_ls: 0.0,
            correlated_vertices: Vec::new(),
            nb_granular: 0,
        }
        .with_granular(params.config.nb_granular)
    }

    fn with_granular(mut self, nb_granular: i32) -> Self {
        self.set_granular(nb_granular);
        self
    }

    /// Use the `nb_granular` closest clients as neighborhoods (HGS
    /// SetCorrelatedVertices, used when the granular size grows).
    pub fn set_granular(&mut self, nb_granular: i32) {
        self.nb_granular = nb_granular.max(1);
        self.correlated_vertices = self
            .params
            .correlated_vertices_for(nb_granular)
            .into_iter()
            .map(|list| list.into_iter().map(|v| v as usize).collect())
            .collect();
    }

    #[inline(always)]
    pub fn depot_start(&self, r: usize) -> usize {
        self.params.nb_clients as usize + 1 + r
    }

    #[inline(always)]
    pub fn depot_end(&self, r: usize) -> usize {
        self.params.nb_clients as usize + 1 + self.params.nb_vehicles as usize + r
    }

    #[inline(always)]
    fn penalty_excess_load(&self, load: i32) -> f64 {
        max(0, load - self.params.vehicle_capacity) as f64 * self.penalty_capacity_ls
    }

    #[inline(always)]
    fn penalty_time_windows(&self, tw_data: &TimeWindowData) -> f64 {
        if matches!(self.params.problem_type, ProblemType::Vrptw | ProblemType::Evrp) {
            (tw_data.time_warp + (tw_data.latest_release_time - tw_data.latest_arrival).max(0.)) as f64
                * self.penalty_time_warp_ls
        } else {
            0.0
        }
    }

    #[inline(always)]
    pub fn insert_node(&mut self, to_insert: usize, insertion_point: usize) {
        let prev = self.nodes[to_insert].prev as usize;
        let next = self.nodes[to_insert].next as usize;
        let ins_next = self.nodes[insertion_point].next as usize;

        self.nodes[prev].next = next as i32;
        self.nodes[next].prev = prev as i32;

        self.nodes[ins_next].prev = to_insert as i32;
        self.nodes[to_insert].prev = insertion_point as i32;
        self.nodes[to_insert].next = ins_next as i32;
        self.nodes[insertion_point].next = to_insert as i32;

        self.nodes[to_insert].route_idx = self.nodes[insertion_point].route_idx;
    }

    #[inline(always)]
    fn insert_unlinked_node(&mut self, to_insert: usize, insertion_point: usize) {
        let ins_next = self.nodes[insertion_point].next as usize;

        self.nodes[ins_next].prev = to_insert as i32;
        self.nodes[to_insert].prev = insertion_point as i32;
        self.nodes[to_insert].next = ins_next as i32;
        self.nodes[insertion_point].next = to_insert as i32;

        self.nodes[to_insert].route_idx = self.nodes[insertion_point].route_idx;
    }

    #[inline(always)]
    pub fn swap_node(&mut self, u: usize, v: usize) {
        let u_pred = self.nodes[u].prev as usize;
        let u_suiv = self.nodes[u].next as usize;
        let v_pred = self.nodes[v].prev as usize;
        let v_suiv = self.nodes[v].next as usize;
        let route_u = self.nodes[u].route_idx;
        let route_v = self.nodes[v].route_idx;

        if u_suiv == v {
            self.nodes[u_pred].next = v as i32;
            self.nodes[v].prev = u_pred as i32;
            self.nodes[v].next = u as i32;
            self.nodes[u].prev = v as i32;
            self.nodes[u].next = v_suiv as i32;
            self.nodes[v_suiv].prev = u as i32;
        } else if v_suiv == u {
            self.nodes[v_pred].next = u as i32;
            self.nodes[u].prev = v_pred as i32;
            self.nodes[u].next = v as i32;
            self.nodes[v].prev = u as i32;
            self.nodes[v].next = u_suiv as i32;
            self.nodes[u_suiv].prev = v as i32;
        } else {
            self.nodes[u_pred].next = v as i32;
            self.nodes[u_suiv].prev = v as i32;
            self.nodes[v_pred].next = u as i32;
            self.nodes[v_suiv].prev = u as i32;

            self.nodes[u].prev = v_pred as i32;
            self.nodes[u].next = v_suiv as i32;
            self.nodes[v].prev = u_pred as i32;
            self.nodes[v].next = u_suiv as i32;
        }

        self.nodes[u].route_idx = route_v;
        self.nodes[v].route_idx = route_u;
    }

    /// Update preprocessed route data (duration, load, TW, penalties, barycenter).
    pub fn update_route_data(&mut self, route_idx: usize) {
        let depot_s = self.depot_start(route_idx);
        let depot_e = self.depot_end(route_idx);

        let mut myplace = 0i32;
        let mut myload = 0i32;
        let mut mytime = 0_f64;
        let mut my_reversal_distance = 0_f64;
        // HGS accumulates the scaled coordinates in ints, truncating each sum.
        let mut cumulated_x = 0i32;
        let mut cumulated_y = 0i32;

        self.nodes[depot_s].position = 0;
        self.nodes[depot_s].cumulated_load = 0;
        self.nodes[depot_s].cumulated_time = 0.;
        self.nodes[depot_s].cumulated_reversal_distance = 0.;

        let mut mynode = depot_s;
        let mut first_it = true;
        let mut seed_tw = TimeWindowData::default();
        let mut seed_node: Option<usize> = None;

        while mynode != depot_e {
            let next_node = self.nodes[mynode].next as usize;
            myplace += 1;
            self.nodes[next_node].position = myplace;
            let cour = self.nodes[next_node].cour as usize;
            let prev_cour = self.nodes[mynode].cour as usize;

            myload += self.params.cli[cour].demand;
            let edge_cost = self.params.time_cost.get(prev_cour, cour);
            let edge_reverse = self.params.time_cost.get(cour, prev_cour);
            mytime += edge_cost + self.params.cli[cour].service_duration;
            my_reversal_distance += edge_reverse - edge_cost;

            self.nodes[next_node].cumulated_load = myload;
            self.nodes[next_node].cumulated_time = mytime;
            self.nodes[next_node].cumulated_reversal_distance = my_reversal_distance;
            self.nodes[next_node].prefix_tw_data = TimeWindowData::merge(
                &self.nodes[mynode].prefix_tw_data,
                &self.nodes[next_node].tw_data,
                self.params,
            );
            self.nodes[next_node].is_seed = false;
            self.nodes[next_node].next_seed = -1;

            if !self.nodes[next_node].is_depot {
                cumulated_x = (cumulated_x as f64 + self.params.cli[cour].coord_x) as i32;
                cumulated_y = (cumulated_y as f64 + self.params.cli[cour].coord_y) as i32;
                if first_it {
                    self.routes[route_idx].sector.initialize(self.params.cli[cour].polar_angle);
                } else {
                    self.routes[route_idx].sector.extend(self.params.cli[cour].polar_angle);
                }
                // Every fourth node stores the merged data up to the next one.
                match myplace % 4 {
                    0 => {
                        if let Some(seed) = seed_node {
                            self.nodes[seed].is_seed = true;
                            self.nodes[seed].to_next_seed_tw_data =
                                TimeWindowData::merge(&seed_tw, &self.nodes[next_node].tw_data, self.params);
                            self.nodes[seed].next_seed = next_node as i32;
                        }
                        seed_node = Some(next_node);
                    }
                    1 => seed_tw = self.nodes[next_node].tw_data.clone(),
                    _ => seed_tw = TimeWindowData::merge(&seed_tw, &self.nodes[next_node].tw_data, self.params),
                }
            }

            first_it = false;
            mynode = next_node;
        }

        let route = &mut self.routes[route_idx];
        route.duration = mytime;
        route.load = myload;
        route.tw_data = self.nodes[depot_e].prefix_tw_data.clone();
        route.nb_customers = myplace - 1;
        route.reversal_distance = my_reversal_distance;
        route.when_last_modified = self.nb_moves;
        route.is_delta_removel_tw_outdated = true;
        self.routes[route_idx].penalty = self.penalty_excess_load(myload)
            + self.penalty_time_windows(&self.routes[route_idx].tw_data);

        // Postfix TW data in reverse direction
        mynode = depot_e;
        while mynode != depot_s {
            let prev_node = self.nodes[mynode].prev as usize;
            self.nodes[prev_node].postfix_tw_data = TimeWindowData::merge(
                &self.nodes[prev_node].tw_data,
                &self.nodes[mynode].postfix_tw_data,
                self.params,
            );
            mynode = prev_node;
        }

        let route = &mut self.routes[route_idx];
        if route.nb_customers == 0 {
            route.polar_angle_barycenter = 1e30;
            self.empty_routes.insert(route_idx);
        } else {
            let inv_n = 1.0 / route.nb_customers as f64;
            let diff_x = cumulated_x as f64 * inv_n - self.params.cli[0].coord_x;
            let diff_y = cumulated_y as f64 * inv_n - self.params.cli[0].coord_y;
            route.polar_angle_barycenter = diff_y.atan2(diff_x);

            // Enforce a minimum circle sector size.
            if self.params.min_circle_sector_size > 0 {
                let sector_size =
                    CircleSector::positive_mod(route.sector.end as i32 - route.sector.start as i32) as i32;
                let grow_by = (self.params.min_circle_sector_size as i32 - sector_size + 1) / 2;
                if grow_by > 0 {
                    let (start, end) = (route.sector.start as i32, route.sector.end as i32);
                    route.sector.extend(CircleSector::positive_mod(start - grow_by));
                    route.sector.extend(CircleSector::positive_mod(end + grow_by));
                }
            }
            self.empty_routes.remove(&route_idx);
        }
    }

    /// Time window data of the route segment from `u` to `v`, both included.
    pub fn get_route_segment_tw_data(&self, u: usize, v: usize) -> TimeWindowData {
        if self.nodes[u].is_depot {
            return self.nodes[v].prefix_tw_data.clone();
        }
        if self.nodes[v].is_depot {
            return self.nodes[u].postfix_tw_data.clone();
        }

        let target = self.nodes[v].position;
        let mut tw_data = self.nodes[u].tw_data.clone();
        let mut curr = u;
        while curr != v {
            let node = &self.nodes[curr];
            if node.is_seed && node.position + 4 <= target {
                tw_data = TimeWindowData::merge(&tw_data, &node.to_next_seed_tw_data, self.params);
                curr = node.next_seed as usize;
            } else {
                curr = node.next as usize;
                tw_data = TimeWindowData::merge(&tw_data, &self.nodes[curr].tw_data, self.params);
            }
        }
        tw_data
    }

    /// Merge consecutive time window segments left to right.
    fn merge_all(&self, parts: &[&TimeWindowData]) -> TimeWindowData {
        let mut result = parts[0].clone();
        for part in &parts[1..] {
            result = TimeWindowData::merge(&result, part, self.params);
        }
        result
    }

    /// Penalty difference of replacing the route time windows `old` by `new`.
    fn delta_penalty_time_windows(&self, new: &TimeWindowData, old: &TimeWindowData) -> f64 {
        self.penalty_time_windows(new) - self.penalty_time_windows(old)
    }

    /// Load an individual into the local search representation.
    pub fn load_individual(&mut self, indiv: &Individual<'_>) {
        self.empty_routes.clear();
        self.nb_moves = 0;
        debug_assert_eq!(
            indiv.chrom_r.iter().map(Vec::len).sum::<usize>(),
            self.params.nb_clients as usize,
            "every client must be routed"
        );

        let depot_tw = TimeWindowData {
            first_node_index: 0,
            last_node_index: 0,
            duration: 0.,
            time_warp: 0.,
            earliest_arrival: self.params.cli[0].earliest_arrival,
            latest_arrival: self.params.cli[0].latest_arrival,
            latest_release_time: self.params.cli[0].release_time,
        };

        for i in 1..=self.params.nb_clients as usize {
            self.nodes[i].cour = i as i32;
            self.nodes[i].is_depot = false;
            self.nodes[i].prev = 0;
            self.nodes[i].next = 0;
            self.nodes[i].route_idx = 0;
            self.nodes[i].when_last_tested_ri = -1;
            self.nodes[i].tw_data = TimeWindowData {
                first_node_index: i as i32,
                last_node_index: i as i32,
                duration: self.params.cli[i].service_duration,
                time_warp: 0.,
                earliest_arrival: self.params.cli[i].earliest_arrival,
                latest_arrival: self.params.cli[i].latest_arrival,
                latest_release_time: self.params.cli[i].release_time,
            };
        }

        for r in 0..self.params.nb_vehicles as usize {
            let start = self.depot_start(r);
            let end = self.depot_end(r);

            self.nodes[start].cour = 0;
            self.nodes[start].is_depot = true;
            self.nodes[start].route_idx = r as i32;
            self.nodes[start].tw_data = depot_tw.clone();
            self.nodes[start].prefix_tw_data = depot_tw.clone();
            self.nodes[start].postfix_tw_data = depot_tw.clone();

            self.nodes[end].cour = 0;
            self.nodes[end].is_depot = true;
            self.nodes[end].route_idx = r as i32;
            self.nodes[end].tw_data = depot_tw.clone();
            self.nodes[end].prefix_tw_data = depot_tw.clone();
            self.nodes[end].postfix_tw_data = depot_tw.clone();

            self.routes[r].cour = r as i32;

            let route = &indiv.chrom_r[r];
            if !route.is_empty() {
                let first = route[0];
                self.nodes[first].route_idx = r as i32;
                self.nodes[first].prev = start as i32;
                self.nodes[start].next = first as i32;

                let mut pred = first;
                for i in 1..route.len() {
                    let curr = route[i];
                    self.nodes[curr].route_idx = r as i32;
                    self.nodes[curr].prev = pred as i32;
                    self.nodes[pred].next = curr as i32;
                    pred = curr;
                }

                let last = pred;
                self.nodes[last].next = end as i32;
                self.nodes[end].prev = last as i32;
            } else {
                self.nodes[start].next = end as i32;
                self.nodes[end].prev = start as i32;
            }

            self.nodes[start].prev = end as i32;
            self.nodes[end].next = start as i32;

            self.update_route_data(r);
            self.routes[r].when_last_tested_large_nb = -1;
            self.best_insert_initialized_for_route[r] = false;
        }
    }

    /// Export the local search solution back to the individual.
    pub fn export_individual(&mut self, indiv: &mut Individual<'_>) {
        let mut route_polar_angles: Vec<(SortableF64, usize)> = Vec::new();
        for r in 0..self.params.nb_vehicles as usize {
            route_polar_angles.push((SortableF64(self.routes[r].polar_angle_barycenter), r));
        }
        route_polar_angles.sort();

        let mut pos = 0;
        for (r, &(_, old_r)) in route_polar_angles.iter().enumerate() {
            indiv.chrom_r[r].clear();
            let start = self.depot_start(old_r);
            let mut curr = self.nodes[start].next as usize;

            while !self.nodes[curr].is_depot {
                let cust = self.nodes[curr].cour as usize;
                if pos < indiv.chrom_t.len() {
                    indiv.chrom_t[pos] = cust;
                } else {
                    indiv.chrom_t.push(cust);
                }
                indiv.chrom_r[r].push(cust);
                pos += 1;
                curr = self.nodes[curr].next as usize;
            }
        }

        indiv.evaluate_complete_cost();
    }

    #[inline(always)]
    fn route_has_tw_violation(&self, r: usize) -> bool {
        matches!(self.params.problem_type, ProblemType::Vrptw | ProblemType::Evrp)
            && self.routes[r].tw_data.time_warp > MY_EPSILON
    }

    #[inline(always)]
    fn route_has_load_penalty(&self, r: usize) -> bool {
        self.routes[r].load > self.params.vehicle_capacity
    }

    /// Move single client U and insert it after V.
    pub fn move_single_client(&mut self, u: usize, v: usize) -> bool {
        let u_prev = self.nodes[u].prev as usize;
        let x = self.nodes[u].next as usize;
        let y = self.nodes[v].next as usize;

        if u == y {
            return false;
        }

        let u_cour = self.nodes[u].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let u_prev_cour = self.nodes[u_prev].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let y_cour = self.nodes[y].cour as usize;

        let mut cost_supp_u = (self.params.time_cost.get(u_prev_cour, x_cour)
            - self.params.time_cost.get(u_prev_cour, u_cour)
            - self.params.time_cost.get(u_cour, x_cour)) as f64;
        let mut cost_supp_v = (self.params.time_cost.get(v_cour, u_cour)
            + self.params.time_cost.get(u_cour, y_cour)
            - self.params.time_cost.get(v_cour, y_cour)) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        let r_v = self.nodes[v].route_idx as usize;
        let load_u = self.params.cli[u_cour].demand;

        let route_u_load_pen = self.route_has_load_penalty(r_u);
        let route_u_tw_pen = self.route_has_tw_violation(r_u);

        if r_u != r_v {
            if !route_u_load_pen && !route_u_tw_pen && cost_supp_u + cost_supp_v > -MY_EPSILON {
                return false;
            }

            let route_u_tw = TimeWindowData::merge(
                &self.nodes[u_prev].prefix_tw_data,
                &self.nodes[x].postfix_tw_data,
                self.params,
            );
            let v_prefix_u = TimeWindowData::merge(
                &self.nodes[v].prefix_tw_data,
                &self.nodes[u].tw_data,
                self.params,
            );
            let route_v_tw = TimeWindowData::merge(
                &v_prefix_u,
                &self.nodes[y].postfix_tw_data,
                self.params,
            );

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load - load_u)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
            cost_supp_v += self.penalty_excess_load(self.routes[r_v].load + load_u)
                + self.penalty_time_windows(&route_v_tw)
                - self.routes[r_v].penalty;
        } else {
            if !route_u_tw_pen && cost_supp_u + cost_supp_v > -MY_EPSILON {
                return false;
            }

            let route_u_tw = if self.nodes[u].position < self.nodes[v].position {
                let seg_tw = self.get_route_segment_tw_data(x, v);
                let p1 = TimeWindowData::merge(
                    &self.nodes[u_prev].prefix_tw_data,
                    &seg_tw,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &self.nodes[u].tw_data, self.params);
                TimeWindowData::merge(&p2, &self.nodes[y].postfix_tw_data, self.params)
            } else {
                let seg_tw = self.get_route_segment_tw_data(y, u_prev);
                let p1 = TimeWindowData::merge(
                    &self.nodes[v].prefix_tw_data,
                    &self.nodes[u].tw_data,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                TimeWindowData::merge(&p2, &self.nodes[x].postfix_tw_data, self.params)
            };

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
        }

        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        self.insert_node(u, v);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        if r_u != r_v {
            self.update_route_data(r_v);
        }

        true
    }

    /// Move two clients (U, X) and insert them after V.
    pub fn move_two_clients(&mut self, u: usize, v: usize) -> bool {
        let x = self.nodes[u].next as usize;
        if self.nodes[x].is_depot {
            return false;
        }
        let y = self.nodes[v].next as usize;
        if u == y || x == v {
            return false;
        }

        let u_prev = self.nodes[u].prev as usize;
        let x_next = self.nodes[x].next as usize;

        let u_cour = self.nodes[u].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let u_prev_cour = self.nodes[u_prev].cour as usize;
        let x_next_cour = self.nodes[x_next].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let y_cour = self.nodes[y].cour as usize;

        let mut cost_supp_u = (self.params.time_cost.get(u_prev_cour, x_next_cour)
            - self.params.time_cost.get(u_prev_cour, u_cour)
            - self.params.time_cost.get(x_cour, x_next_cour)) as f64;
        let mut cost_supp_v = (self.params.time_cost.get(v_cour, u_cour)
            + self.params.time_cost.get(x_cour, y_cour)
            - self.params.time_cost.get(v_cour, y_cour)) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        let r_v = self.nodes[v].route_idx as usize;
        let load_ux = self.params.cli[u_cour].demand + self.params.cli[x_cour].demand;

        let route_u_load_pen = self.route_has_load_penalty(r_u);
        let route_u_tw_pen = self.route_has_tw_violation(r_u);

        if r_u != r_v {
            if !route_u_load_pen && !route_u_tw_pen && cost_supp_u + cost_supp_v > -MY_EPSILON {
                return false;
            }

            let route_u_tw = TimeWindowData::merge(
                &self.nodes[u_prev].prefix_tw_data,
                &self.nodes[x_next].postfix_tw_data,
                self.params,
            );
            let ux_tw = TimeWindowData::merge(
                &self.nodes[u].tw_data,
                &self.nodes[x].tw_data,
                self.params,
            );
            let v_ux = TimeWindowData::merge(&self.nodes[v].prefix_tw_data, &ux_tw, self.params);
            let route_v_tw =
                TimeWindowData::merge(&v_ux, &self.nodes[y].postfix_tw_data, self.params);

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load - load_ux)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
            cost_supp_v += self.penalty_excess_load(self.routes[r_v].load + load_ux)
                + self.penalty_time_windows(&route_v_tw)
                - self.routes[r_v].penalty;
        } else {
            if !route_u_tw_pen && cost_supp_u + cost_supp_v > -MY_EPSILON {
                return false;
            }

            let ux_tw = TimeWindowData::merge(
                &self.nodes[u].tw_data,
                &self.nodes[x].tw_data,
                self.params,
            );
            let route_u_tw = if self.nodes[u].position < self.nodes[v].position {
                let seg_tw = self.get_route_segment_tw_data(x_next, v);
                let p1 = TimeWindowData::merge(
                    &self.nodes[u_prev].prefix_tw_data,
                    &seg_tw,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &ux_tw, self.params);
                TimeWindowData::merge(&p2, &self.nodes[y].postfix_tw_data, self.params)
            } else {
                let seg_tw = self.get_route_segment_tw_data(y, u_prev);
                let p1 = TimeWindowData::merge(
                    &self.nodes[v].prefix_tw_data,
                    &ux_tw,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                TimeWindowData::merge(&p2, &self.nodes[x_next].postfix_tw_data, self.params)
            };

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
        }

        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        self.insert_node(u, v);
        self.insert_node(x, u);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        if r_u != r_v {
            self.update_route_data(r_v);
        }

        true
    }

    /// Move two clients (X, U) reversed and insert them after V.
    pub fn move_two_clients_reversed(&mut self, u: usize, v: usize) -> bool {
        let x = self.nodes[u].next as usize;
        if self.nodes[x].is_depot {
            return false;
        }
        let y = self.nodes[v].next as usize;
        if u == y || x == v {
            return false;
        }

        let u_prev = self.nodes[u].prev as usize;
        let x_next = self.nodes[x].next as usize;

        let u_cour = self.nodes[u].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let u_prev_cour = self.nodes[u_prev].cour as usize;
        let x_next_cour = self.nodes[x_next].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let y_cour = self.nodes[y].cour as usize;

        let mut cost_supp_u = (self.params.time_cost.get(u_prev_cour, x_next_cour)
            - self.params.time_cost.get(u_prev_cour, u_cour)
            - self.params.time_cost.get(u_cour, x_cour)
            - self.params.time_cost.get(x_cour, x_next_cour)) as f64;
        let mut cost_supp_v = (self.params.time_cost.get(v_cour, x_cour)
            + self.params.time_cost.get(x_cour, u_cour)
            + self.params.time_cost.get(u_cour, y_cour)
            - self.params.time_cost.get(v_cour, y_cour)) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        let r_v = self.nodes[v].route_idx as usize;
        let load_ux = self.params.cli[u_cour].demand + self.params.cli[x_cour].demand;

        let route_u_load_pen = self.route_has_load_penalty(r_u);
        let route_u_tw_pen = self.route_has_tw_violation(r_u);

        if r_u != r_v {
            if !route_u_load_pen && !route_u_tw_pen && cost_supp_u + cost_supp_v > -MY_EPSILON {
                return false;
            }

            let route_u_tw = TimeWindowData::merge(
                &self.nodes[u_prev].prefix_tw_data,
                &self.nodes[x_next].postfix_tw_data,
                self.params,
            );
            let xu_tw = TimeWindowData::merge(
                &self.nodes[x].tw_data,
                &self.nodes[u].tw_data,
                self.params,
            );
            let v_xu = TimeWindowData::merge(&self.nodes[v].prefix_tw_data, &xu_tw, self.params);
            let route_v_tw =
                TimeWindowData::merge(&v_xu, &self.nodes[y].postfix_tw_data, self.params);

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load - load_ux)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
            cost_supp_v += self.penalty_excess_load(self.routes[r_v].load + load_ux)
                + self.penalty_time_windows(&route_v_tw)
                - self.routes[r_v].penalty;
        } else {
            if !route_u_tw_pen && cost_supp_u + cost_supp_v > -MY_EPSILON {
                return false;
            }

            let xu_tw = TimeWindowData::merge(
                &self.nodes[x].tw_data,
                &self.nodes[u].tw_data,
                self.params,
            );
            let route_u_tw = if self.nodes[u].position < self.nodes[v].position {
                let seg_tw = self.get_route_segment_tw_data(x_next, v);
                let p1 = TimeWindowData::merge(
                    &self.nodes[u_prev].prefix_tw_data,
                    &seg_tw,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &xu_tw, self.params);
                TimeWindowData::merge(&p2, &self.nodes[y].postfix_tw_data, self.params)
            } else {
                let seg_tw = self.get_route_segment_tw_data(y, u_prev);
                let p1 = TimeWindowData::merge(
                    &self.nodes[v].prefix_tw_data,
                    &xu_tw,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                TimeWindowData::merge(&p2, &self.nodes[x_next].postfix_tw_data, self.params)
            };

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
        }

        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        self.insert_node(x, v);
        self.insert_node(u, x);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        if r_u != r_v {
            self.update_route_data(r_v);
        }

        true
    }

    /// Swap two single clients U and V.
    pub fn swap_two_single_clients(&mut self, u: usize, v: usize) -> bool {
        let u_prev = self.nodes[u].prev as usize;
        let x = self.nodes[u].next as usize;
        let v_prev = self.nodes[v].prev as usize;
        let y = self.nodes[v].next as usize;

        if u == v_prev || u == y {
            return false;
        }

        let u_cour = self.nodes[u].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let u_prev_cour = self.nodes[u_prev].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let v_prev_cour = self.nodes[v_prev].cour as usize;
        let y_cour = self.nodes[y].cour as usize;

        let mut cost_supp_u = (self.params.time_cost.get(u_prev_cour, v_cour)
            + self.params.time_cost.get(v_cour, x_cour)
            - self.params.time_cost.get(u_prev_cour, u_cour)
            - self.params.time_cost.get(u_cour, x_cour)) as f64;
        let mut cost_supp_v = (self.params.time_cost.get(v_prev_cour, u_cour)
            + self.params.time_cost.get(u_cour, y_cour)
            - self.params.time_cost.get(v_prev_cour, v_cour)
            - self.params.time_cost.get(v_cour, y_cour)) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        let r_v = self.nodes[v].route_idx as usize;
        let load_u = self.params.cli[u_cour].demand;
        let load_v = self.params.cli[v_cour].demand;

        let route_u_load_pen = self.route_has_load_penalty(r_u);
        let route_u_tw_pen = self.route_has_tw_violation(r_u);
        let route_v_load_pen = self.route_has_load_penalty(r_v);
        let route_v_tw_pen = self.route_has_tw_violation(r_v);

        if r_u != r_v {
            if !route_u_load_pen
                && !route_u_tw_pen
                && !route_v_load_pen
                && !route_v_tw_pen
                && cost_supp_u + cost_supp_v > -MY_EPSILON
            {
                return false;
            }

            let u_prefix_v = TimeWindowData::merge(
                &self.nodes[u_prev].prefix_tw_data,
                &self.nodes[v].tw_data,
                self.params,
            );
            let route_u_tw =
                TimeWindowData::merge(&u_prefix_v, &self.nodes[x].postfix_tw_data, self.params);

            let v_prefix_u = TimeWindowData::merge(
                &self.nodes[v_prev].prefix_tw_data,
                &self.nodes[u].tw_data,
                self.params,
            );
            let route_v_tw =
                TimeWindowData::merge(&v_prefix_u, &self.nodes[y].postfix_tw_data, self.params);

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load + load_v - load_u)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
            cost_supp_v += self.penalty_excess_load(self.routes[r_v].load + load_u - load_v)
                + self.penalty_time_windows(&route_v_tw)
                - self.routes[r_v].penalty;
        } else {
            if !route_u_tw_pen && cost_supp_u + cost_supp_v > -MY_EPSILON {
                return false;
            }

            let route_u_tw = if self.nodes[u].position < self.nodes[v].position {
                let seg_tw = self.get_route_segment_tw_data(x, v_prev);
                let p1 = TimeWindowData::merge(
                    &self.nodes[u_prev].prefix_tw_data,
                    &self.nodes[v].tw_data,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                let p3 = TimeWindowData::merge(&p2, &self.nodes[u].tw_data, self.params);
                TimeWindowData::merge(&p3, &self.nodes[y].postfix_tw_data, self.params)
            } else {
                let seg_tw = self.get_route_segment_tw_data(y, u_prev);
                let p1 = TimeWindowData::merge(
                    &self.nodes[v_prev].prefix_tw_data,
                    &self.nodes[u].tw_data,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                let p3 = TimeWindowData::merge(&p2, &self.nodes[v].tw_data, self.params);
                TimeWindowData::merge(&p3, &self.nodes[x].postfix_tw_data, self.params)
            };

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
        }

        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        self.swap_node(u, v);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        if r_u != r_v {
            self.update_route_data(r_v);
        }

        true
    }

    /// Swap two clients (U, X) for one client V.
    pub fn swap_two_clients_for_one(&mut self, u: usize, v: usize) -> bool {
        let x = self.nodes[u].next as usize;
        if self.nodes[x].is_depot {
            return false;
        }
        let u_prev = self.nodes[u].prev as usize;
        let x_next = self.nodes[x].next as usize;
        let v_prev = self.nodes[v].prev as usize;
        let y = self.nodes[v].next as usize;

        if u == v_prev || x == v_prev || u == y {
            return false;
        }

        let u_cour = self.nodes[u].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let u_prev_cour = self.nodes[u_prev].cour as usize;
        let x_next_cour = self.nodes[x_next].cour as usize;
        let v_prev_cour = self.nodes[v_prev].cour as usize;
        let y_cour = self.nodes[y].cour as usize;

        let mut cost_supp_u = (self.params.time_cost.get(u_prev_cour, v_cour)
            + self.params.time_cost.get(v_cour, x_next_cour)
            - self.params.time_cost.get(u_prev_cour, u_cour)
            - self.params.time_cost.get(x_cour, x_next_cour)) as f64;
        let mut cost_supp_v = (self.params.time_cost.get(v_prev_cour, u_cour)
            + self.params.time_cost.get(x_cour, y_cour)
            - self.params.time_cost.get(v_prev_cour, v_cour)
            - self.params.time_cost.get(v_cour, y_cour)) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        let r_v = self.nodes[v].route_idx as usize;
        let load_ux = self.params.cli[u_cour].demand + self.params.cli[x_cour].demand;
        let load_v = self.params.cli[v_cour].demand;

        let route_u_load_pen = self.route_has_load_penalty(r_u);
        let route_u_tw_pen = self.route_has_tw_violation(r_u);
        let route_v_load_pen = self.route_has_load_penalty(r_v);
        let route_v_tw_pen = self.route_has_tw_violation(r_v);

        if r_u != r_v {
            if !route_u_load_pen
                && !route_u_tw_pen
                && !route_v_load_pen
                && !route_v_tw_pen
                && cost_supp_u + cost_supp_v > -MY_EPSILON
            {
                return false;
            }

            let u_prefix_v = TimeWindowData::merge(
                &self.nodes[u_prev].prefix_tw_data,
                &self.nodes[v].tw_data,
                self.params,
            );
            let route_u_tw = TimeWindowData::merge(
                &u_prefix_v,
                &self.nodes[x_next].postfix_tw_data,
                self.params,
            );

            let ux_tw = TimeWindowData::merge(
                &self.nodes[u].tw_data,
                &self.nodes[x].tw_data,
                self.params,
            );
            let v_prefix_ux = TimeWindowData::merge(
                &self.nodes[v_prev].prefix_tw_data,
                &ux_tw,
                self.params,
            );
            let route_v_tw = TimeWindowData::merge(
                &v_prefix_ux,
                &self.nodes[y].postfix_tw_data,
                self.params,
            );

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load + load_v - load_ux)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
            cost_supp_v += self.penalty_excess_load(self.routes[r_v].load + load_ux - load_v)
                + self.penalty_time_windows(&route_v_tw)
                - self.routes[r_v].penalty;
        } else {
            if !route_u_tw_pen && cost_supp_u + cost_supp_v > -MY_EPSILON {
                return false;
            }

            let ux_tw = TimeWindowData::merge(
                &self.nodes[u].tw_data,
                &self.nodes[x].tw_data,
                self.params,
            );
            let route_u_tw = if self.nodes[u].position < self.nodes[v].position {
                let seg_tw = self.get_route_segment_tw_data(x_next, v_prev);
                let p1 = TimeWindowData::merge(
                    &self.nodes[u_prev].prefix_tw_data,
                    &self.nodes[v].tw_data,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                let p3 = TimeWindowData::merge(&p2, &ux_tw, self.params);
                TimeWindowData::merge(&p3, &self.nodes[y].postfix_tw_data, self.params)
            } else {
                let seg_tw = self.get_route_segment_tw_data(y, u_prev);
                let p1 = TimeWindowData::merge(
                    &self.nodes[v_prev].prefix_tw_data,
                    &ux_tw,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                let p3 = TimeWindowData::merge(&p2, &self.nodes[v].tw_data, self.params);
                TimeWindowData::merge(&p3, &self.nodes[x_next].postfix_tw_data, self.params)
            };

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
        }

        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        self.swap_node(u, v);
        self.insert_node(x, u);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        if r_u != r_v {
            self.update_route_data(r_v);
        }

        true
    }

    /// Swap two client pairs (U, X) and (V, Y).
    pub fn swap_two_client_pairs(&mut self, u: usize, v: usize) -> bool {
        let x = self.nodes[u].next as usize;
        let y = self.nodes[v].next as usize;
        let u_prev = self.nodes[u].prev as usize;
        let x_next = self.nodes[x].next as usize;

        if self.nodes[x].is_depot
            || self.nodes[y].is_depot
            || y == u_prev
            || u == y
            || x == v
            || v == x_next
        {
            return false;
        }

        let v_prev = self.nodes[v].prev as usize;
        let y_next = self.nodes[y].next as usize;

        let u_cour = self.nodes[u].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let y_cour = self.nodes[y].cour as usize;
        let u_prev_cour = self.nodes[u_prev].cour as usize;
        let x_next_cour = self.nodes[x_next].cour as usize;
        let v_prev_cour = self.nodes[v_prev].cour as usize;
        let y_next_cour = self.nodes[y_next].cour as usize;

        let mut cost_supp_u = (self.params.time_cost.get(u_prev_cour, v_cour)
            + self.params.time_cost.get(y_cour, x_next_cour)
            - self.params.time_cost.get(u_prev_cour, u_cour)
            - self.params.time_cost.get(x_cour, x_next_cour)) as f64;
        let mut cost_supp_v = (self.params.time_cost.get(v_prev_cour, u_cour)
            + self.params.time_cost.get(x_cour, y_next_cour)
            - self.params.time_cost.get(v_prev_cour, v_cour)
            - self.params.time_cost.get(y_cour, y_next_cour)) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        let r_v = self.nodes[v].route_idx as usize;
        let load_ux = self.params.cli[u_cour].demand + self.params.cli[x_cour].demand;
        let load_vy = self.params.cli[v_cour].demand + self.params.cli[y_cour].demand;

        let route_u_load_pen = self.route_has_load_penalty(r_u);
        let route_u_tw_pen = self.route_has_tw_violation(r_u);
        let route_v_load_pen = self.route_has_load_penalty(r_v);
        let route_v_tw_pen = self.route_has_tw_violation(r_v);

        if r_u != r_v {
            if !route_u_load_pen
                && !route_u_tw_pen
                && !route_v_load_pen
                && !route_v_tw_pen
                && cost_supp_u + cost_supp_v > -MY_EPSILON
            {
                return false;
            }

            let vy_tw = TimeWindowData::merge(
                &self.nodes[v].tw_data,
                &self.nodes[y].tw_data,
                self.params,
            );
            let ux_tw = TimeWindowData::merge(
                &self.nodes[u].tw_data,
                &self.nodes[x].tw_data,
                self.params,
            );

            let u_prefix_vy = TimeWindowData::merge(
                &self.nodes[u_prev].prefix_tw_data,
                &vy_tw,
                self.params,
            );
            let route_u_tw = TimeWindowData::merge(
                &u_prefix_vy,
                &self.nodes[x_next].postfix_tw_data,
                self.params,
            );

            let v_prefix_ux = TimeWindowData::merge(
                &self.nodes[v_prev].prefix_tw_data,
                &ux_tw,
                self.params,
            );
            let route_v_tw = TimeWindowData::merge(
                &v_prefix_ux,
                &self.nodes[y_next].postfix_tw_data,
                self.params,
            );

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load + load_vy - load_ux)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
            cost_supp_v += self.penalty_excess_load(self.routes[r_v].load + load_ux - load_vy)
                + self.penalty_time_windows(&route_v_tw)
                - self.routes[r_v].penalty;
        } else {
            if !route_u_tw_pen && cost_supp_u + cost_supp_v > -MY_EPSILON {
                return false;
            }

            let vy_tw = TimeWindowData::merge(
                &self.nodes[v].tw_data,
                &self.nodes[y].tw_data,
                self.params,
            );
            let ux_tw = TimeWindowData::merge(
                &self.nodes[u].tw_data,
                &self.nodes[x].tw_data,
                self.params,
            );

            let route_u_tw = if self.nodes[u].position < self.nodes[v].position {
                let seg_tw = self.get_route_segment_tw_data(x_next, v_prev);
                let p1 = TimeWindowData::merge(
                    &self.nodes[u_prev].prefix_tw_data,
                    &vy_tw,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                let p3 = TimeWindowData::merge(&p2, &ux_tw, self.params);
                TimeWindowData::merge(&p3, &self.nodes[y_next].postfix_tw_data, self.params)
            } else {
                let seg_tw = self.get_route_segment_tw_data(y_next, u_prev);
                let p1 = TimeWindowData::merge(
                    &self.nodes[v_prev].prefix_tw_data,
                    &ux_tw,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                let p3 = TimeWindowData::merge(&p2, &vy_tw, self.params);
                TimeWindowData::merge(&p3, &self.nodes[x_next].postfix_tw_data, self.params)
            };

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
        }

        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        self.swap_node(u, v);
        self.swap_node(x, y);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        if r_u != r_v {
            self.update_route_data(r_v);
        }

        true
    }

    /// 2-opt within a single route.
    pub fn two_opt_within_trip(&mut self, u: usize, v: usize) -> bool {
        if self.nodes[u].position >= self.nodes[v].position - 1 {
            return false;
        }

        let x = self.nodes[u].next as usize;
        let y = self.nodes[v].next as usize;
        let u_cour = self.nodes[u].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let y_cour = self.nodes[y].cour as usize;

        let mut cost = (self.params.time_cost.get(u_cour, v_cour)
            + self.params.time_cost.get(x_cour, y_cour)
            - self.params.time_cost.get(u_cour, x_cour)
            - self.params.time_cost.get(v_cour, y_cour)
            + self.nodes[v].cumulated_reversal_distance
            - self.nodes[x].cumulated_reversal_distance) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        if !self.route_has_tw_violation(r_u) && cost > -MY_EPSILON {
            return false;
        }

        let mut route_tw = self.nodes[u].prefix_tw_data.clone();
        let mut it = v;
        while it != u {
            route_tw = TimeWindowData::merge(&route_tw, &self.nodes[it].tw_data, self.params);
            it = self.nodes[it].prev as usize;
        }
        route_tw = TimeWindowData::merge(&route_tw, &self.nodes[y].postfix_tw_data, self.params);

        cost += self.penalty_excess_load(self.routes[r_u].load)
            + self.penalty_time_windows(&route_tw)
            - self.routes[r_u].penalty;

        if cost > -MY_EPSILON {
            return false;
        }

        let mut it = v;
        let mut ins_point = u;
        while it != x {
            let current = it;
            it = self.nodes[it].prev as usize;
            self.insert_node(current, ins_point);
            ins_point = current;
        }

        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);

        true
    }

    /// 2-opt* between two different routes.
    pub fn two_opt_between_trips(&mut self, u: usize, v: usize) -> bool {
        let x = self.nodes[u].next as usize;
        let y = self.nodes[v].next as usize;

        let u_cour = self.nodes[u].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let y_cour = self.nodes[y].cour as usize;

        let mut cost_supp_u = (self.params.time_cost.get(u_cour, y_cour)
            - self.params.time_cost.get(u_cour, x_cour)) as f64;
        let mut cost_supp_v = (self.params.time_cost.get(v_cour, x_cour)
            - self.params.time_cost.get(v_cour, y_cour)) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        let r_v = self.nodes[v].route_idx as usize;

        if !self.route_has_load_penalty(r_u)
            && !self.route_has_tw_violation(r_u)
            && !self.route_has_load_penalty(r_v)
            && !self.route_has_tw_violation(r_v)
            && cost_supp_u + cost_supp_v > -MY_EPSILON
        {
            return false;
        }

        let route_u_tw = TimeWindowData::merge(
            &self.nodes[u].prefix_tw_data,
            &self.nodes[y].postfix_tw_data,
            self.params,
        );
        let route_v_tw = TimeWindowData::merge(
            &self.nodes[v].prefix_tw_data,
            &self.nodes[x].postfix_tw_data,
            self.params,
        );

        cost_supp_u += self.penalty_excess_load(
            self.nodes[u].cumulated_load + self.routes[r_v].load - self.nodes[v].cumulated_load,
        ) + self.penalty_time_windows(&route_u_tw)
            - self.routes[r_u].penalty;

        cost_supp_v += self.penalty_excess_load(
            self.nodes[v].cumulated_load + self.routes[r_u].load - self.nodes[u].cumulated_load,
        ) + self.penalty_time_windows(&route_v_tw)
            - self.routes[r_v].penalty;

        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        let mut it_v = y;
        let mut ins = u;
        while !self.nodes[it_v].is_depot {
            let curr = it_v;
            it_v = self.nodes[it_v].next as usize;
            self.insert_node(curr, ins);
            ins = curr;
        }

        let mut it_u = x;
        ins = v;
        while !self.nodes[it_u].is_depot {
            let curr = it_u;
            it_u = self.nodes[it_u].next as usize;
            self.insert_node(curr, ins);
            ins = curr;
        }

        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        self.update_route_data(r_v);

        true
    }

    /// Run the full local search with the specified penalty values.
    pub fn run(
        &mut self,
        indiv: &mut Individual<'_>,
        penalty_capacity_ls: f64,
        penalty_time_warp_ls: f64,
    ) {
        let intensification = self.params.config.intensification_probability_ls;
        let run_intensification =
            (self.params.rng.borrow_mut().next() % 100) < intensification.max(0) as u64;
        self.penalty_capacity_ls = penalty_capacity_ls;
        self.penalty_time_warp_ls =
            if matches!(self.params.problem_type, ProblemType::Vrptw | ProblemType::Evrp) {
                penalty_time_warp_ls
            } else {
                0.0
            };

        self.load_individual(indiv);

        {
            let mut rng = self.params.rng.borrow_mut();
            rng.shuffle(&mut self.order_nodes);
            rng.shuffle(&mut self.order_routes);
            // Shuffle neighborhoods only sporadically to avoid O(n^2) work.
            for i in 1..=self.params.nb_clients as usize {
                if rng.next() % self.nb_granular as u64 == 0 {
                    rng.shuffle(&mut self.correlated_vertices[i]);
                }
            }
        }

        self.search_completed = false;
        self.loop_id = 0;
        while !self.search_completed {
            if self.loop_id > 1 {
                self.search_completed = true;
            }

            // Phase 1: classical route improvement (RI) moves.
            for pos_u in 0..self.params.nb_clients as usize {
                let u = self.order_nodes[pos_u];
                let last_test_ri = self.nodes[u].when_last_tested_ri;
                self.nodes[u].when_last_tested_ri = self.nb_moves;

                for k in 0..self.correlated_vertices[u].len() {
                    let v = self.correlated_vertices[u][k];
                    let r_u = self.nodes[u].route_idx as usize;
                    let r_v = self.nodes[v].route_idx as usize;
                    if self.loop_id != 0
                        && max(self.routes[r_u].when_last_modified, self.routes[r_v].when_last_modified)
                            <= last_test_ri
                    {
                        continue;
                    }
                    if self.move_single_client(u, v)
                        || self.move_two_clients(u, v)
                        || self.move_two_clients_reversed(u, v)
                        || (u < v && self.swap_two_single_clients(u, v))
                        || self.swap_two_clients_for_one(u, v)
                        || (u < v && self.swap_two_client_pairs(u, v))
                        || (r_u < r_v && self.two_opt_between_trips(u, v))
                        || (r_u == r_v && self.two_opt_within_trip(u, v))
                    {
                        continue;
                    }

                    // Also try inserting U right after the depot preceding V.
                    let v_prev = self.nodes[v].prev as usize;
                    if self.nodes[v_prev].is_depot {
                        let _ = self.move_single_client(u, v_prev)
                            || self.move_two_clients(u, v_prev)
                            || self.move_two_clients_reversed(u, v_prev)
                            || (r_u < self.nodes[v_prev].route_idx as usize
                                && self.two_opt_between_trips(u, v_prev));
                    }
                }

                // Phase 2: moves into an empty route.
                if self.loop_id > 0 && !self.empty_routes.is_empty() {
                    let depot_v = self.depot_start(*self.empty_routes.iter().next().unwrap());
                    let _ = self.move_single_client(u, depot_v)
                        || self.move_two_clients(u, depot_v)
                        || self.move_two_clients_reversed(u, depot_v)
                        || self.two_opt_between_trips(u, depot_v);
                }
            }

            // Phase 3: SWAP* and RELOCATE* between routes with overlapping sectors.
            if intensification != 0
                && self.search_completed
                && (intensification == 100 || run_intensification)
            {
                for pos_u in 0..self.params.nb_vehicles as usize {
                    let r_u = self.order_routes[pos_u];
                    if self.routes[r_u].nb_customers == 0 {
                        continue;
                    }
                    let last_test_large_nb = self.routes[r_u].when_last_tested_large_nb;
                    self.routes[r_u].when_last_tested_large_nb = self.nb_moves;
                    for pos_v in 0..self.params.nb_vehicles as usize {
                        let r_v = self.order_routes[pos_v];
                        if self.routes[r_v].nb_customers == 0
                            || r_u >= r_v
                            || (self.loop_id > 0
                                && max(self.routes[r_u].when_last_modified, self.routes[r_v].when_last_modified)
                                    <= last_test_large_nb)
                            || !CircleSector::overlaps(
                                &self.routes[r_u].sector,
                                &self.routes[r_v].sector,
                                self.params.circle_sector_overlap_tolerance,
                            )
                        {
                            continue;
                        }
                        if !self.relocate_star(r_u, r_v)
                            && (self.params.config.skip_swap_star_dist || !self.swap_star(r_u, r_v, false))
                            && self.params.config.use_swap_star_tw
                        {
                            self.swap_star(r_u, r_v, true);
                        }
                    }
                }
            }
            self.loop_id += 1;
        }

        self.export_individual(indiv);
    }

    /// Clients of a route in order.
    fn route_clients(&self, r: usize) -> Vec<usize> {
        let mut clients = Vec::with_capacity(self.routes[r].nb_customers.max(0) as usize);
        let mut node = self.nodes[self.depot_start(r)].next as usize;
        while !self.nodes[node].is_depot {
            clients.push(node);
            node = self.nodes[node].next as usize;
        }
        clients
    }

    fn cost(&self, a: usize, b: usize) -> f64 {
        self.params.time_cost.get(self.nodes[a].cour as usize, self.nodes[b].cour as usize)
    }

    /// RELOCATE*: the best move of one client of `r_u` after a client of `r_v`.
    fn relocate_star(&mut self, r_u: usize, r_v: usize) -> bool {
        let mut best_cost = 0.;
        let mut best: Option<(usize, usize)> = None;
        for u in self.route_clients(r_u) {
            let (prev, next) = (self.nodes[u].prev as usize, self.nodes[u].next as usize);
            let route_u_tw = TimeWindowData::merge(
                &self.nodes[prev].prefix_tw_data,
                &self.nodes[next].postfix_tw_data,
                self.params,
            );
            let load_u = self.params.cli[self.nodes[u].cour as usize].demand;
            let cost_supp_u = (self.cost(prev, next) - self.cost(prev, u) - self.cost(u, next)) as f64
                + self.penalty_excess_load(self.routes[r_u].load - load_u)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
            for v in self.route_clients(r_v) {
                let v_next = self.nodes[v].next as usize;
                let route_v_tw = self.merge_all(&[
                    &self.nodes[v].prefix_tw_data,
                    &self.nodes[u].tw_data,
                    &self.nodes[v_next].postfix_tw_data,
                ]);
                let cost_supp_v = (self.cost(v, u) + self.cost(u, v_next) - self.cost(v, v_next)) as f64
                    + self.penalty_excess_load(self.routes[r_v].load + load_u)
                    + self.penalty_time_windows(&route_v_tw)
                    - self.routes[r_v].penalty;
                if cost_supp_u + cost_supp_v < best_cost - MY_EPSILON {
                    best_cost = cost_supp_u + cost_supp_v;
                    best = Some((u, v));
                }
            }
        }
        let Some((u, v)) = best else { return false };
        self.insert_node(u, v);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        self.update_route_data(r_v);
        true
    }

    /// Insertion costs of each client of `r1` into `r2` (SWAP* preprocessing),
    /// with time-warp penalty deltas when `with_tw`.
    fn preprocess_insertions(&mut self, r1: usize, r2: usize, with_tw: bool) {
        let depot2 = self.depot_start(r2);
        let first2 = self.nodes[depot2].next as usize;
        let route2_tw = self.routes[r2].tw_data.clone();
        let tw_outdated = self.routes[r1].is_delta_removel_tw_outdated;
        let route1_tw = self.routes[r1].tw_data.clone();
        for u in self.route_clients(r1) {
            let (prev, next) = (self.nodes[u].prev as usize, self.nodes[u].next as usize);
            let removal = self.cost(prev, next) - self.cost(prev, u) - self.cost(u, next);
            if !with_tw {
                self.nodes[u].delta_removal = removal;
            } else if tw_outdated {
                let tw = TimeWindowData::merge(
                    &self.nodes[prev].prefix_tw_data,
                    &self.nodes[next].postfix_tw_data,
                    self.params,
                );
                // Keep fractional time-window penalties in the insertion cost.
                self.nodes[u].delta_removal_tw =
                    removal + self.delta_penalty_time_windows(&tw, &route1_tw);
            }
            let cour = self.nodes[u].cour as usize;
            let when = if with_tw {
                self.best_insert_client_tw[r2][cour].when_last_calculated
            } else {
                self.best_insert_client[r2][cour].when_last_calculated
            };
            if self.routes[r2].when_last_modified <= when {
                continue;
            }
            let mut option = ThreeBestInsert::new();
            option.when_last_calculated = self.nb_moves;
            let insertion = |ls: &Self, v: usize| {
                let v_next = ls.nodes[v].next as usize;
                let mut delta = ls.cost(v, u) + ls.cost(u, v_next) - ls.cost(v, v_next);
                if with_tw {
                    let tw = ls.merge_all(&[
                        &ls.nodes[v].prefix_tw_data,
                        &ls.nodes[u].tw_data,
                        &ls.nodes[v_next].postfix_tw_data,
                    ]);
                    delta += ls.delta_penalty_time_windows(&tw, &route2_tw);
                }
                delta
            };
            option.best_cost[0] = insertion(self, depot2);
            option.best_location[0] = depot2 as i32;
            let mut v = first2;
            while !self.nodes[v].is_depot {
                option.compare_and_add(insertion(self, v), v as i32);
                v = self.nodes[v].next as usize;
            }
            if with_tw {
                self.best_insert_client_tw[r2][cour] = option;
            } else {
                self.best_insert_client[r2][cour] = option;
            }
        }
        if with_tw {
            self.routes[r1].is_delta_removel_tw_outdated = false;
        }
    }

    /// Cheapest insertion of `u` into the route of `v` once `v` is removed:
    /// one of the three preprocessed positions not adjacent to `v`, or `v`'s
    /// own place. Returns the cost and the node to insert after.
    fn cheapest_insert_simult_removal(&self, u: usize, v: usize, with_tw: bool) -> (f64, usize) {
        let r_v = self.nodes[v].route_idx as usize;
        let cour = self.nodes[u].cour as usize;
        let best = if with_tw {
            &self.best_insert_client_tw[r_v][cour]
        } else {
            &self.best_insert_client[r_v][cour]
        };
        let not_adjacent = |p: i32| p as usize != v && self.nodes[p as usize].next as usize != v;
        let mut position = best.best_location[0];
        let mut cost = best.best_cost[0];
        let mut found = not_adjacent(position);
        if !found && best.best_location[1] >= 0 {
            position = best.best_location[1];
            cost = best.best_cost[1];
            found = not_adjacent(position);
            if !found && best.best_location[2] >= 0 {
                position = best.best_location[2];
                cost = best.best_cost[2];
                found = true;
            }
        }
        let (v_prev, v_next) = (self.nodes[v].prev as usize, self.nodes[v].next as usize);
        let mut delta = self.cost(v_prev, u) + self.cost(u, v_next) - self.cost(v_prev, v_next);
        if with_tw {
            let tw = self.merge_all(&[
                &self.nodes[v_prev].prefix_tw_data,
                &self.nodes[u].tw_data,
                &self.nodes[v_next].postfix_tw_data,
            ]);
            delta += self.delta_penalty_time_windows(&tw, &self.routes[r_v].tw_data);
        }
        if !found || delta < cost {
            (delta, v_prev)
        } else {
            (cost, position as usize)
        }
    }

    /// Route time window data after replacing `removed` by `inserted` placed
    /// after `position`, all in the route of `removed`.
    fn swap_star_route_tw(&self, removed: usize, inserted: usize, position: usize) -> TimeWindowData {
        let (prev, next) = (self.nodes[removed].prev as usize, self.nodes[removed].next as usize);
        let position_next = self.nodes[position].next as usize;
        if self.nodes[position].position == self.nodes[removed].position - 1 {
            self.merge_all(&[
                &self.nodes[position].prefix_tw_data,
                &self.nodes[inserted].tw_data,
                &self.nodes[next].postfix_tw_data,
            ])
        } else if self.nodes[position].position < self.nodes[removed].position {
            let segment = self.get_route_segment_tw_data(position_next, prev);
            self.merge_all(&[
                &self.nodes[position].prefix_tw_data,
                &self.nodes[inserted].tw_data,
                &segment,
                &self.nodes[next].postfix_tw_data,
            ])
        } else {
            let segment = self.get_route_segment_tw_data(next, position);
            self.merge_all(&[
                &self.nodes[prev].prefix_tw_data,
                &segment,
                &self.nodes[inserted].tw_data,
                &self.nodes[position_next].postfix_tw_data,
            ])
        }
    }

    /// Distance change of removing `removed` and inserting `inserted` after
    /// `position` in the same route.
    fn swap_star_route_cost(&self, removed: usize, inserted: usize, position: usize) -> f64 {
        let (prev, next) = (self.nodes[removed].prev as usize, self.nodes[removed].next as usize);
        let mut cost = self.cost(position, inserted) - self.cost(prev, removed) - self.cost(removed, next);
        if position == prev {
            cost += self.cost(inserted, next);
        } else {
            let position_next = self.nodes[position].next as usize;
            cost += self.cost(inserted, position_next) + self.cost(prev, next) - self.cost(position, position_next);
        }
        cost as f64
    }

    /// SWAP* (Vidal 2022): exchange a client of `r_u` with one of `r_v`,
    /// each reinserted at its best position, if it improves the penalized cost.
    fn swap_star(&mut self, r_u: usize, r_v: usize, with_tw: bool) -> bool {
        for r in [r_u, r_v] {
            if !self.best_insert_initialized_for_route[r] {
                self.best_insert_initialized_for_route[r] = true;
                for option in self.best_insert_client[r].iter_mut().chain(self.best_insert_client_tw[r].iter_mut()) {
                    option.when_last_calculated = -1;
                }
            }
        }
        self.preprocess_insertions(r_u, r_v, with_tw);
        self.preprocess_insertions(r_v, r_u, with_tw);

        let (load_u, load_v) = (self.routes[r_u].load, self.routes[r_v].load);
        let current_load_pen = self.penalty_excess_load(load_u) + self.penalty_excess_load(load_v);
        let clients_v = self.route_clients(r_v);
        let mut best = SwapStarElement {
            move_cost: 1e30,
            load_pen_u: 1e30,
            load_pen_v: 1e30,
            u: 0,
            best_position_u: None,
            v: 0,
            best_position_v: None,
        };
        for u in self.route_clients(r_u) {
            let demand_u = self.params.cli[self.nodes[u].cour as usize].demand;
            for &v in &clients_v {
                let demand_v = self.params.cli[self.nodes[v].cour as usize].demand;
                let load_pen_u = self.penalty_excess_load(load_u + demand_v - demand_u);
                let load_pen_v = self.penalty_excess_load(load_v + demand_u - demand_v);
                let delta_load_pen = load_pen_u + load_pen_v - current_load_pen;
                let delta_removal = if with_tw {
                    self.nodes[u].delta_removal_tw + self.nodes[v].delta_removal_tw
                } else {
                    self.nodes[u].delta_removal + self.nodes[v].delta_removal
                };
                // Insertion costs are nonnegative bounds: skip hopeless pairs.
                if delta_load_pen + delta_removal as f64 > 0. {
                    continue;
                }
                let (extra_v, position_u) = self.cheapest_insert_simult_removal(u, v, with_tw);
                let (extra_u, position_v) = self.cheapest_insert_simult_removal(v, u, with_tw);
                let move_cost = delta_load_pen + (delta_removal + extra_u + extra_v) as f64;
                if move_cost < best.move_cost {
                    best = SwapStarElement {
                        move_cost,
                        load_pen_u,
                        load_pen_v,
                        u,
                        best_position_u: Some(position_u),
                        v,
                        best_position_v: Some(position_v),
                    };
                }
            }
        }
        let (Some(position_u), Some(position_v)) = (best.best_position_u, best.best_position_v) else {
            return false;
        };
        let (u, v) = (best.u, best.v);

        // Exact cost, including time warp.
        let route_u_tw = self.swap_star_route_tw(u, v, position_v);
        let route_v_tw = self.swap_star_route_tw(v, u, position_u);
        let cost_supp_u = self.swap_star_route_cost(u, v, position_v) + best.load_pen_u
            + self.penalty_time_windows(&route_u_tw)
            - self.routes[r_u].penalty;
        let cost_supp_v = self.swap_star_route_cost(v, u, position_u) + best.load_pen_v
            + self.penalty_time_windows(&route_v_tw)
            - self.routes[r_v].penalty;
        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        self.insert_node(u, position_u);
        self.insert_node(v, position_v);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        self.update_route_data(r_v);
        true
    }

    /// Initialize construction heuristic data.
    pub fn initialize_construction(&mut self, nodes_to_insert: &mut Vec<NodeToInsert>) {
        self.empty_routes.clear();
        let depot_tw = TimeWindowData {
            first_node_index: 0,
            last_node_index: 0,
            duration: 0.,
            time_warp: 0.,
            earliest_arrival: self.params.cli[0].earliest_arrival,
            latest_arrival: self.params.cli[0].latest_arrival,
            latest_release_time: self.params.cli[0].release_time,
        };

        for i in 1..=self.params.nb_clients as usize {
            self.nodes[i].cour = i as i32;
            self.nodes[i].is_depot = false;
            self.nodes[i].prev = 0;
            self.nodes[i].next = 0;
            self.nodes[i].route_idx = 0;
            self.nodes[i].tw_data = TimeWindowData {
                first_node_index: i as i32,
                last_node_index: i as i32,
                duration: self.params.cli[i].service_duration,
                time_warp: 0.,
                earliest_arrival: self.params.cli[i].earliest_arrival,
                latest_arrival: self.params.cli[i].latest_arrival,
                latest_release_time: self.params.cli[i].release_time,
            };
        }

        for r in 0..self.params.nb_vehicles as usize {
            let start = self.depot_start(r);
            let end = self.depot_end(r);

            self.nodes[start].cour = 0;
            self.nodes[start].is_depot = true;
            self.nodes[start].route_idx = r as i32;
            self.nodes[start].prev = end as i32;
            self.nodes[end].next = start as i32;
            self.nodes[start].next = end as i32;
            self.nodes[end].prev = start as i32;

            self.nodes[start].tw_data = depot_tw.clone();
            self.nodes[start].prefix_tw_data = depot_tw.clone();
            self.nodes[start].postfix_tw_data = depot_tw.clone();

            self.nodes[end].cour = 0;
            self.nodes[end].is_depot = true;
            self.nodes[end].route_idx = r as i32;
            self.nodes[end].tw_data = depot_tw.clone();
            self.nodes[end].prefix_tw_data = depot_tw.clone();
            self.nodes[end].postfix_tw_data = depot_tw.clone();

            self.update_route_data(r);
        }

        nodes_to_insert.clear();
        for i in 1..=self.params.nb_clients as usize {
            let dx = self.params.cli[i].coord_x - self.params.cli[0].coord_x;
            let dy = self.params.cli[i].coord_y - self.params.cli[0].coord_y;
            nodes_to_insert.push(NodeToInsert {
                client_idx: i as i32,
                load: self.params.cli[i].demand,
                service_duration: self.params.cli[i].service_duration,
                angle_from_depot: dy.atan2(dx),
                tw_data: self.nodes[i].tw_data.clone(),
            });
        }
    }

    /// Construct individual by sweep heuristic (Vidal 2012).
    pub fn construct_individual_by_sweep(&mut self, fill_percentage: i32, indiv: &mut Individual<'_>) {
        let mut nodes_to_insert = Vec::new();
        self.initialize_construction(&mut nodes_to_insert);

        nodes_to_insert.sort_by(|a, b| {
            a.angle_from_depot
                .partial_cmp(&b.angle_from_depot)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut node_indices_per_route: Vec<Vec<usize>> = Vec::new();
        let mut current_route_indices = Vec::new();
        let mut current_load = 0;
        let cap_threshold = (fill_percentage as f64 * self.params.vehicle_capacity as f64 / 100.0) as i32;

        for (i, node) in nodes_to_insert.iter().enumerate() {
            if current_load > 0
                && current_load + node.load > cap_threshold
                && node_indices_per_route.len() + 1 < self.routes.len()
            {
                node_indices_per_route.push(current_route_indices);
                current_route_indices = Vec::new();
                current_load = 0;
            }
            current_load += node.load;
            current_route_indices.push(i);
        }
        node_indices_per_route.push(current_route_indices);

        for (r, indices) in node_indices_per_route.iter().enumerate() {
            if r >= self.routes.len() {
                break;
            }
            let depot_s = self.depot_start(r);
            let depot_opening = self.nodes[depot_s].tw_data.latest_arrival
                - self.nodes[depot_s].tw_data.earliest_arrival;

            let mut short_tw_indices = Vec::new();
            let mut long_tw_indices = Vec::new();

            for &idx in indices {
                let tw_dur = nodes_to_insert[idx].tw_data.latest_arrival
                    - nodes_to_insert[idx].tw_data.earliest_arrival;
                if tw_dur * 2. > depot_opening {
                    long_tw_indices.push(idx);
                } else {
                    short_tw_indices.push(idx);
                }
            }

            short_tw_indices.sort_by(|&a, &b| nodes_to_insert[a].tw_data.latest_arrival.total_cmp(&nodes_to_insert[b].tw_data.latest_arrival));

            let mut prev = depot_s;
            for idx in short_tw_indices {
                let to_insert = nodes_to_insert[idx].client_idx as usize;
                self.insert_unlinked_node(to_insert, prev);
                prev = to_insert;
            }
            self.update_route_data(r);

            for idx in long_tw_indices {
                let mut best_cost = f64::MAX;
                let mut best_pred = depot_s;
                let mut curr = depot_s;

                for _ in 0..=self.routes[r].nb_customers {
                    let next = self.nodes[curr].next as usize;
                    let curr_cour = self.nodes[curr].cour as usize;
                    let next_cour = self.nodes[next].cour as usize;
                    let cand_cour = nodes_to_insert[idx].client_idx as usize;

                    let cost = (self.params.time_cost.get(curr_cour, cand_cour)
                        + self.params.time_cost.get(cand_cour, next_cour)
                        - self.params.time_cost.get(curr_cour, next_cour)) as f64;

                    if cost < best_cost {
                        best_cost = cost;
                        best_pred = curr;
                    }
                    curr = next;
                }

                let to_insert = nodes_to_insert[idx].client_idx as usize;
                self.insert_unlinked_node(to_insert, best_pred);
                self.update_route_data(r);
            }
        }

        self.export_individual(indiv);
    }

    /// Construct individual with seed order (Furthest or Nearest seed).
    pub fn construct_individual_with_seed_order(
        &mut self,
        tolerated_capacity_violation: i32,
        tolerated_time_warp: i32,
        use_seed_client_furthest_from_depot: bool,
        indiv: &mut Individual<'_>,
    ) {
        let mut nodes_to_insert = Vec::new();
        self.initialize_construction(&mut nodes_to_insert);

        let mut unassigned: std::collections::BTreeSet<usize> =
            (0..self.params.nb_clients as usize).collect();

        for r in 0..self.routes.len() {
            if unassigned.is_empty() {
                break;
            }
            let depot_s = self.depot_start(r);

            if use_seed_client_furthest_from_depot {
                let mut furthest_idx = None;
                let mut max_insertion_cost = -1.0;

                for &idx in &unassigned {
                    let cand_client = nodes_to_insert[idx].client_idx as usize;
                    let cost = (self.params.time_cost.get(0, cand_client)
                        + self.params.time_cost.get(cand_client, 0)
                        - self.params.time_cost.get(0, 0)) as f64;
                    if cost > max_insertion_cost {
                        max_insertion_cost = cost;
                        furthest_idx = Some(idx);
                    }
                }

                if let Some(f_idx) = furthest_idx {
                    let seed_node = nodes_to_insert[f_idx].client_idx as usize;
                    self.insert_unlinked_node(seed_node, depot_s);
                    self.update_route_data(r);
                    unassigned.remove(&f_idx);
                }
            }

            let mut node_inserted = true;
            while node_inserted {
                node_inserted = false;
                let mut best_insertion_cost = f64::MAX;
                let mut best_pred = depot_s;
                let mut best_node_idx = None;

                for &idx in &unassigned {
                    if self.routes[r].load + nodes_to_insert[idx].load
                        > self.params.vehicle_capacity + tolerated_capacity_violation
                    {
                        continue;
                    }

                    let mut prev = depot_s;
                    for _ in 0..=self.routes[r].nb_customers {
                        let next = self.nodes[prev].next as usize;
                        let cand = nodes_to_insert[idx].client_idx as usize;

                        let merged_prefix_cand = TimeWindowData::merge(
                            &self.nodes[prev].prefix_tw_data,
                            &nodes_to_insert[idx].tw_data,
                            self.params,
                        );
                        let merged_all = TimeWindowData::merge(
                            &merged_prefix_cand,
                            &self.nodes[next].postfix_tw_data,
                            self.params,
                        );

                        if merged_all.time_warp <= f64::from(tolerated_time_warp) {
                            let prev_cour = self.nodes[prev].cour as usize;
                            let next_cour = self.nodes[next].cour as usize;
                            let cost = (self.params.time_cost.get(prev_cour, cand)
                                + self.params.time_cost.get(cand, next_cour)
                                - self.params.time_cost.get(prev_cour, next_cour)) as f64;

                            if cost < best_insertion_cost {
                                best_insertion_cost = cost;
                                best_pred = prev;
                                best_node_idx = Some(idx);
                            }
                        }
                        prev = next;
                    }
                }

                if let Some(best_idx) = best_node_idx {
                    let node_to_ins = nodes_to_insert[best_idx].client_idx as usize;
                    self.insert_unlinked_node(node_to_ins, best_pred);
                    self.update_route_data(r);
                    unassigned.remove(&best_idx);
                    node_inserted = true;
                }
            }
        }

        if !unassigned.is_empty() {
            let last_r = self.routes.len() - 1;
            let depot_e = self.depot_end(last_r);
            // HGS inserts each one right after the same node, reversing them.
            let prev = self.nodes[depot_e].prev as usize;
            for idx in unassigned {
                let to_ins = nodes_to_insert[idx].client_idx as usize;
                self.insert_unlinked_node(to_ins, prev);
            }
            self.update_route_data(last_r);
        }

        self.export_individual(indiv);
    }
}
