//! Individual representation for HGS population.
//!
//! An individual encodes a complete VRP solution with:
//! - A **giant tour** (`chrom_t`) — permutation of all customers
//! - A set of **routes** (`chrom_r`) — feasible vehicle tours derived from the giant tour
//! - Predecessor/successor maps for fast arc queries
//! - Cost, feasibility, and fitness information

use crate::hgs::params::MY_EPSILON;

use std::collections::BTreeSet;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use crate::hgs::{Params, ProblemType, TimeWindow};

static NEXT_INDIVIDUAL_ID: AtomicUsize = AtomicUsize::new(1);

fn next_individual_id() -> usize {
    NEXT_INDIVIDUAL_ID.fetch_add(1, AtomicOrdering::Relaxed)
}

/// Wrapper for f64 that implements Ord by comparing bit patterns.
/// This allows using f64 as a BTreeMap key. Note: NaN and negative zero
/// have special behavior; we assume valid positive distances only.
#[derive(Clone, Copy, Debug)]
pub struct SortableF64(pub f64);


impl PartialEq for SortableF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}

impl Eq for SortableF64 {}

impl PartialOrd for SortableF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SortableF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        // Raw bit patterns order negative values above positive ones.
        self.0.total_cmp(&other.0)
    }
}

/// Cost and penalty breakdown for a solution.
#[derive(Clone, Debug, Default)]
pub struct CostSol {
    /// Penalized cost = distance + capacity_penalty + time_warp_penalty + wait_penalty
    pub penalized_cost: f64,
    /// Number of active routes
    pub nb_routes: usize,
    /// Total travel distance across all routes
    pub distance: f64,
    /// Total capacity excess across all routes
    pub capacity_excess: i32,
    /// Total wait time across all routes
    pub wait_time: f64,
    /// Total time warp across all routes
    pub time_warp: f64,
    /// Whether the solution is feasible (no capacity or time-warp violations)
    pub is_feasible: bool,
}

/// A solution individual in the HGS population.
pub struct Individual<'p> {
    /// Unique identifier for this individual.
    pub id: usize,
    /// Reference to problem parameters.
    pub params: &'p Params,
    /// Number of vehicles available.
    pub nb_vehicles: usize,
    /// Solution cost breakdown.
    pub my_cost_sol: CostSol,
    /// Giant tour — permutation of all customers (1-based indices into cli).
    pub chrom_t: Vec<usize>,
    /// Routes per vehicle. `chrom_r[r]` is the ordered list of customers on route r.
    pub chrom_r: Vec<Vec<usize>>,
    /// `successors[i]` = customer after i in the giant tour (0 = depot).
    pub successors: Vec<usize>,
    /// `predecessors[i]` = customer before i in the giant tour (0 = depot).
    pub predecessors: Vec<usize>,
    /// Proximity index: `(SortableF64, individual_id)` sorted by distance.
    /// Used for diversity management.
    pub indivs_per_proximity: BTreeSet<(SortableF64, usize)>,
    /// Feasibility flag.
    pub is_feasible: bool,
    /// Biased fitness value (penalized cost adjusted by frequency).
    pub biased_fitness: f64,
}

impl<'p> Clone for Individual<'p> {
    fn clone(&self) -> Self {
        Self {
            id: next_individual_id(),
            params: self.params,
            nb_vehicles: self.nb_vehicles,
            my_cost_sol: self.my_cost_sol.clone(),
            chrom_t: self.chrom_t.clone(),
            chrom_r: self.chrom_r.clone(),
            successors: self.successors.clone(),
            predecessors: self.predecessors.clone(),
            indivs_per_proximity: BTreeSet::new(),
            is_feasible: self.is_feasible,
            biased_fitness: self.biased_fitness,
        }
    }
}

impl<'p> Individual<'p> {
    // ── Construction ──────────────────────────────────────────────────────

    /// Clone an individual, copying over map fields (successors, predecessors, proximity).
    pub fn clone_with_maps(indiv: &Self) -> Self {
        indiv.clone()
    }

    /// Create an empty individual (no routes, no clients scheduled).
    pub fn new_empty(params: &'p Params) -> Self {
        let chrom_r = vec![Vec::new(); params.nb_vehicles as usize];
        let chrom_t = Vec::new();
        let nb_clients = params.nb_clients as usize;
        let successors = vec![0; nb_clients + 1];
        let predecessors = vec![0; nb_clients + 1];
        let indivs_per_proximity = BTreeSet::new();
        let mut my_cost_sol = CostSol::default();
        my_cost_sol.penalized_cost = 1e30;
        Self {
            id: next_individual_id(),
            params,
            nb_vehicles: params.nb_vehicles as usize,
            my_cost_sol,
            chrom_t,
            chrom_r,
            successors,
            predecessors,
            indivs_per_proximity,
            is_feasible: false,
            biased_fitness: 0.0,
        }
    }

    /// Create a random individual with a shuffled giant tour.
    pub fn new_random(params: &'p Params) -> Self {
        let nb_clients = params.nb_clients as usize;
        let mut chrom_t: Vec<usize> = (1..=nb_clients).collect();
        params.rng.borrow_mut().shuffle(&mut chrom_t);

        let mut indiv = Self::from_giant_tour(params, &chrom_t);
        // Not evaluated until Split assigns routes.
        indiv.my_cost_sol.penalized_cost = 1e30;
        indiv
    }

    /// Create an individual from a string like `"1 0 3 2 5 0 4"`.
    pub fn from_string(params: &'p Params, solution_str: &str) -> Self {
        let mut chrom_r = vec![Vec::new(); params.nb_vehicles as usize];
        let mut chrom_t = Vec::new();

        let mut route = 0;
        for token in solution_str.split_whitespace() {
            if let Ok(customer) = token.parse::<usize>() {
                if customer == 0 {
                    route += 1;
                } else {
                    if route < chrom_r.len() {
                        chrom_r[route].push(customer);
                    }
                    chrom_t.push(customer);
                }
            }
        }

        let mut indiv = Self::from_components(params, &chrom_t, &chrom_r);
        indiv.evaluate_complete_cost();
        indiv
    }

    /// Create an individual using the Regret-Capacitated-Walk-Start (RCWS) heuristic.
    pub fn from_rcws(params: &'p Params, pattern: Option<&[Vec<usize>]>) -> Self {
        let nb_clients = params.nb_clients as usize;
        let mut chrom_r: Vec<Vec<usize>> = Vec::new();
        let mut load = Vec::new();
        let mut in_route = vec![false; nb_clients + 1];
        let mut interior = vec![false; nb_clients + 1];
        let mut customer_route = vec![isize::MAX as isize; nb_clients + 1];

        let mut next_empty_route = 0;

        // Insert routes from pattern if provided.
        if let Some(pattern) = pattern {
            for route_customers in pattern {
                while next_empty_route >= chrom_r.len() {
                    chrom_r.push(Vec::new());
                    load.push(0);
                }
                for &customer in route_customers {
                    chrom_r[next_empty_route].push(customer);
                    load[next_empty_route] += params.cli[customer].demand;
                    in_route[customer] = true;
                    customer_route[customer] = next_empty_route as isize;
                    if route_customers.len() > 2 && customer > 0 {
                        let idx = route_customers.iter().position(|&c| c == customer);
                        if let Some(i) = idx {
                            if i > 0 && i < route_customers.len() - 1 {
                                interior[customer] = true;
                            }
                        }
                    }
                }
                while next_empty_route < chrom_r.len() && !chrom_r[next_empty_route].is_empty() {
                    next_empty_route += 1;
                }
            }
        }

        // Process savings list to merge routes.
        for savings in &params.savings_list {
            if savings.value <= 0.0 {
                continue;
            }
            let c1 = savings.c1 as usize;
            let c2 = savings.c2 as usize;

            if params.cli[c1].demand + params.cli[c2].demand > params.vehicle_capacity {
                continue;
            }

            match (in_route[c1], in_route[c2]) {
                (false, false) => {
                    // Neither customer is in a route — create new route.
                    while next_empty_route >= chrom_r.len() {
                        chrom_r.push(Vec::new());
                        load.push(0);
                    }
                    chrom_r[next_empty_route].push(c1);
                    chrom_r[next_empty_route].push(c2);
                    load[next_empty_route] = params.cli[c1].demand + params.cli[c2].demand;
                    in_route[c1] = true;
                    in_route[c2] = true;
                    customer_route[c1] = next_empty_route as isize;
                    customer_route[c2] = next_empty_route as isize;
                    while next_empty_route < chrom_r.len() && !chrom_r[next_empty_route].is_empty() {
                        next_empty_route += 1;
                    }
                }
                (true, false) if !interior[c1] => {
                    let route_idx = customer_route[c1] as usize;
                    if route_idx < chrom_r.len() && !chrom_r[route_idx].is_empty() {
                        if chrom_r[route_idx][0] == c1
                            && load[route_idx] + params.cli[c2].demand <= params.vehicle_capacity
                        {
                            chrom_r[route_idx].insert(0, c2);
                            load[route_idx] += params.cli[c2].demand;
                            in_route[c2] = true;
                            customer_route[c2] = route_idx as isize;
                            if chrom_r[route_idx].len() > 2 {
                                interior[c1] = true;
                            }
                        } else if chrom_r[route_idx].last() == Some(&c1)
                            && load[route_idx] + params.cli[c2].demand <= params.vehicle_capacity
                        {
                            chrom_r[route_idx].push(c2);
                            load[route_idx] += params.cli[c2].demand;
                            in_route[c2] = true;
                            customer_route[c2] = route_idx as isize;
                            if chrom_r[route_idx].len() > 2 {
                                interior[c1] = true;
                            }
                        }
                    }
                }
                (false, true) if !interior[c2] => {
                    let route_idx = customer_route[c2] as usize;
                    if route_idx < chrom_r.len() && !chrom_r[route_idx].is_empty() {
                        if chrom_r[route_idx][0] == c2
                            && load[route_idx] + params.cli[c1].demand <= params.vehicle_capacity
                        {
                            chrom_r[route_idx].insert(0, c1);
                            load[route_idx] += params.cli[c1].demand;
                            in_route[c1] = true;
                            customer_route[c1] = route_idx as isize;
                            if chrom_r[route_idx].len() > 2 {
                                interior[c2] = true;
                            }
                        } else if chrom_r[route_idx].last() == Some(&c2)
                            && load[route_idx] + params.cli[c1].demand <= params.vehicle_capacity
                        {
                            chrom_r[route_idx].push(c1);
                            load[route_idx] += params.cli[c1].demand;
                            in_route[c1] = true;
                            customer_route[c1] = route_idx as isize;
                            if chrom_r[route_idx].len() > 2 {
                                interior[c2] = true;
                            }
                        }
                    }
                }
                (true, true) if !interior[c1] && !interior[c2] => {
                    let r1 = customer_route[c1] as usize;
                    let r2 = customer_route[c2] as usize;
                    if r1 != r2 && load[r1] + load[r2] <= params.vehicle_capacity {
                        // Merge routes based on endpoint matching.
                        let front1 = chrom_r[r1].first() == Some(&c1);
                        let back1 = chrom_r[r1].last() == Some(&c1);
                        let front2 = chrom_r[r2].first() == Some(&c2);
                        let back2 = chrom_r[r2].last() == Some(&c2);

                        if front1 && front2 {
                            let mut merged = chrom_r[r2].clone();
                            merged.reverse();
                            merged.extend_from_slice(&chrom_r[r1]);
                            chrom_r[r1] = merged;
                        } else if back1 && front2 {
                            let r2_clone = chrom_r[r2].clone();
                            chrom_r[r1].extend(&r2_clone);
                        } else if front1 && back2 {
                            let mut merged = chrom_r[r2].clone();
                            merged.extend_from_slice(&chrom_r[r1]);
                            chrom_r[r1] = merged;
                        } else if back1 && back2 {
                            let mut merged = chrom_r[r2].clone();
                            merged.reverse();
                            chrom_r[r1].extend(&merged);
                        }

                        load[r1] += load[r2];
                        for &cust in &chrom_r[r2] {
                            customer_route[cust] = r1 as isize;
                        }
                        chrom_r[r2].clear();
                        load[r2] = 0;
                        interior[c1] = true;
                        interior[c2] = true;
                    }
                }
                _ => {}
            }
        }

        // Assign remaining unassigned customers.
        for i in 1..=nb_clients {
            if !in_route[i] {
                let mut best_route: Option<usize> = None;
                let mut best_cost = f64::INFINITY;

                for (r, route) in chrom_r.iter().enumerate() {
                    if !route.is_empty() {
                        let last = *route.last().unwrap();
                        let cost = params.time_cost.get(last, i);
                        if cost < best_cost && load[r] + params.cli[i].demand <= params.vehicle_capacity {
                            best_cost = cost;
                            best_route = Some(r);
                        }
                    }
                }

                if let Some(best) = best_route {
                    chrom_r[best].push(i);
                    load[best] += params.cli[i].demand;
                    in_route[i] = true;
                    customer_route[i] = best as isize;
                }
            }
        }

        // Flatten routes into giant tour.
        let mut chrom_t = Vec::with_capacity(nb_clients);
        for route in &chrom_r {
            chrom_t.extend(route);
        }

        let mut indiv = Self::from_components(params, &chrom_t, &chrom_r);
        indiv.evaluate_complete_cost();
        indiv
    }

    // ── Internal constructors ─────────────────────────────────────────────

    fn from_giant_tour(params: &'p Params, chrom_t: &[usize]) -> Self {
        let nb_clients = params.nb_clients as usize;
        let nb_vehicles = params.nb_vehicles as usize;

        Individual {
            id: next_individual_id(),
            params,
            nb_vehicles,
            my_cost_sol: CostSol::default(),
            chrom_t: chrom_t.to_vec(),
            chrom_r: vec![Vec::new(); nb_vehicles],
            successors: vec![0; nb_clients + 1],
            predecessors: vec![0; nb_clients + 1],
            indivs_per_proximity: BTreeSet::new(),
            is_feasible: false,
            biased_fitness: 0.0,
        }
    }

    fn from_components(params: &'p Params, chrom_t: &[usize], chrom_r: &[Vec<usize>]) -> Self {
        let nb_clients = params.nb_clients as usize;
        let nb_vehicles = params.nb_vehicles as usize;

        let mut indiv = Individual {
            id: next_individual_id(),
            params,
            nb_vehicles,
            my_cost_sol: CostSol::default(),
            chrom_t: chrom_t.to_vec(),
            chrom_r: chrom_r.to_vec(),
            successors: vec![0; nb_clients + 1],
            predecessors: vec![0; nb_clients + 1],
            indivs_per_proximity: BTreeSet::new(),
            is_feasible: false,
            biased_fitness: 0.0,
        };
        indiv.rebuild_maps();
        indiv
    }

    /// Rebuild predecessor/successor maps from chrom_r.
    pub fn rebuild_maps(&mut self) {
        let nb_clients = self.params.nb_clients as usize;
        self.successors = vec![0; nb_clients + 1];
        self.predecessors = vec![0; nb_clients + 1];

        for r in 0..self.nb_vehicles {
            if r >= self.chrom_r.len() {
                continue;
            }
            let route = &self.chrom_r[r];
            if route.is_empty() {
                continue;
            }

            // Depot (0) -> first customer.
            self.successors[0] = route[0];
            self.predecessors[route[0]] = 0;

            // Between customers in the route.
            for i in 1..route.len() {
                self.successors[route[i - 1]] = route[i];
                self.predecessors[route[i]] = route[i - 1];
            }

            // Last customer -> depot (0).
            self.successors[route[route.len() - 1]] = 0;
            self.predecessors[0] = route[route.len() - 1];
        }
    }

    // ── Cost evaluation ───────────────────────────────────────────────────

    /// Evaluate the complete cost of the solution from chrom_r.
    pub fn evaluate_complete_cost(&mut self) {
        let mut cost_sol = CostSol::default();
        let nb_vehicles = self.nb_vehicles.min(self.chrom_r.len());

        // Precompute time-cost cache for each route.
        let mut time_cost_cache: Vec<Vec<f64>> = Vec::with_capacity(nb_vehicles);
        for r in 0..nb_vehicles {
            if self.chrom_r[r].is_empty() {
                time_cost_cache.push(Vec::new());
                continue;
            }
            let mut cache = vec![0.; self.chrom_r[r].len()];
            cache[0] = self.params.time_cost.get(0, self.chrom_r[r][0]);
            for i in 1..self.chrom_r[r].len() {
                cache[i] = self.params.time_cost.get(self.chrom_r[r][i - 1], self.chrom_r[r][i]);
            }
            time_cost_cache.push(cache);
        }

        // Evaluate each route.
        for r in 0..nb_vehicles {
            let route = &self.chrom_r[r];
            if route.is_empty() {
                continue;
            }

            // Find latest release time in route.
            let mut latest_release_time = self.params.cli[route[0]].release_time;
            for &cust in &route[1..] {
                if self.params.cli[cust].release_time > latest_release_time {
                    latest_release_time = self.params.cli[cust].release_time;
                }
            }

            let mut distance = time_cost_cache[r][0];
            let mut load = self.params.cli[route[0]].demand;
            let mut _service = self.params.cli[route[0]].service_duration;
            let mut time = latest_release_time + distance;
            let mut wait_time = 0_f64;
            let mut time_warp = 0_f64;

            // Handle time windows for first customer.
            if self.params.cli[route[0]].n_tw > 0 {
                let tws: Vec<TimeWindow> = self.params.cli[route[0]].time_windows.clone();
                time = Self::adjust_to_time_window(time, &tws, &mut wait_time, &mut time_warp);
            } else {
                if time < self.params.cli[route[0]].earliest_arrival {
                    time = self.params.cli[route[0]].earliest_arrival;
                } else if time > self.params.cli[route[0]].latest_arrival {
                    time_warp += time - self.params.cli[route[0]].latest_arrival;
                    time = self.params.cli[route[0]].latest_arrival;
                }
            }

            self.predecessors[route[0]] = 0;

            // Evaluate remaining customers.
            for i in 1..route.len() {
                let prev = route[i - 1];
                let curr = route[i];
                distance += time_cost_cache[r][i];
                load += self.params.cli[curr].demand;
                _service += self.params.cli[curr].service_duration;
                time += self.params.cli[prev].service_duration + time_cost_cache[r][i];

                if self.params.cli[curr].n_tw > 0 {
                    let tws: Vec<TimeWindow> = self.params.cli[curr].time_windows.clone();
                    time = Self::adjust_to_time_window(time, &tws, &mut wait_time, &mut time_warp);
                } else {
                    if time < self.params.cli[curr].earliest_arrival {
                        wait_time += self.params.cli[curr].earliest_arrival - time;
                        time = self.params.cli[curr].earliest_arrival;
                    } else if time > self.params.cli[curr].latest_arrival {
                        time_warp += time - self.params.cli[curr].latest_arrival;
                        time = self.params.cli[curr].latest_arrival;
                    }
                }

                self.predecessors[curr] = prev;
                self.successors[prev] = curr;
            }

            // Finalize route — return to depot.
            let last = *route.last().unwrap();
            self.successors[last] = 0;
            distance += self.params.time_cost.get(last, 0);
            time += self.params.cli[last].service_duration + self.params.time_cost.get(last, 0);

            // Handle depot time window.
            if time > self.params.cli[0].latest_arrival {
                time_warp += time - self.params.cli[0].latest_arrival;
            }

            // Update solution metrics.
            cost_sol.distance += distance;
            if matches!(self.params.problem_type, ProblemType::Vrptw | ProblemType::Evrp) {
                cost_sol.time_warp += time_warp;
                cost_sol.wait_time += wait_time;
            }
            cost_sol.nb_routes += 1;

            if load > self.params.vehicle_capacity {
                cost_sol.capacity_excess += load - self.params.vehicle_capacity;
            }
        }

        // Calculate penalized cost.
        cost_sol.penalized_cost = cost_sol.distance as f64
            + cost_sol.capacity_excess as f64 * self.params.penalty_capacity.get()
            + cost_sol.time_warp as f64 * self.params.penalty_time_warp.get()
            + cost_sol.wait_time as f64 * self.params.penalty_wait_time;

        cost_sol.is_feasible = cost_sol.capacity_excess == 0 && cost_sol.time_warp <= MY_EPSILON;

        let is_feasible = cost_sol.is_feasible;
        self.my_cost_sol = cost_sol;
        self.is_feasible = is_feasible;
    }

    /// Adjust time based on multiple time windows.
    fn adjust_to_time_window(
        time: f64,
        time_windows: &[TimeWindow],
        wait_time: &mut f64,
        time_warp: &mut f64,
    ) -> f64 {
        for window in time_windows {
            if time < window.earliest {
                *wait_time += window.earliest - time;
                return window.earliest;
            } else if time <= window.latest {
                return time;
            }
        }
        // No valid window — apply time warp to the latest one.
        let last = time_windows.last().unwrap();
        *time_warp += time - last.latest;
        last.latest
    }

    // ── Giant tour operations ─────────────────────────────────────────────

    /// Shuffle the giant tour randomly.
    pub fn shuffle_chrom_t(&mut self) {
        self.chrom_t = (1..=self.params.nb_clients as usize).collect();
        self.params.rng.borrow_mut().shuffle(&mut self.chrom_t);
    }

    // ── Proximity / diversity ─────────────────────────────────────────────

    /// Add a reference to another individual in the proximity structure.
    pub fn add_proximity(&mut self, distance: f64, indiv_id: usize) {
        self.indivs_per_proximity.insert((SortableF64(distance), indiv_id));
    }

    /// Remove an individual from the proximity structure by ID.
    pub fn remove_proximity(&mut self, indiv_id: usize) {
        self.indivs_per_proximity.retain(|&(_, id)| id != indiv_id);
    }

    /// Broken pairs distance between this individual and indiv2.
    /// Measures the fraction of arcs that differ (direct or reverse).
    pub fn broken_pairs_distance(&self, indiv2: &Self) -> f64 {
        let nb_clients = self.params.nb_clients as usize;
        let mut differences = 0usize;

        for j in 1..=nb_clients {
            // Check if successor differs.
            if self.successors[j] != indiv2.successors[j]
                && self.successors[j] != indiv2.predecessors[j]
            {
                differences += 1;
            }
            // Check arc entering j from predecessor vs indiv2's arcs.
            if self.predecessors[j] == 0
                && indiv2.predecessors[j] != 0
                && indiv2.successors[j] != 0
            {
                differences += 1;
            }
        }

        differences as f64 / nb_clients as f64
    }

    /// Average broken pairs distance to the `nb_closest` individuals.
    pub fn average_broken_pairs_distance_closest(&self, nb_closest: usize) -> f64 {
        let size = self.indivs_per_proximity.len().min(nb_closest);
        if size == 0 {
            return 0.0;
        }

        let sum: f64 = self
            .indivs_per_proximity
            .iter()
            .take(size)
            .map(|(d, _)| d.0)
            .sum();

        sum / size as f64
    }

    // ── I/O ───────────────────────────────────────────────────────────────

    /// Export the solution in CVRPLib format to a string.
    pub fn export_cvrplib_string(&self) -> String {
        let mut output = String::new();
        for (r, route) in self.chrom_r.iter().enumerate() {
            if !route.is_empty() {
                output.push_str(&format!("Route {}:\n", r + 1));
                output.push_str("0 ");
                for &cust in route {
                    output.push_str(&format!("{} ", cust));
                }
                output.push_str("0\n");
            }
        }
        output
    }

    /// Print the solution in CVRPLib format.
    pub fn print_cvrplib_format(&self) {
        println!("{}", self.export_cvrplib_string());
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Returns true if the given route is time-window feasible.
    pub fn is_time_window_feasible(&self, route: &[usize]) -> bool {
        let mut arrival_time = self.params.cli[0].earliest_arrival as f64;

        for &current_customer in route {
            let travel_time = self.params.time_cost.get(0, current_customer) as f64;
            arrival_time += travel_time;

            if arrival_time > self.params.cli[current_customer].latest_arrival as f64 {
                return false;
            }

            if arrival_time < self.params.cli[current_customer].earliest_arrival as f64 {
                arrival_time = self.params.cli[current_customer].earliest_arrival as f64;
            }

            arrival_time += self.params.cli[current_customer].service_duration as f64;
        }

        true
    }

    /// Update arrival times for a route.
    pub fn update_arrival_times(&self, route: &[usize]) -> Vec<f64> {
        let mut arrival_times = Vec::with_capacity(route.len());
        let mut arrival_time = self.params.cli[0].earliest_arrival as f64;

        for (i, &current_customer) in route.iter().enumerate() {
            let travel_time = if i == 0 {
                self.params.time_cost.get(0, current_customer) as f64
            } else {
                self.params.time_cost.get(route[i - 1], current_customer) as f64
            };
            arrival_time += travel_time;

            if arrival_time < self.params.cli[current_customer].earliest_arrival as f64 {
                arrival_time = self.params.cli[current_customer].earliest_arrival as f64;
            }

            arrival_times.push(arrival_time);
            arrival_time += self.params.cli[current_customer].service_duration as f64;
        }

        arrival_times
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::Instance;
    use crate::inputs::Customer;

    fn make_test_instance() -> Instance {
        Instance {
            name: "test-5".to_string(),
            vehicles: 3,
            capacity: 100.0,
            customers: vec![
                Customer { x: 0.0, y: 0.0, demand: 0.0, ready: 0.0, due: i32::MAX as f64, service: 0.0 },
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
    fn test_random_individual() {
        let instance = make_test_instance();
        let params = Params::from_instance(&instance);
        let indiv = Individual::new_random(&params);

        assert_eq!(indiv.chrom_t.len(), 4); // 4 customers (depot excluded)
        assert!(indiv.chrom_t.iter().all(|&c| c >= 1 && c <= 4));
        assert_eq!(indiv.nb_vehicles, 3);
    }

    #[test]
    fn test_from_string() {
        let instance = make_test_instance();
        let params = Params::from_instance(&instance);
        let indiv = Individual::from_string(&params, "1 0 2 3 4 0");

        assert_eq!(indiv.chrom_r[0], vec![1]);
        assert_eq!(indiv.chrom_r[1], vec![2, 3, 4]);
        assert_eq!(indiv.chrom_t, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_evaluate_cost() {
        let instance = make_test_instance();
        let params = Params::from_instance(&instance);
        let indiv = Individual::from_string(&params, "1 0 2 3 4 0");

        assert!(indiv.my_cost_sol.nb_routes >= 1);
        assert!(indiv.my_cost_sol.distance > 0.);
    }

    #[test]
    fn test_broken_pairs_distance() {
        let instance = make_test_instance();
        let params = Params::from_instance(&instance);

        let mut indiv1 = Individual::new_random(&params);
        indiv1.evaluate_complete_cost();

        let mut indiv2 = Individual::new_random(&params);
        indiv2.evaluate_complete_cost();

        // Two different random individuals should have some broken pairs.
        let dist = indiv1.broken_pairs_distance(&indiv2);
        assert!(dist >= 0.0 && dist <= 2.0);
    }

    #[test]
    fn test_shuffled_giant_tour_is_permutation() {
        let instance = make_test_instance();
        let params = Params::from_instance(&instance);
        let mut indiv = Individual::new_random(&params);

        indiv.shuffle_chrom_t();

        let mut sorted = indiv.chrom_t.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4]);
    }
}
