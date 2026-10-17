//! Linear split algorithm for converting giant tours to feasible routes.
//!
//! The Split algorithm solves the following problem: given a fixed permutation
//! of customers (a "giant tour"), partition it into the minimum-cost set of
//! vehicle routes subject to capacity and time-window constraints.
//!
//! This is the core decomposition routine used by HGS-CVRP to evaluate
//! individuals represented as giant tours.

use crate::hgs::{Individual, Params};

/// Data for a single client during split processing.
#[derive(Clone, Debug, Default)]
struct ClientSplit {
    /// Demand of the client.
    demand: i32,
    /// Service duration at this client.
    service_time: i32,
    /// Distance from depot to this client.
    d0_x: i32,
    /// Distance from this client back to depot.
    dx_0: i32,
    /// Distance from this client to the next in the giant tour.
    dnext: i32,
    /// Original customer number (1-based index).
    cust_num: usize,
}

/// A lightweight deque implementation for the O(n) split algorithm.
/// Matches the C++ Trivial_Deque structure.
struct TrivialDeque {
    /// Storage for deque elements.
    my_deque: Vec<i32>,
    /// Index of the front element.
    index_front: usize,
    /// Index of the back element (inclusive).
    index_back: usize,
}

impl TrivialDeque {
    /// Create a new deque with space for `nb_elements`, initialized with `first_node`.
    fn new(nb_elements: usize, first_node: i32) -> Self {
        let mut my_deque = Vec::with_capacity(nb_elements);
        my_deque.push(first_node);
        TrivialDeque {
            my_deque,
            index_front: 0,
            index_back: 0,
        }
    }

    /// Remove the front element.
    fn pop_front(&mut self) {
        self.index_front += 1;
    }

    /// Remove the back element.
    fn pop_back(&mut self) {
        self.index_back -= 1;
    }

    /// Append an element to the back.
    fn push_back(&mut self, i: i32) {
        let cap = self.my_deque.len();
        if self.index_back + 1 >= cap {
            self.my_deque.resize(cap * 2, 0);
        }
        self.index_back += 1;
        if self.index_back >= self.my_deque.len() {
            self.my_deque.resize(self.index_back + 1, 0);
        }
        self.my_deque[self.index_back] = i;
    }

    /// Get the front element.
    fn get_front(&self) -> i32 {
        self.my_deque[self.index_front]
    }

    /// Get the next-front element.
    fn get_next_front(&self) -> i32 {
        self.my_deque[self.index_front + 1]
    }

    /// Get the back element.
    fn get_back(&self) -> i32 {
        self.my_deque[self.index_back]
    }

    /// Reset the deque with a single first node.
    fn reset(&mut self, first_node: i32) {
        if self.my_deque.is_empty() {
            self.my_deque.push(first_node);
        } else {
            self.my_deque[0] = first_node;
        }
        self.index_back = 0;
        self.index_front = 0;
    }

    /// Current size of the deque.
    fn size(&self) -> usize {
        (self.index_back as isize - self.index_front as isize + 1).max(0) as usize
    }
}

/// Linear split algorithm for converting giant tours to vehicle routes.
pub struct Split<'p> {
    /// Reference to problem parameters.
    params: &'p Params,
    /// Maximum number of vehicles allowed.
    max_vehicles: usize,

    /// Client data for the current split (indexed 1..=nb_clients).
    cli_split: Vec<ClientSplit>,
    /// Potential[k][i] = best cost to cover first i clients using k routes.
    potential: Vec<Vec<f64>>,
    /// Pred[k][i] = predecessor of client i when k routes are used.
    pred: Vec<Vec<usize>>,
    /// Cumulative distance from depot through the tour.
    sum_distance: Vec<i32>,
    /// Cumulative demand.
    sum_load: Vec<i32>,
    /// Cumulative service time.
    sum_service: Vec<i32>,
}

impl<'p> Split<'p> {
    /// Create a new Split instance for the given parameters.
    pub fn new(params: &'p Params) -> Self {
        let nb_clients = params.nb_clients as usize;
        let nb_vehicles = params.nb_vehicles as usize;

        Split {
            params,
            max_vehicles: 0,
            cli_split: vec![ClientSplit::default(); nb_clients + 1],
            potential: vec![vec![1e30_f64; nb_clients + 1]; nb_vehicles + 1],
            pred: vec![vec![0usize; nb_clients + 1]; nb_vehicles + 1],
            sum_distance: vec![0; nb_clients + 1],
            sum_load: vec![0; nb_clients + 1],
            sum_service: vec![0; nb_clients + 1],
        }
    }

    /// Main entry point: convert the individual's giant tour into routes.
    pub fn general_split(&mut self, indiv: &mut Individual<'p>, nb_max_vehicles: usize) {
        // Don't apply Split with fewer vehicles than the trivial bin-packing bound.
        let bin_packing_bound = (self.params.total_demand as f64 / self.params.vehicle_capacity as f64).ceil() as usize;
        self.max_vehicles = nb_max_vehicles.max(bin_packing_bound)
            .min(self.params.nb_vehicles as usize);

        // Pre-fetch client data from the giant tour.
        for i in 1..=self.params.nb_clients as usize {
            let chrom_idx = indiv.chrom_t[i - 1];

            self.cli_split[i] = ClientSplit {
                demand: self.params.cli[chrom_idx].demand,
                service_time: self.params.cli[chrom_idx].service_duration,
                d0_x: self.params.time_cost.get(0, chrom_idx),
                dx_0: self.params.time_cost.get(chrom_idx, 0),
                cust_num: chrom_idx,
                ..Default::default()
            };

            if i < self.params.nb_clients as usize {
                let next_idx = indiv.chrom_t[i];
                self.cli_split[i].dnext = self.params.time_cost.get(chrom_idx, next_idx);
            } else {
                self.cli_split[i].dnext = i32::MIN;
            }

            self.sum_load[i] = self.sum_load[i - 1] + self.cli_split[i].demand;
            self.sum_service[i] = self.sum_service[i - 1] + self.cli_split[i].service_time;
            self.sum_distance[i] = self.sum_distance[i - 1] + self.cli_split[i - 1].dnext;
        }

        // Try simple split first, then fall back to limited fleet.
        let mut split_ok = self.split_simple(indiv) != 0;
        if !split_ok {
            split_ok = self.split_lf(indiv) != 0;
        }

        if !split_ok {
            panic!("ERROR: no Split solution has been propagated until the last node");
        }

        // Re-evaluate costs on the newly built routes.
        indiv.evaluate_complete_cost();
    }

    /// Split for unlimited fleet — O(n) algorithm using dominance pruning.
    fn split_simple(&mut self, indiv: &mut Individual<'p>) -> i32 {
        // Reset potentials.
        self.potential[0][0] = 0.0;
        for i in 1..=self.params.nb_clients as usize {
            self.potential[0][i] = 1e30;
        }

        if self.params.is_duration_constrained {
            // O(n * B) algorithm for duration-constrained problems.
            for i in 0..self.params.nb_clients as usize {
                let mut load = 0i32;
                let mut distance = 0i32;
                let mut service_duration = 0i32;
                let cumulative_cost = self.potential[0][i];

                for j in (i + 1)..=self.params.nb_clients as usize {
                    load += self.cli_split[j].demand;
                    if load as f64 > 1.5 * self.params.vehicle_capacity as f64 {
                        break;
                    }

                    service_duration += self.cli_split[j].service_time;
                    distance += if j == i + 1 {
                        self.cli_split[j].d0_x
                    } else {
                        self.cli_split[j - 1].dnext
                    };

                    let cost = distance as f64
                        + self.cli_split[j].dx_0 as f64
                        + self.params.penalty_capacity * (load - self.params.vehicle_capacity).max(0) as f64;

                    if cumulative_cost + cost < self.potential[0][j] {
                        self.potential[0][j] = cumulative_cost + cost;
                        self.pred[0][j] = i;
                    }
                }
            }
        } else {
            // O(n) algorithm using dominance pruning.
            let mut queue = TrivialDeque::new(self.params.nb_clients as usize + 1, 0);

            for i in 1..=self.params.nb_clients as usize {
                let front = queue.get_front() as usize;
                self.potential[0][i] = self.propagate(front, i, 0);
                self.pred[0][i] = front;

                if i < self.params.nb_clients as usize {
                    let back = queue.get_back() as usize;

                    if !self.dominates(back, i, 0) {
                        while queue.size() > 0 && self.dominates_right(queue.get_back() as usize, i, 0) {
                            queue.pop_back();
                        }
                        queue.push_back(i as i32);
                    }

                    let next_front_prop = self.propagate(queue.get_front() as usize, i + 1, 0);
                    while queue.size() > 1 && next_front_prop > self.propagate(queue.get_next_front() as usize, i + 1, 0) - 1e-10 {
                        queue.pop_front();
                    }
                }
            }
        }

        // Check if we reached the last client.
        if self.potential[0][self.params.nb_clients as usize] > 1e29 {
            return 0;
        }

        // Build routes from predecessors.
        for k in (self.max_vehicles..indiv.chrom_r.len()).rev() {
            indiv.chrom_r[k].clear();
        }

        let mut end = self.params.nb_clients as usize;
        for k in (0..self.max_vehicles).rev() {
            indiv.chrom_r[k].clear();
            let begin = self.pred[0][end];
            for ii in begin..end {
                indiv.chrom_r[k].push(indiv.chrom_t[ii]);
            }
            end = begin;
        }

        if end == 0 { 1 } else { 0 }
    }

    /// Split for limited fleet — O(n * B) algorithm.
    fn split_lf(&mut self, indiv: &mut Individual<'p>) -> i32 {
        // Initialize potentials.
        for k in 0..=self.max_vehicles {
            self.potential[k][0] = 0.0;
            for i in 1..=self.params.nb_clients as usize {
                self.potential[k][i] = 1e30;
            }
        }

        if self.params.is_duration_constrained {
            let max_load = (1.5 * self.params.vehicle_capacity as f64) as i32;

            for k in 0..self.max_vehicles {
                for i in k..self.params.nb_clients as usize {
                    if self.potential[k][i] > 1e29 {
                        continue;
                    }

                    let mut load = 0i32;
                    let mut service_duration = 0i32;
                    let mut distance = 0i32;
                    let cumulative_cost = self.potential[k][i];

                    for j in (i + 1)..=self.params.nb_clients as usize {
                        load += self.cli_split[j].demand;
                        if load > max_load {
                            break;
                        }

                        service_duration += self.cli_split[j].service_time;
                        distance += if j == i + 1 {
                            self.cli_split[j].d0_x
                        } else {
                            self.cli_split[j - 1].dnext
                        };

                        let cost = distance as f64
                            + self.cli_split[j].dx_0 as f64
                            + self.params.penalty_capacity * (load - self.params.vehicle_capacity).max(0) as f64;

                        if cumulative_cost + cost < self.potential[k + 1][j] {
                            self.potential[k + 1][j] = cumulative_cost + cost;
                            self.pred[k + 1][j] = i;
                        }
                    }
                }
            }
        } else {
            // O(n) algorithm with limited fleet.
            for k in 0..self.max_vehicles {
                let mut queue = TrivialDeque::new(self.params.nb_clients as usize + 1, k as i32);

                for i in (k + 1)..=self.params.nb_clients as usize {
                    if queue.size() == 0 {
                        break;
                    }

                    let front = queue.get_front() as usize;
                    self.potential[k + 1][i] = self.propagate(front, i, k);
                    self.pred[k + 1][i] = front;

                    if i < self.params.nb_clients as usize {
                        let back = queue.get_back() as usize;
                        if !self.dominates(back, i, k) {
                            while queue.size() > 0 && self.dominates_right(queue.get_back() as usize, i, k) {
                                queue.pop_back();
                                if queue.size() == 0 {
                                    break;
                                }
                            }
                            queue.push_back(i as i32);
                        }

                        let mut next_front_prop = self.propagate(queue.get_front() as usize, i + 1, k);
                        while queue.size() > 1 && next_front_prop > self.propagate(queue.get_next_front() as usize, i + 1, k) - 1e-10 {
                            queue.pop_front();
                            if queue.size() > 0 {
                                next_front_prop = self.propagate(queue.get_front() as usize, i + 1, k);
                            }
                        }
                    }
                }
            }
        }

        // Check feasibility.
        if self.potential[self.max_vehicles][self.params.nb_clients as usize] > 1e29 {
            return 0;
        }

        // Find the cheapest number of vehicles.
        let mut min_cost = self.potential[self.max_vehicles][self.params.nb_clients as usize];
        let mut nb_routes = self.max_vehicles;

        for k in 1..self.max_vehicles {
            let potential_cost = self.potential[k][self.params.nb_clients as usize];
            if potential_cost < min_cost {
                min_cost = potential_cost;
                nb_routes = k;
            }
        }

        // Clear excess routes.
        for k in (nb_routes..indiv.chrom_r.len()).rev() {
            indiv.chrom_r[k].clear();
        }

        let mut end = self.params.nb_clients as usize;
        for k in (0..nb_routes).rev() {
            indiv.chrom_r[k].clear();
            let begin = self.pred[k + 1][end];
            indiv.chrom_r[k].reserve(end - begin);
            for ii in begin..end {
                indiv.chrom_r[k].push(indiv.chrom_t[ii]);
            }
            end = begin;
        }

        if end == 0 { 1 } else { 0 }
    }

    /// Compute the cost of extending from predecessor `i` to client `j` using route `k`.
    fn propagate(&self, i: usize, j: usize, _k: usize) -> f64 {
        let dist_diff = self.sum_distance[j] - self.sum_distance[i + 1];
        let load_diff = self.sum_load[j] - self.sum_load[i];

        self.potential[_k][i]
            + dist_diff as f64
            + self.cli_split[i + 1].d0_x as f64
            + self.cli_split[j].dx_0 as f64
            + self.params.penalty_capacity * (load_diff - self.params.vehicle_capacity).max(0) as f64
    }

    /// Test if `i` dominates `j` as a predecessor for all nodes >= j+1. Assumes i < j.
    fn dominates(&self, i: usize, j: usize, k: usize) -> bool {
        let dist_diff = self.sum_distance[j + 1] - self.sum_distance[i + 1];
        let load_diff = self.sum_load[j] - self.sum_load[i];

        self.potential[k][j] + self.cli_split[j + 1].d0_x as f64
            > self.potential[k][i] + self.cli_split[i + 1].d0_x as f64
                + dist_diff as f64
                + self.params.penalty_capacity * load_diff as f64
    }

    /// Test if `j` dominates `i` as a predecessor for all nodes >= j+1. Assumes i < j.
    fn dominates_right(&self, i: usize, j: usize, k: usize) -> bool {
        let dist_diff = self.sum_distance[j + 1] - self.sum_distance[i + 1];

        (self.potential[k][j] + self.cli_split[j + 1].d0_x as f64)
            < (self.potential[k][i] + self.cli_split[i + 1].d0_x as f64 + dist_diff as f64 + 1e-10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hgs::Params;
    use crate::vrptw::{Customer, Instance};

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
            rounding: crate::vrptw::Rounding::None,
            distances: None,
        }
    }

    #[test]
    fn test_split_basic() {
        let instance = make_test_instance();
        let params = Params::from_instance(&instance);
        let mut indiv = Individual::new_random(&params);

        let mut split = Split::new(&params);
        split.general_split(&mut indiv, 3);

        // After split, routes should be populated.
        assert!(indiv.chrom_r.iter().any(|r| !r.is_empty()));
    }

    #[test]
    fn test_split_preserves_all_customers() {
        let instance = make_test_instance();
        let params = Params::from_instance(&instance);
        let mut indiv = Individual::new_random(&params);

        let mut split = Split::new(&params);
        split.general_split(&mut indiv, 3);

        // All customers should appear exactly once across all routes.
        let mut all_customers: Vec<usize> = Vec::new();
        for route in &indiv.chrom_r {
            all_customers.extend(route);
        }
        all_customers.sort();
        assert_eq!(all_customers, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_trivial_deque() {
        let mut dq = TrivialDeque::new(10, 0);
        assert_eq!(dq.size(), 1);
        assert_eq!(dq.get_front(), 0);

        dq.push_back(1);
        dq.push_back(2);
        assert_eq!(dq.size(), 3);
        assert_eq!(dq.get_front(), 0);
        assert_eq!(dq.get_back(), 2);

        dq.pop_front();
        assert_eq!(dq.size(), 2);
        assert_eq!(dq.get_front(), 1);

        dq.pop_back();
        assert_eq!(dq.size(), 1);
        assert_eq!(dq.get_back(), 1);
    }
}
