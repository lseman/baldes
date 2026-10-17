//! Genetic algorithm engine: selection, crossover (OX + SREX), local search orchestration.
//!
//! This module implements the core evolutionary loop of HGS:
//! - Two crossover operators: OX (Order Crossover) and SREX (Segment-based RECombining XOver)
//! - Local search application to offspring
//! - Insertion of unplanned clients into offspring

use crate::hgs::{Individual, Params, Split};
use crate::hgs::local_search::HGSLocalSearch;

/// Number of candidate offspring created from one individual pair.
const NUMBER_OF_CANDIDATE_OFFSPRINGS: usize = 4;

/// Genetic engine that runs the crossover operators.
///
/// Holds references to the problem parameters, split algorithm, and local search structure.
/// Manages crossover operators and offspring pool.
pub struct Genetic<'p> {
    params: &'p Params,                  /// Problem parameters
    split: &'p mut Split<'p>,            /// Split algorithm
    local_search: &'p HGSLocalSearch<'p>, /// Local search structure

    /// Pre-allocated offspring for in-place crossover (0,1 = SREX; 2,3 = OX).
    candidate_offsprings: Vec<Individual<'p>>,
}

impl<'p> Genetic<'p> {
    /// Create a new Genetic engine.
    pub fn new(
        params: &'p Params,
        split: &'p mut Split<'p>,
        local_search: &'p HGSLocalSearch<'p>,
    ) -> Self {
        let mut candidate_offsprings = Vec::with_capacity(NUMBER_OF_CANDIDATE_OFFSPRINGS);
        for _ in 0..NUMBER_OF_CANDIDATE_OFFSPRINGS {
            candidate_offsprings.push(Individual::new_random(params));
        }
        Genetic {
            params,
            split,
            local_search,
            candidate_offsprings,
        }
    }

    /// Order Crossover (OX): creates offspring by copying a segment from parent A
    /// and filling remaining positions from parent B in order.
    fn crossover_ox(&mut self, parents: (&Individual<'p>, &Individual<'p>)) -> Individual<'p> {
        let nb_clients = self.params.nb_clients as usize;
        let start = (self.params.rng.borrow_mut().next_u64(nb_clients as u64)) as usize;
        let mut end: usize;
        loop {
            end = (self.params.rng.borrow_mut().next_u64(nb_clients as u64)) as usize;
            if end != start { break; }
        }

        // Clone parent chrom_t to avoid borrow conflicts
        let p1_chrom: Vec<usize> = parents.0.chrom_t.clone();
        let p2_chrom: Vec<usize> = parents.1.chrom_t.clone();

        // Perform crossovers in-place on candidates 2 and 3 (sequential to avoid double borrow)
        Genetic::do_ox_crossover(self.params, self.split, &mut self.candidate_offsprings[2], &p1_chrom, &p2_chrom, start, end);
        Genetic::do_ox_crossover(self.params, self.split, &mut self.candidate_offsprings[3], &parents.0.chrom_t, &parents.1.chrom_t, start, end);

        // Return best offspring (take ownership)
        if self.candidate_offsprings[2].my_cost_sol.penalized_cost < self.candidate_offsprings[3].my_cost_sol.penalized_cost {
            std::mem::replace(&mut self.candidate_offsprings[2], Individual::new_random(self.params))
        } else {
            std::mem::replace(&mut self.candidate_offsprings[3], Individual::new_random(self.params))
        }
    }

    /// Perform one OX crossover for result from parents, with crossover zone [start, end].
    fn do_ox_crossover(
        params: &'p Params,
        split: &mut Split<'p>,
        result: &mut Individual<'p>,
        p1: &[usize],
        p2: &[usize],
        start: usize,
        end: usize,
    ) {
        let n = params.nb_clients as usize;

        assert!(params.nb_clients > 0);
        assert_eq!(p1.len(), n);
        assert_eq!(p2.len(), n);

        // Detect gene domain: 0..n-1 or 1..n (depot handling varies per codebase)
        let mut min_gene = p1[0];
        let mut max_gene = p1[0];
        for &g in p1.iter() {
            if g < min_gene { min_gene = g; }
            if g > max_gene { max_gene = g; }
        }
        for &g in p2.iter() {
            if g < min_gene { min_gene = g; }
            if g > max_gene { max_gene = g; }
        }

        // Case A: genes in [0..n-1] -> used size n, index = g
        // Case B: genes in [1..n]   -> used size n+1, index = g
        let one_based = min_gene >= 1 && max_gene <= n;
        let _zero_based = min_gene >= 0 && max_gene <= n.saturating_sub(1);
        assert!(one_based || _zero_based);

        let used_size = if one_based { n + 1 } else { n };
        let mut used = vec![false; used_size];
        let mut filled = vec![false; n];
        let mut child = vec![0usize; n];

        // Phase 1: copy segment from p1 (inclusive, with wrap support)
        let mut i = start;
        loop {
            let gene = p1[i];
            assert!(gene < used_size);
            child[i] = gene;
            filled[i] = true;
            used[gene] = true;

            if i == end { break; }
            i = (i + 1) % n;
        }

        // Phase 2: fill remaining positions from p2, skipping already used genes
        let mut write = (end + 1) % n;
        let mut read_pos = (end + 1) % n;
        let mut placed = if start <= end { end - start + 1 } else { (n - start) + (end + 1) };

        while placed < n {
            let g = p2[read_pos];
            assert!(g < used_size);

            if !used[g] {
                while filled[write] {
                    write = (write + 1) % n;
                }
                child[write] = g;
                filled[write] = true;
                used[g] = true;
                placed += 1;
                write = (write + 1) % n;
            }
            read_pos = (read_pos + 1) % n;
        }

        result.chrom_t = child;
        split.general_split(result, params.nb_vehicles as usize);
    }

    /// Segment-based RECombining XOver (SREX): selects adjacent routes from each parent,
    /// locally improves the selection by shifting boundaries, then builds offspring.
    fn crossover_srex(&mut self, parents: (&Individual<'p>, &Individual<'p>)) -> Individual<'p> {
        let n_of_routes_a = parents.0.my_cost_sol.nb_routes as usize;
        let n_of_routes_b = parents.1.my_cost_sol.nb_routes as usize;

        if n_of_routes_a == 0 || n_of_routes_b == 0 {
            // Fallback: copy parent A
            self.candidate_offsprings[0].chrom_r = parents.0.chrom_r.clone();
            self.candidate_offsprings[0].evaluate_complete_cost();
            return std::mem::replace(&mut self.candidate_offsprings[0], Individual::new_random(self.params));
        }

        let mut start_a = self.params.rng.borrow_mut().next_u64(n_of_routes_a as u64) as usize;
        let min_routes = n_of_routes_a.min(n_of_routes_b);
        let n_of_moved_routes = if min_routes == 1 {
            1
        } else {
            (self.params.rng.borrow_mut().next_u64((min_routes - 1) as u64) as usize) + 1
        };
        let mut start_b = if start_a < n_of_routes_b { start_a } else { 0 };

        // Collect clients from selected routes in each parent
        let mut clients_in_selected_a: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for r in 0..n_of_moved_routes {
            let route_idx = (start_a + r) % n_of_routes_a;
            for &c in &parents.0.chrom_r[route_idx] {
                clients_in_selected_a.insert(c);
            }
        }

        let mut clients_in_selected_b: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for r in 0..n_of_moved_routes {
            let route_idx = (start_b + r) % n_of_routes_b;
            for &c in &parents.1.chrom_r[route_idx] {
                clients_in_selected_b.insert(c);
            }
        }

        // Phase 2: Local improvement by shifting selected routes
        let mut improved = true;
        while improved {
            let idx_a_left = (start_a + n_of_routes_a - 1) % n_of_routes_a;
            let idx_a_right = (start_a + n_of_moved_routes) % n_of_routes_a;
            let idx_a_last_moved = (start_a + n_of_moved_routes - 1) % n_of_routes_a;
            let idx_b_left = (start_b + n_of_routes_b - 1) % n_of_routes_b;
            let idx_b_right = (start_b + n_of_moved_routes) % n_of_routes_b;
            let idx_b_last_moved = (start_b + n_of_moved_routes - 1) % n_of_routes_b;

            let count_clients = |route: &[usize], client_set: &std::collections::HashSet<usize>, count_outside: bool| -> i32 {
                route.iter().filter(|&&c| (client_set.contains(&c) != count_outside) as bool).count() as i32
            };

            let left_a_first = count_clients(&parents.0.chrom_r[idx_a_left], &clients_in_selected_b, true);
            let left_a_second = count_clients(&parents.0.chrom_r[idx_a_last_moved], &clients_in_selected_b, true);
            let right_a_first = count_clients(&parents.0.chrom_r[idx_a_right], &clients_in_selected_b, true);
            let right_a_second = count_clients(&parents.0.chrom_r[start_a], &clients_in_selected_b, true);

            let left_b_first = count_clients(&parents.1.chrom_r[idx_b_last_moved], &clients_in_selected_a, false);
            let left_b_second = count_clients(&parents.1.chrom_r[idx_b_left], &clients_in_selected_a, false);
            let right_b_first = count_clients(&parents.1.chrom_r[start_b], &clients_in_selected_a, false);
            let right_b_second = count_clients(&parents.1.chrom_r[idx_b_right], &clients_in_selected_a, false);

            let diff_a_left = left_a_first - left_a_second;
            let diff_a_right = right_a_first - right_a_second;
            let diff_b_left = left_b_first - left_b_second;
            let diff_b_right = right_b_first - right_b_second;

            let best_diff = diff_a_left.min(diff_a_right).min(diff_b_left).min(diff_b_right);

            if best_diff < 0 {
                if best_diff == diff_a_left {
                    for &c in &parents.0.chrom_r[idx_a_last_moved] {
                        clients_in_selected_a.remove(&c);
                    }
                    start_a = (start_a + n_of_routes_a - 1) % n_of_routes_a;
                    let add_idx = start_a;
                    for &c in &parents.0.chrom_r[add_idx] {
                        clients_in_selected_a.insert(c);
                    }
                } else if best_diff == diff_a_right {
                    let remove_idx = start_a;
                    for &c in &parents.0.chrom_r[remove_idx] {
                        clients_in_selected_a.remove(&c);
                    }
                    start_a = (start_a + 1) % n_of_routes_a;
                    let add_idx = (start_a + n_of_moved_routes - 1) % n_of_routes_a;
                    for &c in &parents.0.chrom_r[add_idx] {
                        clients_in_selected_a.insert(c);
                    }
                } else if best_diff == diff_b_left {
                    for &c in &parents.1.chrom_r[idx_b_last_moved] {
                        clients_in_selected_b.remove(&c);
                    }
                    start_b = (start_b + n_of_routes_b - 1) % n_of_routes_b;
                    let add_idx = start_b;
                    for &c in &parents.1.chrom_r[add_idx] {
                        clients_in_selected_b.insert(c);
                    }
                } else if best_diff == diff_b_right {
                    let remove_idx = start_b;
                    for &c in &parents.1.chrom_r[remove_idx] {
                        clients_in_selected_b.remove(&c);
                    }
                    start_b = (start_b + 1) % n_of_routes_b;
                    let add_idx = (start_b + n_of_moved_routes - 1) % n_of_routes_b;
                    for &c in &parents.1.chrom_r[add_idx] {
                        clients_in_selected_b.insert(c);
                    }
                }
            } else {
                improved = false;
            }
        }

        // Phase 3: Identify differences (A - B and B - A)
        let mut sorted_a: Vec<usize> = clients_in_selected_a.into_iter().collect();
        sorted_a.sort();
        let mut sorted_b: Vec<usize> = clients_in_selected_b.into_iter().collect();
        sorted_b.sort();

        let mut clients_a_not_b = Vec::new();
        let mut clients_b_not_a = Vec::new();

        // Set difference: A - B
        let mut ai = 0usize;
        let mut bi = 0usize;
        while ai < sorted_a.len() && bi < sorted_b.len() {
            match sorted_a[ai].cmp(&sorted_b[bi]) {
                std::cmp::Ordering::Less => { clients_a_not_b.push(sorted_a[ai]); ai += 1; }
                std::cmp::Ordering::Equal => { ai += 1; bi += 1; }
                std::cmp::Ordering::Greater => { bi += 1; }
            }
        }
        clients_a_not_b.extend_from_slice(&sorted_a[ai..]);

        // Set difference: B - A
        let mut clients_in_selected_b_not_a: std::collections::HashSet<usize> = std::collections::HashSet::new();
        ai = 0; bi = 0;
        while ai < sorted_a.len() && bi < sorted_b.len() {
            match sorted_a[ai].cmp(&sorted_b[bi]) {
                std::cmp::Ordering::Less => { ai += 1; }
                std::cmp::Ordering::Equal => { ai += 1; bi += 1; }
                std::cmp::Ordering::Greater => { clients_b_not_a.push(sorted_b[bi]); bi += 1; }
            }
        }
        clients_b_not_a.extend_from_slice(&sorted_b[bi..]);
        for &c in &clients_b_not_a {
            clients_in_selected_b_not_a.insert(c);
        }

        // Phase 4: Build offspring by replacing selected routes
        for r in 0..n_of_moved_routes {
            let index_a = (start_a + r) % n_of_routes_a;
            let index_b = (start_b + r) % n_of_routes_b;
            let mut offspring0_route = Vec::new();
            let mut offspring1_route = Vec::new();

            for &c in &parents.1.chrom_r[index_b] {
                offspring0_route.push(c);
            }
            for &c in &parents.1.chrom_r[index_b] {
                if !clients_in_selected_b_not_a.contains(&c) {
                    offspring1_route.push(c);
                }
            }

            self.candidate_offsprings[0].chrom_r[index_a] = offspring0_route;
            self.candidate_offsprings[1].chrom_r[index_a] = offspring1_route;
        }

        for r in n_of_moved_routes..n_of_routes_a {
            let index_a = (start_a + r) % n_of_routes_a;
            let mut offspring0_route = Vec::new();
            let mut offspring1_route = Vec::new();

            for &c in &parents.0.chrom_r[index_a] {
                if !clients_in_selected_b_not_a.contains(&c) {
                    offspring0_route.push(c);
                }
                offspring1_route.push(c);
            }

            self.candidate_offsprings[0].chrom_r[index_a] = offspring0_route;
            self.candidate_offsprings[1].chrom_r[index_a] = offspring1_route;
        }

        for r in n_of_routes_a..self.params.nb_vehicles as usize {
            self.candidate_offsprings[0].chrom_r[r].clear();
            self.candidate_offsprings[1].chrom_r[r].clear();
        }

        // Phase 5: Insert unplanned clients
        Self::insert_unplanned_tasks(self.params, self.split, &mut self.candidate_offsprings[0], &clients_a_not_b);
        Self::insert_unplanned_tasks(self.params, self.split, &mut self.candidate_offsprings[1], &clients_a_not_b);

        self.candidate_offsprings[0].evaluate_complete_cost();
        self.candidate_offsprings[1].evaluate_complete_cost();

        if self.candidate_offsprings[0].my_cost_sol.penalized_cost < self.candidate_offsprings[1].my_cost_sol.penalized_cost {
            std::mem::replace(&mut self.candidate_offsprings[0], Individual::new_random(self.params))
        } else {
            std::mem::replace(&mut self.candidate_offsprings[1], Individual::new_random(self.params))
        }
    }

    /// Insert unplanned clients (those in removed routes but not in inserted routes) into offspring.
    fn insert_unplanned_tasks(
        params: &'p Params,
        _split: &mut Split<'p>,
        offspring: &mut Individual<'p>,
        unplanned_tasks: &[usize],
    ) {
        let mut non_empty_routes = Vec::new();
        for r in 0..params.nb_vehicles as usize {
            if !offspring.chrom_r[r].is_empty() {
                non_empty_routes.push(r);
            }
        }

        let time_cost = &params.time_cost;
        let clients = &params.cli;

        for &client in unplanned_tasks {
            let client_data = &clients[client];
            let client_earliest = client_data.earliest_arrival;
            let client_latest = client_data.latest_arrival;

            #[derive(Clone)]
            struct InsertionPoint {
                route_idx: usize,
                position: usize,
                cost: i32,
            }

            let mut insertion_points = Vec::new();

            for &route_idx in &non_empty_routes {
                let route = &offspring.chrom_r[route_idx];
                let route_size = route.len();

                // Insertion at beginning or between clients or at end
                if route_size > 0 {
                    let first_client = route[0];
                    let cost_to_client = time_cost.get(0, client) as i32;
                    let cost_client_to_first = time_cost.get(client, first_client) as i32;
                    let orig_cost = time_cost.get(0, first_client) as i32;

                    if client_earliest + cost_client_to_first < clients[first_client].latest_arrival {
                        insertion_points.push(InsertionPoint {
                            route_idx,
                            position: 0,
                            cost: cost_to_client + cost_client_to_first - orig_cost,
                        });
                    }

                    // Insertion between clients
                    let mut prev_client = first_client;
                    let prev_earliest = clients[prev_client].earliest_arrival;
                    for pos in 1..route_size {
                        let next_client = route[pos];
                        let cost_from_prev = time_cost.get(prev_client, client) as i32;
                        let cost_to_next = time_cost.get(client, next_client) as i32;
                        let orig_cost_between = time_cost.get(prev_client, next_client) as i32;

                        if prev_earliest + cost_from_prev < client_latest
                            && client_earliest + cost_to_next < clients[next_client].latest_arrival
                        {
                            insertion_points.push(InsertionPoint {
                                route_idx,
                                position: pos,
                                cost: cost_from_prev + cost_to_next - orig_cost_between,
                            });
                        }
                        prev_client = next_client;
                    }

                    // Insertion at end
                    let last_client = route[route_size - 1];
                    let cost_from_last = time_cost.get(last_client, client) as i32;
                    if clients[last_client].earliest_arrival + cost_from_last < client_latest {
                        let cost_from_client_to_depot = time_cost.get(client, 0) as i32;
                        let orig_cost_from_last_to_depot = time_cost.get(last_client, 0) as i32;
                        insertion_points.push(InsertionPoint {
                            route_idx,
                            position: route_size,
                            cost: cost_from_last + cost_from_client_to_depot - orig_cost_from_last_to_depot,
                        });
                    }
                }
            }

            if let Some(best_point) = insertion_points.iter().min_by_key(|p| p.cost) {
                offspring.chrom_r[best_point.route_idx].insert(best_point.position, client);
            }
        }
    }

    /// Run both OX and SREX crossovers, returning the best result.
    pub fn best_of_srex_and_ox_crossovers(
        &mut self,
        parents: (&Individual<'p>, &Individual<'p>),
    ) -> Individual<'p> {
        let mut offspring_ox = self.crossover_ox(parents);
        let ox_cost = offspring_ox.my_cost_sol.penalized_cost;

        let mut offspring_srex = self.crossover_srex(parents);
        let srex_cost = offspring_srex.my_cost_sol.penalized_cost;

        if ox_cost < srex_cost {
            std::mem::replace(&mut offspring_ox, Individual::new_random(self.params))
        } else {
            std::mem::replace(&mut offspring_srex, Individual::new_random(self.params))
        }
    }
}
