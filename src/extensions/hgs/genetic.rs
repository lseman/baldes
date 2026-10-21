//! Genetic algorithm engine: selection, crossover (OX + SREX), local search orchestration.
//!
//! This module implements the core evolutionary loop of HGS:
//! - Two crossover operators: OX (Order Crossover) and SREX (Selective Route Exchange)
//! - Local search application to offspring
//! - Insertion of unplanned clients into offspring

use std::collections::HashSet;

use crate::extensions::hgs::local_search::HGSLocalSearch;
use crate::extensions::hgs::{Individual, Params, Population, Split};

/// Number of candidate offspring created from one individual pair.
pub const NUMBER_OF_CANDIDATE_OFFSPRINGS: usize = 4;

/// Genetic engine that runs the crossover operators and main evolutionary loop.
pub struct Genetic<'a, 'p> {
    params: &'p Params,
    split: &'a mut Split<'p>,
    local_search: &'a mut HGSLocalSearch<'p>,
    /// Reused offspring buffers: SREX writes 0 and 1, OX writes 2 and 3.
    candidate_offsprings: Vec<Individual<'p>>,
}

impl<'a, 'p> Genetic<'a, 'p> {
    /// Create a new Genetic engine.
    pub fn new(
        params: &'p Params,
        split: &'a mut Split<'p>,
        local_search: &'a mut HGSLocalSearch<'p>,
    ) -> Self {
        let candidate_offsprings =
            (0..NUMBER_OF_CANDIDATE_OFFSPRINGS).map(|_| Individual::new_empty(params)).collect();
        Genetic { params, split, local_search, candidate_offsprings }
    }

    /// Run the genetic algorithm until `max_iter_non_prod` consecutive
    /// iterations without improvement or the time limit. With a time limit,
    /// the population restarts instead of stopping on non-improvement.
    pub fn run(&mut self, population: &mut Population<'p>, max_iter_non_prod: usize) {
        if self.params.nb_clients == 1 {
            return;
        }
        const MDM_NON_UPDATING_RESTARTS: f64 = 0.05;
        let config = &self.params.config;
        let repair_threshold = config.repair_probability.max(0) as u64;
        let repeat_until_time_limit = config.time_limit > 0.0 && config.do_repeat_until_time_limit;
        let mut nb_granular = config.nb_granular;
        let mut nb_restarts = 0;
        let mut nb_iter_non_prod = 1;
        let mut nb_iter = 0usize;
        while nb_iter_non_prod <= max_iter_non_prod && !self.params.is_time_limit_exceeded() {
            // Selection and crossover, then local search on the best offspring.
            let best = {
                let (parent_a, parent_b) = population.get_non_identical_parents();
                self.best_of_srex_and_ox_crossovers(parent_a, parent_b)
            };
            let offspring = &mut self.candidate_offsprings[best];
            self.local_search.run(
                offspring,
                self.params.penalty_capacity.get(),
                self.params.penalty_time_warp.get(),
            );
            let mut is_new_best = population.add_individual(offspring, true);

            // Repair half of the infeasible offspring with stronger penalties.
            if !offspring.is_feasible && self.params.rng.borrow_mut().next() % 100 < repair_threshold {
                self.local_search.run(
                    offspring,
                    self.params.penalty_capacity.get() * 10.0,
                    self.params.penalty_time_warp.get() * 10.0,
                );
                if offspring.is_feasible {
                    is_new_best |= population.add_individual(offspring, false);
                }
            }

            nb_iter_non_prod = if is_new_best { 1 } else { nb_iter_non_prod + 1 };

            if nb_iter % 100 == 0 {
                population.manage_penalties();
            }
            let log_interval = if config.log_pool_interval > 0 { config.log_pool_interval as usize } else { 500 };
            if nb_iter % log_interval == 0 {
                population.print_state(nb_iter as i32, nb_iter_non_prod as i32);
            }

            if repeat_until_time_limit && nb_iter_non_prod == max_iter_non_prod {
                nb_restarts += 1;
                let per_restart = self.params.elapsed_secs() / nb_restarts as f64;
                let estimated_restarts = ((config.time_limit / per_restart) as i64).min(1000);
                population.mdm_elite_max_non_updating_restarts =
                    (MDM_NON_UPDATING_RESTARTS * estimated_restarts as f64) as usize;
                population.mine_elite();
                population.restart(self.split, self.local_search);
                nb_iter_non_prod = 1;
            }

            // Growth of the granular neighborhoods and of the population.
            let grows = |every_iter: i32, every_non_improving: i32| {
                (every_iter > 0 && nb_iter % every_iter as usize == 0)
                    || (every_non_improving > 0 && nb_iter_non_prod % every_non_improving as usize == 0)
            };
            if config.grow_nb_granular_size != 0
                && grows(
                    config.grow_nb_granular_after_iterations,
                    config.grow_nb_granular_after_non_improvement_iterations,
                )
            {
                nb_granular += config.grow_nb_granular_size;
                self.local_search.set_granular(nb_granular);
            }
            if config.grow_population_size != 0
                && grows(
                    config.grow_population_after_iterations,
                    config.grow_population_after_non_improvement_iterations,
                )
            {
                population.minimum_population_size += config.grow_population_size;
            }

            nb_iter += 1;
            population.nb_generations += 1;
        }
    }

    /// Order Crossover (OX) on a random segment, in both parent orders.
    /// Returns the index of the better candidate (2 or 3).
    fn crossover_ox(&mut self, parent_a: &Individual<'p>, parent_b: &Individual<'p>) -> usize {
        let nb_clients = self.params.nb_clients as u64;
        let start = self.params.rng.borrow_mut().next() % nb_clients;
        let end = loop {
            let end = self.params.rng.borrow_mut().next() % nb_clients;
            if end != start {
                break end;
            }
        };
        let (start, end) = (start as usize, end as usize);
        for (index, (first, second)) in [(2, (parent_a, parent_b)), (3, (parent_b, parent_a))] {
            let child = &mut self.candidate_offsprings[index];
            Self::do_ox_crossover(&mut child.chrom_t, &first.chrom_t, &second.chrom_t, start, end);
            self.split.general_split(child, self.params.nb_vehicles as usize);
        }
        if self.candidate_offsprings[2].my_cost_sol.penalized_cost
            < self.candidate_offsprings[3].my_cost_sol.penalized_cost
        {
            2
        } else {
            3
        }
    }

    /// One OX crossover: copy `p1[start..=end]` (cyclically), then fill the
    /// positions after `end` with the missing clients in `p2` order from `end + 1`.
    fn do_ox_crossover(child: &mut Vec<usize>, p1: &[usize], p2: &[usize], start: usize, end: usize) {
        let n = p1.len();
        child.clear();
        child.resize(n, 0);
        let mut used = vec![false; n + 1];
        let mut i = start;
        loop {
            child[i] = p1[i];
            used[p1[i]] = true;
            if i == end {
                break;
            }
            i = (i + 1) % n;
        }
        let mut write = (end + 1) % n;
        for k in 1..=n {
            let client = p2[(end + k) % n];
            if !used[client] {
                child[write] = client;
                write = (write + 1) % n;
            }
        }
    }

    /// Selective Route Exchange (SREX, Nagata and Kobayashi 2010): replace
    /// adjacent routes of parent A by routes of parent B, shifted to minimize
    /// the clients they disagree on. Returns the better candidate (0 or 1).
    fn crossover_srex(&mut self, parent_a: &Individual<'p>, parent_b: &Individual<'p>) -> usize {
        let routes_a = &parent_a.chrom_r;
        let routes_b = &parent_b.chrom_r;
        let n_routes_a = parent_a.my_cost_sol.nb_routes;
        let n_routes_b = parent_b.my_cost_sol.nb_routes;
        if n_routes_a == 0 || n_routes_b == 0 {
            self.candidate_offsprings[0].chrom_r.clone_from(routes_a);
            self.candidate_offsprings[0].evaluate_complete_cost();
            return 0;
        }

        let mut start_a = (self.params.rng.borrow_mut().next() % n_routes_a as u64) as usize;
        let n_moved = if n_routes_a.min(n_routes_b) == 1 {
            1
        } else {
            (self.params.rng.borrow_mut().next() % (n_routes_a - 1).min(n_routes_b - 1) as u64) as usize + 1
        };
        let mut start_b = if start_a < n_routes_b { start_a } else { 0 };

        let selected = |routes: &[Vec<usize>], start: usize, count: usize| -> HashSet<usize> {
            (0..n_moved).flat_map(|r| routes[(start + r) % count].iter().copied()).collect()
        };
        let mut selected_a = selected(routes_a, start_a, n_routes_a);
        let mut selected_b = selected(routes_b, start_b, n_routes_b);

        // Shift either selection by one route while that lowers the number of
        // clients in A's selection outside B's (or vice versa).
        loop {
            let a_left = (start_a + n_routes_a - 1) % n_routes_a;
            let a_right = (start_a + n_moved) % n_routes_a;
            let a_last = (start_a + n_moved - 1) % n_routes_a;
            let b_left = (start_b + n_routes_b - 1) % n_routes_b;
            let b_right = (start_b + n_moved) % n_routes_b;
            let b_last = (start_b + n_moved - 1) % n_routes_b;
            let outside =
                |route: &[usize], set: &HashSet<usize>| route.iter().filter(|c| !set.contains(c)).count() as i32;
            let inside =
                |route: &[usize], set: &HashSet<usize>| route.iter().filter(|c| set.contains(c)).count() as i32;
            let differences = [
                outside(&routes_a[a_left], &selected_b) - outside(&routes_a[a_last], &selected_b),
                outside(&routes_a[a_right], &selected_b) - outside(&routes_a[start_a], &selected_b),
                inside(&routes_b[b_last], &selected_a) - inside(&routes_b[b_left], &selected_a),
                inside(&routes_b[start_b], &selected_a) - inside(&routes_b[b_right], &selected_a),
            ];
            let best = *differences.iter().min().unwrap();
            if best >= 0 {
                break;
            }
            // The first shift reaching the minimum: A left, A right, B left, B right.
            let shift = differences.iter().position(|&d| d == best).unwrap();
            let (routes, selection, start, count) = if shift < 2 {
                (routes_a, &mut selected_a, &mut start_a, n_routes_a)
            } else {
                (routes_b, &mut selected_b, &mut start_b, n_routes_b)
            };
            let (removed, added) = if shift % 2 == 0 {
                *start = (*start + count - 1) % count;
                ((*start + n_moved) % count, *start)
            } else {
                let removed = *start;
                *start = (*start + 1) % count;
                (removed, (*start + n_moved - 1) % count)
            };
            for client in &routes[removed] {
                selection.remove(client);
            }
            selection.extend(routes[added].iter().copied());
        }

        let mut a_not_b: Vec<usize> = selected_a.difference(&selected_b).copied().collect();
        a_not_b.sort_unstable();
        let b_not_a: HashSet<usize> = selected_b.difference(&selected_a).copied().collect();

        // Offspring 0 keeps B's routes whole and drops B-only clients from A's
        // other routes; offspring 1 does the opposite.
        let nb_vehicles = self.params.nb_vehicles as usize;
        let (first, rest) = self.candidate_offsprings.split_at_mut(1);
        let (child0, child1) = (&mut first[0], &mut rest[0]);
        for r in 0..n_moved {
            let index_a = (start_a + r) % n_routes_a;
            let route_b = &routes_b[(start_b + r) % n_routes_b];
            child0.chrom_r[index_a].clone_from(route_b);
            child1.chrom_r[index_a] = route_b.iter().copied().filter(|c| !b_not_a.contains(c)).collect();
        }
        for r in n_moved..n_routes_a {
            let index_a = (start_a + r) % n_routes_a;
            child0.chrom_r[index_a] =
                routes_a[index_a].iter().copied().filter(|c| !b_not_a.contains(c)).collect();
            child1.chrom_r[index_a].clone_from(&routes_a[index_a]);
        }
        for r in n_routes_a..nb_vehicles {
            child0.chrom_r[r].clear();
            child1.chrom_r[r].clear();
        }

        for child in [child0, child1] {
            Self::insert_unplanned_tasks(self.params, child, &a_not_b);
            child.evaluate_complete_cost();
        }
        if self.candidate_offsprings[0].my_cost_sol.penalized_cost
            < self.candidate_offsprings[1].my_cost_sol.penalized_cost
        {
            0
        } else {
            1
        }
    }

    /// Insert each unplanned client at its cheapest position whose time
    /// windows look reachable. HGS drops a client with no such position; this
    /// port then uses the cheapest position overall so no client is lost.
    fn insert_unplanned_tasks(params: &Params, offspring: &mut Individual<'p>, unplanned_tasks: &[usize]) {
        let time_cost = &params.time_cost;
        let clients = &params.cli;
        let non_empty: Vec<usize> =
            (0..params.nb_vehicles as usize).filter(|&r| !offspring.chrom_r[r].is_empty()).collect();

        for &client in unplanned_tasks {
            let earliest = clients[client].earliest_arrival;
            let latest = clients[client].latest_arrival;
            // (cost, route, position); the first minimum wins, as std::min_element.
            let mut best_feasible: Option<(f64, usize, usize)> = None;
            let mut best_any: Option<(f64, usize, usize)> = None;
            let mut consider = |cost: f64, route: usize, position: usize, feasible: bool| {
                if best_any.map_or(true, |b| cost < b.0) {
                    best_any = Some((cost, route, position));
                }
                if feasible && best_feasible.map_or(true, |b| cost < b.0) {
                    best_feasible = Some((cost, route, position));
                }
            };
            for &r in &non_empty {
                let route = &offspring.chrom_r[r];
                let first = route[0];
                let cost_to_first = time_cost.get(client, first);
                consider(
                    time_cost.get(0, client) + cost_to_first - time_cost.get(0, first),
                    r,
                    0,
                    earliest + cost_to_first < clients[first].latest_arrival,
                );
                for pos in 1..route.len() {
                    let (prev, next) = (route[pos - 1], route[pos]);
                    let (from_prev, to_next) = (time_cost.get(prev, client), time_cost.get(client, next));
                    consider(
                        from_prev + to_next - time_cost.get(prev, next),
                        r,
                        pos,
                        clients[prev].earliest_arrival + from_prev < latest
                            && earliest + to_next < clients[next].latest_arrival,
                    );
                }
                let last = *route.last().unwrap();
                let from_last = time_cost.get(last, client);
                consider(
                    from_last + time_cost.get(client, 0) - time_cost.get(last, 0),
                    r,
                    route.len(),
                    clients[last].earliest_arrival + from_last < latest,
                );
            }
            match best_feasible.or(best_any) {
                Some((_, route, position)) => offspring.chrom_r[route].insert(position, client),
                None => offspring.chrom_r[0].push(client),
            }
        }
    }

    /// Run both crossovers and return the index of the better offspring.
    pub fn best_of_srex_and_ox_crossovers(
        &mut self,
        parent_a: &Individual<'p>,
        parent_b: &Individual<'p>,
    ) -> usize {
        let ox = self.crossover_ox(parent_a, parent_b);
        let srex = self.crossover_srex(parent_a, parent_b);
        if self.candidate_offsprings[ox].my_cost_sol.penalized_cost
            < self.candidate_offsprings[srex].my_cost_sol.penalized_cost
        {
            ox
        } else {
            srex
        }
    }
}
