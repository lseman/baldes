use super::*;

impl<'p> Population<'p> {
    /// Run local search on individual and add it to population, repairing if needed.
    pub fn do_hgs_local_search_and_add_individual(
        &mut self,
        indiv: &mut Individual<'p>,
        local_search: &mut HGSLocalSearch<'p>,
    ) {
        local_search.run(
            indiv,
            self.params.penalty_capacity.get(),
            self.params.penalty_time_warp.get(),
        );

        self.add_individual(indiv, true);

        if !indiv.is_feasible {
            let repair_prob = self.params.config.repair_probability as u64;
            let roll = self.params.rng.borrow_mut().next_u64(100);
            if roll < repair_prob {
                local_search.run(
                    indiv,
                    self.params.penalty_capacity.get() * 10.0,
                    self.params.penalty_time_warp.get() * 10.0,
                );
                if indiv.is_feasible {
                    self.add_individual(indiv, false);
                }
            }
        }
    }

    /// Generate the initial population using constructive heuristics.
    pub fn generate_population(
        &mut self,
        split: &mut Split<'p>,
        local_search: &mut HGSLocalSearch<'p>,
    ) {
        if self.params.nb_clients == 1 {
            let mut random_indiv = Individual::new_random(self.params);
            split.general_split(&mut random_indiv, self.params.nb_vehicles as usize);
            self.add_individual(&random_indiv, true);
            return;
        }

        let config = &self.params.config;
        let fraction_nearest = config.fraction_generated_nearest;
        let fraction_furthest = config.fraction_generated_furthest;
        let fraction_sweep = config.fraction_generated_sweep;
        let fraction_random = config.fraction_generated_randomly;
        let fraction_mdm = config.fraction_generated_mdm;
        let min_sweep_fill = config.min_sweep_fill_percentage.max(30);
        let max_tol_cap = config.max_tolerated_capacity_violation;
        let max_tol_tw = config.max_tolerated_time_warp;

        let total_individuals = 4 * self.minimum_population_size;

        self.params
            .penalty_time_warp
            .set(config.initial_time_warp_penalty);

        let count_nearest = (fraction_nearest * total_individuals as f64).round() as i32;
        let count_furthest = (fraction_furthest * total_individuals as f64).round() as i32;
        let count_sweep = (fraction_sweep * total_individuals as f64).round() as i32;
        let count_random = (fraction_random * total_individuals as f64).round() as i32;
        let count_mdm = (fraction_mdm * total_individuals as f64).round() as i32;

        // 1. Savings (Clarke and Wright) seeded with the mined MDM patterns
        for _ in 0..count_mdm {
            if self.params.is_time_limit_exceeded() {
                return;
            }
            let mut indiv = self.savings_individual(split);
            self.do_hgs_local_search_and_add_individual(&mut indiv, local_search);
        }
        self.print_state(-1, -1);

        // 2. Nearest
        for i in 0..count_nearest {
            if self.params.is_time_limit_exceeded() {
                return;
            }
            let tol_cap = if i == 0 {
                0
            } else {
                (self
                    .params
                    .rng
                    .borrow_mut()
                    .next_u64((max_tol_cap + 1) as u64)) as i32
            };
            let tol_tw = if i == 0 {
                0
            } else {
                (self
                    .params
                    .rng
                    .borrow_mut()
                    .next_u64((max_tol_tw + 1) as u64)) as i32
            };
            let mut indiv = Individual::new_empty(self.params);
            local_search.construct_individual_with_seed_order(tol_cap, tol_tw, false, &mut indiv);
            self.do_hgs_local_search_and_add_individual(&mut indiv, local_search);
        }
        self.print_state(-1, -1);

        // 3. Furthest
        for i in 0..count_furthest {
            if self.params.is_time_limit_exceeded() {
                return;
            }
            let tol_cap = if i == 0 {
                0
            } else {
                (self
                    .params
                    .rng
                    .borrow_mut()
                    .next_u64((max_tol_cap + 1) as u64)) as i32
            };
            let tol_tw = if i == 0 {
                0
            } else {
                (self
                    .params
                    .rng
                    .borrow_mut()
                    .next_u64((max_tol_tw + 1) as u64)) as i32
            };
            let mut indiv = Individual::new_empty(self.params);
            local_search.construct_individual_with_seed_order(tol_cap, tol_tw, true, &mut indiv);
            self.do_hgs_local_search_and_add_individual(&mut indiv, local_search);
        }
        self.print_state(-1, -1);

        // 4. Sweep
        for i in 0..count_sweep {
            if self.params.is_time_limit_exceeded() {
                return;
            }
            let fill = if i == 0 {
                100
            } else {
                let range = (100 - min_sweep_fill + 1) as u64;
                min_sweep_fill + (self.params.rng.borrow_mut().next_u64(range)) as i32
            };
            let mut indiv = Individual::new_empty(self.params);
            local_search.construct_individual_by_sweep(fill, &mut indiv);
            self.do_hgs_local_search_and_add_individual(&mut indiv, local_search);
        }
        self.print_state(-1, -1);

        // 5. Random
        for _ in 0..count_random {
            if self.params.is_time_limit_exceeded() {
                return;
            }
            let mut indiv = Individual::new_random(self.params);
            split.general_split(&mut indiv, self.params.nb_vehicles as usize);
            self.do_hgs_local_search_and_add_individual(&mut indiv, local_search);
        }
        self.print_state(-1, -1);
    }

    /// A savings (RCWS) individual built around the next MDM pattern. HGS
    /// can leave clients out or exceed the fleet here; the routes are then
    /// concatenated with the missing clients into a giant tour and split.
    pub(super) fn savings_individual(&mut self, split: &mut Split<'p>) -> Individual<'p> {
        let pattern = self.next_mdm_pattern();
        let mut indiv = Individual::from_rcws(self.params, pattern.as_deref());
        let nb_vehicles = self.params.nb_vehicles as usize;
        let mut routes: Vec<Vec<usize>> = indiv
            .chrom_r
            .iter()
            .filter(|r| !r.is_empty())
            .cloned()
            .collect();
        let mut routed = vec![false; self.params.nb_clients as usize + 1];
        for &client in routes.iter().flatten() {
            routed[client] = true;
        }
        let missing: Vec<usize> = (1..routed.len()).filter(|&c| !routed[c]).collect();
        indiv.chrom_t = routes
            .iter()
            .flatten()
            .copied()
            .chain(missing.iter().copied())
            .collect();
        if routes.len() <= nb_vehicles && missing.is_empty() {
            routes.resize(nb_vehicles, Vec::new());
            indiv.chrom_r = routes;
            indiv.evaluate_complete_cost();
        } else {
            indiv.chrom_r = vec![Vec::new(); nb_vehicles];
            split.general_split(&mut indiv, nb_vehicles);
        }
        indiv
    }

    /// Restart the population: clears current pools, regenerates solutions.
    pub fn restart(&mut self, split: &mut Split<'p>, local_search: &mut HGSLocalSearch<'p>) {
        if self.params.verbose {
            crate::log_err!("  HGS restart | Creating a new population");
        }
        self.feasible.clear();
        self.infeasible.clear();
        self.best_restart = Box::new(Individual::new_empty(self.params));
        self.mdm_elite_non_updating_restarts += 1;

        self.generate_population(split, local_search);
    }
}
