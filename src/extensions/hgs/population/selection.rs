use super::*;

impl<'p> Population<'p> {
    /// Add an individual to the population. Returns true if it improves the restart best.
    pub fn add_individual(&mut self, indiv: &Individual<'p>, update_feasible: bool) -> bool {
        if let Some(routes) = self.collected_routes.as_mut() {
            routes.extend(indiv.chrom_r.iter().filter(|r| !r.is_empty()).cloned());
        }
        if update_feasible {
            self.last_feas_load
                .push_back(indiv.my_cost_sol.capacity_excess == 0);
            self.last_feas_tw
                .push_back(indiv.my_cost_sol.time_warp <= MY_EPSILON);
            self.last_feas_load.pop_front();
            self.last_feas_tw.pop_front();
        }

        let is_feas = indiv.is_feasible;
        let mut my_individual = Box::new(indiv.clone());

        // Update proximity with all individuals in the chosen subpopulation
        let subpop = if is_feas {
            &mut self.feasible
        } else {
            &mut self.infeasible
        };
        for other in subpop.iter_mut() {
            let dist = my_individual.broken_pairs_distance(other);
            other.add_proximity(dist, my_individual.id);
            my_individual.add_proximity(dist, other.id);
        }

        // Insert sorted by penalized cost
        let mut place = subpop.len();
        while place > 0
            && subpop[place - 1].my_cost_sol.penalized_cost
                > indiv.my_cost_sol.penalized_cost - MY_EPSILON
        {
            place -= 1;
        }
        subpop.insert(place, my_individual);

        // Survivor selection
        let min_pop = self.minimum_population_size as usize;
        let max_pop = min_pop + self.params.config.generation_size as usize;
        if subpop.len() > max_pop {
            while subpop.len() > min_pop {
                Self::remove_worst_biased_fitness(self.params, subpop);
            }
        }

        // Track best solution
        if is_feas
            && (self.best_restart.my_cost_sol.penalized_cost > 1e29
                || indiv.my_cost_sol.penalized_cost
                    < self.best_restart.my_cost_sol.penalized_cost - MY_EPSILON)
        {
            self.update_mdm_elite(indiv);
            self.best_restart = Box::new(indiv.clone());
            if self.best_overall.my_cost_sol.penalized_cost > 1e29
                || self.best_restart.my_cost_sol.penalized_cost
                    < self.best_overall.my_cost_sol.penalized_cost - MY_EPSILON
            {
                self.best_overall = Box::new(indiv.clone());
            }
            self.search_progress.push((
                self.params.elapsed_secs(),
                self.best_overall.my_cost_sol.penalized_cost,
            ));
            true
        } else {
            false
        }
    }

    /// Update biased fitness values of individuals in a subpopulation.
    pub fn update_biased_fitnesses(params: &Params, pop: &mut [Box<Individual<'p>>]) {
        if pop.is_empty() {
            return;
        }
        if pop.len() == 1 {
            pop[0].biased_fitness = 0.0;
            return;
        }

        let mut ranking: Vec<(SortableF64, usize)> = Vec::with_capacity(pop.len());
        for (i, indiv) in pop.iter().enumerate() {
            ranking.push((
                SortableF64(
                    -indiv.average_broken_pairs_distance_closest(params.config.nb_close as usize),
                ),
                i,
            ));
        }
        ranking.sort();

        let n = pop.len();
        let n_minus_1 = (n - 1) as f64;
        for (div_idx, &(_, indiv_idx)) in ranking.iter().enumerate() {
            let div_rank = div_idx as f64 / n_minus_1;
            let fit_rank = indiv_idx as f64 / n_minus_1;

            if n <= params.config.nb_elite as usize {
                pop[indiv_idx].biased_fitness = fit_rank;
            } else if params.config.diversity_weight > 0.0 {
                pop[indiv_idx].biased_fitness =
                    fit_rank + params.config.diversity_weight * div_rank;
            } else {
                pop[indiv_idx].biased_fitness =
                    fit_rank + (1.0 - (params.config.nb_elite as f64 / n as f64)) * div_rank;
            }
        }
    }

    /// Remove the individual with the worst biased fitness from the subpopulation.
    pub fn remove_worst_biased_fitness(params: &Params, pop: &mut Vec<Box<Individual<'p>>>) {
        Self::update_biased_fitnesses(params, pop);
        if pop.len() <= 1 {
            return;
        }

        let mut worst_pos = 1;
        let mut worst_is_clone = false;
        let mut worst_fitness = -1e30;

        for i in 1..pop.len() {
            let is_clone = pop[i].average_broken_pairs_distance_closest(1) < MY_EPSILON;
            if (is_clone && !worst_is_clone)
                || (is_clone == worst_is_clone && pop[i].biased_fitness > worst_fitness)
            {
                worst_fitness = pop[i].biased_fitness;
                worst_is_clone = is_clone;
                worst_pos = i;
            }
        }

        let worst = pop.remove(worst_pos);
        for other in pop.iter_mut() {
            other.remove_proximity(worst.id);
        }
    }

    /// Select an individual via binary tournament without updating fitness.
    pub fn get_binary_tournament_no_update(&self) -> &Individual<'p> {
        let total = self.feasible.len() + self.infeasible.len();
        assert!(total > 0, "Cannot select from empty population");
        let place1 = (self.params.rng.borrow_mut().next_u64(total as u64)) as usize;
        let place2 = (self.params.rng.borrow_mut().next_u64(total as u64)) as usize;

        let indiv1 = if place1 < self.feasible.len() {
            &self.feasible[place1]
        } else {
            &self.infeasible[place1 - self.feasible.len()]
        };
        let indiv2 = if place2 < self.feasible.len() {
            &self.feasible[place2]
        } else {
            &self.infeasible[place2 - self.feasible.len()]
        };

        if indiv1.biased_fitness < indiv2.biased_fitness {
            indiv1
        } else {
            indiv2
        }
    }

    /// Select two distinct parents for crossover via binary tournament.
    pub fn get_non_identical_parents(&mut self) -> (&Individual<'p>, &Individual<'p>) {
        Self::update_biased_fitnesses(self.params, &mut self.feasible);
        Self::update_biased_fitnesses(self.params, &mut self.infeasible);

        let p1 = self.get_binary_tournament_no_update();
        let mut p2 = self.get_binary_tournament_no_update();
        let mut tries = 1;
        while p1.broken_pairs_distance(p2) < MY_EPSILON && tries < 10 {
            p2 = self.get_binary_tournament_no_update();
            tries += 1;
        }
        (p1, p2)
    }
}
