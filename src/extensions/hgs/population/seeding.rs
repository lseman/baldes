use super::*;

impl<'p> Population<'p> {
    pub(crate) fn initialize_seeds(
        &mut self,
        seeds: &[Vec<Vec<usize>>],
        local: &mut HGSLocalSearch<'p>,
        delay: bool,
    ) {
        for seed in seeds {
            let mut individual = Individual::new_empty(self.params);
            for (target, route) in individual
                .chrom_r
                .iter_mut()
                .zip(seed.iter().filter(|r| !r.is_empty()))
            {
                target.clone_from(route);
            }
            individual.chrom_t = individual.chrom_r.iter().flatten().copied().collect();
            individual.evaluate_complete_cost();
            if delay {
                // Keep the supplied witness available on deadline exhaustion,
                // without making it a parent or a feasibility observation yet.
                if self.best_overall.my_cost_sol.penalized_cost
                    > individual.my_cost_sol.penalized_cost
                {
                    *self.best_overall = individual.clone();
                }
                if let Some(routes) = self.collected_routes.as_mut() {
                    routes.extend(individual.chrom_r.iter().filter(|r| !r.is_empty()).cloned());
                }
                self.reserved_seeds.push(individual);
            } else {
                self.add_individual(&individual, true);
                if !self.params.is_time_limit_exceeded() {
                    self.do_hgs_local_search_and_add_individual(&mut individual, local);
                }
            }
        }
    }

    /// Return None while seeds are still reserved (or already inserted).
    /// Insertion replaces one crossover step and happens at most once.
    pub(crate) fn insert_reserved_seeds(
        &mut self,
        non_productive: usize,
        stagnation: usize,
    ) -> Option<bool> {
        let seed_cost = self
            .reserved_seeds
            .iter()
            .map(|i| i.my_cost_sol.distance)
            .min_by(f64::total_cmp)?;
        let caught_up = self
            .get_best_feasible()
            .is_some_and(|i| i.my_cost_sol.distance <= seed_cost);
        let time_trigger = self.params.config.time_limit > 0.
            && self.params.elapsed_secs() >= self.params.config.time_limit * 0.5;
        let generation_trigger = self
            .params
            .config
            .generation_limit
            .is_some_and(|limit| self.nb_generations >= limit / 2);
        if !caught_up
            && !time_trigger
            && !generation_trigger
            && non_productive < (stagnation / 2).max(1)
        {
            return None;
        }
        let improves_pool = self
            .get_best_feasible()
            .is_none_or(|i| seed_cost < i.my_cost_sol.distance);
        if self.params.verbose {
            crate::log_err!(
                "  HGS seed insertion | {} reserved plans | generation {}",
                self.reserved_seeds.len(),
                self.nb_generations
            );
        }
        let mut improved = improves_pool;
        for seed in std::mem::take(&mut self.reserved_seeds) {
            improved |= self.add_individual(&seed, true);
        }
        Some(improved)
    }
}
