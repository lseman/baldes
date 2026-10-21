use super::*;

impl<'p> Population<'p> {
    /// Create a new empty population.
    pub fn new(params: &'p Params) -> Self {
        let mut last_feas_load = VecDeque::with_capacity(100);
        let mut last_feas_tw = VecDeque::with_capacity(100);
        for _ in 0..100 {
            last_feas_load.push_back(true);
            last_feas_tw.push_back(true);
        }

        let best_restart = Box::new(Individual::new_empty(params));
        let best_overall = Box::new(Individual::new_empty(params));

        Population {
            params,
            collected_routes: None,
            feasible: Vec::new(),
            infeasible: Vec::new(),
            last_feas_load,
            last_feas_tw,
            search_progress: Vec::new(),
            best_restart,
            best_overall,
            mdm_elite: Vec::new(),
            mdm_elite_updated: false,
            mdm_elite_max_non_updating_restarts: 0,
            mdm_elite_non_updating_restarts: 0,
            mdm_patterns: Vec::new(),
            mdm_next_pattern: 0,
            nb_generations: 0,
            minimum_population_size: params.config.minimum_population_size,
        }
    }

    /// Manage penalty parameters based on feasibility history.
    pub fn manage_penalties(&mut self) {
        let count_load_feas = self.last_feas_load.iter().filter(|&&b| b).count();
        let fraction_load = count_load_feas as f64 / self.last_feas_load.len() as f64;
        let target = self.params.config.target_feasible;

        let cur_cap = self.params.penalty_capacity.get();
        if fraction_load <= 0.01 && self.params.config.penalty_booster > 0.0 && cur_cap < 100000.0 {
            self.params
                .penalty_capacity
                .set((cur_cap * self.params.config.penalty_booster).min(100000.0));
        } else if fraction_load < target - 0.05 && cur_cap < 100000.0 {
            self.params
                .penalty_capacity
                .set((cur_cap * 1.2).min(100000.0));
        } else if fraction_load > target + 0.05 && cur_cap > 0.1 {
            self.params.penalty_capacity.set((cur_cap * 0.85).max(0.1));
        }

        let count_tw_feas = self.last_feas_tw.iter().filter(|&&b| b).count();
        let fraction_tw = count_tw_feas as f64 / self.last_feas_tw.len() as f64;

        let cur_tw = self.params.penalty_time_warp.get();
        if fraction_tw <= 0.01 && self.params.config.penalty_booster > 0.0 && cur_tw < 100000.0 {
            self.params
                .penalty_time_warp
                .set((cur_tw * self.params.config.penalty_booster).min(100000.0));
        } else if fraction_tw < target - 0.05 && cur_tw < 100000.0 {
            self.params
                .penalty_time_warp
                .set((cur_tw * 1.2).min(100000.0));
        } else if fraction_tw > target + 0.05 && cur_tw > 0.1 {
            self.params.penalty_time_warp.set((cur_tw * 0.85).max(0.1));
        }

        let pen_cap = self.params.penalty_capacity.get();
        let pen_tw = self.params.penalty_time_warp.get();
        for indiv in self.infeasible.iter_mut() {
            indiv.my_cost_sol.penalized_cost = indiv.my_cost_sol.distance as f64
                + pen_cap * indiv.my_cost_sol.capacity_excess as f64
                + pen_tw * indiv.my_cost_sol.time_warp as f64;
        }

        // Reorder with HGS's bubble sort, which swaps only beyond MY_EPSILON.
        let n = self.infeasible.len();
        for i in 0..n {
            for j in 0..n - i - 1 {
                if self.infeasible[j].my_cost_sol.penalized_cost
                    > self.infeasible[j + 1].my_cost_sol.penalized_cost + MY_EPSILON
                {
                    self.infeasible.swap(j, j + 1);
                }
            }
        }
    }

    /// Return the best feasible individual if any exists.
    pub fn get_best_feasible(&self) -> Option<&Individual<'p>> {
        self.feasible.first().map(|b| b.as_ref())
    }

    /// Return the best infeasible individual if any exists.
    pub fn get_best_infeasible(&self) -> Option<&Individual<'p>> {
        self.infeasible.first().map(|b| b.as_ref())
    }

    /// Return the best overall individual found.
    pub fn get_best_found(&self) -> Option<&Individual<'p>> {
        if self.best_overall.my_cost_sol.penalized_cost < 1e29 {
            Some(&self.best_overall)
        } else if let Some(feas) = self.get_best_feasible() {
            Some(feas)
        } else {
            self.get_best_infeasible()
        }
    }

    /// Print one aligned solver progress row. Negative iterations denote initialization.
    pub fn print_state(&self, nb_iter: i32, nb_iter_non_prod: i32) {
        if !self.params.verbose {
            return;
        }
        let cost = |value: Option<f64>| {
            value
                .filter(|v| v.is_finite())
                .map_or_else(|| "--".into(), |v| format!("{:.5}", v / 10.))
        };
        let best = self.get_best_found().filter(|i| i.is_feasible);
        let iteration = if nb_iter < 0 {
            "init".into()
        } else {
            nb_iter.to_string()
        };
        let stall = if nb_iter_non_prod < 0 {
            "--".into()
        } else {
            nb_iter_non_prod.to_string()
        };
        crate::log_err!(
            "{:>6} {:>7} {:>13} {:>6} {:>6} {:>6} {:>13} {:>13} {:>8.2}",
            iteration,
            stall,
            cost(best.map(|i| f64::from(i.my_cost_sol.distance))),
            best.map_or_else(
                || "--".into(),
                |i| i
                    .chrom_r
                    .iter()
                    .filter(|r| !r.is_empty())
                    .count()
                    .to_string()
            ),
            self.feasible.len(),
            self.infeasible.len(),
            cost(self.feasible.first().map(|i| i.my_cost_sol.penalized_cost)),
            cost(
                self.infeasible
                    .first()
                    .map(|i| i.my_cost_sol.penalized_cost)
            ),
            self.params.elapsed_secs(),
        );
    }

    /// Average distance to the closest individuals over the best mu of a
    /// pool, or -1 when the pool is empty.
    pub fn get_diversity(&self, pop: &SubPop<'p>) -> f64 {
        let size = (self.minimum_population_size.max(0) as usize).min(pop.len());
        if size == 0 {
            return -1.0;
        }
        pop[..size]
            .iter()
            .map(|indiv| indiv.average_broken_pairs_distance_closest(size))
            .sum::<f64>()
            / size as f64
    }

    /// Average penalized cost over the best mu of a pool, or -1 when empty.
    pub fn get_average_cost(&self, pop: &SubPop<'p>) -> f64 {
        let size = (self.minimum_population_size.max(0) as usize).min(pop.len());
        if size == 0 {
            return -1.0;
        }
        pop[..size]
            .iter()
            .map(|indiv| indiv.my_cost_sol.penalized_cost)
            .sum::<f64>()
            / size as f64
    }
}
