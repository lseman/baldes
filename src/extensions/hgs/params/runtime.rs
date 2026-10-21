use super::*;

impl Params {
    /// Set the seed of the random number generator.
    pub fn set_seed(&mut self, seed: i32) {
        self.config.seed = seed;
        self.rng = std::cell::RefCell::new(Xoroshiro128Plus::new(seed));
    }

    /// Restart the clock used by the time limit.
    pub fn reset_clock(&mut self) {
        self.start_time = Instant::now();
    }

    /// Elapsed seconds since creation.
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Whether the time limit has been exceeded.
    pub fn is_time_limit_exceeded(&self) -> bool {
        if self.config.time_limit <= 0.0 {
            return false;
        }
        self.elapsed_secs() >= self.config.time_limit
    }

    /// Update adaptive penalty coefficients based on solution feasibility history.
    ///
    /// This implements the penalty adaptation scheme from Vidal 2012.
    pub fn update_penalties(
        &mut self,
        num_feasible: usize,
        num_infeasible: usize,
        best_capacity_excess: i32,
        best_time_warp: f64,
    ) {
        let total = num_feasible + num_infeasible;
        if total == 0 {
            return;
        }

        let ratio_feasible = num_feasible as f64 / total as f64;

        // Adapt capacity penalty
        if best_capacity_excess > 0 {
            self.penalty_capacity.set(self.penalty_capacity.get() * 1.5);
        } else if best_capacity_excess == 0 && ratio_feasible > self.config.target_feasible {
            self.penalty_capacity.set(self.penalty_capacity.get() / 1.5);
        }

        // Adapt time warp penalty
        if best_time_warp > MY_EPSILON {
            self.penalty_time_warp
                .set(self.penalty_time_warp.get() * 1.5);
        } else if best_time_warp <= MY_EPSILON && ratio_feasible > self.config.target_feasible {
            self.penalty_time_warp
                .set(self.penalty_time_warp.get() / 1.5);
        }

        // Ensure penalties stay in reasonable bounds
        self.penalty_capacity
            .set(self.penalty_capacity.get().max(0.1).min(1000.0));
        self.penalty_time_warp
            .set(self.penalty_time_warp.get().max(0.1).min(1000.0));

        // Penalty booster: if no feasible solutions, increase penalties faster
        if num_feasible == 0 && self.config.penalty_booster > 0.0 {
            self.penalty_capacity
                .set(self.penalty_capacity.get() * self.config.penalty_booster);
            self.penalty_time_warp
                .set(self.penalty_time_warp.get() * self.config.penalty_booster);
        }
    }

    /// Format as a human-readable string for logging.
    pub fn summary(&self) -> String {
        format!(
            "Params{{{} clients, {} vehicles, capacity={}, total_demand={}, \
             penalty_cap={:.2}, penalty_tw={:.2}}}",
            self.nb_clients,
            self.nb_vehicles,
            self.vehicle_capacity,
            self.total_demand,
            self.penalty_capacity.get(),
            self.penalty_time_warp.get(),
        )
    }
}
