use super::*;

impl<'p> Individual<'p> {
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
                cache[i] = self
                    .params
                    .time_cost
                    .get(self.chrom_r[r][i - 1], self.chrom_r[r][i]);
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
            if matches!(
                self.params.problem_type,
                ProblemType::Vrptw | ProblemType::Evrp
            ) {
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
    pub(super) fn adjust_to_time_window(
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
