use super::*;

impl<'p> HGSLocalSearch<'p> {
    #[inline(always)]
    pub fn insert_node(&mut self, to_insert: usize, insertion_point: usize) {
        let prev = self.nodes[to_insert].prev as usize;
        let next = self.nodes[to_insert].next as usize;
        let ins_next = self.nodes[insertion_point].next as usize;

        self.nodes[prev].next = next as i32;
        self.nodes[next].prev = prev as i32;

        self.nodes[ins_next].prev = to_insert as i32;
        self.nodes[to_insert].prev = insertion_point as i32;
        self.nodes[to_insert].next = ins_next as i32;
        self.nodes[insertion_point].next = to_insert as i32;

        self.nodes[to_insert].route_idx = self.nodes[insertion_point].route_idx;
    }

    #[inline(always)]
    pub(super) fn insert_unlinked_node(&mut self, to_insert: usize, insertion_point: usize) {
        let ins_next = self.nodes[insertion_point].next as usize;

        self.nodes[ins_next].prev = to_insert as i32;
        self.nodes[to_insert].prev = insertion_point as i32;
        self.nodes[to_insert].next = ins_next as i32;
        self.nodes[insertion_point].next = to_insert as i32;

        self.nodes[to_insert].route_idx = self.nodes[insertion_point].route_idx;
    }

    #[inline(always)]
    pub fn swap_node(&mut self, u: usize, v: usize) {
        let u_pred = self.nodes[u].prev as usize;
        let u_suiv = self.nodes[u].next as usize;
        let v_pred = self.nodes[v].prev as usize;
        let v_suiv = self.nodes[v].next as usize;
        let route_u = self.nodes[u].route_idx;
        let route_v = self.nodes[v].route_idx;

        if u_suiv == v {
            self.nodes[u_pred].next = v as i32;
            self.nodes[v].prev = u_pred as i32;
            self.nodes[v].next = u as i32;
            self.nodes[u].prev = v as i32;
            self.nodes[u].next = v_suiv as i32;
            self.nodes[v_suiv].prev = u as i32;
        } else if v_suiv == u {
            self.nodes[v_pred].next = u as i32;
            self.nodes[u].prev = v_pred as i32;
            self.nodes[u].next = v as i32;
            self.nodes[v].prev = u as i32;
            self.nodes[v].next = u_suiv as i32;
            self.nodes[u_suiv].prev = v as i32;
        } else {
            self.nodes[u_pred].next = v as i32;
            self.nodes[u_suiv].prev = v as i32;
            self.nodes[v_pred].next = u as i32;
            self.nodes[v_suiv].prev = u as i32;

            self.nodes[u].prev = v_pred as i32;
            self.nodes[u].next = v_suiv as i32;
            self.nodes[v].prev = u_pred as i32;
            self.nodes[v].next = u_suiv as i32;
        }

        self.nodes[u].route_idx = route_v;
        self.nodes[v].route_idx = route_u;
    }

    /// Update preprocessed route data (duration, load, TW, penalties, barycenter).
    pub fn update_route_data(&mut self, route_idx: usize) {
        let depot_s = self.depot_start(route_idx);
        let depot_e = self.depot_end(route_idx);

        let mut myplace = 0i32;
        let mut myload = 0i32;
        let mut mytime = 0_f64;
        let mut my_reversal_distance = 0_f64;
        // HGS accumulates the scaled coordinates in ints, truncating each sum.
        let mut cumulated_x = 0i32;
        let mut cumulated_y = 0i32;

        self.nodes[depot_s].position = 0;
        self.nodes[depot_s].cumulated_load = 0;
        self.nodes[depot_s].cumulated_time = 0.;
        self.nodes[depot_s].cumulated_reversal_distance = 0.;

        let mut mynode = depot_s;
        let mut first_it = true;
        let mut seed_tw = TimeWindowData::default();
        let mut seed_node: Option<usize> = None;

        while mynode != depot_e {
            let next_node = self.nodes[mynode].next as usize;
            myplace += 1;
            self.nodes[next_node].position = myplace;
            let cour = self.nodes[next_node].cour as usize;
            let prev_cour = self.nodes[mynode].cour as usize;

            myload += self.params.cli[cour].demand;
            let edge_cost = self.params.time_cost.get(prev_cour, cour);
            let edge_reverse = self.params.time_cost.get(cour, prev_cour);
            mytime += edge_cost + self.params.cli[cour].service_duration;
            my_reversal_distance += edge_reverse - edge_cost;

            self.nodes[next_node].cumulated_load = myload;
            self.nodes[next_node].cumulated_time = mytime;
            self.nodes[next_node].cumulated_reversal_distance = my_reversal_distance;
            self.nodes[next_node].prefix_tw_data = TimeWindowData::merge(
                &self.nodes[mynode].prefix_tw_data,
                &self.nodes[next_node].tw_data,
                self.params,
            );
            self.nodes[next_node].is_seed = false;
            self.nodes[next_node].next_seed = -1;

            if !self.nodes[next_node].is_depot {
                cumulated_x = (cumulated_x as f64 + self.params.cli[cour].coord_x) as i32;
                cumulated_y = (cumulated_y as f64 + self.params.cli[cour].coord_y) as i32;
                if first_it {
                    self.routes[route_idx]
                        .sector
                        .initialize(self.params.cli[cour].polar_angle);
                } else {
                    self.routes[route_idx]
                        .sector
                        .extend(self.params.cli[cour].polar_angle);
                }
                // Every fourth node stores the merged data up to the next one.
                match myplace % 4 {
                    0 => {
                        if let Some(seed) = seed_node {
                            self.nodes[seed].is_seed = true;
                            self.nodes[seed].to_next_seed_tw_data = TimeWindowData::merge(
                                &seed_tw,
                                &self.nodes[next_node].tw_data,
                                self.params,
                            );
                            self.nodes[seed].next_seed = next_node as i32;
                        }
                        seed_node = Some(next_node);
                    }
                    1 => seed_tw = self.nodes[next_node].tw_data.clone(),
                    _ => {
                        seed_tw = TimeWindowData::merge(
                            &seed_tw,
                            &self.nodes[next_node].tw_data,
                            self.params,
                        )
                    }
                }
            }

            first_it = false;
            mynode = next_node;
        }

        let route = &mut self.routes[route_idx];
        route.duration = mytime;
        route.load = myload;
        route.tw_data = self.nodes[depot_e].prefix_tw_data.clone();
        route.nb_customers = myplace - 1;
        route.reversal_distance = my_reversal_distance;
        route.when_last_modified = self.nb_moves;
        route.is_delta_removel_tw_outdated = true;
        self.routes[route_idx].penalty = self.penalty_excess_load(myload)
            + self.penalty_time_windows(&self.routes[route_idx].tw_data);

        // Postfix TW data in reverse direction
        mynode = depot_e;
        while mynode != depot_s {
            let prev_node = self.nodes[mynode].prev as usize;
            self.nodes[prev_node].postfix_tw_data = TimeWindowData::merge(
                &self.nodes[prev_node].tw_data,
                &self.nodes[mynode].postfix_tw_data,
                self.params,
            );
            mynode = prev_node;
        }

        let route = &mut self.routes[route_idx];
        if route.nb_customers == 0 {
            route.polar_angle_barycenter = 1e30;
            self.empty_routes.insert(route_idx);
        } else {
            let inv_n = 1.0 / route.nb_customers as f64;
            let diff_x = cumulated_x as f64 * inv_n - self.params.cli[0].coord_x;
            let diff_y = cumulated_y as f64 * inv_n - self.params.cli[0].coord_y;
            route.polar_angle_barycenter = diff_y.atan2(diff_x);

            // Enforce a minimum circle sector size.
            if self.params.min_circle_sector_size > 0 {
                let sector_size =
                    CircleSector::positive_mod(route.sector.end as i32 - route.sector.start as i32)
                        as i32;
                let grow_by = (self.params.min_circle_sector_size as i32 - sector_size + 1) / 2;
                if grow_by > 0 {
                    let (start, end) = (route.sector.start as i32, route.sector.end as i32);
                    route
                        .sector
                        .extend(CircleSector::positive_mod(start - grow_by));
                    route
                        .sector
                        .extend(CircleSector::positive_mod(end + grow_by));
                }
            }
            self.empty_routes.remove(&route_idx);
        }
    }

    /// Time window data of the route segment from `u` to `v`, both included.
    pub fn get_route_segment_tw_data(&self, u: usize, v: usize) -> TimeWindowData {
        if self.nodes[u].is_depot {
            return self.nodes[v].prefix_tw_data.clone();
        }
        if self.nodes[v].is_depot {
            return self.nodes[u].postfix_tw_data.clone();
        }

        let target = self.nodes[v].position;
        let mut tw_data = self.nodes[u].tw_data.clone();
        let mut curr = u;
        while curr != v {
            let node = &self.nodes[curr];
            if node.is_seed && node.position + 4 <= target {
                tw_data = TimeWindowData::merge(&tw_data, &node.to_next_seed_tw_data, self.params);
                curr = node.next_seed as usize;
            } else {
                curr = node.next as usize;
                tw_data = TimeWindowData::merge(&tw_data, &self.nodes[curr].tw_data, self.params);
            }
        }
        tw_data
    }

    /// Load an individual into the local search representation.
    pub fn load_individual(&mut self, indiv: &Individual<'_>) {
        self.empty_routes.clear();
        self.nb_moves = 0;
        debug_assert_eq!(
            indiv.chrom_r.iter().map(Vec::len).sum::<usize>(),
            self.params.nb_clients as usize,
            "every client must be routed"
        );

        let depot_tw = TimeWindowData {
            first_node_index: 0,
            last_node_index: 0,
            duration: 0.,
            time_warp: 0.,
            earliest_arrival: self.params.cli[0].earliest_arrival,
            latest_arrival: self.params.cli[0].latest_arrival,
            latest_release_time: self.params.cli[0].release_time,
        };

        for i in 1..=self.params.nb_clients as usize {
            self.nodes[i].cour = i as i32;
            self.nodes[i].is_depot = false;
            self.nodes[i].prev = 0;
            self.nodes[i].next = 0;
            self.nodes[i].route_idx = 0;
            self.nodes[i].when_last_tested_ri = -1;
            self.nodes[i].tw_data = TimeWindowData {
                first_node_index: i as i32,
                last_node_index: i as i32,
                duration: self.params.cli[i].service_duration,
                time_warp: 0.,
                earliest_arrival: self.params.cli[i].earliest_arrival,
                latest_arrival: self.params.cli[i].latest_arrival,
                latest_release_time: self.params.cli[i].release_time,
            };
        }

        for r in 0..self.params.nb_vehicles as usize {
            let start = self.depot_start(r);
            let end = self.depot_end(r);

            self.nodes[start].cour = 0;
            self.nodes[start].is_depot = true;
            self.nodes[start].route_idx = r as i32;
            self.nodes[start].tw_data = depot_tw.clone();
            self.nodes[start].prefix_tw_data = depot_tw.clone();
            self.nodes[start].postfix_tw_data = depot_tw.clone();

            self.nodes[end].cour = 0;
            self.nodes[end].is_depot = true;
            self.nodes[end].route_idx = r as i32;
            self.nodes[end].tw_data = depot_tw.clone();
            self.nodes[end].prefix_tw_data = depot_tw.clone();
            self.nodes[end].postfix_tw_data = depot_tw.clone();

            self.routes[r].cour = r as i32;

            let route = &indiv.chrom_r[r];
            if !route.is_empty() {
                let first = route[0];
                self.nodes[first].route_idx = r as i32;
                self.nodes[first].prev = start as i32;
                self.nodes[start].next = first as i32;

                let mut pred = first;
                for i in 1..route.len() {
                    let curr = route[i];
                    self.nodes[curr].route_idx = r as i32;
                    self.nodes[curr].prev = pred as i32;
                    self.nodes[pred].next = curr as i32;
                    pred = curr;
                }

                let last = pred;
                self.nodes[last].next = end as i32;
                self.nodes[end].prev = last as i32;
            } else {
                self.nodes[start].next = end as i32;
                self.nodes[end].prev = start as i32;
            }

            self.nodes[start].prev = end as i32;
            self.nodes[end].next = start as i32;

            self.update_route_data(r);
            self.routes[r].when_last_tested_large_nb = -1;
            self.best_insert_initialized_for_route[r] = false;
        }
    }

    /// Export the local search solution back to the individual.
    pub fn export_individual(&mut self, indiv: &mut Individual<'_>) {
        let mut route_polar_angles: Vec<(SortableF64, usize)> = Vec::new();
        for r in 0..self.params.nb_vehicles as usize {
            route_polar_angles.push((SortableF64(self.routes[r].polar_angle_barycenter), r));
        }
        route_polar_angles.sort();

        let mut pos = 0;
        for (r, &(_, old_r)) in route_polar_angles.iter().enumerate() {
            indiv.chrom_r[r].clear();
            let start = self.depot_start(old_r);
            let mut curr = self.nodes[start].next as usize;

            while !self.nodes[curr].is_depot {
                let cust = self.nodes[curr].cour as usize;
                if pos < indiv.chrom_t.len() {
                    indiv.chrom_t[pos] = cust;
                } else {
                    indiv.chrom_t.push(cust);
                }
                indiv.chrom_r[r].push(cust);
                pos += 1;
                curr = self.nodes[curr].next as usize;
            }
        }

        indiv.evaluate_complete_cost();
    }

    /// Clients of a route in order.
    pub(super) fn route_clients(&self, r: usize) -> Vec<usize> {
        let mut clients = Vec::with_capacity(self.routes[r].nb_customers.max(0) as usize);
        let mut node = self.nodes[self.depot_start(r)].next as usize;
        while !self.nodes[node].is_depot {
            clients.push(node);
            node = self.nodes[node].next as usize;
        }
        clients
    }
}
