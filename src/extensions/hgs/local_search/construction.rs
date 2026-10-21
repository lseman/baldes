use super::*;

impl<'p> HGSLocalSearch<'p> {
    /// Initialize construction heuristic data.
    pub fn initialize_construction(&mut self, nodes_to_insert: &mut Vec<NodeToInsert>) {
        self.empty_routes.clear();
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
            self.nodes[start].prev = end as i32;
            self.nodes[end].next = start as i32;
            self.nodes[start].next = end as i32;
            self.nodes[end].prev = start as i32;

            self.nodes[start].tw_data = depot_tw.clone();
            self.nodes[start].prefix_tw_data = depot_tw.clone();
            self.nodes[start].postfix_tw_data = depot_tw.clone();

            self.nodes[end].cour = 0;
            self.nodes[end].is_depot = true;
            self.nodes[end].route_idx = r as i32;
            self.nodes[end].tw_data = depot_tw.clone();
            self.nodes[end].prefix_tw_data = depot_tw.clone();
            self.nodes[end].postfix_tw_data = depot_tw.clone();

            self.update_route_data(r);
        }

        nodes_to_insert.clear();
        for i in 1..=self.params.nb_clients as usize {
            let dx = self.params.cli[i].coord_x - self.params.cli[0].coord_x;
            let dy = self.params.cli[i].coord_y - self.params.cli[0].coord_y;
            nodes_to_insert.push(NodeToInsert {
                client_idx: i as i32,
                load: self.params.cli[i].demand,
                service_duration: self.params.cli[i].service_duration,
                angle_from_depot: dy.atan2(dx),
                tw_data: self.nodes[i].tw_data.clone(),
            });
        }
    }

    /// Construct individual by sweep heuristic (Vidal 2012).
    pub fn construct_individual_by_sweep(
        &mut self,
        fill_percentage: i32,
        indiv: &mut Individual<'_>,
    ) {
        let mut nodes_to_insert = Vec::new();
        self.initialize_construction(&mut nodes_to_insert);

        nodes_to_insert.sort_by(|a, b| {
            a.angle_from_depot
                .partial_cmp(&b.angle_from_depot)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut node_indices_per_route: Vec<Vec<usize>> = Vec::new();
        let mut current_route_indices = Vec::new();
        let mut current_load = 0;
        let cap_threshold =
            (fill_percentage as f64 * self.params.vehicle_capacity as f64 / 100.0) as i32;

        for (i, node) in nodes_to_insert.iter().enumerate() {
            if current_load > 0
                && current_load + node.load > cap_threshold
                && node_indices_per_route.len() + 1 < self.routes.len()
            {
                node_indices_per_route.push(current_route_indices);
                current_route_indices = Vec::new();
                current_load = 0;
            }
            current_load += node.load;
            current_route_indices.push(i);
        }
        node_indices_per_route.push(current_route_indices);

        for (r, indices) in node_indices_per_route.iter().enumerate() {
            if r >= self.routes.len() {
                break;
            }
            let depot_s = self.depot_start(r);
            let depot_opening = self.nodes[depot_s].tw_data.latest_arrival
                - self.nodes[depot_s].tw_data.earliest_arrival;

            let mut short_tw_indices = Vec::new();
            let mut long_tw_indices = Vec::new();

            for &idx in indices {
                let tw_dur = nodes_to_insert[idx].tw_data.latest_arrival
                    - nodes_to_insert[idx].tw_data.earliest_arrival;
                if tw_dur * 2. > depot_opening {
                    long_tw_indices.push(idx);
                } else {
                    short_tw_indices.push(idx);
                }
            }

            short_tw_indices.sort_by(|&a, &b| {
                nodes_to_insert[a]
                    .tw_data
                    .latest_arrival
                    .total_cmp(&nodes_to_insert[b].tw_data.latest_arrival)
            });

            let mut prev = depot_s;
            for idx in short_tw_indices {
                let to_insert = nodes_to_insert[idx].client_idx as usize;
                self.insert_unlinked_node(to_insert, prev);
                prev = to_insert;
            }
            self.update_route_data(r);

            for idx in long_tw_indices {
                let mut best_cost = f64::MAX;
                let mut best_pred = depot_s;
                let mut curr = depot_s;

                for _ in 0..=self.routes[r].nb_customers {
                    let next = self.nodes[curr].next as usize;
                    let curr_cour = self.nodes[curr].cour as usize;
                    let next_cour = self.nodes[next].cour as usize;
                    let cand_cour = nodes_to_insert[idx].client_idx as usize;

                    let cost = (self.params.time_cost.get(curr_cour, cand_cour)
                        + self.params.time_cost.get(cand_cour, next_cour)
                        - self.params.time_cost.get(curr_cour, next_cour))
                        as f64;

                    if cost < best_cost {
                        best_cost = cost;
                        best_pred = curr;
                    }
                    curr = next;
                }

                let to_insert = nodes_to_insert[idx].client_idx as usize;
                self.insert_unlinked_node(to_insert, best_pred);
                self.update_route_data(r);
            }
        }

        self.export_individual(indiv);
    }

    /// Construct individual with seed order (Furthest or Nearest seed).
    pub fn construct_individual_with_seed_order(
        &mut self,
        tolerated_capacity_violation: i32,
        tolerated_time_warp: i32,
        use_seed_client_furthest_from_depot: bool,
        indiv: &mut Individual<'_>,
    ) {
        let mut nodes_to_insert = Vec::new();
        self.initialize_construction(&mut nodes_to_insert);

        let mut unassigned: std::collections::BTreeSet<usize> =
            (0..self.params.nb_clients as usize).collect();

        for r in 0..self.routes.len() {
            if unassigned.is_empty() {
                break;
            }
            let depot_s = self.depot_start(r);

            if use_seed_client_furthest_from_depot {
                let mut furthest_idx = None;
                let mut max_insertion_cost = -1.0;

                for &idx in &unassigned {
                    let cand_client = nodes_to_insert[idx].client_idx as usize;
                    let cost = (self.params.time_cost.get(0, cand_client)
                        + self.params.time_cost.get(cand_client, 0)
                        - self.params.time_cost.get(0, 0)) as f64;
                    if cost > max_insertion_cost {
                        max_insertion_cost = cost;
                        furthest_idx = Some(idx);
                    }
                }

                if let Some(f_idx) = furthest_idx {
                    let seed_node = nodes_to_insert[f_idx].client_idx as usize;
                    self.insert_unlinked_node(seed_node, depot_s);
                    self.update_route_data(r);
                    unassigned.remove(&f_idx);
                }
            }

            let mut node_inserted = true;
            while node_inserted {
                node_inserted = false;
                let mut best_insertion_cost = f64::MAX;
                let mut best_pred = depot_s;
                let mut best_node_idx = None;

                for &idx in &unassigned {
                    if self.routes[r].load + nodes_to_insert[idx].load
                        > self.params.vehicle_capacity + tolerated_capacity_violation
                    {
                        continue;
                    }

                    let mut prev = depot_s;
                    for _ in 0..=self.routes[r].nb_customers {
                        let next = self.nodes[prev].next as usize;
                        let cand = nodes_to_insert[idx].client_idx as usize;

                        let merged_prefix_cand = TimeWindowData::merge(
                            &self.nodes[prev].prefix_tw_data,
                            &nodes_to_insert[idx].tw_data,
                            self.params,
                        );
                        let merged_all = TimeWindowData::merge(
                            &merged_prefix_cand,
                            &self.nodes[next].postfix_tw_data,
                            self.params,
                        );

                        if merged_all.time_warp <= f64::from(tolerated_time_warp) {
                            let prev_cour = self.nodes[prev].cour as usize;
                            let next_cour = self.nodes[next].cour as usize;
                            let cost = (self.params.time_cost.get(prev_cour, cand)
                                + self.params.time_cost.get(cand, next_cour)
                                - self.params.time_cost.get(prev_cour, next_cour))
                                as f64;

                            if cost < best_insertion_cost {
                                best_insertion_cost = cost;
                                best_pred = prev;
                                best_node_idx = Some(idx);
                            }
                        }
                        prev = next;
                    }
                }

                if let Some(best_idx) = best_node_idx {
                    let node_to_ins = nodes_to_insert[best_idx].client_idx as usize;
                    self.insert_unlinked_node(node_to_ins, best_pred);
                    self.update_route_data(r);
                    unassigned.remove(&best_idx);
                    node_inserted = true;
                }
            }
        }

        if !unassigned.is_empty() {
            let last_r = self.routes.len() - 1;
            let depot_e = self.depot_end(last_r);
            // HGS inserts each one right after the same node, reversing them.
            let prev = self.nodes[depot_e].prev as usize;
            for idx in unassigned {
                let to_ins = nodes_to_insert[idx].client_idx as usize;
                self.insert_unlinked_node(to_ins, prev);
            }
            self.update_route_data(last_r);
        }

        self.export_individual(indiv);
    }
}
