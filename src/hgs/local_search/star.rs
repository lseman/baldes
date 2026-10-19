use super::*;

impl<'p> HGSLocalSearch<'p> {
    /// RELOCATE*: the best move of one client of `r_u` after a client of `r_v`.
    pub(super) fn relocate_star(&mut self, r_u: usize, r_v: usize) -> bool {
        let mut best_cost = 0.;
        let mut best: Option<(usize, usize)> = None;
        for u in self.route_clients(r_u) {
            let (prev, next) = (self.nodes[u].prev as usize, self.nodes[u].next as usize);
            let route_u_tw = TimeWindowData::merge(
                &self.nodes[prev].prefix_tw_data,
                &self.nodes[next].postfix_tw_data,
                self.params,
            );
            let load_u = self.params.cli[self.nodes[u].cour as usize].demand;
            let cost_supp_u = (self.cost(prev, next) - self.cost(prev, u) - self.cost(u, next))
                as f64
                + self.penalty_excess_load(self.routes[r_u].load - load_u)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
            for v in self.route_clients(r_v) {
                let v_next = self.nodes[v].next as usize;
                let route_v_tw = self.merge_all(&[
                    &self.nodes[v].prefix_tw_data,
                    &self.nodes[u].tw_data,
                    &self.nodes[v_next].postfix_tw_data,
                ]);
                let cost_supp_v = (self.cost(v, u) + self.cost(u, v_next) - self.cost(v, v_next))
                    as f64
                    + self.penalty_excess_load(self.routes[r_v].load + load_u)
                    + self.penalty_time_windows(&route_v_tw)
                    - self.routes[r_v].penalty;
                if cost_supp_u + cost_supp_v < best_cost - MY_EPSILON {
                    best_cost = cost_supp_u + cost_supp_v;
                    best = Some((u, v));
                }
            }
        }
        let Some((u, v)) = best else { return false };
        self.insert_node(u, v);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        self.update_route_data(r_v);
        true
    }

    /// Insertion costs of each client of `r1` into `r2` (SWAP* preprocessing),
    /// with time-warp penalty deltas when `with_tw`.
    pub(super) fn preprocess_insertions(&mut self, r1: usize, r2: usize, with_tw: bool) {
        let depot2 = self.depot_start(r2);
        let first2 = self.nodes[depot2].next as usize;
        let route2_tw = self.routes[r2].tw_data.clone();
        let tw_outdated = self.routes[r1].is_delta_removel_tw_outdated;
        let route1_tw = self.routes[r1].tw_data.clone();
        for u in self.route_clients(r1) {
            let (prev, next) = (self.nodes[u].prev as usize, self.nodes[u].next as usize);
            let removal = self.cost(prev, next) - self.cost(prev, u) - self.cost(u, next);
            if !with_tw {
                self.nodes[u].delta_removal = removal;
            } else if tw_outdated {
                let tw = TimeWindowData::merge(
                    &self.nodes[prev].prefix_tw_data,
                    &self.nodes[next].postfix_tw_data,
                    self.params,
                );
                // Keep fractional time-window penalties in the insertion cost.
                self.nodes[u].delta_removal_tw =
                    removal + self.delta_penalty_time_windows(&tw, &route1_tw);
            }
            let cour = self.nodes[u].cour as usize;
            let when = if with_tw {
                self.best_insert_client_tw[r2][cour].when_last_calculated
            } else {
                self.best_insert_client[r2][cour].when_last_calculated
            };
            if self.routes[r2].when_last_modified <= when {
                continue;
            }
            let mut option = ThreeBestInsert::new();
            option.when_last_calculated = self.nb_moves;
            let insertion = |ls: &Self, v: usize| {
                let v_next = ls.nodes[v].next as usize;
                let mut delta = ls.cost(v, u) + ls.cost(u, v_next) - ls.cost(v, v_next);
                if with_tw {
                    let tw = ls.merge_all(&[
                        &ls.nodes[v].prefix_tw_data,
                        &ls.nodes[u].tw_data,
                        &ls.nodes[v_next].postfix_tw_data,
                    ]);
                    delta += ls.delta_penalty_time_windows(&tw, &route2_tw);
                }
                delta
            };
            option.best_cost[0] = insertion(self, depot2);
            option.best_location[0] = depot2 as i32;
            let mut v = first2;
            while !self.nodes[v].is_depot {
                option.compare_and_add(insertion(self, v), v as i32);
                v = self.nodes[v].next as usize;
            }
            if with_tw {
                self.best_insert_client_tw[r2][cour] = option;
            } else {
                self.best_insert_client[r2][cour] = option;
            }
        }
        if with_tw {
            self.routes[r1].is_delta_removel_tw_outdated = false;
        }
    }

    /// Cheapest insertion of `u` into the route of `v` once `v` is removed:
    /// one of the three preprocessed positions not adjacent to `v`, or `v`'s
    /// own place. Returns the cost and the node to insert after.
    pub(super) fn cheapest_insert_simult_removal(
        &self,
        u: usize,
        v: usize,
        with_tw: bool,
    ) -> (f64, usize) {
        let r_v = self.nodes[v].route_idx as usize;
        let cour = self.nodes[u].cour as usize;
        let best = if with_tw {
            &self.best_insert_client_tw[r_v][cour]
        } else {
            &self.best_insert_client[r_v][cour]
        };
        let not_adjacent = |p: i32| p as usize != v && self.nodes[p as usize].next as usize != v;
        let mut position = best.best_location[0];
        let mut cost = best.best_cost[0];
        let mut found = not_adjacent(position);
        if !found && best.best_location[1] >= 0 {
            position = best.best_location[1];
            cost = best.best_cost[1];
            found = not_adjacent(position);
            if !found && best.best_location[2] >= 0 {
                position = best.best_location[2];
                cost = best.best_cost[2];
                found = true;
            }
        }
        let (v_prev, v_next) = (self.nodes[v].prev as usize, self.nodes[v].next as usize);
        let mut delta = self.cost(v_prev, u) + self.cost(u, v_next) - self.cost(v_prev, v_next);
        if with_tw {
            let tw = self.merge_all(&[
                &self.nodes[v_prev].prefix_tw_data,
                &self.nodes[u].tw_data,
                &self.nodes[v_next].postfix_tw_data,
            ]);
            delta += self.delta_penalty_time_windows(&tw, &self.routes[r_v].tw_data);
        }
        if !found || delta < cost {
            (delta, v_prev)
        } else {
            (cost, position as usize)
        }
    }

    /// Route time window data after replacing `removed` by `inserted` placed
    /// after `position`, all in the route of `removed`.
    pub(super) fn swap_star_route_tw(
        &self,
        removed: usize,
        inserted: usize,
        position: usize,
    ) -> TimeWindowData {
        let (prev, next) = (
            self.nodes[removed].prev as usize,
            self.nodes[removed].next as usize,
        );
        let position_next = self.nodes[position].next as usize;
        if self.nodes[position].position == self.nodes[removed].position - 1 {
            self.merge_all(&[
                &self.nodes[position].prefix_tw_data,
                &self.nodes[inserted].tw_data,
                &self.nodes[next].postfix_tw_data,
            ])
        } else if self.nodes[position].position < self.nodes[removed].position {
            let segment = self.get_route_segment_tw_data(position_next, prev);
            self.merge_all(&[
                &self.nodes[position].prefix_tw_data,
                &self.nodes[inserted].tw_data,
                &segment,
                &self.nodes[next].postfix_tw_data,
            ])
        } else {
            let segment = self.get_route_segment_tw_data(next, position);
            self.merge_all(&[
                &self.nodes[prev].prefix_tw_data,
                &segment,
                &self.nodes[inserted].tw_data,
                &self.nodes[position_next].postfix_tw_data,
            ])
        }
    }

    /// Distance change of removing `removed` and inserting `inserted` after
    /// `position` in the same route.
    pub(super) fn swap_star_route_cost(
        &self,
        removed: usize,
        inserted: usize,
        position: usize,
    ) -> f64 {
        let (prev, next) = (
            self.nodes[removed].prev as usize,
            self.nodes[removed].next as usize,
        );
        let mut cost =
            self.cost(position, inserted) - self.cost(prev, removed) - self.cost(removed, next);
        if position == prev {
            cost += self.cost(inserted, next);
        } else {
            let position_next = self.nodes[position].next as usize;
            cost += self.cost(inserted, position_next) + self.cost(prev, next)
                - self.cost(position, position_next);
        }
        cost as f64
    }

    /// SWAP* (Vidal 2022): exchange a client of `r_u` with one of `r_v`,
    /// each reinserted at its best position, if it improves the penalized cost.
    pub(super) fn swap_star(&mut self, r_u: usize, r_v: usize, with_tw: bool) -> bool {
        for r in [r_u, r_v] {
            if !self.best_insert_initialized_for_route[r] {
                self.best_insert_initialized_for_route[r] = true;
                for option in self.best_insert_client[r]
                    .iter_mut()
                    .chain(self.best_insert_client_tw[r].iter_mut())
                {
                    option.when_last_calculated = -1;
                }
            }
        }
        self.preprocess_insertions(r_u, r_v, with_tw);
        self.preprocess_insertions(r_v, r_u, with_tw);

        let (load_u, load_v) = (self.routes[r_u].load, self.routes[r_v].load);
        let current_load_pen = self.penalty_excess_load(load_u) + self.penalty_excess_load(load_v);
        let clients_v = self.route_clients(r_v);
        let mut best = SwapStarElement {
            move_cost: 1e30,
            load_pen_u: 1e30,
            load_pen_v: 1e30,
            u: 0,
            best_position_u: None,
            v: 0,
            best_position_v: None,
        };
        for u in self.route_clients(r_u) {
            let demand_u = self.params.cli[self.nodes[u].cour as usize].demand;
            for &v in &clients_v {
                let demand_v = self.params.cli[self.nodes[v].cour as usize].demand;
                let load_pen_u = self.penalty_excess_load(load_u + demand_v - demand_u);
                let load_pen_v = self.penalty_excess_load(load_v + demand_u - demand_v);
                let delta_load_pen = load_pen_u + load_pen_v - current_load_pen;
                let delta_removal = if with_tw {
                    self.nodes[u].delta_removal_tw + self.nodes[v].delta_removal_tw
                } else {
                    self.nodes[u].delta_removal + self.nodes[v].delta_removal
                };
                // Insertion costs are nonnegative bounds: skip hopeless pairs.
                if delta_load_pen + delta_removal as f64 > 0. {
                    continue;
                }
                let (extra_v, position_u) = self.cheapest_insert_simult_removal(u, v, with_tw);
                let (extra_u, position_v) = self.cheapest_insert_simult_removal(v, u, with_tw);
                let move_cost = delta_load_pen + (delta_removal + extra_u + extra_v) as f64;
                if move_cost < best.move_cost {
                    best = SwapStarElement {
                        move_cost,
                        load_pen_u,
                        load_pen_v,
                        u,
                        best_position_u: Some(position_u),
                        v,
                        best_position_v: Some(position_v),
                    };
                }
            }
        }
        let (Some(position_u), Some(position_v)) = (best.best_position_u, best.best_position_v)
        else {
            return false;
        };
        let (u, v) = (best.u, best.v);

        // Exact cost, including time warp.
        let route_u_tw = self.swap_star_route_tw(u, v, position_v);
        let route_v_tw = self.swap_star_route_tw(v, u, position_u);
        let cost_supp_u = self.swap_star_route_cost(u, v, position_v)
            + best.load_pen_u
            + self.penalty_time_windows(&route_u_tw)
            - self.routes[r_u].penalty;
        let cost_supp_v = self.swap_star_route_cost(v, u, position_u)
            + best.load_pen_v
            + self.penalty_time_windows(&route_v_tw)
            - self.routes[r_v].penalty;
        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        self.insert_node(u, position_u);
        self.insert_node(v, position_v);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        self.update_route_data(r_v);
        true
    }
}
