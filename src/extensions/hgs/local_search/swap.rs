use super::*;

impl<'p> HGSLocalSearch<'p> {
    /// Swap two single clients U and V.
    pub fn swap_two_single_clients(&mut self, u: usize, v: usize) -> bool {
        let u_prev = self.nodes[u].prev as usize;
        let x = self.nodes[u].next as usize;
        let v_prev = self.nodes[v].prev as usize;
        let y = self.nodes[v].next as usize;

        if u == v_prev || u == y {
            return false;
        }

        let u_cour = self.nodes[u].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let u_prev_cour = self.nodes[u_prev].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let v_prev_cour = self.nodes[v_prev].cour as usize;
        let y_cour = self.nodes[y].cour as usize;

        let mut cost_supp_u = (self.params.time_cost.get(u_prev_cour, v_cour)
            + self.params.time_cost.get(v_cour, x_cour)
            - self.params.time_cost.get(u_prev_cour, u_cour)
            - self.params.time_cost.get(u_cour, x_cour)) as f64;
        let mut cost_supp_v = (self.params.time_cost.get(v_prev_cour, u_cour)
            + self.params.time_cost.get(u_cour, y_cour)
            - self.params.time_cost.get(v_prev_cour, v_cour)
            - self.params.time_cost.get(v_cour, y_cour)) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        let r_v = self.nodes[v].route_idx as usize;
        let load_u = self.params.cli[u_cour].demand;
        let load_v = self.params.cli[v_cour].demand;

        let route_u_load_pen = self.route_has_load_penalty(r_u);
        let route_u_tw_pen = self.route_has_tw_violation(r_u);
        let route_v_load_pen = self.route_has_load_penalty(r_v);
        let route_v_tw_pen = self.route_has_tw_violation(r_v);

        if r_u != r_v {
            if !route_u_load_pen
                && !route_u_tw_pen
                && !route_v_load_pen
                && !route_v_tw_pen
                && cost_supp_u + cost_supp_v > -MY_EPSILON
            {
                return false;
            }

            // Time-window penalties are nonnegative. Reject using distance and
            // capacity alone before merging any candidate time-window segments.
            let load_pen_u = self.penalty_excess_load(self.routes[r_u].load + load_v - load_u);
            let load_pen_v = self.penalty_excess_load(self.routes[r_v].load + load_u - load_v);
            let lower_bound = (cost_supp_u + (load_pen_u - self.routes[r_u].penalty))
                + (cost_supp_v + (load_pen_v - self.routes[r_v].penalty));
            if lower_bound > MY_EPSILON {
                return false;
            }

            let u_prefix_v = TimeWindowData::merge(
                &self.nodes[u_prev].prefix_tw_data,
                &self.nodes[v].tw_data,
                self.params,
            );
            let route_u_tw =
                TimeWindowData::merge(&u_prefix_v, &self.nodes[x].postfix_tw_data, self.params);

            let v_prefix_u = TimeWindowData::merge(
                &self.nodes[v_prev].prefix_tw_data,
                &self.nodes[u].tw_data,
                self.params,
            );
            let route_v_tw =
                TimeWindowData::merge(&v_prefix_u, &self.nodes[y].postfix_tw_data, self.params);

            cost_supp_u +=
                load_pen_u + self.penalty_time_windows(&route_u_tw) - self.routes[r_u].penalty;
            cost_supp_v +=
                load_pen_v + self.penalty_time_windows(&route_v_tw) - self.routes[r_v].penalty;
        } else {
            if !route_u_tw_pen && cost_supp_u + cost_supp_v > -MY_EPSILON {
                return false;
            }

            let route_u_tw = if self.nodes[u].position < self.nodes[v].position {
                let seg_tw = self.get_route_segment_tw_data(x, v_prev);
                let p1 = TimeWindowData::merge(
                    &self.nodes[u_prev].prefix_tw_data,
                    &self.nodes[v].tw_data,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                let p3 = TimeWindowData::merge(&p2, &self.nodes[u].tw_data, self.params);
                TimeWindowData::merge(&p3, &self.nodes[y].postfix_tw_data, self.params)
            } else {
                let seg_tw = self.get_route_segment_tw_data(y, u_prev);
                let p1 = TimeWindowData::merge(
                    &self.nodes[v_prev].prefix_tw_data,
                    &self.nodes[u].tw_data,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                let p3 = TimeWindowData::merge(&p2, &self.nodes[v].tw_data, self.params);
                TimeWindowData::merge(&p3, &self.nodes[x].postfix_tw_data, self.params)
            };

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
        }

        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        self.swap_node(u, v);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        if r_u != r_v {
            self.update_route_data(r_v);
        }

        true
    }

    /// Swap two clients (U, X) for one client V.
    pub fn swap_two_clients_for_one(&mut self, u: usize, v: usize) -> bool {
        let x = self.nodes[u].next as usize;
        if self.nodes[x].is_depot {
            return false;
        }
        let u_prev = self.nodes[u].prev as usize;
        let x_next = self.nodes[x].next as usize;
        let v_prev = self.nodes[v].prev as usize;
        let y = self.nodes[v].next as usize;

        if u == v_prev || x == v_prev || u == y {
            return false;
        }

        let u_cour = self.nodes[u].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let u_prev_cour = self.nodes[u_prev].cour as usize;
        let x_next_cour = self.nodes[x_next].cour as usize;
        let v_prev_cour = self.nodes[v_prev].cour as usize;
        let y_cour = self.nodes[y].cour as usize;

        let mut cost_supp_u = (self.params.time_cost.get(u_prev_cour, v_cour)
            + self.params.time_cost.get(v_cour, x_next_cour)
            - self.params.time_cost.get(u_prev_cour, u_cour)
            - self.params.time_cost.get(x_cour, x_next_cour)) as f64;
        let mut cost_supp_v = (self.params.time_cost.get(v_prev_cour, u_cour)
            + self.params.time_cost.get(x_cour, y_cour)
            - self.params.time_cost.get(v_prev_cour, v_cour)
            - self.params.time_cost.get(v_cour, y_cour)) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        let r_v = self.nodes[v].route_idx as usize;
        let load_ux = self.params.cli[u_cour].demand + self.params.cli[x_cour].demand;
        let load_v = self.params.cli[v_cour].demand;

        let route_u_load_pen = self.route_has_load_penalty(r_u);
        let route_u_tw_pen = self.route_has_tw_violation(r_u);
        let route_v_load_pen = self.route_has_load_penalty(r_v);
        let route_v_tw_pen = self.route_has_tw_violation(r_v);

        if r_u != r_v {
            if !route_u_load_pen
                && !route_u_tw_pen
                && !route_v_load_pen
                && !route_v_tw_pen
                && cost_supp_u + cost_supp_v > -MY_EPSILON
            {
                return false;
            }

            // Time-window penalties are nonnegative. Reject using distance and
            // capacity alone before merging any candidate time-window segments.
            let load_pen_u = self.penalty_excess_load(self.routes[r_u].load + load_v - load_ux);
            let load_pen_v = self.penalty_excess_load(self.routes[r_v].load + load_ux - load_v);
            let lower_bound = (cost_supp_u + (load_pen_u - self.routes[r_u].penalty))
                + (cost_supp_v + (load_pen_v - self.routes[r_v].penalty));
            if lower_bound > MY_EPSILON {
                return false;
            }

            let u_prefix_v = TimeWindowData::merge(
                &self.nodes[u_prev].prefix_tw_data,
                &self.nodes[v].tw_data,
                self.params,
            );
            let route_u_tw = TimeWindowData::merge(
                &u_prefix_v,
                &self.nodes[x_next].postfix_tw_data,
                self.params,
            );

            let ux_tw =
                TimeWindowData::merge(&self.nodes[u].tw_data, &self.nodes[x].tw_data, self.params);
            let v_prefix_ux =
                TimeWindowData::merge(&self.nodes[v_prev].prefix_tw_data, &ux_tw, self.params);
            let route_v_tw =
                TimeWindowData::merge(&v_prefix_ux, &self.nodes[y].postfix_tw_data, self.params);

            cost_supp_u +=
                load_pen_u + self.penalty_time_windows(&route_u_tw) - self.routes[r_u].penalty;
            cost_supp_v +=
                load_pen_v + self.penalty_time_windows(&route_v_tw) - self.routes[r_v].penalty;
        } else {
            if !route_u_tw_pen && cost_supp_u + cost_supp_v > -MY_EPSILON {
                return false;
            }

            let ux_tw =
                TimeWindowData::merge(&self.nodes[u].tw_data, &self.nodes[x].tw_data, self.params);
            let route_u_tw = if self.nodes[u].position < self.nodes[v].position {
                let seg_tw = self.get_route_segment_tw_data(x_next, v_prev);
                let p1 = TimeWindowData::merge(
                    &self.nodes[u_prev].prefix_tw_data,
                    &self.nodes[v].tw_data,
                    self.params,
                );
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                let p3 = TimeWindowData::merge(&p2, &ux_tw, self.params);
                TimeWindowData::merge(&p3, &self.nodes[y].postfix_tw_data, self.params)
            } else {
                let seg_tw = self.get_route_segment_tw_data(y, u_prev);
                let p1 =
                    TimeWindowData::merge(&self.nodes[v_prev].prefix_tw_data, &ux_tw, self.params);
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                let p3 = TimeWindowData::merge(&p2, &self.nodes[v].tw_data, self.params);
                TimeWindowData::merge(&p3, &self.nodes[x_next].postfix_tw_data, self.params)
            };

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
        }

        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        self.swap_node(u, v);
        self.insert_node(x, u);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        if r_u != r_v {
            self.update_route_data(r_v);
        }

        true
    }

    /// Swap two client pairs (U, X) and (V, Y).
    pub fn swap_two_client_pairs(&mut self, u: usize, v: usize) -> bool {
        let x = self.nodes[u].next as usize;
        let y = self.nodes[v].next as usize;
        let u_prev = self.nodes[u].prev as usize;
        let x_next = self.nodes[x].next as usize;

        if self.nodes[x].is_depot
            || self.nodes[y].is_depot
            || y == u_prev
            || u == y
            || x == v
            || v == x_next
        {
            return false;
        }

        let v_prev = self.nodes[v].prev as usize;
        let y_next = self.nodes[y].next as usize;

        let u_cour = self.nodes[u].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let y_cour = self.nodes[y].cour as usize;
        let u_prev_cour = self.nodes[u_prev].cour as usize;
        let x_next_cour = self.nodes[x_next].cour as usize;
        let v_prev_cour = self.nodes[v_prev].cour as usize;
        let y_next_cour = self.nodes[y_next].cour as usize;

        let mut cost_supp_u = (self.params.time_cost.get(u_prev_cour, v_cour)
            + self.params.time_cost.get(y_cour, x_next_cour)
            - self.params.time_cost.get(u_prev_cour, u_cour)
            - self.params.time_cost.get(x_cour, x_next_cour)) as f64;
        let mut cost_supp_v = (self.params.time_cost.get(v_prev_cour, u_cour)
            + self.params.time_cost.get(x_cour, y_next_cour)
            - self.params.time_cost.get(v_prev_cour, v_cour)
            - self.params.time_cost.get(y_cour, y_next_cour)) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        let r_v = self.nodes[v].route_idx as usize;
        let load_ux = self.params.cli[u_cour].demand + self.params.cli[x_cour].demand;
        let load_vy = self.params.cli[v_cour].demand + self.params.cli[y_cour].demand;

        let route_u_load_pen = self.route_has_load_penalty(r_u);
        let route_u_tw_pen = self.route_has_tw_violation(r_u);
        let route_v_load_pen = self.route_has_load_penalty(r_v);
        let route_v_tw_pen = self.route_has_tw_violation(r_v);

        if r_u != r_v {
            if !route_u_load_pen
                && !route_u_tw_pen
                && !route_v_load_pen
                && !route_v_tw_pen
                && cost_supp_u + cost_supp_v > -MY_EPSILON
            {
                return false;
            }

            // Time-window penalties are nonnegative. Reject using distance and
            // capacity alone before merging any candidate time-window segments.
            let load_pen_u = self.penalty_excess_load(self.routes[r_u].load + load_vy - load_ux);
            let load_pen_v = self.penalty_excess_load(self.routes[r_v].load + load_ux - load_vy);
            let lower_bound = (cost_supp_u + (load_pen_u - self.routes[r_u].penalty))
                + (cost_supp_v + (load_pen_v - self.routes[r_v].penalty));
            if lower_bound > MY_EPSILON {
                return false;
            }

            let vy_tw =
                TimeWindowData::merge(&self.nodes[v].tw_data, &self.nodes[y].tw_data, self.params);
            let ux_tw =
                TimeWindowData::merge(&self.nodes[u].tw_data, &self.nodes[x].tw_data, self.params);

            let u_prefix_vy =
                TimeWindowData::merge(&self.nodes[u_prev].prefix_tw_data, &vy_tw, self.params);
            let route_u_tw = TimeWindowData::merge(
                &u_prefix_vy,
                &self.nodes[x_next].postfix_tw_data,
                self.params,
            );

            let v_prefix_ux =
                TimeWindowData::merge(&self.nodes[v_prev].prefix_tw_data, &ux_tw, self.params);
            let route_v_tw = TimeWindowData::merge(
                &v_prefix_ux,
                &self.nodes[y_next].postfix_tw_data,
                self.params,
            );

            cost_supp_u +=
                load_pen_u + self.penalty_time_windows(&route_u_tw) - self.routes[r_u].penalty;
            cost_supp_v +=
                load_pen_v + self.penalty_time_windows(&route_v_tw) - self.routes[r_v].penalty;
        } else {
            if !route_u_tw_pen && cost_supp_u + cost_supp_v > -MY_EPSILON {
                return false;
            }

            let vy_tw =
                TimeWindowData::merge(&self.nodes[v].tw_data, &self.nodes[y].tw_data, self.params);
            let ux_tw =
                TimeWindowData::merge(&self.nodes[u].tw_data, &self.nodes[x].tw_data, self.params);

            let route_u_tw = if self.nodes[u].position < self.nodes[v].position {
                let seg_tw = self.get_route_segment_tw_data(x_next, v_prev);
                let p1 =
                    TimeWindowData::merge(&self.nodes[u_prev].prefix_tw_data, &vy_tw, self.params);
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                let p3 = TimeWindowData::merge(&p2, &ux_tw, self.params);
                TimeWindowData::merge(&p3, &self.nodes[y_next].postfix_tw_data, self.params)
            } else {
                let seg_tw = self.get_route_segment_tw_data(y_next, u_prev);
                let p1 =
                    TimeWindowData::merge(&self.nodes[v_prev].prefix_tw_data, &ux_tw, self.params);
                let p2 = TimeWindowData::merge(&p1, &seg_tw, self.params);
                let p3 = TimeWindowData::merge(&p2, &vy_tw, self.params);
                TimeWindowData::merge(&p3, &self.nodes[x_next].postfix_tw_data, self.params)
            };

            cost_supp_u += self.penalty_excess_load(self.routes[r_u].load)
                + self.penalty_time_windows(&route_u_tw)
                - self.routes[r_u].penalty;
        }

        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        self.swap_node(u, v);
        self.swap_node(x, y);
        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        if r_u != r_v {
            self.update_route_data(r_v);
        }

        true
    }
}
