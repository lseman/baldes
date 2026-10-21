use super::*;

impl<'p> HGSLocalSearch<'p> {
    /// 2-opt within a single route.
    pub fn two_opt_within_trip(&mut self, u: usize, v: usize) -> bool {
        if self.nodes[u].position >= self.nodes[v].position - 1 {
            return false;
        }

        let x = self.nodes[u].next as usize;
        let y = self.nodes[v].next as usize;
        let u_cour = self.nodes[u].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let y_cour = self.nodes[y].cour as usize;

        let mut cost = (self.params.time_cost.get(u_cour, v_cour)
            + self.params.time_cost.get(x_cour, y_cour)
            - self.params.time_cost.get(u_cour, x_cour)
            - self.params.time_cost.get(v_cour, y_cour)
            + self.nodes[v].cumulated_reversal_distance
            - self.nodes[x].cumulated_reversal_distance) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        if !self.route_has_tw_violation(r_u) && cost > -MY_EPSILON {
            return false;
        }

        let mut route_tw = self.nodes[u].prefix_tw_data.clone();
        let mut it = v;
        while it != u {
            route_tw = TimeWindowData::merge(&route_tw, &self.nodes[it].tw_data, self.params);
            it = self.nodes[it].prev as usize;
        }
        route_tw = TimeWindowData::merge(&route_tw, &self.nodes[y].postfix_tw_data, self.params);

        cost += self.penalty_excess_load(self.routes[r_u].load)
            + self.penalty_time_windows(&route_tw)
            - self.routes[r_u].penalty;

        if cost > -MY_EPSILON {
            return false;
        }

        let mut it = v;
        let mut ins_point = u;
        while it != x {
            let current = it;
            it = self.nodes[it].prev as usize;
            self.insert_node(current, ins_point);
            ins_point = current;
        }

        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);

        true
    }

    /// 2-opt* between two different routes.
    pub fn two_opt_between_trips(&mut self, u: usize, v: usize) -> bool {
        let x = self.nodes[u].next as usize;
        let y = self.nodes[v].next as usize;

        let u_cour = self.nodes[u].cour as usize;
        let v_cour = self.nodes[v].cour as usize;
        let x_cour = self.nodes[x].cour as usize;
        let y_cour = self.nodes[y].cour as usize;

        let mut cost_supp_u = (self.params.time_cost.get(u_cour, y_cour)
            - self.params.time_cost.get(u_cour, x_cour)) as f64;
        let mut cost_supp_v = (self.params.time_cost.get(v_cour, x_cour)
            - self.params.time_cost.get(v_cour, y_cour)) as f64;

        let r_u = self.nodes[u].route_idx as usize;
        let r_v = self.nodes[v].route_idx as usize;

        if !self.route_has_load_penalty(r_u)
            && !self.route_has_tw_violation(r_u)
            && !self.route_has_load_penalty(r_v)
            && !self.route_has_tw_violation(r_v)
            && cost_supp_u + cost_supp_v > -MY_EPSILON
        {
            return false;
        }

        let route_u_tw = TimeWindowData::merge(
            &self.nodes[u].prefix_tw_data,
            &self.nodes[y].postfix_tw_data,
            self.params,
        );
        let route_v_tw = TimeWindowData::merge(
            &self.nodes[v].prefix_tw_data,
            &self.nodes[x].postfix_tw_data,
            self.params,
        );

        cost_supp_u += self.penalty_excess_load(
            self.nodes[u].cumulated_load + self.routes[r_v].load - self.nodes[v].cumulated_load,
        ) + self.penalty_time_windows(&route_u_tw)
            - self.routes[r_u].penalty;

        cost_supp_v += self.penalty_excess_load(
            self.nodes[v].cumulated_load + self.routes[r_u].load - self.nodes[u].cumulated_load,
        ) + self.penalty_time_windows(&route_v_tw)
            - self.routes[r_v].penalty;

        if cost_supp_u + cost_supp_v > -MY_EPSILON {
            return false;
        }

        let mut it_v = y;
        let mut ins = u;
        while !self.nodes[it_v].is_depot {
            let curr = it_v;
            it_v = self.nodes[it_v].next as usize;
            self.insert_node(curr, ins);
            ins = curr;
        }

        let mut it_u = x;
        ins = v;
        while !self.nodes[it_u].is_depot {
            let curr = it_u;
            it_u = self.nodes[it_u].next as usize;
            self.insert_node(curr, ins);
            ins = curr;
        }

        self.nb_moves += 1;
        self.search_completed = false;
        self.update_route_data(r_u);
        self.update_route_data(r_v);

        true
    }
}
