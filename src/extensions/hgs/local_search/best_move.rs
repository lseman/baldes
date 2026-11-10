//! Candidate evaluation shares the exact delta and pruning code with application.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) enum Move {
    Single(usize, usize),
    Pair(usize, usize),
    ReversedPair(usize, usize),
    Swap(usize, usize),
    PairSingle(usize, usize),
    PairPair(usize, usize),
    TwoOpt(usize, usize),
    TwoOptStar(usize, usize),
    RelocateStar(usize, usize),
    SwapStar(usize, usize, bool),
}

#[cfg(test)]
#[path = "../../../../tests/unit/extensions/hgs/local_search/best_move.rs"]
mod tests;

impl<'p> HGSLocalSearch<'p> {
    #[inline]
    pub(super) fn delta_limit(&self) -> f64 {
        self.move_credit.unwrap_or(-MY_EPSILON)
    }

    #[inline]
    pub(super) fn bound_limit(&self) -> f64 {
        self.move_credit.unwrap_or(0.) + MY_EPSILON
    }

    pub(super) fn register_delta(&mut self, delta: f64) {
        if let Some(credit) = self.move_credit.as_mut() {
            let cap = self.params.config.local_search.max_deterioration * 10.;
            debug_assert!(delta <= *credit + MY_EPSILON);
            *credit = (*credit - delta).clamp(0., cap);
        }
    }

    pub(super) fn evaluate_move(&mut self, mv: Move, apply: bool) -> Option<f64> {
        match mv {
            Move::Single(u, v) => self.evaluate_move_single_client(u, v, apply),
            Move::Pair(u, v) => self.evaluate_move_two_clients(u, v, apply),
            Move::ReversedPair(u, v) => self.evaluate_move_two_clients_reversed(u, v, apply),
            Move::Swap(u, v) => self.evaluate_swap_two_single_clients(u, v, apply),
            Move::PairSingle(u, v) => self.evaluate_swap_two_clients_for_one(u, v, apply),
            Move::PairPair(u, v) => self.evaluate_swap_two_client_pairs(u, v, apply),
            Move::TwoOpt(u, v) => self.evaluate_two_opt_within_trip(u, v, apply),
            Move::TwoOptStar(u, v) => self.evaluate_two_opt_between_trips(u, v, apply),
            Move::RelocateStar(u, route) => {
                let source = self.nodes[u].route_idx as usize;
                self.evaluate_relocate_star(source, route, Some(u), apply)
            }
            Move::SwapStar(u, route, tw) => {
                let source = self.nodes[u].route_idx as usize;
                self.evaluate_swap_star(source, route, tw, Some(u), apply)
            }
        }
    }

    fn consider(&mut self, mv: Move, best: &mut Option<(f64, Move)>) {
        if let Some(delta) = self.evaluate_move(mv, false) {
            if best
                .as_ref()
                .is_none_or(|(cost, _)| delta < *cost - MY_EPSILON)
            {
                *best = Some((delta, mv));
            }
        }
    }

    pub(super) fn best_customer_move(&mut self, u: usize) -> Option<(f64, Move)> {
        let mut best = None;
        let r_u = self.nodes[u].route_idx as usize;
        let last_test = self.nodes[u].when_last_tested_ri;
        self.nodes[u].when_last_tested_ri = self.nb_moves;
        let mut star_routes = std::mem::take(&mut self.candidate_routes);
        star_routes.fill(false);
        for k in 0..self.correlated_vertices[u].len() {
            let v = self.correlated_vertices[u][k];
            let r_v = self.nodes[v].route_idx as usize;
            if r_u != r_v {
                star_routes[r_v] = true;
            }
            if r_u == r_v {
                continue;
            }
            if self.loop_id == 0 && self.move_credit.is_none() && self.reuses_ri_pair(r_u, r_v) {
                continue;
            }
            if self.loop_id > 1
                && max(
                    self.routes[r_u].when_last_modified,
                    self.routes[r_v].when_last_modified,
                ) <= last_test
            {
                continue;
            }
            for mv in [
                Move::Single(u, v),
                Move::Pair(u, v),
                Move::ReversedPair(u, v),
                Move::Swap(u, v),
                Move::PairSingle(u, v),
                Move::PairPair(u, v),
            ] {
                self.consider(mv, &mut best);
            }
            self.consider(
                if r_u == r_v {
                    Move::TwoOpt(u, v)
                } else {
                    Move::TwoOptStar(u, v)
                },
                &mut best,
            );
            // The depot is a separate insertion location, independent of the
            // first customer's eligibility in the granular neighborhood.
            let depot = self.depot_start(r_v);
            for mv in [
                Move::Single(u, depot),
                Move::Pair(u, depot),
                Move::ReversedPair(u, depot),
            ] {
                self.consider(mv, &mut best);
            }
            if r_u != r_v {
                self.consider(Move::TwoOptStar(u, depot), &mut best);
            }
        }
        // Intra-route education scans every insertion/cut position. It is
        // independent of the inter-route granular shortlist.
        let inherited_intra =
            self.loop_id == 0 && self.move_credit.is_none() && self.reuses_ri_pair(r_u, r_u);
        if !inherited_intra
            && (self.loop_id <= 1 || self.routes[r_u].when_last_modified > last_test)
        {
            let depot = self.depot_start(r_u);
            let mut v = depot;
            loop {
                for mv in [
                    Move::Single(u, v),
                    Move::Pair(u, v),
                    Move::ReversedPair(u, v),
                ] {
                    self.consider(mv, &mut best);
                }
                if !self.nodes[v].is_depot {
                    for mv in [
                        Move::Swap(u, v),
                        Move::PairSingle(u, v),
                        Move::PairPair(u, v),
                        Move::TwoOpt(u, v),
                    ] {
                        self.consider(mv, &mut best);
                    }
                    if self.nodes[u].prev as usize == depot {
                        self.consider(Move::TwoOpt(depot, v), &mut best);
                    }
                }
                v = self.nodes[v].next as usize;
                if self.nodes[v].is_depot {
                    break;
                }
            }
        }
        if self.loop_id > 0 {
            if let Some(&r) = self.empty_routes.iter().next() {
                let depot = self.depot_start(r);
                for mv in [
                    Move::Single(u, depot),
                    Move::Pair(u, depot),
                    Move::ReversedPair(u, depot),
                    Move::TwoOptStar(u, depot),
                ] {
                    self.consider(mv, &mut best);
                }
            }
        }
        // SWAP* has its own shortlist, so its route coverage is independent of RI.
        if self.params.config.intensification_probability_ls != 0 {
            for (r_v, eligible) in star_routes.iter().enumerate() {
                if r_u == r_v || self.routes[r_v].nb_customers == 0 {
                    continue;
                }
                if self.loop_id > 1
                    && !self.refresh_swap
                    && max(
                        self.routes[r_u].when_last_modified,
                        self.routes[r_v].when_last_modified,
                    ) <= last_test
                {
                    continue;
                }
                let swap_candidate = self
                    .swap_neighborhood
                    .candidates(
                        u,
                        self.nodes[self.depot_start(r_v)].next as usize,
                        r_v,
                        &self.nodes,
                    )
                    .next()
                    .is_some();
                if !eligible
                    && !swap_candidate
                    && !CircleSector::overlaps(
                        &self.routes[r_u].sector,
                        &self.routes[r_v].sector,
                        self.params.circle_sector_overlap_tolerance,
                    )
                {
                    continue;
                }
                self.consider(Move::RelocateStar(u, r_v), &mut best);
                if !self.params.config.skip_swap_star_dist {
                    self.consider(Move::SwapStar(u, r_v, false), &mut best);
                }
                if self.params.config.use_swap_star_tw {
                    self.consider(Move::SwapStar(u, r_v, true), &mut best);
                }
            }
        }
        self.candidate_routes = star_routes;
        best
    }

    fn save_feasible(&mut self, best: &mut Option<(f64, Vec<Vec<usize>>)>, service: f64) {
        if self.routes.iter().any(|r| {
            r.load > self.params.vehicle_capacity
                || r.tw_data.time_warp > MY_EPSILON
                || r.tw_data.latest_release_time > r.tw_data.latest_arrival + MY_EPSILON
        }) {
            return;
        }
        // Service is constant across complete plans; route durations exclude waiting.
        let distance = self
            .routes
            .iter()
            .filter(|r| r.nb_customers > 0)
            .map(|r| r.duration - self.params.cli[0].service_duration)
            .sum::<f64>()
            - service;
        if best
            .as_ref()
            .is_some_and(|(cost, _)| distance >= *cost - MY_EPSILON)
        {
            return;
        }
        let routes = (0..self.routes.len())
            .map(|r| self.route_clients(r).collect())
            .collect();
        *best = Some((distance, routes));
    }

    pub(super) fn run_best_moves(&mut self, indiv: &mut Individual<'_>) {
        let service = self
            .params
            .cli
            .iter()
            .skip(1)
            .map(|c| c.service_duration)
            .sum();
        let mut best_feasible = indiv
            .is_feasible
            .then(|| (indiv.my_cost_sol.distance, indiv.chrom_r.clone()));
        self.move_credit = (self.params.config.local_search.max_deterioration > 0.).then_some(0.);
        self.loop_id = 0;
        let mut changed = true;
        let mut complete = true;
        while changed || self.loop_id < 2 {
            changed = false;
            if self.loop_id > 0 {
                self.move_credit = None;
            }
            self.refresh_swap = self.swap_neighborhood.begin_pass();
            for index in 0..self.order_nodes.len() {
                let u = self.order_nodes[index];
                loop {
                    if self.params.is_time_limit_exceeded() {
                        complete = false;
                        break;
                    }
                    let Some((expected, mv)) = self.best_customer_move(u) else {
                        break;
                    };
                    let delta = self
                        .evaluate_move(mv, true)
                        .expect("selected move must remain applicable");
                    debug_assert!((delta - expected).abs() <= 1e-8 * expected.abs().max(1.));
                    changed = true;
                    self.save_feasible(&mut best_feasible, service);
                    if delta >= -MY_EPSILON {
                        break;
                    }
                }
                if !complete {
                    break;
                }
            }
            if !complete {
                break;
            }
            self.loop_id += 1;
        }
        self.move_credit = None;
        self.search_completed = complete;
        self.export_individual(indiv);
        if let Some((cost, routes)) = best_feasible {
            if !indiv.is_feasible || cost < indiv.my_cost_sol.distance - MY_EPSILON {
                indiv.chrom_t = routes.iter().flatten().copied().collect();
                indiv.chrom_r = routes;
                indiv.evaluate_complete_cost();
                self.load_individual(indiv);
                return;
            }
        }
        if complete {
            self.remember_searched_plan(indiv);
        }
    }
}
