use super::*;

impl<'p> HGSLocalSearch<'p> {
    /// Create a new local search engine.
    pub fn new(params: &'p Params) -> Self {
        let nb_clients = params.nb_clients as usize;
        let nb_vehicles = params.nb_vehicles as usize;
        let total_nodes = nb_clients + 2 * nb_vehicles + 1;

        let mut nodes = vec![Node::default(); total_nodes];
        for i in 1..=nb_clients {
            nodes[i].cour = i as i32;
            nodes[i].is_depot = false;
        }

        let routes = (0..nb_vehicles)
            .map(|r| HGSRoute {
                cour: r as i32,
                ..Default::default()
            })
            .collect();

        let order_nodes: Vec<usize> = (1..=nb_clients).collect();
        let order_routes: Vec<usize> = (0..nb_vehicles).collect();

        Self {
            params,
            search_completed: false,
            nb_moves: 0,
            order_nodes,
            order_routes,
            empty_routes: std::collections::BTreeSet::new(),
            loop_id: 0,
            nodes,
            routes,
            best_insert_initialized_for_route: vec![false; nb_vehicles],
            best_insert_client: vec![vec![ThreeBestInsert::new(); nb_clients + 1]; nb_vehicles],
            best_insert_client_tw: vec![vec![ThreeBestInsert::new(); nb_clients + 1]; nb_vehicles],
            penalty_capacity_ls: 0.0,
            penalty_time_warp_ls: 0.0,
            correlated_vertices: Vec::new(),
            nb_granular: 0,
        }
        .with_granular(params.config.nb_granular)
    }

    pub(super) fn with_granular(mut self, nb_granular: i32) -> Self {
        self.set_granular(nb_granular);
        self
    }

    /// Use the `nb_granular` closest clients as neighborhoods (HGS
    /// SetCorrelatedVertices, used when the granular size grows).
    pub fn set_granular(&mut self, nb_granular: i32) {
        self.nb_granular = nb_granular.max(1);
        self.correlated_vertices = self
            .params
            .correlated_vertices_for(nb_granular)
            .into_iter()
            .map(|list| list.into_iter().map(|v| v as usize).collect())
            .collect();
    }

    #[inline(always)]
    pub fn depot_start(&self, r: usize) -> usize {
        self.params.nb_clients as usize + 1 + r
    }

    #[inline(always)]
    pub fn depot_end(&self, r: usize) -> usize {
        self.params.nb_clients as usize + 1 + self.params.nb_vehicles as usize + r
    }

    #[inline(always)]
    pub(super) fn penalty_excess_load(&self, load: i32) -> f64 {
        max(0, load - self.params.vehicle_capacity) as f64 * self.penalty_capacity_ls
    }

    #[inline(always)]
    pub(super) fn penalty_time_windows(&self, tw_data: &TimeWindowData) -> f64 {
        if matches!(
            self.params.problem_type,
            ProblemType::Vrptw | ProblemType::Evrp
        ) {
            (tw_data.time_warp + (tw_data.latest_release_time - tw_data.latest_arrival).max(0.))
                as f64
                * self.penalty_time_warp_ls
        } else {
            0.0
        }
    }

    /// Merge consecutive time window segments left to right.
    pub(super) fn merge_all(&self, parts: &[&TimeWindowData]) -> TimeWindowData {
        let mut result = parts[0].clone();
        for part in &parts[1..] {
            result = TimeWindowData::merge(&result, part, self.params);
        }
        result
    }

    /// Penalty difference of replacing the route time windows `old` by `new`.
    pub(super) fn delta_penalty_time_windows(
        &self,
        new: &TimeWindowData,
        old: &TimeWindowData,
    ) -> f64 {
        self.penalty_time_windows(new) - self.penalty_time_windows(old)
    }

    #[inline(always)]
    pub(super) fn route_has_tw_violation(&self, r: usize) -> bool {
        matches!(
            self.params.problem_type,
            ProblemType::Vrptw | ProblemType::Evrp
        ) && self.routes[r].tw_data.time_warp > MY_EPSILON
    }

    #[inline(always)]
    pub(super) fn route_has_load_penalty(&self, r: usize) -> bool {
        self.routes[r].load > self.params.vehicle_capacity
    }

    /// Run the full local search with the specified penalty values.
    pub fn run(
        &mut self,
        indiv: &mut Individual<'_>,
        penalty_capacity_ls: f64,
        penalty_time_warp_ls: f64,
    ) {
        let intensification = self.params.config.intensification_probability_ls;
        let run_intensification =
            (self.params.rng.borrow_mut().next() % 100) < intensification.max(0) as u64;
        self.penalty_capacity_ls = penalty_capacity_ls;
        self.penalty_time_warp_ls = if matches!(
            self.params.problem_type,
            ProblemType::Vrptw | ProblemType::Evrp
        ) {
            penalty_time_warp_ls
        } else {
            0.0
        };

        self.load_individual(indiv);

        {
            let mut rng = self.params.rng.borrow_mut();
            rng.shuffle(&mut self.order_nodes);
            rng.shuffle(&mut self.order_routes);
            // Shuffle neighborhoods only sporadically to avoid O(n^2) work.
            for i in 1..=self.params.nb_clients as usize {
                if rng.next() % self.nb_granular as u64 == 0 {
                    rng.shuffle(&mut self.correlated_vertices[i]);
                }
            }
        }

        self.search_completed = false;
        self.loop_id = 0;
        while !self.search_completed {
            if self.loop_id > 1 {
                self.search_completed = true;
            }

            // Phase 1: classical route improvement (RI) moves.
            for pos_u in 0..self.params.nb_clients as usize {
                let u = self.order_nodes[pos_u];
                let last_test_ri = self.nodes[u].when_last_tested_ri;
                self.nodes[u].when_last_tested_ri = self.nb_moves;

                for k in 0..self.correlated_vertices[u].len() {
                    let v = self.correlated_vertices[u][k];
                    let r_u = self.nodes[u].route_idx as usize;
                    let r_v = self.nodes[v].route_idx as usize;
                    if self.loop_id != 0
                        && max(
                            self.routes[r_u].when_last_modified,
                            self.routes[r_v].when_last_modified,
                        ) <= last_test_ri
                    {
                        continue;
                    }
                    if self.move_single_client(u, v)
                        || self.move_two_clients(u, v)
                        || self.move_two_clients_reversed(u, v)
                        || (u < v && self.swap_two_single_clients(u, v))
                        || self.swap_two_clients_for_one(u, v)
                        || (u < v && self.swap_two_client_pairs(u, v))
                        || (r_u < r_v && self.two_opt_between_trips(u, v))
                        || (r_u == r_v && self.two_opt_within_trip(u, v))
                    {
                        continue;
                    }

                    // Also try inserting U right after the depot preceding V.
                    let v_prev = self.nodes[v].prev as usize;
                    if self.nodes[v_prev].is_depot {
                        let _ = self.move_single_client(u, v_prev)
                            || self.move_two_clients(u, v_prev)
                            || self.move_two_clients_reversed(u, v_prev)
                            || (r_u < self.nodes[v_prev].route_idx as usize
                                && self.two_opt_between_trips(u, v_prev));
                    }
                }

                // Phase 2: moves into an empty route.
                if self.loop_id > 0 && !self.empty_routes.is_empty() {
                    let depot_v = self.depot_start(*self.empty_routes.iter().next().unwrap());
                    let _ = self.move_single_client(u, depot_v)
                        || self.move_two_clients(u, depot_v)
                        || self.move_two_clients_reversed(u, depot_v)
                        || self.two_opt_between_trips(u, depot_v);
                }
            }

            // Phase 3: SWAP* and RELOCATE* between routes with overlapping sectors.
            if intensification != 0
                && self.search_completed
                && (intensification == 100 || run_intensification)
            {
                for pos_u in 0..self.params.nb_vehicles as usize {
                    let r_u = self.order_routes[pos_u];
                    if self.routes[r_u].nb_customers == 0 {
                        continue;
                    }
                    let last_test_large_nb = self.routes[r_u].when_last_tested_large_nb;
                    self.routes[r_u].when_last_tested_large_nb = self.nb_moves;
                    for pos_v in 0..self.params.nb_vehicles as usize {
                        let r_v = self.order_routes[pos_v];
                        if self.routes[r_v].nb_customers == 0
                            || r_u >= r_v
                            || (self.loop_id > 0
                                && max(
                                    self.routes[r_u].when_last_modified,
                                    self.routes[r_v].when_last_modified,
                                ) <= last_test_large_nb)
                            || !CircleSector::overlaps(
                                &self.routes[r_u].sector,
                                &self.routes[r_v].sector,
                                self.params.circle_sector_overlap_tolerance,
                            )
                        {
                            continue;
                        }
                        if !self.relocate_star(r_u, r_v)
                            && (self.params.config.skip_swap_star_dist
                                || !self.swap_star(r_u, r_v, false))
                            && self.params.config.use_swap_star_tw
                        {
                            self.swap_star(r_u, r_v, true);
                        }
                    }
                }
            }
            self.loop_id += 1;
        }

        self.export_individual(indiv);
    }

    pub(super) fn cost(&self, a: usize, b: usize) -> f64 {
        self.params
            .time_cost
            .get(self.nodes[a].cour as usize, self.nodes[b].cour as usize)
    }
}
