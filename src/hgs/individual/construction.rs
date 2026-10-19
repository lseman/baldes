use super::*;

impl<'p> Individual<'p> {
    /// Create a random individual with a shuffled giant tour.
    pub fn new_random(params: &'p Params) -> Self {
        let nb_clients = params.nb_clients as usize;
        let mut chrom_t: Vec<usize> = (1..=nb_clients).collect();
        params.rng.borrow_mut().shuffle(&mut chrom_t);

        let mut indiv = Self::from_giant_tour(params, &chrom_t);
        // Not evaluated until Split assigns routes.
        indiv.my_cost_sol.penalized_cost = 1e30;
        indiv
    }

    /// Create an individual from a string like `"1 0 3 2 5 0 4"`.
    pub fn from_string(params: &'p Params, solution_str: &str) -> Self {
        let mut chrom_r = vec![Vec::new(); params.nb_vehicles as usize];
        let mut chrom_t = Vec::new();

        let mut route = 0;
        for token in solution_str.split_whitespace() {
            if let Ok(customer) = token.parse::<usize>() {
                if customer == 0 {
                    route += 1;
                } else {
                    if route < chrom_r.len() {
                        chrom_r[route].push(customer);
                    }
                    chrom_t.push(customer);
                }
            }
        }

        let mut indiv = Self::from_components(params, &chrom_t, &chrom_r);
        indiv.evaluate_complete_cost();
        indiv
    }

    /// Create an individual using the Regret-Capacitated-Walk-Start (RCWS) heuristic.
    pub fn from_rcws(params: &'p Params, pattern: Option<&[Vec<usize>]>) -> Self {
        let nb_clients = params.nb_clients as usize;
        let mut chrom_r: Vec<Vec<usize>> = Vec::new();
        let mut load = Vec::new();
        let mut in_route = vec![false; nb_clients + 1];
        let mut interior = vec![false; nb_clients + 1];
        let mut customer_route = vec![isize::MAX as isize; nb_clients + 1];

        let mut next_empty_route = 0;

        // Insert routes from pattern if provided.
        if let Some(pattern) = pattern {
            for route_customers in pattern {
                while next_empty_route >= chrom_r.len() {
                    chrom_r.push(Vec::new());
                    load.push(0);
                }
                for &customer in route_customers {
                    chrom_r[next_empty_route].push(customer);
                    load[next_empty_route] += params.cli[customer].demand;
                    in_route[customer] = true;
                    customer_route[customer] = next_empty_route as isize;
                    if route_customers.len() > 2 && customer > 0 {
                        let idx = route_customers.iter().position(|&c| c == customer);
                        if let Some(i) = idx {
                            if i > 0 && i < route_customers.len() - 1 {
                                interior[customer] = true;
                            }
                        }
                    }
                }
                while next_empty_route < chrom_r.len() && !chrom_r[next_empty_route].is_empty() {
                    next_empty_route += 1;
                }
            }
        }

        // Process savings list to merge routes.
        for savings in &params.savings_list {
            if savings.value <= 0.0 {
                continue;
            }
            let c1 = savings.c1 as usize;
            let c2 = savings.c2 as usize;

            if params.cli[c1].demand + params.cli[c2].demand > params.vehicle_capacity {
                continue;
            }

            match (in_route[c1], in_route[c2]) {
                (false, false) => {
                    // Neither customer is in a route — create new route.
                    while next_empty_route >= chrom_r.len() {
                        chrom_r.push(Vec::new());
                        load.push(0);
                    }
                    chrom_r[next_empty_route].push(c1);
                    chrom_r[next_empty_route].push(c2);
                    load[next_empty_route] = params.cli[c1].demand + params.cli[c2].demand;
                    in_route[c1] = true;
                    in_route[c2] = true;
                    customer_route[c1] = next_empty_route as isize;
                    customer_route[c2] = next_empty_route as isize;
                    while next_empty_route < chrom_r.len() && !chrom_r[next_empty_route].is_empty()
                    {
                        next_empty_route += 1;
                    }
                }
                (true, false) if !interior[c1] => {
                    let route_idx = customer_route[c1] as usize;
                    if route_idx < chrom_r.len() && !chrom_r[route_idx].is_empty() {
                        if chrom_r[route_idx][0] == c1
                            && load[route_idx] + params.cli[c2].demand <= params.vehicle_capacity
                        {
                            chrom_r[route_idx].insert(0, c2);
                            load[route_idx] += params.cli[c2].demand;
                            in_route[c2] = true;
                            customer_route[c2] = route_idx as isize;
                            if chrom_r[route_idx].len() > 2 {
                                interior[c1] = true;
                            }
                        } else if chrom_r[route_idx].last() == Some(&c1)
                            && load[route_idx] + params.cli[c2].demand <= params.vehicle_capacity
                        {
                            chrom_r[route_idx].push(c2);
                            load[route_idx] += params.cli[c2].demand;
                            in_route[c2] = true;
                            customer_route[c2] = route_idx as isize;
                            if chrom_r[route_idx].len() > 2 {
                                interior[c1] = true;
                            }
                        }
                    }
                }
                (false, true) if !interior[c2] => {
                    let route_idx = customer_route[c2] as usize;
                    if route_idx < chrom_r.len() && !chrom_r[route_idx].is_empty() {
                        if chrom_r[route_idx][0] == c2
                            && load[route_idx] + params.cli[c1].demand <= params.vehicle_capacity
                        {
                            chrom_r[route_idx].insert(0, c1);
                            load[route_idx] += params.cli[c1].demand;
                            in_route[c1] = true;
                            customer_route[c1] = route_idx as isize;
                            if chrom_r[route_idx].len() > 2 {
                                interior[c2] = true;
                            }
                        } else if chrom_r[route_idx].last() == Some(&c2)
                            && load[route_idx] + params.cli[c1].demand <= params.vehicle_capacity
                        {
                            chrom_r[route_idx].push(c1);
                            load[route_idx] += params.cli[c1].demand;
                            in_route[c1] = true;
                            customer_route[c1] = route_idx as isize;
                            if chrom_r[route_idx].len() > 2 {
                                interior[c2] = true;
                            }
                        }
                    }
                }
                (true, true) if !interior[c1] && !interior[c2] => {
                    let r1 = customer_route[c1] as usize;
                    let r2 = customer_route[c2] as usize;
                    if r1 != r2 && load[r1] + load[r2] <= params.vehicle_capacity {
                        // Merge routes based on endpoint matching.
                        let front1 = chrom_r[r1].first() == Some(&c1);
                        let back1 = chrom_r[r1].last() == Some(&c1);
                        let front2 = chrom_r[r2].first() == Some(&c2);
                        let back2 = chrom_r[r2].last() == Some(&c2);

                        if front1 && front2 {
                            let mut merged = chrom_r[r2].clone();
                            merged.reverse();
                            merged.extend_from_slice(&chrom_r[r1]);
                            chrom_r[r1] = merged;
                        } else if back1 && front2 {
                            let r2_clone = chrom_r[r2].clone();
                            chrom_r[r1].extend(&r2_clone);
                        } else if front1 && back2 {
                            let mut merged = chrom_r[r2].clone();
                            merged.extend_from_slice(&chrom_r[r1]);
                            chrom_r[r1] = merged;
                        } else if back1 && back2 {
                            let mut merged = chrom_r[r2].clone();
                            merged.reverse();
                            chrom_r[r1].extend(&merged);
                        }

                        load[r1] += load[r2];
                        for &cust in &chrom_r[r2] {
                            customer_route[cust] = r1 as isize;
                        }
                        chrom_r[r2].clear();
                        load[r2] = 0;
                        interior[c1] = true;
                        interior[c2] = true;
                    }
                }
                _ => {}
            }
        }

        // Assign remaining unassigned customers.
        for i in 1..=nb_clients {
            if !in_route[i] {
                let mut best_route: Option<usize> = None;
                let mut best_cost = f64::INFINITY;

                for (r, route) in chrom_r.iter().enumerate() {
                    if !route.is_empty() {
                        let last = *route.last().unwrap();
                        let cost = params.time_cost.get(last, i);
                        if cost < best_cost
                            && load[r] + params.cli[i].demand <= params.vehicle_capacity
                        {
                            best_cost = cost;
                            best_route = Some(r);
                        }
                    }
                }

                if let Some(best) = best_route {
                    chrom_r[best].push(i);
                    load[best] += params.cli[i].demand;
                    in_route[i] = true;
                    customer_route[i] = best as isize;
                }
            }
        }

        // Flatten routes into giant tour.
        let mut chrom_t = Vec::with_capacity(nb_clients);
        for route in &chrom_r {
            chrom_t.extend(route);
        }

        let mut indiv = Self::from_components(params, &chrom_t, &chrom_r);
        indiv.evaluate_complete_cost();
        indiv
    }

    // ── Internal constructors ─────────────────────────────────────────────

    pub(super) fn from_giant_tour(params: &'p Params, chrom_t: &[usize]) -> Self {
        let nb_clients = params.nb_clients as usize;
        let nb_vehicles = params.nb_vehicles as usize;

        Individual {
            id: next_individual_id(),
            params,
            nb_vehicles,
            my_cost_sol: CostSol::default(),
            chrom_t: chrom_t.to_vec(),
            chrom_r: vec![Vec::new(); nb_vehicles],
            successors: vec![0; nb_clients + 1],
            predecessors: vec![0; nb_clients + 1],
            indivs_per_proximity: BTreeSet::new(),
            is_feasible: false,
            biased_fitness: 0.0,
        }
    }

    pub(super) fn from_components(
        params: &'p Params,
        chrom_t: &[usize],
        chrom_r: &[Vec<usize>],
    ) -> Self {
        let nb_clients = params.nb_clients as usize;
        let nb_vehicles = params.nb_vehicles as usize;

        let mut indiv = Individual {
            id: next_individual_id(),
            params,
            nb_vehicles,
            my_cost_sol: CostSol::default(),
            chrom_t: chrom_t.to_vec(),
            chrom_r: chrom_r.to_vec(),
            successors: vec![0; nb_clients + 1],
            predecessors: vec![0; nb_clients + 1],
            indivs_per_proximity: BTreeSet::new(),
            is_feasible: false,
            biased_fitness: 0.0,
        };
        indiv.rebuild_maps();
        indiv
    }
}
