use super::*;

impl<'p> Individual<'p> {
    // ── Construction ──────────────────────────────────────────────────────

    /// Clone an individual, copying over map fields (successors, predecessors, proximity).
    pub fn clone_with_maps(indiv: &Self) -> Self {
        indiv.clone()
    }

    /// Create an empty individual (no routes, no clients scheduled).
    pub fn new_empty(params: &'p Params) -> Self {
        let chrom_r = vec![Vec::new(); params.nb_vehicles as usize];
        let chrom_t = Vec::new();
        let nb_clients = params.nb_clients as usize;
        let successors = vec![0; nb_clients + 1];
        let predecessors = vec![0; nb_clients + 1];
        let indivs_per_proximity = BTreeSet::new();
        let mut my_cost_sol = CostSol::default();
        my_cost_sol.penalized_cost = 1e30;
        Self {
            id: next_individual_id(),
            params,
            nb_vehicles: params.nb_vehicles as usize,
            my_cost_sol,
            chrom_t,
            chrom_r,
            successors,
            predecessors,
            indivs_per_proximity,
            is_feasible: false,
            biased_fitness: 0.0,
        }
    }

    /// Rebuild predecessor/successor maps from chrom_r.
    pub fn rebuild_maps(&mut self) {
        let nb_clients = self.params.nb_clients as usize;
        self.successors = vec![0; nb_clients + 1];
        self.predecessors = vec![0; nb_clients + 1];

        for r in 0..self.nb_vehicles {
            if r >= self.chrom_r.len() {
                continue;
            }
            let route = &self.chrom_r[r];
            if route.is_empty() {
                continue;
            }

            // Depot (0) -> first customer.
            self.successors[0] = route[0];
            self.predecessors[route[0]] = 0;

            // Between customers in the route.
            for i in 1..route.len() {
                self.successors[route[i - 1]] = route[i];
                self.predecessors[route[i]] = route[i - 1];
            }

            // Last customer -> depot (0).
            self.successors[route[route.len() - 1]] = 0;
            self.predecessors[0] = route[route.len() - 1];
        }
    }

    // ── Giant tour operations ─────────────────────────────────────────────

    /// Shuffle the giant tour randomly.
    pub fn shuffle_chrom_t(&mut self) {
        self.chrom_t = (1..=self.params.nb_clients as usize).collect();
        self.params.rng.borrow_mut().shuffle(&mut self.chrom_t);
    }

    // ── I/O ───────────────────────────────────────────────────────────────

    /// Export the solution in CVRPLib format to a string.
    pub fn export_cvrplib_string(&self) -> String {
        let mut output = String::new();
        for (r, route) in self.chrom_r.iter().enumerate() {
            if !route.is_empty() {
                output.push_str(&format!("Route {}:\n", r + 1));
                output.push_str("0 ");
                for &cust in route {
                    output.push_str(&format!("{} ", cust));
                }
                output.push_str("0\n");
            }
        }
        output
    }

    /// Print the solution in CVRPLib format.
    pub fn print_cvrplib_format(&self) {
        println!("{}", self.export_cvrplib_string());
    }
}
