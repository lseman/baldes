use super::*;

struct IntegerSearch<'a, const D: usize> {
    problem: &'a RootProblem<D>,
    pool: &'a Pool,
    by_customer: Vec<Vec<usize>>,
    customers: &'a [usize],
    fleet: usize,
    used: Vec<bool>,
    chosen: Vec<usize>,
    best: &'a mut f64,
    solution: &'a mut Vec<Route>,
    nodes: usize,
    limit: usize,
    deadline: Instant,
}
impl<const D: usize> IntegerSearch<'_, D> {
    fn search(&mut self, cost: f64) {
        if self.nodes >= self.limit || Instant::now() >= self.deadline || cost >= *self.best {
            return;
        }
        self.nodes += 1;
        let mut selected = None;
        let mut count = usize::MAX;
        for &v in self.customers {
            if self.used[v] {
                continue;
            }
            let available = self.by_customer[v]
                .iter()
                .filter(|&&r| self.pool.customers[r].iter().all(|&c| !self.used[c]))
                .count();
            if available == 0 {
                return;
            }
            if available < count {
                count = available;
                selected = Some(v);
            }
        }
        let Some(v) = selected else {
            let routes = self
                .chosen
                .iter()
                .map(|&r| self.pool.routes[r].clone())
                .collect::<Vec<_>>();
            if let Some(upper) = self.problem.unit_route_incumbent(&routes) {
                if upper < *self.best {
                    *self.best = upper;
                    *self.solution = routes;
                }
            }
            return;
        };
        if self.chosen.len() >= self.fleet {
            return;
        }
        // Binary set partitioning: branch on the route covering the most
        // constrained uncovered customer. This enumeration is only a heuristic
        // for the full problem because its column universe is restricted.
        for i in 0..self.by_customer[v].len() {
            let r = self.by_customer[v][i];
            if self.pool.customers[r].iter().any(|&c| self.used[c]) {
                continue;
            }
            for &c in &self.pool.customers[r] {
                self.used[c] = true;
            }
            self.chosen.push(r);
            self.search(cost + self.pool.routes[r].original_cost);
            self.chosen.pop();
            for &c in &self.pool.customers[r] {
                self.used[c] = false;
            }
            if self.nodes >= self.limit || Instant::now() >= self.deadline {
                break;
            }
        }
    }
}
pub(super) fn integer_heuristic<const D: usize>(
    problem: &RootProblem<D>,
    pool: &Pool,
    best: &mut f64,
    solution: &mut Vec<Route>,
    options: &TreeOptions,
    deadline: Instant,
) -> usize {
    if options.mip_node_limit == 0 {
        return 0;
    }
    let mut by_customer = vec![vec![]; problem.graph.vertices.len()];
    for (r, customers) in pool.customers.iter().enumerate() {
        for &v in customers {
            by_customer[v].push(r);
        }
    }
    for rows in &mut by_customer {
        rows.sort_by(|&a, &b| {
            (pool.routes[a].original_cost / pool.customers[a].len() as f64)
                .total_cmp(&(pool.routes[b].original_cost / pool.customers[b].len() as f64))
                .then(a.cmp(&b))
        });
    }
    let mut search = IntegerSearch {
        problem,
        pool,
        by_customer,
        customers: &problem.customers,
        fleet: problem.rows[0].rhs.floor() as usize,
        used: vec![false; problem.graph.vertices.len()],
        chosen: vec![],
        best,
        solution,
        nodes: 0,
        limit: options.mip_node_limit,
        deadline: deadline.min(Instant::now() + options.mip_time_limit),
    };
    search.search(0.);
    search.nodes
}
