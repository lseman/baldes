//! Numerical arc branch-and-price using the existing LP backend and BALDES labels.
use super::pricing::route_uses;
use super::{
    ArcBranch, ColumnGenerationOptions, ColumnGenerationStatus, Master, Policy, Problem,
    RouteEvaluation,
};
use crate::master::LpMaster;
use crate::{invalid, Result};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct SolveOptions {
    pub column_generation: ColumnGenerationOptions,
    pub max_nodes: usize,
    pub time_limit: Duration,
    pub integrality_tolerance: f64,
}
impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            column_generation: ColumnGenerationOptions::default(),
            max_nodes: 10_000,
            time_limit: Duration::from_secs(3600),
            integrality_tolerance: 1e-7,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveStatus {
    /// Exhausted arc branch-and-price at the configured floating tolerances.
    /// This is not a directed arithmetic certificate.
    Optimal,
    Infeasible,
    NodeLimit,
    TimeLimit,
    NodeIncomplete(ColumnGenerationStatus),
    NumericalFailure,
}
#[derive(Clone, Debug)]
pub struct Solution {
    pub routes: Vec<Vec<usize>>,
    pub evaluations: Vec<RouteEvaluation>,
    pub total_cost: f64,
}

#[derive(Clone, Debug)]
pub struct SolveResult {
    pub status: SolveStatus,
    pub solution: Option<Solution>,
    /// Numerical lower bound from degree/walk relaxations and fully priced
    /// node LPs. Restricted LP objectives alone are never used as bounds.
    pub lower_bound: f64,
    pub nodes: usize,
    pub pricing_calls: usize,
    pub generated_labels: usize,
}
struct Node {
    branches: Vec<ArcBranch>,
    lower: f64,
}

impl Problem {
    fn order_customers(&self, customers: &[usize]) -> Vec<usize> {
        let mut route = vec![];
        let mut remaining = customers.to_vec();
        while !remaining.is_empty() {
            let mut best = (f64::INFINITY, 0, 0);
            for (index, &customer) in remaining.iter().enumerate() {
                for k in 0..=route.len() {
                    let i = if k == 0 { 0 } else { route[k - 1] };
                    let j = if k == route.len() { 0 } else { route[k] };
                    let cost = self.costs[i][customer] + self.costs[customer][j] - self.costs[i][j];
                    if cost < best.0 {
                        best = (cost, index, k);
                    }
                }
            }
            route.insert(best.2, remaining.remove(best.1));
        }
        // A few travel-only 2-opt sweeps order the candidate. Its final
        // stochastic objective is independently evaluated before acceptance.
        let travel = |r: &[usize]| {
            let mut from = 0;
            let mut cost = 0.;
            for to in r.iter().copied().chain(std::iter::once(0)) {
                cost += self.costs[from][to];
                from = to;
            }
            cost
        };
        let mut cost = travel(&route);
        for _ in 0..3 {
            let mut improved = false;
            for a in 0..route.len() {
                for b in a + 1..route.len() {
                    route[a..=b].reverse();
                    let candidate = travel(&route);
                    if candidate < cost - 1e-9 {
                        cost = candidate;
                        improved = true;
                    } else {
                        route[a..=b].reverse();
                    }
                }
            }
            if !improved {
                break;
            }
        }
        route
    }

    /// Construct a feasible incumbent without relying on pricing completion.
    /// Failed greedy packing says nothing about feasibility.
    fn initial_solution(
        &self,
        policy: Policy,
        vehicles: usize,
        deadline: Instant,
    ) -> Result<Option<Solution>> {
        let n = self.demands.len();
        if n > 0 && vehicles == 0 {
            return Ok(None);
        }
        let mut order: Vec<_> = (1..=n).collect();
        order.sort_by(|&a, &b| {
            self.demands[b - 1]
                .mean()
                .total_cmp(&self.demands[a - 1].mean())
        });
        let pack = |order: &[usize]| -> Result<Option<Solution>> {
            let mut routes: Vec<Vec<usize>> = vec![];
            for &customer in order {
                let mut selected = None;
                let mut fullest = -1.;
                for (k, route) in routes.iter().enumerate() {
                    let load: f64 = (1..=n)
                        .filter(|j| *j == customer || route.contains(j))
                        .map(|j| self.demands[j - 1].mean())
                        .sum();
                    if load <= self.capacity as f64 && load > fullest {
                        fullest = load;
                        selected = Some(k);
                    }
                }
                if selected.is_none() {
                    if routes.len() == vehicles {
                        return Ok(None);
                    }
                    routes.push(vec![]);
                    selected = Some(routes.len() - 1);
                }
                let route = &mut routes[selected.expect("available bin")];
                let position = (0..=route.len())
                    .min_by(|&a, &b| {
                        let delta = |k: usize| {
                            let i = if k == 0 { 0 } else { route[k - 1] };
                            let j = if k == route.len() { 0 } else { route[k] };
                            self.costs[i][customer] + self.costs[customer][j] - self.costs[i][j]
                        };
                        delta(a).total_cmp(&delta(b))
                    })
                    .expect("insertion position");
                route.insert(position, customer);
            }
            if n > 9 && Instant::now() < deadline {
                for route in &mut routes {
                    *route = self.order_customers(route);
                }
            }
            let evaluations: Vec<_> = routes
                .iter()
                .map(|r| self.evaluate_route(r, policy))
                .collect::<Result<_>>()?;
            let total_cost = evaluations.iter().map(|e| e.total_cost).sum();
            Ok(Some(Solution {
                routes,
                evaluations,
                total_cost,
            }))
        };
        let mut best = pack(&order)?;
        let mut random = 0x6a09e667f3bcc909u64;
        if n > 9 {
            for attempt in 0..n + 64 {
                if Instant::now() >= deadline {
                    break;
                }
                let mut trial: Vec<_> = (1..=n).collect();
                if attempt < n {
                    // Geographic order from each possible first customer.
                    let mut from = trial.remove(attempt);
                    let mut nearest = vec![from];
                    while !trial.is_empty() {
                        let index = (0..trial.len())
                            .min_by(|&a, &b| {
                                self.costs[from][trial[a]].total_cmp(&self.costs[from][trial[b]])
                            })
                            .unwrap();
                        from = trial.remove(index);
                        nearest.push(from);
                    }
                    trial = nearest;
                } else {
                    for k in (1..trial.len()).rev() {
                        random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                        trial.swap(k, (random as usize) % (k + 1));
                    }
                }
                if let Some(solution) = pack(&trial)? {
                    if best
                        .as_ref()
                        .map_or(true, |s| solution.total_cost < s.total_cost)
                    {
                        best = Some(solution);
                    }
                }
            }
        }
        Ok(best)
    }

    // A bounded exact-cover search over known routes is a primal heuristic:
    // failure/exhaustion never changes a dual bound or proves infeasibility.
    fn improve_from_pool(
        &self,
        policy: Policy,
        vehicles: usize,
        pool: &BTreeSet<Vec<usize>>,
        incumbent: &mut Option<Solution>,
        deadline: Instant,
    ) -> Result<()> {
        if Instant::now() >= deadline || self.demands.is_empty() {
            return Ok(());
        }
        let n = self.demands.len();
        let mut best = std::collections::BTreeMap::<u64, (Vec<usize>, RouteEvaluation)>::new();
        for route in pool {
            if Instant::now() >= deadline {
                return Ok(());
            }
            let mask = route.iter().fold(0u64, |m, &j| m | (1 << (j - 1)));
            let evaluation = self.evaluate_route(route, policy)?;
            // For a two-vehicle solution, a priced route plus its complement
            // supplies a useful primal candidate even when the complementary
            // column has not yet been priced.
            if vehicles >= 2 {
                let complement: Vec<_> = (1..=n).filter(|&j| mask & (1 << (j - 1)) == 0).collect();
                let load: f64 = complement.iter().map(|&j| self.demands[j - 1].mean()).sum();
                if !complement.is_empty() && load <= self.capacity as f64 {
                    let other = self.order_customers(&complement);
                    let other_evaluation = self.evaluate_route(&other, policy)?;
                    let cost = evaluation.total_cost + other_evaluation.total_cost;
                    if incumbent.as_ref().map_or(true, |s| cost < s.total_cost) {
                        *incumbent = Some(Solution {
                            routes: vec![route.clone(), other],
                            evaluations: vec![evaluation, other_evaluation],
                            total_cost: cost,
                        });
                    }
                }
            }
            if best
                .get(&mask)
                .map_or(true, |(_, e)| evaluation.total_cost < e.total_cost)
            {
                best.insert(mask, (route.clone(), evaluation));
            }
        }
        let columns: Vec<_> = best.into_iter().collect();
        let mut containing = vec![vec![]; n];
        for (k, (mask, _)) in columns.iter().enumerate() {
            for (j, list) in containing.iter_mut().enumerate() {
                if mask & (1 << j) != 0 {
                    list.push(k);
                }
            }
        }
        for list in &mut containing {
            list.sort_by(|&a, &b| {
                let score =
                    |k: usize| columns[k].1 .1.total_cost / columns[k].0.count_ones() as f64;
                score(a).total_cmp(&score(b))
            });
        }
        struct Search<'a> {
            columns: &'a [(u64, (Vec<usize>, RouteEvaluation))],
            containing: &'a [Vec<usize>],
            all: u64,
            vehicles: usize,
            work: usize,
            deadline: Instant,
            best_cost: f64,
            best: Option<Vec<usize>>,
        }
        impl Search<'_> {
            fn visit(&mut self, covered: u64, cost: f64, chosen: &mut Vec<usize>) {
                if covered == self.all {
                    if cost < self.best_cost {
                        self.best_cost = cost;
                        self.best = Some(chosen.clone());
                    }
                    return;
                }
                if self.work == 0
                    || chosen.len() == self.vehicles
                    || Instant::now() >= self.deadline
                {
                    return;
                }
                self.work -= 1;
                let customer = (self.all & !covered).trailing_zeros() as usize;
                for index in 0..self.containing[customer].len() {
                    let k = self.containing[customer][index];
                    let (mask, (_, evaluation)) = &self.columns[k];
                    if covered & mask != 0 {
                        continue;
                    }
                    let (next, cost) = (covered | mask, cost + evaluation.total_cost);
                    chosen.push(k);
                    self.visit(next, cost, chosen);
                    chosen.pop();
                    if self.work == 0 || Instant::now() >= self.deadline {
                        break;
                    }
                }
            }
        }
        let mut search = Search {
            columns: &columns,
            containing: &containing,
            all: (1u64 << n) - 1,
            vehicles,
            work: 20_000,
            deadline,
            best_cost: incumbent.as_ref().map_or(f64::INFINITY, |s| s.total_cost),
            best: None,
        };
        search.visit(0, 0., &mut vec![]);
        if let Some(chosen) = search.best {
            let routes = chosen.iter().map(|&k| columns[k].1 .0.clone()).collect();
            let evaluations = chosen.iter().map(|&k| columns[k].1 .1).collect();
            *incumbent = Some(Solution {
                routes,
                evaluations,
                total_cost: search.best_cost,
            });
        }
        Ok(())
    }

    fn initial_lower_bound(&self) -> f64 {
        // Every customer has one incoming and one outgoing arc. Dropping depot
        // arcs and capacity gives two independent degree relaxations. Account
        // for possibly negative recourse on rounded/nonmetric cost matrices.
        let n = self.demands.len();
        let outgoing: f64 = (1..=n)
            .map(|i| {
                (0..=n)
                    .filter(|&j| j != i)
                    .map(|j| self.costs[i][j])
                    .fold(f64::INFINITY, f64::min)
            })
            .sum();
        let incoming: f64 = (1..=n)
            .map(|j| {
                (0..=n)
                    .filter(|&i| i != j)
                    .map(|i| self.costs[i][j])
                    .fold(f64::INFINITY, f64::min)
            })
            .sum();
        let bound = outgoing.max(incoming) + self.recourse_lower_bound();
        (bound - 1e-9 * (1. + bound.abs())).max(0.)
    }

    /// Solve using a factory for fresh, empty LP backends. Each node performs
    /// complete two-phase column generation before it may be bounded/pruned.
    /// Branches split a fractional directed arc flow into zero and one.
    /// No restricted-pool integer solve is used as an optimality argument.
    pub fn solve_with<M: LpMaster>(
        &self,
        policy: Policy,
        vehicles: usize,
        options: SolveOptions,
        mut factory: impl FnMut() -> Result<M>,
    ) -> Result<SolveResult> {
        options.column_generation.validate()?;
        if !options.integrality_tolerance.is_finite()
            || options.integrality_tolerance <= 0.
            || options.integrality_tolerance >= 0.5
        {
            return Err(invalid("invalid stochastic integrality tolerance"));
        }
        self.pricing_graph(policy, &vec![0.; self.demands.len()], 0., &[], &[], true)?;
        let started = Instant::now();
        let initial_lower = self.initial_lower_bound();
        let mut result = SolveResult {
            status: SolveStatus::NodeLimit,
            solution: self.initial_solution(
                policy,
                vehicles,
                started + options.time_limit.min(Duration::from_millis(25)),
            )?,
            lower_bound: initial_lower,
            nodes: 0,
            pricing_calls: 0,
            generated_labels: 0,
        };
        let mut pending = vec![Node {
            branches: vec![],
            lower: initial_lower,
        }];
        let mut pool = BTreeSet::<Vec<usize>>::new();
        for j in 1..=self.demands.len() {
            pool.insert(vec![j]);
        }
        if let Some(solution) = &result.solution {
            pool.extend(solution.routes.iter().cloned());
        }
        let mut closed_lower = f64::INFINITY;
        while !pending.is_empty() {
            if started.elapsed() >= options.time_limit {
                result.status = SolveStatus::TimeLimit;
                break;
            }
            if result.nodes >= options.max_nodes {
                result.status = SolveStatus::NodeLimit;
                break;
            }
            let node = pending.pop().expect("nonempty node stack");
            if result
                .solution
                .as_ref()
                .is_some_and(|s| node.lower >= s.total_cost)
            {
                closed_lower = closed_lower.min(node.lower);
                continue;
            }
            let mut master = Master::new(
                self.clone(),
                policy,
                vehicles,
                node.branches.clone(),
                factory()?,
            )?;
            for route in &pool {
                if node
                    .branches
                    .iter()
                    .any(|b| !b.required && route_uses(route, *b))
                {
                    continue;
                }
                master.add_route(route)?;
            }
            let mut cg = options.column_generation;
            cg.time_limit = cg
                .time_limit
                .min(options.time_limit.saturating_sub(started.elapsed()));
            let mut observed_columns = 0;
            let answer = master.optimize_with_observer(cg, |columns, snapshot| {
                if cg.ng_size > 0
                    && self.demands.len() > cg.ng_size
                    && snapshot.phase == super::MasterPhase::ExpectedCost
                    && columns.len() > observed_columns
                {
                    for column in &columns[observed_columns..] {
                        pool.insert(column.customers.clone());
                    }
                    observed_columns = columns.len();
                    let remaining = options.time_limit.saturating_sub(started.elapsed());
                    self.improve_from_pool(
                        policy,
                        vehicles,
                        &pool,
                        &mut result.solution,
                        Instant::now() + remaining.min(Duration::from_millis(8)),
                    )?;
                }
                Ok(())
            })?;
            result.nodes += 1;
            result.pricing_calls += answer.pricing_calls;
            result.generated_labels = result
                .generated_labels
                .saturating_add(answer.generated_labels);
            for column in master.columns() {
                pool.insert(column.customers.clone());
            }
            // A feasible integer restricted LP solution is an incumbent even
            // if pricing stopped; only its optimality still needs a full bound.
            if let Some(snapshot) = answer.snapshot.as_ref().filter(|s| {
                s.phase == super::MasterPhase::ExpectedCost && s.artificial_usage <= cg.lp_tolerance
            }) {
                let mut covered = vec![0usize; self.demands.len()];
                let mut routes = vec![];
                let mut evaluations = vec![];
                let mut valid = true;
                for (column, &x) in master.columns().iter().zip(&snapshot.route_values) {
                    if (x - x.round()).abs() > options.integrality_tolerance
                        || !(0. ..=1.).contains(&x.round())
                    {
                        valid = false;
                        break;
                    }
                    if x.round() == 1. {
                        for &j in &column.customers {
                            covered[j - 1] += 1;
                        }
                        evaluations.push(self.evaluate_route(&column.customers, policy)?);
                        routes.push(column.customers.clone());
                    }
                }
                valid &= covered.iter().all(|&v| v == 1) && routes.len() <= vehicles;
                let total_cost: f64 = evaluations.iter().map(|e| e.total_cost).sum();
                if valid
                    && total_cost.is_finite()
                    && result
                        .solution
                        .as_ref()
                        .map_or(true, |s| total_cost < s.total_cost)
                {
                    result.solution = Some(Solution {
                        routes,
                        evaluations,
                        total_cost,
                    });
                }
            }
            // Reserve at most a small slice for primal work after this node.
            let remaining = options.time_limit.saturating_sub(started.elapsed());
            let heuristic_deadline = Instant::now() + remaining.min(Duration::from_millis(20));
            self.improve_from_pool(
                policy,
                vehicles,
                &pool,
                &mut result.solution,
                heuristic_deadline,
            )?;
            if answer.status == ColumnGenerationStatus::Infeasible {
                continue;
            }
            if answer.status != ColumnGenerationStatus::Converged {
                result.status = if started.elapsed() >= options.time_limit {
                    SolveStatus::TimeLimit
                } else {
                    SolveStatus::NodeIncomplete(answer.status)
                };
                pending.push(Node {
                    lower: answer.lower_bound.unwrap_or(node.lower).max(node.lower),
                    ..node
                });
                break;
            }
            let snapshot = answer
                .snapshot
                .as_ref()
                .ok_or_else(|| invalid("missing stochastic node snapshot"))?;
            let lower = answer
                .lower_bound
                .ok_or_else(|| invalid("missing stochastic node lower bound"))?
                .max(node.lower);
            if result
                .solution
                .as_ref()
                .is_some_and(|s| lower >= s.total_cost)
            {
                closed_lower = closed_lower.min(lower);
                continue;
            }
            let integral = snapshot
                .route_values
                .iter()
                .all(|x| (*x - x.round()).abs() <= options.integrality_tolerance);
            if integral {
                let mut routes = vec![];
                let mut evaluations = vec![];
                let mut covered = vec![false; self.demands.len()];
                let mut valid = true;
                let mut cost = 0.;
                for (column, &x) in master.columns().iter().zip(&snapshot.route_values) {
                    if x.round() == 0. {
                        continue;
                    }
                    if x.round() != 1. {
                        valid = false;
                        break;
                    }
                    for &j in &column.customers {
                        if covered[j - 1] {
                            valid = false;
                        }
                        covered[j - 1] = true;
                    }
                    let evaluation = self.evaluate_route(&column.customers, policy)?;
                    cost += evaluation.total_cost;
                    routes.push(column.customers.clone());
                    evaluations.push(evaluation);
                }
                valid &= covered.iter().all(|&v| v) && routes.len() <= vehicles && cost.is_finite();
                for b in &node.branches {
                    valid &= routes.iter().filter(|r| route_uses(r, *b)).count()
                        == usize::from(b.required);
                }
                // Integrality tolerances cannot close a genuine LP/integer gap.
                let gap_tolerance = cg.lp_tolerance * (1. + cost.abs())
                    + self.demands.len() as f64 * cg.reduced_cost_tolerance;
                valid &= lower <= cost + gap_tolerance && cost - lower <= 2. * gap_tolerance;
                if !valid {
                    result.status = SolveStatus::NumericalFailure;
                    pending.push(Node { lower, ..node });
                    break;
                }
                if result
                    .solution
                    .as_ref()
                    .map_or(true, |s| cost < s.total_cost)
                {
                    result.solution = Some(Solution {
                        routes,
                        evaluations,
                        total_cost: cost,
                    });
                }
                closed_lower = closed_lower.min(lower);
                continue;
            }
            let n = self.demands.len();
            let mut flow = vec![vec![0.; n + 1]; n + 1];
            for (column, &x) in master.columns().iter().zip(&snapshot.route_values) {
                let mut i = 0;
                for j in column.customers.iter().copied().chain(std::iter::once(0)) {
                    flow[i][j] += x;
                    i = j;
                }
            }
            let mut chosen = None;
            let mut distance = f64::INFINITY;
            for (i, row) in flow.iter().enumerate() {
                for (j, &x) in row.iter().enumerate() {
                    if i == j
                        || x <= options.integrality_tolerance
                        || x >= 1. - options.integrality_tolerance
                        || node.branches.iter().any(|b| b.from == i && b.to == j)
                    {
                        continue;
                    }
                    if (x - 0.5).abs() < distance {
                        distance = (x - 0.5).abs();
                        chosen = Some((i, j));
                    }
                }
            }
            let Some((from, to)) = chosen else {
                result.status = SolveStatus::NumericalFailure;
                pending.push(Node { lower, ..node });
                break;
            };
            for required in [false, true] {
                let mut branches = node.branches.clone();
                branches.push(ArcBranch { from, to, required });
                pending.push(Node { branches, lower });
            }
        }
        if pending.is_empty() {
            result.status = if result.solution.is_some() {
                SolveStatus::Optimal
            } else {
                SolveStatus::Infeasible
            };
        }
        result.lower_bound = pending.iter().map(|n| n.lower).fold(closed_lower, f64::min);
        if let Some(solution) = &result.solution {
            result.lower_bound = result.lower_bound.min(solution.total_cost);
        }
        Ok(result)
    }

    /// Clp-backed convenience entrypoint for numerical arc branch-and-price.
    #[cfg(feature = "clp")]
    pub fn solve(
        &self,
        policy: Policy,
        vehicles: usize,
        options: SolveOptions,
    ) -> Result<SolveResult> {
        self.solve_with(
            policy,
            vehicles,
            options,
            crate::master::clp::ClpMaster::new,
        )
    }
}
