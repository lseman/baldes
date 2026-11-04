//! Expected recourse through Theta optimality cuts (Tomazella, Munari and
//! Morabito 2026) in the deterministic BALDES routing branch-cut-and-price.
//! Pricing is the classic capacitated one: mean demands, arc cost travel plus
//! the head customer's recourse lower bound. The routing master separates a
//! Theta cut for every integral route whose expected cost exceeds that.
use super::{Policy, Problem};
use crate::master::model::separation::{CapacityCutConfig, Recourse, RouteRecourse};
use crate::master::model::Problem as Model;
use crate::master::tree::{ModelTreeOptions, ModelTreeOutput};
use crate::{invalid, Graph, Result};
use std::sync::Arc;

/// Expected route cost above `relaxed_route_cost`, from the exact Bellman
/// evaluation of the route under `policy`.
struct ExpectedRecourse {
    problem: Problem,
    policy: Policy,
}

impl RouteRecourse for ExpectedRecourse {
    fn excess(&self, customers: &[usize]) -> Result<f64> {
        let total = self
            .problem
            .evaluate_route(customers, self.policy)?
            .total_cost;
        let excess = total - self.problem.relaxed_route_cost(customers);
        if excess < -1e-7 * (1. + total.abs()) {
            return Err(invalid("relaxed route cost exceeds expected route cost"));
        }
        Ok(excess.max(0.))
    }
}

impl Problem {
    // A customer can incur at most one negative preventive detour per visit.
    // Charging each customer's own bound is tighter than n times the worst
    // detour, and remains safe for rounded/nonmetric distance matrices.
    pub(super) fn recourse_customer_lower_bounds(&self) -> Vec<f64> {
        (1..=self.demands.len())
            .map(|j| {
                (0..=self.demands.len())
                    .map(|i| self.costs[i][0] + self.costs[0][j] - self.costs[i][j])
                    .fold(0., f64::min)
                    * (1. + 1e-10)
            })
            .collect()
    }

    /// Travel cost plus per-customer recourse lower bounds: a lower bound on
    /// the expected cost of the route under either policy.
    pub fn relaxed_route_cost(&self, route: &[usize]) -> f64 {
        let charges = self.recourse_customer_lower_bounds();
        let mut from = 0;
        let mut cost = 0.;
        for &to in route.iter().chain(std::iter::once(&0)) {
            cost += self.costs[from][to];
            if to != 0 {
                cost += charges[to - 1];
            }
            from = to;
        }
        cost
    }

    /// Classic capacitated routing graph on mean demands. Vertex `j` is
    /// customer `j`, the sink is `n + 1`. Arc costs add the head customer's
    /// recourse lower bound, so a route costs `relaxed_route_cost`. NG
    /// neighborhoods hold the `ng` nearest vertices.
    pub fn theta_cut_graph(&self, ng: usize) -> Result<Graph> {
        let n = self.demands.len();
        let sink = n + 1;
        let means: Vec<f64> = std::iter::once(0.)
            .chain(self.demands.iter().map(|d| d.mean()))
            .collect();
        // Mean sums carry rounding error; never exclude a feasible route.
        let capacity = self.capacity as f64 * (1. + 1e-9) + 1e-9;
        let charges = self.recourse_customer_lower_bounds();
        let mut graph = Graph::new(n + 2, 2)?;
        for customer in 1..=n {
            graph.add_elementarity_set(&[customer])?;
        }
        let mut add = |u: usize, v: usize| -> Result<()> {
            let (i, j) = (u, if v == sink { 0 } else { v });
            if u != 0 && v != sink && means[i] + means[j] > capacity {
                return Ok(());
            }
            let charge = if j == 0 { 0. } else { charges[j - 1] };
            let id = graph.add_arc(u, v, self.costs[i][j] + charge)?;
            graph.arc_mut(id)?.consumption = [0., means[j]].into();
            Ok(())
        };
        for j in 1..=n {
            add(0, j)?;
            add(j, sink)?;
            for k in 1..=n {
                if j != k {
                    add(j, k)?;
                }
            }
        }
        // A conservative recourse charge can make an individual arc slightly
        // negative. Vertex potentials preserve every source-to-sink route cost
        // while moving that charge to other arcs on the same route.
        if graph.arcs.iter().any(|arc| arc.cost < 0.) {
            let mut potential = vec![0.; n + 2];
            for _ in 0..n + 1 {
                let mut changed = false;
                for arc in &graph.arcs {
                    let candidate = potential[arc.tail] + arc.cost;
                    if candidate < potential[arc.head] {
                        potential[arc.head] = candidate;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            if potential[sink] != 0.
                || graph
                    .arcs
                    .iter()
                    .any(|arc| potential[arc.tail] + arc.cost < potential[arc.head])
            {
                return Err(invalid(
                    "stochastic relaxed graph cannot be reweighted to nonnegative arcs",
                ));
            }
            for arc in &mut graph.edit().arcs {
                arc.cost = (arc.cost + potential[arc.tail] - potential[arc.head]).max(0.);
                arc.reduced_cost = arc.cost;
            }
        }
        for v in 0..=sink {
            let i = if v == sink { 0 } else { v };
            graph.set_bounds(v, 0, 0., 1.)?;
            graph.set_bounds(v, 1, 0., capacity)?;
            let distance = |a: usize| {
                let a = if a == sink { 0 } else { a };
                self.costs[i][a] + self.costs[a][i]
            };
            let mut order: Vec<_> = (0..=sink).collect();
            order.sort_by(|&a, &b| distance(a).total_cmp(&distance(b)).then(a.cmp(&b)));
            order.truncate(ng.min(n + 2));
            let elements: Vec<_> = order
                .into_iter()
                .filter(|&vertex| vertex > 0 && vertex <= n)
                .map(|vertex| vertex - 1)
                .collect();
            graph.set_ng(v, &elements)?;
        }
        graph.edit().initial[0] = Some(0.);
        Ok(graph)
    }

    /// One-network set-partitioning model with at most `vehicles` routes and
    /// the recourse oracle whose excess Theta cuts impose.
    pub fn theta_cut_model(
        &self,
        policy: Policy,
        vehicles: usize,
        ng: usize,
    ) -> Result<(Model, Recourse)> {
        if vehicles == 0 {
            return Err(invalid("Theta-cut routing requires at least one vehicle"));
        }
        let problem = Model::set_partitioning(self.theta_cut_graph(ng)?, vehicles as f64)?;
        Ok((
            problem,
            Recourse(Arc::new(ExpectedRecourse {
                problem: self.clone(),
                policy,
            })),
        ))
    }

    /// Rounded capacity cuts on mean demands; every route respects them.
    pub fn mean_capacity_cut_config(&self) -> CapacityCutConfig {
        let demands = std::iter::once(0.)
            .chain(self.demands.iter().map(|d| d.mean()))
            .collect();
        CapacityCutConfig::new(demands, self.capacity as f64 * (1. + 1e-9))
    }

    /// Branch-cut-and-price with Theta optimality cuts on the network-model
    /// tree, branching on route count and then arc flow. Rank-one cuts are
    /// enabled, with eight cuts per round unless explicitly overridden.
    /// Capacity cuts on mean demands are added unless `options.node`
    /// already configures them. Dynamic NG strengthening runs after capacity
    /// separation, with up to 15 remembered customers per vertex and on
    /// average. Explicit pricing options override these limits;
    /// `BALDESdynamicNGmode = 0` disables it.
    /// Load-indexed walk bounds prune non-improving exact-pricing completions by
    /// default, conditioning on up to four selected NG elements.
    /// `BALDESloadCompletionSubsetSize = 0` keeps unconditional bounds only;
    /// `BALDESloadCompletionResource = -1` disables both.
    /// NG mode defaults to -1: strengthening pauses once rank-one rows are
    /// present. Set `BALDESdynamicNGmode = 1` to continue after their insertion.
    pub fn solve_theta_cuts(
        &self,
        policy: Policy,
        vehicles: usize,
        ng: usize,
        mut options: ModelTreeOptions,
    ) -> Result<ModelTreeOutput> {
        let node = &mut options.node;
        for (name, _) in &mut node.pricing_options {
            *name = crate::core::options::Options::canonical_name(name)?.into();
        }
        let separation = &mut node.separation;
        separation.rank_one.enabled = true;
        if separation.relaxation_improvement_priority == 0. {
            separation.relaxation_improvement_priority = 1.;
        }
        if separation.capacity.is_none() {
            separation.capacity = Some(self.mean_capacity_cut_config());
        }
        let size = 15.min(crate::graph::ng::NG_MEMORY_CAPACITY) as f64;
        // Fleet, customer, capacity and Theta prices are all on physical arcs
        // in this formulation, so zero is the improving-route threshold.
        for (name, value) in [
            ("BALDESrankOneCutsMaxNumPerRound", 8.),
            ("BALDESloadCompletionResource", 1.),
            ("BALDESloadCompletionSubsetSize", 4.),
            ("BALDESdynamicNGmode", -1.),
            ("BALDESmaxNGneighbourhoodSize", size),
            ("BALDESmaxNGaverNeighbourhoodSize", size),
        ] {
            if !node.pricing_options.iter().any(|(key, _)| key == name) {
                node.pricing_options.push((name.into(), value));
            }
        }
        let (problem, recourse) = self.theta_cut_model(policy, vehicles, ng)?;
        node.separation.route_recourse = Some(recourse);
        if options.branching_registry.mapped_variables.is_empty() {
            options.branching_registry.register_batch(
                problem
                    .arc_flow_variables(1.)
                    .into_iter()
                    .map(crate::master::model::BranchingRegistration::Mapped),
            )?;
        }
        crate::master::tree::solve(problem, options)
    }
}
