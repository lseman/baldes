//! Expected recourse through Theta optimality cuts (Tomazella, Munari and
//! Morabito 2026) in the deterministic BALDES routing branch-cut-and-price.
//! Pricing is the classic capacitated one: mean demands, arc cost travel plus
//! an arc-specific recourse lower bound. The routing master separates a
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
    // At transition i -> j, direct service adds a nonnegative failure cost;
    // preventive restocking adds c_i0 + c_0j - c_ij. Thus every Bellman
    // transition is bounded below by min(0, c_i0 + c_0j - c_ij), plus the
    // suffix's lower bound. Charge that bound only on its actual arc.
    fn relaxed_arc_cost(&self, from: usize, to: usize) -> f64 {
        let via_depot = crate::core::rounding::add_down(self.costs[from][0], self.costs[0][to])
            .unwrap_or(f64::INFINITY);
        self.costs[from][to].min(via_depot)
    }

    /// Travel plus arc-specific lower bounds on replenishment recourse.
    /// This is a lower bound on the expected route cost under either policy,
    /// including rounded/nonmetric costs with negative preventive detours.
    pub fn relaxed_route_cost(&self, route: &[usize]) -> f64 {
        let mut from = 0;
        let mut cost = 0.;
        for &to in route.iter().chain(std::iter::once(&0)) {
            cost += self.relaxed_arc_cost(from, to);
            from = to;
        }
        cost
    }

    /// Classic capacitated routing graph on mean demands. Vertex `j` is
    /// customer `j`, the sink is `n + 1`. Each arc costs the smaller of direct
    /// travel and travel via the depot, so a route costs `relaxed_route_cost`. NG
    /// neighborhoods hold the `ng` nearest vertices.
    pub fn theta_cut_graph(&self, ng: usize) -> Result<Graph> {
        let n = self.demands.len();
        let sink = n + 1;
        let means: Vec<f64> = std::iter::once(0.)
            .chain(self.demands.iter().map(|d| d.mean()))
            .collect();
        // Mean sums carry rounding error; never exclude a feasible route.
        let capacity = self.capacity as f64 * (1. + 1e-9) + 1e-9;
        let mut graph = Graph::new(n + 2, 2)?;
        for customer in 1..=n {
            graph.add_elementarity_set(&[customer])?;
        }
        let mut add = |u: usize, v: usize| -> Result<()> {
            let (i, j) = (u, if v == sink { 0 } else { v });
            if u != 0 && v != sink && means[i] + means[j] > capacity {
                return Ok(());
            }
            let id = graph.add_arc(u, v, self.relaxed_arc_cost(i, j))?;
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
