//! Conventional routing solve with independently evaluated HGS seeds.
use super::{HgsWarmStart, HgsWarmStartSummary, Policy, Problem};
use crate::labeling::Route;
use crate::master::routing::RoutingContext;
use crate::master::tree::{TreeOptions, TreeOutput};
use crate::{invalid, Result};
use std::time::Instant;

/// Stochastic routing controls. HGS and routing share `tree.time_limit`.
#[derive(Clone, Debug)]
pub struct SolveOptions {
    pub tree: TreeOptions,
    pub ng_size: usize,
    pub hgs_warm_start: Option<HgsWarmStart>,
}
impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            tree: TreeOptions::default(),
            ng_size: 8,
            hgs_warm_start: None,
        }
    }
}

#[derive(Debug)]
pub struct SolveResult {
    pub tree: TreeOutput,
    pub hgs_warm_start: Option<HgsWarmStartSummary>,
}

impl Problem {
    /// Solve using conventional pricing and Theta recourse cuts. HGS proposals
    /// become initial columns only after original stochastic feasibility checks;
    /// a complete validated plan also becomes the tree's initial incumbent.
    pub fn solve(
        &self,
        policy: Policy,
        vehicles: usize,
        mut options: SolveOptions,
    ) -> Result<SolveResult> {
        let started = Instant::now();
        let mut summary = None;
        if let Some(warm) = options.hgs_warm_start {
            let (columns, solution, counts) = self.hgs_seed(
                policy,
                vehicles,
                warm,
                options.tree.time_limit.saturating_sub(started.elapsed()),
            )?;
            let context = self.theta_cut_context(policy, vehicles, options.ng_size)?;
            for customers in columns {
                options
                    .tree
                    .root
                    .initial_routes
                    .push(seed_route(&context, &customers)?);
            }
            if let Some(solution) = solution {
                let routes = solution
                    .routes
                    .iter()
                    .map(|customers| seed_route(&context, customers))
                    .collect::<Result<Vec<_>>>()?;
                let old = if options.tree.initial_solution.is_empty() {
                    None
                } else {
                    Some(
                        context
                            .unit_route_incumbent(&options.tree.initial_solution)
                            .ok_or_else(|| invalid("invalid supplied initial solution"))?,
                    )
                };
                if old.is_none_or(|cost| solution.total_cost < cost) {
                    options.tree.initial_solution = routes;
                }
            }
            summary = Some(counts);
        }
        options.tree.time_limit = options.tree.time_limit.saturating_sub(started.elapsed());
        options.tree.root.cg.time_limit =
            options.tree.root.cg.time_limit.min(options.tree.time_limit);
        let tree = self.solve_theta_cuts(policy, vehicles, options.ng_size, options.tree)?;
        Ok(SolveResult {
            tree,
            hgs_warm_start: summary,
        })
    }
}

fn seed_route(context: &RoutingContext, customers: &[usize]) -> Result<Route> {
    let graph = &context.model.networks[0].graph;
    let mut from = graph.source;
    let mut arcs = Vec::with_capacity(customers.len() + 1);
    for to in customers.iter().copied().chain(std::iter::once(graph.sink)) {
        let arc = graph
            .arcs
            .iter()
            .find(|a| a.tail == from && a.head == to)
            .ok_or_else(|| invalid("HGS route has no routing arc"))?;
        arcs.push(arc.id);
        from = to;
    }
    context
        .elementary_route(&arcs)
        .ok_or_else(|| invalid("HGS route failed routing resource replay"))
}
