//! Conventional routing solve with independently evaluated HGS seeds.
use super::{HgsWarmStart, HgsWarmStartSummary, Policy, Problem};
use crate::labeling::Route;
use crate::master::model::{FixedRoute, NetworkRoute};
use crate::master::tree::{ModelTreeOptions, ModelTreeOutput};
use crate::Graph;
use crate::{invalid, Result};
use std::time::Instant;

/// Stochastic routing controls. HGS and routing share `tree.time_limit`.
#[derive(Clone, Debug)]
pub struct SolveOptions {
    pub tree: ModelTreeOptions,
    pub ng_size: usize,
    pub hgs_warm_start: Option<HgsWarmStart>,
    /// Optional true-recourse refinement of an existing initial plan. Zero disables it.
    /// Its work is charged to `tree.time_limit`; pricing objectives stay unchanged.
    pub incumbent_search_time_limit: std::time::Duration,
}
impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            tree: ModelTreeOptions::default(),
            ng_size: 8,
            hgs_warm_start: None,
            incumbent_search_time_limit: std::time::Duration::ZERO,
        }
    }
}

#[derive(Debug)]
pub struct SolveResult {
    pub tree: ModelTreeOutput,
    pub hgs_warm_start: Option<HgsWarmStartSummary>,
    pub incumbent_search: Option<super::IncumbentSearchSummary>,
    /// Source of the reported lower bound; a relaxation does not prove pricing completion.
    pub lower_bound_source: &'static str,
}

impl Problem {
    /// Solve using conventional pricing and Theta recourse cuts. HGS proposals
    /// become initial columns only after original stochastic feasibility checks;
    /// a complete validated plan also becomes the tree's initial incumbent.
    pub fn solve(
        &self,
        policy: Policy,
        vehicles: usize,
        options: SolveOptions,
    ) -> Result<SolveResult> {
        self.solve_formulation(policy, vehicles, options, false)
    }

    pub(crate) fn solve_formulation(
        &self,
        policy: Policy,
        vehicles: usize,
        mut options: SolveOptions,
        bellman: bool,
    ) -> Result<SolveResult> {
        if vehicles == 0 {
            return Err(invalid("stochastic routing requires a positive fleet"));
        }
        if bellman && policy != Policy::OptimalRestocking {
            return Err(invalid("Bellman pricing requires optimal restocking"));
        }
        if bellman {
            // The scalar NG fork can use the shared load/subset completion
            // bounds once the Bellman resource is removed. Match the Theta
            // adapter's mean-load default, preserving explicit overrides.
            for (name, _) in &mut options.tree.node.pricing_options {
                *name = crate::core::options::Options::canonical_name(name)?.into();
            }
            for (name, value) in [
                ("BALDESloadCompletionResource", 1.),
                ("BALDESloadCompletionSubsetSize", 6.),
            ] {
                if !options
                    .tree
                    .node
                    .pricing_options
                    .iter()
                    .any(|(key, _)| key == name)
                {
                    options.tree.node.pricing_options.push((name.into(), value));
                }
            }
        }
        let make_graph = |reversed: bool| -> Result<Graph> {
            let mut graph = self.theta_cut_graph(options.ng_size)?;
            if bellman {
                // Recover the exact pricer's advancing hop coordinate. NG
                // candidate pricing separately selects mean load as its axis.
                let hops = (self.demands().len() + 1) as f64;
                for vertex in 0..graph.vertices.len() {
                    graph.set_bounds(vertex, 0, 0., hops)?;
                }
                graph.set_bounds(graph.source, 0, 0., 0.)?;
                graph.edit().main_resource = Some(0);
                // The historical exact pricer kept the whole visited set in
                // the engine's NG word. Larger instances retain the plugin's
                // exact mask when the selected build's NG word is smaller.
                graph.edit().elementary =
                    reversed && self.demands().len() <= crate::graph::ng::NG_MEMORY_CAPACITY;
                let sink = graph.sink;
                let endpoints: Vec<_> = graph
                    .arcs
                    .iter()
                    .map(|a| (a.tail, if a.head == sink { 0 } else { a.head }))
                    .collect();
                for (arc, &(i, j)) in graph.edit().arcs.iter_mut().zip(&endpoints) {
                    arc.cost = self.costs()[i][j];
                    arc.reduced_cost = arc.cost;
                    arc.consumption[0] = 1.;
                    if reversed {
                        // Reverse the route while relabeling the depot copies
                        // so public source/sink IDs stay 0/n+1.
                        let tail = arc.tail;
                        arc.tail = if arc.head == sink { 0 } else { arc.head };
                        arc.head = if tail == 0 { sink } else { tail };
                        arc.consumption[1] = if i == 0 {
                            0.
                        } else {
                            self.demands()[i - 1].mean()
                        };
                    }
                }
                use crate::extensions::stochastic_restocking::RestockingResource;
                use crate::graph::coupled::{CoupledResource, PluginResource};
                let resource = RestockingResource::new(self.clone(), endpoints)?;
                let resource = if reversed {
                    resource.with_reversed_pricing()
                } else {
                    resource
                };
                graph.set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(
                    resource,
                ))])?;
            }
            Ok(graph)
        };
        let started = Instant::now();
        let mut summary = None;
        if bellman
            && (!options.tree.node.seed_routes.is_empty()
                || options.tree.initial_solution.is_some())
        {
            let graph = make_graph(true)?;
            let reverse = |route: &mut Route| -> Result<()> {
                let ids: Vec<_> = route.arcs.iter().copied().rev().collect();
                *route = graph
                    .elementary_route(&ids)?
                    .ok_or_else(|| invalid("initial Bellman route failed reversed replay"))?;
                Ok(())
            };
            for route in &mut options.tree.node.seed_routes {
                reverse(&mut route.route)?;
            }
            if let Some((_, routes)) = &mut options.tree.initial_solution {
                for fixed in routes {
                    reverse(&mut fixed.route.route)?;
                }
            }
        }
        if let Some(warm) = options.hgs_warm_start {
            let (columns, solution, counts) = self.hgs_seed(
                policy,
                vehicles,
                warm,
                options.tree.time_limit.saturating_sub(started.elapsed()),
            )?;
            let graph = make_graph(bellman)?;
            for customers in columns {
                options.tree.node.seed_routes.push(NetworkRoute {
                    network: 0,
                    route: seed_route(&graph, &customers, bellman)?,
                });
            }
            if let Some(solution) = solution {
                let routes = solution
                    .routes
                    .iter()
                    .map(|customers| {
                        Ok(FixedRoute {
                            route: NetworkRoute {
                                network: 0,
                                route: seed_route(&graph, customers, bellman)?,
                            },
                            quantity: 1.,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let old = options
                    .tree
                    .initial_solution
                    .as_ref()
                    .map(|(cost, _)| *cost);
                if old.is_none_or(|cost| solution.total_cost < cost) {
                    options.tree.initial_solution = Some((solution.total_cost, routes));
                }
            }
            summary = Some(counts);
        }
        let mut incumbent_search = None;
        let search_budget = options
            .incumbent_search_time_limit
            .min(options.tree.time_limit.saturating_sub(started.elapsed()));
        if !search_budget.is_zero() {
            if let Some((old_cost, initial)) = &options.tree.initial_solution {
                if initial
                    .iter()
                    .all(|r| r.quantity == 1. && r.route.network == 0)
                {
                    let graph = make_graph(bellman)?;
                    let routes = initial
                        .iter()
                        .map(|fixed| {
                            let replay = graph
                                .elementary_route(&fixed.route.route.arcs)?
                                .ok_or_else(|| {
                                    invalid("infeasible initial route for incumbent search")
                                })?;
                            let nodes = &replay.nodes;
                            let mut route: Vec<_> = nodes[1..nodes.len() - 1]
                                .iter()
                                .map(|&v| v as usize)
                                .collect();
                            if bellman {
                                route.reverse();
                            }
                            Ok(route)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let deadline = Instant::now()
                        .checked_add(search_budget)
                        .ok_or_else(|| invalid("incumbent search deadline overflow"))?;
                    let (improved, statistics) =
                        self.search_incumbent(policy, vehicles, routes, deadline)?;
                    if improved.total_cost < *old_cost {
                        let graph = make_graph(bellman)?;
                        let routes = improved
                            .routes
                            .iter()
                            .map(|customers| {
                                Ok(FixedRoute {
                                    route: NetworkRoute {
                                        network: 0,
                                        route: seed_route(&graph, customers, bellman)?,
                                    },
                                    quantity: 1.,
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        options
                            .tree
                            .node
                            .seed_routes
                            .extend(routes.iter().map(|r| r.route.clone()));
                        options.tree.initial_solution = Some((improved.total_cost, routes));
                    }
                    incumbent_search = Some(statistics);
                }
            }
        }
        options.tree.time_limit = options.tree.time_limit.saturating_sub(started.elapsed());
        options.tree.node.cg.time_limit =
            options.tree.node.cg.time_limit.min(options.tree.time_limit);
        let mut tree = if bellman {
            let graph = make_graph(bellman)?;
            let model = crate::master::model::Problem::set_partitioning(graph, vehicles as f64)?;
            options.tree.node.separation.rank_one.enabled = true;
            options.tree.node.separation.capacity = Some(self.mean_capacity_cut_config());
            if options.tree.branching_registry.mapped_variables.is_empty() {
                options.tree.branching_registry.register_batch(
                    model
                        .arc_flow_variables(1.)
                        .into_iter()
                        .map(crate::master::model::BranchingRegistration::Mapped),
                )?;
            }
            crate::master::tree::solve(model, options.tree)?
        } else {
            self.solve_theta_cuts(policy, vehicles, options.ng_size, options.tree)?
        };
        if bellman && !tree.routes.is_empty() {
            let original = make_graph(false)?;
            for fixed in &mut tree.routes {
                let route = &mut fixed.route.route;
                let ids: Vec<_> = route.arcs.iter().copied().rev().collect();
                *route = original
                    .elementary_route(&ids)?
                    .ok_or_else(|| invalid("Bellman solution failed original-direction replay"))?;
            }
        }
        Ok(SolveResult {
            tree,
            hgs_warm_start: summary,
            incumbent_search,
            lower_bound_source: "Tree",
        })
    }
}

fn seed_route(graph: &Graph, customers: &[usize], reversed: bool) -> Result<Route> {
    let customers: Vec<_> = if reversed {
        customers.iter().copied().rev().collect()
    } else {
        customers.to_vec()
    };
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
    graph
        .elementary_route(&arcs)?
        .ok_or_else(|| invalid("HGS route failed routing resource replay"))
}
