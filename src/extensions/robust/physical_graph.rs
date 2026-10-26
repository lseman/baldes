//! Robust VRPTW on the ordinary physical-graph engine.
//!
//! Scalar resources hold the nominal (γ = 0) service start and load, so
//! buckets, completion bounds and arc fixing work on a valid relaxation.
//! The coupled `BudgetedTime` and `BudgetedLoad` resources hold the full
//! worst-case vectors and decide robust feasibility, including joins.
use super::Problem;
use crate::graph::coupled::CoupledResource;
use crate::{invalid, Graph, Result, Window};

impl Problem {
    /// Coupled resources for a graph whose arcs mirror `arcs` (tail, head)
    /// in arc-vector order. Heads equal to `sink` use the depot's data.
    pub fn budgeted_resources(
        &self,
        arcs: &[(usize, usize)],
        sink: usize,
    ) -> Result<Vec<CoupledResource>> {
        let node = |v: usize| if v == sink { self.depot } else { v };
        let mut durations = Vec::with_capacity(arcs.len());
        let mut time_deviations = Vec::with_capacity(arcs.len());
        let mut windows = Vec::with_capacity(arcs.len());
        let mut demands = Vec::with_capacity(arcs.len());
        let mut demand_deviations = Vec::with_capacity(arcs.len());
        for &(tail, head) in arcs {
            let (tail, head) = (node(tail), node(head));
            let travel =
                self.travel[tail][head].ok_or_else(|| invalid("unavailable robust arc"))?;
            durations.push(self.nodes[tail].service + travel.time);
            time_deviations.push(travel.time_deviation);
            let (open, close) = self.nodes[head].window;
            windows.push(Window::new(open, close)?);
            demands.push(self.nodes[head].demand);
            demand_deviations.push(self.nodes[head].demand_deviation);
        }
        Ok(vec![
            CoupledResource::BudgetedTime {
                initial: self.nodes[self.depot].window.0,
                budget: self.time_budget,
                durations,
                deviations: time_deviations,
                windows,
            },
            CoupledResource::BudgetedLoad {
                capacity: self.capacity,
                budget: self.demand_budget,
                demands,
                deviations: demand_deviations,
            },
        ])
    }

    /// Pricing graph under customer duals (zero at the depot) and a fleet
    /// dual charged once per route. Vertex ids equal node ids; one extra
    /// vertex, `nodes.len()`, is the sink copy of the depot. Each customer's
    /// NG set holds its `ng_size` nearest customers by nominal travel time,
    /// itself included; `ng_size >= customers` prices elementary routes.
    /// Arcs that no single-arc robust extension can use are omitted.
    pub fn pricing_graph(&self, duals: &[f64], fleet_dual: f64, ng_size: usize) -> Result<Graph> {
        self.validate()?;
        let n = self.nodes.len();
        let customers: Vec<usize> = (0..n).filter(|&v| v != self.depot).collect();
        let element = |c: usize| c - usize::from(c > self.depot);
        let memory = (0..=n)
            .map(|v| {
                if v == self.depot || v == n {
                    return vec![];
                }
                let time = |c: usize| self.travel[v][c].map_or(f64::INFINITY, |t| t.time);
                let mut near = customers.clone();
                near.sort_by(|&a, &b| {
                    (a != v)
                        .cmp(&(b != v))
                        .then(time(a).total_cmp(&time(b)))
                        .then(a.cmp(&b))
                });
                near.truncate(ng_size.max(1));
                near.into_iter().map(element).collect()
            })
            .collect::<Vec<_>>();
        self.pricing_graph_with_memory(duals, fleet_dual, &memory)
    }

    /// As `pricing_graph`, with explicit NG memories: one list of element ids
    /// per vertex, sink included. Customer c is element c, less one when it
    /// follows the depot.
    pub fn pricing_graph_with_memory(
        &self,
        duals: &[f64],
        fleet_dual: f64,
        memory: &[Vec<usize>],
    ) -> Result<Graph> {
        self.validate()?;
        let n = self.nodes.len();
        if duals.len() != n || duals[self.depot] != 0. || duals.iter().any(|d| !d.is_finite()) {
            return Err(invalid(
                "robust duals need one finite value per node, zero at the depot",
            ));
        }
        if memory.len() != n + 1 {
            return Err(invalid("robust NG memory needs one list per vertex"));
        }
        let sink = n;
        let mut graph = Graph::new(n + 1, 2)?;
        graph.source = self.depot;
        graph.sink = sink;
        for v in (0..n).filter(|&v| v != self.depot) {
            graph.add_elementarity_set(&[v])?;
        }
        let mut arcs = Vec::new();
        let root = self.initial_state();
        for tail in 0..n {
            for head in 0..n {
                let Some(travel) = self.travel[tail][head] else {
                    continue;
                };
                if tail == head || (tail == self.depot && head == self.depot) {
                    continue;
                }
                // Necessary robust condition: the tail at its earliest start
                // and least worst-case load, then this arc alone.
                let node = &self.nodes[tail];
                let mut from = root.clone();
                from.start
                    .iter_mut()
                    .for_each(|s| *s = s.max(node.window.0));
                for (g, load) in from.load.iter_mut().enumerate() {
                    *load = node.demand + if g > 0 { node.demand_deviation } else { 0. };
                }
                if self.extend(&from, tail, head).is_none() {
                    continue;
                }
                let to = if head == self.depot { sink } else { head };
                let id = graph.add_arc(tail, to, travel.cost)?;
                let arc = graph.arc_mut(id)?;
                arc.reduced_cost =
                    travel.cost - duals[head] - if tail == self.depot { fleet_dual } else { 0. };
                arc.consumption = [node.service + travel.time, self.nodes[head].demand].into();
                arcs.push((tail, to));
            }
        }
        for v in 0..=sink {
            let node = &self.nodes[if v == sink { self.depot } else { v }];
            graph.set_bounds(v, 0, node.window.0, node.window.1)?;
            graph.set_bounds(v, 1, 0., self.capacity)?;
            graph.set_ng(v, &memory[v])?;
        }
        graph.initial[0] = Some(self.nodes[self.depot].window.0);
        graph.initial[1] = Some(0.);
        graph.set_coupled_resources(&self.budgeted_resources(&arcs, sink)?)?;
        graph.validate()?;
        Ok(graph)
    }

    /// Node sequence `depot, ..., depot` of a pricing route's vertex list.
    pub fn route_nodes(&self, vertices: &[i32]) -> Vec<usize> {
        let sink = self.nodes.len();
        vertices
            .iter()
            .map(|&v| {
                if v as usize == sink {
                    self.depot
                } else {
                    v as usize
                }
            })
            .collect()
    }
}
