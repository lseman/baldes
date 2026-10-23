use super::state::ExactState;
use super::{Options, Problem};
use crate::{incomplete, Result};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub(super) struct Node {
    pub location: usize,
    pub outgoing: Vec<usize>,
    pub depth: usize,
    pub state: Option<ExactState>,
}

#[derive(Clone, Debug)]
pub(super) struct Arc {
    pub from: usize,
    pub to: usize,
    pub cost: f64,
    pub active: bool,
}

pub(super) struct Diagram {
    pub nodes: Vec<Node>,
    pub arcs: Vec<Arc>,
    pub problem: Problem,
    exact_nodes: HashMap<(usize, ExactState), usize>,
}

impl Diagram {
    /// Resolve a location prefix against the current diagram. Earlier splits
    /// may redirect its arcs or remove it entirely. Each node has at most one
    /// active successor per location, and splitting preserves this invariant.
    pub fn separate_locations(
        &mut self,
        locations: &[usize],
        options: &Options,
    ) -> Result<Option<bool>> {
        let mut current = 0;
        let mut prefix = Vec::with_capacity(locations.len());
        for &location in locations {
            let Some(id) = self.nodes[current].outgoing.iter().copied().find(|&id| {
                let arc = &self.arcs[id];
                arc.active && self.nodes[arc.to].location == location
            }) else {
                return Ok(None);
            };
            prefix.push(id);
            current = self.arcs[id].to;
        }
        self.separate(&prefix, options).map(Some)
    }

    pub fn new(problem: &Problem, options: &Options) -> Result<Self> {
        let n = problem.demands.len();
        if n >= 20 {
            if let Some(dd) = Self::time_diagram(problem, options)? {
                return Ok(dd);
            }
            if let Some(dd) = Self::capacity_diagram(problem, options)? {
                return Ok(dd);
            }
        }
        if n.checked_mul(n)
            .and_then(|v| v.checked_add(2))
            .map_or(true, |v| v > options.max_nodes)
        {
            return Err(incomplete("column elimination initial node limit reached"));
        }
        let mut dd = Self::empty(problem);
        // Necessary local feasibility checks. No shortest-path or triangle
        // inequality assumption: even leaving at the source window's earliest
        // time must permit service at the destination before its deadline.
        let possible = |from: usize, to: usize| {
            let arrival = (problem.windows[from].0
                + problem.service_times[from]
                + problem.travel_times[from][to])
                .max(problem.windows[to].0);
            arrival.is_finite()
                && arrival <= problem.windows[to].1
                && (from == 0 || problem.demands[from - 1] <= problem.capacity)
                && (to == n + 1 || problem.demands[to - 1] <= problem.capacity)
        };
        // Layer counts customer visits. Every elementary route has at most n.
        for layer in 0..n {
            for location in 1..=n {
                dd.nodes.push(Node {
                    location,
                    outgoing: vec![],
                    depth: layer + 1,
                    state: None,
                });
            }
        }
        for customer in 1..=n {
            if !possible(0, customer) {
                continue;
            }
            if let Some(cost) = problem.costs[0][customer] {
                dd.add_arc(0, customer + 1, cost, options)?;
            }
        }
        for layer in 0..n {
            for customer in 1..=n {
                let from = 2 + layer * n + customer - 1;
                if possible(customer, n + 1) {
                    if let Some(cost) = problem.costs[customer][n + 1] {
                        dd.add_arc(from, 1, cost, options)?;
                    }
                }
                if layer + 1 == n {
                    continue;
                }
                for next in 1..=n {
                    if next == customer || !possible(customer, next) {
                        continue;
                    }
                    if let Some(cost) = problem.costs[customer][next] {
                        dd.add_arc(from, 2 + (layer + 1) * n + next - 1, cost, options)?;
                    }
                }
            }
        }
        dd.reduce();
        Ok(dd)
    }

    pub(super) fn empty(problem: &Problem) -> Self {
        Self {
            nodes: vec![
                Node {
                    location: 0,
                    outgoing: vec![],
                    depth: 0,
                    state: Some(ExactState::root(problem)),
                },
                Node {
                    location: problem.demands.len() + 1,
                    outgoing: vec![],
                    depth: usize::MAX,
                    state: None,
                },
            ],
            arcs: vec![],
            problem: problem.clone(),
            exact_nodes: HashMap::new(),
        }
    }

    pub(super) fn rebuild_exact_index(&mut self) {
        self.exact_nodes.clear();
        for (id, node) in self.nodes.iter().enumerate() {
            if let Some(state) = &node.state {
                self.exact_nodes.insert((node.location, state.clone()), id);
            }
        }
    }

    pub(super) fn add_arc(
        &mut self,
        from: usize,
        to: usize,
        cost: f64,
        options: &Options,
    ) -> Result<()> {
        if self.arcs.len() >= options.max_arcs {
            return Err(incomplete("column elimination arc limit reached"));
        }
        self.nodes[from].outgoing.push(self.arcs.len());
        self.arcs.push(Arc {
            from,
            to,
            cost,
            active: true,
        });
        Ok(())
    }

    /// Refine a root prefix into exact resource states. Equivalent states are
    /// shared, and all infeasible outgoing transitions are filtered at once.
    /// Preflight limits so a failed refinement leaves the diagram untouched.
    pub fn separate(&mut self, prefix: &[usize], options: &Options) -> Result<bool> {
        let last = prefix.len() - 1;
        let mut copies = 0;
        let mut extra_arcs = 0;
        let mut state = ExactState::root(&self.problem);
        let mut current = 0;
        for &arc in &prefix[..last] {
            let next_location = self.nodes[self.arcs[arc].to].location;
            let Some(selected) = self.nodes[current].outgoing.iter().copied().find(|&id| {
                let arc = &self.arcs[id];
                arc.active && self.nodes[arc.to].location == next_location
            }) else {
                break;
            };
            let target = self.arcs[selected].to;
            let Some(next) =
                state.advance(&self.problem, self.nodes[current].location, next_location)
            else {
                break;
            };
            if let Some(&existing) = self.exact_nodes.get(&(next_location, next.clone())) {
                current = existing;
            } else {
                copies += 1;
                extra_arcs += self.nodes[target]
                    .outgoing
                    .iter()
                    .filter(|&&id| {
                        let arc = &self.arcs[id];
                        arc.active
                            && next.allows(
                                &self.problem,
                                next_location,
                                self.nodes[arc.to].location,
                            )
                    })
                    .count();
                // A planned clone inherits this target's outgoing choices.
                current = target;
            }
            state = next;
        }
        if copies > options.max_nodes.saturating_sub(self.nodes.len())
            || extra_arcs > options.max_arcs.saturating_sub(self.arcs.len())
        {
            return Ok(false);
        }
        let mut current = 0;
        let mut state = ExactState::root(&self.problem);
        for (index, &old_arc) in prefix.iter().enumerate() {
            let location = self.nodes[self.arcs[old_arc].to].location;
            let Some(selected) = self.nodes[current].outgoing.iter().copied().find(|&id| {
                let arc = &self.arcs[id];
                arc.active && self.nodes[arc.to].location == location
            }) else {
                // An earlier exact-state filter already removed this conflict.
                return Ok(true);
            };
            let Some(next) = state.advance(&self.problem, self.nodes[current].location, location)
            else {
                self.arcs[selected].active = false;
                return Ok(true);
            };
            if index == last {
                // Callers supply a prefix ending at its first infeasibility.
                debug_assert!(false, "refinement prefix must end in a conflict");
                return Ok(true);
            }
            let old_to = self.arcs[selected].to;
            let key = (location, next.clone());
            let exact = if let Some(&id) = self.exact_nodes.get(&key) {
                id
            } else {
                let id = self.nodes.len();
                self.nodes.push(Node {
                    location,
                    outgoing: vec![],
                    depth: self.nodes[old_to].depth,
                    state: Some(next.clone()),
                });
                for old_id in self.nodes[old_to].outgoing.clone() {
                    let arc = self.arcs[old_id].clone();
                    if arc.active
                        && next.allows(&self.problem, location, self.nodes[arc.to].location)
                    {
                        self.add_arc(id, arc.to, arc.cost, options)?;
                    }
                }
                self.exact_nodes.insert(key, id);
                id
            };
            self.arcs[selected].to = exact;
            current = exact;
            state = next;
        }
        unreachable!("nonempty conflict prefix")
    }
}
