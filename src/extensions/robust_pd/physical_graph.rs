use super::Problem;
use crate::graph::coupled::{
    plugin::{CoupledPlugin, PluginResource, PluginState},
    CoupledResource,
};
use crate::{invalid, Graph, Result, Window};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
struct State {
    /// Pickups already visited forward; deliveries already visited backward.
    seen: Vec<bool>,
    /// Requests crossing the current cut of the path.
    open: Vec<bool>,
}

#[derive(Debug)]
struct Pairing {
    problem: Arc<Problem>,
    arcs: Vec<(usize, usize)>,
}
impl CoupledPlugin for Pairing {
    fn validate(&self, arcs: usize) -> Result<()> {
        if arcs != self.arcs.len() {
            return Err(invalid("PD plugin arc count"));
        }
        Ok(())
    }

    fn root(&self, _forward: bool) -> PluginState {
        let n = self.problem.instance.requests;
        PluginState::new(State {
            seen: vec![false; n],
            open: vec![false; n],
        })
    }

    fn extend(
        &self,
        state: &PluginState,
        arc: usize,
        forward: bool,
    ) -> Result<Option<PluginState>> {
        let mut state = state
            .get::<State>()
            .ok_or_else(|| invalid("PD plugin state"))?
            .clone();
        let &(tail, head) = self.arcs.get(arc).ok_or_else(|| invalid("PD plugin arc"))?;
        let n = self.problem.instance.requests;
        let v = if forward { head } else { tail };
        if v == 0 || v == 2 * n + 1 {
            return Ok((!state.open.iter().any(|o| *o)).then(|| PluginState::new(state)));
        }
        let pickup = v <= n;
        let request = if pickup { v - 1 } else { v - n - 1 };
        if pickup == forward {
            if state.seen[request] {
                return Ok(None);
            }
            state.seen[request] = true;
            state.open[request] = true;
        } else {
            if !state.open[request] {
                return Ok(None);
            }
            state.open[request] = false;
        }
        // Reverse traversal checks the same physical cuts, from the end.
        if self.problem.load(&state.open) > self.problem.instance.capacity {
            return Ok(None);
        }
        Ok(Some(PluginState::new(state)))
    }

    fn join(&self, state: &PluginState, suffix: &mut dyn Iterator<Item = usize>) -> Result<bool> {
        let mut state = state.clone();
        for arc in suffix {
            let Some(next) = self.extend(&state, arc, true)? else {
                return Ok(false);
            };
            state = next;
        }
        Ok(!state
            .get::<State>()
            .ok_or_else(|| invalid("PD plugin state"))?
            .open
            .iter()
            .any(|o| *o))
    }
    // Equality dominance is conservative: both history and on-board set matter.

    fn certify(&self, arcs: &[usize], work: &mut usize, deadline: Option<Instant>) -> Option<bool> {
        use crate::core::rounding::{add_down, add_up};
        let n = self.problem.instance.requests;
        let sink = 2 * n + 1;
        let mut seen = vec![false; n];
        let mut open = vec![false; n];
        let mut vertex = 0;
        if arcs.is_empty() {
            return Some(false);
        }
        for &arc in arcs {
            // Bound both scanning and sorting work before touching this arc.
            let charge = n.checked_add(2)?.checked_mul(n)?.checked_add(1)?;
            *work = work.checked_sub(charge)?;
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            let &(tail, head) = self.arcs.get(arc)?;
            if tail != vertex || vertex == sink {
                return Some(false);
            }
            vertex = head;
            if head == sink {
                if open.iter().any(|o| *o) {
                    return Some(false);
                }
                continue;
            }
            if head == 0 || head > sink {
                return Some(false);
            }
            let r = if head <= n { head - 1 } else { head - n - 1 };
            if head <= n {
                if seen[r] {
                    return Some(false);
                }
                seen[r] = true;
                open[r] = true;
            } else {
                if !open[r] {
                    return Some(false);
                }
                open[r] = false;
            }
            // Independent outward interval for the exact-real load of this cut.
            let (mut lower, mut upper) = (0., 0.);
            let mut deviations = Vec::new();
            for (r, &onboard) in open.iter().enumerate() {
                if onboard {
                    lower = add_down(lower, self.problem.demands[r])?;
                    upper = add_up(upper, self.problem.demands[r])?;
                    deviations.push(self.problem.deviations[r]);
                }
            }
            deviations.sort_by(|a, b| b.total_cmp(a));
            for deviation in deviations
                .iter()
                .take(self.problem.parameters.demand_budget)
            {
                lower = add_down(lower, *deviation)?;
                upper = add_up(upper, *deviation)?;
            }
            if lower > self.problem.instance.capacity {
                return Some(false);
            }
            if upper > self.problem.instance.capacity {
                return None;
            }
        }
        Some(vertex == sink && !open.iter().any(|o| *o))
    }
}

impl Problem {
    pub(super) fn build_graph(&self, duals: &[f64], fleet_dual: f64) -> Result<Graph> {
        let n = self.instance.requests;
        if duals.len() != n || duals.iter().any(|d| !d.is_finite()) || !fleet_dual.is_finite() {
            return Err(invalid(
                "one finite dual per request and finite fleet dual required",
            ));
        }
        let sink = 2 * n + 1;
        let mut graph = Graph::new(sink + 1, 1)?;
        graph.source = 0;
        graph.sink = sink;
        graph.initial[0] = Some(self.instance.nodes[0].open);
        for (v, node) in self.instance.nodes.iter().enumerate() {
            graph.set_bounds(v, 0, node.open, node.close)?;
        }
        let mut arcs = vec![];
        let mut durations = vec![];
        let mut deviations = vec![];
        let mut windows = vec![];
        for tail in 0..sink {
            for head in 1..=sink {
                if tail == head
                    || (tail == 0 && head > n)
                    || (head == sink && tail <= n)
                    || tail == head + n
                {
                    continue;
                }
                let travel = self.travel[tail][head];
                let duration = self.instance.nodes[tail].service + travel;
                let deviation = (self.parameters.time_deviation * travel).ceil();
                if self.instance.nodes[tail].open
                    + duration
                    + if self.parameters.time_budget > 0 {
                        deviation
                    } else {
                        0.
                    }
                    > self.instance.nodes[head].close
                {
                    continue;
                }
                let cost = travel
                    + if tail == 0 {
                        self.parameters.vehicle_cost
                    } else {
                        0.
                    };
                let id = graph.add_arc(tail, head, cost)?;
                let arc = graph.arc_mut(id)?;
                arc.consumption[0] = duration;
                arc.reduced_cost = cost
                    - if head <= n { duals[head - 1] } else { 0. }
                    - if tail == 0 { fleet_dual } else { 0. };
                arcs.push((tail, head));
                durations.push(duration);
                deviations.push(deviation);
                windows.push(Window::new(
                    self.instance.nodes[head].open,
                    self.instance.nodes[head].close,
                )?);
            }
        }
        graph.set_coupled_resources(&[
            CoupledResource::BudgetedTime {
                initial: self.instance.nodes[0].open,
                budget: self.parameters.time_budget.min(sink),
                durations,
                deviations,
                windows,
            },
            CoupledResource::Plugin(PluginResource::new(Pairing {
                problem: Arc::new(self.clone()),
                arcs,
            })),
        ])?;
        graph.validate()?;
        Ok(graph)
    }
}
