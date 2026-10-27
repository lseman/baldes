use super::Problem;
use crate::graph::coupled::{
    plugin::{CoupledPlugin, PluginResource, PluginState},
    CoupledResource,
};
use crate::graph::RouteDomain;
use crate::{invalid, Graph, Result, Window};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
struct State {
    /// NG memory of requests whose pickup may not be entered next: picked
    /// up within the memory, or proven unreachable from this label.
    memory: Vec<bool>,
    /// Requests crossing the current cut of the path.
    open: Vec<bool>,
    /// The γ = 0 time entry: earliest service start forward, latest start
    /// excluding the vertex's own window backward. Used only for marking.
    time: f64,
}

#[derive(Debug)]
struct Pairing {
    problem: Arc<Problem>,
    arcs: Vec<(usize, usize)>,
    /// Tail service plus nominal travel, per arc.
    durations: Vec<f64>,
    /// Shortest nominal duration between vertices over the graph's arcs; a
    /// lower bound on any path, so marking needs no triangle inequality.
    shortest: Vec<Vec<f64>>,
    /// NG neighbourhood of each vertex as request flags.
    neighbours: Vec<Vec<bool>>,
    /// Every neighbourhood holds every request, so routes are elementary.
    elementary: bool,
}

impl Pairing {
    fn window(&self, v: usize) -> (f64, f64) {
        let node = &self.problem.instance.nodes[v];
        (node.open, node.close)
    }
    /// Whether no continuation after service starts at `v` no earlier than
    /// `time` can serve request `k`: pickup, delivery and return in time.
    fn unreachable_forward(&self, v: usize, time: f64, k: usize) -> bool {
        let n = self.problem.instance.requests;
        let (p, d, sink) = (k + 1, k + 1 + n, 2 * n + 1);
        let (ep, lp) = self.window(p);
        let pickup = ep.max(time + self.shortest[v][p]);
        let (ed, ld) = self.window(d);
        let delivery = ed.max(pickup + self.shortest[p][d]);
        !(pickup <= lp
            && delivery <= ld
            && delivery + self.shortest[d][sink] <= self.window(sink).1)
    }
    /// Whether no prefix before `v`, whose latest start excluding its own
    /// window is `latest`, can serve request `k`.
    fn unreachable_backward(&self, v: usize, latest: f64, k: usize) -> bool {
        let n = self.problem.instance.requests;
        let (p, d) = (k + 1, k + 1 + n);
        let at_v = self.window(v).1.min(latest);
        let (ed, ld) = self.window(d);
        let delivery_latest = ld.min(at_v - self.shortest[d][v]);
        let (ep, lp) = self.window(p);
        let pickup_earliest = ep.max(self.window(0).0 + self.shortest[0][p]);
        let pickup_latest = lp.min(delivery_latest - self.shortest[p][d]);
        !(pickup_earliest <= pickup_latest
            && ed.max(pickup_earliest + self.shortest[p][d]) <= delivery_latest)
    }
}

impl CoupledPlugin for Pairing {
    fn validate(&self, arcs: usize) -> Result<()> {
        if arcs != self.arcs.len() || arcs != self.durations.len() {
            return Err(invalid("PD plugin arc count"));
        }
        Ok(())
    }

    fn root(&self, forward: bool) -> PluginState {
        let n = self.problem.instance.requests;
        PluginState::new(State {
            memory: vec![false; n],
            open: vec![false; n],
            time: if forward {
                self.window(0).0
            } else {
                f64::INFINITY
            },
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
        let (open_time, close_time) = self.window(head);
        state.time = if forward {
            open_time.max(state.time + self.durations[arc])
        } else {
            close_time.min(state.time) - self.durations[arc]
        };
        let n = self.problem.instance.requests;
        let v = if forward { head } else { tail };
        if v == 0 || v == 2 * n + 1 {
            return Ok((!state.open.iter().any(|o| *o)).then(|| PluginState::new(state)));
        }
        let pickup = v <= n;
        let request = if pickup { v - 1 } else { v - n - 1 };
        // A remembered pickup closes an NG cycle in either direction. With
        // complete memories a backward delivery after its pickup also would.
        if pickup == forward {
            if state.open[request] || (state.memory[request] && (pickup || self.elementary)) {
                return Ok(None);
            }
            state.open[request] = true;
        } else {
            if !state.open[request] || (pickup && state.memory[request]) {
                return Ok(None);
            }
            state.open[request] = false;
        }
        for (m, &keep) in state.memory.iter_mut().zip(&self.neighbours[v]) {
            *m &= keep;
        }
        if pickup {
            state.memory[request] = true;
        }
        // Reverse traversal checks the same physical cuts, from the end.
        if self.problem.load(&state.open) > self.problem.instance.capacity {
            return Ok(None);
        }
        for k in 0..n {
            if !state.memory[k]
                && !state.open[k]
                && if forward {
                    self.unreachable_forward(v, state.time, k)
                } else {
                    self.unreachable_backward(v, state.time, k)
                }
            {
                state.memory[k] = true;
            }
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

    /// Equal on-board sets and a smaller memory. No delivery is skipped, so
    /// no triangle inequality is assumed; budgeted times are compared by the
    /// kernel, and marking is monotone in its nominal time entry.
    fn dominates(&self, a: &PluginState, b: &PluginState, _forward: bool) -> bool {
        let (Some(a), Some(b)) = (a.get::<State>(), b.get::<State>()) else {
            return false;
        };
        a.open == b.open && a.memory.iter().zip(&b.memory).all(|(a, b)| !a || *b)
    }

    fn certify(&self, arcs: &[usize], work: &mut usize, deadline: Option<Instant>) -> Option<bool> {
        use crate::core::rounding::{add_down, add_up};
        let n = self.problem.instance.requests;
        let sink = 2 * n + 1;
        // NG memory of picked-up requests, as in `extend`.
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
                if seen[r] || open[r] {
                    return Some(false);
                }
                open[r] = true;
            } else {
                if !open[r] {
                    return Some(false);
                }
                open[r] = false;
            }
            // The pricing domain: NG memory without unreachable marking.
            // Complete neighbourhoods make it elementary; a repeated pickup
            // has coefficient 2 in an equality cover row, never integral.
            for (m, &keep) in seen.iter_mut().zip(&self.neighbours[head]) {
                *m &= keep;
            }
            if head <= n {
                seen[r] = true;
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
        let mut shortest = vec![vec![f64::INFINITY; sink + 1]; sink + 1];
        for (v, row) in shortest.iter_mut().enumerate() {
            row[v] = 0.;
        }
        for (&(tail, head), &duration) in arcs.iter().zip(&durations) {
            shortest[tail][head] = shortest[tail][head].min(duration);
        }
        for k in 0..=sink {
            for i in 0..=sink {
                for j in 0..=sink {
                    let through = shortest[i][k] + shortest[k][j];
                    if through < shortest[i][j] {
                        shortest[i][j] = through;
                    }
                }
            }
        }
        // Each customer remembers its own request and the requests whose
        // pickups are nearest by travel time.
        let size = self.parameters.ng_size.max(1);
        let neighbours = (0..=sink)
            .map(|v| {
                let mut near = vec![size >= n; n];
                if size < n {
                    let own = (1..sink).contains(&v).then(|| (v - 1) % n);
                    let mut order: Vec<usize> = (0..n).collect();
                    order.sort_by(|&a, &b| {
                        (Some(a) != own)
                            .cmp(&(Some(b) != own))
                            .then(self.travel[v][a + 1].total_cmp(&self.travel[v][b + 1]))
                            .then(a.cmp(&b))
                    });
                    for &r in &order[..size] {
                        near[r] = true;
                    }
                }
                near
            })
            .collect();
        graph.set_coupled_resources(&[
            CoupledResource::BudgetedTime {
                initial: self.instance.nodes[0].open,
                budget: self.parameters.time_budget.min(sink),
                durations: durations.clone(),
                deviations,
                windows,
            },
            CoupledResource::Plugin(PluginResource::new(Pairing {
                problem: Arc::new(self.clone()),
                arcs,
                durations,
                shortest,
                neighbours,
                elementary: size >= n,
            })),
        ])?;
        // NG routes may revisit a pickup after delivering it, so the route
        // domain must admit walks. Twice the elementary length keeps every
        // elementary route and every walk that repeats each request once.
        if size < n {
            graph.set_route_domain(RouteDomain::BoundedWalks {
                max_arcs: 2 * (2 * n + 1),
            })?;
        }
        graph.validate()?;
        Ok(graph)
    }
}
