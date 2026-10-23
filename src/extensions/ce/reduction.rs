//! Feasibility reductions: none of these depend on an integer incumbent.
use super::diagram::Diagram;

impl Diagram {
    /// Propagate independent lower bounds on load and service start through
    /// the DAG, then retain only root-to-sink arcs. Independent minima relax
    /// path correlations, so they can miss conflicts but cannot cut a feasible
    /// route. Monotone floating-point addition matches exact route replay.
    pub(super) fn reduce(&mut self) {
        let n = self.problem.demands.len();
        // Suffix sharing can join different original visit layers. Recompute
        // a topological order instead of relying on the original layer number.
        let mut incoming = vec![0usize; self.nodes.len()];
        for arc in self.arcs.iter().filter(|arc| arc.active) {
            incoming[arc.to] += 1;
        }
        let mut order: Vec<_> = (0..self.nodes.len())
            .filter(|&id| incoming[id] == 0)
            .collect();
        let mut cursor = 0;
        while cursor < order.len() {
            let from = order[cursor];
            for &id in &self.nodes[from].outgoing {
                let arc = &self.arcs[id];
                if arc.active {
                    incoming[arc.to] -= 1;
                    if incoming[arc.to] == 0 {
                        order.push(arc.to);
                    }
                }
            }
            cursor += 1;
        }
        assert_eq!(order.len(), self.nodes.len(), "CE diagram must be acyclic");
        for (depth, &id) in order.iter().enumerate() {
            self.nodes[id].depth = depth;
        }
        let mut load = vec![f64::INFINITY; self.nodes.len()];
        let mut time = vec![f64::INFINITY; self.nodes.len()];
        load[0] = 0.0;
        time[0] = self.problem.windows[0].0;
        for &from in &order {
            let location = self.nodes[from].location;
            for &id in &self.nodes[from].outgoing {
                let arc = &mut self.arcs[id];
                if !arc.active {
                    continue;
                }
                let to = self.nodes[arc.to].location;
                let next_load = load[from]
                    + if to <= n {
                        self.problem.demands[to - 1]
                    } else {
                        0.0
                    };
                let next_time = (time[from]
                    + self.problem.service_times[location]
                    + self.problem.travel_times[location][to])
                    .max(self.problem.windows[to].0);
                if !next_load.is_finite()
                    || next_load > self.problem.capacity
                    || !next_time.is_finite()
                    || next_time > self.problem.windows[to].1
                {
                    arc.active = false;
                } else {
                    load[arc.to] = load[arc.to].min(next_load);
                    time[arc.to] = time[arc.to].min(next_time);
                }
            }
        }
        let mut reaches_sink = vec![false; self.nodes.len()];
        reaches_sink[1] = true;
        let mut latest = vec![f64::NEG_INFINITY; self.nodes.len()];
        latest[1] = self.problem.windows[self.nodes[1].location].1;
        for &from in order.iter().rev() {
            for &id in &self.nodes[from].outgoing {
                let arc = &mut self.arcs[id];
                let location = self.nodes[from].location;
                let to = self.nodes[arc.to].location;
                let arrival = (time[from]
                    + self.problem.service_times[location]
                    + self.problem.travel_times[location][to])
                    .max(self.problem.windows[to].0);
                if arc.active && reaches_sink[arc.to] && arrival <= latest[arc.to] {
                    reaches_sink[from] = true;
                    // Invert each rounded addition with outward rounding. This
                    // is an upper bound, including at floating-point boundaries;
                    // ordinary subtraction could incorrectly remove a route.
                    let before_travel = upper_float(
                        upper_float(latest[arc.to]) - self.problem.travel_times[location][to],
                    );
                    let departure = upper_float(
                        upper_float(before_travel) - self.problem.service_times[location],
                    );
                    latest[from] =
                        latest[from].max(departure.min(self.problem.windows[location].1));
                } else {
                    arc.active = false;
                }
            }
        }
        let mut remap = vec![usize::MAX; self.nodes.len()];
        let mut nodes = Vec::new();
        for (id, mut node) in std::mem::take(&mut self.nodes).into_iter().enumerate() {
            if id < 2 || (load[id].is_finite() && reaches_sink[id]) {
                remap[id] = nodes.len();
                node.outgoing.clear();
                nodes.push(node);
            }
        }
        let mut arcs = Vec::new();
        for mut arc in std::mem::take(&mut self.arcs) {
            if arc.active && remap[arc.from] != usize::MAX && remap[arc.to] != usize::MAX {
                arc.from = remap[arc.from];
                arc.to = remap[arc.to];
                nodes[arc.from].outgoing.push(arcs.len());
                arcs.push(arc);
            }
        }
        self.nodes = nodes;
        self.arcs = arcs;
        self.merge_suffixes();
        self.rebuild_exact_index();
    }

    /// Identical relaxed suffixes represent exactly the same location paths.
    /// Merge them bottom-up; exact prefix states retain their resource identity.
    fn merge_suffixes(&mut self) {
        use std::collections::HashMap;
        let mut order: Vec<_> = (2..self.nodes.len()).collect();
        order.sort_unstable_by_key(|&id| std::cmp::Reverse(self.nodes[id].depth));
        let mut canonical: Vec<_> = (0..self.nodes.len()).collect();
        let mut signatures = HashMap::new();
        for id in order {
            if self.nodes[id].state.is_some() {
                continue;
            }
            let mut successors: Vec<_> = self.nodes[id]
                .outgoing
                .iter()
                .map(|&arc| {
                    let arc = &self.arcs[arc];
                    (canonical[arc.to], arc.cost.to_bits())
                })
                .collect();
            successors.sort_unstable();
            let key = (self.nodes[id].location, successors);
            if let Some(&existing) = signatures.get(&key) {
                canonical[id] = existing;
            } else {
                signatures.insert(key, id);
            }
        }
        if canonical
            .iter()
            .enumerate()
            .all(|(id, &target)| id == target)
        {
            return;
        }
        let mut remap = vec![usize::MAX; self.nodes.len()];
        let mut nodes = Vec::new();
        for (id, mut node) in std::mem::take(&mut self.nodes).into_iter().enumerate() {
            if canonical[id] == id {
                remap[id] = nodes.len();
                node.outgoing.clear();
                nodes.push(node);
            }
        }
        let mut arcs = Vec::new();
        for mut arc in std::mem::take(&mut self.arcs) {
            if canonical[arc.from] != arc.from {
                continue;
            }
            arc.from = remap[arc.from];
            arc.to = remap[canonical[arc.to]];
            nodes[arc.from].outgoing.push(arcs.len());
            arcs.push(arc);
        }
        self.nodes = nodes;
        self.arcs = arcs;
    }
}

/// Next representable upper bound, without requiring a newer Rust MSRV.
fn upper_float(value: f64) -> f64 {
    if value == f64::INFINITY || value.is_nan() {
        value
    } else if value == 0.0 {
        f64::from_bits(1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}
