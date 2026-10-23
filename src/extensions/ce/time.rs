//! Acyclic time-bucket relaxation. Bucket anchors round service starts down;
//! exact replay and prefix refinement still enforce the original resources.
use super::{
    diagram::{Diagram, Node},
    Options, Problem,
};
use crate::Result;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

impl Diagram {
    pub(super) fn time_diagram(p: &Problem, options: &Options) -> Result<Option<Self>> {
        let n = p.demands.len();
        if p.windows
            .iter()
            .any(|&(a, b)| !a.is_finite() || !b.is_finite() || a < 0.0 || b > (1u64 << 40) as f64)
        {
            return Ok(None);
        }
        let mut step = f64::INFINITY;
        for from in 1..=n {
            for to in 1..=n {
                if from != to && p.costs[from][to].is_some() {
                    step = step.min(p.service_times[from] + p.travel_times[from][to]);
                }
            }
        }
        if !step.is_finite() || step < 1.0 {
            return Ok(None);
        }
        // Power-of-two anchors are exact throughout the admitted range.
        let quantum = 2f64.powi(step.log2().floor().min(30.0) as i32);
        let bucket = |time: f64| (time / quantum).floor() as usize;
        let estimate: usize = p.windows[1..=n]
            .iter()
            .map(|&(a, b)| bucket(b) - bucket(a) + 1)
            .sum();
        if estimate > n.saturating_mul(n) / 4 || estimate + 2 > options.max_nodes {
            return Ok(None);
        }
        let mut dd = Self::empty(p);
        let mut states = HashMap::new();
        let mut pending = BinaryHeap::new();
        // Each queued state is processed once; every customer transition must
        // strictly increase its bucket, otherwise use the layered fallback.
        let mut current = Some((0usize, 0usize));
        while let Some((from, tick)) = current {
            let location = dd.nodes[from].location;
            let time = if from == 0 {
                p.windows[0].0
            } else {
                (tick as f64 * quantum).max(p.windows[location].0)
            };
            for to in 1..=n + 1 {
                if from == 0 && to == n + 1 {
                    continue;
                }
                if to == location || (to <= n && p.demands[to - 1] > p.capacity) {
                    continue;
                }
                let Some(cost) = p.costs[location][to] else {
                    continue;
                };
                let arrival = (time + p.service_times[location] + p.travel_times[location][to])
                    .max(p.windows[to].0);
                if !arrival.is_finite() || arrival > p.windows[to].1 {
                    continue;
                }
                let target = if to == n + 1 {
                    1
                } else {
                    let next = bucket(arrival);
                    if from != 0 && next <= tick {
                        return Ok(None);
                    }
                    if let Some(&id) = states.get(&(to, next)) {
                        id
                    } else {
                        if dd.nodes.len() >= options.max_nodes {
                            return Ok(None);
                        }
                        let id = dd.nodes.len();
                        dd.nodes.push(Node {
                            location: to,
                            outgoing: vec![],
                            depth: next + 1,
                            state: None,
                        });
                        states.insert((to, next), id);
                        pending.push(Reverse((next, id)));
                        id
                    }
                };
                if dd.arcs.len() >= options.max_arcs {
                    return Ok(None);
                }
                dd.add_arc(from, target, cost, options)?;
            }
            current = pending.pop().map(|Reverse((tick, id))| (id, tick));
        }
        dd.reduce();
        Ok(Some(dd))
    }
}
