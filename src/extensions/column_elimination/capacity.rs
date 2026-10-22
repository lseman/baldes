//! A compact capacity-indexed relaxation with two-customer ng neighborhoods.
use super::{
    diagram::{Diagram, Node},
    Options, Problem,
};
use crate::Result;

impl Diagram {
    pub(super) fn capacity_diagram(problem: &Problem, options: &Options) -> Result<Option<Self>> {
        let n = problem.demands.len();
        if problem.windows.iter().any(|&(_, due)| due.is_finite())
            || problem
                .demands
                .iter()
                .any(|&q| q <= 0.0 || q.fract() != 0.0 || q > u32::MAX as f64)
            || problem.capacity > u32::MAX as f64
        {
            return Ok(None);
        }
        let gcd = |mut a: usize, mut b: usize| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let unit = problem
            .demands
            .iter()
            .map(|&q| q as usize)
            .reduce(gcd)
            .unwrap();
        let capacity = (problem.capacity as usize) / unit;
        // Preflight a conservative dense-state budget. Prefer the layered
        // fallback to exhausting limits on a large pseudo-polynomial model.
        let Some(states) = capacity
            .checked_add(1)
            .and_then(|q| q.checked_mul(n))
            .and_then(|s| s.checked_mul(2))
        else {
            return Ok(None);
        };
        if capacity > n.saturating_mul(4)
            || states.saturating_add(2) > options.max_nodes
            || states.saturating_mul(n + 1).saturating_add(n) > options.max_arcs
        {
            return Ok(None);
        }
        let demand: Vec<_> = problem.demands.iter().map(|&q| q as usize / unit).collect();
        // A neighborhood contains the customer and its cheapest reachable
        // neighbor. A state remembers whether that neighbor was recently seen.
        let neighbor: Vec<_> = (1..=n)
            .map(|from| {
                (1..=n)
                    .filter(|&to| to != from && problem.costs[from][to].is_some())
                    .min_by(|&a, &b| {
                        problem.costs[from][a]
                            .unwrap()
                            .total_cmp(&problem.costs[from][b].unwrap())
                    })
                    .unwrap_or(from)
            })
            .collect();
        let mut dd = Self::empty(problem);
        let mut index = vec![usize::MAX; states];
        for load in 0..=capacity {
            let sources: Vec<_> = if load == 0 {
                vec![(0, 0, false)]
            } else {
                (1..=n)
                    .flat_map(|last| [false, true].map(move |remembered| (last, remembered)))
                    .filter_map(|(last, remembered)| {
                        let id = index[(load * n + last - 1) * 2 + usize::from(remembered)];
                        (id != usize::MAX).then_some((id, last, remembered))
                    })
                    .collect()
            };
            for (from, last, remembered) in sources {
                if last != 0 {
                    if let Some(cost) = problem.costs[last][n + 1] {
                        dd.add_arc(from, 1, cost, options)?;
                    }
                }
                for next in 1..=n {
                    if next == last || (remembered && next == neighbor[last - 1]) {
                        continue;
                    }
                    let Some(cost) = problem.costs[last][next] else {
                        continue;
                    };
                    if demand[next - 1] > capacity - load {
                        continue;
                    }
                    let next_load = load + demand[next - 1];
                    let next_remembered = last == neighbor[next - 1]
                        || (remembered && neighbor[last - 1] == neighbor[next - 1]);
                    let slot = (next_load * n + next - 1) * 2 + usize::from(next_remembered);
                    let to = if index[slot] != usize::MAX {
                        index[slot]
                    } else {
                        let id = dd.nodes.len();
                        dd.nodes.push(Node {
                            location: next,
                            outgoing: vec![],
                            depth: next_load,
                            state: None,
                        });
                        index[slot] = id;
                        id
                    };
                    dd.add_arc(from, to, cost, options)?;
                }
            }
        }
        dd.reduce();
        Ok(Some(dd))
    }
}
