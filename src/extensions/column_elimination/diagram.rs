use super::{Options, Problem};
use crate::{incomplete, Result};

#[derive(Clone, Debug)]
pub(super) struct Node {
    pub location: usize,
    pub outgoing: Vec<usize>,
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
}

impl Diagram {
    pub fn new(problem: &Problem, options: &Options) -> Result<Self> {
        let n = problem.demands.len();
        if n.checked_mul(n)
            .and_then(|v| v.checked_add(2))
            .map_or(true, |v| v > options.max_nodes)
        {
            return Err(incomplete("column elimination initial node limit reached"));
        }
        let mut dd = Self {
            nodes: vec![
                Node {
                    location: 0,
                    outgoing: vec![],
                },
                Node {
                    location: n + 1,
                    outgoing: vec![],
                },
            ],
            arcs: vec![],
        };
        // Layer counts customer visits. Every elementary route has at most n.
        for _ in 0..n {
            for location in 1..=n {
                dd.nodes.push(Node {
                    location,
                    outgoing: vec![],
                });
            }
        }
        for customer in 1..=n {
            if let Some(cost) = problem.costs[0][customer] {
                dd.add_arc(0, customer + 1, cost, options)?;
            }
        }
        for layer in 0..n {
            for customer in 1..=n {
                let from = 2 + layer * n + customer - 1;
                if let Some(cost) = problem.costs[customer][n + 1] {
                    dd.add_arc(from, 1, cost, options)?;
                }
                if layer + 1 == n {
                    continue;
                }
                for next in 1..=n {
                    if next == customer {
                        continue;
                    }
                    if let Some(cost) = problem.costs[customer][next] {
                        dd.add_arc(from, 2 + (layer + 1) * n + next - 1, cost, options)?;
                    }
                }
            }
        }
        Ok(dd)
    }

    fn add_arc(&mut self, from: usize, to: usize, cost: f64, options: &Options) -> Result<()> {
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

    /// Isolate this root prefix by copying successor nodes and redirecting
    /// only the selected incoming arc. All other incoming paths keep their
    /// original successors. The final conflicting transition is then removed.
    /// Preflight limits so a failed refinement leaves the diagram untouched.
    pub fn separate(&mut self, prefix: &[usize], options: &Options) -> Result<bool> {
        let copies = prefix.len() - 1;
        let extra_arcs: usize = prefix[..copies]
            .iter()
            .map(|&a| {
                self.nodes[self.arcs[a].to]
                    .outgoing
                    .iter()
                    .filter(|&&id| self.arcs[id].active)
                    .count()
            })
            .sum();
        if copies > options.max_nodes.saturating_sub(self.nodes.len())
            || extra_arcs > options.max_arcs.saturating_sub(self.arcs.len())
        {
            return Ok(false);
        }
        let mut current = 0;
        for (index, &old_arc) in prefix.iter().enumerate() {
            // A preceding split copied this arc into the isolated node.
            let selected = if current == self.arcs[old_arc].from {
                old_arc
            } else {
                let old = &self.arcs[old_arc];
                *self.nodes[current]
                    .outgoing
                    .iter()
                    .find(|&&id| {
                        let a = &self.arcs[id];
                        a.active && a.to == old.to
                    })
                    .expect("copied prefix arc")
            };
            if index == copies {
                self.arcs[selected].active = false;
                return Ok(true);
            }
            let old_to = self.arcs[selected].to;
            let cloned = self.nodes.len();
            self.nodes.push(Node {
                location: self.nodes[old_to].location,
                outgoing: vec![],
            });
            for id in self.nodes[old_to].outgoing.clone() {
                let arc = self.arcs[id].clone();
                if arc.active {
                    self.add_arc(cloned, arc.to, arc.cost, options)?;
                }
            }
            self.arcs[selected].to = cloned;
            current = cloned;
        }
        unreachable!("nonempty conflict prefix")
    }
}
