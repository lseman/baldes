//! Ryan–Foster branching constraints enforced with binary resources.
use crate::{invalid, Graph, Result};

/// Restrict the resource at an arc's head, after consumption and waiting.
/// Both sides include the threshold, so their union covers every schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccumulatedResourceBranch {
    pub arc: i32,
    pub resource: usize,
    pub threshold: f64,
    /// Upper side: resource >= threshold; otherwise resource <= threshold.
    pub upper: bool,
}
impl AccumulatedResourceBranch {
    pub(crate) fn window(&self, graph: &Graph) -> Result<Option<crate::graph::Window>> {
        if self.resource >= graph.resources.len() || !self.threshold.is_finite() {
            return Err(invalid("accumulated-resource branch coordinate/threshold"));
        }
        let arc = graph.arcs.iter().find(|a| a.id == self.arc)
            .ok_or_else(|| invalid("accumulated-resource branch arc ID"))?;
        let mut window = graph.vertices[arc.head].bounds[self.resource];
        if let Some(w) = arc.bounds[self.resource] {
            window.lower = window.lower.max(w.lower);
            window.upper = window.upper.min(w.upper);
        }
        if self.upper { window.lower = window.lower.max(self.threshold); }
        else { window.upper = window.upper.min(self.threshold); }
        Ok((window.lower <= window.upper).then_some(window))
    }
}

/// A pair of physical customer vertices (the standard routing packing sets).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RyanFosterConstraint {
    pub first: usize,
    pub second: usize,
    pub together: bool,
}
impl RyanFosterConstraint {
    pub fn elementary_on_pair(&self, vertices: impl IntoIterator<Item = usize>) -> bool {
        let (mut a, mut b) = (false, false);
        for v in vertices {
            if v == self.first {
                if a {
                    return false;
                }
                a = true;
            }
            if v == self.second {
                if b {
                    return false;
                }
                b = true;
            }
        }
        true
    }
    pub fn satisfied_by(&self, vertices: impl IntoIterator<Item = usize>) -> bool {
        let (mut a, mut b) = (false, false);
        for v in vertices {
            a |= v == self.first;
            b |= v == self.second;
        }
        if self.together {
            a == b
        } else {
            !(a && b)
        }
    }
}

/// Allocate no label words when no branch resource is active.
pub type BranchState = Option<Box<[u32; 16]>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryConsumption {
    pub resource: usize,
    pub toggle: bool,
}

#[derive(Clone, Debug, Default)]
pub struct RyanFosterResources {
    pub resources: usize,
    pub vertices: Vec<Vec<BinaryConsumption>>,
    /// Native together constraints require even parity at the terminal/join.
    pub parity: [u32; 16],
}
impl RyanFosterResources {
    pub fn prepare(g: &Graph, constraints: &[RyanFosterConstraint]) -> Result<Self> {
        let mut result = Self {
            vertices: vec![vec![]; g.vertices.len()],
            ..Default::default()
        };
        for c in constraints {
            if c.first == c.second
                || [c.first, c.second]
                    .iter()
                    .any(|&v| v >= g.vertices.len() || v == g.source || v == g.sink)
            {
                return Err(invalid(
                    "Ryan–Foster requires two distinct customer vertices",
                ));
            }
            let i = result.resources;
            result.resources += if c.together { 3 } else { 1 };
            if result.resources > 512 {
                return Err(invalid("Ryan–Foster exceeds 512 binary resources"));
            }
            if c.together {
                result.parity[i / 32] |= 1 << (i % 32);
                for (v, seen) in [(c.first, i + 1), (c.second, i + 2)] {
                    result.vertices[v].push(BinaryConsumption {
                        resource: i,
                        toggle: true,
                    });
                    result.vertices[v].push(BinaryConsumption {
                        resource: seen,
                        toggle: false,
                    });
                }
            } else {
                for v in [c.first, c.second] {
                    result.vertices[v].push(BinaryConsumption {
                        resource: i,
                        toggle: false,
                    });
                }
            }
        }
        Ok(result)
    }
    pub fn can_visit(&self, state: &BranchState, vertex: usize) -> bool {
        state.as_ref().map_or(true, |words| {
            self.vertices[vertex]
                .iter()
                .all(|op| op.toggle || words[op.resource / 32] & (1 << (op.resource % 32)) == 0)
        })
    }
    pub fn visit(&self, state: &mut BranchState, vertex: usize) -> bool {
        let ops = &self.vertices[vertex];
        if ops.is_empty() {
            return true;
        }
        if !self.can_visit(state, vertex) {
            return false;
        }
        let words = state.get_or_insert_with(|| Box::new([0; 16]));
        for op in ops {
            let mask = 1 << (op.resource % 32);
            if op.toggle {
                words[op.resource / 32] ^= mask;
            } else {
                words[op.resource / 32] |= mask;
            }
        }
        true
    }
    pub fn accepts(&self, state: &BranchState) -> bool {
        state
            .as_ref()
            .map_or(true, |w| (0..16).all(|i| w[i] & self.parity[i] == 0))
    }
    pub fn joins(&self, a: &BranchState, b: &BranchState) -> bool {
        if self.resources == 0 {
            return true;
        }
        (0..16).all(|i| {
            let a = a.as_ref().map_or(0, |w| w[i]);
            let b = b.as_ref().map_or(0, |w| w[i]);
            a & b & !self.parity[i] == 0 && (a ^ b) & self.parity[i] == 0
        })
    }
}
