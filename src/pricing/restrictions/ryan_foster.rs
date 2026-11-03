//! Ryan–Foster constraints and binary resources for labeling.
use crate::{invalid, Graph, Result};
use std::sync::Arc;

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

/// Allocate no label words when no branch resource is active. Clones share
/// words until a visit changes a branch resource.
pub type BranchState = Option<Arc<[u32; 16]>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryConsumption {
    pub resource: usize,
    pub toggle: bool,
}

/// Packing-set membership on physical arc IDs, including any head membership.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArcSetConstraint {
    pub first: Vec<i32>,
    pub second: Vec<i32>,
    pub together: bool,
}
impl ArcSetConstraint {
    /// Check a complete route using physical arc IDs.
    pub fn satisfied_by(&self, arcs: impl IntoIterator<Item = i32>) -> bool {
        let (mut first, mut second) = (false, false);
        for arc in arcs {
            first |= self.first.contains(&arc);
            second |= self.second.contains(&arc);
        }
        if self.together {
            first == second
        } else {
            !(first && second)
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RyanFosterResources {
    pub resources: usize,
    pub vertices: Vec<Vec<BinaryConsumption>>,
    /// Branch events indexed by physical graph arc position.
    pub arcs: Vec<Vec<BinaryConsumption>>,
    /// Native together constraints require even parity at the terminal/join.
    pub parity: [u32; 16],
}
impl RyanFosterResources {
    /// Prepare arc-set operations on top of existing vertex restrictions.
    /// Each physical arc is listed once per set, even when both the arc and
    /// its head have the same packing-set membership.
    pub fn prepare_arc_sets<const D: usize>(
        g: &Graph<D>,
        vertices: &[RyanFosterConstraint],
        constraints: &[ArcSetConstraint],
    ) -> Result<Self> {
        let mut result = Self::prepare(g, vertices)?;
        if constraints.is_empty() {
            return Ok(result);
        }
        result.arcs = vec![vec![]; g.arcs.len()];
        let indices: std::collections::BTreeMap<_, _> = g
            .arcs
            .iter()
            .enumerate()
            .map(|(index, arc)| (arc.id, index))
            .collect();
        for constraint in constraints {
            let index = result.resources;
            result.resources += if constraint.together { 3 } else { 1 };
            if result.resources > 512 {
                return Err(invalid("Ryan–Foster exceeds 512 binary resources"));
            }
            if constraint.together {
                result.parity[index / 32] |= 1 << (index % 32);
            }
            for (arcs, seen) in [
                (&constraint.first, index + 1),
                (&constraint.second, index + 2),
            ] {
                let mut unique = std::collections::BTreeSet::new();
                for id in arcs {
                    let physical = *indices
                        .get(id)
                        .ok_or_else(|| invalid("Ryan–Foster unknown arc ID"))?;
                    if !unique.insert(physical) {
                        continue;
                    }
                    result.arcs[physical].push(BinaryConsumption {
                        resource: index,
                        toggle: constraint.together,
                    });
                    if constraint.together {
                        result.arcs[physical].push(BinaryConsumption {
                            resource: seen,
                            toggle: false,
                        });
                    }
                }
            }
        }
        Ok(result)
    }

    /// Apply all events on an arc atomically; rejected arcs leave state intact.
    pub fn traverse_arc(&self, state: &mut BranchState, physical: usize) -> bool {
        let Some(ops) = self.arcs.get(physical) else {
            return self.arcs.is_empty();
        };
        if ops.is_empty() {
            return true;
        }
        let mut words = state.as_deref().copied().unwrap_or([0; 16]);
        for op in ops {
            let mask = 1 << (op.resource % 32);
            let word = &mut words[op.resource / 32];
            if op.toggle {
                *word ^= mask;
            } else if *word & mask != 0 {
                return false;
            } else {
                *word |= mask;
            }
        }
        *state = Some(Arc::new(words));
        true
    }

    pub fn prepare<const D: usize>(
        g: &Graph<D>,
        constraints: &[RyanFosterConstraint],
    ) -> Result<Self> {
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
        state.as_ref().is_none_or(|words| {
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
        let words = Arc::make_mut(state.get_or_insert_with(|| Arc::new([0; 16])));
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
            .is_none_or(|w| (0..16).all(|i| w[i] & self.parity[i] == 0))
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

#[cfg(test)]
#[path = "../../../tests/unit/pricing/ryan_foster_tests.rs"]
mod tests;
