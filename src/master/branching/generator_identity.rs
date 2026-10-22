//! Exact generator identities and optional original model display names.
use super::BranchConstraint;
use std::collections::BTreeMap;

/// Original model names used only to break equal-score history ties. Entries
/// are keyed by the exact branch variable/disjunction, independently of their
/// display text. Register either child to name both sides of the generator.
#[derive(Clone, Debug, Default)]
pub struct BranchGeneratorNames {
    entries: BTreeMap<String, String>,
}
impl BranchGeneratorNames {
    pub fn insert(
        &mut self,
        constraint: &BranchConstraint,
        name: impl Into<String>,
    ) -> Option<String> {
        self.entries.insert(key(constraint), name.into())
    }
    pub fn get(&self, constraint: &BranchConstraint) -> Option<&str> {
        self.entries.get(&key(constraint)).map(String::as_str)
    }
    pub fn remove(&mut self, constraint: &BranchConstraint) -> Option<String> {
        self.entries.remove(&key(constraint))
    }
    pub(crate) fn label(&self, constraint: &BranchConstraint) -> String {
        self.get(constraint)
            .map(str::to_owned)
            .unwrap_or_else(|| match constraint {
                BranchConstraint::AccumulatedResourceGroup(group) => group.generator_name(),
                _ => key(constraint),
            })
    }
}

// Drop RHS/sense so estimates generalize between bounds on the same variable.
// Keep schedule thresholds: distinct resource disjunctions need distinct costs.
pub(crate) fn key(c: &BranchConstraint) -> String {
    match c {
        BranchConstraint::ArcFlow { arc, .. } => format!("arc:{arc}"),
        BranchConstraint::NumberOfVehicles { graph_id, .. } => format!("vehicles:{graph_id}"),
        BranchConstraint::PackingSetAssignment {
            vertex, graph_id, ..
        } => format!("assignment:{vertex}:{graph_id}"),
        BranchConstraint::RyanFoster(c) => {
            format!("rf:{}:{}", c.first.min(c.second), c.first.max(c.second))
        }
        BranchConstraint::ClusterBoundary { vertices, .. } => {
            let mut v = vertices.clone();
            v.sort_unstable();
            format!("cluster:{v:?}")
        }
        BranchConstraint::ClusterPair { first, second, .. } => {
            let mut a = first.clone();
            let mut b = second.clone();
            a.sort_unstable();
            b.sort_unstable();
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            format!("pair:{a:?}:{b:?}")
        }
        BranchConstraint::AccumulatedResourceGroup(c) => format!(
            "resource-group:{}:{:?}:{}:{:016x}",
            c.packing_row(),
            c.arcs(),
            c.resource(),
            if c.boundary() == 0. {
                0
            } else {
                c.boundary().to_bits()
            }
        ),
        BranchConstraint::AccumulatedResource(c) => format!(
            "resource:{}:{}:{:016x}",
            c.arc,
            c.resource,
            if c.boundary() == 0. {
                0
            } else {
                c.boundary().to_bits()
            }
        ),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/branching/generator_identity/tests.rs"]
mod tests;
