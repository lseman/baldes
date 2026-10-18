//! Routing branches corresponding to arc, paths-per-network,
//! packing-set assignment and packing-set Ryan–Foster families.
use super::{root::RootProblem, RowSense};
use crate::{invalid, pricing::branching::RyanFosterConstraint, Result, Route};

#[derive(Clone, Debug, PartialEq)]
pub enum BranchConstraint {
    ArcFlow {
        arc: usize,
        sense: RowSense,
        rhs: f64,
    },
    NumberOfVehicles {
        graph_id: i32,
        sense: RowSense,
        rhs: f64,
    },
    /// In the standard routing master a packing set is a customer vertex.
    PackingSetAssignment {
        vertex: usize,
        graph_id: i32,
        assigned: bool,
    },
    RyanFoster(RyanFosterConstraint),
}
impl BranchConstraint {
    pub fn row(&self) -> (RowSense, f64) {
        match *self {
            Self::ArcFlow { sense, rhs, .. } | Self::NumberOfVehicles { sense, rhs, .. } => {
                (sense, rhs)
            }
            Self::PackingSetAssignment { assigned, .. } => {
                if assigned {
                    (RowSense::GreaterEqual, 1.)
                } else {
                    (RowSense::LessEqual, 0.)
                }
            }
            // Native generic coefficient is 1 iff the route violates the rule.
            Self::RyanFoster(_) => (RowSense::LessEqual, 0.),
        }
    }
    pub fn validate(&self, p: &RootProblem) -> Result<()> {
        let (_, rhs) = self.row();
        if !rhs.is_finite() {
            return Err(invalid("finite branch RHS required"));
        }
        match *self {
            Self::ArcFlow { arc, .. } if arc >= p.graph.arcs.len() => {
                return Err(invalid("branch arc index"))
            }
            Self::NumberOfVehicles { rhs, graph_id, .. } if rhs < 0. || graph_id < 0 => {
                return Err(invalid("vehicle branch graph/count"))
            }
            Self::PackingSetAssignment {
                vertex, graph_id, ..
            } if !p.customers.contains(&vertex) || graph_id < 0 => {
                return Err(invalid(
                    "assignment requires a customer packing set and nonnegative graph ID",
                ))
            }
            Self::RyanFoster(c)
                if c.first == c.second
                    || !p.customers.contains(&c.first)
                    || !p.customers.contains(&c.second) =>
            {
                return Err(invalid(
                    "Ryan–Foster requires distinct customer packing sets",
                ))
            }
            _ => (),
        }
        Ok(())
    }
    /// Per-arc contribution, so new columns and pricing receive identical rows.
    pub(crate) fn arc_coefficient(&self, p: &RootProblem, arc: usize) -> f64 {
        let a = &p.graph.arcs[arc];
        match *self {
            Self::ArcFlow { arc: target, .. } => (arc == target) as u8 as f64,
            Self::NumberOfVehicles { graph_id, .. } => {
                (graph_id == p.graph.graph_id.max(0)
                    && a.head == p.graph.sink
                    && a.tail != p.graph.source) as u8 as f64
            }
            Self::PackingSetAssignment {
                vertex, graph_id, ..
            } => (graph_id == p.graph.graph_id.max(0) && a.head == vertex) as u8 as f64,
            Self::RyanFoster(_) => 0.,
        }
    }
    pub fn coefficient(&self, p: &RootProblem, route: &Route) -> f64 {
        if let Self::RyanFoster(c) = self {
            return (!c.satisfied_by(route.nodes.iter().map(|&v| v as usize))) as u8 as f64;
        }
        route
            .arcs
            .iter()
            .map(|&id| self.arc_coefficient(p, p.arc_position(id).expect("validated route arc")))
            .sum()
    }
}

/// The order of families is explicit; the first fractional family is evaluated
/// by the existing phased strong-branching schedule. Arc flow is the fallback.
#[derive(Clone, Debug)]
pub struct BranchingOptions {
    pub number_of_vehicles: bool,
    pub packing_set_assignment: bool,
    pub ryan_foster: bool,
}
impl Default for BranchingOptions {
    fn default() -> Self {
        Self {
            number_of_vehicles: true,
            packing_set_assignment: true,
            ryan_foster: true,
        }
    }
}
impl BranchingOptions {
    pub fn arc_flow_only() -> Self {
        Self {
            number_of_vehicles: false,
            packing_set_assignment: false,
            ryan_foster: false,
        }
    }
}
