//! Routing branch constraints. Family-specific behavior lives in sibling modules.
mod accumulated_resource;
mod aggregate_flow;
mod arc_flow;
mod cluster;
pub mod enforcement;
pub(crate) mod generator_identity;
mod grouped_resource;
mod number_of_vehicles;
mod options;
mod packing_set_assignment;
pub mod resource_intervals;
mod ryan_foster;

use crate::master::{root::RootProblem, RowSense};
use crate::{invalid, pricing::restrictions::RyanFosterConstraint, Result, Route};
pub(crate) use aggregate_flow::edges as aggregate_edges;
pub use aggregate_flow::AggregateVariable;
pub use cluster::ClusterBranchingOptions;
pub use generator_identity::BranchGeneratorNames;
pub use grouped_resource::AccumulatedResourceGroupBranch;
pub use options::{BranchPriorities, BranchingOptions};

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
    /// An integer combination of physical arc traversals: a model-registered
    /// generic variable or an undirected edge. `arcs` holds sorted physical
    /// arc positions and integer coefficients; `name` identifies the variable.
    AggregateFlow {
        name: String,
        arcs: Vec<(usize, f64)>,
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
    /// Number of entries into a customer cluster (half its undirected cutset).
    ClusterBoundary {
        vertices: Vec<usize>,
        sense: RowSense,
        rhs: f64,
    },
    /// Total flow in both directions between two disjoint clusters.
    ClusterPair {
        first: Vec<usize>,
        second: Vec<usize>,
        sense: RowSense,
        rhs: f64,
    },
    AccumulatedResource(crate::pricing::restrictions::AccumulatedResourceBranch),
    AccumulatedResourceGroup(AccumulatedResourceGroupBranch),
}
impl BranchConstraint {
    pub fn row(&self) -> (RowSense, f64) {
        match *self {
            Self::ArcFlow { sense, rhs, .. }
            | Self::NumberOfVehicles { sense, rhs, .. }
            | Self::AggregateFlow { sense, rhs, .. }
            | Self::ClusterBoundary { sense, rhs, .. }
            | Self::ClusterPair { sense, rhs, .. } => (sense, rhs),
            Self::PackingSetAssignment { assigned, .. } => packing_set_assignment::row(assigned),
            // Nonnegative route-violation rows for resource-based rules.
            Self::RyanFoster(_)
            | Self::AccumulatedResource(_)
            | Self::AccumulatedResourceGroup(_) => (RowSense::LessEqual, 0.),
        }
    }
    pub fn validate<const D: usize>(&self, p: &RootProblem<D>) -> Result<()> {
        if !self.row().1.is_finite() {
            return Err(invalid("finite branch RHS required"));
        }
        match self {
            Self::ArcFlow { arc, .. } => arc_flow::validate(p, *arc),
            Self::AggregateFlow { arcs, .. } => aggregate_flow::validate(p, arcs),
            Self::NumberOfVehicles { graph_id, rhs, .. } => {
                number_of_vehicles::validate(*graph_id, *rhs)
            }
            Self::PackingSetAssignment {
                vertex, graph_id, ..
            } => packing_set_assignment::validate(p, *vertex, *graph_id),
            Self::RyanFoster(c) => ryan_foster::validate(p, *c),
            Self::ClusterBoundary { vertices, .. } => cluster::validate(p, vertices),
            Self::ClusterPair { first, second, .. } => cluster::validate_pair(p, first, second),
            Self::AccumulatedResource(c) => accumulated_resource::validate(p, *c),
            Self::AccumulatedResourceGroup(c) => c.validate(p),
        }
    }
    /// Per-arc contribution, so new columns and pricing receive identical rows.
    pub(crate) fn arc_coefficient<const D: usize>(&self, p: &RootProblem<D>, arc: usize) -> f64 {
        match self {
            Self::ArcFlow { arc: target, .. } => arc_flow::coefficient(arc, *target),
            Self::AggregateFlow { arcs, .. } => aggregate_flow::coefficient(arcs, arc),
            Self::NumberOfVehicles { graph_id, .. } => {
                number_of_vehicles::coefficient(p, arc, *graph_id)
            }
            Self::PackingSetAssignment {
                vertex, graph_id, ..
            } => packing_set_assignment::coefficient(p, arc, *vertex, *graph_id),
            Self::RyanFoster(_) | Self::AccumulatedResource(_) => 0.,
            Self::AccumulatedResourceGroup(c) if enforcement::ACCUMULATED_RESOURCE_IN_PRICING => {
                c.additive(p.graph.arcs[arc].id)
            }
            Self::AccumulatedResourceGroup(_) => 0.,
            Self::ClusterBoundary { vertices, .. } => cluster::boundary_coefficient(
                vertices,
                p.graph.arcs[arc].tail,
                p.graph.arcs[arc].head,
            ),
            Self::ClusterPair { first, second, .. } => cluster::pair_coefficient(
                first,
                second,
                p.graph.arcs[arc].tail,
                p.graph.arcs[arc].head,
            ),
        }
    }
    pub fn coefficient<const D: usize>(&self, p: &RootProblem<D>, route: &Route) -> f64 {
        match self {
            Self::RyanFoster(c) => ryan_foster::coefficient(*c, route),
            Self::AccumulatedResource(c) => accumulated_resource::coefficient(*c, route),
            Self::AccumulatedResourceGroup(c) => c.coefficient(route, false),
            _ => route
                .arcs
                .iter()
                .map(|&id| {
                    self.arc_coefficient(p, p.arc_position(id).expect("validated route arc"))
                })
                .sum(),
        }
    }
    pub(crate) fn nonlinear(&self) -> bool {
        matches!(
            self,
            Self::RyanFoster(_) | Self::AccumulatedResource(_) | Self::AccumulatedResourceGroup(_)
        )
    }
}
