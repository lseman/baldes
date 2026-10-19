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
    /// Number of entries into a customer cluster (half its undirected cutset).
    ClusterBoundary { vertices: Vec<usize>, sense: RowSense, rhs: f64 },
    /// Total flow in both directions between two disjoint clusters.
    ClusterPair { first: Vec<usize>, second: Vec<usize>, sense: RowSense, rhs: f64 },
    AccumulatedResource(crate::pricing::branching::AccumulatedResourceBranch),
}
impl BranchConstraint {
    pub fn row(&self) -> (RowSense, f64) {
        match *self {
            Self::ArcFlow { sense, rhs, .. } | Self::NumberOfVehicles { sense, rhs, .. } => {
                (sense, rhs)
            }
            Self::ClusterBoundary { sense, rhs, .. } | Self::ClusterPair { sense, rhs, .. } => (sense, rhs),
            Self::PackingSetAssignment { assigned, .. } => {
                if assigned {
                    (RowSense::GreaterEqual, 1.)
                } else {
                    (RowSense::LessEqual, 0.)
                }
            }
            // Native generic coefficient is 1 iff the route violates the rule.
            Self::RyanFoster(_) | Self::AccumulatedResource(_) => (RowSense::LessEqual, 0.),
        }
    }
    pub fn validate(&self, p: &RootProblem) -> Result<()> {
        let (_, rhs) = self.row();
        if !rhs.is_finite() {
            return Err(invalid("finite branch RHS required"));
        }
        match *self {
            Self::AccumulatedResource(c) => { c.window(&p.graph)?; }
            Self::ClusterBoundary { ref vertices, .. } => validate_cluster(p, vertices)?,
            Self::ClusterPair { ref first, ref second, .. } => {
                validate_cluster(p, first)?;
                validate_cluster(p, second)?;
                if first.iter().any(|v| second.contains(v)) {
                    return Err(invalid("cluster pair must be disjoint"));
                }
            }
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
            Self::AccumulatedResource(_) => 0.,
            Self::ClusterBoundary { ref vertices, .. } =>
                (!vertices.contains(&a.tail) && vertices.contains(&a.head)) as u8 as f64,
            Self::ClusterPair { ref first, ref second, .. } =>
                ((first.contains(&a.tail) && second.contains(&a.head))
                    || (second.contains(&a.tail) && first.contains(&a.head))) as u8 as f64,
        }
    }
    pub fn coefficient(&self, p: &RootProblem, route: &Route) -> f64 {
        if let Self::RyanFoster(c) = self {
            return (!c.satisfied_by(route.nodes.iter().map(|&v| v as usize))) as u8 as f64;
        }
        if let Self::AccumulatedResource(c) = self {
            return route.arcs.iter().enumerate().filter(|&(i, &id)| {
                if id != c.arc { return false; }
                route.resource_profile.get(i + 1).and_then(|q| q.get(c.resource))
                    .map_or(true, |&q| !q.is_finite() || if c.upper { q < c.threshold } else { q > c.threshold })
            }).count() as f64;
        }
        route
            .arcs
            .iter()
            .map(|&id| self.arc_coefficient(p, p.arc_position(id).expect("validated route arc")))
            .sum()
    }
    pub(crate) fn nonlinear(&self) -> bool {
        matches!(self, Self::RyanFoster(_) | Self::AccumulatedResource(_))
    }
}

fn validate_cluster(p: &RootProblem, vertices: &[usize]) -> Result<()> {
    let depot = vertices.len() == 2 && vertices.contains(&p.graph.source) && vertices.contains(&p.graph.sink);
    if vertices.is_empty() || vertices.iter().enumerate().any(|(i, v)| vertices[..i].contains(v))
        || (!depot && vertices.iter().any(|v| !p.customers.contains(v))) {
        return Err(invalid("cluster must contain distinct customers or the split depot"));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ClusterBranchingOptions {
    /// Explicit customer partition, or an automatically constructed MST partition.
    pub partition: Option<Vec<Vec<usize>>>,
    /// Remove MST edges above mean + theta * standard deviation.
    pub theta: f64,
}
impl Default for ClusterBranchingOptions {
    fn default() -> Self { Self { partition: None, theta: 1. } }
}

/// The order of families is explicit; the first fractional family is evaluated
/// by the existing phased strong-branching schedule. Arc flow is the fallback.
#[derive(Clone, Debug)]
pub struct BranchingOptions {
    pub number_of_vehicles: bool,
    pub packing_set_assignment: bool,
    pub ryan_foster: bool,
    pub clusters: Option<ClusterBranchingOptions>,
    pub accumulated_resource: bool,
}
impl Default for BranchingOptions {
    fn default() -> Self {
        Self {
            number_of_vehicles: true,
            packing_set_assignment: true,
            ryan_foster: true,
            clusters: None,
            accumulated_resource: false,
        }
    }
}
impl BranchingOptions {
    pub fn arc_flow_only() -> Self {
        Self {
            number_of_vehicles: false,
            packing_set_assignment: false,
            ryan_foster: false,
            clusters: None,
            accumulated_resource: false,
        }
    }
}
