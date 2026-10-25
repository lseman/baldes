//! Candidate-family configuration and defaults.
use super::ClusterBranchingOptions;

/// Model registration priorities. Defaults preserve the routing adapter's
/// family order; native interfaces likewise accept priorities from the model.
#[derive(Clone, Debug)]
pub struct BranchPriorities {
    pub number_of_vehicles: f64,
    pub accumulated_resource: f64,
    pub clusters: f64,
    pub packing_set_assignment: f64,
    pub ryan_foster: f64,
    pub aggregate_flow: f64,
    pub arc_flow: f64,
    /// Time-window selection counts. Below arc flow by default, so they are
    /// branched on last, after flows summed over each input arc's copies.
    pub time_window_selection: f64,
}
impl Default for BranchPriorities {
    fn default() -> Self {
        Self {
            number_of_vehicles: 11.,
            accumulated_resource: 10.,
            clusters: 10.,
            packing_set_assignment: 3.,
            ryan_foster: 2.,
            aggregate_flow: 1.5,
            arc_flow: 1.,
            time_window_selection: 0.5,
        }
    }
}
impl BranchPriorities {
    pub(crate) fn validate(&self) -> crate::Result<()> {
        if [
            self.number_of_vehicles,
            self.accumulated_resource,
            self.clusters,
            self.packing_set_assignment,
            self.ryan_foster,
            self.aggregate_flow,
            self.arc_flow,
            self.time_window_selection,
        ]
        .iter()
        .all(|p| p.is_finite())
        {
            Ok(())
        } else {
            Err(crate::invalid("finite branch family priorities required"))
        }
    }
    pub(crate) fn priority(&self, constraint: &super::BranchConstraint) -> f64 {
        use super::BranchConstraint::*;
        match constraint {
            NumberOfVehicles { .. } => self.number_of_vehicles,
            AccumulatedResource(_) | AccumulatedResourceGroup(_) => self.accumulated_resource,
            ClusterBoundary { .. } | ClusterPair { .. } => self.clusters,
            PackingSetAssignment { .. } => self.packing_set_assignment,
            RyanFoster(_) => self.ryan_foster,
            AggregateFlow { .. } => self.aggregate_flow,
            ArcFlow { .. }
            | WindowFlow {
                group: super::WindowGroup::Arc { .. },
                ..
            } => self.arc_flow,
            WindowFlow {
                group: super::WindowGroup::Selection { .. },
                ..
            } => self.time_window_selection,
        }
    }
}

/// The order of families is explicit; the first fractional family is evaluated
/// by the existing phased strong-branching schedule. Arc flow is the fallback.
#[derive(Clone, Debug)]
pub struct BranchingOptions {
    pub priorities: BranchPriorities,
    /// Optional original model labels for equal-score history ties.
    pub generator_names: super::BranchGeneratorNames,
    pub number_of_vehicles: bool,
    pub packing_set_assignment: bool,
    pub ryan_foster: bool,
    pub clusters: Option<ClusterBranchingOptions>,
    pub accumulated_resource: bool,
    /// Model-registered generic variables, branched on when fractional.
    pub aggregates: Vec<super::AggregateVariable>,
    /// Branch on undirected edge flows: both arc directions between a pair.
    pub edge_flow: bool,
    /// Branch on how often each time window of a multi-window vertex is
    /// selected, on this disposable resource (multi-time-windows builds).
    /// Following Li et al. (2020), the tree copies each arc into such a vertex
    /// once per window, branches on input-arc flows summed over the copies
    /// (arc-flow priority) and then on window selection counts (their own,
    /// lower priority). Reported routes use the input arc IDs.
    pub time_window_selection: Option<usize>,
    /// Window copies of the split graph; set by the tree, not by callers.
    #[doc(hidden)]
    pub window_arcs: Option<crate::graph::WindowArcs>,
}
impl Default for BranchingOptions {
    fn default() -> Self {
        Self {
            priorities: BranchPriorities::default(),
            generator_names: Default::default(),
            number_of_vehicles: true,
            packing_set_assignment: true,
            ryan_foster: true,
            clusters: None,
            accumulated_resource: false,
            aggregates: vec![],
            edge_flow: false,
            time_window_selection: None,
            window_arcs: None,
        }
    }
}
impl BranchingOptions {
    /// Whether any aggregate-flow generator is registered.
    pub(crate) fn aggregate_flow(&self) -> bool {
        self.edge_flow || !self.aggregates.is_empty()
    }
    pub fn arc_flow_only() -> Self {
        Self {
            priorities: BranchPriorities::default(),
            generator_names: Default::default(),
            number_of_vehicles: false,
            packing_set_assignment: false,
            ryan_foster: false,
            clusters: None,
            accumulated_resource: false,
            aggregates: vec![],
            edge_flow: false,
            time_window_selection: None,
            window_arcs: None,
        }
    }
}
