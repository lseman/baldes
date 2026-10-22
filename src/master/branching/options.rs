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
    pub arc_flow: f64,
}
impl Default for BranchPriorities {
    fn default() -> Self {
        Self {
            number_of_vehicles: 11.,
            accumulated_resource: 10.,
            clusters: 10.,
            packing_set_assignment: 3.,
            ryan_foster: 2.,
            arc_flow: 1.,
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
            self.arc_flow,
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
            ArcFlow { .. } => self.arc_flow,
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
        }
    }
}
impl BranchingOptions {
    pub fn arc_flow_only() -> Self {
        Self {
            priorities: BranchPriorities::default(),
            generator_names: Default::default(),
            number_of_vehicles: false,
            packing_set_assignment: false,
            ryan_foster: false,
            clusters: None,
            accumulated_resource: false,
        }
    }
}
