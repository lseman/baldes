//! Candidate-family configuration and defaults.
use super::ClusterBranchingOptions;

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
