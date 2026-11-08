//! Branching restrictions installed in pricing networks.
mod grouped_resource;
pub mod resource_intervals;
pub use grouped_resource::AccumulatedResourceGroupBranch;

/// Whether master-selected Ryan–Foster branches also restrict pricing.
/// Master-only mode retains a relaxed pricing domain and its lower bound.
pub const RYAN_FOSTER_IN_PRICING: bool = !cfg!(feature = "ryan-foster-master");

/// Whether pricing enforces accumulated-resource branches.
pub const ACCUMULATED_RESOURCE_IN_PRICING: bool = !cfg!(feature = "accumulated-resource-master");
