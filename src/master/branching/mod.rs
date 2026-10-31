//! Branching restrictions installed in pricing networks.
mod grouped_resource;
pub mod resource_intervals;
pub use grouped_resource::AccumulatedResourceGroupBranch;

/// Whether pricing enforces accumulated-resource branches.
pub const ACCUMULATED_RESOURCE_IN_PRICING: bool = !cfg!(feature = "accumulated-resource-master");
