//! Restricted-master and dual-price smoothing ports. Cuts are outside this scope.
pub use lp_master::{LpMaster, LpSolveOutcome};
pub mod solvers;
#[cfg(feature = "clp")]
pub use solvers::clp;
#[cfg(feature = "highs")]
pub use solvers::highs;
pub mod column_generation;
mod progress;
pub mod certificates;
pub(crate) mod rounding;
pub mod lp_master;
pub mod stabilization;
pub mod preprocessing;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSense {
    Equal,
    LessEqual,
    GreaterEqual,
}

#[cfg(feature = "clp")]
pub mod branching;
#[cfg(feature = "clp")]
pub mod root;
#[cfg(feature = "clp")]
pub mod multi_pricing;
/// Compatibility path; prefer [`multi_pricing`] for shared-master models.
#[cfg(feature = "clp")]
pub use multi_pricing as networks;
#[cfg(feature = "clp")]
pub mod tree;
