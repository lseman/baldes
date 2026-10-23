//! Restricted-master and dual-price smoothing ports. Cuts are outside this scope.
pub use lp_master::{LpMaster, LpSolveOutcome};
pub mod solvers;
#[cfg(feature = "clp")]
pub use solvers::clp;
#[cfg(feature = "highs")]
pub use solvers::highs;
pub mod certificates;
pub mod cg;
pub mod deluxing;
pub mod lp_master;
pub mod preprocessing;
mod progress;
pub mod stabilization;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSense {
    Equal,
    LessEqual,
    GreaterEqual,
}

#[cfg(feature = "clp")]
pub mod branching;
#[cfg(feature = "clp")]
pub mod mp;
#[cfg(feature = "clp")]
pub mod root;
/// Compatibility path; prefer [`mp`] for shared-master models.
#[cfg(feature = "clp")]
pub use mp as networks;
#[cfg(feature = "clp")]
pub mod tree;
