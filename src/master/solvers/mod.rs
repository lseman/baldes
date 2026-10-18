//! LP solver backends and shared integer-solve controls.
//! HiGHS also solves mixed-integer copies of [`super::LpMaster`] models.
#[cfg(feature = "clp")]
pub mod clp;
#[cfg(feature = "highs")]
pub mod highs;

#[cfg(feature = "clp")]
pub use clp::ClpMaster;
#[cfg(feature = "highs")]
pub use highs::HighsMaster;

pub mod mip;
