//! Algorithms and specialized models built on the core solver interfaces.
//!
//! `hgs` provides primal heuristics, `ipm` a linear-programming backend, and
//! `evrp` electric-vehicle routing models and `pstep` a compact CVRP relaxation.
//! These modules remain in the same
//! crate; this namespace does not introduce runtime plugins or Cargo features.
/// Decision-diagram arc-flow relaxation with infeasible-prefix refinement.
pub mod column_elimination;
pub mod evrp;
pub mod hgs;
pub mod ipm;
pub mod pstep;
