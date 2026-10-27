//! Algorithms and specialized models built on the core solver interfaces.
//!
//! `hgs` provides primal heuristics, `ipm` a linear-programming backend, and
//! `evrp` electric-vehicle routing models, `pstep` a compact CVRP relaxation
//! and `robust` the budgeted-uncertainty VRPTW.
//! These modules remain in the same
//! crate; this namespace does not introduce runtime plugins or Cargo features.
/// Decision-diagram arc-flow relaxation with infeasible-prefix refinement.
pub mod ce;
pub mod evrp;
pub mod hgs;
pub mod ipm;
pub mod pstep;
pub mod robust;
/// One-to-one robust pickup and delivery with time windows.
pub mod robust_pd;
/// Stochastic demands with replenishment recourse and a tiny exact baseline.
pub mod stochastic;
