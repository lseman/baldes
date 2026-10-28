//! Hybrid Genetic Search (HGS) for VRPTW/CVRP.
//!
//! A Rust port of HGS-VRPTW (Kool et al., after Vidal's HGS-CVRP). It provides:
//! - **Params**: Problem parameters, instance data, penalty coefficients
//! - **Individual**: Solution representation (giant tour + routes)
//! - **Split**: Linear split algorithm for converting giant tours to feasible routes
//! - **LocalSearch**: Local search moves (RI, SWAP*, 2-opt, etc.)
//! - **Genetic**: Crossover and mutation operators
//! - **Population**: Population management for the genetic algorithm
//!
//! ## Architecture
//!
//! ```text
//! Instance → Params → Individual (giant tour)
//!                     ↓
//!                 Split → Routes
//!                     ↓
//!               LocalSearch → Improved routes
//!                     ↓
//!                Genetic → New individuals
//!                     ↓
//!              Population → Selection & diversity
//! ```

pub mod circle_sector;
pub mod dri;
pub mod driver;
pub mod genetic;
pub mod individual;
pub mod intensify;
pub mod local_search;
pub mod matrix;
pub mod params;
pub mod population;
pub mod solution;
pub mod split;

pub use circle_sector::CircleSector;
pub use driver::{run_hgs, run_hgs_seeded, HgsOptions, HgsResult};
pub use genetic::Genetic;
pub use individual::{Individual, SortableF64};
pub use matrix::Matrix;
pub use params::{
    Client, Config, InstanceData, Params, ProblemType, Savings, TimeWindow, Xoroshiro128Plus,
};
pub use population::Population;
pub use split::Split;
