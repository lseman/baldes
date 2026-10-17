//! Hybrid Genetic Search (HGS) for VRPTW/CVRP.
//!
//! This module converts the HGS-CVRP algorithm from C++ to Rust. It provides:
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
pub mod genetic;
pub mod individual;
pub mod local_search;
pub mod matrix;
pub mod params;
pub mod split;

// Future module (to be converted):
// pub mod population;

pub use circle_sector::CircleSector;
pub use genetic::Genetic;
pub use individual::Individual;
pub use matrix::Matrix;
pub use params::{Client, Config, InstanceData, Params, ProblemType, Savings, TimeWindow, Xoroshiro128Plus};
pub use split::Split;
