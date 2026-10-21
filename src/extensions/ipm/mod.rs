//! Native Rust port of the IPM in `solvers/ipm` and `solvers/src/ipm`.
//! Minimizes cᵀx subject to mixed linear rows and variable bounds.
#![doc = include_str!("README.md")]
#![forbid(unsafe_code)]

mod factor;
mod matrix;
mod model;
mod solver;
mod sparse_solver;

pub use matrix::SparseMatrix;
pub use model::{ModelData, Sense};
pub use solver::{solve, Options, Solution, Solver, SolverStatistics, Status};
pub use sparse_solver::{
    estimate_fill, FillEstimate, HiPOLDLT, NormalEqLDLT, SolverType, SparseSolver,
};

/// Invalid input and numerical failures are distinct from an iteration limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidInput(String),
    NumericalFailure(String),
    NotFactorized,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
            Self::NumericalFailure(s) => write!(f, "numerical failure: {s}"),
            Self::NotFactorized => f.write_str("matrix has not been factorized"),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn finite(x: &[f64]) -> bool {
    x.iter().all(|v| v.is_finite())
}
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
pub(crate) fn norm(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}
