//! Shared integer-solver options and explicit proof/limit outcomes.
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct MipOptions {
    pub node_limit: usize,
    pub time_limit: Duration,
    /// Search only for objectives strictly better than this value. No solution
    /// is implied by a finite cutoff supplied by the caller.
    pub objective_cutoff: f64,
}
impl Default for MipOptions {
    fn default() -> Self {
        Self {
            node_limit: 2_000_000,
            time_limit: Duration::from_secs(360_000),
            objective_cutoff: f64::INFINITY,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MipStatus {
    Optimal,
    Infeasible,
    /// No solution improves the supplied cutoff; this is not infeasibility.
    ObjectiveCutoff,
    NodeLimit,
    TimeLimit,
    /// A relaxation did not provide a usable certificate (including unbounded LPs).
    Incomplete,
}
#[derive(Clone, Debug)]
pub struct MipSolution {
    pub objective: f64,
    /// Values in the original model's column order, including continuous columns.
    pub values: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct MipOutput {
    pub status: MipStatus,
    pub solution: Option<MipSolution>,
    pub bound: f64,
    pub nodes: usize,
    pub simplex_iterations: usize,
    pub seconds: f64,
    /// Original HiGHS model status; 0 when no solver call was made.
    pub solver_status: i32,
}
