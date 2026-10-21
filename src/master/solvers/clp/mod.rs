//! Ownership-safe bridge to COIN-OR Clp's C interface. No original solver code
//! is called: labeling, pricing, stabilization and CG belong to this Rust port.
use crate::master::{LpMaster, LpSolveOutcome, RowSense};
use crate::{incomplete, invalid, Result};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::{Duration, Instant};
type Model = c_void;
extern "C" {
    fn Clp_newModel() -> *mut Model;
    fn baldes_clp_clone(m: *const Model) -> *mut Model;
    fn baldes_clp_dual_limited(
        m: *mut Model,
        iterations: i32,
        seconds: f64,
        keep_work_areas: i32,
    ) -> i32;
    #[cfg(test)]
    fn baldes_clp_solve_limits(m: *mut Model, values: *mut f64);
    #[cfg(test)]
    fn Clp_setMaximumSeconds(m: *mut Model, seconds: f64);
    fn Clp_numberIterations(m: *mut Model) -> i32;
    fn Clp_deleteModel(m: *mut Model);
    fn Clp_setLogLevel(m: *mut Model, level: i32);
    fn Clp_setOptimizationDirection(m: *mut Model, direction: f64);
    fn Clp_addRows(
        m: *mut Model,
        n: i32,
        lower: *const f64,
        upper: *const f64,
        start: *const i32,
        columns: *const i32,
        values: *const f64,
    );
    fn Clp_addColumns(
        m: *mut Model,
        n: i32,
        lower: *const f64,
        upper: *const f64,
        objective: *const f64,
        start: *const i32,
        rows: *const i32,
        values: *const f64,
    );
    fn Clp_modifyCoefficient(m: *mut Model, row: i32, column: i32, value: f64, keep: bool);
    fn Clp_chgObjCoefficients(m: *mut Model, costs: *const f64);
    fn Clp_chgRowLower(m: *mut Model, lower: *const f64);
    fn Clp_chgRowUpper(m: *mut Model, upper: *const f64);
    fn Clp_chgColumnUpper(m: *mut Model, upper: *const f64);
    fn Clp_deleteRows(m: *mut Model, n: i32, indices: *const i32);
    fn Clp_deleteColumns(m: *mut Model, n: i32, indices: *const i32);
    fn Clp_scaling(m: *mut Model, mode: i32);
    fn Clp_status(m: *mut Model) -> i32;
    fn Clp_secondaryStatus(m: *mut Model) -> i32;
    fn Clp_objectiveValue(m: *mut Model) -> f64;
    fn Clp_dualRowSolution(m: *mut Model) -> *const f64;
    fn Clp_primalColumnSolution(m: *mut Model) -> *const f64;
    fn Clp_getReducedCost(m: *mut Model) -> *const f64;
    fn Clp_setDualTolerance(m: *mut Model, tolerance: f64);
    fn Clp_setPrimalTolerance(m: *mut Model, tolerance: f64);
    fn Clp_getRowStatus(m: *mut Model, row: i32) -> i32;
    fn Clp_getVectorStarts(m: *mut Model) -> *const i32;
    fn Clp_getVectorLengths(m: *mut Model) -> *const i32;
    fn Clp_getIndices(m: *mut Model) -> *const i32;
    fn Clp_getElements(m: *mut Model) -> *const f64;
}
/// Outcome for this LP only; it is not an original routing-domain certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClpSolveOutcome {
    Optimal,
    PrimalInfeasible,
    DualInfeasible,
    Stopped,
    Failed,
}
/// Algorithm used for restricted-master LP solves. The selected native backend
/// owns the mutable matrix; IPM uses Clp storage and computes its own solutions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MasterSolver {
    #[default]
    Clp,
    Ipm,
    Highs,
}

impl MasterSolver {
    pub fn ensure_available(self) -> Result<()> {
        if self == Self::Highs && !cfg!(feature = "highs") {
            return Err(invalid(
                "HiGHS master requires the highs build feature (build with clp,highs)",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ClpMaster {
    // Exactly one native matrix is owned: Clp here, or highs_backend below.
    // A requested solver change is materialized by ensure_backend.
    model: Option<NonNull<Model>>,
    solver: MasterSolver,
    clp_options: ClpOptions,
    ipm: ipm_state::IpmState,
    highs_row_basic: Vec<bool>,
    #[cfg(feature = "highs")]
    highs_backend: Option<super::highs::HighsMaster>,
    primal_tolerance: f64,
    pub rhs: Vec<f64>,
    pub senses: Vec<RowSense>,
    pub costs: Vec<f64>,
    pub upper: Vec<f64>,
    column_default_upper: f64,
    pub duals: Vec<f64>,
    pub values: Vec<f64>,
    pub reduced_costs: Vec<f64>,
    pub objective: f64,
    /// Constant objective contribution, retained by forks and integer transfer.
    pub objective_offset: f64,
    pub status: i32,
    pub solved: bool,
    pub dual_tolerance: f64,
    pub simplex_iterations: usize,
    /// Completed Newton iterations reported by IPM solves on this model.
    pub ipm_iterations: usize,
    /// Primal values and duals with magnitude at most this are read as zero,
    /// using a default tolerance of 1e-7.
    pub zero_tolerance: f64,
}

impl Drop for ClpMaster {
    fn drop(&mut self) {
        if let Some(model) = self.model {
            unsafe { Clp_deleteModel(model.as_ptr()) };
        }
    }
}

// ── LpMaster trait implementation ─────────────────────────────────────

fn entries_of(entries: &[(usize, f64)], size: usize) -> Result<(Vec<i32>, Vec<f64>)> {
    // Routing columns and separated rows already have sorted, unique indices.
    // Validate and copy them directly instead of allocating a tree to sort
    // and accumulate an incidence that is already canonical.
    if entries.windows(2).all(|pair| pair[0].0 < pair[1].0) {
        let mut ids = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());
        for &(id, value) in entries {
            if id >= size || id > i32::MAX as usize || !value.is_finite() {
                return Err(invalid("sparse master entry"));
            }
            if value != 0. {
                ids.push(id as i32);
                values.push(value);
            }
        }
        return Ok((ids, values));
    }
    let mut accumulated = std::collections::BTreeMap::new();
    for &(id, value) in entries {
        if id >= size || id > i32::MAX as usize || !value.is_finite() {
            return Err(invalid("sparse master entry"));
        }
        *accumulated.entry(id).or_insert(0.) += value;
    }
    let (ids, values) = accumulated
        .into_iter()
        .filter(|(_, v)| *v != 0.)
        .map(|(i, v)| (i as i32, v))
        .unzip();
    Ok((ids, values))
}

#[cfg(test)]
mod continuation_tests;

#[cfg(all(test, feature = "highs"))]
mod highs_warm_tests;

mod ipm_state;
mod matrix;
mod model;
mod solve;

#[cfg(test)]
mod ipm_warm_tests;

mod interface;

mod options;
pub use options::{ClpOptions, ClpScaling};

#[cfg(test)]
mod reoptimization_tests;

#[cfg(test)]
mod recovery_tests;
