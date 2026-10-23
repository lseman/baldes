//! Ownership-safe bridge to the HiGHS C interface (HiGHS 1.x `highs_c_api.h`).
//!
//! `HighsMaster` keeps the same bookkeeping as `ClpMaster`: columns have
//! bounds [0, 1e12], solutions are read after each optimal solve with values
//! of magnitude at most the zero tolerance snapped to zero, appending
//! dual-feasible columns keeps the current solution, and cut cleanup uses the
//! shared [`crate::master::lp_master::cleanup_cut_rows`]. HiGHS follows its own pivot
//! rules, so trajectories differ from Clp's even on identical masters. Native
//! models retain their basis across edits; `try_clone` installs an independent
//! copy of that basis in a child model.
use crate::master::{LpMaster, LpSolveOutcome, RowSense};
use crate::{incomplete, invalid, Result};
use std::cell::OnceCell;
use std::ffi::{c_char, c_double, c_void, CString};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

/// `HighsInt` of the default (32-bit) HiGHS build.
type HighsInt = i32;

extern "C" {
    fn Highs_create() -> *mut c_void;
    fn Highs_destroy(highs: *mut c_void);
    fn Highs_run(highs: *mut c_void) -> HighsInt;
    fn Highs_clearSolver(highs: *mut c_void) -> HighsInt;
    fn Highs_setBoolOptionValue(
        highs: *mut c_void,
        option: *const c_char,
        value: HighsInt,
    ) -> HighsInt;
    fn Highs_setDoubleOptionValue(
        highs: *mut c_void,
        option: *const c_char,
        value: c_double,
    ) -> HighsInt;
    fn Highs_getDoubleOptionValue(
        highs: *const c_void,
        option: *const c_char,
        value: *mut f64,
    ) -> HighsInt;
    fn Highs_getRunTime(highs: *const c_void) -> f64;
    fn Highs_setIntOptionValue(
        highs: *mut c_void,
        option: *const c_char,
        value: HighsInt,
    ) -> HighsInt;
    fn Highs_setStringOptionValue(
        highs: *mut c_void,
        option: *const c_char,
        value: *const c_char,
    ) -> HighsInt;
    fn Highs_getIntOptionValues(
        highs: *const c_void,
        option: *const c_char,
        current: *mut HighsInt,
        minimum: *mut HighsInt,
        maximum: *mut HighsInt,
        default_value: *mut HighsInt,
    ) -> HighsInt;
    fn Highs_getInfinity(highs: *const c_void) -> c_double;
    #[allow(clippy::too_many_arguments)]
    fn Highs_addRows(
        highs: *mut c_void,
        num_new_row: HighsInt,
        lower: *const f64,
        upper: *const f64,
        num_new_nz: HighsInt,
        starts: *const HighsInt,
        index: *const HighsInt,
        value: *const f64,
    ) -> HighsInt;
    #[allow(clippy::too_many_arguments)]
    fn Highs_addCols(
        highs: *mut c_void,
        num_new_col: HighsInt,
        costs: *const c_double,
        lower: *const c_double,
        upper: *const c_double,
        num_new_nz: HighsInt,
        starts: *const HighsInt,
        index: *const HighsInt,
        value: *const c_double,
    ) -> HighsInt;
    fn Highs_changeColCost(highs: *mut c_void, col: HighsInt, cost: c_double) -> HighsInt;
    fn Highs_changeColBounds(
        highs: *mut c_void,
        col: HighsInt,
        lower: c_double,
        upper: c_double,
    ) -> HighsInt;
    fn Highs_changeCoeff(
        highs: *mut c_void,
        row: HighsInt,
        col: HighsInt,
        value: c_double,
    ) -> HighsInt;
    fn Highs_deleteRowsBySet(
        highs: *mut c_void,
        num_set_entries: HighsInt,
        set: *const HighsInt,
    ) -> HighsInt;
    fn Highs_deleteColsBySet(
        highs: *mut c_void,
        num_set_entries: HighsInt,
        set: *const HighsInt,
    ) -> HighsInt;
    fn Highs_getModelStatus(highs: *const c_void) -> HighsInt;
    fn Highs_getIntInfoValue(
        highs: *const c_void,
        info: *const c_char,
        value: *mut HighsInt,
    ) -> HighsInt;
    fn Highs_setBasis(
        highs: *mut c_void,
        col_status: *const HighsInt,
        row_status: *const HighsInt,
    ) -> HighsInt;
    fn Highs_changeRowsBoundsByRange(
        highs: *mut c_void,
        from_row: HighsInt,
        to_row: HighsInt,
        lower: *const f64,
        upper: *const f64,
    ) -> HighsInt;
    fn Highs_getObjectiveValue(highs: *const c_void) -> c_double;
    fn Highs_getSolution(
        highs: *const c_void,
        col_value: *mut c_double,
        col_dual: *mut c_double,
        row_value: *mut c_double,
        row_dual: *mut c_double,
    ) -> HighsInt;
    fn Highs_getBasis(
        highs: *const c_void,
        col_status: *mut HighsInt,
        row_status: *mut HighsInt,
    ) -> HighsInt;
    #[allow(clippy::too_many_arguments)]
    fn Highs_getColsByRange(
        highs: *const c_void,
        from_col: HighsInt,
        to_col: HighsInt,
        num_col: *mut HighsInt,
        costs: *mut c_double,
        lower: *mut c_double,
        upper: *mut c_double,
        num_nz: *mut HighsInt,
        matrix_start: *mut HighsInt,
        matrix_index: *mut HighsInt,
        matrix_value: *mut c_double,
    ) -> HighsInt;
}

const STATUS_ERROR: HighsInt = -1;
const MODEL_OPTIMAL: HighsInt = 7;
const MODEL_SOLVE_ERROR: HighsInt = 4;
const MODEL_INFEASIBLE: HighsInt = 8;
const MODEL_UNBOUNDED_OR_INFEASIBLE: HighsInt = 9;
const MODEL_UNBOUNDED: HighsInt = 10;
const MODEL_TIME_LIMIT: HighsInt = 13;
const MODEL_ITERATION_LIMIT: HighsInt = 14;
const MODEL_UNKNOWN: HighsInt = 15;
const BASIS_BASIC: HighsInt = 1;

struct ScopedTimeLimit {
    highs: NonNull<c_void>,
    previous: f64,
    active: bool,
}
impl ScopedTimeLimit {
    fn restore(&mut self) -> Result<()> {
        // SAFETY: this private guard is scoped within the owned model's borrow.
        let status = unsafe {
            Highs_setDoubleOptionValue(
                self.highs.as_ptr(),
                b"time_limit\0".as_ptr().cast(),
                self.previous,
            )
        };
        HighsMaster::check(status, "restore time_limit")?;
        self.active = false;
        Ok(())
    }
}
impl Drop for ScopedTimeLimit {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: the model outlives this local guard, including unwinding.
            unsafe {
                Highs_setDoubleOptionValue(
                    self.highs.as_ptr(),
                    b"time_limit\0".as_ptr().cast(),
                    self.previous,
                );
            }
        }
    }
}
/// Column upper bound, as the Clp master (columns carry 1e12).
const COLUMN_UPPER: f64 = 1e12;

/// Immutable CSC snapshot, populated in bulk only when matrix reads are needed.
/// Native edits invalidate it; it owns no solver state or factorization.
#[derive(Debug)]
struct SparseColumns {
    starts: Vec<HighsInt>,
    indices: Vec<HighsInt>,
    values: Vec<f64>,
}

#[derive(Debug)]
pub struct HighsMaster {
    highs: NonNull<c_void>,
    thread_config: Option<(usize, usize)>,
    adaptive_threads: bool,
    matrix_cache: OnceCell<SparseColumns>,
    pub rhs: Vec<f64>,
    pub senses: Vec<RowSense>,
    pub costs: Vec<f64>,
    pub upper: Vec<f64>,
    pub duals: Vec<f64>,
    pub values: Vec<f64>,
    pub reduced_costs: Vec<f64>,
    /// Basis status of each row after the last optimal solve.
    row_status: Vec<HighsInt>,
    row_values: Vec<f64>,
    col_status: Vec<HighsInt>,
    primal_tolerance: f64,
    iteration_limit: Option<usize>,
    pub objective: f64,
    pub objective_offset: f64,
    /// -1 unsolved, 0 optimal, 1 primal infeasible, 2 dual infeasible.
    pub status: i32,
    pub solved: bool,
    pub dual_tolerance: f64,
    /// Values and duals of magnitude at most this read as zero (0 keeps them).
    pub zero_tolerance: f64,
}

impl Drop for HighsMaster {
    fn drop(&mut self) {
        // SAFETY: the handle is owned and destroyed once.
        unsafe { Highs_destroy(self.highs.as_ptr()) }
    }
}

/// Old-to-new map after removing `removed`, and the removed indices ascending.
fn mapping(size: usize, removed: &[usize]) -> Result<(Vec<Option<usize>>, Vec<HighsInt>)> {
    let mut ids = removed.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.iter().any(|&i| i >= size) {
        return Err(invalid("master deletion index"));
    }
    let mut next = 0;
    let map = (0..size)
        .map(|i| {
            if ids.binary_search(&i).is_ok() {
                None
            } else {
                next += 1;
                Some(next - 1)
            }
        })
        .collect();
    Ok((map, ids.into_iter().map(|i| i as HighsInt).collect()))
}

fn entries_of(entries: &[(usize, f64)], size: usize) -> Result<(Vec<HighsInt>, Vec<f64>)> {
    let mut accumulated = std::collections::BTreeMap::new();
    for &(id, value) in entries {
        if id >= size || id > HighsInt::MAX as usize || !value.is_finite() {
            return Err(invalid("sparse master entry"));
        }
        *accumulated.entry(id).or_insert(0.) += value;
    }
    Ok(accumulated
        .into_iter()
        .filter(|(_, v)| *v != 0.)
        .map(|(i, v)| (i as HighsInt, v))
        .unzip())
}

mod integer;
pub use integer::{solve_integer, solve_integer_with_deluxing};

#[cfg(test)]
#[path = "../../../../tests/unit/master/solvers/highs/limit_tests.rs"]
mod limit_tests;

mod interface;

mod model;
