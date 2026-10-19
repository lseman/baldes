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
    fn Highs_setBoolOptionValue(highs: *mut c_void, option: *const c_char, value: HighsInt) -> HighsInt;
    fn Highs_setDoubleOptionValue(highs: *mut c_void, option: *const c_char, value: c_double) -> HighsInt;
    fn Highs_getDoubleOptionValue(highs: *const c_void, option: *const c_char, value: *mut f64) -> HighsInt;
    fn Highs_getRunTime(highs: *const c_void) -> f64;
    fn Highs_setIntOptionValue(highs: *mut c_void, option: *const c_char, value: HighsInt) -> HighsInt;
    #[cfg(test)]
    fn Highs_setStringOptionValue(highs: *mut c_void, option: *const c_char, value: *const c_char) -> HighsInt;
    fn Highs_getInfinity(highs: *const c_void) -> c_double;
    #[allow(clippy::too_many_arguments)]
    fn Highs_addRows(highs: *mut c_void, num_new_row: HighsInt, lower: *const f64, upper: *const f64,
        num_new_nz: HighsInt, starts: *const HighsInt, index: *const HighsInt, value: *const f64) -> HighsInt;
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
    fn Highs_changeColBounds(highs: *mut c_void, col: HighsInt, lower: c_double, upper: c_double) -> HighsInt;
    fn Highs_changeCoeff(highs: *mut c_void, row: HighsInt, col: HighsInt, value: c_double) -> HighsInt;
    fn Highs_deleteRowsBySet(highs: *mut c_void, num_set_entries: HighsInt, set: *const HighsInt) -> HighsInt;
    fn Highs_deleteColsBySet(highs: *mut c_void, num_set_entries: HighsInt, set: *const HighsInt) -> HighsInt;
    fn Highs_getModelStatus(highs: *const c_void) -> HighsInt;
    fn Highs_getIntInfoValue(highs: *const c_void, info: *const c_char, value: *mut HighsInt) -> HighsInt;
    fn Highs_setBasis(highs: *mut c_void, col_status: *const HighsInt, row_status: *const HighsInt) -> HighsInt;
    fn Highs_changeRowsBoundsByRange(highs: *mut c_void, from_row: HighsInt, to_row: HighsInt, lower: *const f64, upper: *const f64) -> HighsInt;
    fn Highs_getObjectiveValue(highs: *const c_void) -> c_double;
    fn Highs_getSolution(
        highs: *const c_void,
        col_value: *mut c_double,
        col_dual: *mut c_double,
        row_value: *mut c_double,
        row_dual: *mut c_double,
    ) -> HighsInt;
    fn Highs_getBasis(highs: *const c_void, col_status: *mut HighsInt, row_status: *mut HighsInt) -> HighsInt;
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
        let status = unsafe { Highs_setDoubleOptionValue(self.highs.as_ptr(), b"time_limit\0".as_ptr().cast(), self.previous) };
        HighsMaster::check(status, "restore time_limit")?;
        self.active = false;
        Ok(())
    }
}
impl Drop for ScopedTimeLimit {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: the model outlives this local guard, including unwinding.
            unsafe { Highs_setDoubleOptionValue(self.highs.as_ptr(), b"time_limit\0".as_ptr().cast(), self.previous); }
        }
    }
}
/// Column upper bound, as the Clp master (columns carry 1e12).
const COLUMN_UPPER: f64 = 1e12;

#[derive(Debug)]
pub struct HighsMaster {
    highs: NonNull<c_void>,
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

impl HighsMaster {
    pub fn new() -> Result<Self> {
        // SAFETY: Highs_create returns an exclusively owned handle.
        let highs = NonNull::new(unsafe { Highs_create() })
            .ok_or_else(|| incomplete("HiGHS model allocation failed"))?;
        let master = Self {
            highs,
            rhs: vec![],
            senses: vec![],
            costs: vec![],
            upper: vec![],
            duals: vec![],
            values: vec![],
            reduced_costs: vec![],
            row_status: vec![],
            row_values: vec![],
            col_status: vec![],
            primal_tolerance: 1e-7,
            iteration_limit: None,
            objective: 0.,
            objective_offset: 0.,
            status: -1,
            solved: false,
            dual_tolerance: 1e-7,
            zero_tolerance: 0.,
        };
        master.set_bool("output_flag", false)?;
        // Adding columns preserves primal feasibility but can invalidate dual
        // feasibility; adding cuts or changing bounds can do the reverse.
        // Let simplex choose the direction from the retained basis instead of
        // HiGHS's default of always using dual simplex.
        // SAFETY: live owned handle and a static NUL-terminated option name.
        unsafe {
            Self::check(Highs_setIntOptionValue(
                highs.as_ptr(), b"simplex_strategy\0".as_ptr().cast(), 0,
            ), "set adaptive simplex strategy")?;
        }
        Ok(master)
    }
    /// Independently copy the model and install its current native basis.
    /// Native factorization storage is private to each instance; the child
    /// refactorizes this basis once, then reuses its own solver state.
    pub fn try_clone(&self) -> Result<Self> {
        let mut copy = Self::new()?;
        copy.set_tolerances(self.primal_tolerance, self.dual_tolerance)?;
        let rows = self.senses.iter().zip(&self.rhs).map(|(&s,&b)| (s,b,&[][..])).collect::<Vec<_>>();
        copy.add_rows(&rows)?;
        let count = self.costs.len();
        if count > 0 {
            let mut num_col = 0;
            let mut nnz = 0;
            let mut costs = vec![0.;count];
            let mut lower = vec![0.;count];
            let mut upper = vec![0.;count];
            // SAFETY: first call sizes the sparse matrix, second fills arrays
            // of that size. The destination copies the complete CSC in one call.
            unsafe {
                Self::check(Highs_getColsByRange(self.highs.as_ptr(),0,count as HighsInt-1,
                    &mut num_col,costs.as_mut_ptr(),lower.as_mut_ptr(),upper.as_mut_ptr(),&mut nnz,
                    std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut()),"size clone columns")?;
                if nnz < 0 || num_col != count as HighsInt { return Err(incomplete("HiGHS clone matrix dimensions")); }
                let mut starts = vec![0;count+1];
                let mut index = vec![0;nnz as usize];
                let mut values = vec![0.;nnz as usize];
                Self::check(Highs_getColsByRange(self.highs.as_ptr(),0,count as HighsInt-1,
                    &mut num_col,costs.as_mut_ptr(),lower.as_mut_ptr(),upper.as_mut_ptr(),&mut nnz,
                    starts.as_mut_ptr(),index.as_mut_ptr(),values.as_mut_ptr()),"copy clone columns")?;
                starts[count] = nnz;
                Self::check(Highs_addCols(copy.highs.as_ptr(),count as HighsInt, costs.as_ptr(),lower.as_ptr(),upper.as_ptr(),
                    nnz,starts.as_ptr(),index.as_ptr(),values.as_ptr()),"add clone columns")?;
            }
            copy.costs.clone_from(&self.costs);
            copy.upper.clone_from(&self.upper);
        }
        // Query the current basis directly: edits may invalidate solution info
        // while retaining a valid (possibly infeasible) simplex warm basis.
        let mut columns = vec![0; self.costs.len()];
        let mut rows = vec![0; self.rhs.len()];
        // SAFETY: source is borrowed, destination independently owns its basis.
        unsafe {
            if Highs_getBasis(self.highs.as_ptr(), columns.as_mut_ptr(), rows.as_mut_ptr()) == 0 {
                Self::check(Highs_setBasis(copy.highs.as_ptr(), columns.as_ptr(), rows.as_ptr()), "setBasis for clone")?;
            }
        }
        copy.values.clone_from(&self.values);
        copy.duals.clone_from(&self.duals);
        copy.reduced_costs.clone_from(&self.reduced_costs);
        copy.row_status.clone_from(&self.row_status);
        copy.col_status.clone_from(&self.col_status);
        copy.objective = self.objective;
        copy.objective_offset = self.objective_offset;
        copy.zero_tolerance = self.zero_tolerance;
        copy.status = self.status;
        copy.solved = self.solved;
        Ok(copy)
    }
    /// Replace RHS values without rebuilding the model or discarding the basis.
    pub fn set_rhs(&mut self, rhs: &[f64]) -> Result<()> {
        if rhs.len() != self.rhs.len() || rhs.iter().any(|v| !v.is_finite()) {
            return Err(invalid("HiGHS RHS dimensions or values"));
        }
        if rhs == self.rhs { return Ok(()); }
        let infinity = self.infinity();
        let lower: Vec<_> = rhs.iter().zip(&self.senses).map(|(&b,&s)| if s == RowSense::LessEqual {-infinity} else {b}).collect();
        let upper: Vec<_> = rhs.iter().zip(&self.senses).map(|(&b,&s)| if s == RowSense::GreaterEqual {infinity} else {b}).collect();
        if !rhs.is_empty() {
            // SAFETY: one lower/upper bound for every native row; HiGHS copies.
            unsafe { Self::check(Highs_changeRowsBoundsByRange(self.highs.as_ptr(), 0, (rhs.len()-1) as HighsInt, lower.as_ptr(), upper.as_ptr()), "changeRowsBoundsByRange")?; }
        }
        self.rhs.copy_from_slice(rhs);
        self.solved = false;
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn native_identity(&self) -> usize { self.highs.as_ptr() as usize }
    /// Simplex pivots from the most recent native solve.
    pub fn simplex_iterations(&self) -> Result<usize> {
        let mut count = 0;
        // SAFETY: live instance and one integer output.
        unsafe { Self::check(Highs_getIntInfoValue(self.highs.as_ptr(), b"simplex_iteration_count\0".as_ptr().cast(), &mut count), "simplex_iteration_count")?; }
        usize::try_from(count).map_err(|_| incomplete("negative HiGHS iteration count"))
    }
    pub(crate) fn set_lp_iteration_limit(&mut self, iterations: usize) -> Result<()> {
        if self.iteration_limit == Some(iterations) { return Ok(()); }
        let limit = i32::try_from(iterations).map_err(|_| invalid("HiGHS iteration limit"))?;
        // SAFETY: live owned handle and static NUL-terminated option names.
        unsafe {
            Self::check(Highs_setIntOptionValue(self.highs.as_ptr(), b"simplex_iteration_limit\0".as_ptr().cast(), limit), "simplex_iteration_limit")?;
            Self::check(Highs_setIntOptionValue(self.highs.as_ptr(), b"ipm_iteration_limit\0".as_ptr().cast(), limit), "ipm_iteration_limit")?;
        }
        self.iteration_limit = Some(iterations);
        Ok(())
    }
    fn set_bool(&self, name: &str, value: bool) -> Result<()> {
        let name_c = CString::new(name).map_err(|_| invalid("HiGHS option name"))?;
        // SAFETY: live handle, NUL-terminated option name.
        let rc = unsafe { Highs_setBoolOptionValue(self.highs.as_ptr(), name_c.as_ptr(), value as HighsInt) };
        if rc == STATUS_ERROR {
            return Err(invalid(format!("HiGHS option {name}")));
        }
        Ok(())
    }
    fn set_double(&self, name: &str, value: f64) -> Result<()> {
        let name_c = CString::new(name).map_err(|_| invalid("HiGHS option name"))?;
        // SAFETY: live handle, NUL-terminated option name.
        let rc = unsafe { Highs_setDoubleOptionValue(self.highs.as_ptr(), name_c.as_ptr(), value) };
        if rc == STATUS_ERROR {
            return Err(invalid(format!("HiGHS option {name}")));
        }
        Ok(())
    }
    fn infinity(&self) -> f64 {
        // SAFETY: live handle.
        unsafe { Highs_getInfinity(self.highs.as_ptr()) }
    }
    fn check(rc: HighsInt, what: &str) -> Result<()> {
        if rc == STATUS_ERROR {
            Err(invalid(format!("HiGHS {what} failed")))
        } else {
            Ok(())
        }
    }
}

impl Drop for HighsMaster {
    fn drop(&mut self) {
        // SAFETY: the handle is owned and destroyed once.
        unsafe { Highs_destroy(self.highs.as_ptr()) }
    }
}

impl LpMaster for HighsMaster {
    fn add_row(&mut self, sense: RowSense, rhs: f64, entries: &[(usize, f64)]) -> Result<usize> {
        Ok(self.add_rows(&[(sense,rhs,entries)])?.start)
    }

    fn add_rows(&mut self, rows: &[(RowSense, f64, &[(usize, f64)])]) -> Result<std::ops::Range<usize>> {
        let first = self.rhs.len();
        if rows.is_empty() { return Ok(first..first); }
        let infinity = self.infinity();
        let mut lower = Vec::with_capacity(rows.len());
        let mut upper = Vec::with_capacity(rows.len());
        let mut starts = vec![0];
        let mut indices = vec![];
        let mut values = vec![];
        for &(sense,rhs,entries) in rows {
            if !rhs.is_finite() { return Err(invalid("finite row RHS required")); }
            let (i,v) = entries_of(entries,self.costs.len())?;
            lower.push(if sense == RowSense::LessEqual {-infinity} else {rhs});
            upper.push(if sense == RowSense::GreaterEqual {infinity} else {rhs});
            indices.extend(i); values.extend(v); starts.push(indices.len() as HighsInt);
        }
        // SAFETY: buffers contain all row bounds and declared sparse entries;
        // HiGHS copies them without invalidating the surviving basis columns.
        unsafe { Self::check(Highs_addRows(self.highs.as_ptr(),rows.len() as HighsInt,lower.as_ptr(),upper.as_ptr(),
            indices.len() as HighsInt,starts.as_ptr(),indices.as_ptr(),values.as_ptr()),"addRows")?; }
        for &(sense,rhs,_) in rows { self.senses.push(sense); self.rhs.push(rhs); }
        self.solved = false;
        Ok(first..self.rhs.len())
    }

    fn add_column(&mut self, cost: f64, entries: &[(usize, f64)]) -> Result<usize> {
        Ok(self.add_columns(&[(cost, entries)])?.start)
    }

    /// As `ClpMaster::add_columns`: one call; a solution is kept, with zero
    /// values, for the leading columns that are dual feasible.
    fn add_columns(&mut self, columns: &[(f64, &[(usize, f64)])]) -> Result<std::ops::Range<usize>> {
        let first = self.costs.len();
        if columns.is_empty() {
            return Ok(first..first);
        }
        let mut starts = vec![];
        let mut indices = vec![];
        let mut values = vec![];
        let mut costs = Vec::with_capacity(columns.len());
        let mut reduced = Vec::with_capacity(columns.len());
        let mut preserve = self.solved;
        let mut preserved = 0;
        for &(cost, entries) in columns {
            if !cost.is_finite() {
                return Err(invalid("finite column cost required"));
            }
            let (ids, vals) = entries_of(entries, self.rhs.len())?;
            let mut rc = cost;
            for (&r, &a) in ids.iter().zip(&vals) {
                rc -= self.duals.get(r as usize).copied().unwrap_or(0.) * a;
            }
            preserve = preserve && rc >= -self.dual_tolerance;
            preserved += preserve as usize;
            starts.push(indices.len() as HighsInt);
            indices.extend(ids);
            values.extend(vals);
            costs.push(cost);
            reduced.push(rc);
        }
        let lower = vec![0.; columns.len()];
        let upper = vec![COLUMN_UPPER; columns.len()];
        // SAFETY: `starts` has one entry per column, `indices`/`values` the
        // declared nonzeros; HiGHS copies them.
        let rc = unsafe {
            Highs_addCols(
                self.highs.as_ptr(),
                columns.len() as HighsInt,
                costs.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
                indices.len() as HighsInt,
                starts.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
            )
        };
        Self::check(rc, "addCols")?;
        self.costs.extend(&costs);
        self.upper.extend(&upper);
        self.solved = preserve;
        self.values.resize(self.values.len() + preserved, 0.);
        self.reduced_costs.extend(&reduced[..preserved]);
        Ok(first..self.costs.len())
    }

    fn set_coefficient(&mut self, row: usize, column: usize, value: f64) -> Result<()> {
        if row >= self.rhs.len() || column >= self.costs.len() || !value.is_finite() {
            return Err(invalid("master coefficient index/value"));
        }
        // SAFETY: indices checked above.
        let rc = unsafe { Highs_changeCoeff(self.highs.as_ptr(), row as HighsInt, column as HighsInt, value) };
        Self::check(rc, "changeCoeff")?;
        self.solved = false;
        Ok(())
    }

    fn set_cost(&mut self, column: usize, cost: f64) -> Result<()> {
        if column >= self.costs.len() || !cost.is_finite() {
            return Err(invalid("master cost index/value"));
        }
        // SAFETY: index checked above.
        let rc = unsafe { Highs_changeColCost(self.highs.as_ptr(), column as HighsInt, cost) };
        Self::check(rc, "changeColCost")?;
        self.costs[column] = cost;
        self.solved = false;
        Ok(())
    }

    fn set_upper(&mut self, column: usize, upper: f64) -> Result<()> {
        if column >= self.upper.len() || !upper.is_finite() || upper < 0. {
            return Err(invalid("master upper bound"));
        }
        // SAFETY: index checked above.
        let rc = unsafe { Highs_changeColBounds(self.highs.as_ptr(), column as HighsInt, 0., upper) };
        Self::check(rc, "changeColBounds")?;
        self.upper[column] = upper;
        self.solved = false;
        Ok(())
    }

    fn set_tolerances(&mut self, primal: f64, dual: f64) -> Result<()> {
        if !primal.is_finite() || !dual.is_finite() || primal <= 0. || dual <= 0. {
            return Err(invalid("positive finite HiGHS tolerances required"));
        }
        self.set_double("primal_feasibility_tolerance", primal)?;
        self.set_double("dual_feasibility_tolerance", dual)?;
        if self.primal_tolerance != primal || self.dual_tolerance != dual { self.solved = false; }
        self.primal_tolerance = primal;
        self.dual_tolerance = dual;
        Ok(())
    }

    fn solve(&mut self) -> bool {
        if self.costs.is_empty() {
            // No variable can satisfy a nonzero required activity. Do not
            // interpret a backend "empty model" status as feasibility.
            let feasible = self.senses.iter().zip(&self.rhs).all(|(sense, &rhs)| match sense {
                RowSense::Equal => rhs == 0.,
                RowSense::LessEqual => rhs >= 0.,
                RowSense::GreaterEqual => rhs <= 0.,
            });
            self.status = if feasible { 0 } else { 1 };
            self.solved = feasible;
            if feasible {
                self.objective = self.objective_offset;
                self.values.clear();
                self.reduced_costs.clear();
                self.duals = vec![0.; self.rhs.len()];
                self.row_status = vec![BASIS_BASIC; self.rhs.len()];
            }
            return feasible;
        }
        let h = self.highs.as_ptr();
        // SAFETY: live handle; the buffers below have the model's dimensions.
        unsafe {
            let mut result = Highs_run(h);
            if matches!(Highs_getModelStatus(h), MODEL_SOLVE_ERROR | MODEL_UNKNOWN) {
                // A retained simplex basis can fail numerically after column
                // edits even though a cold solve of the same LP succeeds.
                // Retry once, discarding only solver state. Options (including
                // the cumulative time limit) and the model stay unchanged.
                if Highs_clearSolver(h) != STATUS_ERROR {
                    result = Highs_run(h);
                }
            }
            if result == STATUS_ERROR {
                self.status = 4;
                self.solved = false;
                return false;
            }
            self.status = match Highs_getModelStatus(h) {
                MODEL_OPTIMAL => 0,
                MODEL_INFEASIBLE => 1,
                MODEL_UNBOUNDED => 2,
                MODEL_TIME_LIMIT | MODEL_ITERATION_LIMIT => 3,
                MODEL_UNBOUNDED_OR_INFEASIBLE => -1,
                _ => -1,
            };
            self.solved = self.status == 0;
            if !self.solved {
                return false;
            }
            self.objective = Highs_getObjectiveValue(h) + self.objective_offset;
            let (rows, cols) = (self.rhs.len(), self.costs.len());
            self.values.resize(cols, 0.);
            self.reduced_costs.resize(cols, 0.);
            self.duals.resize(rows, 0.);
            self.row_values.resize(rows, 0.);
            Highs_getSolution(
                h,
                self.values.as_mut_ptr(),
                self.reduced_costs.as_mut_ptr(),
                self.row_values.as_mut_ptr(),
                self.duals.as_mut_ptr(),
            );
            self.col_status.resize(cols, 0);
            self.row_status.resize(rows, 0);
            Highs_getBasis(h, self.col_status.as_mut_ptr(), self.row_status.as_mut_ptr());
        }
        let tolerance = self.zero_tolerance;
        if tolerance > 0. {
            for x in self.duals.iter_mut().chain(self.values.iter_mut()).chain(self.reduced_costs.iter_mut()) {
                if !(*x > tolerance || *x < -tolerance) {
                    *x = 0.;
                }
            }
        }
        true
    }

    fn solve_with_time_limit(&mut self, remaining: Duration) -> Result<LpSolveOutcome> {
        let started = Instant::now();
        if remaining.is_zero() {
            self.status = 3;
            self.solved = false;
            return Ok(LpSolveOutcome::TimeLimit);
        }
        let mut previous = 0.;
        // SAFETY: live handle; option getter writes one double. HiGHS run time
        // is cumulative across Highs_run calls, as is its time_limit option.
        let runtime = unsafe {
            Self::check(Highs_getDoubleOptionValue(self.highs.as_ptr(), b"time_limit\0".as_ptr().cast(), &mut previous), "get time_limit")?;
            Highs_getRunTime(self.highs.as_ptr())
        };
        if !runtime.is_finite() || runtime < 0. || previous.is_nan() || previous < 0. {
            return Err(invalid("HiGHS time budget"));
        }
        if previous <= runtime {
            self.status = 3;
            self.solved = false;
            return Ok(LpSolveOutcome::TimeLimit);
        }
        self.set_double("time_limit", previous.min(runtime + remaining.as_secs_f64()))?;
        let mut guard = ScopedTimeLimit { highs: self.highs, previous, active: true };
        let solved = self.solve();
        if let Err(error) = guard.restore() {
            self.status = 4;
            self.solved = false;
            return Err(error);
        }
        if started.elapsed() >= remaining {
            self.status = 3;
            self.solved = false;
            return Ok(LpSolveOutcome::TimeLimit);
        }
        Ok(if solved { LpSolveOutcome::Optimal } else {
            match self.status {
                1 => LpSolveOutcome::PrimalInfeasible,
                2 => LpSolveOutcome::DualInfeasible,
                3 => {
                    // SAFETY: live model; only inspect a successfully returned stop.
                    match unsafe { Highs_getModelStatus(self.highs.as_ptr()) } {
                        MODEL_TIME_LIMIT => LpSolveOutcome::TimeLimit,
                        MODEL_ITERATION_LIMIT => LpSolveOutcome::IterationLimit,
                        _ => LpSolveOutcome::Stopped,
                    }
                }
                4 => LpSolveOutcome::Failed,
                _ => LpSolveOutcome::Unknown,
            }
        })
    }

    fn duals(&self) -> &[f64] {
        &self.duals
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
    fn reduced_costs(&self) -> &[f64] {
        &self.reduced_costs
    }
    fn objective(&self) -> f64 {
        self.objective
    }
    fn objective_offset(&self) -> f64 { self.objective_offset }
    fn status(&self) -> i32 {
        self.status
    }
    fn solved(&self) -> bool {
        self.solved
    }
    fn rhs(&self) -> &[f64] {
        &self.rhs
    }
    fn senses(&self) -> &[RowSense] {
        &self.senses
    }
    fn costs(&self) -> &[f64] {
        &self.costs
    }
    fn upper(&self) -> &[f64] {
        &self.upper
    }

    fn row_is_basic(&self, row: usize) -> Result<bool> {
        if row >= self.rhs.len() {
            return Err(invalid("master row index"));
        }
        if !self.solved {
            return Err(invalid("HiGHS basis requested before an optimal solve"));
        }
        Ok(self.row_status[row] == BASIS_BASIC)
    }

    fn cleanup_cut_rows(&self, initial_rows: usize) -> Result<Vec<usize>> {
        crate::master::lp_master::cleanup_cut_rows(self, initial_rows)
    }

    fn column_entries(&self, column: usize) -> Result<Vec<(usize, f64)>> {
        if column >= self.costs.len() {
            return Err(invalid("master column index"));
        }
        let h = self.highs.as_ptr();
        let col = column as HighsInt;
        let (mut num_col, mut num_nz) = (0, 0);
        let (mut cost, mut lower, mut upper) = (0., 0., 0.);
        let mut start = 0;
        // SAFETY: first call sizes the column; the second fills buffers of that size.
        unsafe {
            Self::check(
                Highs_getColsByRange(
                    h, col, col, &mut num_col, &mut cost, &mut lower, &mut upper, &mut num_nz,
                    std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
                ),
                "getColsByRange",
            )?;
            let mut index = vec![0; num_nz as usize];
            let mut value = vec![0.; num_nz as usize];
            Self::check(
                Highs_getColsByRange(
                    h, col, col, &mut num_col, &mut cost, &mut lower, &mut upper, &mut num_nz,
                    &mut start, index.as_mut_ptr(), value.as_mut_ptr(),
                ),
                "getColsByRange",
            )?;
            let mut entries: Vec<(usize, f64)> =
                index.into_iter().zip(value).filter(|&(_, v)| v != 0.).map(|(r, v)| (r as usize, v)).collect();
            entries.sort_by_key(|&(r, _)| r);
            Ok(entries)
        }
    }

    fn remove(&mut self, rows: &[usize], columns: &[usize]) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)> {
        let (rm, ri) = mapping(self.rhs.len(), rows)?;
        let (cm, ci) = mapping(self.costs.len(), columns)?;
        // SAFETY: sets hold distinct, in-range, ascending indices.
        unsafe {
            if !ri.is_empty() {
                Self::check(Highs_deleteRowsBySet(self.highs.as_ptr(), ri.len() as HighsInt, ri.as_ptr()), "deleteRowsBySet")?;
            }
            if !ci.is_empty() {
                Self::check(Highs_deleteColsBySet(self.highs.as_ptr(), ci.len() as HighsInt, ci.as_ptr()), "deleteColsBySet")?;
            }
        }
        let keep = |values: &mut Vec<f64>, map: &[Option<usize>]| {
            *values = values.iter().zip(map).filter_map(|(&x, m)| m.map(|_| x)).collect();
        };
        self.rhs = self.rhs.iter().zip(&rm).filter_map(|(&x, m)| m.map(|_| x)).collect();
        self.senses = self.senses.iter().zip(&rm).filter_map(|(&x, m)| m.map(|_| x)).collect();
        keep(&mut self.costs, &cm);
        keep(&mut self.upper, &cm);
        self.solved = false;
        Ok((rm, cm))
    }

    fn remove_cuts(&mut self, rows: &[usize], columns: &[usize]) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)> {
        let solved = self.solved;
        let (duals, values, reduced_costs, row_status) = (
            self.duals.clone(),
            self.values.clone(),
            self.reduced_costs.clone(),
            self.row_status.clone(),
        );
        let (rm, cm) = self.remove(rows, columns)?;
        if solved {
            let pick = |x: Vec<f64>, map: &[Option<usize>]| -> Vec<f64> {
                x.into_iter().zip(map).filter_map(|(x, m)| m.map(|_| x)).collect()
            };
            self.duals = pick(duals, &rm);
            self.values = pick(values, &cm);
            self.reduced_costs = pick(reduced_costs, &cm);
            self.row_status = row_status.into_iter().zip(&rm).filter_map(|(x, m)| m.map(|_| x)).collect();
            self.solved = true;
        }
        Ok((rm, cm))
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
    Ok(accumulated.into_iter().filter(|(_, v)| *v != 0.).map(|(i, v)| (i as HighsInt, v)).unzip())
}

mod integer;
pub use integer::solve_integer;

#[cfg(test)]
mod limit_tests {
    use super::*;
    use crate::master::column_generation::{self as cg, NoController};

    fn time_limit(master: &HighsMaster) -> f64 {
        let mut value = 0.;
        // SAFETY: live model and valid output scalar / NUL-terminated option.
        unsafe { HighsMaster::check(Highs_getDoubleOptionValue(master.highs.as_ptr(), b"time_limit\0".as_ptr().cast(), &mut value), "get time_limit").unwrap(); }
        value
    }
    fn model() -> HighsMaster {
        let mut master = HighsMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_column(2., &[(0, 1.)]).unwrap();
        master
    }
    #[test]
    fn scoped_time_budget_restores_options_and_resumes() {
        let mut master = model();
        master.set_double("time_limit", 3600.).unwrap();
        for duration in [Duration::from_secs(60), Duration::ZERO, Duration::from_nanos(1)] {
            let outcome = master.solve_with_time_limit(duration).unwrap();
            assert_eq!(outcome, if duration == Duration::from_secs(60) { LpSolveOutcome::Optimal } else { LpSolveOutcome::TimeLimit });
            assert_eq!(time_limit(&master), 3600.);
            if outcome != LpSolveOutcome::Optimal { assert!(!master.solved); }
            assert!(master.solve());
            assert_eq!(master.objective, 2.);
        }
    }
    #[test]
    fn cg_backend_limits_do_not_report_infeasibility_or_call_pricing() {
        for iteration_stop in [false, true] {
            let mut master = HighsMaster::new().unwrap();
            for _ in 0..128 { master.add_row(RowSense::Equal, 1., &[]).unwrap(); }
            for row in 0..128 { master.add_column(1., &[(row, 1.)]).unwrap(); }
            if iteration_stop {
                // SAFETY: live handle and NUL-terminated option strings.
                unsafe {
                    HighsMaster::check(Highs_setStringOptionValue(master.highs.as_ptr(), b"presolve\0".as_ptr().cast(), b"off\0".as_ptr().cast()), "presolve").unwrap();
                    HighsMaster::check(Highs_setStringOptionValue(master.highs.as_ptr(), b"solver\0".as_ptr().cast(), b"simplex\0".as_ptr().cast()), "solver").unwrap();
                    HighsMaster::check(Highs_setIntOptionValue(master.highs.as_ptr(), b"simplex_iteration_limit\0".as_ptr().cast(), 0), "simplex_iteration_limit").unwrap();
                }
            } else { master.set_double("time_limit", 0.).unwrap(); }
            let mut options = cg::Options::default(); options.stabilize = false;
            let output = cg::optimize(&mut master, &[], |_, _| -> Result<cg::Answer> { panic!("stopped LP must not invoke pricing"); }, &mut NoController, options).unwrap();
            assert_eq!(output.status, if iteration_stop { cg::Status::IterationLimit } else { cg::Status::TimeLimit });
            assert_eq!(output.pricing_calls, 0);
            assert_eq!(output.bound, f64::NEG_INFINITY);
            assert!(!master.solved);
        }
    }
    #[test]
    fn empty_master_checks_its_required_row_activity() {
        for (rhs, expected) in [(0., LpSolveOutcome::Optimal), (1., LpSolveOutcome::PrimalInfeasible)] {
            let mut master = HighsMaster::new().unwrap();
            master.add_row(RowSense::Equal, rhs, &[]).unwrap();
            assert_eq!(master.solve_with_time_limit(Duration::from_secs(60)).unwrap(), expected);
        }
    }

    #[test]
    fn cg_stop_after_column_insertion_keeps_the_last_certificate() {
        struct StopNext;
        impl cg::Controller<HighsMaster> for StopNext {
            fn master_solved(&mut self, master: &HighsMaster, _: &[i32], _: &[usize]) -> Result<()> {
                master.set_double("time_limit", 0.)
            }
        }
        let mut master = model();
        let mut options = cg::Options::default(); options.stabilize = false;
        let output = cg::optimize(&mut master, &[], |_, duals| {
            Ok(cg::Answer {
                columns: vec![cg::Column { cost: 1., reduced_cost: 1.-duals[0], entries: vec![(0,1.)], acceptable: true, tag: 0, identity: "improving".into() }],
                minimum: 1.-duals[0], exact: true, bound_minimum: None,
            })
        }, &mut StopNext, options).unwrap();
        assert_eq!(output.status, cg::Status::TimeLimit);
        assert_eq!(output.pricing_calls, 1);
        assert_eq!(output.bound, 1.);
        assert_eq!(output.lp, 2.);
        assert_eq!(output.history.len(), 1);
        assert!(!master.solved);
    }
}
