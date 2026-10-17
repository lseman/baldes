//! Ownership-safe bridge to the HiGHS C interface (HiGHS 1.x `highs_c_api.h`).
//!
//! `HighsMaster` keeps the same bookkeeping as `ClpMaster`: columns have
//! bounds [0, 1e12], solutions are read after each optimal solve with values
//! of magnitude at most the zero tolerance snapped to zero, appending
//! dual-feasible columns keeps the current solution, and cut cleanup uses the
//! shared [`super::lp_master::cleanup_cut_rows`]. HiGHS follows its own pivot
//! rules, so trajectories differ from Clp's even on identical masters.
use super::{LpMaster, RowSense};
use crate::{incomplete, invalid, Result};
use std::ffi::{c_char, c_double, c_void, CString};
use std::ptr::NonNull;

/// `HighsInt` of the default (32-bit) HiGHS build.
type HighsInt = i32;

extern "C" {
    fn Highs_create() -> *mut c_void;
    fn Highs_destroy(highs: *mut c_void);
    fn Highs_run(highs: *mut c_void) -> HighsInt;
    fn Highs_setBoolOptionValue(highs: *mut c_void, option: *const c_char, value: HighsInt) -> HighsInt;
    fn Highs_setDoubleOptionValue(highs: *mut c_void, option: *const c_char, value: c_double) -> HighsInt;
    fn Highs_getInfinity(highs: *const c_void) -> c_double;
    fn Highs_addRow(
        highs: *mut c_void,
        lower: c_double,
        upper: c_double,
        num_new_nz: HighsInt,
        index: *const HighsInt,
        value: *const c_double,
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
    fn Highs_changeColBounds(highs: *mut c_void, col: HighsInt, lower: c_double, upper: c_double) -> HighsInt;
    fn Highs_changeCoeff(highs: *mut c_void, row: HighsInt, col: HighsInt, value: c_double) -> HighsInt;
    fn Highs_deleteRowsBySet(highs: *mut c_void, num_set_entries: HighsInt, set: *const HighsInt) -> HighsInt;
    fn Highs_deleteColsBySet(highs: *mut c_void, num_set_entries: HighsInt, set: *const HighsInt) -> HighsInt;
    fn Highs_getModelStatus(highs: *const c_void) -> HighsInt;
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
const MODEL_INFEASIBLE: HighsInt = 8;
const MODEL_UNBOUNDED_OR_INFEASIBLE: HighsInt = 9;
const MODEL_UNBOUNDED: HighsInt = 10;
const BASIS_BASIC: HighsInt = 1;
/// Column upper bound, as the Clp master (BaPCod's columns carry 1e12).
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
    pub objective: f64,
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
            objective: 0.,
            status: -1,
            solved: false,
            dual_tolerance: 1e-7,
            zero_tolerance: 0.,
        };
        master.set_bool("output_flag", false)?;
        Ok(master)
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
        if !rhs.is_finite() {
            return Err(invalid("finite row RHS required"));
        }
        let (indices, values) = entries_of(entries, self.costs.len())?;
        let infinity = self.infinity();
        let lower = if sense == RowSense::LessEqual { -infinity } else { rhs };
        let upper = if sense == RowSense::GreaterEqual { infinity } else { rhs };
        // SAFETY: arrays hold `indices.len()` entries; HiGHS copies them.
        let rc = unsafe {
            Highs_addRow(
                self.highs.as_ptr(),
                lower,
                upper,
                indices.len() as HighsInt,
                indices.as_ptr(),
                values.as_ptr(),
            )
        };
        Self::check(rc, "addRow")?;
        let id = self.rhs.len();
        self.rhs.push(rhs);
        self.senses.push(sense);
        self.solved = false;
        Ok(id)
    }

    fn add_rows(&mut self, rows: &[(RowSense, f64, &[(usize, f64)])]) -> Result<std::ops::Range<usize>> {
        let first = self.rhs.len();
        for &(sense, rhs, entries) in rows {
            self.add_row(sense, rhs, entries)?;
        }
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
        self.dual_tolerance = dual;
        Ok(())
    }

    fn solve(&mut self) -> bool {
        let h = self.highs.as_ptr();
        // SAFETY: live handle; the buffers below have the model's dimensions.
        unsafe {
            if Highs_run(h) == STATUS_ERROR {
                self.status = -1;
                self.solved = false;
                return false;
            }
            self.status = match Highs_getModelStatus(h) {
                MODEL_OPTIMAL => 0,
                MODEL_INFEASIBLE => 1,
                MODEL_UNBOUNDED | MODEL_UNBOUNDED_OR_INFEASIBLE => 2,
                _ => -1,
            };
            self.solved = self.status == 0;
            if !self.solved {
                return false;
            }
            self.objective = Highs_getObjectiveValue(h);
            let (rows, cols) = (self.rhs.len(), self.costs.len());
            self.values = vec![0.; cols];
            self.reduced_costs = vec![0.; cols];
            self.duals = vec![0.; rows];
            let mut row_values = vec![0.; rows];
            Highs_getSolution(
                h,
                self.values.as_mut_ptr(),
                self.reduced_costs.as_mut_ptr(),
                row_values.as_mut_ptr(),
                self.duals.as_mut_ptr(),
            );
            let mut col_status = vec![0; cols];
            self.row_status = vec![0; rows];
            Highs_getBasis(h, col_status.as_mut_ptr(), self.row_status.as_mut_ptr());
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
        super::lp_master::cleanup_cut_rows(self, initial_rows)
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
