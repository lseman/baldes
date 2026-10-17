//! Ownership-safe bridge to COIN-OR Clp's C interface. No original solver code
//! is called: labeling, pricing, stabilization and CG belong to this Rust port.
use super::RowSense;
use crate::{incomplete, invalid, Result};
use std::ffi::c_void;
use std::ptr::NonNull;
type Model = c_void;
extern "C" {
    fn Clp_newModel() -> *mut Model;
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
    fn Clp_chgColumnUpper(m: *mut Model, upper: *const f64);
    fn Clp_deleteRows(m: *mut Model, n: i32, indices: *const i32);
    fn Clp_deleteColumns(m: *mut Model, n: i32, indices: *const i32);
    fn Clp_dual(m: *mut Model, values_pass: i32) -> i32;
    fn Clp_status(m: *mut Model) -> i32;
    fn Clp_objectiveValue(m: *mut Model) -> f64;
    fn Clp_dualRowSolution(m: *mut Model) -> *const f64;
    fn Clp_primalColumnSolution(m: *mut Model) -> *const f64;
    fn Clp_setDualTolerance(m: *mut Model, tolerance: f64);
    fn Clp_setPrimalTolerance(m: *mut Model, tolerance: f64);
    fn Clp_getRowStatus(m: *mut Model, row: i32) -> i32;
    fn Clp_getVectorStarts(m: *mut Model) -> *const i32;
    fn Clp_getVectorLengths(m: *mut Model) -> *const i32;
    fn Clp_getIndices(m: *mut Model) -> *const i32;
    fn Clp_getElements(m: *mut Model) -> *const f64;
}
#[derive(Debug)]
pub struct ClpMaster {
    model: NonNull<Model>,
    pub rhs: Vec<f64>,
    pub senses: Vec<RowSense>,
    pub costs: Vec<f64>,
    pub upper: Vec<f64>,
    pub duals: Vec<f64>,
    pub values: Vec<f64>,
    pub objective: f64,
    pub status: i32,
    pub solved: bool,
    pub dual_tolerance: f64,
    /// Primal values and duals with magnitude at most this are read as zero,
    /// as BaPCod's LpClpInterface::getSol (ELF 0x30d4b0, 0x30df00) does with 1e-7.
    pub zero_tolerance: f64,
}
impl ClpMaster {
    pub fn new() -> Result<Self> {
        // SAFETY: Clp returns an exclusively owned model, freed by Drop below.
        let model = NonNull::new(unsafe { Clp_newModel() })
            .ok_or_else(|| incomplete("Clp model allocation failed"))?;
        unsafe {
            Clp_setLogLevel(model.as_ptr(), 0);
            Clp_setOptimizationDirection(model.as_ptr(), 1.);
        }
        Ok(Self {
            model,
            rhs: vec![],
            senses: vec![],
            costs: vec![],
            upper: vec![],
            duals: vec![],
            values: vec![],
            objective: 0.,
            status: -1,
            solved: false,
            dual_tolerance: 1e-7,
            zero_tolerance: 0.,
        })
    }
    pub fn add_row(
        &mut self,
        sense: RowSense,
        rhs: f64,
        entries: &[(usize, f64)],
    ) -> Result<usize> {
        if !rhs.is_finite() {
            return Err(invalid("finite row RHS required"));
        }
        let (indices, values) = entries_of(entries, self.costs.len())?;
        let lower = if sense == RowSense::LessEqual {
            -f64::MAX
        } else {
            rhs
        };
        let upper = if sense == RowSense::GreaterEqual {
            f64::MAX
        } else {
            rhs
        };
        let starts = [0, indices.len() as i32];
        // SAFETY: All arrays contain the declared row and nonzero counts; Clp copies them.
        unsafe {
            Clp_addRows(
                self.model.as_ptr(),
                1,
                &lower,
                &upper,
                starts.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
            );
        }
        let id = self.rhs.len();
        self.rhs.push(rhs);
        self.senses.push(sense);
        self.solved = false;
        Ok(id)
    }
    pub fn add_column(&mut self, cost: f64, entries: &[(usize, f64)]) -> Result<usize> {
        if !cost.is_finite() {
            return Err(invalid("finite column cost required"));
        }
        let (indices, values) = entries_of(entries, self.rhs.len())?;
        let mut reduced = cost;
        for (&r, &a) in indices.iter().zip(&values) {
            reduced -= self.duals.get(r as usize).copied().unwrap_or(0.) * a;
        }
        let preserve = self.solved && reduced >= -self.dual_tolerance;
        let lower = 0.;
        let upper = 1e12;
        let starts = [0, indices.len() as i32];
        // SAFETY: Model is exclusively borrowed, Clp copies the supplied arrays.
        unsafe {
            Clp_addColumns(
                self.model.as_ptr(),
                1,
                &lower,
                &upper,
                &cost,
                starts.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
            );
        }
        let id = self.costs.len();
        self.costs.push(cost);
        self.upper.push(upper);
        self.solved = preserve;
        if preserve {
            self.values.push(0.);
        }
        Ok(id)
    }
    pub fn set_coefficient(&mut self, row: usize, column: usize, value: f64) -> Result<()> {
        if row >= self.rhs.len() || column >= self.costs.len() || !value.is_finite() {
            return Err(invalid("master coefficient index/value"));
        }
        unsafe {
            Clp_modifyCoefficient(self.model.as_ptr(), row as i32, column as i32, value, true);
        }
        self.solved = false;
        Ok(())
    }
    pub fn set_cost(&mut self, column: usize, cost: f64) -> Result<()> {
        if column >= self.costs.len() || !cost.is_finite() {
            return Err(invalid("master cost index/value"));
        }
        self.costs[column] = cost;
        unsafe {
            Clp_chgObjCoefficients(self.model.as_ptr(), self.costs.as_ptr());
        }
        self.solved = false;
        Ok(())
    }
    pub fn set_upper(&mut self, column: usize, upper: f64) -> Result<()> {
        if column >= self.upper.len() || !upper.is_finite() || upper < 0. {
            return Err(invalid("master upper bound"));
        }
        self.upper[column] = upper;
        unsafe {
            Clp_chgColumnUpper(self.model.as_ptr(), self.upper.as_ptr());
        }
        self.solved = false;
        Ok(())
    }
    pub fn set_tolerances(&mut self, primal: f64, dual: f64) -> Result<()> {
        if !primal.is_finite() || !dual.is_finite() || primal <= 0. || dual <= 0. {
            return Err(invalid("positive finite Clp tolerances required"));
        }
        unsafe {
            Clp_setPrimalTolerance(self.model.as_ptr(), primal);
            Clp_setDualTolerance(self.model.as_ptr(), dual);
        }
        self.dual_tolerance = dual;
        Ok(())
    }
    pub fn solve(&mut self) -> bool {
        // SAFETY: All returned arrays are owned by this model. Copy before further mutation.
        unsafe {
            Clp_dual(self.model.as_ptr(), 0);
            self.status = Clp_status(self.model.as_ptr());
            self.solved = self.status == 0;
            if self.solved {
                self.objective = Clp_objectiveValue(self.model.as_ptr());
                self.duals = if self.rhs.is_empty() {
                    vec![]
                } else {
                    std::slice::from_raw_parts(
                        Clp_dualRowSolution(self.model.as_ptr()),
                        self.rhs.len(),
                    )
                    .to_vec()
                };
                self.values = if self.costs.is_empty() {
                    vec![]
                } else {
                    std::slice::from_raw_parts(
                        Clp_primalColumnSolution(self.model.as_ptr()),
                        self.costs.len(),
                    )
                    .to_vec()
                };
                let tolerance = self.zero_tolerance;
                if tolerance > 0. {
                    for x in self.duals.iter_mut().chain(self.values.iter_mut()) {
                        if !(*x > tolerance || *x < -tolerance) {
                            *x = 0.;
                        }
                    }
                }
            }
        }
        self.solved
    }
    pub fn row_is_basic(&self, row: usize) -> Result<bool> {
        if row >= self.rhs.len() {
            return Err(invalid("master row index"));
        }
        Ok(unsafe { Clp_getRowStatus(self.model.as_ptr(), row as i32) } == 1)
    }
    /// BaPCod cleanupRestrictedMastCuts (ELF 0xe1010), default threshold 1
    /// and retention ratio 0.66. The normalization divides by the SIGNED RHS
    /// when |rhs| > 1: a slack packing cut with rhs < -1 is retained.
    pub fn cleanup_cut_rows(&self, initial_rows: usize) -> Result<Vec<usize>> {
        if initial_rows > self.rhs.len() || !self.solved {
            return Err(invalid("cut cleanup requires a solved master and valid base rows"));
        }
        let count = self.rhs.len() - initial_rows;
        if count <= 1 { return Ok(vec![]); }
        let mut activity = vec![0.; self.rhs.len()];
        for (column, &value) in self.values.iter().enumerate() {
            if value == 0. { continue; }
            for (row, coefficient) in self.column_entries(column)? {
                activity[row] += coefficient * value;
            }
        }
        let mut kept = 0;
        let mut candidates = vec![];
        for row in initial_rows..self.rhs.len() {
            if !self.row_is_basic(row)? { kept += 1; continue; }
            let rhs = self.rhs[row];
            let mut slack = match self.senses[row] {
                RowSense::LessEqual => rhs - activity[row],
                RowSense::GreaterEqual => activity[row] - rhs,
                RowSense::Equal => 0.,
            };
            if rhs.abs() > 1. { slack /= rhs; }
            // VarConstr::violation(Double) snaps values in [-precision,precision].
            if slack.abs() <= 1e-6 { slack = 0.; }
            if slack < 0. { kept += 1; } else { candidates.push((row, slack)); }
        }
        let target = 1.min((count as f32 * 0.66_f32) as usize);
        if target > kept {
            // SortConstraintsByViolation: smaller normalized slack first, then
            // creation order for Double comparisons equal within tolerance.
            candidates.sort_by(|a,b| {
                let tolerance = 1e-6 + a.1.abs().max(b.1.abs()) * 1e-10;
                if a.1 < b.1 - tolerance { std::cmp::Ordering::Less }
                else if b.1 < a.1 - tolerance { std::cmp::Ordering::Greater }
                else { a.0.cmp(&b.0) }
            });
        }
        let mut rows: Vec<_> = candidates.into_iter().skip(target.saturating_sub(kept))
            .map(|(row,_)| row).collect();
        rows.sort_unstable();
        Ok(rows)
    }
    /// Sparse column of the current matrix, sorted by row.
    pub fn column_entries(&self, column: usize) -> Result<Vec<(usize, f64)>> {
        if column >= self.costs.len() {
            return Err(invalid("master column index"));
        }
        let m = self.model.as_ptr();
        let mut entries = unsafe {
            let start = *Clp_getVectorStarts(m).add(column) as usize;
            let length = *Clp_getVectorLengths(m).add(column) as usize;
            let indices = std::slice::from_raw_parts(Clp_getIndices(m).add(start), length);
            let values = std::slice::from_raw_parts(Clp_getElements(m).add(start), length);
            indices
                .iter()
                .zip(values)
                .map(|(&r, &v)| (r as usize, v))
                .collect::<Vec<_>>()
        };
        entries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(entries)
    }
    /// ClpMaster::remove_cuts: basic cut rows and their nonbasic artificials.
    /// The current solution stays available, compacted to the kept indices.
    pub fn remove_cuts(
        &mut self,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)> {
        let solved = self.solved;
        let duals = self.duals.clone();
        let values = self.values.clone();
        let (rm, cm) = self.remove(rows, columns)?;
        if solved {
            self.duals = duals
                .into_iter()
                .zip(&rm)
                .filter_map(|(x, m)| m.map(|_| x))
                .collect();
            self.values = values
                .into_iter()
                .zip(&cm)
                .filter_map(|(x, m)| m.map(|_| x))
                .collect();
            self.solved = true;
        }
        Ok((rm, cm))
    }
    pub fn remove(
        &mut self,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)> {
        fn mapping(size: usize, removed: &[usize]) -> Result<(Vec<Option<usize>>, Vec<i32>)> {
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
                        let id = next;
                        next += 1;
                        Some(id)
                    }
                })
                .collect();
            Ok((map, ids.into_iter().map(|i| i as i32).collect()))
        }
        let (rm, ri) = mapping(self.rhs.len(), rows)?;
        let (cm, ci) = mapping(self.costs.len(), columns)?;
        unsafe {
            Clp_deleteRows(self.model.as_ptr(), ri.len() as i32, ri.as_ptr());
            Clp_deleteColumns(self.model.as_ptr(), ci.len() as i32, ci.as_ptr());
        }
        self.rhs = self
            .rhs
            .iter()
            .zip(&rm)
            .filter_map(|(&x, m)| m.map(|_| x))
            .collect();
        self.senses = self
            .senses
            .iter()
            .zip(&rm)
            .filter_map(|(&x, m)| m.map(|_| x))
            .collect();
        self.costs = self
            .costs
            .iter()
            .zip(&cm)
            .filter_map(|(&x, m)| m.map(|_| x))
            .collect();
        self.upper = self
            .upper
            .iter()
            .zip(&cm)
            .filter_map(|(&x, m)| m.map(|_| x))
            .collect();
        self.solved = false;
        Ok((rm, cm))
    }
}
impl Drop for ClpMaster {
    fn drop(&mut self) {
        unsafe {
            Clp_deleteModel(self.model.as_ptr());
        }
    }
}
fn entries_of(entries: &[(usize, f64)], size: usize) -> Result<(Vec<i32>, Vec<f64>)> {
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
