//! Ownership-safe bridge to COIN-OR Clp's C interface. No original solver code
//! is called: labeling, pricing, stabilization and CG belong to this Rust port.
use super::{LpMaster, RowSense};
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
    fn Clp_getReducedCost(m: *mut Model) -> *const f64;
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
    pub reduced_costs: Vec<f64>,
    pub objective: f64,
    pub status: i32,
    pub solved: bool,
    pub dual_tolerance: f64,
    /// Primal values and duals with magnitude at most this are read as zero,
    /// as BaPCod's LpClpInterface::getSol (ELF 0x30d4b0, 0x30df00) does with 1e-7.
    pub zero_tolerance: f64,
}
impl ClpMaster {
    /// Returns a copy of the dual (shadow) prices for each row.
    pub fn duals(&self) -> Vec<f64> { self.duals.clone() }
    /// Returns a copy of the primal column values.
    pub fn values(&self) -> Vec<f64> { self.values.clone() }

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
            reduced_costs: vec![],
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
        Ok(self.add_rows(&[(sense, rhs, entries)])?.start)
    }
    /// Adds the rows in order with one Clp call, as BaPCod adds a cut round.
    pub fn add_rows(&mut self, rows: &[(RowSense, f64, &[(usize, f64)])]) -> Result<std::ops::Range<usize>> {
        let first = self.rhs.len();
        if rows.is_empty() {
            return Ok(first..first);
        }
        let (mut lower, mut upper, mut starts) = (vec![], vec![], vec![0i32]);
        let (mut indices, mut values) = (vec![], vec![]);
        for &(sense, rhs, entries) in rows {
            if !rhs.is_finite() {
                return Err(invalid("finite row RHS required"));
            }
            let (i, v) = entries_of(entries, self.costs.len())?;
            lower.push(if sense == RowSense::LessEqual { -f64::MAX } else { rhs });
            upper.push(if sense == RowSense::GreaterEqual { f64::MAX } else { rhs });
            indices.extend(i);
            values.extend(v);
            starts.push(indices.len() as i32);
        }
        // SAFETY: All arrays contain the declared row and nonzero counts; Clp copies them.
        unsafe {
            Clp_addRows(
                self.model.as_ptr(),
                rows.len() as i32,
                lower.as_ptr(),
                upper.as_ptr(),
                starts.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
            );
        }
        for &(sense, rhs, _) in rows {
            self.rhs.push(rhs);
            self.senses.push(sense);
        }
        self.solved = false;
        Ok(first..self.rhs.len())
    }
    pub fn add_column(&mut self, cost: f64, entries: &[(usize, f64)]) -> Result<usize> {
        Ok(self.add_columns(&[(cost, entries)])?.start)
    }
    /// Adds the columns in order with one Clp call: Clp copies the whole
    /// matrix on every append, so adding them one by one was quadratic.
    /// The model and bookkeeping match adding each with `add_column`.
    pub fn add_columns(&mut self, columns: &[(f64, &[(usize, f64)])]) -> Result<std::ops::Range<usize>> {
        let first = self.costs.len();
        if columns.is_empty() {
            return Ok(first..first);
        }
        let mut starts = vec![0i32];
        let mut indices = vec![];
        let mut values = vec![];
        let mut costs = Vec::with_capacity(columns.len());
        // `add_column` kept a zero value per column while `solved` held.
        let mut preserve = self.solved;
        let mut preserved = 0;
        for &(cost, entries) in columns {
            if !cost.is_finite() {
                return Err(invalid("finite column cost required"));
            }
            let (ids, vals) = entries_of(entries, self.rhs.len())?;
            let mut reduced = cost;
            for (&r, &a) in ids.iter().zip(&vals) {
                reduced -= self.duals.get(r as usize).copied().unwrap_or(0.) * a;
            }
            preserve = preserve && reduced >= -self.dual_tolerance;
            preserved += preserve as usize;
            indices.extend(ids);
            values.extend(vals);
            starts.push(indices.len() as i32);
            costs.push(cost);
        }
        let lower = vec![0.; columns.len()];
        let upper = vec![1e12; columns.len()];
        // SAFETY: Model is exclusively borrowed, Clp copies the supplied arrays.
        unsafe {
            Clp_addColumns(
                self.model.as_ptr(),
                columns.len() as i32,
                lower.as_ptr(),
                upper.as_ptr(),
                costs.as_ptr(),
                starts.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
            );
        }
        self.costs.extend(&costs);
        self.upper.extend(&upper);
        self.solved = preserve;
        self.values.resize(self.values.len() + preserved, 0.);
        Ok(first..self.costs.len())
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
                self.reduced_costs = if self.costs.is_empty() {
                    vec![]
                } else {
                    std::slice::from_raw_parts(
                        Clp_getReducedCost(self.model.as_ptr()),
                        self.costs.len(),
                    )
                    .to_vec()
                };
                let tolerance = self.zero_tolerance;
                if tolerance > 0. {
                    for x in self
                        .duals
                        .iter_mut()
                        .chain(self.values.iter_mut())
                        .chain(self.reduced_costs.iter_mut())
                    {
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
        super::lp_master::cleanup_cut_rows(self, initial_rows)
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
        let reduced_costs = self.reduced_costs.clone();
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
            self.reduced_costs = reduced_costs
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
        // Empty deletions are skipped: BaPCod issues none, and a Clp call
        // with no index still resets internal state.
        unsafe {
            if !ri.is_empty() {
                Clp_deleteRows(self.model.as_ptr(), ri.len() as i32, ri.as_ptr());
            }
            if !ci.is_empty() {
                Clp_deleteColumns(self.model.as_ptr(), ci.len() as i32, ci.as_ptr());
            }
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

// ── LpMaster trait implementation ─────────────────────────────────────

impl LpMaster for ClpMaster {
    fn add_row(
        &mut self,
        sense: RowSense,
        rhs: f64,
        entries: &[(usize, f64)],
    ) -> Result<usize> {
        Self::add_row(self, sense, rhs, entries)
    }

    fn add_rows(&mut self, rows: &[(RowSense, f64, &[(usize, f64)])]) -> Result<std::ops::Range<usize>> {
        Self::add_rows(self, rows)
    }

    fn add_column(&mut self, cost: f64, entries: &[(usize, f64)]) -> Result<usize> {
        Self::add_column(self, cost, entries)
    }

    fn add_columns(
        &mut self,
        columns: &[(f64, &[(usize, f64)])],
    ) -> Result<std::ops::Range<usize>> {
        Self::add_columns(self, columns)
    }

    fn set_coefficient(&mut self, row: usize, column: usize, value: f64) -> Result<()> {
        Self::set_coefficient(self, row, column, value)
    }

    fn set_cost(&mut self, column: usize, cost: f64) -> Result<()> {
        Self::set_cost(self, column, cost)
    }

    fn set_upper(&mut self, column: usize, upper: f64) -> Result<()> {
        Self::set_upper(self, column, upper)
    }

    fn set_tolerances(&mut self, primal: f64, dual: f64) -> Result<()> {
        Self::set_tolerances(self, primal, dual)
    }

    fn solve(&mut self) -> bool {
        Self::solve(self)
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
        Self::row_is_basic(self, row)
    }
    fn cleanup_cut_rows(&self, initial_rows: usize) -> Result<Vec<usize>> {
        Self::cleanup_cut_rows(self, initial_rows)
    }
    fn column_entries(&self, column: usize) -> Result<Vec<(usize, f64)>> {
        Self::column_entries(self, column)
    }
    fn remove(
        &mut self,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)> {
        Self::remove(self, rows, columns)
    }
    fn remove_cuts(
        &mut self,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)> {
        Self::remove_cuts(self, rows, columns)
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
