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
    fn baldes_clp_dual_limited(m: *mut Model, iterations: i32, seconds: f64) -> i32;
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
    fn Clp_dual(m: *mut Model, values_pass: i32) -> i32;
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
    /// Constant objective contribution, retained by forks and integer transfer.
    pub objective_offset: f64,
    pub status: i32,
    pub solved: bool,
    pub dual_tolerance: f64,
    pub simplex_iterations: usize,
    /// Primal values and duals with magnitude at most this are read as zero,
    /// using a default tolerance of 1e-7.
    pub zero_tolerance: f64,
}
impl ClpMaster {
    /// Reuse a preceding solve when deleting its trailing rows/columns gives
    /// this exact ordered matrix and costs. Bounds remain those of this model.
    /// Returns false without changing either model when they do not match.
    pub(crate) fn continue_from(&mut self, previous: &Self) -> Result<bool> {
        let rows = self.rhs.len();
        let columns = self.costs.len();
        if previous.rhs.len() < rows
            || previous.costs.len() < columns
            || previous.costs[..columns] != self.costs
            || previous.senses[..rows] != self.senses
            || previous.objective_offset != self.objective_offset
        {
            return Ok(false);
        }
        for column in 0..columns {
            let retained: Vec<_> = previous
                .column_entries(column)?
                .into_iter()
                .filter(|&(row, _)| row < rows)
                .collect();
            if retained != self.column_entries(column)? {
                return Ok(false);
            }
        }
        let mut continued = previous.try_clone()?;
        // Native node restoration deletes branch artificials before rows.
        let removed_columns: Vec<_> = (columns..previous.costs.len()).collect();
        let removed_rows: Vec<_> = (rows..previous.rhs.len()).collect();
        continued.remove(&[], &removed_columns)?;
        continued.remove(&removed_rows, &[])?;
        continued.duals.truncate(rows);
        continued.values.truncate(columns);
        continued.reduced_costs.truncate(columns);
        continued.set_rhs(&self.rhs)?;
        for column in 0..columns {
            if continued.upper[column] != self.upper[column] {
                continued.set_upper(column, self.upper[column])?;
            }
        }
        continued.zero_tolerance = self.zero_tolerance;
        continued.set_tolerances(1e-7, self.dual_tolerance)?;
        *self = continued;
        Ok(true)
    }
    /// Fork the model through Clp's deep-copy operation, retaining its basis,
    /// primal/dual solution, scaling and simplex state. The fork owns all data.
    pub fn try_clone(&self) -> Result<Self> {
        // SAFETY: source stays alive and is not mutated; the C++ bridge returns
        // an independently owned Clp wrapper, released by this type's Drop.
        let model = NonNull::new(unsafe { baldes_clp_clone(self.model.as_ptr()) })
            .ok_or_else(|| incomplete("Clp model copy failed"))?;
        unsafe { Clp_setLogLevel(model.as_ptr(), 0) };
        Ok(Self { model, rhs:self.rhs.clone(), senses:self.senses.clone(), costs:self.costs.clone(),
            upper:self.upper.clone(), duals:self.duals.clone(), values:self.values.clone(),
            reduced_costs:self.reduced_costs.clone(), objective:self.objective, objective_offset:self.objective_offset, status:self.status,
            solved:self.solved, dual_tolerance:self.dual_tolerance, zero_tolerance:self.zero_tolerance,
            simplex_iterations:0 })
    }
    /// Replace all current row bounds, preserving row identities and the basis.
    pub fn set_rhs(&mut self, rhs: &[f64]) -> Result<()> {
        if rhs.len() != self.rhs.len() || rhs.iter().any(|x| !x.is_finite()) {
            return Err(invalid("Clp RHS dimensions or values"));
        }
        let lower: Vec<_> = rhs.iter().zip(&self.senses).map(|(&b,&s)| if s == RowSense::LessEqual { -f64::MAX } else { b }).collect();
        let upper: Vec<_> = rhs.iter().zip(&self.senses).map(|(&b,&s)| if s == RowSense::GreaterEqual { f64::MAX } else { b }).collect();
        // SAFETY: both buffers contain one bound for every live model row;
        // Clp copies them during these calls and the model is exclusively owned.
        unsafe {
            Clp_chgRowLower(self.model.as_ptr(), lower.as_ptr());
            Clp_chgRowUpper(self.model.as_ptr(), upper.as_ptr());
        }
        self.rhs.copy_from_slice(rhs);
        self.solved = false;
        Ok(())
    }
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
            objective_offset: 0.,
            status: -1,
            solved: false,
            dual_tolerance: 1e-7,
            simplex_iterations: 0,
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
    /// Adds the rows in order with one Clp call.
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
        // SAFETY: The model is exclusively borrowed and owned by this handle.
        unsafe { Clp_dual(self.model.as_ptr(), 0); }
        self.read_solution()
    }
    /// Cooperatively limit one LP solve by elapsed wall time and simplex pivots.
    /// Existing stricter backend limits remain effective and are restored
    /// exactly. Only `Optimal` marks the cached solution as solved; other
    /// outcomes do not validate the cached primal/dual arrays for this call.
    pub fn solve_limited(&mut self, iterations: usize, time_limit: Duration) -> Result<ClpSolveOutcome> {
        let iterations = i32::try_from(iterations).map_err(|_| invalid("Clp iteration limit"))?;
        let started = Instant::now();
        if iterations == 0 || time_limit.is_zero() {
            self.status = 3;
            self.solved = false;
            return Ok(ClpSolveOutcome::Stopped);
        }
        // SAFETY: The native scope guard restores this owned model's limits
        // before returning, including on caught C++ exceptions.
        let failed = unsafe { baldes_clp_dual_limited(self.model.as_ptr(), iterations, time_limit.as_secs_f64()) } != 0;
        if failed {
            self.status = 4;
            self.solved = false;
            return Ok(ClpSolveOutcome::Failed);
        }
        self.read_solution();
        if started.elapsed() >= time_limit {
            self.status = 3;
            self.solved = false;
        }
        Ok(match self.status {
            0 => ClpSolveOutcome::Optimal,
            1 => ClpSolveOutcome::PrimalInfeasible,
            2 => ClpSolveOutcome::DualInfeasible,
            3 => ClpSolveOutcome::Stopped,
            _ => ClpSolveOutcome::Failed,
        })
    }
    fn read_solution(&mut self) -> bool {
        // SAFETY: All returned arrays are owned by this model. Copy before further mutation.
        unsafe {
            self.simplex_iterations += Clp_numberIterations(self.model.as_ptr()).max(0) as usize;
            self.status = Clp_status(self.model.as_ptr());
            self.solved = self.status == 0;
            if self.solved {
                self.objective = Clp_objectiveValue(self.model.as_ptr()) + self.objective_offset;
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
    /// Cut cleanup with default threshold 1
    /// and retention ratio 0.66. The normalization divides by the SIGNED RHS
    /// when |rhs| > 1: a slack packing cut with rhs < -1 is retained.
    pub fn cleanup_cut_rows(&self, initial_rows: usize) -> Result<Vec<usize>> {
        crate::master::lp_master::cleanup_cut_rows(self, initial_rows)
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
        // Skip empty deletions: a Clp call
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

    fn solve_with_time_limit(&mut self, remaining: Duration) -> Result<LpSolveOutcome> {
        let started = Instant::now();
        Ok(match self.solve_limited(i32::MAX as usize, remaining)? {
            ClpSolveOutcome::Optimal => LpSolveOutcome::Optimal,
            ClpSolveOutcome::PrimalInfeasible => LpSolveOutcome::PrimalInfeasible,
            ClpSolveOutcome::DualInfeasible => LpSolveOutcome::DualInfeasible,
            ClpSolveOutcome::Failed => LpSolveOutcome::Failed,
            ClpSolveOutcome::Stopped => {
                // SAFETY: live, exclusively borrowed model. Clp secondary
                // status 9 identifies a time stop rather than an iteration stop.
                if remaining.is_zero() || started.elapsed() >= remaining
                    || unsafe { Clp_secondaryStatus(self.model.as_ptr()) } == 9
                { LpSolveOutcome::TimeLimit } else { LpSolveOutcome::IterationLimit }
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

#[cfg(test)]
mod continuation_tests {
    use super::*;

    #[test]
    fn clone_preserves_sparse_layout_through_column_insertion() {
        fn layout(master: &ClpMaster) -> (Vec<i32>, Vec<i32>) {
            // SAFETY: live Clp models expose n+1 starts and n column lengths;
            // both buffers are copied before the model can be mutated.
            unsafe {
                let model = master.model.as_ptr();
                (
                    std::slice::from_raw_parts(Clp_getVectorStarts(model), master.costs.len() + 1)
                        .to_vec(),
                    std::slice::from_raw_parts(Clp_getVectorLengths(model), master.costs.len())
                        .to_vec(),
                )
            }
        }
        let mut master = ClpMaster::new().unwrap();
        for _ in 0..3 {
            master.add_row(RowSense::Equal, 1., &[]).unwrap();
        }
        for cost in [1., 2., 3.] {
            master
                .add_column(cost, &[(0, 1.), (1, 1.), (2, 1.)])
                .unwrap();
        }
        master.remove(&[], &[1]).unwrap();
        let before = layout(&master);
        assert!(before.0.last().unwrap() > &before.1.iter().sum());
        let mut copied = master.try_clone().unwrap();
        assert_eq!(layout(&copied), before);
        for model in [&mut master, &mut copied] {
            model.add_column(4., &[(0, 1.)]).unwrap();
        }
        assert_eq!(layout(&copied), layout(&master));
        assert!(master.solve() && copied.solve());
        assert_eq!(master.objective, 1.);
        assert_eq!(copied.values(), master.values());
        copied.set_cost(0, 5.).unwrap();
        assert!(copied.solve());
        assert_eq!(copied.objective, 3.);
        assert_eq!(master.costs[0], 1.);
        assert_eq!(master.objective, 1.);
    }

    fn base() -> ClpMaster {
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_column(2., &[(0, 1.)]).unwrap();
        assert!(master.solve());
        master
    }

    fn limits(master: &ClpMaster) -> [f64; 3] {
        let mut values = [0.; 3];
        // SAFETY: The native getter writes exactly three values for this model.
        unsafe { baldes_clp_solve_limits(master.model.as_ptr(), values.as_mut_ptr()); }
        values
    }

    #[test]
    fn limited_solve_preserves_absolute_deadlines_and_can_resume() {
        let mut master = base();
        // SAFETY: This exclusively owned model accepts a relative CPU limit.
        unsafe { Clp_setMaximumSeconds(master.model.as_ptr(), 60.); }
        let before = limits(&master);
        for (iterations, time) in [(0, Duration::from_secs(60)), (1000, Duration::ZERO), (1000, Duration::from_nanos(1))] {
            assert_eq!(master.solve_limited(iterations, time).unwrap(), ClpSolveOutcome::Stopped);
            assert!(!master.solved);
            assert_eq!(master.status, 3);
            assert_eq!(limits(&master), before);
            assert!(master.solve());
            assert_eq!(master.objective, 2.);
        }
        let status = master.status;
        assert!(master.solve_limited(usize::MAX, Duration::from_secs(60)).is_err());
        assert_eq!(master.status, status);
        assert!(master.solved);
        assert_eq!(limits(&master), before);
    }

    #[test]
    fn iteration_stopped_model_is_not_an_infeasibility_certificate() {
        let mut master = ClpMaster::new().unwrap();
        for _ in 0..128 { master.add_row(RowSense::Equal, 1., &[]).unwrap(); }
        for row in 0..128 { master.add_column(1., &[(row, 1.)]).unwrap(); }
        let before = limits(&master);
        assert_eq!(master.solve_limited(1, Duration::from_secs(60)).unwrap(), ClpSolveOutcome::Stopped);
        assert_eq!(master.status, 3);
        assert!(!master.solved);
        assert_eq!(limits(&master), before);
        assert!(master.solve());
        assert_eq!(master.objective, 128.);
        assert!(master.values.iter().all(|&value| value == 1.));
    }

    #[test]
    fn cg_time_stop_preserves_bound_and_does_not_price() {
        use crate::master::column_generation::{self as cg, NoController};
        let mut master = ClpMaster::new().unwrap();
        for _ in 0..128 { master.add_row(RowSense::Equal, 1., &[]).unwrap(); }
        for row in 0..128 { master.add_column(1., &[(row, 1.)]).unwrap(); }
        // SAFETY: expire the CPU budget of this exclusively owned model.
        unsafe { Clp_setMaximumSeconds(master.model.as_ptr(), 0.); }
        let before = limits(&master);
        let mut options = cg::Options::default(); options.stabilize = false;
        let output = cg::optimize(&mut master, &[], |_, _| -> Result<cg::Answer> { panic!("stopped LP must not price"); }, &mut NoController, options).unwrap();
        assert_eq!(output.status, cg::Status::TimeLimit);
        assert_eq!(output.bound, f64::NEG_INFINITY);
        assert_eq!(output.pricing_calls, 0);
        assert!(!master.solved);
        assert_eq!(limits(&master), before);
        // SAFETY: remove the explicit test budget before resuming the model.
        unsafe { Clp_setMaximumSeconds(master.model.as_ptr(), -1.); }
        assert!(master.solve());
        assert_eq!(master.objective, 128.);
    }

    #[test]
    fn limited_solve_distinguishes_backend_infeasibility_statuses() {
        let mut infeasible = ClpMaster::new().unwrap();
        infeasible.add_row(RowSense::Equal, 1., &[]).unwrap();
        assert_eq!(infeasible.solve_limited(1000, Duration::from_secs(60)).unwrap(), ClpSolveOutcome::PrimalInfeasible);
        let mut dual_infeasible = ClpMaster::new().unwrap();
        let column = dual_infeasible.add_column(-1., &[]).unwrap();
        // Columns default to the finite upper bound 1e12. Explicitly use
        // Clp's infinity convention for this backend-status fixture.
        dual_infeasible.set_upper(column, f64::MAX).unwrap();
        assert_eq!(dual_infeasible.solve_limited(1000, Duration::from_secs(60)).unwrap(), ClpSolveOutcome::DualInfeasible);
        let mut optimal = base();
        assert_eq!(optimal.solve_limited(1000, Duration::from_secs(60)).unwrap(), ClpSolveOutcome::Optimal);
        assert!(optimal.solved);
    }

    #[test]
    fn continuation_restores_target_bounds_and_keeps_the_source_owned() {
        let mut previous = base();
        previous
            .add_row(RowSense::GreaterEqual, 0., &[(0, 1.)])
            .unwrap();
        previous.add_column(10., &[(1, 1.)]).unwrap();
        assert!(previous.solve());
        let previous_values = previous.values.clone();
        let mut target = base();
        target.set_rhs(&[2.]).unwrap();
        target.set_upper(0, 3.).unwrap();
        assert!(target.continue_from(&previous).unwrap());
        assert_eq!(target.rhs, [2.]);
        assert_eq!(target.upper, [3.]);
        assert_eq!(target.values.len(), 1);
        assert_eq!(target.duals.len(), 1);
        assert!(!target.solved);
        assert!(target.solve());
        assert_eq!(target.objective, 4.);
        assert_eq!(previous.rhs, [1., 0.]);
        assert_eq!(previous.costs, [2., 10.]);
        assert_eq!(previous.values, previous_values);
        assert!(previous.solved);
    }

    #[test]
    fn continuation_rejects_a_different_model_without_mutation() {
        let previous = base();
        for change in 0..4 {
            let mut target = base();
            match change {
                0 => target.set_cost(0, 3.).unwrap(),
                1 => target.set_coefficient(0, 0, 2.).unwrap(),
                2 => {
                    target.add_column(3., &[(0, 1.)]).unwrap();
                }
                _ => {
                    target.add_row(RowSense::LessEqual, 5., &[(0, 1.)]).unwrap();
                }
            }
            let before = (
                target.rhs.clone(),
                target.costs.clone(),
                target.column_entries(0).unwrap(),
                target.solved,
                target.values.clone(),
            );
            assert!(!target.continue_from(&previous).unwrap());
            assert_eq!(
                before,
                (
                    target.rhs.clone(),
                    target.costs.clone(),
                    target.column_entries(0).unwrap(),
                    target.solved,
                    target.values.clone()
                )
            );
        }
        assert_eq!(previous.objective, 2.);
        assert!(previous.solved);
    }
}
