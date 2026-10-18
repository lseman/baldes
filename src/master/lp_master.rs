//! Abstract LP master interface for column generation backends.
//!
//! Both Clp and HiGHS backends implement this trait so that the column
//! generation loop, root driver, and Python bindings are solver-agnostic.
//! Accessors borrow the backend's solution and model vectors: the loop reads
//! them per column and per row, so they must not copy.
use super::RowSense;
use crate::{invalid, Result};

/// The operations the column generation algorithm needs from an LP backend.
pub trait LpMaster: Sized + 'static {
    /// Add a constraint row to the master problem.
    fn add_row(&mut self, sense: RowSense, rhs: f64, entries: &[(usize, f64)]) -> Result<usize>;

    /// Add several constraint rows in one call, in order. Returns their
    /// index range. Must leave the model as adding them one by one would.
    fn add_rows(&mut self, rows: &[(RowSense, f64, &[(usize, f64)])]) -> Result<std::ops::Range<usize>>;

    /// Add a single variable (column), with lower bound zero.
    fn add_column(&mut self, cost: f64, entries: &[(usize, f64)]) -> Result<usize>;

    /// Add several columns in one call, in order. Returns their index range.
    /// Must leave the model and bookkeeping as adding them one by one would.
    fn add_columns(&mut self, columns: &[(f64, &[(usize, f64)])]) -> Result<std::ops::Range<usize>>;

    /// Modify a single matrix coefficient.
    fn set_coefficient(&mut self, row: usize, column: usize, value: f64) -> Result<()>;

    /// Change the objective cost of an existing column.
    fn set_cost(&mut self, column: usize, cost: f64) -> Result<()>;

    /// Change the upper bound of an existing column.
    fn set_upper(&mut self, column: usize, upper: f64) -> Result<()>;

    /// Set primal and dual feasibility tolerances.
    fn set_tolerances(&mut self, primal: f64, dual: f64) -> Result<()>;

    /// Solve the current LP. Returns `true` on an optimal solution.
    fn solve(&mut self) -> bool;

    /// Row duals of the last optimal solve.
    fn duals(&self) -> &[f64];
    /// Column values of the last optimal solve.
    fn values(&self) -> &[f64];
    /// Column reduced costs of the last optimal solve.
    fn reduced_costs(&self) -> &[f64];
    /// Objective value of the last solve.
    fn objective(&self) -> f64;
    /// Solver status: 0 optimal, 1 primal infeasible, 2 dual infeasible.
    fn status(&self) -> i32;
    /// Whether the current solution vectors belong to the current model.
    fn solved(&self) -> bool;

    /// Right-hand side of each row.
    fn rhs(&self) -> &[f64];
    /// Sense of each row.
    fn senses(&self) -> &[RowSense];
    /// Objective cost of each column.
    fn costs(&self) -> &[f64];
    /// Upper bound of each column.
    fn upper(&self) -> &[f64];

    /// Whether a row's slack is basic in the current basis. Cut cleanup
    /// depends on it, so every backend must report the real basis.
    fn row_is_basic(&self, row: usize) -> Result<bool>;

    /// Cut cleanup: the cut rows (from `initial_rows`
    /// on) to remove, ascending. Backends normally forward to the shared
    /// [`cleanup_cut_rows`].
    fn cleanup_cut_rows(&self, initial_rows: usize) -> Result<Vec<usize>>;

    /// Sparse column of the current matrix, sorted by row.
    fn column_entries(&self, column: usize) -> Result<Vec<(usize, f64)>>;

    /// Remove rows and columns, returning old-to-new index maps. The
    /// current solution is dropped.
    fn remove(
        &mut self,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)>;

    /// Remove rows and columns while keeping the current solution, compacted
    /// to the surviving indices (cut cleanup, column cleanup).
    fn remove_cuts(
        &mut self,
        rows: &[usize],
        columns: &[usize],
    ) -> Result<(Vec<Option<usize>>, Vec<Option<usize>>)>;
}

/// Cut cleanup with default threshold 1 and
/// retention ratio 0.66, on any backend. The normalization divides by the
/// SIGNED RHS when |rhs| > 1: a slack packing cut with rhs < -1 is retained.
pub fn cleanup_cut_rows<M: LpMaster>(master: &M, initial_rows: usize) -> Result<Vec<usize>> {
    let (rhs, senses, values) = (master.rhs(), master.senses(), master.values());
    if initial_rows > rhs.len() || !master.solved() {
        return Err(invalid("cut cleanup requires a solved master and valid base rows"));
    }
    let count = rhs.len() - initial_rows;
    if count <= 1 {
        return Ok(vec![]);
    }
    let mut activity = vec![0.; rhs.len()];
    for (column, &value) in values.iter().enumerate() {
        if value == 0. {
            continue;
        }
        for (row, coefficient) in master.column_entries(column)? {
            activity[row] += coefficient * value;
        }
    }
    let mut kept = 0;
    let mut candidates = vec![];
    for row in initial_rows..rhs.len() {
        if !master.row_is_basic(row)? {
            kept += 1;
            continue;
        }
        let b = rhs[row];
        let mut slack = match senses[row] {
            RowSense::LessEqual => b - activity[row],
            RowSense::GreaterEqual => activity[row] - b,
            RowSense::Equal => 0.,
        };
        if b.abs() > 1. {
            slack /= b;
        }
        // VarConstr::violation(Double) snaps values in [-precision,precision].
        if slack.abs() <= 1e-6 {
            slack = 0.;
        }
        if slack < 0. {
            kept += 1;
        } else {
            candidates.push((row, slack));
        }
    }
    let target = 1.min((count as f32 * 0.66_f32) as usize);
    if target > kept {
        // SortConstraintsByViolation: smaller normalized slack first, then
        // creation order for Double comparisons equal within tolerance.
        candidates.sort_by(|a, b| {
            let tolerance = 1e-6 + a.1.abs().max(b.1.abs()) * 1e-10;
            if a.1 < b.1 - tolerance {
                std::cmp::Ordering::Less
            } else if b.1 < a.1 - tolerance {
                std::cmp::Ordering::Greater
            } else {
                a.0.cmp(&b.0)
            }
        });
    }
    let mut rows: Vec<_> = candidates.into_iter().skip(target.saturating_sub(kept)).map(|(row, _)| row).collect();
    rows.sort_unstable();
    Ok(rows)
}
