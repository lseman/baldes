use super::*;

/// The controller owns incumbent tracking and the pricing route universe.
pub trait Controller<M> {
    /// Optional original-model cap, restored whenever this column is loaded.
    /// The controller must justify the cap for every omitted/future column
    /// with the same tag. Numerical/certified bounds must account for it.
    fn column_upper_bound(&self, _tag: i32) -> Option<f64> {
        None
    }
    /// Optional Lagrangian estimate for the most recent batch at these prices.
    /// Only exact batches publish a numerical bound. Multiple pricing domains
    /// can contribute different multiplicities.
    fn pricing_bound(&self, _master: &M, _duals: &[f64]) -> Result<Option<f64>> {
        Ok(None)
    }
    /// Validate a conservative bound when an exact pass may close pricing or
    /// the incumbent gap. None/negative infinity means unknown, never fallback
    /// to a numerical estimate in certified mode.
    /// This may certify the original integer domain; it is not an LP pricing
    /// convergence estimate, including when inherited through a warm start.
    fn certified_pricing_bound(&self, _master: &M, _duals: &[f64]) -> Result<Option<f64>> {
        Ok(None)
    }
    /// Optional aggregate subgradient for the most recent pricing batch.
    fn pricing_subgradient(&self, _rhs: &[f64]) -> Option<Vec<f64>> {
        None
    }
    /// Logical base/branch rows can follow inherited cut rows. Cleanup must
    /// never delete these rows, even when their slacks are basic.
    fn protected_rows(&self) -> Vec<usize> {
        vec![]
    }
    fn unstabilized_rows(&self) -> Vec<usize> {
        vec![]
    }
    fn master_solved(&mut self, _master: &M, _tags: &[i32], _artificials: &[usize]) -> Result<()> {
        Ok(())
    }
    fn incumbent(&self) -> f64 {
        f64::INFINITY
    }
    /// Optional original-model feasible columns after exact pricing leaves
    /// positive artificials. These columns do not certify a bound or prove
    /// infeasibility. The controller must validate their original constraints.
    /// In certified mode, recovery from restricted pricing still requires an
    /// original-domain publication certificate; an oracle estimate is insufficient.
    fn restore_feasible_columns(&mut self, _duals: &[f64]) -> Result<Vec<Column>> {
        Ok(vec![])
    }
    fn enumerated_universe(&mut self, _duals: &[f64], _bound: f64) -> Result<Option<Vec<Column>>> {
        Ok(None)
    }
    /// Update cached, possibly unloaded columns after a cut-row change.
    fn refresh_generated_columns(&mut self, _pool: &mut [Column]) -> Result<()> {
        Ok(())
    }
    /// Synchronize cached columns using the refreshed loaded metadata where
    /// the controller can establish identity. The default preserves callbacks
    /// that need to update every cached column themselves.
    fn synchronize_generated_columns(
        &mut self,
        pool: &mut [Column],
        _loaded: &[Option<Column>],
    ) -> Result<()> {
        self.refresh_generated_columns(pool)
    }
    /// Refresh the generated column pool after successful enumeration. A
    /// representative universe need not contain every admissible cached path.
    fn regenerate_enumerated_columns(
        &mut self,
        pool: &[Column],
        _duals: &[f64],
        _bound: f64,
        universe: &[Column],
    ) -> Result<Vec<Column>> {
        let allowed: HashMap<_, _> = universe.iter().map(|c| (c.signature(), c)).collect();
        Ok(pool
            .iter()
            .filter_map(|column| {
                let current = allowed.get(&column.signature())?;
                let mut column = column.clone();
                column.entries = current.entries.clone();
                Some(column)
            })
            .collect())
    }
    fn remap_columns(&mut self, _map: &[Option<usize>]) -> Result<()> {
        Ok(())
    }
    /// indices_remapped after a row (cut) or column deletion.
    fn remap_rows(&mut self, _rows: &[Option<usize>]) -> Result<()> {
        Ok(())
    }
    /// Retain a completed feasible pricing round before cut mutations.
    fn save_pricing_checkpoint(&mut self, _master: &M, _output: &Output) -> Result<()> {
        Ok(())
    }
    /// Transfer the completed checkpoint when its owner can retain it without
    /// copying the generated pool again. Existing borrowed callbacks still run.
    fn store_pricing_checkpoint(&mut self, master: &M, output: Output) -> Result<()> {
        self.save_pricing_checkpoint(master, &output)
    }
    /// Whether a separator is installed.
    fn separates(&self) -> bool {
        false
    }
    /// Adds a cut family once pricing has converged on an exact phase;
    /// returns how many rows were added (or changed). Zero ends the rounds.
    fn separate(&mut self, _master: &mut M, _tags: &[i32]) -> Result<usize> {
        Ok(0)
    }
    /// Master columns to delete after `separate`, ascending: columns that a
    /// strengthened pricing relaxation no longer allows
    /// (removeColumnsNotSatisfyingColGenSpRelaxation).
    fn take_removed_columns(&mut self) -> Vec<usize> {
        vec![]
    }
}

pub struct NoController;
impl<M> Controller<M> for NoController {}

pub(super) fn apply_column_cap<M: LpMaster, C: Controller<M>>(
    master: &mut M,
    controller: &C,
    column: usize,
    tag: i32,
) -> Result<()> {
    if let Some(upper) = controller.column_upper_bound(tag) {
        if !upper.is_finite() || upper < 0. {
            return Err(invalid("finite nonnegative column cap required"));
        }
        if upper < master.upper()[column] {
            master.set_upper(column, upper)?;
        }
    }
    Ok(())
}
