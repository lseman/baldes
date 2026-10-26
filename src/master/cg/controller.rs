use super::*;

/// The controller owns incumbent tracking and the pricing route universe.
pub trait Controller<M> {
    /// Explicit physical-row roles; append position and row sense do not
    /// identify dynamic constraints. Only active ordinary rows form the mean.
    fn curvature_rows(&self, row_count: usize) -> Vec<CurvatureRow> {
        vec![CurvatureRow::default(); row_count]
    }
    /// Whether the most recent batch visited every enabled pricing domain at
    /// its current dual point. A partial batch cannot become exact merely
    /// because the requested phase uses an exact oracle.
    fn pricing_complete(&self) -> bool {
        true
    }
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
    /// Transfer an aggregate computed for this pricing pass when its owner can
    /// relinquish it. The compatibility default retains borrowed callbacks.
    fn take_pricing_subgradient(&mut self, rhs: &[f64]) -> Option<Vec<f64>> {
        self.pricing_subgradient(rhs)
    }
    /// Opt in only when cached columns can be revalidated against the current
    /// pricing domain. Cached improvement never proves pricing exhaustion.
    fn uses_inactive_column_pool(&self) -> bool {
        false
    }
    /// Refresh one promising inactive column, or decline it if its path is no
    /// longer permitted. The default leaves domain ownership with the oracle.
    fn reprice_inactive_column(
        &mut self,
        _column: &Column,
        _duals: &[f64],
    ) -> Result<Option<Column>> {
        Ok(None)
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
    /// Refresh loaded coefficients after a pricing-universe transition.
    /// Return true when the LP or its coefficient contract changed, requiring
    /// a new solve and pricing convergence estimate before termination.
    fn refresh_master_columns(
        &mut self,
        _master: &mut M,
        _loaded: &mut [Option<Column>],
    ) -> Result<bool> {
        Ok(false)
    }
    /// A validated original-model primal is retained for publication. This
    /// permits an inherited bound to prune before pricing, without claiming
    /// that the pricing LP has converged. Compatibility controllers opt out.
    fn original_witness_available(&self) -> bool {
        false
    }
    fn incumbent(&self) -> f64 {
        f64::INFINITY
    }
    /// Enabled cut families and their current master-row counts for progress
    /// logging. Keep names and order stable during a solve; omit disabled families.
    fn cut_counts(&self) -> Vec<(&'static str, usize)> {
        vec![]
    }
    /// Current active bucket widths in resource order for progress logging.
    /// Empty means the pricing implementation does not expose a bucket grid.
    fn bucket_steps(&self) -> Vec<f64> {
        vec![]
    }
    /// Optional original-model feasible columns after exact pricing leaves
    /// positive artificials. These columns do not certify a bound or prove
    /// infeasibility. The controller must validate their original constraints.
    /// In certified mode, recovery from restricted pricing still requires an
    /// original-domain publication certificate; an oracle estimate is insufficient.
    fn restore_feasible_columns(&mut self, _duals: &[f64]) -> Result<Vec<Column>> {
        Ok(vec![])
    }
    /// Price a zero-real-objective phase I on independent model/pricing state.
    /// None declines recovery. An error must leave live state unchanged.
    /// Returned columns carry original costs, not feasibility costs. Infeasible
    /// requires whole-domain coverage; certified mode additionally requires a
    /// directed original-domain proof. Limits retain valid partial columns.
    fn recover_feasibility(
        &mut self,
        _master: &M,
        _warm: &WarmStart,
        _remaining: Duration,
        _iterations: usize,
        _tolerance: f64,
        _bound_mode: BoundMode,
    ) -> Result<Option<FeasibilityRecovery>> {
        Ok(None)
    }
    /// Save every controller/pricing field that enumeration, regeneration,
    /// column remapping and retained-pool synchronization can change.
    /// False declines the optional handoff.
    /// An error must leave state unchanged. Mutable enumeration callbacks must
    /// opt in and implement `finish_enumeration_transaction` for rollback.
    fn begin_enumeration_transaction(&mut self, _bound: f64) -> Result<bool> {
        Ok(false)
    }
    /// Commit on success (including a partial fixing pass returning no universe),
    /// or restore the saved state on any failed preparation/model edit.
    /// Rollback must be infallible and must not retain invalidated route tags.
    fn finish_enumeration_transaction(&mut self, _commit: bool) {}
    fn enumerated_universe(&mut self, _duals: &[f64], _bound: f64) -> Result<Option<Vec<Column>>> {
        Ok(None)
    }
    /// Yield a warm checkpoint before termination or in-place enumeration.
    /// Called only after accurate, exact pricing with no improving columns or
    /// positive artificials. The owner may atomically replace the model and
    /// resume with the returned warm start. This request proves no coverage.
    fn request_enumeration_checkpoint(&mut self, _master: &M) -> Result<bool> {
        Ok(false)
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
    /// Permit an owner to decline a redundant checkpoint before the driver
    /// copies column metadata. Existing controllers keep their callback flow.
    fn needs_pricing_checkpoint(&self, _master: &M) -> bool {
        true
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
