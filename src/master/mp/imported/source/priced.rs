//! Bounded phase-I/objective request orchestration for external column sources.
use super::*;

impl<const D: usize> ColumnImport<D> {
    /// Collect source batches with fresh numerical prices for each accepted
    /// restricted domain. A positive phase-I violation selects feasibility
    /// prices; zero selects objective prices. The source's completion flag
    /// still means full enumeration, never merely no negative reduced cost.
    /// Pricing calls share the collection deadline and respect stricter LP
    /// budgets. Errors leave the input session unchanged.
    pub fn collect_priced<F>(
        &self,
        limits: &ImportOptions,
        options: Options,
        mut source: F,
    ) -> Result<ImportProgress<D>>
    where
        F: FnMut(&ImportPricing<D>, Duration) -> Result<ColumnBatch>,
    {
        let deadline = Instant::now()
            .checked_add(limits.time_limit)
            .ok_or_else(|| invalid("import time limit"))?;
        let mut session = self.clone();
        let mut batches = 0;
        let mut added_columns = 0;
        let status = loop {
            if session.complete {
                break ImportStatus::Complete;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break ImportStatus::TimeLimit;
            }
            if batches == limits.max_batches {
                break ImportStatus::BatchLimit;
            }
            let Some(point) = super::super::pricing::next_point(&session, &options, deadline)?
            else {
                break ImportStatus::TimeLimit;
            };
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break ImportStatus::TimeLimit;
            }
            let batch = source(&point, remaining)?;
            let count = batch.columns.len();
            session = session.accept(batch)?;
            batches += 1;
            added_columns += count;
        };
        Ok(ImportProgress {
            session,
            status,
            batches,
            added_columns,
        })
    }
}
