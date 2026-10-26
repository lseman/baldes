//! Fair bounded collection over a shared immutable domain definition.
use super::*;
use std::time::Duration;

impl<const D: usize> MultiImport<D> {
    fn collect_with<F>(
        &self,
        limits: &ImportOptions,
        mut source: F,
    ) -> Result<MultiImportProgress<D>>
    where
        F: FnMut(&Self, i32, Instant) -> Result<Option<ColumnBatch>>,
    {
        let deadline = Instant::now()
            .checked_add(limits.time_limit)
            .ok_or_else(|| invalid("import time limit"))?;
        let mut session = self.clone();
        let mut batches = 0;
        let mut added_columns = 0;
        let status = loop {
            if session.pending.is_empty() {
                break ImportStatus::Complete;
            }
            if Instant::now() >= deadline {
                break ImportStatus::TimeLimit;
            }
            if batches == limits.max_batches {
                break ImportStatus::BatchLimit;
            }
            let group = session.order[session.cursor];
            session.cursor = (session.cursor + 1) % session.order.len();
            if !session.pending.contains(&group) {
                continue;
            }
            let Some(batch) = source(&session, group, deadline)? else {
                break ImportStatus::TimeLimit;
            };
            let count = batch.columns.len();
            let next = session.view(group)?.accept(batch)?;
            session.definition = next.definition;
            session.pending = next.open_groups;
            batches += 1;
            added_columns += count;
        };
        Ok(MultiImportProgress {
            session,
            status,
            batches,
            added_columns,
        })
    }
    /// Round-robin batches with a shared deadline. Completion applies only to
    /// the requested group. Errors leave the input session unchanged.
    pub fn collect<F>(
        &self,
        limits: &ImportOptions,
        mut source: F,
    ) -> Result<MultiImportProgress<D>>
    where
        F: FnMut(&Subproblem, Duration) -> Result<ColumnBatch>,
    {
        self.collect_with(limits, |session, group, deadline| {
            source(
                session.subproblem(group)?,
                deadline.saturating_duration_since(Instant::now()),
            )
            .map(Some)
        })
    }
    /// Requests see columns accepted from every group. Numerical lack of
    /// improvement never completes an external domain.
    pub fn collect_priced<F>(
        &self,
        limits: &ImportOptions,
        options: Options,
        mut source: F,
    ) -> Result<MultiImportProgress<D>>
    where
        F: FnMut(&ImportPricing<D>, Duration) -> Result<ColumnBatch>,
    {
        let mut points = super::super::pricing::session::PointCache::default();
        self.collect_with(limits, |session, group, deadline| {
            let view = session.view(group)?;
            let Some(point) = points.next(&view, &options, deadline)? else {
                return Ok(None);
            };
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            let batch = source(&point, remaining)?;
            points.accepted(&batch);
            Ok(Some(batch))
        })
    }
}
