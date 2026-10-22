//! Incremental finite-domain imports with an explicit completion boundary.
use super::*;
use std::time::{Duration, Instant};
mod priced;

/// A source declares completion only after providing its entire intended domain.
/// This is domain enumeration, not a reduced-cost pricing certificate.
#[derive(Clone, Debug)]
pub struct ColumnBatch {
    pub columns: Vec<Column>,
    pub complete: bool,
}
#[derive(Clone, Debug)]
pub struct ImportOptions {
    pub max_batches: usize,
    pub time_limit: Duration,
}
impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            max_batches: 256,
            time_limit: Duration::from_secs(60),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportStatus {
    Complete,
    BatchLimit,
    TimeLimit,
}

/// An immutable collection session. Incomplete sessions cannot solve or inherit
/// finite-parent proofs. Explicit envelope-covered certificates are separate.
#[derive(Clone, Debug)]
pub struct ColumnImport<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub(super) context: Rc<Context<D>>,
    pub(super) definition: Rc<Definition<D>>,
    pub(super) position: usize,
    pub(super) complete: bool,
    pub(super) open_groups: BTreeSet<i32>,
}
#[derive(Debug)]
pub struct ImportProgress<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub session: ColumnImport<D>,
    pub status: ImportStatus,
    pub batches: usize,
    pub added_columns: usize,
}
impl<const D: usize> ImportedSnapshot<D> {
    /// Begin extending one existing imported group through an external source.
    /// All other groups retain their declared domains. The source receives the
    /// original group definition plus every column accepted in this session.
    pub fn begin_import(&self, subproblem: i32) -> Result<ColumnImport<D>> {
        let position = self
            .context
            .definition
            .subproblems
            .iter()
            .position(|s| s.id == subproblem)
            .ok_or_else(|| invalid("unknown imported subproblem"))?;
        Ok(ColumnImport {
            context: self.context.clone(),
            definition: self.context.definition.clone(),
            position,
            complete: false,
            open_groups: BTreeSet::from([subproblem]),
        })
    }
}
impl<const D: usize> ColumnImport<D> {
    pub fn subproblem(&self) -> &Subproblem {
        &self.definition.subproblems[self.position]
    }
    pub fn is_complete(&self) -> bool {
        self.complete && self.open_groups.is_empty()
    }

    /// Collect bounded batches without solving a restricted domain. The time
    /// limit is checked between callbacks; callbacks receive the remaining time
    /// and must cooperate with it. A synchronous callback cannot be preempted.
    /// Invalid batches/errors leave this session and its parent unchanged.
    pub fn collect<F>(&self, options: &ImportOptions, mut source: F) -> Result<ImportProgress<D>>
    where
        F: FnMut(&Subproblem, Duration) -> Result<ColumnBatch>,
    {
        let deadline = Instant::now()
            .checked_add(options.time_limit)
            .ok_or_else(|| invalid("import time limit"))?;
        let mut session = Self {
            context: self.context.clone(),
            definition: self.definition.clone(),
            position: self.position,
            complete: self.complete,
            open_groups: self.open_groups.clone(),
        };
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
            if batches == options.max_batches {
                break ImportStatus::BatchLimit;
            }
            let batch = source(session.subproblem(), remaining)?;
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

    pub(super) fn accept(&self, batch: ColumnBatch) -> Result<Self> {
        let old = &self.definition;
        let mut definition = Definition {
            problem: old.problem.clone(),
            variables: old.variables.clone(),
            subproblems: old.subproblems.clone(),
            envelopes: old.envelopes.clone(),
        };
        definition.subproblems[self.position]
            .columns
            .extend(batch.columns);
        super::envelope::validate(&definition)?;
        // No LP or publication is initialized while validating a declaration.
        compile::model(
            definition.problem.clone(),
            definition.variables.clone(),
            definition.subproblems.clone(),
        )?;
        let mut open_groups = self.open_groups.clone();
        if batch.complete {
            open_groups.remove(&self.subproblem().id);
        } else {
            open_groups.insert(self.subproblem().id);
        }
        Ok(Self {
            context: self.context.clone(),
            definition: Rc::new(definition),
            position: self.position,
            complete: batch.complete,
            open_groups,
        })
    }

    /// Solve only after the source explicitly completes the intended domain.
    /// Rebuild all explicit restrictions and discard old finite-domain proofs.
    pub fn solve(&self, options: Options) -> Result<ImportedOutput<D>> {
        if !self.is_complete() {
            return Err(invalid("import domain is not complete"));
        }
        lifecycle::rebuild(self.definition.clone(), Some(&self.context), options)
    }
}
