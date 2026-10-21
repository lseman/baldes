//! Several external domains sharing one original model and restriction journal.
use super::*;
mod driver;
mod restrictions;
pub(super) mod tree;
pub use tree::{ImportTree, ImportTreeProgress, ImportTreeSolution, ImportTreeStatus};

#[derive(Clone, Debug)]
pub struct MultiImport<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    context: Rc<Context<D>>,
    definition: Rc<Definition<D>>,
    order: Vec<i32>,
    pending: BTreeSet<i32>,
    cursor: usize,
}
#[derive(Debug)]
pub struct MultiImportProgress<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub session: MultiImport<D>,
    pub status: ImportStatus,
    pub batches: usize,
    pub added_columns: usize,
}
impl<const D: usize> ImportedSnapshot<D> {
    /// Open selected groups together. Other groups remain finite. Duplicate or
    /// unknown IDs are errors; an empty selection is already complete.
    pub fn begin_imports(&self, groups: &[i32]) -> Result<MultiImport<D>> {
        let pending: BTreeSet<_> = groups.iter().copied().collect();
        if pending.len() != groups.len()
            || groups.iter().any(|id| {
                !self
                    .context
                    .definition
                    .subproblems
                    .iter()
                    .any(|g| g.id == *id)
            })
        {
            return Err(invalid("duplicate or unknown open import group"));
        }
        Ok(MultiImport {
            context: self.context.clone(),
            definition: self.context.definition.clone(),
            order: groups.to_vec(),
            pending,
            cursor: 0,
        })
    }
}
impl<const D: usize> MultiImport<D> {
    fn view(&self, group: i32) -> Result<ColumnImport<D>> {
        if !self.order.contains(&group) {
            return Err(invalid("group not selected for import"));
        }
        let position = self
            .definition
            .subproblems
            .iter()
            .position(|g| g.id == group)
            .ok_or_else(|| invalid("unknown imported subproblem"))?;
        Ok(ColumnImport {
            context: self.context.clone(),
            definition: self.definition.clone(),
            position,
            complete: !self.pending.contains(&group),
            open_groups: self.pending.clone(),
        })
    }
    pub fn pending_groups(&self) -> Vec<i32> {
        self.pending.iter().copied().collect()
    }
    pub fn is_complete(&self) -> bool {
        self.pending.is_empty()
    }
    pub fn subproblem(&self, id: i32) -> Result<&Subproblem> {
        self.view(id)?;
        self.definition
            .subproblems
            .iter()
            .find(|g| g.id == id)
            .ok_or_else(|| invalid("unknown imported subproblem"))
    }
    pub fn with_envelope(&self, group: i32, envelope: ColumnEnvelope) -> Result<Self> {
        let view = self.view(group)?.with_envelope(envelope)?;
        let mut session = self.clone();
        session.definition = view.definition;
        Ok(session)
    }
    /// Prices for one group at a point covering all currently accepted columns.
    pub fn pricing(&self, group: i32, options: Options) -> Result<ImportPricing<D>> {
        self.view(group)?.pricing(options)
    }
    pub fn feasibility_pricing(&self, group: i32, options: Options) -> Result<ImportPricing<D>> {
        self.view(group)?.feasibility_pricing(options)
    }
    pub(super) fn certificate_view(&self) -> Result<ColumnImport<D>> {
        let group = self
            .order
            .first()
            .ok_or_else(|| invalid("no selected import group for pricing"))?;
        self.view(*group)
    }
    /// Covers every unfinished group, requiring each one's envelope/cap proof.
    pub fn objective_certificate(&self, options: Options) -> Result<Option<ImportCertificate>> {
        self.certificate_view()?.objective_certificate(options)
    }
    pub fn feasibility_certificate(&self, options: Options) -> Result<Option<ImportCertificate>> {
        self.certificate_view()?.feasibility_certificate(options)
    }
    pub fn infeasibility_certificate(&self, options: Options) -> Result<Option<ImportCertificate>> {
        self.certificate_view()?.infeasibility_certificate(options)
    }
    pub fn solve(&self, options: Options) -> Result<ImportedOutput<D>> {
        if !self.is_complete() {
            return Err(invalid("import domains are not complete"));
        }
        lifecycle::rebuild(self.definition.clone(), Some(&self.context), options)
    }
}
