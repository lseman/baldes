//! A persistent frontier whose external domains need not be enumerated.
use super::*;
use crate::master::tree::TreeStatus;
use std::time::Duration;
mod branching;
mod driver;
mod round;

#[derive(Clone, Debug)]
pub struct ImportTreeSolution {
    pub objective: f64,
    pub routes: Vec<FixedRoute>,
    /// Original IDs and qualified column IDs at the incumbent's own lineage.
    pub values: Values,
}
#[derive(Clone, Debug)]
struct Pending<const D: usize> {
    session: MultiImport<D>,
    bound: f64,
}
/// Immutable, resumable branch-and-price frontier. Open-domain bounds use
/// directed envelopes or explicit numerical oracle replies under their chosen
/// policy. Finite-RMP objectives alone never prune a node.
#[derive(Clone, Debug)]
pub struct ImportTree<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pending: Vec<Pending<D>>,
    solution: Option<ImportTreeSolution>,
    bound_mode: Option<cg::BoundMode>,
}
#[derive(Debug)]
pub struct ImportTreeProgress<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub tree: ImportTree<D>,
    pub status: TreeStatus,
    pub bound_mode: cg::BoundMode,
    pub bound: f64,
    pub incumbent: f64,
    pub open_nodes: usize,
    /// Node evaluations during this invocation, including resumed nodes.
    pub nodes: usize,
    pub batches: usize,
    pub added_columns: usize,
}
impl<const D: usize> MultiImport<D> {
    /// Start without inheriting the finite parent's bound or infeasibility.
    /// Calling this declares every as-yet-unseen column optional (quantity
    /// lower bound zero). Mandatory columns must already be registered.
    /// Source completion must enumerate the group's entire domain, including
    /// patterns excluded by this node's aggregate branch restrictions.
    pub fn begin_tree(&self) -> ImportTree<D> {
        ImportTree {
            pending: vec![Pending {
                session: self.clone(),
                bound: f64::NEG_INFINITY,
            }],
            solution: None,
            bound_mode: None,
        }
    }
}
impl<const D: usize> ImportTree<D> {
    pub fn solution(&self) -> Option<&ImportTreeSolution> {
        self.solution.as_ref()
    }
    fn incumbent(&self) -> f64 {
        self.solution
            .as_ref()
            .map_or(f64::INFINITY, |s| s.objective)
    }
    fn retain_solution(&mut self, objective: f64, routes: Vec<FixedRoute>, values: Values) {
        if objective < self.incumbent() {
            self.solution = Some(ImportTreeSolution {
                objective,
                routes,
                values,
            });
        }
    }
    /// Run under shared node, callback and wall-clock budgets. The callback's
    /// completion flag still means full enumeration, never just no improvement.
    /// Each child replays original restrictions before requesting fresh prices.
    /// Errors preserve this frontier; callback side effects are caller-owned.
    pub fn run<F>(
        &self,
        options: &ModelTreeOptions,
        imports: &ImportOptions,
        source: F,
    ) -> Result<ImportTreeProgress<D>>
    where
        F: FnMut(&ImportPricing<D>, Duration) -> Result<ColumnBatch>,
    {
        let mut source = source;
        driver::run(self, options, imports, false, |point, remaining| {
            source(point, remaining).map(ImportOracleReply::columns)
        })
    }
    /// Query every unfinished group at one common point before branching.
    /// Numerical mode can consume trusted global numerical minima. Certified
    /// mode ignores those scalars for publication and pruning. Enumeration
    /// completion is independent of pricing optimality.
    pub fn run_with_oracles<F>(
        &self,
        options: &ModelTreeOptions,
        imports: &ImportOptions,
        source: F,
    ) -> Result<ImportTreeProgress<D>>
    where
        F: FnMut(&ImportPricing<D>, Duration) -> Result<ImportOracleReply>,
    {
        driver::run(self, options, imports, true, source)
    }
}
