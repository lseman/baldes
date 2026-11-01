//! External pricing state snapshots independent of numerical pricing proofs.
use super::*;
use std::{any::Any, time::Duration};

/// Immutable node-local oracle snapshot, shared by children without deep cloning.
/// Interior mutation of snapshot payloads violates the isolation contract.
#[derive(Clone)]
pub struct ImportOracleState {
    value: Rc<dyn Any>,
    name: &'static str,
}
impl std::fmt::Debug for ImportOracleState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ImportOracleState")
            .field(&self.name)
            .finish()
    }
}
impl ImportOracleState {
    pub fn new<T: Any>(value: T) -> Self {
        Self {
            value: Rc::new(value),
            name: std::any::type_name::<T>(),
        }
    }
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.value.downcast_ref()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportOracleSetup {
    Ready,
    /// Trusted numerical external-oracle conclusion. It never establishes a
    /// directed certificate: certified trees retain the node as incomplete.
    Infeasible,
}

/// Stateful external optimizer. Setup observes replayed child restrictions
/// before CG; retained nodes record state before their children are queued.
/// Defaults match an oracle with no optional node-state hooks. Synchronous
/// callbacks must cooperate with the supplied remaining wall-clock budget.
pub trait ImportOracle<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    fn setup_node(
        &mut self,
        _node: &ImportOracleNode<D>,
        _state: Option<&ImportOracleState>,
        _remaining: Duration,
    ) -> Result<ImportOracleSetup> {
        Ok(ImportOracleSetup::Ready)
    }
    fn price(&mut self, point: &ImportPricing<D>, remaining: Duration)
        -> Result<ImportOracleReply>;
    fn record_node(
        &mut self,
        _node: &ImportOracleNode<D>,
        _remaining: Duration,
    ) -> Result<Option<ImportOracleState>> {
        Ok(None)
    }
}

impl<const D: usize, F> ImportOracle<D> for F
where
    F: FnMut(&ImportPricing<D>, Duration) -> Result<ImportOracleReply>,
{
    fn price(
        &mut self,
        point: &ImportPricing<D>,
        remaining: Duration,
    ) -> Result<ImportOracleReply> {
        self(point, remaining)
    }
}
