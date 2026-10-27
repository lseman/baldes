//! Label resources supplied by extensions.
//!
//! A plugin owns one resource's per-label state and its transitions over
//! physical arc positions (the original graph's arc vector order). The kernel
//! carries the state, calls the plugin on every extension and join, and
//! treats a plugin like any other resource. Directed route certification is
//! opt-in; ordinary numerical transitions are never taken as proof.
use crate::Result;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Instant;

/// One label's plugin value. Clones share it; equality compares values.
#[derive(Clone, Debug)]
pub struct PluginState(Arc<dyn PluginValue>);
impl PluginState {
    pub fn new<T: PluginValue>(value: T) -> Self {
        Self(Arc::new(value))
    }
    /// The value as its concrete type, or None for another plugin's state.
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.0.as_any().downcast_ref()
    }
}
impl PartialEq for PluginState {
    fn eq(&self, other: &Self) -> bool {
        same(&self.0, &other.0) || self.0.equals(&*other.0)
    }
}

/// Value stored in a [`PluginState`]; implemented for every comparable type.
pub trait PluginValue: Any + Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn equals(&self, other: &dyn PluginValue) -> bool;
}
impl<T: Any + Debug + Send + Sync + PartialEq> PluginValue for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn equals(&self, other: &dyn PluginValue) -> bool {
        other.as_any().downcast_ref::<T>() == Some(self)
    }
}

/// A resource that an extension adds to coupled labeling. Errors abort pricing;
/// they never prove infeasibility. Objective costs require explicit opt-in.
pub trait CoupledPlugin: Debug + Send + Sync {
    /// Whether this resource contributes an objective. Cost-bearing plugins
    /// use equality-only dominance, canonical suffix replay for joins, and no
    /// generic objective bounds or directed cost certification.
    fn has_costs(&self) -> bool {
        false
    }
    /// Accumulated resource objective in this directional state. Must be zero
    /// at both roots, finite at every reachable state, and deterministic.
    /// Forward replay of a complete route defines its objective. Backward
    /// objectives are provisional; joins replay the suffix from the prefix.
    /// Phase-I suppression is applied by the kernel, not the plugin.
    fn objective(&self, state: &PluginState, forward: bool) -> Result<f64> {
        let _ = (state, forward);
        Ok(0.0)
    }
    /// Check per-arc data against the graph's arc count.
    fn validate(&self, arcs: usize) -> Result<()>;
    /// The state at the source (forward) or the sink (backward).
    fn root(&self, forward: bool) -> PluginState;
    /// Extend along physical arc `arc`; None when infeasible.
    fn extend(&self, state: &PluginState, arc: usize, forward: bool)
        -> Result<Option<PluginState>>;
    /// Whether forward prefix `state` stays feasible along `suffix`, the
    /// backward label's arcs in route order.
    fn join(&self, state: &PluginState, suffix: &mut dyn Iterator<Item = usize>) -> Result<bool>;
    /// Whether label state `a` dominates `b` in this direction: every
    /// completion feasible for `b` is feasible for `a`. Default: equality.
    fn dominates(&self, a: &PluginState, b: &PluginState, forward: bool) -> bool {
        let _ = forward;
        a == b
    }
    /// Optional original-domain proof for a complete physical-arc path.
    ///
    /// `Some(true)` proves that the path admits this resource; `Some(false)`
    /// proves that it does not. `None` means unavailable, exhausted or unknown.
    /// Implementations must establish these claims independently of tolerant
    /// numerical labeling, debit bounded proof work through `work`, respect the
    /// cooperative deadline, and never increase the supplied work allowance.
    /// This feasibility-only proof contributes zero cost. Cost-bearing plugins
    /// cannot use it to certify their objective.
    ///
    /// The default supplies no proof; adding numerical `extend`/`join` methods
    /// alone never opts an existing plugin into certification.
    fn certify(&self, arcs: &[usize], work: &mut usize, deadline: Option<Instant>) -> Option<bool> {
        let _ = (arcs, work, deadline);
        None
    }
}

/// Shared handle to a plugin inside a resource definition. Two handles are
/// equal when they share one plugin instance.
#[derive(Clone, Debug)]
pub struct PluginResource(pub Arc<dyn CoupledPlugin>);
impl PluginResource {
    pub fn new<P: CoupledPlugin + 'static>(plugin: P) -> Self {
        Self(Arc::new(plugin))
    }
}
impl PartialEq for PluginResource {
    fn eq(&self, other: &Self) -> bool {
        same(&self.0, &other.0)
    }
}

/// Same allocation; vtable addresses are not compared.
fn same<T: ?Sized>(a: &Arc<T>, b: &Arc<T>) -> bool {
    std::ptr::eq(Arc::as_ptr(a) as *const (), Arc::as_ptr(b) as *const ())
}
