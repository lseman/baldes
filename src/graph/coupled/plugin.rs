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
    /// Prefer a complete backward sweep over bidirectional/forward labeling.
    fn backward_pricing(&self) -> bool {
        false
    }
    /// Use scalar candidate prepricing followed by complete threshold stages.
    /// A reversed physical graph can implement this with forward labeling.
    fn staged_pricing(&self) -> bool {
        self.backward_pricing()
    }
    /// Optional additive surrogate for heuristic scalar prepricing. One charge
    /// per physical arc, in vector order. Candidates are replayed with the
    /// original resource; this hook never supplies a minimum certificate.
    fn scalar_pricing_charges(&self) -> Option<Vec<f64>> {
        None
    }
    /// Optional call-local resource with completion pruning at these additive
    /// arc costs. Must preserve every route below threshold; forward replay
    /// must remain unrestricted. None keeps the original resource.
    fn prepare_pricing(
        &self,
        costs: &[f64],
        threshold: f64,
        deadline: Instant,
    ) -> Option<PluginResource> {
        let _ = (costs, threshold, deadline);
        None
    }
    /// Directed lower bound on additive arc costs plus this resource's cost
    /// over every complete feasible route, independently of generated columns.
    fn certify_pricing_minimum(
        &self,
        costs: &[f64],
        work: &mut usize,
        deadline: Instant,
    ) -> Option<f64> {
        let _ = (costs, work, deadline);
        None
    }
    /// Independent outward enclosure of the resource objective on every feasible
    /// complete path with at most `max_arcs` physical arcs. This covers unseen
    /// paths, not just generated columns. None leaves global bounds unknown.
    /// Debit work, respect the deadline, and never increase the work allowance.
    fn objective_bounds(
        &self,
        max_arcs: usize,
        work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<[f64; 2]> {
        let _ = (max_arcs, work, deadline);
        None
    }
    /// Whether this resource contributes an objective. Cost-bearing plugins
    /// default to equality dominance, canonical suffix replay for joins, and no
    /// generic objective bounds or directed cost certification.
    fn has_costs(&self) -> bool {
        false
    }
    /// Opt into `dominates` for objective-bearing states. The implementation
    /// must also guarantee a no-greater objective after every common feasible
    /// completion, independently of the scalar arc objective. Default: equality.
    fn supports_cost_dominance(&self) -> bool {
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
    /// Optional directed compensation for cost-bearing label dominance.
    /// Feasibility must be preserved for every common completion. The amount
    /// pays for the future resource-objective excess after removing the current
    /// resource-objective difference already present in the scalar label key.
    /// None retains both labels. Only the exclusive staged-pricing adapter
    /// currently admits this hook; it never establishes a pricing certificate.
    fn cost_dominance_penalty(
        &self,
        a: &PluginState,
        b: &PluginState,
        forward: bool,
    ) -> Option<f64> {
        self.dominates(a, b, forward).then_some(0.)
    }
    /// Optional independent feasibility proof and outward objective enclosure
    /// for a complete physical-arc path. Cost-bearing plugins must implement
    /// this to certify incumbents. The interval covers this resource's original
    /// objective before kernel coefficients and phase-I suppression. Returning
    /// None means unknown; numerical replay alone is not a certificate.
    /// Debit work and respect deadlines as in `certify`.
    fn certify_objective(
        &self,
        arcs: &[usize],
        work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<(bool, [f64; 2])> {
        let _ = (arcs, work, deadline);
        None
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

#[cfg(test)]
#[path = "../../../tests/unit/graph/coupled/plugin_certificate_tests.rs"]
mod certificate_tests;
