//! Owned data crossing tree-worker boundaries, never Rc or native LP pointers.
use super::*;
use crate::master::solvers::clp::MasterTransfer;

pub(crate) struct SnapshotTransfer<const D: usize> {
    master: MasterTransfer,
    problem: RoutingContext<D>,
    cut: CutState,
    pricing: Solver<D>,
    incumbent: f64,
    routes: Vec<Route>,
    warm: cg::WarmStart,
}

impl<const D: usize> NodeSnapshot<D> {
    pub(crate) fn export_for_worker(&self) -> Result<SnapshotTransfer<D>> {
        Ok(SnapshotTransfer {
            master: self.master.export_for_worker()?,
            problem: self.problem.clone(),
            cut: self.cut.clone(),
            pricing: {
                let mut pricing = self.pricing.fork_for_node();
                pricing.detach_step_versions();
                pricing
            },
            incumbent: self.incumbent,
            routes: self.routes.clone(),
            warm: self.warm.clone(),
        })
    }
}

impl<const D: usize> SnapshotTransfer<D> {
    pub(crate) fn restore(self) -> Result<NodeSnapshot<D>> {
        Ok(NodeSnapshot {
            master: self.master.restore()?,
            problem: self.problem,
            cut: self.cut,
            pricing: self.pricing,
            incumbent: self.incumbent,
            routes: self.routes,
            warm: self.warm,
        })
    }
}

impl<const D: usize> RootOutput<D> {
    /// Return a Send closure that reconstructs the output on the receiver.
    /// Its captures contain only owned data; the returned Rc is created there.
    pub(crate) fn into_worker_transfer(self) -> Result<impl FnOnce() -> Result<Self> + Send> {
        let Self {
            fixed_routes,
            fixed_cost,
            preprocessing,
            snapshot,
            simplex_iterations,
            ipm_iterations,
            inherited_cuts,
            master,
            routes,
            values,
            duals,
            incumbent,
            pricing_seconds,
            oracle_calls,
            solver_seconds,
            cuts,
            lp_before_cuts,
            fixing_calls,
            enumerated_routes,
        } = self;
        let snapshot = snapshot.map(|s| s.export_for_worker()).transpose()?;
        Ok(move || {
            Ok(Self {
                fixed_routes,
                fixed_cost,
                preprocessing,
                snapshot: snapshot.map(|s| s.restore().map(Rc::new)).transpose()?,
                simplex_iterations,
                ipm_iterations,
                inherited_cuts,
                master,
                routes,
                values,
                duals,
                incumbent,
                pricing_seconds,
                oracle_calls,
                solver_seconds,
                cuts,
                lp_before_cuts,
                fixing_calls,
                enumerated_routes,
            })
        })
    }
}
