//! Concurrent evaluation of a batch of open nodes. The coordinator alone owns
//! the frontier, incumbent and branching decisions; workers only re-solve
//! independent node copies. Native LP handles never cross threads: each copy
//! is exported to owned data and rebuilt on its receiver.
use super::*;
use crate::master::solvers::clp::MasterTransfer;
use rayon::prelude::*;

pub(super) struct SnapshotTransfer<const D: usize> {
    master: MasterTransfer,
    warm: cg::WarmStart,
    state: State<D>,
}

impl<const D: usize> NodeSnapshot<D> {
    fn into_transfer(mut self) -> Result<SnapshotTransfer<D>> {
        for oracle in &mut self.state.oracles {
            oracle.solver.detach_step_versions();
        }
        Ok(SnapshotTransfer {
            master: self.master.export_for_worker()?,
            warm: self.warm,
            state: self.state,
        })
    }
}

impl<const D: usize> SnapshotTransfer<D> {
    fn restore(self) -> Result<NodeSnapshot<D>> {
        Ok(NodeSnapshot {
            master: self.master.restore()?,
            warm: self.warm,
            state: self.state,
        })
    }
}

pub(super) struct OutputTransfer<const D: usize> {
    master: cg::Output,
    snapshot: SnapshotTransfer<D>,
    fixed_routes: Vec<FixedRoute>,
    routes: Vec<NetworkRoute>,
    values: Vec<f64>,
    duals: Vec<f64>,
    master_values: Vec<(i32, f64)>,
    oracle_calls: Vec<(i32, usize)>,
    cuts: usize,
    lp_before_cuts: f64,
}

impl<const D: usize> Output<D> {
    fn into_transfer(self) -> Result<OutputTransfer<D>> {
        Ok(OutputTransfer {
            master: self.master,
            snapshot: self.snapshot.into_transfer()?,
            fixed_routes: self.fixed_routes,
            routes: self.routes,
            values: self.values,
            duals: self.duals,
            master_values: self.master_values,
            oracle_calls: self.oracle_calls,
            cuts: self.cuts,
            lp_before_cuts: self.lp_before_cuts,
        })
    }
}

impl<const D: usize> OutputTransfer<D> {
    fn restore(self) -> Result<Output<D>> {
        Ok(Output {
            master: self.master,
            snapshot: self.snapshot.restore()?,
            fixed_routes: self.fixed_routes,
            routes: self.routes,
            values: self.values,
            duals: self.duals,
            master_values: self.master_values,
            oracle_calls: self.oracle_calls,
            cuts: self.cuts,
            lp_before_cuts: self.lp_before_cuts,
        })
    }
}

pub(super) fn workers(threads: usize) -> Result<Option<rayon::ThreadPool>> {
    if threads <= 1 {
        return Ok(None);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map(Some)
        .map_err(|e| invalid(format!("tree worker pool: {e}")))
}

/// Solve independent copies of `nodes` in parallel, preserving their order.
pub(super) fn evaluate<const D: usize>(
    workers: &rayon::ThreadPool,
    nodes: &[&NodeSnapshot<D>],
    options: &Options,
) -> Result<Vec<Result<Output<D>>>> {
    let jobs = nodes
        .iter()
        .map(|node| node.fork()?.into_transfer())
        .collect::<Result<Vec<_>>>()?;
    let results: Vec<Result<OutputTransfer<D>>> = workers.install(|| {
        jobs.into_par_iter()
            .map(|job| {
                job.restore()?
                    .solve_forked(options.clone())
                    .and_then(Output::into_transfer)
            })
            .collect()
    });
    Ok(results
        .into_iter()
        .map(|result| result.and_then(OutputTransfer::restore))
        .collect())
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/tree/parallel_tests.rs"]
mod tests;
