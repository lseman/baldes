//! Recover original-objective columns and install them transactionally.
use super::*;

pub(super) fn attempt<M: LpMaster, C: Controller<M>>(
    master: &mut M,
    controller: &mut C,
    warm: &WarmStart,
    options: &Options,
    started: Instant,
    out: &mut Output,
    columns: super::super::feasibility::Columns<'_>,
) -> Result<Option<bool>> {
    let recovery = controller.recover_feasibility(
        master,
        warm,
        options.time_limit.saturating_sub(started.elapsed()),
        options.max_iterations.saturating_sub(out.iterations),
        options.tolerance,
        options.bound_mode,
    )?;
    recovery
        .map(|recovery| {
            super::super::feasibility::install(master, controller, recovery, out, columns)
        })
        .transpose()
}
