//! Bounded node batches. The coordinator alone owns the frontier and incumbent.
use super::*;
use rayon::prelude::*;

pub(super) fn evaluate_entries<const D: usize>(
    workers: &rayon::ThreadPool,
    problem: &RoutingContext<D>,
    nodes: &[crate::master::tree::QueuedNode<Node<D>, routing::RootOutput<D>>],
    options: &TreeOptions,
    incumbent: f64,
    deadline: Instant,
) -> Result<Vec<Result<routing::RootOutput<D>>>> {
    // Clone on the coordinator: RoutingContext has RefCell data and snapshots
    // contain Rc/native handles. No borrowed source crosses the boundary.
    let jobs = nodes
        .iter()
        .map(|entry| {
            let node = &entry.node;
            Ok((
                problem.clone(),
                node.branches.clone(),
                node.parent
                    .as_ref()
                    .map(|p| p.export_for_worker())
                    .transpose()?,
                options.clone(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let results = workers.install(|| {
        jobs.into_par_iter()
            .map(|(problem, branches, parent, options)| {
                let parent = parent.map(|p| p.restore().map(Rc::new)).transpose()?;
                let pool = Pool {
                    routes: vec![],
                    customers: vec![],
                    seen: HashSet::new(),
                };
                evaluate(
                    parent.as_ref(),
                    None,
                    &problem,
                    &branches,
                    &pool,
                    &options,
                    incumbent,
                    deadline,
                    options.root.cg.max_iterations,
                    0,
                    true,
                )?
                .into_worker_transfer()
            })
            .collect::<Vec<_>>()
    });
    // Indexed parallel iteration retains dispatch order. Rebuild thread-local
    // models here before the existing sequential node-processing logic runs.
    Ok(results
        .into_iter()
        .map(|r| r.and_then(|restore| restore()))
        .collect())
}
