//! Imported-domain evaluation hooks for the common branch-and-price driver.
use super::*;
use crate::master::tree::{
    Decision, Evaluation, Limits, Model, NodeInfo, QueuedNode, Stop, TreeFrontier,
};

struct ImportedEvaluation<const D: usize> {
    children: Vec<Pending<D>>,
    status: Option<TreeStatus>,
    requeue: bool,
}
impl<const D: usize> Evaluation for ImportedEvaluation<D> {
    fn proven_infeasible(&self) -> bool {
        false
    }
}
impl<const D: usize> NodeInfo for Pending<D> {
    fn depth(&self) -> usize {
        0
    }
    fn numerical_bound(&self) -> f64 {
        self.bound
    }
}
struct ImportedSearch<'a, const D: usize, F> {
    tree: &'a mut ImportTree<D>,
    options: &'a ModelTreeOptions,
    imports: &'a ImportOptions,
    deadline: Instant,
    oracle_rounds: bool,
    source: F,
    nodes: usize,
    batches: usize,
    added_columns: usize,
    status: Option<TreeStatus>,
}
impl<const D: usize, F> Model for ImportedSearch<'_, D, F>
where
    F: FnMut(&ImportPricing<D>, Duration) -> Result<ImportOracleReply>,
{
    type Node = Pending<D>;
    type Output = ImportedEvaluation<D>;
    fn limits(&self) -> Limits {
        Limits {
            deadline: self.deadline,
            remaining_nodes: self.options.node_limit.saturating_sub(self.nodes),
            batch_size: 1,
            time_first: true,
        }
    }
    fn prune(&mut self, frontier: &mut TreeFrontier<Self::Node, Self::Output>) {
        let cutoff = self.tree.incumbent() - self.options.tolerance;
        frontier.retain(|node| node.bound < cutoff);
    }
    fn admit(&mut self, _entry: &mut QueuedNode<Self::Node, Self::Output>) -> bool {
        self.nodes += 1;
        true
    }
    fn evaluate(
        &mut self,
        entries: &mut [QueuedNode<Self::Node, Self::Output>],
        frontier: &mut TreeFrontier<Self::Node, Self::Output>,
    ) -> Result<Vec<Result<Self::Output>>> {
        let pending = &mut entries[0].node;
        let tree = &mut *self.tree;
        let options = self.options;
        let imports = self.imports;
        let deadline = self.deadline;
        let oracle_rounds = self.oracle_rounds;
        let mut source = &mut self.source;
        let nodes = &mut self.nodes;
        let batches = &mut self.batches;
        let added_columns = &mut self.added_columns;
        let configured = || {
            let mut node = options.node.clone();
            node.cg.time_limit = node
                .cg
                .time_limit
                .min(deadline.saturating_duration_since(Instant::now()));
            node
        };
        let suspended = |status| -> Result<Vec<Result<ImportedEvaluation<D>>>> {
            Ok(vec![Ok(ImportedEvaluation {
                children: vec![],
                status: Some(status),
                requeue: true,
            })])
        };
        let branched = |children: Vec<Pending<D>>| -> Result<Vec<Result<ImportedEvaluation<D>>>> {
            Ok(vec![Ok(ImportedEvaluation {
                children,
                status: None,
                requeue: false,
            })])
        };
        let mut empty_batches = 0;
        loop {
            if Instant::now() >= deadline {
                return suspended(TreeStatus::TimeLimit);
            }
            if pending.session.is_complete() {
                // Finite-only preprocessing, enumeration and MIP become valid
                // exactly at this boundary, after fresh journal replay.
                let output = pending.session.solve(configured())?;
                let mut finite = options.clone();
                finite.node_limit = options.node_limit.saturating_sub(*nodes).saturating_add(1);
                finite.time_limit = deadline.saturating_duration_since(Instant::now());
                finite.node.cg.time_limit = finite.node.cg.time_limit.min(finite.time_limit);
                let result = output.output.snapshot.solve_tree(finite)?;
                *nodes = nodes.saturating_add(result.nodes.saturating_sub(1));
                if result.incumbent.is_finite() {
                    tree.retain_solution(
                        result.incumbent,
                        result.routes,
                        output.context.registry.values(&result.master_values)?,
                    );
                }
                match result.status {
                    TreeStatus::Optimal | TreeStatus::Infeasible => break,
                    other => {
                        pending.bound = pending.bound.max(result.bound);

                        return suspended(match other {
                            TreeStatus::NodeLimit => TreeStatus::NodeLimit,
                            TreeStatus::TimeLimit => TreeStatus::TimeLimit,
                            _ => TreeStatus::Incomplete,
                        });
                    }
                }
            }
            // The selected point covers all pending groups. Cursor selection
            // also prevents repeated empty batches from starving other groups.
            let group = loop {
                let group = pending.session.order[pending.session.cursor];
                if pending.session.pending.contains(&group) {
                    break group;
                }
                pending.session.cursor = (pending.session.cursor + 1) % pending.session.order.len();
            };
            let view = pending.session.view(group)?;
            let phase = match view.feasibility_point(configured(), true) {
                Ok(point) => point,
                Err(_) if Instant::now() >= deadline => {
                    return suspended(TreeStatus::TimeLimit);
                }
                Err(error) => return Err(error),
            };
            if phase
                .domain_certificate()?
                .is_some_and(|c| c.proves_infeasible())
            {
                break;
            }
            let mut deferred_children = None;
            let point = if phase.restricted_violation().unwrap_or(f64::INFINITY) > 0. {
                phase
            } else {
                let (output, registry) = match view.objective_output(configured()) {
                    Ok(output) => output,
                    Err(_) if Instant::now() >= deadline => {
                        return suspended(TreeStatus::TimeLimit);
                    }
                    Err(error) => return Err(error),
                };
                let point = super::super::super::pricing::certificate::objective_request_at_output(
                    &view, &output, registry,
                );
                if let Some(certificate) = point.domain_certificate()? {
                    pending.bound = pending.bound.max(certificate.lower_bound());
                }
                if let Some((objective, routes, master_values)) =
                    output.snapshot.integer_witness(options.tolerance)?
                {
                    let compiled = lifecycle::compile_definition(
                        &pending.session.definition,
                        Some(&pending.session.context),
                    )?;
                    tree.retain_solution(
                        objective,
                        routes,
                        compiled.registry.values(&master_values)?,
                    );
                }
                if pending.bound >= tree.incumbent() - options.tolerance {
                    break;
                }
                if let Some(children) =
                    branching::children(&pending.session, &output, options.tolerance)?
                {
                    if oracle_rounds {
                        deferred_children = Some(children);
                    } else {
                        return branched(
                            children
                                .into_iter()
                                .map(|session| Pending {
                                    session,
                                    bound: pending.bound,
                                })
                                .collect(),
                        );
                    }
                }
                point
            };
            let collected = round::collect(
                pending,
                &mut |current: &Pending<D>| {
                    for peer in frontier.iter_mut() {
                        peer.node.session.definition = current.session.definition.clone();
                        peer.node.session.pending = current.session.pending.clone();
                    }
                },
                &point,
                imports.max_batches.saturating_sub(*batches),
                deadline,
                options.node.cg.tolerance,
                oracle_rounds,
                oracle_rounds && options.node.cg.bound_mode == cg::BoundMode::Numerical,
                &mut source,
            )?;
            *batches += collected.batches;
            *added_columns += collected.added;
            if let Some(bound) = collected.numerical_bound {
                if point.kind() == ImportPricingKind::Feasibility {
                    if bound > options.tolerance {
                        break;
                    }
                } else {
                    pending.bound = pending.bound.max(bound);
                }
            }
            if pending.bound >= tree.incumbent() - options.tolerance {
                break;
            }
            if let Some(status) = collected.status {
                return suspended(status);
            }
            let progress = collected.added > 0 || collected.completed > 0;
            if oracle_rounds {
                if progress {
                    continue;
                }
                if let Some(children) = deferred_children {
                    return branched(
                        children
                            .into_iter()
                            .map(|mut session| {
                                session.definition = pending.session.definition.clone();
                                session.pending = pending.session.pending.clone();
                                Pending {
                                    session,
                                    bound: pending.bound,
                                }
                            })
                            .collect(),
                    );
                }

                return suspended(TreeStatus::Incomplete);
            }
            // Empty one-group batches must visit a complete fair round before
            // suspending. An oracle round has already visited every open group.
            empty_batches = if progress { 0 } else { empty_batches + 1 };
            if empty_batches >= pending.session.pending.len().max(1) {
                return suspended(TreeStatus::Incomplete);
            }
        }
        Ok(vec![Ok(ImportedEvaluation {
            children: vec![],
            status: None,
            requeue: false,
        })])
    }
    fn consume(
        &mut self,
        entry: Option<QueuedNode<Self::Node, Self::Output>>,
        output: Result<Self::Output>,
        _reused: bool,
        frontier: &mut TreeFrontier<Self::Node, Self::Output>,
    ) -> Result<Decision> {
        let entry =
            entry.ok_or_else(|| invalid("imported pricing requires its session payload"))?;
        let output = output?;
        if output.requeue {
            frontier.push_evaluated(entry.node, entry.parent_branch_order, None)?;
        }
        for child in output.children {
            frontier.push_evaluated(child, entry.branch_order, None)?;
        }
        if let Some(status) = output.status {
            self.status = Some(status);
            Ok(Decision::Stop {
                reason: Stop::Requested,
                restore_ready: false,
            })
        } else {
            Ok(Decision::Continue)
        }
    }
}

pub(super) fn run<const D: usize, F>(
    input: &ImportTree<D>,
    options: &ModelTreeOptions,
    imports: &ImportOptions,
    oracle_rounds: bool,
    source: F,
) -> Result<ImportTreeProgress<D>>
where
    F: FnMut(&ImportPricing<D>, Duration) -> Result<ImportOracleReply>,
{
    super::super::super::super::validate_options(&options.node)?;
    if !options.tolerance.is_finite() || options.tolerance <= 0. || options.tolerance >= 0.5 {
        return Err(invalid(
            "import tree tolerance must be between zero and one half",
        ));
    }
    if input
        .bound_mode
        .is_some_and(|mode| mode != options.node.cg.bound_mode)
    {
        return Err(invalid("resumed import tree must retain its bound mode"));
    }
    if options.enumerated_mip.is_some() && !cfg!(feature = "highs") {
        return Err(invalid("enumerated MIP requires the highs feature"));
    }
    let deadline = Instant::now()
        .checked_add(options.time_limit.min(imports.time_limit))
        .ok_or_else(|| invalid("import tree time limit"))?;
    let mut tree = input.clone();
    tree.bound_mode = Some(options.node.cg.bound_mode);
    let mut frontier = TreeFrontier::exact_best_bound();
    for pending in tree.pending.drain(..) {
        frontier.push_evaluated(pending, 0, None)?;
    }
    let mut model = ImportedSearch {
        tree: &mut tree,
        options,
        imports,
        deadline,
        oracle_rounds,
        source,
        nodes: 0,
        batches: 0,
        added_columns: 0,
        status: None,
    };
    let stop = crate::master::tree::run(&mut model, &mut frontier, None)?;
    let nodes = model.nodes;
    let batches = model.batches;
    let added_columns = model.added_columns;
    let status = model.status.unwrap_or(match stop {
        Stop::TimeLimit => TreeStatus::TimeLimit,
        Stop::NodeLimit => TreeStatus::NodeLimit,
        _ => TreeStatus::Incomplete,
    });
    drop(model);
    tree.pending = frontier
        .into_nodes()
        .into_iter()
        .map(|entry| entry.node)
        .collect();
    let incumbent = tree.incumbent();
    tree.pending
        .retain(|n| n.bound < incumbent - options.tolerance);
    let status = if tree.pending.is_empty() {
        if incumbent.is_finite() {
            TreeStatus::Optimal
        } else {
            TreeStatus::Infeasible
        }
    } else {
        status
    };
    let bound = tree
        .pending
        .iter()
        .map(|n| n.bound)
        .fold(incumbent, f64::min);
    Ok(ImportTreeProgress {
        open_nodes: tree.pending.len(),
        bound_mode: options.node.cg.bound_mode,
        tree,
        status,
        bound,
        incumbent,
        nodes,
        batches,
        added_columns,
    })
}
