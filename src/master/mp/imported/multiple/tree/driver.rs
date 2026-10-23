//! Certificate-gated processing of a persistent external-pricing frontier.
use super::*;

pub(super) fn run<const D: usize, F>(
    input: &ImportTree<D>,
    options: &AssignmentTreeOptions,
    imports: &ImportOptions,
    oracle_rounds: bool,
    mut source: F,
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
    let mut nodes = 0;
    let mut batches = 0;
    let mut added_columns = 0;
    let configured = || {
        let mut node = options.node.clone();
        node.cg.time_limit = node
            .cg
            .time_limit
            .min(deadline.saturating_duration_since(Instant::now()));
        node
    };
    let status = 'frontier: loop {
        let incumbent = tree.incumbent();
        tree.pending
            .retain(|n| n.bound < incumbent - options.tolerance);
        if tree.pending.is_empty() {
            break if incumbent.is_finite() {
                ImportTreeStatus::Optimal
            } else {
                ImportTreeStatus::Infeasible
            };
        }
        if Instant::now() >= deadline {
            break ImportTreeStatus::TimeLimit;
        }
        if nodes >= options.node_limit {
            break ImportTreeStatus::NodeLimit;
        }
        let best = (0..tree.pending.len())
            .min_by(|&a, &b| tree.pending[a].bound.total_cmp(&tree.pending[b].bound))
            .unwrap();
        let mut pending = tree.pending.swap_remove(best);
        nodes += 1;
        let mut empty_batches = 0;
        loop {
            if Instant::now() >= deadline {
                tree.pending.push(pending);
                break 'frontier ImportTreeStatus::TimeLimit;
            }
            if pending.session.is_complete() {
                // Finite-only preprocessing, enumeration and MIP become valid
                // exactly at this boundary, after fresh journal replay.
                let output = pending.session.solve(configured())?;
                let mut finite = options.clone();
                finite.node_limit = options.node_limit.saturating_sub(nodes).saturating_add(1);
                finite.time_limit = deadline.saturating_duration_since(Instant::now());
                finite.node.cg.time_limit = finite.node.cg.time_limit.min(finite.time_limit);
                let result = output.output.snapshot.solve_assignment_tree(finite)?;
                nodes = nodes.saturating_add(result.nodes.saturating_sub(1));
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
                        tree.pending.push(pending);
                        break 'frontier match other {
                            TreeStatus::NodeLimit => ImportTreeStatus::NodeLimit,
                            TreeStatus::TimeLimit => ImportTreeStatus::TimeLimit,
                            _ => ImportTreeStatus::Incomplete,
                        };
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
                    tree.pending.push(pending);
                    break 'frontier ImportTreeStatus::TimeLimit;
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
                        tree.pending.push(pending);
                        break 'frontier ImportTreeStatus::TimeLimit;
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
                        tree.pending
                            .extend(children.into_iter().map(|session| Pending {
                                session,
                                bound: pending.bound,
                            }));
                        break;
                    }
                }
                point
            };
            let collected = round::collect(
                &mut pending,
                &mut tree.pending,
                &point,
                imports.max_batches.saturating_sub(batches),
                deadline,
                options.node.cg.tolerance,
                oracle_rounds,
                oracle_rounds && options.node.cg.bound_mode == cg::BoundMode::Numerical,
                &mut source,
            )?;
            batches += collected.batches;
            added_columns += collected.added;
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
                tree.pending.push(pending);
                break 'frontier status;
            }
            let progress = collected.added > 0 || collected.completed > 0;
            if oracle_rounds {
                if progress {
                    continue;
                }
                if let Some(children) = deferred_children {
                    tree.pending.extend(children.into_iter().map(|mut session| {
                        session.definition = pending.session.definition.clone();
                        session.pending = pending.session.pending.clone();
                        Pending {
                            session,
                            bound: pending.bound,
                        }
                    }));
                    break;
                }
                tree.pending.push(pending);
                break 'frontier ImportTreeStatus::Incomplete;
            }
            // Empty one-group batches must visit a complete fair round before
            // suspending. An oracle round has already visited every open group.
            empty_batches = if progress { 0 } else { empty_batches + 1 };
            if empty_batches >= pending.session.pending.len().max(1) {
                tree.pending.push(pending);
                break 'frontier ImportTreeStatus::Incomplete;
            }
        }
    };
    let incumbent = tree.incumbent();
    tree.pending
        .retain(|n| n.bound < incumbent - options.tolerance);
    let status = if tree.pending.is_empty() {
        if incumbent.is_finite() {
            ImportTreeStatus::Optimal
        } else {
            ImportTreeStatus::Infeasible
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
