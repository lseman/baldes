//! General-network evaluation and branching hooks for the shared tree driver.
use super::*;
use crate::master::tree::{Decision, Limits, Model, QueuedNode, Stop};

pub(super) struct NetworkSearch<'a, const D: usize> {
    pub options: &'a ModelTreeOptions,
    pub result: &'a mut ModelTreeOutput,
    pub deadline: Instant,
    pub windows: &'a [time_windows::NetworkWindows],
    pub base_depth: usize,
    pub subtree_history: &'a mut Option<crate::master::tree::strong::SubtreeHistory>,
    pub mapped_history: &'a mut model_strong::History,
    pub selected_nodes: &'a mut usize,
    pub unresolved: &'a mut Option<f64>,
}
impl<const D: usize> Model for NetworkSearch<'_, D> {
    type Node = NodeSnapshot<D>;
    type Output = Output<D>;
    fn limits(&self) -> Limits {
        Limits {
            deadline: self.deadline,
            remaining_nodes: self.options.node_limit.saturating_sub(*self.selected_nodes),
            batch_size: 1,
            time_first: false,
        }
    }
    fn admit(&mut self, entry: &mut QueuedNode<Self::Node, Self::Output>) -> bool {
        *self.selected_nodes += 1;
        if entry.evaluated.is_none() {
            self.result.processed_nodes += 1;
        }
        if entry.proven_infeasible()
            || entry.node.warm.bound >= self.result.incumbent - self.options.tolerance
        {
            if let Some(history) = self.subtree_history.as_mut() {
                history.close(entry.node.state.mapped_lineage);
            }
            false
        } else {
            true
        }
    }
    fn evaluate(
        &mut self,
        entries: &mut [QueuedNode<Self::Node, Self::Output>],
        _frontier: &mut crate::master::tree::TreeFrontier<Self::Node, Self::Output>,
    ) -> Result<Vec<Result<Self::Output>>> {
        entries
            .iter_mut()
            .map(|entry| {
                if self.options.share_inactive_columns {
                    self.mapped_history
                        .column_pool
                        .seed(&mut entry.node, self.deadline)?;
                }
                let mut options = self.options.node.clone();
                options.search_inactive_columns |= self.options.share_inactive_columns;
                options.cg.time_limit = options
                    .cg
                    .time_limit
                    .min(self.deadline.saturating_duration_since(Instant::now()));
                Ok(entry.node.solve(options))
            })
            .collect()
    }
    fn consume(
        &mut self,
        entry: Option<QueuedNode<Self::Node, Self::Output>>,
        output: Result<Self::Output>,
        reused: bool,
        pending: &mut TreeFrontier<D>,
    ) -> Result<Decision> {
        let mut output = output?;
        let current_branch_order = if let Some(entry) = entry {
            if reused {
                output.snapshot.state.mapped_lineage = entry.node.state.mapped_lineage;
            }
            entry.branch_order
        } else {
            self.result.processed_nodes += 1;
            *self.selected_nodes += 1;
            1
        };
        let options = self.options;
        let result = &mut *self.result;
        let deadline = self.deadline;
        let windows = self.windows;
        let base_depth = self.base_depth;
        let mut subtree_history = &mut *self.subtree_history;
        let mut mapped_history = &mut *self.mapped_history;
        let selected_nodes = &mut *self.selected_nodes;
        let unresolved = &mut *self.unresolved;
        if !reused {
            result.nodes += 1;
            result.iterations += output.master.iterations;
            result.pricing_calls += output.master.pricing_calls;
            for &(id, calls) in &output.oracle_calls {
                if let Some((_, total)) = result
                    .oracle_calls
                    .iter_mut()
                    .find(|(network, _)| *network == id)
                {
                    *total += calls;
                }
            }
            result
                .enumeration_reports
                .push(output.snapshot.enumeration_report().clone());
        }
        if options.share_inactive_columns {
            mapped_history.column_pool.absorb(&output.snapshot);
        }
        let bound = output.master.bound;
        let lineage = output.snapshot.state.mapped_lineage;
        mapped_history.completed_estimate = subtree_history
            .as_ref()
            .and_then(|h| h.mean(output.snapshot.state.node_depth.saturating_sub(base_depth)));
        // The whole original-flow relaxation proves LP infeasibility without
        // integer rounding. Complete-master integer deductions retain their
        // explicitly requested integer-domain policy.
        let preprocessing = output.snapshot.preprocessing_report();
        if preprocessing.propagation.infeasible
            && (!preprocessing.complete_domain || options.node.preprocessing.integer_domains)
        {
            if let Some(history) = &mut subtree_history {
                history.close(lineage);
            }
            return Ok(Decision::Continue);
        }
        if let Some(configured) = &options.enumerated_mip {
            let mut configured = configured.clone();
            configured.time_limit = configured
                .time_limit
                .min(deadline.saturating_duration_since(Instant::now()));
            configured.objective_cutoff = configured.objective_cutoff.min(result.incumbent);
            let attempt = if output.snapshot.enumeration_report().complete {
                Some(output.snapshot.solve_enumerated_mip(&configured))
            } else if configured.objective_cutoff.is_finite() {
                let mut limits = options.node.enumeration.clone().unwrap_or_default();
                limits.time_limit = limits
                    .time_limit
                    .min(deadline.saturating_duration_since(Instant::now()));
                match output
                    .snapshot
                    .enumerate_below(configured.objective_cutoff, &limits)
                {
                    Ok(view) => {
                        result.enumeration_reports.push(view.report().clone());
                        configured.time_limit = configured
                            .time_limit
                            .min(deadline.saturating_duration_since(Instant::now()));
                        Some(view.solve_mip(&configured).map(Some))
                    }
                    Err(error) => {
                        result.enumeration_reports.push(EnumerationReport {
                            attempts: 1,
                            coverage_cutoff: Some(configured.objective_cutoff),
                            failure: Some(error.to_string()),
                            ..Default::default()
                        });
                        None
                    }
                }
            } else {
                None
            };
            if let Some(attempt) = attempt {
                result.enumerated_mip_calls += 1;
                match attempt {
                    Ok(Some(mip)) => {
                        use crate::master::solvers::mip::MipStatus;
                        result.enumerated_mip_nodes += mip.mip.nodes;
                        if let Some(solution) = &mip.mip.solution {
                            if solution.objective < result.incumbent {
                                result.incumbent = solution.objective;
                                result.routes = mip.route_values;
                                result.master_values = mip.master_values;
                            }
                        }
                        // HiGHS supplies a numerical integer bound; it never
                        // becomes a directed-arithmetic publication certificate.
                        if options.node.cg.bound_mode == cg::BoundMode::Numerical {
                            if mip.mip.status == MipStatus::Infeasible {
                                if let Some(history) = &mut subtree_history {
                                    history.close(lineage);
                                }
                                return Ok(Decision::Continue);
                            }
                            if matches!(
                                mip.mip.status,
                                MipStatus::Optimal | MipStatus::ObjectiveCutoff
                            ) && result.incumbent.is_finite()
                                && mip.mip.bound >= result.incumbent - options.tolerance
                            {
                                if let Some(history) = &mut subtree_history {
                                    history.close(lineage);
                                }
                                return Ok(Decision::Continue);
                            }
                        }
                    }
                    Err(_) => {
                        result.enumerated_mip_failures += 1;
                    }
                    Ok(None) => {}
                }
            }
        }
        if model_strong::proven_infeasible(&output) {
            if let Some(history) = &mut subtree_history {
                history.close(lineage);
            }
            return Ok(Decision::Continue);
        }
        // Exact pricing can already rule out improvement while phase I remains
        // unable to certify original infeasibility, or the LP hits a limit.
        // Use the selected lower-bound channel, never the artificial objective,
        // and require an existing original-model incumbent to close such a node.
        if result.incumbent.is_finite()
            && bound.is_finite()
            && bound >= result.incumbent - options.tolerance
        {
            if let Some(history) = &mut subtree_history {
                history.close(lineage);
            }
            return Ok(Decision::Continue);
        }
        if output.master.status != cg::Status::Optimal {
            result.status = if output.master.status == cg::Status::TimeLimit {
                TreeStatus::TimeLimit
            } else {
                TreeStatus::Incomplete
            };
            *unresolved = Some(bound);
            return Ok(Decision::Stop {
                reason: Stop::Requested,
                restore_ready: false,
            });
        }
        if let Some((objective, routes, master_values)) =
            output.snapshot.integer_witness(options.tolerance)?
        {
            if objective < result.incumbent {
                result.incumbent = objective;
                result.routes = routes;
                result.master_values = master_values;
            }
            if bound >= result.incumbent - options.tolerance {
                if let Some(history) = &mut subtree_history {
                    history.close(lineage);
                }
                return Ok(Decision::Continue);
            }
        }
        let count_children = if options.branching == NetworkBranchingPolicy::MultiplicityFirst {
            branching::multiplicity_children(&output, options.tolerance)?
        } else {
            None
        };
        if options.mapped_strong_branching.is_some()
            && count_children.is_none()
            && *selected_nodes >= options.node_limit
            && (options.branching == NetworkBranchingPolicy::Phased
                || !model_strong::mapped_proposals(
                    &output,
                    &options.mapped_variables,
                    options.tolerance,
                )?
                .is_empty())
        {
            result.status = TreeStatus::NodeLimit;
            *unresolved = Some(bound);
            return Ok(Decision::Stop {
                reason: Stop::Requested,
                restore_ready: false,
            });
        }
        let trials_before = mapped_history.processed_trials;
        let children = if options.branching == NetworkBranchingPolicy::Phased {
            let proposals = proposals::collect(&output, &options, windows)?;
            if let Some(mut children) = model_strong::choose_proposals(
                &output,
                &options,
                &mut mapped_history,
                result.incumbent,
                deadline,
                proposals,
            )? {
                children.admit(NetworkBranchDirection::UpperFirst);
                Some(children)
            } else {
                phased_fallback_selection(branching::phased_children(
                    &output,
                    options.tolerance,
                    windows,
                )?)
            }
        } else if count_children.is_some() {
            count_children.map(model_strong::Selection::fresh)
        } else if let Some(mut children) = model_strong::choose(
            &output,
            &options,
            &mut mapped_history,
            result.incumbent,
            deadline,
        )? {
            children.admit(
                if options.branching == NetworkBranchingPolicy::MultiplicityFirst {
                    NetworkBranchDirection::UpperFirst
                } else {
                    NetworkBranchDirection::LowerFirst
                },
            );
            Some(children)
        } else {
            branching::children(&output, options.tolerance, options.branching, windows, &[])?
                .map(model_strong::Selection::fresh)
        };
        result.processed_nodes += mapped_history.processed_trials - trials_before;
        if let Some((objective, routes, master_values)) = mapped_history.incumbent.take() {
            if objective < result.incumbent {
                result.incumbent = objective;
                result.routes = routes;
                result.master_values = master_values;
            }
        }
        if let Some(mut selection) = children {
            let children = &mut selection.children;
            if children.len() != 2 {
                return Err(invalid("network branch proposal must have two children"));
            }
            let lineages = subtree_history
                .as_mut()
                .map(|h| h.split(lineage))
                .unwrap_or([0; 2]);
            for (side, child) in children.iter_mut().enumerate() {
                child.state.node_depth = output.snapshot.state.node_depth.saturating_add(1);
                child.state.mapped_lineage = lineages[side];
            }
            for (side, child) in selection.children.into_iter().enumerate() {
                pending.push_evaluated(
                    child,
                    current_branch_order,
                    selection.evaluated[side].take(),
                )?;
            }
        } else {
            *unresolved = Some(bound);
            return Ok(Decision::Stop {
                reason: Stop::Requested,
                restore_ready: false,
            });
        }
        Ok(Decision::Continue)
    }
}
