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
    pub workers: Option<&'a rayon::ThreadPool>,
    pub local_search_rng: crate::rng::GnuRand,
}
impl<const D: usize> Model for NetworkSearch<'_, D> {
    type Node = NodeSnapshot<D>;
    type Output = Output<D>;
    fn limits(&self) -> Limits {
        Limits {
            deadline: self.deadline,
            remaining_nodes: self.options.node_limit.saturating_sub(*self.selected_nodes),
            batch_size: if self.workers.is_some() {
                self.options.threads
            } else {
                1
            },
            time_first: false,
        }
    }
    fn admit(&mut self, entry: &mut QueuedNode<Self::Node, Self::Output>) -> bool {
        *self.selected_nodes += 1;
        if entry.evaluated.is_none() {
            self.result.processed_nodes += 1;
        }
        if entry.proven_infeasible()
            || self
                .options
                .closes(entry.node.tree_bound(), self.result.incumbent)
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
        if let Some(workers) = self.workers.filter(|_| entries.len() > 1) {
            if self.options.share_inactive_columns {
                for entry in entries.iter_mut() {
                    self.mapped_history
                        .column_pool
                        .seed(&mut entry.node, self.deadline)?;
                }
            }
            let mut options = self.options.node.clone();
            options.fixing_upper_bound = self
                .result
                .incumbent
                .is_finite()
                .then_some(self.result.incumbent);
            options.search_inactive_columns |= self.options.share_inactive_columns;
            options.cg.time_limit = options
                .cg
                .time_limit
                .min(self.deadline.saturating_duration_since(Instant::now()));
            let nodes: Vec<_> = entries.iter().map(|entry| &entry.node).collect();
            return parallel::evaluate(workers, &nodes, &options);
        }
        entries
            .iter_mut()
            .map(|entry| {
                if self.options.share_inactive_columns {
                    self.mapped_history
                        .column_pool
                        .seed(&mut entry.node, self.deadline)?;
                }
                let mut options = self.options.node.clone();
                options.fixing_upper_bound = self
                    .result
                    .incumbent
                    .is_finite()
                    .then_some(self.result.incumbent);
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
        let mapped_history = &mut *self.mapped_history;
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
            // Native Alg4EvalByMip::setupAlgo reserves 2% of the remaining
            // global budget for the rest of the search.
            configured.time_limit = configured.time_limit.min(
                deadline
                    .saturating_duration_since(Instant::now())
                    .mul_f64(0.98),
            );
            configured.objective_cutoff = configured.objective_cutoff.min(result.incumbent);
            let route_limit = output.snapshot.end_of_node_mip_route_limit()?;
            let attempt = if output.snapshot.enumeration_report().complete
                || output.snapshot.state.gap_pool_cutoff.is_some()
            {
                // Native end-of-node MIP runs only on a small enough universe.
                (output.snapshot.enumerated_route_count() <= route_limit)
                    .then(|| output.snapshot.solve_enumerated_mip(&configured))
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
                        configured.time_limit = configured.time_limit.min(
                            deadline
                                .saturating_duration_since(Instant::now())
                                .mul_f64(0.98),
                        );
                        (view.report().routes <= route_limit)
                            .then(|| view.solve_mip(&configured).map(Some))
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
                                && options.closes(mip.mip.bound, result.incumbent)
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
            && options.closes(bound, result.incumbent)
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
        primal_callbacks::run(&output, options, result, deadline)?;
        if options.closes(bound, result.incumbent) {
            if let Some(history) = &mut subtree_history {
                history.close(lineage);
            }
            return Ok(Decision::Continue);
        }
        if let Some((objective, routes, master_values)) =
            output.snapshot.integer_witness(options.tolerance)?
        {
            if objective < result.incumbent {
                result.incumbent = objective;
                result.routes = routes;
                result.master_values = master_values;
            }
            if options.closes(bound, result.incumbent) {
                if let Some(history) = &mut subtree_history {
                    history.close(lineage);
                }
                return Ok(Decision::Continue);
            }
        }
        if let Some(config) = options.diving.as_ref().filter(|config| {
            config.due(
                output.snapshot.state.node_depth.saturating_sub(base_depth),
                *selected_nodes,
            )
        }) {
            let dive = diving::run(&output, options, config, result.incumbent, deadline)?;
            result.diving_nodes += dive.nodes;
            result.iterations += dive.iterations;
            result.pricing_calls += dive.pricing_calls;
            for (id, calls) in dive.oracle_calls {
                if let Some((_, total)) = result
                    .oracle_calls
                    .iter_mut()
                    .find(|(network, _)| *network == id)
                {
                    *total += calls;
                }
            }
            if let Some((objective, routes, master_values)) = dive.incumbent {
                if objective < result.incumbent {
                    result.incumbent = objective;
                    result.routes = routes;
                    result.master_values = master_values;
                }
            }
            // The dive's own bounds are heuristic; only this node's bound prunes.
            if options.closes(bound, result.incumbent) {
                if let Some(history) = &mut subtree_history {
                    history.close(lineage);
                }
                return Ok(Decision::Continue);
            }
        }
        if let Some(config) = options
            .local_search
            .as_ref()
            .filter(|config| config.due(output.snapshot.state.node_depth))
        {
            let incumbent = result.incumbent.is_finite().then(|| {
                (
                    result.incumbent,
                    result.routes.clone(),
                    result.master_values.clone(),
                )
            });
            let dive = local_search::run(
                &output,
                options,
                config,
                incumbent,
                &mut self.local_search_rng,
                deadline,
            )?;
            result.diving_nodes += dive.nodes;
            result.iterations += dive.iterations;
            result.pricing_calls += dive.pricing_calls;
            for (id, calls) in dive.oracle_calls {
                if let Some((_, total)) = result
                    .oracle_calls
                    .iter_mut()
                    .find(|(network, _)| *network == id)
                {
                    *total += calls;
                }
            }
            if let Some((objective, routes, master_values)) = dive.incumbent {
                if objective < result.incumbent {
                    result.incumbent = objective;
                    result.routes = routes;
                    result.master_values = master_values;
                }
            }
            // Neighborhood bounds cover only that private dive. The node's
            // selected-channel bound remains the sole pruning authority.
            if options.closes(bound, result.incumbent) {
                if let Some(history) = &mut subtree_history {
                    history.close(lineage);
                }
                return Ok(Decision::Continue);
            }
        }
        if let Some(config) = options.restricted_master_mip.as_ref().filter(|config| {
            config.due(
                output.snapshot.state.node_depth.saturating_sub(base_depth),
                *selected_nodes,
            )
        }) {
            let mip = crate::master::solvers::mip::MipOptions {
                time_limit: Duration::try_from_secs_f64(config.time_limit)
                    .unwrap_or(Duration::ZERO)
                    .min(deadline.saturating_duration_since(Instant::now())),
                node_limit: config.node_limit,
                objective_cutoff: result.incumbent,
                // A restricted master is not a complete pool.
                deluxing: None,
            };
            result.restricted_mip_calls += 1;
            // A failed heuristic leaves normal branching available.
            if let Ok(Some((objective, routes, master_values))) =
                output.snapshot.solve_restricted_mip(&mip)
            {
                if objective < result.incumbent {
                    result.incumbent = objective;
                    result.routes = routes;
                    result.master_values = master_values;
                }
            }
            if options.closes(bound, result.incumbent) {
                if let Some(history) = &mut subtree_history {
                    history.close(lineage);
                }
                return Ok(Decision::Continue);
            }
        }
        // One priority dispatcher for every policy; the policy only presets
        // family priorities (`proposals::collect`) and child direction.
        let proposals = proposals::collect(&output, options, windows)?;
        if options.mapped_strong_branching.is_some()
            && *selected_nodes >= options.node_limit
            && (!proposals.is_empty() || options.branching == NetworkBranchingPolicy::Phased)
        {
            result.status = TreeStatus::NodeLimit;
            *unresolved = Some(bound);
            return Ok(Decision::Stop {
                reason: Stop::Requested,
                restore_ready: false,
            });
        }
        let upper_first = options.branching != NetworkBranchingPolicy::ResourceFirst;
        let trials_before = mapped_history.processed_trials;
        let children = if let Some(mut children) = model_strong::choose_proposals(
            &output,
            options,
            mapped_history,
            result.incumbent,
            deadline,
            proposals,
        )? {
            children.admit(if upper_first {
                NetworkBranchDirection::UpperFirst
            } else {
                NetworkBranchDirection::LowerFirst
            });
            Some(children)
        } else {
            // Non floor/ceiling disjunctions (synchronization, window
            // conflicts) and generators whose children could not be built.
            let fallback = branching::phased_children(&output, options.tolerance, windows)?;
            if upper_first {
                phased_fallback_selection(fallback)
            } else {
                fallback.map(|(nodes, _)| model_strong::Selection::fresh(nodes))
            }
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
            // Objective-dependent exclusions only remove solutions costing at
            // least the incumbent, so this floor holds for every child even
            // when its fork dropped incumbent-justified warm bounds.
            let floor = bound.min(result.incumbent);
            for (side, child) in children.iter_mut().enumerate() {
                child.state.node_depth = output.snapshot.state.node_depth.saturating_add(1);
                child.state.mapped_lineage = lineages[side];
                if floor.is_finite() {
                    child.state.tree_bound_floor = child.state.tree_bound_floor.max(floor);
                }
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
