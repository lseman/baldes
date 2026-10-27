//! Collect complete or cost-bounded route universes under shared budgets.
use super::*;

pub(super) fn collect_routes<const D: usize>(
    state: &State<D>,
    options: &EnumerationOptions,
) -> std::result::Result<(Vec<NetworkRoute>, Vec<(i32, usize)>), String> {
    collect_routes_with_plan(state, options, None)
}
pub(super) fn collect_routes_with_plan<const D: usize>(
    state: &State<D>,
    options: &EnumerationOptions,
    plan: Option<&[super::super::bounds::NetworkCutoff]>,
) -> std::result::Result<(Vec<NetworkRoute>, Vec<(i32, usize)>), String> {
    let deadline = Instant::now()
        .checked_add(options.time_limit)
        .ok_or("enumeration deadline overflow")?
        .min(state.deadline);
    let mut paths_left = options.max_routes;
    let mut work_left = options.max_work;
    let mut arcs_left = options.max_stored_arcs;
    let mut control_iterations = 100_000;
    let mut routes = vec![];
    let mut counts = vec![];
    for (network_index, oracle) in state.oracles.iter().enumerate() {
        if Instant::now() >= deadline {
            return Err("TimeLimit".into());
        }
        if oracle.multiplicity_upper() == 0. {
            counts.push((oracle.network.id, 0));
            continue;
        }
        let cutoff = plan.map(|p| &p[network_index]);
        if let Some(super::super::bounds::NetworkCutoff::Empty { network }) = cutoff {
            if *network != oracle.network.id {
                return Err("cutoff network mismatch".into());
            }
            counts.push((oracle.network.id, 0));
            continue;
        }
        // Replacing a control cannot preserve an identity-specific cap on
        // that column. Invariant Gamma needs no such dominance argument.
        let objective_projection = !state
            .route_upper_bounds
            .keys()
            .any(|&tag| state.routes[tag].network == oracle.network.id);
        let domain = super::super::resource_groups::residual_domain_graph(oracle)
            .map_err(|e| e.to_string())?;
        let graph = domain.as_ref();
        // Disposable transitions relax scalar equality transitions: clamping a
        // too-small load adds paths, never removes an equality-feasible path.
        // Collect that superset, then replay every path on the original graph.
        // Coupled restrictions are also relaxed only in this outer path
        // superset. Every candidate is certified and repriced against `graph`
        // below before the atomic transfer. The scalar collectors themselves
        // retain their explicit unsupported-coupled proof contract.
        let relaxed = (graph.has_coupled_resources()
            || graph
                .resources
                .iter()
                .any(|r| *r == crate::graph::ResourceKind::Equality))
        .then(|| -> Result<_> {
            let mut relaxed = graph.clone();
            relaxed
                .resources
                .fill(crate::graph::ResourceKind::Disposable);
            relaxed.set_coupled_resources(&[])?;
            Ok(relaxed)
        })
        .transpose()
        .map_err(|e| e.to_string())?;
        // A cost plan includes a directed route-wide coupled-objective lower
        // bound in its constant. The scalar superset can therefore prune by
        // its arc ceiling even when original resource charges are signed.
        let limits = PathDomainLimits {
            max_paths: paths_left,
            max_work: work_left,
            max_stored_arcs: arcs_left,
            deadline: Some(deadline),
        };
        let full_domain;
        let pruned_domain;
        let costs;
        let (paths, work_used, stored_arcs) = match cutoff {
            Some(super::super::bounds::NetworkCutoff::Cost {
                network,
                arc_cost_lower_bounds,
                arc_ceiling,
                ..
            }) => {
                if *network != oracle.network.id {
                    return Err("cutoff network mismatch".into());
                }
                costs = graph
                    .arcs
                    .iter()
                    .map(|a| arc_cost_lower_bounds[oracle.index[&a.id]])
                    .collect::<Vec<_>>();
                pruned_domain = CostBoundedResourceRelaxation::collect(
                    relaxed.as_ref().unwrap_or(graph),
                    &costs,
                    *arc_ceiling,
                    limits,
                )
                .map_err(|e| e.to_string())?;
                (
                    pruned_domain.paths(),
                    pruned_domain.work_used(),
                    pruned_domain.stored_arcs(),
                )
            }
            None => {
                full_domain =
                    CompleteResourceRelaxation::collect(relaxed.as_ref().unwrap_or(graph), limits)
                        .map_err(|e| e.to_string())?;
                (
                    full_domain.paths(),
                    full_domain.work_used(),
                    full_domain.stored_arcs(),
                )
            }
            Some(super::super::bounds::NetworkCutoff::Empty { .. }) => unreachable!(),
        };
        paths_left -= paths.len();
        work_left -= work_used;
        arcs_left -= stored_arcs;
        let problem = crate::master::routing::RoutingContext::set_partitioning(graph.clone(), 1.)
            .map_err(|e| e.to_string())?;
        let before = routes.len();
        for path in paths {
            if Instant::now() >= deadline {
                return Err("TimeLimit".into());
            }
            work_left = work_left
                .checked_sub(path.arc_positions().len() + 1)
                .ok_or("WorkLimit")?;
            let ids: Vec<_> = path
                .arc_positions()
                .iter()
                .map(|&a| graph.arcs[a].id)
                .collect();
            let mut probe = Route {
                coupled_controls: Vec::new(),
                coupled_values: Vec::new(),
                nodes: std::iter::once(graph.source as i32)
                    .chain(
                        path.arc_positions()
                            .iter()
                            .map(|&a| graph.arcs[a].head as i32),
                    )
                    .collect(),
                arcs: ids,
                cost: 0.,
                original_cost: 0.,
                resources: vec![],
                resource_profile: vec![],
            };
            let mut controls_complete = true;
            if super::super::control_pricing::needed(oracle) {
                match crate::master::routing::exact_scalar_path_feasibility(&problem, &probe) {
                    Some(false) => continue,
                    Some(true) => {}
                    None => return Err("Arithmetic: exact scalar replay is unknown".into()),
                }
            }
            let positions: Vec<_> = probe.arcs.iter().map(|id| oracle.index[id]).collect();
            if super::super::control_pricing::needed(oracle) {
                let controls = super::super::control_pricing::enumeration_controls(
                    oracle,
                    &positions,
                    &mut work_left,
                    &mut control_iterations,
                    deadline,
                    objective_projection,
                    &state.shared_rows,
                )
                .ok_or(
                    "paired control domain is not proved invariant or objective-dominated; path enumeration is incomplete",
                )?;
                let Some(controls) = controls else {
                    continue;
                };
                controls_complete = controls.complete;
                probe.coupled_controls = controls.controls;
            }
            match crate::master::routing::exact_route_feasibility_with_budget(
                &problem,
                &probe,
                &mut work_left,
                deadline,
            ) {
                Some(false) => continue,
                Some(true) => {}
                None => return Err("Arithmetic: exact resource replay is unknown".into()),
            }
            if !oracle
                .solver
                .route_satisfies_relaxation(&probe.nodes, &probe.arcs)
                .map_err(|e| e.to_string())?
            {
                continue;
            }
            if !controls_complete {
                return Err(
                    "paired control domain is not proved invariant or objective-dominated; path enumeration is incomplete"
                        .into(),
                );
            }
            // Never silently omit an exact-feasible path whose rounded forward
            // schedule cannot be reconstructed by the current scalar kernel.
            let mut canonical = problem
                .elementary_route(&probe.arcs)
                .ok_or("Arithmetic: canonical resource replay")?;
            let controlled_cost = if probe.coupled_controls.is_empty() {
                None
            } else {
                let selection = oracle
                    .network
                    .graph
                    .coupled_kernel()
                    .ok_or("missing control kernel")?
                    .replay_with_controls_using(
                        &positions,
                        &probe.coupled_controls,
                        &mut work_left,
                        Some(deadline),
                    )
                    .map_err(|e| e.to_string())?
                    .ok_or("Arithmetic: controlled resource replay")?;
                canonical.coupled_controls = std::mem::take(&mut probe.coupled_controls);
                canonical.coupled_values = selection.paired_values;
                Some(selection.cost)
            };
            probe.resources = canonical.resources;
            probe.resource_profile = canonical.resource_profile;
            probe.coupled_controls = canonical.coupled_controls;
            probe.coupled_values = canonical.coupled_values;
            let mut ids = probe.arcs.clone();
            ids.sort_unstable();
            probe.original_cost = ids
                .iter()
                .map(|id| oracle.network.graph.arcs[oracle.index[id]].cost)
                .sum::<f64>()
                + match controlled_cost {
                    Some(cost) => cost,
                    None => probe
                        .coupled_cost(
                            &oracle.network.graph,
                            &probe
                                .arcs
                                .iter()
                                .map(|id| oracle.index[id])
                                .collect::<Vec<_>>(),
                        )
                        .map_err(|e| e.to_string())?
                        .ok_or("infeasible coupled path")?,
                };
            probe.cost = probe.original_cost;
            if !probe.cost.is_finite() {
                return Err("Arithmetic: route cost".into());
            }
            routes.push(NetworkRoute {
                network: oracle.network.id,
                route: probe,
            });
        }
        counts.push((oracle.network.id, routes.len() - before));
    }
    if Instant::now() >= deadline {
        return Err("TimeLimit".into());
    }
    Ok((routes, counts))
}
