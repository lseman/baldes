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
        if let Some(super::super::bounds::NetworkCutoff::Cost { network, .. }) = cutoff {
            if *network != oracle.network.id {
                return Err("cutoff network mismatch".into());
            }
        }
        let domain = super::super::resource_groups::residual_domain_graph(oracle)
            .map_err(|e| e.to_string())?;
        let graph = domain.as_ref();
        if graph
            .coupled_kernel()
            .is_some_and(|kernel| kernel.machine_paired_join_complete())
        {
            let (collected, candidate_count, generated, stored_arcs) =
                collect_paired_program_routes(
                    oracle, graph, cutoff, paths_left, work_left, arcs_left, deadline,
                )?;
            paths_left = paths_left.checked_sub(candidate_count).ok_or("PathLimit")?;
            work_left = work_left.checked_sub(generated).ok_or("WorkLimit")?;
            arcs_left = arcs_left.checked_sub(stored_arcs).ok_or("StoredArcLimit")?;
            counts.push((oracle.network.id, collected.len()));
            for mut route in collected {
                route.original_cost = super::super::original_costs::route_cost(oracle, &route)
                    .map_err(|error| error.to_string())?;
                route.cost = route.original_cost;
                routes.push(NetworkRoute {
                    network: oracle.network.id,
                    route,
                });
            }
            continue;
        }
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
            // Never silently omit an exact-feasible path whose rounded forward
            // schedule cannot be reconstructed by the current scalar kernel.
            let canonical = problem
                .elementary_route(&probe.arcs)
                .ok_or("Arithmetic: canonical resource replay")?;
            probe.resources = canonical.resources;
            probe.resource_profile = canonical.resource_profile;
            probe.coupled_values = canonical.coupled_values;
            let mut ids = probe.arcs.clone();
            ids.sort_unstable();
            probe.original_cost = ids
                .iter()
                .map(|id| oracle.network.graph.arcs[oracle.index[id]].cost)
                .sum::<f64>()
                + probe
                    .coupled_cost(
                        &oracle.network.graph,
                        &probe
                            .arcs
                            .iter()
                            .map(|id| oracle.index[id])
                            .collect::<Vec<_>>(),
                    )
                    .map_err(|e| e.to_string())?
                    .ok_or("infeasible coupled path")?;
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

fn collect_paired_program_routes<const D: usize>(
    oracle: &Oracle<D>,
    graph: &Graph<D>,
    cutoff: Option<&super::super::bounds::NetworkCutoff>,
    max_paths: usize,
    max_work: usize,
    max_arcs: usize,
    deadline: Instant,
) -> std::result::Result<(Vec<Route>, usize, usize, usize), String> {
    if max_work == 0 {
        return Err("WorkLimit".into());
    }
    let mut solver = crate::pricing::Solver::<D>::new(graph.vertices.len(), graph.resources.len())
        .map_err(|error| error.to_string())?;
    solver.graph = graph.clone();
    for arc in &mut solver.graph.arcs {
        arc.reduced_cost = 0.;
    }
    solver.graph.suppress_coupled_costs();
    solver.enumeration_preserve_arc_paths = true;
    solver.controls.max_labels = max_work;
    solver.controls.time_limit = deadline.saturating_duration_since(Instant::now());
    solver
        .options
        .set("BALDESmaxNumOfLabelsInEnumeration", max_work as f64)
        .map_err(|error| error.to_string())?;
    solver
        .options
        .set("BALDESmaxNumOfEnumeratedSolutions", max_paths.max(1) as f64)
        .map_err(|error| error.to_string())?;
    let candidates = solver
        .enumerate(f64::MAX)
        .map_err(|error| error.to_string())?;
    if Instant::now() >= deadline {
        return Err("TimeLimit".into());
    }
    let candidate_count = candidates.len();
    if candidate_count > max_paths {
        return Err("PathLimit".into());
    }
    let enumeration = solver
        .enumeration
        .as_ref()
        .ok_or("paired enumeration unavailable")?;
    if !enumeration.completed {
        return Err("paired enumeration incomplete".into());
    }
    let mut work = enumeration.generated;
    let mut stored_arcs = 0usize;
    let problem = crate::master::routing::RoutingContext::set_partitioning(graph.clone(), 1.)
        .map_err(|error| error.to_string())?;
    let mut routes = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if Instant::now() >= deadline {
            return Err("TimeLimit".into());
        }
        stored_arcs = stored_arcs
            .checked_add(candidate.arcs.len())
            .ok_or("StoredArcLimit")?;
        if stored_arcs > max_arcs {
            return Err("StoredArcLimit".into());
        }
        work = work
            .checked_add(candidate.arcs.len().checked_add(1).ok_or("WorkLimit")?)
            .ok_or("WorkLimit")?;
        if work > max_work {
            return Err("WorkLimit".into());
        }
        if let Some(super::super::bounds::NetworkCutoff::Cost {
            arc_cost_lower_bounds,
            arc_ceiling,
            ..
        }) = cutoff
        {
            let mut lower = Some(0.);
            for id in &candidate.arcs {
                let index = *oracle.index.get(id).ok_or("unknown enumeration arc")?;
                let value = *arc_cost_lower_bounds
                    .get(index)
                    .ok_or("missing enumeration cost bound")?;
                lower = lower.and_then(|sum| crate::core::rounding::add_down(sum, value));
            }
            if lower.is_some_and(|value| value > *arc_ceiling) {
                continue;
            }
        }
        if !oracle
            .solver
            .route_satisfies_relaxation(&candidate.nodes, &candidate.arcs)
            .map_err(|error| error.to_string())?
        {
            continue;
        }
        let mut remaining = max_work - work;
        match crate::master::routing::exact_route_feasibility_with_budget(
            &problem,
            &candidate,
            &mut remaining,
            deadline,
        ) {
            Some(false) => {
                work = max_work - remaining;
                continue;
            }
            Some(true) => {}
            None => return Err("Arithmetic: paired front replay is unknown".into()),
        }
        work = max_work - remaining;
        let route = problem
            .elementary_route(&candidate.arcs)
            .ok_or("Arithmetic: paired route reconstruction")?;
        routes.push(route);
    }
    if Instant::now() >= deadline {
        return Err("TimeLimit".into());
    }
    Ok((routes, candidate_count, work, stored_arcs))
}
