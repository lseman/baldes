//! Initial residual construction by shifted-cost subproblem solves.
use super::*;

fn shift(costs: impl IntoIterator<Item = f64>) -> f64 {
    let mut maximum: f64 = -1e12;
    for cost in costs {
        let margin = cost.abs().max(maximum.abs()) * 1e-10 + 1e-6;
        if maximum < cost - margin {
            maximum = cost;
        }
    }
    let margin = maximum.abs() * 1e-10 + 1e-6;
    if maximum < -margin {
        0.
    } else if maximum - margin <= 0. {
        1.
    } else {
        maximum * 1.1
    }
}

struct ShiftedCosts {
    arcs: Vec<f64>,
    activities: Vec<(usize, f64)>,
}

fn costs<const D: usize>(node: &NodeSnapshot<D>, index: usize) -> Result<ShiftedCosts> {
    let oracle = &node.state.oracles[index];
    let mut activities = BTreeMap::<usize, f64>::new();
    let mut costs: Vec<_> = oracle.network.graph.arcs.iter().map(|a| a.cost).collect();
    if let Some(model) = node
        .state
        .original_variable_costs
        .iter()
        .find(|m| m.network == oracle.network.id)
    {
        // Original variable associations, rather than physical arc count, define
        // the shift when an explicit original-variable registry is available.
        let delta = shift(model.variables.iter().map(|v| v.cost));
        for variable in &model.variables {
            if let Some(resource) = variable.accumulated_resource {
                // The raw paired solver ignores the owner coefficient, as in
                // ordinary pricing. Deterministic activities honor the shift.
                if !activity_rows::paired(&oracle.network.graph, resource) {
                    *activities.entry(resource).or_default() -= delta;
                }
                continue;
            }
            for &(id, coefficient) in &variable.arcs {
                costs[oracle.index[&id]] -= delta * coefficient;
            }
        }
    } else {
        let delta = shift(costs.iter().copied());
        for cost in &mut costs {
            *cost -= delta;
        }
    }
    if costs.iter().any(|v| !v.is_finite()) {
        return Err(incomplete("greedy shifted cost overflow"));
    }
    Ok(ShiftedCosts {
        arcs: costs,
        activities: activities.into_iter().collect(),
    })
}

fn quantity<const D: usize>(node: &NodeSnapshot<D>, index: usize, route: &Route) -> Result<f64> {
    let oracle = &node.state.oracles[index];
    let upper = oracle.multiplicity_upper();
    let mut cap = (upper + upper * 1e-10 + 1e-6).floor().min(i32::MAX as f64);
    for (row, coefficient) in entries(oracle, route, node.state.shared_offset)? {
        if node.state.retired_rows.contains(&row) {
            continue;
        }
        let sense = node.master.senses[row];
        let restrict = match sense {
            RowSense::Equal => coefficient != 0.,
            RowSense::LessEqual => coefficient > 0.,
            RowSense::GreaterEqual => coefficient < 0.,
        };
        if restrict {
            cap = cap.min((node.state.rhs[row] / coefficient).floor());
        }
    }
    Ok(cap.max(0.))
}

fn snapshot<const D: usize>(
    master: ClpMaster,
    state: State<D>,
    options: &mut Options,
) -> NodeSnapshot<D> {
    let mut warm = cg::WarmStart::empty(options.cg.bound_mode);
    warm.initial_rows = master.rhs.len();
    warm.columns = std::mem::take(&mut options.cg.known)
        .into_iter()
        .map(Some)
        .collect();
    NodeSnapshot {
        master,
        state,
        warm: warm.into(),
    }
}

/// No heuristic bound is returned. Only full original-model witness replay can
/// publish a primal, and all pricing/fixing takes place on private snapshots.
pub(super) fn run<const D: usize>(
    master: &ClpMaster,
    state: &State<D>,
    configured: &Options,
    warm: Option<&RetainedWarmStart>,
    options: &ModelTreeOptions,
    result: &mut ModelTreeOutput,
    deadline: Instant,
) -> Result<()> {
    let mut config = configured.clone();
    let mut current = if let Some(warm) = warm {
        NodeSnapshot {
            master: master.try_clone()?,
            state: state.fork_state(),
            warm: warm.clone(),
        }
    } else {
        snapshot(master.try_clone()?, state.fork_state(), &mut config)
    };
    current.state.deadline = deadline;
    let shifted = (0..state.oracles.len())
        .map(|i| costs(&current, i))
        .collect::<Result<Vec<_>>>();
    let shifted = match shifted {
        Ok(v) => v,
        Err(e) if e.kind == crate::core::error::ErrorKind::Runtime => return Ok(()),
        Err(e) => return Err(e),
    };
    let mut cursor = 0;
    while result.greedy_nodes < options.node_limit && Instant::now() < deadline {
        current = current.preprocess_with_options(&options.node.preprocessing)?;
        result.greedy_nodes += 1;
        if current.state.preprocessing.propagation.infeasible {
            break;
        }
        let mut values: Vec<_> = current
            .state
            .master_variables
            .iter()
            .map(|v| (v.id, 0.))
            .collect();
        let routes = current.state.fixed_routes.clone();
        if let Some(objective) = validate_routes(&current.state, &routes, &mut values)? {
            if objective < result.incumbent {
                result.incumbent = objective;
                result.routes = routes;
                result.master_values = values;
            }
        }
        if let Some(mip) = &options.restricted_master_mip {
            let config = crate::master::solvers::mip::MipOptions {
                node_limit: mip.node_limit,
                time_limit: Duration::try_from_secs_f64(mip.time_limit)
                    .unwrap_or_default()
                    .min(
                        deadline
                            .saturating_duration_since(Instant::now())
                            .mul_f64(0.98),
                    ),
                objective_cutoff: result.incumbent,
                deluxing: None,
            };
            result.restricted_mip_calls += 1;
            if let Ok(Some((objective, routes, values))) = current.solve_restricted_mip(&config) {
                if objective < result.incumbent {
                    result.incumbent = objective;
                    result.routes = routes;
                    result.master_values = values;
                }
            }
        }
        let count = current.state.oracles.len();
        if count == 0 {
            break;
        }
        let index = (0..count)
            .map(|offset| (cursor + offset) % count)
            .find(|&i| current.state.oracles[i].multiplicity_upper() > 1e-6);
        let Some(index) = index else {
            break;
        };
        let duals = vec![0.; current.state.rhs.len()];
        // A retained pricing pool can disable heuristic phases or carry old
        // objective exclusions. Rebuild private labels in the original domain;
        // hard arc/branch restrictions survive this cache restoration.
        current.state.oracles[index].solver.restore_arcs();
        super::super::pricing::cache_duals(&mut current.state, index, &duals)?;
        let oracle = &mut current.state.oracles[index];
        // Complete retained domains can bypass ordinary dual preparation.
        // Greedy pricing uses only its shifted variable objective, including
        // when resuming a snapshot with previously priced nonlinear rows.
        for cut in &mut oracle.solver.cuts.cuts {
            cut.dual = 0.;
        }
        for cut in &mut oracle.solver.cuts.knapsack_cuts {
            cut.dual = 0.;
        }
        if oracle
            .solver
            .presence_rows()
            .iter()
            .any(|row| row.dual != 0.)
        {
            oracle
                .solver
                .set_presence_duals(&vec![0.; oracle.solver.presence_rows().len()])?;
        }
        oracle
            .solver
            .cache_physical_costs(&shifted[index].arcs, &shifted[index].arcs)?;
        oracle
            .solver
            .graph
            .set_coupled_pricing_adjustments(&shifted[index].activities)?;
        oracle.solver.controls.time_limit = deadline.saturating_duration_since(Instant::now());
        result.pricing_calls += 1;
        if let Some((_, calls)) = result
            .oracle_calls
            .iter_mut()
            .find(|(id, _)| *id == oracle.network.id)
        {
            *calls += 1;
        }
        // The subproblem supplies its best solution; additional callback-
        // recorded solutions have no implicit counterpart in this solver.
        let priced = match oracle.solver.price_phase_partial(0, 1, false) {
            Ok(priced) => priced,
            Err(error)
                if error.message.contains("time limit")
                    || error.message.contains("label limit") =>
            {
                break
            }
            Err(error) => return Err(error),
        };
        let network = oracle.network.id;
        let mut selected = vec![];
        for route in priced.columns {
            if !lifecycle::integer_route_feasible(&current.state.oracles[index], &route)? {
                continue;
            }
            let quantity = quantity(&current, index, &route)?;
            if quantity > 0. {
                selected.push((NetworkRoute { network, route }, quantity));
            }
        }
        if selected.is_empty() {
            break;
        }
        for (seed, _) in &selected {
            let identity = seed.route.identity();
            let key = (network, identity.clone());
            let tag = if let Some(&tag) = current.state.tags.get(&key) {
                tag
            } else {
                let tag = i32::try_from(current.state.routes.len())
                    .map_err(|_| invalid("greedy route tag overflow"))?;
                let mut route = seed.clone();
                route.route.original_cost =
                    original_costs::route_cost(&current.state.oracles[index], &route.route)?;
                Arc::make_mut(&mut current.state.routes).push(Arc::new(route));
                Arc::make_mut(&mut current.state.tags).insert(key, tag);
                tag
            };
            if !current
                .warm
                .columns
                .iter()
                .flatten()
                .any(|column| column.tag == tag)
            {
                let route = &current.state.routes[tag as usize];
                let coefficients = retirement::active_entries(
                    &current.state,
                    &current.state.oracles[index],
                    &route.route,
                )?;
                let cost = route.route.original_cost;
                let column = current.master.add_column(cost, &coefficients)?;
                debug_assert_eq!(column, current.warm.columns.len());
                current.warm.columns.push(Some(
                    cg::Column {
                        cost,
                        reduced_cost: 0.,
                        entries: coefficients,
                        acceptable: true,
                        tag,
                        identity: format!("{network}:{identity}"),
                    }
                    .into(),
                ));
            }
        }
        let mut child = current.fork()?;
        let mut valid = true;
        for (route, quantity) in selected {
            let tag = child.state.tags[&(network, route.route.identity())] as usize;
            match child.fix_route(tag, quantity) {
                Ok(next) => child = next,
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            break;
        }
        child.state.node_depth = current.state.node_depth.saturating_add(1);
        current = child;
        cursor = (index + 1) % count;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/tree/greedy_tests.rs"]
mod tests;
