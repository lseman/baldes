//! Whole-domain arc-flow relaxation; never inferred from restricted columns.
use super::*;
mod links;
use crate::master::preprocessing::{propagate_mixed_conservative, Bounds, Constraint};

use crate::master::preprocessing::flow::row;

pub(super) fn apply<const D: usize>(
    master: &mut ClpMaster,
    state: &mut State<D>,
    loaded: &[(usize, usize)],
    work_limit: usize,
    integer_domains: bool,
) -> Result<PreprocessingReport> {
    let mut report = PreprocessingReport::default();
    if !bounds::exact_fixed_reduction(state, master)? {
        report.skipped = Some("RoundedFixedReduction".into());
        return Ok(report);
    }
    let build_work = state.oracles.iter().fold(0usize, |work, oracle| {
        work.saturating_add(
            oracle
                .presence_rows
                .len()
                .saturating_mul(state.shared_rows.len())
                .saturating_mul(2),
        )
        .saturating_add(oracle.network.graph.arcs.len().saturating_mul(4))
        .saturating_add(oracle.network.graph.vertices.len())
        .saturating_add(
            oracle
                .network
                .arc_entries
                .iter()
                .map(Vec::len)
                .sum::<usize>(),
        )
    });
    if build_work > work_limit {
        report.skipped = Some("WorkLimit".into());
        return Ok(report);
    }
    let mut limits: Vec<_> = state.master_variables.iter().map(|v| v.bounds).collect();
    let mut starts = vec![];
    let mut raw = vec![vec![]; state.shared_rows.len()];
    let mut constraints = vec![];
    for (v, variable) in state.master_variables.iter().enumerate() {
        for &(r, a) in &variable.entries {
            raw[r].push((v, a));
        }
    }
    for oracle in &state.oracles {
        let start = limits.len();
        starts.push(start);
        let g = &oracle.network.graph;
        limits.extend(g.arcs.iter().enumerate().map(|(a, _)| {
            Bounds {
                lower: 0.,
                upper: if oracle
                    .solver
                    .forbidden_arcs
                    .get(a)
                    .copied()
                    .unwrap_or(false)
                {
                    0.
                } else {
                    f64::INFINITY
                },
            }
        }));
        let m = limits.len();
        limits.push(Bounds {
            lower: oracle.network.multiplicity.lower,
            upper: oracle.multiplicity_upper(),
        });
        for (a, entries) in oracle.network.arc_entries.iter().enumerate() {
            for &(r, c) in entries {
                raw[r].push((start + a, c));
            }
        }
        for &(r, c) in oracle.network.route_entries.iter() {
            raw[r].push((m, c));
        }
        constraints.extend(crate::master::preprocessing::flow::conservation(
            g,
            start,
            Some(m),
        ));
    }
    // Independent owner coordinates retain registration/row propagation without
    // projecting a bound onto individual machine-selected path values.
    let mut paired_coordinates = BTreeMap::new();
    let mut owner_states = BTreeMap::new();
    for (position, oracle) in state.oracles.iter().enumerate() {
        let Some(kernel) = oracle.network.graph.coupled_kernel() else {
            continue;
        };
        let resources: std::collections::BTreeSet<_> = oracle
            .activity_rows
            .iter()
            .map(|entry| entry.resource)
            .chain(
                kernel
                    .paired_associated_costs()
                    .iter()
                    .map(|&(resource, _)| resource),
            )
            .chain(
                kernel
                    .paired_owner_registration_bounds()
                    .iter()
                    .map(|&(resource, _)| resource),
            )
            .filter(|&resource| super::activity_rows::paired(&oracle.network.graph, resource))
            .collect();
        for resource in resources {
            let input = kernel
                .paired_owner_registration_bounds()
                .iter()
                .find(|&&(index, _)| index == resource)
                .map(|&(_, bounds)| bounds);
            let mut fixed_upper = crate::master::numerics::ExactSum::default();
            let mut negative_fixed_lower = crate::master::numerics::ExactSum::default();
            for fixed in state
                .fixed_routes
                .iter()
                .filter(|fixed| fixed.route.network == oracle.network.id)
            {
                let arcs = fixed
                    .route
                    .route
                    .arcs
                    .iter()
                    .map(|id| {
                        oracle
                            .index
                            .get(id)
                            .copied()
                            .ok_or_else(|| invalid("unknown fixed owner activity arc"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let value = super::activity_rows::column_value(
                    oracle,
                    &fixed.route.route,
                    &arcs,
                    resource,
                )?;
                if fixed_upper.product(value, fixed.quantity).is_none()
                    || negative_fixed_lower
                        .product(-value, fixed.quantity)
                        .is_none()
                {
                    return Err(invalid("fixed owner activity overflow"));
                }
            }
            let fixed = [
                -negative_fixed_lower
                    .upper()
                    .ok_or_else(|| invalid("fixed owner activity overflow"))?,
                fixed_upper
                    .upper()
                    .ok_or_else(|| invalid("fixed owner activity overflow"))?,
            ];
            let quantity = starts[position] + oracle.network.graph.arcs.len();
            let owner = crate::master::preprocessing::paired_owner::OwnerBounds::new(
                input,
                limits[quantity],
                fixed,
            );
            let index = limits.len();
            limits.push(owner.aggregate);
            owner_states.insert((oracle.network.id, resource), (owner, quantity));
            paired_coordinates.insert((oracle.network.id, resource), index);
            for entry in oracle
                .activity_rows
                .iter()
                .filter(|entry| entry.resource == resource)
            {
                raw[entry.row].push((index, entry.coefficient));
            }
        }
    }
    let shared_start = constraints.len();
    let mut equalities = vec![];
    for (r, shared) in state.shared_rows.iter().enumerate() {
        // Signed nonlinear activities have no additive flow envelope here.
        // Dropping this entire row enlarges the domain; omitting only the
        // activity term could incorrectly prove infeasibility or fix arcs.
        if state.oracles.iter().any(|o| {
            o.activity_rows.iter().any(|entry| {
                entry.row == r && !paired_coordinates.contains_key(&(o.network.id, entry.resource))
            })
        }) {
            continue;
        }
        let nonlinear = state
            .oracles
            .iter()
            .any(|o| o.nonlinear_rows.iter().any(|&(_, row)| row == r));
        if shared.sense == RowSense::Equal
            && raw[r].len() > 1
            && raw[r]
                .iter()
                .any(|&(v, _)| v < state.master_variables.len())
            && !state.oracles.iter().any(|o| {
                o.nonlinear_rows
                    .iter()
                    .chain(&o.knapsack_rows)
                    .chain(&o.presence_rows)
                    .any(|&(_, row)| row == r)
            })
        {
            if let Some(equality) = row(
                raw[r].clone(),
                RowSense::Equal,
                state.rhs[state.shared_offset + r],
            ) {
                equalities.push(equality);
            }
        }
        // For a >= row, use the registered kernel's proved increment upper
        // envelope, including arc weights and limited-memory resets. For a <=
        // row, omit it, a lower envelope. Equality requires both relaxations.
        for sense in [RowSense::LessEqual, RowSense::GreaterEqual] {
            if shared.sense != RowSense::Equal && shared.sense != sense {
                continue;
            }
            let mut terms = raw[r].clone();
            if nonlinear && sense == RowSense::GreaterEqual {
                for (oracle, &start) in state.oracles.iter().zip(&starts) {
                    for &(cut, row_id) in &oracle.nonlinear_rows {
                        if row_id == r {
                            for a in 0..oracle.network.graph.arcs.len() {
                                let bound = super::nonlinear::arc_upper(oracle, cut, a);
                                if bound > 0. {
                                    terms.push((start + a, bound));
                                }
                            }
                        }
                    }
                }
            }
            if sense == RowSense::GreaterEqual {
                for (oracle, &start) in state.oracles.iter().zip(&starts) {
                    for &(input, row_id) in &oracle.presence_rows {
                        if row_id == r {
                            terms.push((
                                start + oracle.network.graph.arcs.len(),
                                oracle.solver.presence_rows()[input]
                                    .row
                                    .maximum_coefficient() as f64,
                            ));
                        }
                    }
                }
            }
            let Some(c) = row(terms, sense, state.rhs[state.shared_offset + r]) else {
                report.skipped = Some("RoundedFlowCoefficients".into());
                return Ok(report);
            };
            constraints.push(c);
        }
    }
    let mut integers = vec![integer_domains; limits.len()];
    for (i, v) in state.master_variables.iter().enumerate() {
        integers[i] = integer_domains && v.integer;
    }
    for &index in paired_coordinates.values() {
        integers[index] = false;
    }
    let mut remaining = work_limit - build_work;
    let (derived, link_work, link_complete) = links::derive(
        &constraints[shared_start..],
        &equalities,
        (remaining / 4).min(100_000),
        state.deadline,
    );
    constraints.extend(derived);
    remaining -= link_work;
    report.propagation.work_used = build_work + link_work;
    let resources = state.oracles.iter().any(|o| {
        !o.knapsack_rows.is_empty()
            || !o.presence_rows.is_empty()
            || !o.nonlinear_rows.is_empty()
            || !o.activity_rows.is_empty()
    });
    loop {
        if remaining == 0 || Instant::now() >= state.deadline {
            break;
        }
        let next = propagate_mixed_conservative(&mut limits, &constraints, &integers, remaining)?;
        remaining -= next.work_used;
        report.propagation.work_used += next.work_used;
        report.propagation.bound_updates += next.bound_updates;
        report.propagation.rows_processed += next.rows_processed;
        report.propagation.infeasible |= next.infeasible;
        if report.propagation.infeasible {
            report.propagation.complete = true;
            return Ok(report);
        }
        if !next.complete {
            break;
        }
        let mut owner_changed = false;
        for (identity, (owner, quantity)) in &mut owner_states {
            let index = paired_coordinates[identity];
            owner_changed |= owner.update(limits[index], limits[*quantity]);
            limits[index] = owner.aggregate;
        }
        if owner_changed {
            continue;
        }
        if !resources {
            report.propagation.complete = true;
            break;
        }
        let (zeros, complete, spent) =
            super::resource_rows::propagation::zero_arcs(state, &limits, &starts, remaining);
        remaining -= spent;
        report.propagation.work_used += spent;
        let mut changed = false;
        for v in zeros {
            if limits[v].lower > 0. {
                report.propagation.infeasible = true;
                report.propagation.complete = true;
                return Ok(report);
            }
            if limits[v].upper != 0. {
                limits[v].upper = 0.;
                report.propagation.bound_updates += 1;
                changed = true;
            }
        }
        if !complete {
            break;
        }
        if !changed {
            report.propagation.complete = true;
            break;
        }
        // A nonlinear zero may tighten static variables through the original
        // flow rows, enabling another shared nonlinear implication. Repeat
        // only with the unspent budget; no progress resets the work counter.
    }
    report.propagation.complete &= link_complete;
    for (&(network, resource), &index) in &paired_coordinates {
        report.paired_owner_bounds.push((
            network,
            resource,
            owner_states[&(network, resource)].0.local,
        ));
        report
            .paired_owner_aggregate_bounds
            .push((network, resource, limits[index]));
    }
    for (index, variable) in state.master_variables.iter_mut().enumerate() {
        let bound = limits[index];
        variable.bounds = bound;
        report.master_bounds.push((variable.id, bound));
        // Canonical signed components preserve every feasible original x:
        // choose max(x,0) and max(-x,0). Coefficients and cost depend only on
        // their difference, so removing cancellation does not change the model.
        for (side, sign) in [(0usize, 1.), (1, -1.)] {
            let column = 2 * index + side;
            let upper = super::master_variables::upper(variable, sign);
            if upper < master.upper[column] {
                master.set_upper(column, upper)?;
                report.tightened_columns += 1;
                report.fixed_columns += usize::from(upper == 0.);
            }
        }
    }
    for (oracle, &start) in state.oracles.iter_mut().zip(&starts) {
        let count = oracle.network.graph.arcs.len();
        let upper = limits[start + count].upper;
        if upper.is_finite() {
            oracle.proved_multiplicity_upper = Some(upper);
        }
        report.networks.push(NetworkActivityBounds {
            network: oracle.network.id,
            arcs: limits[start..start + count].to_vec(),
            multiplicity: limits[start + count],
        });
        let forbidden: Vec<_> = (0..oracle.network.graph.arcs.len())
            .filter(|&a| limits[start + a].upper == 0.)
            .collect();
        report.forbidden_arcs += forbidden
            .iter()
            .filter(|&&a| {
                !oracle
                    .solver
                    .forbidden_arcs
                    .get(a)
                    .copied()
                    .unwrap_or(false)
            })
            .count();
        oracle
            .proved_zero_arcs
            .resize(oracle.network.graph.arcs.len(), false);
        for &a in &forbidden {
            oracle.proved_zero_arcs[a] = true;
        }
        oracle.solver.set_forbidden_arcs(forbidden)?;
    }
    let mut disabled = HashSet::new();
    for &(column, tag) in loaded {
        let route = &state.routes[tag];
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == route.network)
            .unwrap();
        if route
            .route
            .arcs
            .iter()
            .any(|id| oracle.solver.forbidden_arcs[oracle.index[id]])
        {
            if master.upper[column] != 0. {
                master.set_upper(column, 0.)?;
                report.fixed_columns += 1;
                report.tightened_columns += 1;
            }
            disabled.insert(tag);
        }
    }
    for &tag in &disabled {
        state.route_upper_bounds.insert(tag, 0.);
    }
    if let Some(universe) = &mut state.universe {
        universe.retain(|tag| !disabled.contains(tag));
    }
    Ok(report)
}
