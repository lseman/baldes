// Cross-path witness recovery and realization of source controls.
use super::*;
/// Recover an original integer solution from fractional controls on integer
/// physical-path counts. Only registered, replayable source points are tried;
/// failure or budget exhaustion supplies no bound or infeasibility conclusion.
pub(in crate::master::model) fn repair_witness<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    metadata: impl Fn(usize) -> Option<i32>,
    tolerance: f64,
) -> Result<Option<IntegerWitness>> {
    if !master.solved() || Instant::now() >= state.deadline {
        return Ok(None);
    }
    let mut groups = BTreeMap::<(i32, Vec<i32>), (f64, bool, Vec<(usize, f64, FixedRoute)>)>::new();
    let mut masters: Vec<_> = state.master_variables.iter().map(|v| (v.id, 0.)).collect();
    for (column, &value) in master.values().iter().enumerate() {
        let tag = metadata(column);
        // Clp can leave tiny signed residuals on artificial/unmapped columns.
        // They are only proposal data: exact source replay must still satisfy
        // every original row without any artificial contribution.
        if tag.is_none() || tag == Some(-1) {
            if value.is_finite() && value.abs() <= tolerance {
                continue;
            }
            return Ok(None);
        }
        if !value.is_finite()
            || !(0. ..=9_007_199_254_740_992.).contains(&value)
            || value > master.upper[column]
        {
            return Ok(None);
        }
        if let Some((index, sign)) = tag.and_then(master_variables::decode) {
            let Some(variable) = state.master_variables.get(index) else {
                return Ok(None);
            };
            let quantity = if variable.integer {
                value.round()
            } else {
                value
            };
            if (value - quantity).abs() > tolerance || quantity > master.upper[column] {
                return Ok(None);
            }
            masters[index].1 += sign * quantity;
            continue;
        }
        if value <= tolerance {
            continue;
        }
        let Some(route_tag) = tag.and_then(|tag| usize::try_from(tag).ok()) else {
            return Ok(None);
        };
        let Some(route) = state.routes.get(route_tag) else {
            return Ok(None);
        };
        let group = groups
            .entry((route.network, route.route.arcs.clone()))
            .or_default();
        group.0 += value;
        group.1 |= (value - value.round()).abs() > tolerance;
        group.2.push((
            route_tag,
            value,
            FixedRoute {
                route: route.as_ref().clone(),
                quantity: value.round(),
            },
        ));
    }
    let mut witness = state.fixed_routes.clone();
    let mut allocation_groups: Vec<(Vec<(usize, f64, NetworkRoute)>, f64)> = vec![];
    for ((network, arcs), (quantity, fractional, lp_routes)) in groups {
        // Integer-tolerance path counts can still carry a fractional paired
        // control. Reconstruct source controls before trusting their rows.
        if !fractional
            && lp_routes
                .iter()
                .all(|(_, _, route)| route.route.route.coupled_controls.is_empty())
        {
            witness.extend(lp_routes.into_iter().map(|(_, _, route)| route));
            continue;
        }
        let rounded = quantity.round();
        if (quantity - rounded).abs() > tolerance
            || rounded <= 0.
            || rounded > 9_007_199_254_740_992.
        {
            return Ok(None);
        }
        let mut choices: Vec<_> = state
            .routes
            .iter()
            .enumerate()
            .filter(|(tag, route)| {
                route.network == network
                    && route.route.arcs == arcs
                    && state
                        .universe
                        .as_ref()
                        .map_or(true, |universe| universe.contains(tag))
                    && enumeration::route_upper(state, *tag).is_some_and(|cap| cap >= 1.)
            })
            .map(|(tag, route)| {
                let lp_quantity = lp_routes
                    .iter()
                    .filter(|(lp_tag, _, _)| *lp_tag == tag)
                    .map(|(_, value, _)| value)
                    .sum();
                (tag, lp_quantity, route.as_ref().clone())
            })
            .collect();
        // Try LP-supported identities first, then other registered points.
        // An empty control list is the registered minimum-front choice, so it
        // remains a valid identity candidate when a path count is split across
        // fractional LP columns.
        choices.sort_by(|a, b| (b.1 > 0.).cmp(&(a.1 > 0.)).then_with(|| a.0.cmp(&b.0)));
        allocation_groups.push((choices, rounded));
    }
    if allocation_groups.is_empty() {
        return Ok(None);
    }
    let mut routes = witness.clone();
    let now = Instant::now();
    let identity_search_deadline = now + state.deadline.saturating_duration_since(now) / 2;
    let mut recovery_work = 4_000_000;
    let mut recovery_iterations = 100_000;
    let joint_work_reserve = recovery_work / 2;
    let mut identity_work = recovery_work - joint_work_reserve;
    let identity_work_initial = identity_work;
    if let Some(result) = search_identity_groups(
        state,
        &allocation_groups,
        0,
        &mut routes,
        &masters,
        identity_search_deadline,
        &mut identity_work,
    )? {
        return Ok(Some(result));
    }
    recovery_work = recovery_work.saturating_sub(identity_work_initial - identity_work);
    Ok(joint_witness_targets(
        state,
        &witness,
        &allocation_groups,
        &masters,
        &mut recovery_work,
        &mut recovery_iterations,
    ))
}

/// Search registered identity allocations lazily. This avoids a fixed
/// identity shortlist and materializing the Cartesian product of all groups.
/// Exhausting the phase deadline is inconclusive.
pub(super) fn search_identity_groups<const D: usize>(
    state: &State<D>,
    groups: &[(Vec<(usize, f64, NetworkRoute)>, f64)],
    index: usize,
    routes: &mut Vec<FixedRoute>,
    masters: &[(i32, f64)],
    deadline: Instant,
    work: &mut usize,
) -> Result<Option<IntegerWitness>> {
    if Instant::now() >= deadline || *work == 0 {
        return Ok(None);
    }
    if index == groups.len() {
        if let Some(scheduled) = synchronization::witness(state, routes)? {
            if scheduled_identities_allowed(state, &scheduled[state.fixed_routes.len()..]) {
                if let Some(objective) = tree_adapter::validate_routes(state, &scheduled, masters)?
                {
                    return Ok(Some((objective, scheduled, masters.to_vec())));
                }
            }
        }
        return Ok(None);
    }

    let (choices, quantity) = &groups[index];
    let mut found = None;
    for_each_identity_allocation(
        state,
        choices,
        *quantity,
        deadline,
        work,
        &mut |allocation, deadline, work| {
            let old_len = routes.len();
            routes.extend_from_slice(allocation);
            found =
                search_identity_groups(state, groups, index + 1, routes, masters, deadline, work)?;
            routes.truncate(old_len);
            Ok(found.is_some())
        },
    )?;
    Ok(found)
}

/// Enumerate bounded integral allocations of one physical-path count across
/// registered route identities, respecting each identity's residual cap.
#[cfg(test)]
pub(super) fn identity_allocations<const D: usize>(
    state: &State<D>,
    choices: &[(usize, f64, NetworkRoute)],
    quantity: f64,
    limit: usize,
    deadline: Instant,
) -> Vec<Vec<FixedRoute>> {
    pub(super) fn visit<const D: usize>(
        state: &State<D>,
        choices: &[(usize, f64, NetworkRoute)],
        index: usize,
        remaining: f64,
        selected: &mut Vec<FixedRoute>,
        allocations: &mut Vec<Vec<FixedRoute>>,
        limit: usize,
        deadline: Instant,
    ) {
        if allocations.len() >= limit || Instant::now() >= deadline {
            return;
        }
        if remaining == 0. {
            allocations.push(selected.clone());
            return;
        }
        if index == choices.len() {
            return;
        }

        let (tag, preferred, route) = &choices[index];
        let Some(cap) = enumeration::route_upper(state, *tag) else {
            return;
        };
        let max_count = remaining.min(cap.floor());
        if max_count <= 0. {
            visit(
                state,
                choices,
                index + 1,
                remaining,
                selected,
                allocations,
                limit,
                deadline,
            );
            return;
        }

        // Try LP-near and cap-boundary counts first, then enumerate remaining
        // integral counts until the allocation limit or phase deadline.
        let preferred_floor = preferred.floor().clamp(0., max_count);
        let preferred_ceil = preferred.ceil().clamp(0., max_count);
        let mut ordered = vec![
            preferred_floor,
            preferred_ceil,
            max_count,
            max_count - 1.,
            1.,
            0.,
        ];
        ordered.retain(|value| *value >= 0. && *value <= max_count);
        let mut seen = Vec::new();
        ordered.retain(|value| {
            if seen.contains(value) {
                false
            } else {
                seen.push(*value);
                true
            }
        });
        let try_count = |count: f64,
                         selected: &mut Vec<FixedRoute>,
                         allocations: &mut Vec<Vec<FixedRoute>>,
                         deadline: Instant| {
            if allocations.len() >= limit || Instant::now() >= deadline {
                return;
            }
            if count > 0. {
                selected.push(FixedRoute {
                    route: route.clone(),
                    quantity: count,
                });
            }
            visit(
                state,
                choices,
                index + 1,
                remaining - count,
                selected,
                allocations,
                limit,
                deadline,
            );
            if count > 0. {
                selected.pop();
            }
        };
        for count in ordered {
            try_count(count, selected, allocations, deadline);
        }
        let mut count = 0.;
        while count <= max_count && allocations.len() < limit && Instant::now() < deadline {
            if !seen.contains(&count) {
                try_count(count, selected, allocations, deadline);
            }
            count += 1.;
        }
    }

    if !quantity.is_finite() || quantity <= 0. || quantity.fract() != 0. || limit == 0 {
        return Vec::new();
    }
    let mut allocations = Vec::new();
    visit(
        state,
        choices,
        0,
        quantity,
        &mut Vec::new(),
        &mut allocations,
        limit,
        deadline,
    );
    allocations
}

/// Stream bounded allocations under the shared work budget and phase deadline;
/// no Cartesian candidate list is materialized.
pub(super) fn for_each_identity_allocation<const D: usize>(
    state: &State<D>,
    choices: &[(usize, f64, NetworkRoute)],
    quantity: f64,
    deadline: Instant,
    work: &mut usize,
    on_allocation: &mut impl FnMut(&[FixedRoute], Instant, &mut usize) -> Result<bool>,
) -> Result<bool> {
    pub(super) fn try_count<const D: usize>(
        state: &State<D>,
        choices: &[(usize, f64, NetworkRoute)],
        index: usize,
        remaining: f64,
        count: f64,
        route: &NetworkRoute,
        selected: &mut Vec<FixedRoute>,
        deadline: Instant,
        work: &mut usize,
        on_allocation: &mut impl FnMut(&[FixedRoute], Instant, &mut usize) -> Result<bool>,
    ) -> Result<bool> {
        if count > 0. {
            selected.push(FixedRoute {
                route: route.clone(),
                quantity: count,
            });
        }
        let stop = visit(
            state,
            choices,
            index + 1,
            remaining - count,
            selected,
            deadline,
            work,
            on_allocation,
        )?;
        if count > 0. {
            selected.pop();
        }
        Ok(stop)
    }

    pub(super) fn visit<const D: usize>(
        state: &State<D>,
        choices: &[(usize, f64, NetworkRoute)],
        index: usize,
        remaining: f64,
        selected: &mut Vec<FixedRoute>,
        deadline: Instant,
        work: &mut usize,
        on_allocation: &mut impl FnMut(&[FixedRoute], Instant, &mut usize) -> Result<bool>,
    ) -> Result<bool> {
        let Some(remaining_work) = work.checked_sub(1) else {
            return Ok(true);
        };
        *work = remaining_work;
        if Instant::now() >= deadline {
            return Ok(false);
        }
        if remaining == 0. {
            return on_allocation(selected, deadline, work);
        }
        let Some((_, preferred, route)) = choices.get(index) else {
            return Ok(false);
        };
        let Some(cap) = enumeration::route_upper(state, choices[index].0) else {
            return Ok(false);
        };
        let max_count = remaining.min(cap.floor());
        if max_count <= 0. {
            return visit(
                state,
                choices,
                index + 1,
                remaining,
                selected,
                deadline,
                work,
                on_allocation,
            );
        }

        // Preserve LP-near and cap-boundary preference, then enumerate every
        // remaining integer count until the shared budget or deadline ends.
        let preferred_floor = preferred.floor().clamp(0., max_count);
        let preferred_ceil = preferred.ceil().clamp(0., max_count);
        let mut ordered = vec![
            preferred_floor,
            preferred_ceil,
            max_count,
            max_count - 1.,
            1.,
            0.,
        ];
        ordered.retain(|value| *value >= 0. && *value <= max_count);
        let mut seen = Vec::new();
        ordered.retain(|value| {
            if seen.contains(value) {
                false
            } else {
                seen.push(*value);
                true
            }
        });
        for count in ordered.iter().copied() {
            if try_count(
                state,
                choices,
                index,
                remaining,
                count,
                route,
                selected,
                deadline,
                work,
                on_allocation,
            )? {
                return Ok(true);
            }
        }
        let mut count = 0.;
        while count <= max_count && Instant::now() < deadline {
            if !seen.contains(&count)
                && try_count(
                    state,
                    choices,
                    index,
                    remaining,
                    count,
                    route,
                    selected,
                    deadline,
                    work,
                    on_allocation,
                )?
            {
                return Ok(true);
            }
            count += 1.;
        }
        Ok(false)
    }

    if !quantity.is_finite() || quantity <= 0. || quantity.fract() != 0. {
        return Ok(false);
    }
    visit(
        state,
        choices,
        0,
        quantity,
        &mut Vec::new(),
        deadline,
        work,
        on_allocation,
    )
}

// If registered points cannot realize the LP's integral physical path counts,
// solve all shared Gamma rows together, retaining fixed and integral background
// contributions. Each new terminal target still needs a source trajectory.
pub(super) fn joint_witness_targets<const D: usize>(
    state: &State<D>,
    background: &[FixedRoute],
    allocation_groups: &[(Vec<(usize, f64, NetworkRoute)>, f64)],
    masters: &[(i32, f64)],
    work: &mut usize,
    iterations: &mut usize,
) -> Option<IntegerWitness> {
    if Instant::now() >= state.deadline {
        return None;
    }
    let mut seed = Vec::new();
    search_joint_witness_seeds(
        state,
        background,
        allocation_groups,
        0,
        &mut seed,
        masters,
        work,
        iterations,
    )
}

/// Enumerate fallback identity combinations lazily under a single shared
/// work, LP-iteration and solve-deadline budget.
pub(super) fn search_joint_witness_seeds<const D: usize>(
    state: &State<D>,
    background: &[FixedRoute],
    groups: &[(Vec<(usize, f64, NetworkRoute)>, f64)],
    index: usize,
    selected: &mut Vec<FixedRoute>,
    masters: &[(i32, f64)],
    work: &mut usize,
    iterations: &mut usize,
) -> Option<IntegerWitness> {
    let mut found = None;
    for_each_joint_witness_seed(
        state,
        groups,
        index,
        selected,
        work,
        iterations,
        &mut |allocation, work, iterations| {
            found = joint_witness_targets_for_seed(
                state, background, allocation, masters, work, iterations,
            );
            Ok(found.is_some()
                || *work == 0
                || *iterations == 0
                || Instant::now() >= state.deadline)
        },
    )
    .ok()?;
    found
}

/// Visit complete cross-group identity allocations without materializing their
/// Cartesian product. Callers stop at a witness or shared budget boundary.
pub(super) fn for_each_joint_witness_seed<const D: usize>(
    state: &State<D>,
    groups: &[(Vec<(usize, f64, NetworkRoute)>, f64)],
    index: usize,
    selected: &mut Vec<FixedRoute>,
    work: &mut usize,
    iterations: &mut usize,
    on_seed: &mut impl FnMut(&[FixedRoute], &mut usize, &mut usize) -> Result<bool>,
) -> Result<bool> {
    if Instant::now() >= state.deadline || *work == 0 || *iterations == 0 {
        return Ok(false);
    }
    if index == groups.len() {
        return on_seed(selected, work, iterations);
    }
    let (choices, quantity) = &groups[index];
    for_each_identity_allocation(
        state,
        choices,
        *quantity,
        state.deadline,
        work,
        &mut |allocation, _, work| {
            let old_len = selected.len();
            selected.extend_from_slice(allocation);
            let stop = for_each_joint_witness_seed(
                state,
                groups,
                index + 1,
                selected,
                work,
                iterations,
                on_seed,
            )?;
            selected.truncate(old_len);
            Ok(stop || *work == 0 || *iterations == 0 || Instant::now() >= state.deadline)
        },
    )
}

pub(super) fn joint_witness_targets_for_seed<const D: usize>(
    state: &State<D>,
    background: &[FixedRoute],
    seed: &[FixedRoute],
    masters: &[(i32, f64)],
    shared_work: &mut usize,
    shared_iterations: &mut usize,
) -> Option<IntegerWitness> {
    if Instant::now() >= state.deadline {
        return None;
    }
    let mut work = *shared_work;
    let mut iterations = *shared_iterations;
    let result = (|| -> Option<IntegerWitness> {
        let mut routes = background.to_vec();
        routes.extend_from_slice(seed);
        let mut rhs: Vec<_> = state.shared_rows.iter().map(|r| r.rhs).collect();
        for (index, fixed) in routes.iter().enumerate() {
            let oracle = state
                .oracles
                .iter()
                .find(|o| o.network.id == fixed.route.network)?;
            let terms = if index < background.len() {
                entries(oracle, &fixed.route.route, state.shared_offset).ok()?
            } else {
                // Variable paired contributions belong to the target matrix, not
                // to a rounded baseline which would then need cancellation.
                entries_base(oracle, &fixed.route.route, state.shared_offset)
                    .ok()?
                    .into_iter()
                    .collect()
            };
            for (row, a) in terms {
                if let Some(row) = row
                    .checked_sub(state.shared_offset)
                    .and_then(|r| rhs.get_mut(r))
                {
                    *row -= a * fixed.quantity;
                }
            }
        }
        for variable in &state.master_variables {
            let value = masters.iter().find(|(id, _)| *id == variable.id)?.1;
            for &(row, a) in &variable.entries {
                *rhs.get_mut(row)? -= a * value;
            }
        }
        let mut domains = vec![];
        let mut costs = vec![];
        let mut variables = vec![];
        let mut coefficients = vec![Vec::new(); rhs.len()];
        for (index, fixed) in routes.iter().enumerate().skip(background.len()) {
            let oracle = state
                .oracles
                .iter()
                .find(|o| o.network.id == fixed.route.network)?;
            let path: Vec<_> = fixed
                .route
                .route
                .arcs
                .iter()
                .map(|id| oracle.index.get(id).copied())
                .collect::<Option<_>>()?;
            for resource in resources(oracle) {
                let domain = domain(
                    oracle,
                    &path,
                    resource,
                    &mut work,
                    &mut iterations,
                    state.deadline,
                )??;
                let variable = domains.len();
                let lower = domain.lower * fixed.quantity;
                let upper = domain.upper * fixed.quantity;
                if !lower.is_finite() || !upper.is_finite() {
                    return None;
                }
                for entry in oracle
                    .activity_rows
                    .iter()
                    .filter(|e| e.resource == resource)
                {
                    let a = entry.coefficient;
                    coefficients.get_mut(entry.row)?.push((variable, a));
                    *rhs.get_mut(entry.row)? -= a * lower;
                }
                costs.push(
                    oracle
                        .network
                        .graph
                        .coupled_kernel()?
                        .effective_objective_coefficient(resource)?,
                );
                domains.push(Interval {
                    lo: lower,
                    hi: upper,
                });
                variables.push((index, resource, path.clone()));
            }
        }
        if variables.is_empty() {
            return None;
        }
        let row_count = rhs.len();
        work = work.checked_sub(variables.len().checked_mul(row_count.checked_add(1)?)?)?;
        let rows: Vec<_> = state
            .shared_rows
            .iter()
            .zip(rhs)
            .zip(coefficients)
            .map(|((row, rhs), a)| (row.sense, rhs, a))
            .collect();
        let proposal_count = variables.len().checked_mul(2)?.checked_add(2)?;
        let vertex_enumeration_reserve = work / 3;
        let vertex_realization_reserve = work / 3;
        let vertex_iteration_reserve = iterations / 3;
        let mut vertex_iterations = vertex_iteration_reserve;
        let mut coordinate_iterations = iterations.checked_sub(vertex_iteration_reserve)?;
        work =
            work.checked_sub(vertex_enumeration_reserve.checked_add(vertex_realization_reserve)?)?;
        let mut enumeration_work = vertex_enumeration_reserve;
        let mut vertex_realization_work = vertex_realization_reserve;
        let proposal_result = (|| -> Option<IntegerWitness> {
            for proposal in 0..proposal_count {
                if Instant::now() >= state.deadline {
                    return None;
                }
                let proposal_work = variables.len().checked_mul(row_count.checked_add(1)?)?;
                let Some(remaining_work) = work.checked_sub(proposal_work) else {
                    break;
                };
                work = remaining_work;
                let objective = terminal_objective(&costs, proposal)?;
                let Some(totals) = solve_terminal_targets(
                    &domains,
                    &rows,
                    &objective,
                    &mut coordinate_iterations,
                    state.deadline,
                ) else {
                    continue;
                };
                let targets: Vec<_> = variables
                    .iter()
                    .zip(totals)
                    .map(|((index, _, _), total)| TerminalTarget {
                        average: total / routes[*index].quantity,
                        total,
                    })
                    .collect();
                for split in [false, true] {
                    let Some(candidate) = realize_joint_targets(
                        state,
                        &routes,
                        background.len(),
                        &variables,
                        &targets,
                        split,
                        (&mut work, &mut coordinate_iterations),
                    ) else {
                        continue;
                    };
                    let Some(scheduled) = synchronization::witness(state, &candidate).ok()? else {
                        continue;
                    };
                    if !scheduled_identities_allowed(state, &scheduled[state.fixed_routes.len()..])
                    {
                        continue;
                    }
                    if let Some(objective) =
                        tree_adapter::validate_routes(state, &scheduled, masters).ok()?
                    {
                        return Some((objective, scheduled, masters.to_vec()));
                    }
                }
            }
            // Global cross-path/network faces can have vertices that are not exposed
            // by any one terminal coordinate. Use the same bounded LP vertex sweep as
            // route-local pricing, while reserving half of the remaining work for
            // source realization and whole-witness validation.
            let vertices =
                terminal_vertices(&domains, &rows, &mut enumeration_work, state.deadline);
            for vertex in vertices {
                if Instant::now() >= state.deadline || vertex_realization_work == 0 {
                    break;
                }
                let totals: Vec<_> = domains
                    .iter()
                    .zip(vertex)
                    .map(|(domain, shift)| domain.lo + shift)
                    .collect();
                if totals.iter().any(|total| !total.is_finite()) {
                    continue;
                }
                let targets: Vec<_> = variables
                    .iter()
                    .zip(totals)
                    .map(|((index, _, _), total)| TerminalTarget {
                        average: total / routes[*index].quantity,
                        total,
                    })
                    .collect();
                for split in [false, true] {
                    let Some(candidate) = realize_joint_targets(
                        state,
                        &routes,
                        background.len(),
                        &variables,
                        &targets,
                        split,
                        (&mut vertex_realization_work, &mut vertex_iterations),
                    ) else {
                        continue;
                    };
                    let Some(scheduled) = synchronization::witness(state, &candidate).ok()? else {
                        continue;
                    };
                    if !scheduled_identities_allowed(state, &scheduled[state.fixed_routes.len()..])
                    {
                        continue;
                    }
                    if let Some(objective) =
                        tree_adapter::validate_routes(state, &scheduled, masters).ok()?
                    {
                        return Some((objective, scheduled, masters.to_vec()));
                    }
                }
            }
            None
        })();
        work = work
            .saturating_add(enumeration_work)
            .saturating_add(vertex_realization_work);
        iterations = coordinate_iterations.saturating_add(vertex_iterations);
        proposal_result
    })();
    *shared_work = work;
    *shared_iterations = iterations;
    result
}

pub(super) type TerminalVariable = (usize, usize, Vec<usize>);
#[derive(Clone, Copy)]
pub(super) struct TerminalTarget {
    pub(super) average: f64,
    pub(super) total: f64,
}

// A nonrepresentable average need not rule out an exact integer witness.
// Try integer copy blocks at dyadic neighboring Gamma values, then replay
// the complete source model; rounded aggregate targets never become proofs.
pub(super) fn realize_joint_targets<const D: usize>(
    state: &State<D>,
    routes: &[FixedRoute],
    first: usize,
    variables: &[TerminalVariable],
    targets: &[TerminalTarget],
    split: bool,
    budget: (&mut usize, &mut usize),
) -> Option<Vec<FixedRoute>> {
    let (work, iterations) = budget;
    let mut result = routes[..first].to_vec();
    let mut divided = false;
    for (index, fixed) in routes.iter().enumerate().skip(first) {
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == fixed.route.network)?;
        let q = fixed.quantity;
        let mut breaks = vec![0., q];
        let mut points = vec![];
        for ((_, resource, path), &target) in variables
            .iter()
            .zip(targets)
            .filter(|((i, _, _), _)| *i == index)
        {
            let (low, high, cut) = if split && q > 1. && target.average.fract() != 0. {
                let points = dyadic_copy_controls(
                    oracle,
                    path,
                    *resource,
                    target,
                    q,
                    (work, iterations),
                    state.deadline,
                )?;
                divided = true;
                points
            } else {
                let control = target_control(
                    oracle,
                    path,
                    *resource,
                    target.average,
                    work,
                    iterations,
                    state.deadline,
                )?
                .control?;
                (control.clone(), control, q)
            };
            breaks.push(cut);
            points.push((*resource, low, high, cut));
        }
        breaks.sort_by(f64::total_cmp);
        breaks.dedup();
        for segment in breaks.windows(2) {
            if segment[0] == segment[1] {
                continue;
            }
            let mut controls = fixed.route.route.coupled_controls.clone();
            for (resource, low, high, cut) in &points {
                controls.retain(|c| c.resource != *resource);
                controls.push(if segment[0] >= *cut {
                    high.clone()
                } else {
                    low.clone()
                });
            }
            let mut route = fixed.route.clone();
            route.route = route
                .route
                .with_paired_controls(&oracle.network.graph, sorted(controls))
                .ok()?;
            result.push(FixedRoute {
                route,
                quantity: segment[1] - segment[0],
            });
        }
    }
    (!split || divided).then_some(result)
}

// Search at most 53 binary grids, from the target's magnitude down to its
// binary64 lattice. Counts are exact expansion identities for the proposed
// aggregate; only source replay can establish the model's row feasibility.
pub(super) fn dyadic_copy_controls<const D: usize>(
    oracle: &Oracle<D>,
    path: &[usize],
    resource: usize,
    target: TerminalTarget,
    copies: f64,
    budget: (&mut usize, &mut usize),
    deadline: Instant,
) -> Option<(PairedControl, PairedControl, f64)> {
    use crate::master::numerics::ExactSum;
    let (work, iterations) = budget;
    let total = target.total;
    let target = target.average;
    if !target.is_finite() || !total.is_finite() || copies <= 1. {
        return None;
    }
    let exponent = (((target.abs().to_bits() >> 52) & 0x7ff) as i32 - 1023).max(-1022);
    for precision in 0..=52 {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let power = exponent - precision;
        let step = if power >= -1022 {
            f64::from_bits(((power + 1023) as u64) << 52)
        } else {
            f64::from_bits(1u64 << (power + 1074))
        };
        let lo = (target / step).floor() * step;
        let hi = lo + step;
        if !lo.is_finite() || !hi.is_finite() || lo == hi {
            continue;
        }
        let mut residual = ExactSum::default();
        if residual.add(total).is_none() || residual.product(-lo, copies).is_none() {
            continue;
        }
        let high_count = residual.upper()? / step;
        if !high_count.is_finite()
            || high_count.fract() != 0.
            || !(0. ..=copies).contains(&high_count)
        {
            continue;
        }
        if residual.product(-step, high_count).is_none() || residual.sign() != 0. {
            continue;
        }
        let Some(low) = attained_control(
            oracle,
            path,
            resource,
            if high_count == copies { hi } else { lo },
            work,
            iterations,
            deadline,
        ) else {
            continue;
        };
        let high = if high_count == 0. || high_count == copies {
            low.clone()
        } else {
            let Some(high) =
                attained_control(oracle, path, resource, hi, work, iterations, deadline)
            else {
                continue;
            };
            high
        };
        return Some((low, high, copies - high_count));
    }
    None
}

pub(super) fn attained_control<const D: usize>(
    oracle: &Oracle<D>,
    path: &[usize],
    resource: usize,
    value: f64,
    work: &mut usize,
    iterations: &mut usize,
    deadline: Instant,
) -> Option<PairedControl> {
    let control =
        target_control(oracle, path, resource, value, work, iterations, deadline)?.control?;
    let proof = oracle
        .network
        .graph
        .coupled_kernel()?
        .certify_paired_control(path, &control, work, Some(deadline))
        .ok()??;
    proof.certifies_value(value).then_some(control)
}

// Scheduling can change a scalar profile, hence the column identity. Check
// caps and retained coverage on the actual variable copies being returned;
// fixed copies already have their contribution removed from residual caps.
pub(super) fn scheduled_identities_allowed<const D: usize>(
    state: &State<D>,
    routes: &[FixedRoute],
) -> bool {
    use crate::master::numerics::ExactSum;
    if state.universe.is_none() && state.route_upper_bounds.is_empty() {
        return true;
    }
    let identities: Vec<_> = routes
        .iter()
        .map(|r| (r.route.network, r.route.route.identity(), r.quantity))
        .collect();
    if let Some(universe) = &state.universe {
        if identities.iter().any(|(network, identity, _)| {
            !universe.iter().any(|&tag| {
                let candidate = &state.routes[tag];
                candidate.network == *network && candidate.route.identity() == *identity
            })
        }) {
            return false;
        }
    }
    for (&tag, &cap) in &state.route_upper_bounds {
        if cap == f64::INFINITY {
            continue;
        }
        let candidate = &state.routes[tag];
        let identity = candidate.route.identity();
        let mut quantity = ExactSum::default();
        for (network, selected, value) in &identities {
            if *network == candidate.network
                && *selected == identity
                && quantity.add(*value).is_none()
            {
                return false;
            }
        }
        if quantity.add(-cap).is_none() || quantity.sign() > 0.0 {
            return false;
        }
    }
    true
}
