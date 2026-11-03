//! Redundant constraints keep their original identity and replay definition.
//! Only their restricted-master slot is neutralized. This avoids renumbering
//! packing identities and inherited nonlinear registries in descendant nodes.
use super::*;

pub(super) fn project<const D: usize>(state: &State<D>, entries: &mut Vec<(usize, f64)>) {
    entries.retain(|(row, _)| !state.retired_rows.contains(row));
}

pub(super) fn active_entries<const D: usize>(
    state: &State<D>,
    oracle: &Oracle<D>,
    route: &Route,
) -> Result<Vec<(usize, f64)>> {
    let mut result = entries(oracle, route, state.shared_offset)?;
    project(state, &mut result);
    Ok(result)
}

pub(super) fn physical_rhs<const D: usize>(state: &State<D>) -> Vec<f64> {
    let mut rhs = state.rhs.clone();
    for &row in &state.retired_rows {
        rhs[row] = 0.;
    }
    rhs
}

/// New rows extend the semantic RHS; backend zero slots never overwrite it.
pub(super) fn sync_rhs<const D: usize>(state: &mut State<D>, master: &ClpMaster) {
    let mut rhs = master.rhs.clone();
    for &row in &state.retired_rows {
        rhs[row] = state.rhs[row];
    }
    state.rhs = rhs;
}

pub(super) fn duals<'a, const D: usize>(
    state: &State<D>,
    values: &'a [f64],
) -> std::borrow::Cow<'a, [f64]> {
    if state.retired_rows.is_empty() {
        return std::borrow::Cow::Borrowed(values);
    }
    let mut values = values.to_vec();
    for &row in &state.retired_rows {
        if let Some(value) = values.get_mut(row) {
            *value = 0.;
        }
    }
    std::borrow::Cow::Owned(values)
}

fn retained_quantity<const D: usize>(oracle: &Oracle<D>) -> f64 {
    // The physical quantity row enforces this bound independently of any
    // shared row we may remove. A merely inferred aggregate upper endpoint
    // cannot justify deleting its own only encoding in the master.
    oracle.network.multiplicity.upper
}

/// Outward interval evaluation uses only bounds retained in the node domain.
/// A current LP solution or an incomplete route pool supplies no evidence.
fn envelope<const D: usize>(state: &State<D>, row: usize) -> Option<(f64, f64)> {
    use crate::core::rounding::{add_down, add_up, multiply_down};
    let multiply_up = |a: f64, b: f64| multiply_down(-a, b).map(|v| -v);
    let mut lower = 0.;
    let mut upper = 0.;
    let mut add = |a: f64, lo: f64, hi: f64| -> Option<()> {
        if a == 0. {
            return Some(());
        }
        let (lo, hi) = if a > 0. { (lo, hi) } else { (hi, lo) };
        lower = add_down(lower, multiply_down(a, lo)?)?;
        upper = add_up(upper, multiply_up(a, hi)?)?;
        Some(())
    };
    for variable in &state.master_variables {
        for &(r, a) in &variable.entries {
            if r == row {
                add(a, variable.bounds.lower, variable.bounds.upper)?;
            }
        }
    }
    for oracle in &state.oracles {
        let quantity = retained_quantity(oracle);
        if quantity == 0. {
            continue;
        }
        // General schedule/class/floor coefficients require their own proved
        // whole-domain envelopes. Never substitute additive route activity.
        if oracle
            .nonlinear_rows
            .iter()
            .chain(&oracle.knapsack_rows)
            .chain(&oracle.presence_rows)
            .any(|&(_, r)| r == row)
            || oracle.activity_rows.iter().any(|entry| entry.row == row)
            || oracle.resource_groups.iter().any(|&(r, _)| r == row)
            || oracle.ryan_foster_rows.iter().any(|&(r, _)| r == row)
            || oracle.arc_set_branch_rows.iter().any(|(r, _)| *r == row)
        {
            return None;
        }
        for &(r, a) in oracle.network.route_entries.iter() {
            if r == row {
                add(a, oracle.network.multiplicity.lower, quantity)?;
            }
        }
        for (arc, terms) in oracle.network.arc_entries.iter().enumerate() {
            for &(r, a) in terms {
                if r != row || a == 0. {
                    continue;
                }
                let cap = if oracle.proved_zero_arcs.get(arc).copied().unwrap_or(false) {
                    0.
                } else {
                    let count = oracle.network.graph.max_route_arcs()? as f64;
                    multiply_up(count, quantity)?
                };
                add(a, 0., cap)?;
            }
        }
    }
    Some((lower, upper))
}

/// A complete original-domain route universe can also enclose nonlinear rows.
/// Gap-limited pools and an ordinary restricted master cannot use this path.
fn complete_envelope<const D: usize>(
    state: &State<D>,
    row: usize,
    work: &mut usize,
) -> Option<(f64, f64)> {
    use crate::core::rounding::{add_down, add_up, multiply_down};
    use crate::master::numerics::ExactSum;
    if !state.enumeration_report.complete || state.enumeration_report.coverage_cutoff.is_some() {
        return None;
    }
    let universe = state.universe.as_ref()?;
    let mut lower = 0.;
    let mut upper = 0.;
    let mut contribution = |a: f64, lo: f64, hi: f64| -> Option<()> {
        if a == 0. {
            return Some(());
        }
        let (lo, hi) = if a > 0. { (lo, hi) } else { (hi, lo) };
        lower = add_down(lower, multiply_down(a, lo)?)?;
        upper = add_up(upper, -multiply_down(-a, hi)?)?;
        Some(())
    };
    for variable in &state.master_variables {
        for &(r, a) in &variable.entries {
            if r == row {
                contribution(a, variable.bounds.lower, variable.bounds.upper)?;
            }
        }
    }
    // End the mutable capture before adding network intervals.
    drop(contribution);
    for oracle in &state.oracles {
        let quantity = retained_quantity(oracle);
        if quantity == 0. {
            continue;
        }
        let (mut lo, mut hi) = (0_f64, 0_f64);
        for &tag in universe {
            let route = &state.routes[tag];
            if route.network != oracle.network.id || state.route_upper_bounds.get(&tag) == Some(&0.)
            {
                continue;
            }
            *work = work.checked_sub(route.route.arcs.len().saturating_add(1))?;
            if Instant::now() >= state.deadline {
                return None;
            }
            // Retain schedule-dependent activities unless a separate source
            // envelope exists. For registered/discrete penalties use bounds
            // covering all pricing memory encodings, not the current label.
            if oracle.activity_rows.iter().any(|entry| entry.row == row) {
                return None;
            }
            let mut high = ExactSum::default();
            for &(r, a) in oracle.network.route_entries.iter() {
                if r == row {
                    high.add(a)?;
                }
            }
            for id in &route.route.arcs {
                for &(r, a) in &oracle.network.arc_entries[oracle.index[id]] {
                    if r == row {
                        high.add(a)?;
                    }
                }
            }
            for &(cut, r) in &oracle.nonlinear_rows {
                if r == row {
                    for id in &route.route.arcs {
                        high.add(nonlinear::arc_upper(oracle, cut, oracle.index[id]))?;
                    }
                }
            }
            for &(cut, r) in &oracle.knapsack_rows {
                if r == row {
                    let (_, b) = oracle.solver.cuts.knapsack_cuts[cut]
                        .cut
                        .coefficients
                        .values()
                        .fold((0, 0), |(a, b), &v| (a.min(v), b.max(v)));
                    high.add(b as f64)?;
                }
            }
            for &(input, r) in &oracle.presence_rows {
                if r == row {
                    let maximum = oracle.solver.presence_rows()[input]
                        .row
                        .maximum_coefficient();
                    if maximum as u128 > 1_u128 << 53 {
                        return None;
                    }
                    high.add(maximum as f64)?;
                }
            }
            for r in oracle
                .resource_groups
                .iter()
                .map(|(r, _)| *r)
                .chain(oracle.ryan_foster_rows.iter().map(|(r, _)| *r))
                .chain(oracle.arc_set_branch_rows.iter().map(|(r, _)| *r))
            {
                if r == row {
                    high.add(1.)?;
                }
            }
            hi = hi.max(high.upper()?);
            // ExactSum has an upward conversion; negate its source terms to
            // obtain the opposite directed endpoint without double rounding.
            let mut negative = ExactSum::default();
            for &(r, a) in oracle.network.route_entries.iter() {
                if r == row {
                    negative.add(-a)?;
                }
            }
            for id in &route.route.arcs {
                for &(r, a) in &oracle.network.arc_entries[oracle.index[id]] {
                    if r == row {
                        negative.add(-a)?;
                    }
                }
            }
            for &(cut, r) in &oracle.knapsack_rows {
                if r == row {
                    let minimum = oracle.solver.cuts.knapsack_cuts[cut]
                        .cut
                        .coefficients
                        .values()
                        .copied()
                        .min()
                        .unwrap_or(0)
                        .min(0);
                    negative.add(-(minimum as f64))?;
                }
            }
            lo = lo.min(-negative.upper()?);
        }
        lower = add_down(lower, multiply_down(lo, quantity)?)?;
        upper = add_up(upper, -multiply_down(-hi, quantity)?)?;
    }
    Some((lower, upper))
}

/// Called before CG creates its checkpoint store or private stabilization LP.
pub(super) fn apply<const D: usize>(
    master: &mut ClpMaster,
    state: &mut State<D>,
    options: &mut cg::Options,
    work_limit: usize,
) -> Result<()> {
    apply_to_metadata(
        master,
        state,
        &mut options.known,
        options.warm_start.as_mut(),
        work_limit,
    )
}

pub(super) fn apply_to_metadata<const D: usize, W: WarmMetadata>(
    master: &mut ClpMaster,
    state: &mut State<D>,
    known: &mut [cg::Column],
    warm: Option<&mut W>,
    work_limit: usize,
) -> Result<()> {
    state.preprocessing.retired_rows = state.retired_rows.iter().copied().collect();
    state.preprocessing.retired_rows.sort_unstable();
    if state.preprocessing.skipped.is_some()
        || state.preprocessing.propagation.infeasible
        || state.objective_fixed
    {
        return Ok(());
    }
    let mut removed = vec![];
    let mut work = work_limit.saturating_sub(state.preprocessing.propagation.work_used);
    let available = work;
    for (row, definition) in state.shared_rows.iter().enumerate() {
        let physical = state.shared_offset + row;
        if state.retired_rows.contains(&physical)
            || state.master_variable_bound_rows.contains(&physical)
        {
            continue;
        }
        let cost = state
            .master_variables
            .iter()
            .map(|v| v.entries.len())
            .sum::<usize>()
            .saturating_add(
                state
                    .oracles
                    .iter()
                    .map(|o| {
                        o.network
                            .arc_entries
                            .iter()
                            .map(Vec::len)
                            .sum::<usize>()
                            .saturating_add(o.network.route_entries.len())
                    })
                    .sum::<usize>(),
            )
            .saturating_mul(8)
            .saturating_add(1);
        let Some(remaining) = work.checked_sub(cost) else {
            break;
        };
        work = remaining;
        if Instant::now() >= state.deadline {
            break;
        }
        let Some((lower, upper)) =
            envelope(state, row).or_else(|| complete_envelope(state, row, &mut work))
        else {
            continue;
        };
        let rhs = state.rhs[physical];
        let redundant = match definition.sense {
            RowSense::LessEqual => upper <= rhs,
            RowSense::GreaterEqual => lower >= rhs,
            RowSense::Equal => lower >= rhs && upper <= rhs,
        };
        if redundant {
            removed.push(physical);
        }
    }
    state.preprocessing.propagation.work_used = state
        .preprocessing
        .propagation
        .work_used
        .saturating_add(available - work);
    if removed.is_empty() {
        return Ok(());
    }
    state.retired_rows.extend(removed.iter().copied());
    for oracle in &mut state.oracles {
        for &(cut, row) in &oracle.nonlinear_rows {
            if removed.contains(&(state.shared_offset + row)) {
                oracle.solver.cuts.cuts[cut].dual = 0.;
            }
        }
        for &(cut, row) in &oracle.knapsack_rows {
            if removed.contains(&(state.shared_offset + row)) {
                oracle.solver.cuts.knapsack_cuts[cut].dual = 0.;
            }
        }
        if !oracle.presence_rows.is_empty() {
            let mut prices: Vec<_> = oracle
                .solver
                .presence_rows()
                .iter()
                .map(|r| r.dual)
                .collect();
            for &(input, row) in &oracle.presence_rows {
                if removed.contains(&(state.shared_offset + row)) {
                    prices[input] = 0.;
                }
            }
            oracle.solver.set_presence_duals(&prices)?;
        }
    }
    // Artificial columns are identified by ownership metadata, never by cost.
    let artificials: HashSet<_> = warm.as_ref().map_or_else(HashSet::new, |warm| {
        warm.artificials().iter().copied().collect()
    });
    for column in 0..master.costs.len() {
        let coefficients = master.column_entries(column)?;
        if artificials.contains(&column) && coefficients.iter().any(|(r, _)| removed.contains(r)) {
            master.set_upper(column, 0.)?;
        }
        for (row, _) in coefficients {
            if removed.contains(&row) {
                master.set_coefficient(row, column, 0.)?;
            }
        }
    }
    master.set_rhs(&physical_rhs(state))?;
    for &row in &removed {
        if let Some(role) = state.curvature_rows.get_mut(row) {
            role.active = false;
        }
    }
    for column in known {
        project(state, &mut column.entries);
    }
    if let Some(warm) = warm {
        warm.edit_columns(
            |column| {
                column
                    .entries
                    .iter()
                    .any(|(row, _)| state.retired_rows.contains(row))
            },
            |column| project(state, &mut column.entries),
        );
        warm.clear_stabilization();
        warm.clear_proximal();
    }
    state.gradient = None;
    state.correction = None;
    state.preprocessing.retired_rows.extend(removed);
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/retirement_tests.rs"]
mod tests;
