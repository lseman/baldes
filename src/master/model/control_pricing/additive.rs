//! Exact additive bounds for constant paired recurrences. This is a proof
//! specialization; source feasibility and returned controls are still replayed.
use super::*;

const LIMIT: i128 = 1_i128 << 40;
const COST_LIMIT: i128 = 1_i128 << 50;
// Resource and price operands occupy separate exact binary lattices. Their
// products, physical costs and route constants share the finer cost lattice.
const OPERAND_SCALE: f64 = 1024.;
const RESOURCE_UNITS: i128 = 1024;
const COST_SCALE: f64 = OPERAND_SCALE * OPERAND_SCALE;

fn price(value: f64) -> Option<i128> {
    integer(value * OPERAND_SCALE)
}

fn integer(value: f64) -> Option<i128> {
    (value.is_finite() && value.fract() == 0. && value.abs() <= LIMIT as f64)
        .then_some(value as i128)
}

fn slope_constant(
    function: &crate::graph::coupled::PreparedSlopeFunction,
    work: &mut usize,
) -> Option<i128> {
    *work = work.checked_sub(function.pairs().len())?;
    function
        .pairs()
        .iter()
        .skip(1)
        .all(|pair| pair.0 == 0.)
        .then(|| integer(function.pairs()[0].0 * OPERAND_SCALE))
        .flatten()
}

fn knot_constant(
    function: &crate::graph::coupled::PiecewiseLinearFunction,
    work: &mut usize,
) -> Option<i128> {
    *work = work.checked_sub(function.points.len())?;
    let value = function.points.first()?[1];
    function
        .points
        .iter()
        .all(|point| point[1] == value)
        .then(|| integer(value * OPERAND_SCALE))
        .flatten()
}

pub(super) fn prepare<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<PhysicalSuffixBounds> {
    let graph = &oracle.network.graph;
    if graph.front_program_domain()
        || graph.source == graph.sink
        || !oracle.nonlinear_rows.is_empty()
        || !oracle.knapsack_rows.is_empty()
        || !oracle.presence_rows.is_empty()
        || !oracle.resource_groups.is_empty()
    {
        return None;
    }
    let kernel = graph.coupled_kernel()?;
    let hops = graph.original_max_arcs();
    let mut costs = Vec::with_capacity(graph.arcs.len());
    let mut magnitude = 0_i128;
    let mut largest_arc = 0_i128;
    for (index, arc) in graph.arcs.iter().enumerate() {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let mut cost = price(arc.cost)?.checked_mul(RESOURCE_UNITS)?;
        let mut absolute = cost.abs();
        for &(row, coefficient) in oracle.network.arc_entries.get(index)? {
            *work = work.checked_sub(1)?;
            let term = price(*duals.get(offset.checked_add(row)?)?)?
                .checked_mul(integer(coefficient)?)?
                .checked_mul(RESOURCE_UNITS)?;
            cost = cost.checked_sub(term)?;
            absolute = absolute.checked_add(term.abs())?;
        }
        largest_arc = largest_arc.max(absolute);
        costs.push(cost);
    }
    magnitude = magnitude.checked_add(largest_arc.checked_mul(hops as i128)?)?;
    let mut constant = 0_i128;
    for (base, entries) in [
        (0, oracle.route_entries.as_slice()),
        (offset, oracle.network.route_entries.as_slice()),
    ] {
        for &(row, coefficient) in entries {
            *work = work.checked_sub(1)?;
            let term = price(*duals.get(base.checked_add(row)?)?)?
                .checked_mul(integer(coefficient)?)?
                .checked_mul(RESOURCE_UNITS)?;
            constant = constant.checked_sub(term)?;
            magnitude = magnitude.checked_add(term.abs())?;
        }
    }
    for (resource, definition) in kernel.definitions().iter().enumerate() {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let (initial, increments, mode, normalization) = match definition {
            CoupledResource::SlopePaired {
                initial,
                arcs,
                mode,
                normalization,
                ..
            } => {
                let initial = slope_constant(initial, work)?;
                let mut increments = Vec::with_capacity(arcs.len());
                for arc in arcs {
                    *work = work.checked_sub(1)?;
                    if arc.waiting != 0 || slope_constant(&arc.lower, work)? > initial {
                        return None;
                    }
                    increments.push(slope_constant(&arc.increment, work)?);
                }
                (initial, increments, *mode, *normalization)
            }
            CoupledResource::PairedPiecewiseLinear {
                initial,
                arcs,
                mode,
                normalization,
                ..
            } => {
                let initial = knot_constant(initial, work)?;
                let mut increments = Vec::with_capacity(arcs.len());
                for arc in arcs {
                    *work = work.checked_sub(1)?;
                    if arc.waiting != 0 || knot_constant(&arc.lower, work)? > initial {
                        return None;
                    }
                    increments.push(knot_constant(&arc.increment, work)?);
                }
                (initial, increments, *mode, *normalization)
            }
            _ => return None,
        };
        if increments.len() != costs.len() || increments.iter().any(|&value| value < 0) {
            return None;
        }
        let maximum = initial.abs().checked_add(
            increments
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
                .checked_mul(hops as i128)?,
        )?;
        if maximum > LIMIT {
            return None;
        }
        let objective = if mode == crate::graph::coupled::PairedResourceMode::Feasibility {
            0
        } else {
            price(kernel.effective_objective_coefficient(resource)?)?
        };
        let mut coefficient = objective;
        let mut absolute = objective.abs();
        let baseline = if normalization
            == crate::graph::coupled::PairedCostNormalization::AbsoluteFinalValue
        {
            initial
        } else {
            0
        };
        constant = constant.checked_add(objective.checked_mul(baseline)?)?;
        for row in oracle
            .activity_rows
            .iter()
            .filter(|row| row.resource == resource && row.coefficient != 0.)
        {
            *work = work.checked_sub(1)?;
            let term = price(*duals.get(offset.checked_add(row.row)?)?)?
                .checked_mul(integer(row.coefficient)?)?;
            coefficient = coefficient.checked_sub(term)?;
            absolute = absolute.checked_add(term.abs())?;
            constant = constant.checked_sub(term.checked_mul(initial)?)?;
        }
        // Bound every intermediate of the original terminal charge as well
        // as its redistributed form, including relative normalization.
        magnitude =
            magnitude.checked_add(absolute.checked_mul(maximum.checked_add(initial.abs())?)?)?;
        if magnitude > COST_LIMIT {
            return None;
        }
        for (cost, increment) in costs.iter_mut().zip(increments) {
            *cost = cost.checked_add(coefficient.checked_mul(increment)?)?;
        }
    }
    if magnitude > COST_LIMIT || Instant::now() >= deadline {
        return None;
    }
    let arc_costs: Vec<_> = costs
        .into_iter()
        .map(|cost| cost as f64 / COST_SCALE)
        .collect();
    let suffix = suffix_table(graph, &arc_costs, hops, work, deadline)?;
    Some(PhysicalSuffixBounds {
        suffix,
        arc_costs,
        route_constant: constant as f64 / COST_SCALE,
        other_coupled_lower: 0.,
        paired_terminal_charges: vec![],
        exact_additive: true,
    })
}

pub(super) fn route_upper<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    arcs: &[usize],
    route: &Route,
    work: &mut usize,
    deadline: Instant,
) -> Option<f64> {
    let resources: Vec<_> = route
        .coupled_controls
        .iter()
        .map(|control| control.resource)
        .collect();
    let source = SourceColumn {
        kernel: oracle.network.graph.coupled_kernel()?,
        controls: &route.coupled_controls,
        variable_resources: &resources,
    };
    let mut terms = components(oracle, offset, duals, arcs, work, deadline, source)??;
    for row in oracle
        .activity_rows
        .iter()
        .filter(|row| row.coefficient != 0. && resources.contains(&row.resource))
    {
        let range = activity_rows::certificate_value(oracle, route, row.resource, work, deadline)?;
        let term = Interval {
            lo: range[0],
            hi: range[1],
        }
        .scale(row.coefficient)?;
        let value = terms
            .rows
            .entry(offset.checked_add(row.row)?)
            .or_insert(Interval::point(0.));
        *value = value.add(term)?;
    }
    for (row, value) in terms.rows {
        terms.cost = terms.cost.add(value.scale(-*duals.get(row)?)?)?;
    }
    // prepare proves that every original reduced-cost operation is on the
    // exact cost lattice. Tighten the certified upper enclosure onto
    // that lattice, rather than using a tolerance to assert attainment.
    terms
        .cost
        .hi
        .is_finite()
        .then_some((terms.cost.hi * COST_SCALE).floor() / COST_SCALE)
}

pub(super) fn attained_minimum<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    let bounds = prepare(oracle, offset, duals, work, deadline)?;
    let graph = &oracle.network.graph;
    let mut remaining = bounds.suffix.len().checked_sub(1)?;
    let Some(tail) = bounds.suffix.get(remaining)?.get(graph.source).copied()? else {
        return Some(None);
    };
    let lower = bounds.route_constant + tail;
    if lower >= 0. {
        return Some(Some(lower));
    }
    // Stable suffix rows represent every larger horizon. Reconstruct from
    // their stored horizon instead of spending work decrementing a possibly
    // enormous declared cap through identical virtual rows.
    if bounds.suffix.stable {
        remaining = remaining.min(bounds.suffix.rows.len().checked_sub(1)?);
    }
    let mut vertex = graph.source;
    let mut arcs = Vec::new();
    while vertex != graph.sink {
        *work = work.checked_sub(1)?;
        if remaining == 0 || Instant::now() >= deadline {
            return None;
        }
        let target = bounds
            .suffix
            .get(remaining)?
            .get(vertex)
            .copied()
            .flatten()?;
        if bounds
            .suffix
            .get(remaining - 1)?
            .get(vertex)
            .copied()
            .flatten()
            == Some(target)
        {
            remaining -= 1;
            continue;
        }
        let mut selected = None;
        for (index, arc) in graph.arcs.iter().enumerate() {
            *work = work.checked_sub(1)?;
            if arc.tail == vertex
                && bounds
                    .suffix
                    .get(remaining - 1)?
                    .get(arc.head)
                    .copied()
                    .flatten()
                    .is_some_and(|tail| bounds.arc_costs[index] + tail == target)
            {
                selected = Some(index);
                break;
            }
        }
        let index = selected?;
        arcs.push(index);
        vertex = graph.arcs[index].head;
        remaining -= 1;
    }
    // The walk DP drops scalar/coupled restrictions. Only an independently
    // feasible, source-certified control can establish that its bound is
    // attained; otherwise retain the complete-path refinement fallback.
    let mut iterations = 100_000;
    let result = path(
        oracle,
        offset,
        duals,
        &arcs,
        work,
        &mut iterations,
        deadline,
    )??;
    let route = result
        .route?
        .with_paired_controls(graph, sorted(result.greedy))
        .ok()?;
    if !lifecycle::integer_route_feasible(oracle, &route).ok()? {
        return None;
    }
    let upper = route_upper(oracle, offset, duals, &arcs, &route, work, deadline)?;
    (upper <= lower).then_some(Some(lower))
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/additive_control_bounds_tests.rs"]
mod tests;
