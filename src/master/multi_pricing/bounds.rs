//! Conservative original-graph walk relaxations for a network pricing batch.
//! Resource-derived hop limits can bound positive-consumption cycles. Cycles
//! without a justified finite hop limit leave the certificate unknown.
use super::*;

use crate::master::rounding::{add_down as add, multiply_down as multiply};
mod cutoff;
mod presence;
mod resource;
mod resource_groups;
pub(super) use cutoff::{cutoff_plan, NetworkCutoff};

fn subtract_product(value: f64, a: f64, b: f64) -> Option<f64> {
    // Negate an input so the subtraction itself is rounded downward.
    add(value, multiply(-a, b)?)
}

fn arc_cost_lower_bounds<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
) -> Option<Vec<f64>> {
    if oracle
        .resource_groups
        .iter()
        .any(|&(row, _)| duals[offset + row] > 0.)
    {
        return None;
    }
    let graph = &oracle.network.graph;
    let mut weights = vec![];
    for (index, (arc, coefficients)) in graph
        .arcs
        .iter()
        .zip(&oracle.network.arc_entries)
        .enumerate()
    {
        let mut weight = arc.cost;
        for &(row, coefficient) in coefficients {
            weight = subtract_product(weight, duals[offset + row], coefficient)?;
        }
        // Registered nonnegative coefficients have a proved per-arc bound
        // on their increments, including vertex and arc-weight carries.
        for &(cut, row) in &oracle.nonlinear_rows {
            let dual = duals[offset + row];
            if dual > 0. {
                weight = subtract_product(
                    weight,
                    dual,
                    super::nonlinear::arc_upper(oracle, cut, index),
                )?;
            }
        }
        weights.push(weight);
    }
    Some(weights)
}

fn network_minimum<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    resource_work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    // Resource step rows have nonnegative coefficients and <= sense. Dropping
    // their nonnegative reduced-cost penalty is a valid lower relaxation.
    // A sign-incompatible dual cannot use this argument.
    if oracle
        .knapsack_rows
        .iter()
        .any(|&(_, row)| duals[offset + row] > 0.)
    {
        return None;
    }
    let domain = super::resource_groups::residual_domain_graph(oracle).ok()?;
    let graph = domain.as_ref();
    let positions = (graph.arcs.len() != oracle.network.graph.arcs.len()).then(|| {
        graph
            .arcs
            .iter()
            .map(|a| oracle.index[&a.id])
            .collect::<Vec<_>>()
    });
    let position = |a| positions.as_ref().map_or(a, |p| p[a]);
    // Only cycles on a source-to-sink walk can affect the relaxation.
    let mut reaches_sink = vec![false; graph.vertices.len()];
    reaches_sink[graph.sink] = true;
    for _ in 0..graph.vertices.len() {
        let mut changed = false;
        for (a, arc) in graph.arcs.iter().enumerate() {
            if oracle
                .proved_zero_arcs
                .get(position(a))
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            if reaches_sink[arc.head] && !reaches_sink[arc.tail] {
                reaches_sink[arc.tail] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let original_weights = arc_cost_lower_bounds(oracle, offset, duals)?;
    let weights: std::borrow::Cow<'_, [f64]> = match &positions {
        Some(positions) => {
            std::borrow::Cow::Owned(positions.iter().map(|&i| original_weights[i]).collect())
        }
        None => std::borrow::Cow::Borrowed(&original_weights),
    };
    let mut distances = vec![None; graph.vertices.len()];
    distances[graph.source] = Some(0.);
    let mut cyclic = false;
    for pass in 0..graph.vertices.len() {
        let mut changed = false;
        for (a, (arc, &weight)) in graph.arcs.iter().zip(weights.iter()).enumerate() {
            // These flags have an exact original-row activity proof. They are
            // separate from heuristic or reduced-cost pricing restrictions.
            if oracle
                .proved_zero_arcs
                .get(position(a))
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            if !reaches_sink[arc.head] {
                continue;
            }
            if let Some(tail) = distances[arc.tail] {
                let candidate = add(tail, weight)?;
                if distances[arc.head].map_or(true, |previous| candidate < previous) {
                    distances[arc.head] = Some(candidate);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
        if pass + 1 == graph.vertices.len() {
            cyclic = true;
        }
    }
    let mut minimum = match if cyclic {
        resource::minimum(graph, &weights, resource_work, deadline)?
    } else {
        distances[graph.sink]
    } {
        Some(value) => value,
        None => return Some(None),
    };
    if !oracle.resource_groups.is_empty() {
        match resource_groups::minimum(graph, &weights, resource_work, deadline) {
            Some(Some(value)) => minimum = minimum.max(value),
            Some(None) => return Some(None),
            None => {}
        }
    }
    // Keep all active presence rows jointly, so event overlap, duplicate pairs
    // and repeated emissions cannot be priced independently by mistake.
    let inputs: Vec<_> = oracle
        .presence_rows
        .iter()
        .filter_map(|&(input, row)| {
            let dual = duals[offset + row];
            (dual != 0.).then_some((&oracle.solver.presence_rows()[input], dual))
        })
        .collect();
    let joint = if inputs.is_empty() {
        None
    } else {
        // A state expansion cannot consume this batch's entire resource budget.
        let allowance = (*resource_work / 4).min(100_000);
        let mut remaining = allowance;
        let result = presence::minimum(
            graph,
            &weights,
            &inputs,
            |a| {
                oracle
                    .proved_zero_arcs
                    .get(position(a))
                    .copied()
                    .unwrap_or(false)
            },
            &mut remaining,
            deadline,
        );
        *resource_work -= allowance - remaining;
        result
    };
    for &(input, row) in &oracle.presence_rows {
        let dual = duals[offset + row];
        if dual > 0. {
            minimum = subtract_product(
                minimum,
                dual,
                oracle.solver.presence_rows()[input]
                    .row
                    .maximum_coefficient() as f64,
            )?;
        }
    }
    match joint {
        Some(Some(bound)) => minimum = minimum.max(bound),
        Some(None) => return Some(None),
        None => {}
    }
    for &(row, coefficient) in &oracle.route_entries {
        minimum = subtract_product(minimum, duals[row], coefficient)?;
    }
    for &(row, coefficient) in &oracle.network.route_entries {
        minimum = subtract_product(minimum, duals[offset + row], coefficient)?;
    }
    Some(Some(minimum))
}

pub(super) fn certificate<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    duals: &[f64],
) -> Option<f64> {
    certificate_inner(state, master, duals, true)
}

/// A split-domain certificate whose original-variable caps remain valid when
/// external columns are added. Known-column projections cannot justify caps.
pub(super) fn extension_certificate<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    duals: &[f64],
) -> Option<f64> {
    certificate_inner(state, master, duals, false)
}

fn master_dual_base<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    duals: &[f64],
    allow_projection: bool,
) -> Option<f64> {
    if !exact_fixed_reduction(state, master).unwrap_or(false) {
        return None;
    }
    if duals.len() != state.rhs.len() || duals.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut value = master.objective_offset;
    for ((&dual, &rhs), sense) in duals.iter().zip(&state.rhs).zip(&master.senses) {
        if (*sense == RowSense::LessEqual && dual > 0.)
            || (*sense == RowSense::GreaterEqual && dual < 0.)
        {
            return None;
        }
        value = add(value, multiply(dual, rhs)?)?;
    }
    // Static master variables have semantic component bounds. Their
    // negative reduced costs require compensation even though no oracle prices
    // them. An infinite adverse endpoint leaves this certificate unknown;
    // backend infinity sentinels never turn it into a finite correction.
    let mut projected = None;
    for (index, variable) in state.master_variables.iter().enumerate() {
        for sign in [1., -1.] {
            let mut reduced = sign * variable.cost;
            for &(row, a) in &variable.entries {
                reduced = subtract_product(reduced, duals[state.shared_offset + row], sign * a)?;
            }
            if reduced < 0. {
                let mut cap = super::master_variables::upper(variable, sign);
                if !cap.is_finite() {
                    if !allow_projection {
                        return None;
                    }
                    if projected.is_none() {
                        projected = Some(super::master_variables::projected_bounds(state)?);
                    }
                    cap = super::master_variables::component_upper(
                        projected.as_ref().unwrap()[index],
                        sign,
                    );
                }
                value = add(value, multiply(reduced, cap)?)?;
            }
        }
    }
    Some(value)
}

fn certificate_inner<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    duals: &[f64],
    allow_projection: bool,
) -> Option<f64> {
    let mut value = master_dual_base(state, master, duals, allow_projection)?;
    if state.universe.is_some() {
        return add(value, super::route_bounds::certificate(state, duals)?);
    }
    // Shared across this batch: one difficult network cannot allocate an
    // unbounded amount of resource-aware certification work per oracle.
    let mut resource_work = 1_000_000;
    for oracle in &state.oracles {
        let limits = oracle.network.multiplicity;
        if oracle.multiplicity_upper() == 0. {
            continue;
        }
        let minimum = network_minimum(
            oracle,
            state.shared_offset,
            duals,
            &mut resource_work,
            state.deadline,
        )?;
        match minimum {
            None if limits.lower > 0. => return None,
            None => {}
            Some(minimum) => {
                let quantity = if minimum < 0. {
                    oracle.multiplicity_upper()
                } else {
                    limits.lower
                };
                value = add(value, multiply(minimum, quantity)?)?;
            }
        }
    }
    Some(value)
}

// Full original-model contributions shared by witness replay and validation of
// fixed-route reductions. Never trust cached rounded path costs or resources.
type Contributions = (
    Vec<crate::master::root::ExactSum>,
    crate::master::root::ExactSum,
    Vec<crate::master::root::ExactSum>,
);
pub(super) fn exact_contributions<const D: usize>(
    state: &State<D>,
    routes: &[FixedRoute],
) -> Result<Option<Contributions>> {
    exact_contributions_in_domain(state, routes, true)
}
fn exact_contributions_in_domain<const D: usize>(
    state: &State<D>,
    routes: &[FixedRoute],
    branch_domain: bool,
) -> Result<Option<Contributions>> {
    use crate::master::root::ExactSum;
    let mut rows = vec![ExactSum::default(); state.shared_rows.len()];
    let mut objective = ExactSum::default();
    let mut counts = vec![ExactSum::default(); state.oracles.len()];
    for fixed in routes {
        let index = state
            .oracles
            .iter()
            .position(|o| o.network.id == fixed.route.network)
            .ok_or_else(|| invalid("unknown witness network"))?;
        let oracle = &state.oracles[index];
        let feasible = if branch_domain {
            super::lifecycle::integer_route_feasible(oracle, &fixed.route.route)?
        } else {
            super::lifecycle::original_route_feasible(oracle, &fixed.route.route)?
        };
        if !feasible {
            return Ok(None);
        }
        if counts[index].add(fixed.quantity).is_none() {
            return Ok(None);
        }
        for &(row, coefficient) in &oracle.network.route_entries {
            if rows[row].product(coefficient, fixed.quantity).is_none() {
                return Ok(None);
            }
        }
        for (row, coefficient) in super::nonlinear::entries(oracle, &fixed.route.route)? {
            if rows[row].product(coefficient, fixed.quantity).is_none() {
                return Ok(None);
            }
        }
        for id in &fixed.route.route.arcs {
            let arc = oracle.index[id];
            if objective
                .product(oracle.network.graph.arcs[arc].cost, fixed.quantity)
                .is_none()
            {
                return Ok(None);
            }
            for &(row, coefficient) in &oracle.network.arc_entries[arc] {
                if rows[row].product(coefficient, fixed.quantity).is_none() {
                    return Ok(None);
                }
            }
        }
    }
    Ok(Some((rows, objective, counts)))
}

// A rounded RHS or offset after fixing is not an exact model reduction. Keep
// its original-domain certificate unknown instead of certifying the rounded LP.
pub(super) fn exact_fixed_reduction<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
) -> Result<bool> {
    if state.fixed_routes.is_empty() {
        return Ok(true);
    }
    let Some((rows, mut objective, counts)) =
        exact_contributions_in_domain(state, &state.fixed_routes, false)?
    else {
        return Ok(false);
    };
    if objective.add(-master.objective_offset).is_none() || objective.sign() != 0. {
        return Ok(false);
    }
    for (i, mut activity) in rows.into_iter().enumerate() {
        if activity.add(state.rhs[state.shared_offset + i]).is_none()
            || activity.add(-state.shared_rows[i].rhs).is_none()
            || activity.sign() != 0.
        {
            return Ok(false);
        }
    }
    for (count, oracle) in counts.iter().zip(&state.oracles) {
        for &(row, _) in &oracle.route_entries {
            let original = if master.senses[row] == RowSense::LessEqual {
                oracle.total_multiplicity.upper
            } else {
                oracle.total_multiplicity.lower
            };
            let mut difference = count.clone();
            if difference.add(state.rhs[row]).is_none()
                || difference.add(-original).is_none()
                || difference.sign() != 0.
            {
                return Ok(false);
            }
        }
        if oracle.total_multiplicity.upper.is_finite() {
            let mut upper = count.clone();
            if upper.add(oracle.network.multiplicity.upper).is_none()
                || upper.add(-oracle.total_multiplicity.upper).is_none()
                || upper.sign() != 0.
            {
                return Ok(false);
            }
        } else if oracle.network.multiplicity.upper != f64::INFINITY {
            return Ok(false);
        }
        let mut lower = count.clone();
        if lower.add(oracle.network.multiplicity.lower).is_none()
            || lower.add(-oracle.total_multiplicity.lower).is_none()
        {
            return Ok(false);
        }
        if oracle.network.multiplicity.lower > 0. && lower.sign() != 0.
            || oracle.network.multiplicity.lower == 0. && lower.sign() < 0.
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounded_fixed_rhs_cannot_supply_a_new_certificate() {
        let mut graph = Graph::<2>::new(3, 1).unwrap();
        graph.add_arc(0, 1, 1.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        let problem = Problem {
            rows: vec![Row {
                sense: RowSense::Equal,
                rhs: 1.,
            }],
            networks: vec![Network {
                id: 1,
                graph,
                multiplicity: Window::new(0., 20.).unwrap(),
                arc_entries: vec![vec![(0, 0.1)], vec![]],
                route_entries: vec![],
            }],
        };
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.artificial_cost = 100.;
        let root = solve(problem, options).unwrap();
        assert_eq!(root.master.status, cg::Status::Optimal);
        let node = root.snapshot.fix_route(0, 1.).unwrap();
        // f64(0.9) + f64(0.1) differs from one in exact arithmetic.
        assert!(!exact_fixed_reduction(&node.state, &node.master).unwrap());
        assert!(certificate(&node.state, &node.master, &root.duals).is_none());
    }
}

#[cfg(test)]
mod extension_tests {
    use super::*;
    use crate::master::preprocessing::Bounds;
    #[test]
    fn open_import_certificates_do_not_use_known_column_projected_caps() {
        let mut options = Options::default();
        options.preprocessing.enabled = false;
        let result = imported::solve(
            Problem::<2> {
                rows: vec![],
                networks: vec![],
            },
            vec![MasterVariable {
                id: 0,
                cost: -1.,
                bounds: Bounds {
                    lower: 0.,
                    upper: f64::INFINITY,
                },
                integer: false,
                entries: vec![],
            }],
            vec![imported::Subproblem {
                id: 7,
                variables: vec![0],
                multiplicity: Bounds {
                    lower: 0.,
                    upper: 1.,
                },
                columns: vec![imported::Column {
                    id: 0,
                    cost: 2.,
                    bounds: Bounds {
                        lower: 0.,
                        upper: f64::INFINITY,
                    },
                    integer: false,
                    entries: vec![(0, 1.)],
                }],
            }],
            options,
        )
        .unwrap();
        let node = &result.output.snapshot;
        assert!(node.state.master_variables[0].bounds.upper.is_infinite());
        let duals = vec![0.; node.state.rhs.len()];
        // x<=1 follows only from currently known lambda vectors. An unseen
        // vector x=100, cost=0 could yield objective -100 at the same total cap.
        assert_eq!(certificate(&node.state, &node.master, &duals), Some(-1.));
        assert_eq!(
            extension_certificate(&node.state, &node.master, &duals),
            None
        );
    }
}
