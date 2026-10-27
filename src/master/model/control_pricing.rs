//! Paired objective/shared-row pricing over independently bounded source controls.
//! Physical paths are covered by the common finite-domain collector. LP cells
//! bound each terminal coordinate; affine symbols retain objective/row coupling.
use super::*;
use crate::graph::coupled::{CoupledResource, PairedControl};
use crate::master::controls::{
    affine::{Expression, Interval},
    query, terminal_domain, ControlOptimization, ControlSearchLimits, TerminalDomain,
};
use crate::master::LpMaster;
use bounds::columns::{components, physical_paths, SourceColumn};

pub(super) fn needed<const D: usize>(oracle: &Oracle<D>) -> bool {
    oracle
        .network
        .graph
        .coupled_kernel()
        .is_some_and(|kernel| kernel.has_negative_paired_objectives())
        || oracle.activity_rows.iter().any(|c| {
            c.coefficient != 0. && activity_rows::paired(&oracle.network.graph, c.resource)
        })
}

fn target_control<const D: usize>(
    oracle: &Oracle<D>,
    path: &[usize],
    resource: usize,
    value: f64,
    work: &mut usize,
    iterations: &mut usize,
    deadline: Instant,
) -> Option<ControlOptimization> {
    let mut limits = ControlSearchLimits {
        max_cells: 4096,
        max_work: *work,
        max_lp_iterations: *iterations,
        deadline: Some(deadline),
    };
    let result = query(
        &oracle.network.graph,
        path,
        resource,
        false,
        Some(Window {
            lower: value,
            upper: value,
        }),
        &mut limits,
    );
    *work = limits.max_work;
    *iterations = limits.max_lp_iterations;
    result
}
fn domain<const D: usize>(
    oracle: &Oracle<D>,
    path: &[usize],
    resource: usize,
    work: &mut usize,
    iterations: &mut usize,
    deadline: Instant,
) -> Option<Option<TerminalDomain>> {
    let mut limits = ControlSearchLimits {
        max_cells: 4096,
        max_work: *work,
        max_lp_iterations: *iterations,
        deadline: Some(deadline),
    };
    let result = terminal_domain(&oracle.network.graph, path, resource, &mut limits);
    *work = limits.max_work;
    *iterations = limits.max_lp_iterations;
    result
}

pub(super) fn resources<const D: usize>(oracle: &Oracle<D>) -> Vec<usize> {
    let graph = &oracle.network.graph;
    let Some(kernel) = graph.coupled_kernel() else {
        return vec![];
    };
    let mut resources: Vec<_> = oracle
        .activity_rows
        .iter()
        .filter(|c| c.coefficient != 0. && activity_rows::paired(graph, c.resource))
        .map(|c| c.resource)
        .collect();
    resources.extend(
        kernel
            .definitions()
            .iter()
            .enumerate()
            .filter_map(|(r, _)| {
                (matches!(
                    kernel.definitions()[r],
                    CoupledResource::SlopePaired { .. }
                        | CoupledResource::PairedPiecewiseLinear { .. }
                ) && kernel.objective_coefficient(r)? < 0.)
                    .then_some(r)
            }),
    );
    resources.sort_by_key(|&r| {
        (
            !matches!(kernel.definitions()[r], CoupledResource::SlopePaired { .. }),
            r,
        )
    });
    resources.dedup();
    resources
}

/// Direction that weakly improves the objective and every Gamma-dependent row.
/// Repeated terms are checked individually to preserve ordered row arithmetic.
pub(super) fn endpoint_direction<const D: usize>(
    oracle: &Oracle<D>,
    resource: usize,
    rows: &[Row],
    pending: &[AccumulatedActivityEntry],
) -> Option<bool> {
    let q = oracle
        .network
        .graph
        .coupled_kernel()?
        .objective_coefficient(resource)?;
    let mut increase = q <= 0.;
    let mut decrease = q >= 0.;
    for c in oracle
        .activity_rows
        .iter()
        .chain(pending)
        .filter(|c| c.resource == resource && c.coefficient != 0.)
    {
        match rows.get(c.row)?.sense {
            RowSense::LessEqual => {
                if c.coefficient > 0. {
                    increase = false;
                } else {
                    decrease = false;
                }
            }
            RowSense::GreaterEqual => {
                if c.coefficient > 0. {
                    decrease = false;
                } else {
                    increase = false;
                }
            }
            RowSense::Equal => {
                increase = false;
                decrease = false;
            }
        }
    }
    if increase {
        Some(true)
    } else if decrease {
        Some(false)
    } else {
        None
    }
}

pub(super) struct EnumerationControls {
    pub controls: Vec<PairedControl>,
    pub complete: bool,
}
/// Retain checked invariant controls or attained dominating objective endpoints.
/// The collector separately proves whole-path feasibility before accepting
/// preservation of feasibility and the best objective on that path.
pub(super) fn enumeration_controls<const D: usize>(
    oracle: &Oracle<D>,
    path: &[usize],
    work: &mut usize,
    iterations: &mut usize,
    deadline: Instant,
    allow_endpoint_projection: bool,
    rows: &[Row],
) -> Option<Option<EnumerationControls>> {
    let mut controls = vec![];
    let mut complete = true;
    let kernel = oracle.network.graph.coupled_kernel()?;
    for resource in resources(oracle) {
        let Some(domain) = domain(oracle, path, resource, work, iterations, deadline)? else {
            return Some(None);
        };
        if domain.lower == domain.upper {
            controls.push(domain.minimum);
            continue;
        }
        // Independent paired programs can move to an attained endpoint when
        // the objective and every row agree on the direction. In-place charge
        // and row accumulation retain monotonicity under nearest rounding.
        if allow_endpoint_projection {
            if let Some(maximize) = endpoint_direction(oracle, resource, rows, &[]) {
                let (control, value) = if maximize {
                    (domain.maximum, domain.upper)
                } else {
                    (domain.minimum.clone(), domain.lower)
                };
                let point = kernel
                    .certify_paired_control(path, &control, work, Some(deadline))
                    .ok()??;
                if point.certifies_value(value) {
                    controls.push(control);
                    continue;
                }
            }
        }
        complete = false;
        controls.push(domain.minimum);
    }
    Some(Some(EnumerationControls {
        controls: sorted(controls),
        complete,
    }))
}

struct PathPricing {
    lower: f64,
    route: Option<Route>,
    resources: Vec<usize>,
    domains: Vec<Interval>,
    minimum: Vec<PairedControl>,
    maximum: Vec<PairedControl>,
    greedy: Vec<PairedControl>,
}

// Propose one jointly row-feasible unit column. This small terminal relaxation
// supplies targets only; every coordinate must subsequently have a replayed
// source control. Its LP status/objective never certifies pricing or feasibility.
fn joint_controls<const D: usize>(
    state: &State<D>,
    oracle: &Oracle<D>,
    path: &[usize],
    priced: &PathPricing,
    initial: &Route,
    quantity: f64,
    budget: (&mut usize, &mut usize),
) -> Option<Vec<PairedControl>> {
    use crate::master::clp::{ClpMaster, ClpSolveOutcome};
    let (work, iterations) = budget;
    if !quantity.is_finite() || quantity < 1. || quantity.fract() != 0. {
        return None;
    }
    let entries: BTreeMap<_, _> = entries(oracle, initial, state.shared_offset)
        .ok()?
        .into_iter()
        .collect();
    let n = priced.resources.len();
    if n < 2 || *iterations == 0 || Instant::now() >= state.deadline {
        return None;
    }
    *work = work.checked_sub(n.checked_mul(oracle.activity_rows.len().checked_add(1)?)?)?;
    let mut rows: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for entry in &oracle.activity_rows {
        let Some(i) = priced.resources.iter().position(|&r| r == entry.resource) else {
            continue;
        };
        rows.entry(entry.row).or_insert_with(|| vec![0.; n])[i] += entry.coefficient;
    }
    let mut master = ClpMaster::new().ok()?;
    for domain in &priced.domains {
        let column = master.add_column(0., &[]).ok()?;
        master.set_upper(column, domain.hi - domain.lo).ok()?;
    }
    for (row, coefficients) in rows {
        if coefficients.iter().all(|&a| a == 0.) {
            continue;
        }
        let index = state.shared_offset + row;
        let mut rhs =
            *state.rhs.get(index)? / quantity - entries.get(&index).copied().unwrap_or(0.);
        for (i, &a) in coefficients.iter().enumerate() {
            let current = initial
                .coupled_values
                .iter()
                .find(|(r, _)| *r == priced.resources[i])?
                .1;
            rhs += a * (current - priced.domains[i].lo);
        }
        let coefficients: Vec<_> = coefficients
            .into_iter()
            .enumerate()
            .filter(|(_, a)| *a != 0.)
            .collect();
        master
            .add_row(state.shared_rows.get(row)?.sense, rhs, &coefficients)
            .ok()?;
    }
    let before = master.simplex_iterations;
    let solved = master
        .solve_limited(
            *iterations,
            state.deadline.saturating_duration_since(Instant::now()),
        )
        .ok()?;
    *iterations =
        iterations.checked_sub(master.simplex_iterations.saturating_sub(before).max(1))?;
    if solved != ClpSolveOutcome::Optimal {
        return None;
    }
    let mut controls = vec![];
    for (i, &resource) in priced.resources.iter().enumerate() {
        let target = priced.domains[i].lo + master.values[i];
        let proposal = target_control(
            oracle,
            path,
            resource,
            target,
            work,
            iterations,
            state.deadline,
        )?;
        controls.push(proposal.control?);
    }
    Some(controls)
}
fn sorted(mut controls: Vec<PairedControl>) -> Vec<PairedControl> {
    controls.sort_by_key(|c| c.resource);
    controls
}

/// Recover an original integer solution from fractional controls on integer
/// physical-path counts. Only registered, replayable source points are tried;
/// failure or budget exhaustion supplies no bound or infeasibility conclusion.
pub(super) fn repair_witness<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    metadata: impl Fn(usize) -> Option<i32>,
    tolerance: f64,
) -> Result<Option<IntegerWitness>> {
    if !master.solved() || Instant::now() >= state.deadline {
        return Ok(None);
    }
    let mut groups = BTreeMap::<(i32, Vec<i32>), (f64, bool, Vec<FixedRoute>)>::new();
    let mut masters: Vec<_> = state.master_variables.iter().map(|v| (v.id, 0.)).collect();
    for (column, &value) in master.values().iter().enumerate() {
        if !value.is_finite()
            || !(0. ..=9_007_199_254_740_992.).contains(&value)
            || value > master.upper[column]
        {
            return Ok(None);
        }
        if let Some((index, sign)) = metadata(column).and_then(master_variables::decode) {
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
        let Some(route) = metadata(column)
            .and_then(|tag| usize::try_from(tag).ok())
            .and_then(|tag| state.routes.get(tag))
        else {
            return Ok(None);
        };
        let group = groups
            .entry((route.network, route.route.arcs.clone()))
            .or_default();
        group.0 += value;
        group.1 |= (value - value.round()).abs() > tolerance;
        group.2.push(FixedRoute {
            route: route.as_ref().clone(),
            quantity: value.round(),
        });
    }
    let mut witness = state.fixed_routes.clone();
    let mut alternatives = vec![];
    for ((network, arcs), (quantity, fractional, routes)) in groups {
        if !fractional {
            witness.extend(routes);
            continue;
        }
        let rounded = quantity.round();
        if (quantity - rounded).abs() > tolerance
            || rounded <= 0.
            || rounded > 9_007_199_254_740_992.
        {
            return Ok(None);
        }
        let choices: Vec<_> = state
            .routes
            .iter()
            .enumerate()
            .filter(|(tag, route)| {
                route.network == network
                    && route.route.arcs == arcs
                    && !route.route.coupled_controls.is_empty()
                    && state
                        .universe
                        .as_ref()
                        .map_or(true, |universe| universe.contains(tag))
                    && enumeration::route_upper(state, *tag).is_some_and(|cap| rounded <= cap)
            })
            .map(|(_, route)| FixedRoute {
                route: route.as_ref().clone(),
                quantity: rounded,
            })
            .collect();
        if choices.is_empty() || alternatives.len() >= 128 {
            return Ok(None);
        }
        alternatives.push(choices);
    }
    if alternatives.is_empty() {
        return Ok(None);
    }
    let mut positions = vec![0; alternatives.len()];
    for _ in 0..512 {
        if Instant::now() >= state.deadline {
            return Ok(None);
        }
        let mut routes = witness.clone();
        routes.extend(
            alternatives
                .iter()
                .zip(&positions)
                .map(|(choices, &i)| choices[i].clone()),
        );
        if let Some(objective) = tree_adapter::validate_routes(state, &routes, &masters)? {
            return Ok(Some((objective, routes, masters)));
        }
        let mut carry = true;
        for (i, choices) in positions.iter_mut().zip(&alternatives).rev() {
            *i += 1;
            if *i < choices.len() {
                carry = false;
                break;
            }
            *i = 0;
        }
        if carry {
            break;
        }
    }
    Ok(None)
}
fn hull(a: Interval, b: f64) -> Option<Interval> {
    b.is_finite().then_some(Interval {
        lo: a.lo.min(b),
        hi: a.hi.max(b),
    })
}

// Outer None is Unknown; inner None is independently proved infeasibility.
fn path<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    arcs: &[usize],
    work: &mut usize,
    iterations: &mut usize,
    deadline: Instant,
) -> Option<Option<PathPricing>> {
    let graph = &oracle.network.graph;
    let mut scalar = resource_groups::domain_graph(oracle).ok()?.into_owned();
    scalar.set_coupled_resources(&[]).ok()?;
    let problem = crate::master::routing::RoutingContext::set_partitioning(scalar, 1.).ok()?;
    let ids: Vec<_> = arcs.iter().map(|&p| graph.arcs[p].id).collect();
    let skeleton = Route {
        nodes: std::iter::once(graph.source as i32)
            .chain(arcs.iter().map(|&p| graph.arcs[p].head as i32))
            .collect(),
        arcs: ids.clone(),
        cost: 0.,
        original_cost: 0.,
        resources: vec![],
        resource_profile: vec![],
        coupled_values: vec![],
        coupled_controls: vec![],
    };
    if crate::master::routing::exact_scalar_path_feasibility(&problem, &skeleton) == Some(false) {
        return Some(None);
    }
    let route = problem.elementary_route(&ids);
    let kernel = graph.coupled_kernel()?;
    let resources = resources(oracle);
    let mut domains = vec![];
    let mut minimum = vec![];
    let mut maximum = vec![];
    for &r in &resources {
        let Some(domain) = domain(oracle, arcs, r, work, iterations, deadline)? else {
            return Some(None);
        };
        domains.push(Interval {
            lo: domain.lower,
            hi: domain.upper,
        });
        minimum.push(domain.minimum);
        maximum.push(domain.maximum);
    }
    // Match ordinary replay: numeric/front prefix, slope pairs, then knot pairs.
    // Every paired charge is removed from the prefix before being inserted at
    // its own position; witness metadata must not change parenthesization.
    let mut paired_order: Vec<_> = kernel
        .definitions()
        .iter()
        .enumerate()
        .filter_map(|(r, definition)| match definition {
            CoupledResource::SlopePaired { mode, .. }
            | CoupledResource::PairedPiecewiseLinear { mode, .. }
                if *mode != crate::graph::coupled::PairedResourceMode::Feasibility =>
            {
                Some(r)
            }
            _ => None,
        })
        .collect();
    paired_order.sort_by_key(|&r| {
        (
            !matches!(kernel.definitions()[r], CoupledResource::SlopePaired { .. }),
            r,
        )
    });
    let mut base = graph.as_ref().clone();
    let weights: Vec<_> = kernel
        .definitions()
        .iter()
        .enumerate()
        .filter_map(|(r, _)| {
            let q = kernel.objective_coefficient(r)?;
            (paired_order.contains(&r) || q != 1.)
                .then_some((r, if paired_order.contains(&r) { 0. } else { q }))
        })
        .collect();
    base.set_coupled_objective_coefficients(&weights).ok()?;
    let controls = sorted(minimum.clone());
    let source = SourceColumn {
        kernel: base.coupled_kernel()?,
        controls: &controls,
        variable_resources: &resources,
    };
    let terms = components(oracle, offset, duals, arcs, work, deadline, source)?;
    let Some(mut terms) = terms else {
        return Some(None);
    };
    let numeric = base
        .coupled_kernel()?
        .replay_with_controls_using(arcs, &controls, work, Some(deadline))
        .ok()??;
    terms.coupled = hull(terms.coupled, numeric.cost)?;
    // Enclose both source coefficients and the actual registered constant terms.
    if let Some(route) = route.as_ref() {
        for (r, value) in entries_base(oracle, route, offset).ok()? {
            let old = terms.rows.entry(r).or_insert(Interval::point(0.));
            *old = hull(*old, value)?;
        }
    }
    let n = resources.len();
    let numeric_components = if paired_order.iter().any(|r| !resources.contains(r)) {
        *work = work.checked_sub(
            arcs.len()
                .checked_mul(kernel.definitions().len().checked_add(1)?)?,
        )?;
        Some(kernel.replay_cost_components(arcs).ok()??)
    } else {
        None
    };
    let mut coupled = Expression::constant(terms.coupled, n);
    for r in paired_order {
        let charge = if let Some(i) = resources.iter().position(|&resource| resource == r) {
            let coefficient = kernel.effective_objective_coefficient(r)?;
            let mut value = Expression::variable(i, n);
            if coefficient != 0.
                && matches!(
                    kernel.definitions()[r],
                    CoupledResource::SlopePaired {
                        normalization:
                            crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum,
                        ..
                    } | CoupledResource::PairedPiecewiseLinear {
                        normalization:
                            crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum,
                        ..
                    }
                )
            {
                let baseline = kernel.paired_initial_minimum(r, work, Some(deadline))?;
                value = value.add(
                    &Expression::constant(
                        Interval {
                            lo: -baseline[1],
                            hi: -baseline[0],
                        },
                        n,
                    ),
                    &domains,
                )?;
            }
            value.scale(coefficient, &domains)?
        } else {
            let (valid, range) =
                kernel.certify_paired_objective_component(arcs, r, work, Some(deadline))?;
            if !valid {
                return Some(None);
            }
            Expression::constant(
                hull(
                    Interval {
                        lo: range[0],
                        hi: range[1],
                    },
                    *numeric_components.as_ref()?.resources.get(r)?,
                )?,
                n,
            )
        };
        coupled = coupled.add(&charge, &domains)?;
    }
    let mut reduced = Expression::constant(terms.physical, n).add(&coupled, &domains)?;
    let mut rows: BTreeMap<_, _> = terms
        .rows
        .into_iter()
        .map(|(r, value)| (r, Expression::constant(value, n)))
        .collect();
    let mut paired = BTreeMap::<usize, Expression>::new();
    for c in &oracle.activity_rows {
        if c.coefficient == 0. {
            continue;
        }
        let Some(i) = resources.iter().position(|&r| r == c.resource) else {
            continue;
        };
        let value = Expression::variable(i, n).scale(c.coefficient, &domains)?;
        let row = paired
            .entry(offset + c.row)
            .or_insert_with(|| Expression::constant(Interval::point(0.), n));
        *row = row.add(&value, &domains)?;
    }
    for (r, value) in paired {
        let row = rows
            .entry(r)
            .or_insert_with(|| Expression::constant(Interval::point(0.), n));
        *row = row.add(&value, &domains)?;
    }
    for (r, value) in rows {
        reduced = reduced.add(&value.scale(-*duals.get(r)?, &domains)?, &domains)?;
    }
    let lower = reduced.range(&domains)?.lo;
    let greedy = resources
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if reduced.coefficient(i).hi < 0. {
                maximum[i].clone()
            } else {
                minimum[i].clone()
            }
        })
        .collect();
    Some(Some(PathPricing {
        lower,
        route,
        resources,
        domains,
        minimum,
        maximum,
        greedy,
    }))
}

pub(super) fn minimum<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    let paths = physical_paths(oracle, work, deadline)?;
    let mut iterations = 100_000;
    let mut lower: Option<f64> = None;
    for arcs in paths {
        if let Some(result) = path(
            oracle,
            offset,
            duals,
            &arcs,
            work,
            &mut iterations,
            deadline,
        )? {
            lower = Some(lower.map_or(result.lower, |old| old.min(result.lower)));
        }
    }
    (Instant::now() < deadline).then_some(lower)
}

pub(super) fn column_bound<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    arcs: &[usize],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    column_bound_using(oracle, offset, duals, arcs, work, &mut 100_000, deadline)
}
pub(super) fn column_bound_using<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    arcs: &[usize],
    work: &mut usize,
    iterations: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    Some(path(oracle, offset, duals, arcs, work, iterations, deadline)?.map(|p| p.lower))
}

pub(super) fn price<const D: usize>(
    state: &State<D>,
    index: usize,
    duals: &[f64],
) -> Result<(Vec<Route>, f64)> {
    let oracle = &state.oracles[index];
    let mut work = 4_000_000;
    let mut iterations = 100_000;
    let paths = physical_paths(oracle, &mut work, state.deadline)
        .ok_or_else(|| incomplete("paired source-control pricing path coverage incomplete"))?;
    let mut minimum = f64::INFINITY;
    let mut routes = vec![];
    let mut seen = HashSet::new();
    for arcs in paths {
        let result = path(
            oracle,
            state.shared_offset,
            duals,
            &arcs,
            &mut work,
            &mut iterations,
            state.deadline,
        )
        .ok_or_else(|| incomplete("paired source-control pricing control coverage incomplete"))?;
        let Some(result) = result else {
            continue;
        };
        minimum = minimum.min(result.lower);
        let Some(ref route) = result.route else {
            continue;
        };
        // Row boundary proposals enrich the restricted master. They are never
        // used to assert exhaustive control coverage or source infeasibility.
        let initial = route
            .clone()
            .with_paired_controls(&oracle.network.graph, sorted(result.greedy.clone()))?;
        let coefficients: BTreeMap<_, _> = entries(oracle, &initial, state.shared_offset)?
            .into_iter()
            .collect();
        let mut quantities = vec![1.];
        for count in [
            oracle.network.multiplicity.lower.ceil(),
            oracle.multiplicity_upper().floor(),
        ] {
            if count.is_finite() && (1. ..=9_007_199_254_740_992.).contains(&count) {
                quantities.push(count);
            }
        }
        quantities.sort_by(f64::total_cmp);
        quantities.dedup();
        let mut choices = vec![];
        for quantity in quantities {
            if let Some(controls) = joint_controls(
                state,
                oracle,
                &arcs,
                &result,
                &initial,
                quantity,
                (&mut work, &mut iterations),
            ) {
                choices.push(controls);
            }
        }
        choices.extend([
            result.minimum.clone(),
            result.maximum.clone(),
            result.greedy.clone(),
        ]);
        for (i, &resource) in result.resources.iter().enumerate() {
            for c in oracle
                .activity_rows
                .iter()
                .filter(|c| c.resource == resource && c.coefficient != 0.)
            {
                let r = state.shared_offset + c.row;
                let coefficient: f64 = oracle
                    .activity_rows
                    .iter()
                    .filter(|other| other.resource == resource && other.row == c.row)
                    .map(|other| other.coefficient)
                    .sum();
                let value = initial
                    .coupled_values
                    .iter()
                    .find(|(r, _)| *r == resource)
                    .map(|(_, v)| *v)
                    .unwrap_or(0.);
                let target = value
                    + (state.rhs[r] - coefficients.get(&r).copied().unwrap_or(0.)) / coefficient;
                if !target.is_finite()
                    || target < result.domains[i].lo
                    || target > result.domains[i].hi
                {
                    continue;
                }
                if let Some(proposal) = target_control(
                    oracle,
                    &arcs,
                    resource,
                    target,
                    &mut work,
                    &mut iterations,
                    state.deadline,
                ) {
                    if let Some(control) = proposal.control {
                        let mut selection = result.greedy.clone();
                        selection[i] = control;
                        choices.push(selection);
                    }
                }
            }
        }
        for controls in choices {
            let candidate = route
                .clone()
                .with_paired_controls(&oracle.network.graph, sorted(controls))?;
            if lifecycle::integer_route_feasible(oracle, &candidate)?
                && seen.insert(candidate.identity())
            {
                routes.push(candidate);
            }
        }
    }
    Ok((
        routes,
        if minimum == f64::INFINITY {
            0.
        } else {
            minimum
        },
    ))
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/joint_control_tests.rs"]
mod tests;
