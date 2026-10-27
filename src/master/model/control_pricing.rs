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
use bounds::columns::{components, SourceColumn};

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
        max_cells: usize::MAX,
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
        max_cells: usize::MAX,
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
    sign_ambiguous: Vec<usize>,
}

fn sign_ambiguous(coefficient: Interval) -> bool {
    coefficient.lo < 0. && coefficient.hi >= 0.
}

/// Certify an affine reduced-cost lower bound over each descriptor's complete
/// source-control regions. Paired descriptors own independent base-control
/// programs, so the affine objective separates by terminal resource after all
/// objective and row-dual terms have been combined in `reduced`. `domains`
/// comes from complete min/max source-control searches. Coefficient intervals
/// account for floating-point accumulation; checking both endpoints covers
/// every coefficient value represented by the enclosure without repeating
/// those source-control searches.
fn source_control_lower_bound(domains: &[Interval], reduced: &Expression) -> Option<f64> {
    if domains.len() != reduced.coefficient_count() {
        return None;
    }
    let mut lower = reduced.constant_interval().lo;
    for (index, &domain) in domains.iter().enumerate() {
        let coefficient = reduced.coefficient(index);
        let mut resource_lower = f64::INFINITY;
        for weight in [coefficient.lo, coefficient.hi] {
            if weight == 0. {
                resource_lower = resource_lower.min(0.);
                continue;
            }
            let signed_bound = if weight < 0. { -domain.hi } else { domain.lo };
            // A negative coefficient uses -upper(Gamma), a lower bound on
            // -Gamma. Multiplying by |weight| bounds weight*Gamma below.
            let contribution = crate::core::rounding::multiply_down(weight.abs(), signed_bound)?;
            resource_lower = resource_lower.min(contribution);
        }
        lower = crate::core::rounding::add_down(lower, resource_lower)?;
    }
    Some(lower)
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
) -> Option<Vec<Vec<PairedControl>>> {
    let (work, iterations) = budget;
    if !quantity.is_finite() || quantity < 1. || quantity.fract() != 0. {
        return None;
    }
    let entries: BTreeMap<_, _> = entries(oracle, initial, state.shared_offset)
        .ok()?
        .into_iter()
        .collect();
    let n = priced.resources.len();
    if n == 0 || *iterations == 0 || Instant::now() >= state.deadline {
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
    let mut constraints = vec![];
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
        constraints.push((state.shared_rows.get(row)?.sense, rhs, coefficients));
    }
    // A single paired resource needs joint projection only when multiple rows
    // constrain it. One-row targets are already proposed by `price` below.
    if n == 1 && constraints.len() < 2 {
        return None;
    }
    // A zero-cost feasible point can be an unattainable interior Gamma even
    // when an endpoint of the same equality-constrained face is source-
    // attainable. Sweep both coordinate directions (plus the original
    // feasibility point) to expose those bounded transfer candidates.
    let mut objectives = vec![vec![0.; n]];
    for i in 0..n {
        for cost in [-1., 1.] {
            let mut objective = vec![0.; n];
            objective[i] = cost;
            objectives.push(objective);
        }
    }
    // Keep the coordinate proposals from consuming the entire path budget;
    // reserve work both to enumerate joint vertices and to realize their
    // source controls. Reserve simplex iterations for vertex realization too,
    // so coordinate proposals cannot consume its entire source-query budget.
    let vertex_enumeration_reserve = *work / 3;
    let vertex_realization_reserve = *work / 3;
    let vertex_iteration_reserve = *iterations / 3;
    let mut vertex_iterations = vertex_iteration_reserve;
    let mut coordinate_iterations = iterations.checked_sub(vertex_iteration_reserve)?;
    *work =
        work.checked_sub(vertex_enumeration_reserve.checked_add(vertex_realization_reserve)?)?;
    let mut proposals = Vec::with_capacity(objectives.len());
    for objective in objectives {
        let Some(row_work) = oracle.activity_rows.len().checked_add(1) else {
            break;
        };
        let Some(proposal_work) = n.checked_mul(row_work) else {
            break;
        };
        let Some(remaining_work) = work.checked_sub(proposal_work) else {
            break;
        };
        *work = remaining_work;
        let Some(targets) = solve_terminal_targets(
            &priced.domains,
            &constraints,
            &objective,
            &mut coordinate_iterations,
            state.deadline,
        ) else {
            continue;
        };
        let mut controls = Vec::with_capacity(n);
        let mut attainable = true;
        for (i, &resource) in priced.resources.iter().enumerate() {
            let Some(proposal) = target_control(
                oracle,
                path,
                resource,
                targets[i],
                work,
                &mut coordinate_iterations,
                state.deadline,
            ) else {
                attainable = false;
                break;
            };
            let Some(control) = proposal.control else {
                attainable = false;
                break;
            };
            controls.push(control);
        }
        if attainable {
            proposals.push(controls);
        }
    }
    // Coordinate objectives miss vertices whose normal cone has no
    // coordinate-axis direction (for example, a three-resource simplex with
    // pairwise-sum rows). Enumerate bounded terminal-LP vertices as additional
    // witness targets. This is still proposal generation: each coordinate is
    // queried against its source domain and final rows are replayed later.
    let mut enumeration_work = vertex_enumeration_reserve;
    let vertices = terminal_vertices(
        &priced.domains,
        &constraints,
        &mut enumeration_work,
        state.deadline,
    );
    let mut vertex_realization_work = vertex_realization_reserve;
    for vertex in vertices {
        if vertex_realization_work == 0 || Instant::now() >= state.deadline {
            break;
        }
        let mut controls = Vec::with_capacity(n);
        let mut attainable = true;
        for (i, &resource) in priced.resources.iter().enumerate() {
            let target = priced.domains[i].lo + vertex[i];
            if !target.is_finite() {
                attainable = false;
                break;
            }
            let Some(proposal) = target_control(
                oracle,
                path,
                resource,
                target,
                &mut vertex_realization_work,
                &mut vertex_iterations,
                state.deadline,
            ) else {
                attainable = false;
                break;
            };
            let Some(control) = proposal.control else {
                attainable = false;
                break;
            };
            controls.push(control);
        }
        if attainable {
            proposals.push(controls);
        }
    }
    *work = work
        .checked_add(enumeration_work)?
        .checked_add(vertex_realization_work)?;
    *iterations = coordinate_iterations.checked_add(vertex_iterations)?;
    (!proposals.is_empty()).then_some(proposals)
}
// Both route-local proposals and cross-path witness recovery use the same
// shifted terminal-box LP. Its solution is a proposal, never a proof bound.
type TerminalRow = (RowSense, f64, Vec<(usize, f64)>);

// The cost optimum is useful for witness recovery, but it can land on a
// source-unattainable point of a coupled face. Also propose a feasible point
// selected by a zero objective and both extrema of every terminal
// coordinate. These targets are candidates only; source replay remains the
// acceptance test.
fn terminal_objective(costs: &[f64], proposal: usize) -> Option<Vec<f64>> {
    if proposal == 0 {
        return Some(costs.to_vec());
    }
    if proposal == 1 {
        return Some(vec![0.; costs.len()]);
    }
    let coordinate = (proposal - 2) / 2;
    let sign = if proposal % 2 == 0 { -1. } else { 1. };
    let mut objective = vec![0.; costs.len()];
    *objective.get_mut(coordinate)? = sign;
    Some(objective)
}

fn solve_terminal_targets(
    domains: &[Interval],
    rows: &[TerminalRow],
    costs: &[f64],
    iterations: &mut usize,
    deadline: Instant,
) -> Option<Vec<f64>> {
    use crate::master::clp::{ClpMaster, ClpSolveOutcome};
    if *iterations == 0 || Instant::now() >= deadline {
        return None;
    }
    let mut master = ClpMaster::new().ok()?;
    for (domain, &cost) in domains.iter().zip(costs) {
        let column = master.add_column(cost, &[]).ok()?;
        master.set_upper(column, domain.hi - domain.lo).ok()?;
    }
    for (sense, rhs, entries) in rows {
        master.add_row(*sense, *rhs, entries).ok()?;
    }
    let before = master.simplex_iterations;
    let solved = master
        .solve_limited(
            *iterations,
            deadline.saturating_duration_since(Instant::now()),
        )
        .ok()?;
    *iterations =
        iterations.checked_sub(master.simplex_iterations.saturating_sub(before).max(1))?;
    if solved != ClpSolveOutcome::Optimal {
        return None;
    }
    Some(
        domains
            .iter()
            .zip(master.values())
            .map(|(domain, v)| domain.lo + v)
            .collect(),
    )
}

/// Enumerate vertices of a bounded shifted terminal LP, up to shared work.
/// The returned points are numerical proposals only; callers must realize
/// every coordinate in the source model and validate the complete row set.
fn terminal_vertices(
    domains: &[Interval],
    rows: &[TerminalRow],
    work: &mut usize,
    deadline: Instant,
) -> Vec<Vec<f64>> {
    #[derive(Clone)]
    struct Constraint {
        coefficients: Vec<f64>,
        rhs: f64,
    }

    let n = domains.len();
    if n == 0 {
        return vec![];
    }
    let Some(bound_constraints) = n.checked_mul(2) else {
        return vec![];
    };
    let Some(setup_work) = rows
        .len()
        .checked_add(bound_constraints)
        .and_then(|constraints| constraints.checked_mul(n))
    else {
        return vec![];
    };
    let Some(remaining) = work.checked_sub(setup_work) else {
        return vec![];
    };
    *work = remaining;
    let mut constraints = Vec::with_capacity(rows.len() + n * 2);
    for &(_sense, rhs, ref entries) in rows {
        let mut coefficients = vec![0.; n];
        for &(index, coefficient) in entries {
            let Some(value) = coefficients.get_mut(index) else {
                return vec![];
            };
            *value += coefficient;
        }
        if coefficients.iter().any(|value| !value.is_finite()) || !rhs.is_finite() {
            return vec![];
        }
        constraints.push(Constraint { coefficients, rhs });
    }
    for (index, domain) in domains.iter().enumerate() {
        let width = domain.hi - domain.lo;
        if !width.is_finite() || width < 0. {
            return vec![];
        }
        let mut upper = vec![0.; n];
        upper[index] = 1.;
        constraints.push(Constraint {
            coefficients: upper,
            rhs: width,
        });
        let mut lower = vec![0.; n];
        lower[index] = -1.;
        constraints.push(Constraint {
            coefficients: lower,
            rhs: 0.,
        });
    }

    fn solve_active(active: &[&Constraint], n: usize) -> Option<Vec<f64>> {
        let mut matrix = vec![vec![0.; n + 1]; n];
        for (row, constraint) in active.iter().enumerate() {
            matrix[row][..n].copy_from_slice(&constraint.coefficients);
            matrix[row][n] = constraint.rhs;
        }
        for column in 0..n {
            let pivot = (column..n)
                .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))?;
            let scale = matrix[pivot][..n]
                .iter()
                .fold(0_f64, |largest, value| largest.max(value.abs()));
            if matrix[pivot][column].abs() <= 64. * f64::EPSILON * scale.max(1.) {
                return None;
            }
            matrix.swap(column, pivot);
            let divisor = matrix[column][column];
            for value in &mut matrix[column][column..=n] {
                *value /= divisor;
            }
            for row in 0..n {
                if row == column {
                    continue;
                }
                let factor = matrix[row][column];
                for index in column..=n {
                    matrix[row][index] -= factor * matrix[column][index];
                }
            }
        }
        let point: Vec<_> = matrix.into_iter().map(|row| row[n]).collect();
        point.iter().all(|value| value.is_finite()).then_some(point)
    }

    fn feasible(point: &[f64], domains: &[Interval], rows: &[TerminalRow]) -> bool {
        for (value, domain) in point.iter().zip(domains) {
            let width = domain.hi - domain.lo;
            let tolerance = 1e-9 * (1. + width.abs() + value.abs());
            if *value < -tolerance || *value > width + tolerance {
                return false;
            }
        }
        for &(sense, rhs, ref entries) in rows {
            let lhs = entries
                .iter()
                .map(|&(index, coefficient)| point[index] * coefficient)
                .sum::<f64>();
            if !lhs.is_finite() {
                return false;
            }
            let tolerance =
                1e-9 * (1. + rhs.abs() + entries.iter().map(|&(_, a)| a.abs()).sum::<f64>());
            match sense {
                RowSense::LessEqual if lhs > rhs + tolerance => return false,
                RowSense::GreaterEqual if lhs < rhs - tolerance => return false,
                RowSense::Equal if (lhs - rhs).abs() > tolerance => return false,
                _ => {}
            }
        }
        true
    }

    fn visit(
        start: usize,
        active: &mut Vec<usize>,
        constraints: &[Constraint],
        domains: &[Interval],
        rows: &[TerminalRow],
        work: &mut usize,
        deadline: Instant,
        vertices: &mut Vec<Vec<f64>>,
        seen: &mut std::collections::HashSet<Vec<u64>>,
    ) {
        if Instant::now() >= deadline || *work == 0 {
            return;
        }
        let n = domains.len();
        if active.len() == n {
            let Some(cost) = n.checked_mul(n).and_then(|value| value.checked_add(1)) else {
                return;
            };
            let Some(remaining) = work.checked_sub(cost) else {
                *work = 0;
                return;
            };
            *work = remaining;
            let selected: Vec<_> = active.iter().map(|&index| &constraints[index]).collect();
            let Some(point) = solve_active(&selected, n) else {
                return;
            };
            if !feasible(&point, domains, rows) {
                return;
            }
            let key: Vec<_> = point
                .iter()
                .map(|value| if *value == 0. { 0 } else { value.to_bits() })
                .collect();
            if seen.insert(key) {
                vertices.push(point);
            }
            return;
        }
        let needed = n - active.len();
        if constraints.len().saturating_sub(start) < needed {
            return;
        }
        for index in start..constraints.len() {
            active.push(index);
            visit(
                index + 1,
                active,
                constraints,
                domains,
                rows,
                work,
                deadline,
                vertices,
                seen,
            );
            active.pop();
            if Instant::now() >= deadline || *work == 0 {
                break;
            }
        }
    }

    let mut vertices = Vec::new();
    let mut seen = std::collections::HashSet::new();
    visit(
        0,
        &mut Vec::with_capacity(n),
        &constraints,
        domains,
        rows,
        work,
        deadline,
        &mut vertices,
        &mut seen,
    );
    vertices
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
    if let Some(result) = search_identity_groups(
        state,
        &allocation_groups,
        0,
        &mut routes,
        &masters,
        identity_search_deadline,
    )? {
        return Ok(Some(result));
    }
    Ok(joint_witness_targets(
        state,
        &witness,
        &allocation_groups,
        &masters,
    ))
}

/// Search registered identity allocations lazily. This avoids a fixed
/// identity shortlist and materializing the Cartesian product of all groups.
/// Exhausting the phase deadline is inconclusive.
fn search_identity_groups<const D: usize>(
    state: &State<D>,
    groups: &[(Vec<(usize, f64, NetworkRoute)>, f64)],
    index: usize,
    routes: &mut Vec<FixedRoute>,
    masters: &[(i32, f64)],
    deadline: Instant,
) -> Result<Option<IntegerWitness>> {
    if Instant::now() >= deadline {
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
        &mut |allocation, deadline| {
            let old_len = routes.len();
            routes.extend_from_slice(allocation);
            found = search_identity_groups(state, groups, index + 1, routes, masters, deadline)?;
            routes.truncate(old_len);
            Ok(found.is_some())
        },
    )?;
    Ok(found)
}

/// Enumerate bounded integral allocations of one physical-path count across
/// registered route identities, respecting each identity's residual cap.
#[cfg(test)]
fn identity_allocations<const D: usize>(
    state: &State<D>,
    choices: &[(usize, f64, NetworkRoute)],
    quantity: f64,
    limit: usize,
    deadline: Instant,
) -> Vec<Vec<FixedRoute>> {
    fn visit<const D: usize>(
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

/// Stream every bounded allocation to a caller without materializing a large
/// candidate list. A phase deadline bounds recursion across path groups.
fn for_each_identity_allocation<const D: usize>(
    state: &State<D>,
    choices: &[(usize, f64, NetworkRoute)],
    quantity: f64,
    deadline: Instant,
    on_allocation: &mut impl FnMut(&[FixedRoute], Instant) -> Result<bool>,
) -> Result<bool> {
    fn try_count<const D: usize>(
        state: &State<D>,
        choices: &[(usize, f64, NetworkRoute)],
        index: usize,
        remaining: f64,
        count: f64,
        route: &NetworkRoute,
        selected: &mut Vec<FixedRoute>,
        deadline: Instant,
        on_allocation: &mut impl FnMut(&[FixedRoute], Instant) -> Result<bool>,
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
            on_allocation,
        )?;
        if count > 0. {
            selected.pop();
        }
        Ok(stop)
    }

    fn visit<const D: usize>(
        state: &State<D>,
        choices: &[(usize, f64, NetworkRoute)],
        index: usize,
        remaining: f64,
        selected: &mut Vec<FixedRoute>,
        deadline: Instant,
        on_allocation: &mut impl FnMut(&[FixedRoute], Instant) -> Result<bool>,
    ) -> Result<bool> {
        if Instant::now() >= deadline {
            return Ok(false);
        }
        if remaining == 0. {
            return on_allocation(selected, deadline);
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
        on_allocation,
    )
}

// If registered points cannot realize the LP's integral physical path counts,
// solve all shared Gamma rows together, retaining fixed and integral background
// contributions. Each new terminal target still needs a source trajectory.
fn joint_witness_targets<const D: usize>(
    state: &State<D>,
    background: &[FixedRoute],
    allocation_groups: &[(Vec<(usize, f64, NetworkRoute)>, f64)],
    masters: &[(i32, f64)],
) -> Option<IntegerWitness> {
    if Instant::now() >= state.deadline {
        return None;
    }
    let mut work = 4_000_000;
    let mut iterations = 100_000;
    let mut seed = Vec::new();
    search_joint_witness_seeds(
        state,
        background,
        allocation_groups,
        0,
        &mut seed,
        masters,
        &mut work,
        &mut iterations,
    )
}

/// Enumerate fallback identity combinations lazily under a single shared
/// work, LP-iteration and solve-deadline budget.
fn search_joint_witness_seeds<const D: usize>(
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
fn for_each_joint_witness_seed<const D: usize>(
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
        &mut |allocation, _| {
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

fn joint_witness_targets_for_seed<const D: usize>(
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

type TerminalVariable = (usize, usize, Vec<usize>);
#[derive(Clone, Copy)]
struct TerminalTarget {
    average: f64,
    total: f64,
}

// A nonrepresentable average need not rule out an exact integer witness.
// Try integer copy blocks at dyadic neighboring Gamma values, then replay
// the complete source model; rounded aggregate targets never become proofs.
fn realize_joint_targets<const D: usize>(
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
fn dyadic_copy_controls<const D: usize>(
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

fn attained_control<const D: usize>(
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
fn scheduled_identities_allowed<const D: usize>(state: &State<D>, routes: &[FixedRoute]) -> bool {
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
    let lower = source_control_lower_bound(&domains, &reduced)?;
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
    let sign_ambiguous = (0..resources.len())
        .filter(|&index| sign_ambiguous(reduced.coefficient(index)))
        .collect();
    Some(Some(PathPricing {
        lower,
        route,
        resources,
        domains,
        minimum,
        maximum,
        greedy,
        sign_ambiguous,
    }))
}

pub(super) fn minimum<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    let mut iterations = 100_000;
    let mut lower: Option<f64> = None;
    bounds::columns::visit_physical_paths(oracle, work, deadline, |arcs, work| {
        let Some(result) = path(oracle, offset, duals, arcs, work, &mut iterations, deadline)
        else {
            return Err(crate::enumeration::PathDomainFailure::Arithmetic);
        };
        if let Some(result) = result {
            lower = Some(lower.map_or(result.lower, |old| old.min(result.lower)));
        }
        Ok(())
    })?;
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

fn process_path<const D: usize>(
    state: &State<D>,
    oracle: &Oracle<D>,
    duals: &[f64],
    arcs: &[usize],
    work: &mut usize,
    iterations: &mut usize,
    minimum: &mut f64,
    routes: &mut Vec<Route>,
    seen: &mut HashSet<String>,
) -> Result<()> {
    let result = path(
        oracle,
        state.shared_offset,
        duals,
        arcs,
        work,
        iterations,
        state.deadline,
    )
    .ok_or_else(|| incomplete("paired source-control pricing control coverage incomplete"))?;
    let Some(result) = result else {
        return Ok(());
    };
    *minimum = minimum.min(result.lower);
    let Some(ref route) = result.route else {
        return Ok(());
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
    let mut choices = vec![
        result.minimum.clone(),
        result.maximum.clone(),
        result.greedy.clone(),
    ];
    let combinations = if result.sign_ambiguous.is_empty() {
        0
    } else {
        let shift = u32::try_from(result.sign_ambiguous.len())
            .map_err(|_| incomplete("paired endpoint proposal dimension overflow"))?;
        let count = 1usize
            .checked_shl(shift)
            .ok_or_else(|| incomplete("paired endpoint proposal dimension overflow"))?;
        *work = work
            .checked_sub(count)
            .ok_or_else(|| incomplete("paired endpoint proposal work budget exhausted"))?;
        count
    };
    for quantity in quantities {
        if let Some(controls) = joint_controls(
            state,
            oracle,
            arcs,
            &result,
            &initial,
            quantity,
            (work, iterations),
        ) {
            choices.extend(controls);
        }
    }
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
            let target =
                value + (state.rhs[r] - coefficients.get(&r).copied().unwrap_or(0.)) / coefficient;
            if !target.is_finite() || target < result.domains[i].lo || target > result.domains[i].hi
            {
                continue;
            }
            if let Some(proposal) = target_control(
                oracle,
                arcs,
                resource,
                target,
                work,
                iterations,
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
    // Stream the exponential endpoint sweep: the work budget bounds its
    // count, while the deadline also bounds cloning and source replay.
    let endpoints = (0..combinations).map(|mask| {
        let mut controls = result.greedy.clone();
        for (bit, &index) in result.sign_ambiguous.iter().enumerate() {
            controls[index] = if mask & (1usize << bit) == 0 {
                result.minimum[index].clone()
            } else {
                result.maximum[index].clone()
            };
        }
        controls
    });
    for controls in choices.into_iter().chain(endpoints) {
        if Instant::now() >= state.deadline {
            return Err(incomplete("paired endpoint proposal deadline exhausted"));
        }
        let candidate = route
            .clone()
            .with_paired_controls(&oracle.network.graph, sorted(controls))?;
        if lifecycle::integer_route_feasible(oracle, &candidate)?
            && seen.insert(candidate.identity())
        {
            routes.push(candidate);
        }
    }
    Ok(())
}

struct PhysicalSuffixBounds {
    suffix: Vec<Vec<Option<f64>>>,
    arc_costs: Vec<f64>,
    route_constant: f64,
    other_coupled_lower: f64,
}

fn row_simple_supported<const D: usize>(oracle: &Oracle<D>) -> bool {
    let graph = &oracle.network.graph;
    let Some(kernel) = graph.coupled_kernel() else {
        return false;
    };
    oracle.activity_rows.iter().all(|entry| {
        activity_rows::paired(graph, entry.resource)
            || matches!(
                kernel.definitions().get(entry.resource),
                Some(
                    CoupledResource::CumulativeCost { .. } | CoupledResource::CustomResource { .. }
                )
            )
    }) && (0..oracle.solver.cuts.cuts.len()).all(|cut| {
        oracle
            .nonlinear_rows
            .iter()
            .any(|&(mapped, _)| mapped == cut)
    }) && (0..oracle.solver.cuts.knapsack_cuts.len()).all(|cut| {
        oracle
            .knapsack_rows
            .iter()
            .any(|&(mapped, _)| mapped == cut)
    }) && (0..oracle.solver.presence_rows().len()).all(|input| {
        oracle
            .presence_rows
            .iter()
            .any(|&(mapped, _)| mapped == input)
    })
}

fn coupled_lower_bounds<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<(f64, Vec<f64>)> {
    let graph = &oracle.network.graph;
    let kernel = graph.coupled_kernel()?;
    let mut lower = 0.;
    let mut paired_arc_lower = vec![0.; graph.arcs.len()];
    for (resource, definition) in kernel.definitions().iter().enumerate() {
        let accumulated = match definition {
            CoupledResource::CumulativeCost {
                initial,
                increments,
                weights,
                ..
            } => Some((*initial, increments.as_slice(), weights.as_slice())),
            CoupledResource::CustomResource {
                increments,
                weights,
                ..
            } => Some((0., increments.as_slice(), weights.as_slice())),
            _ => None,
        };
        if let Some((initial, increments, weights)) = accumulated {
            let coefficient = kernel.effective_objective_coefficient(resource)?;
            let has_activity = oracle
                .activity_rows
                .iter()
                .any(|entry| entry.resource == resource && entry.coefficient != 0.);
            if coefficient != 0. || has_activity {
                let amount = accumulated_cost_hull(
                    initial,
                    increments,
                    weights,
                    graph.original_max_arcs(),
                    work,
                    deadline,
                )?;
                if coefficient != 0. {
                    lower = crate::core::rounding::add_down(lower, amount.scale(coefficient)?.lo)?;
                }
                for entry in oracle
                    .activity_rows
                    .iter()
                    .filter(|entry| entry.resource == resource && entry.coefficient != 0.)
                {
                    let dual = *duals.get(offset.checked_add(entry.row)?)?;
                    lower = crate::core::rounding::add_down(
                        lower,
                        amount.scale(entry.coefficient)?.scale(-dual)?.lo,
                    )?;
                }
            }
            continue;
        }
        let charge_hull = match definition {
            CoupledResource::IntervalCost { functions, .. }
            | CoupledResource::PiecewiseLinearCost { functions, .. } => {
                piecewise_charge_hull(functions, work, deadline)
            }
            CoupledResource::EncodedPiecewiseLinearCost { functions, .. } => {
                encoded_charge_hull(functions, work, deadline)
            }
            CoupledResource::SlopeIntervalCost { functions, .. } => {
                slope_charge_hull(functions, work, deadline)
            }
            _ => None,
        };
        if let Some(charge_hull) = charge_hull {
            let coefficient = kernel.effective_objective_coefficient(resource)?;
            if coefficient != 0. {
                let amount =
                    repeated_charge_hull(charge_hull, graph.original_max_arcs(), work, deadline)?;
                lower = crate::core::rounding::add_down(lower, amount.scale(coefficient)?.lo)?;
            }
            continue;
        }
        let (normalization, cost_mode) = match definition {
            CoupledResource::PairedPiecewiseLinear {
                mode,
                normalization,
                arcs,
                ..
            } => {
                if *mode == crate::graph::coupled::PairedResourceMode::CostUnbounded
                    && arcs
                        .first()
                        .and_then(|arc| arc.upper.points.first())
                        .is_some_and(|point| point[1] >= 1e12)
                {
                    // This descriptor uses 1e12 as infinity; the state may
                    // grow beyond the stored upper function.
                    return None;
                }
                if *mode == crate::graph::coupled::PairedResourceMode::Feasibility
                    && !oracle
                        .activity_rows
                        .iter()
                        .any(|entry| entry.resource == resource)
                {
                    continue;
                }
                (
                    *normalization,
                    *mode != crate::graph::coupled::PairedResourceMode::Feasibility,
                )
            }
            CoupledResource::SlopePaired { .. } => {
                let CoupledResource::SlopePaired {
                    mode,
                    normalization,
                    arcs,
                    policy,
                    ..
                } = definition
                else {
                    unreachable!()
                };
                if *mode == crate::graph::coupled::PairedResourceMode::CostUnbounded
                    && *policy == crate::graph::coupled::FrontOperationPolicy::FrontAlgebra
                    && arcs
                        .first()
                        .and_then(|arc| arc.upper.pairs().first())
                        .is_some_and(|pair| pair.0 >= 1e12)
                {
                    // Match the raw-slope shift policy: this is an unbounded
                    // resource, so its declared sentinel is not a cap.
                    return None;
                }
                if *mode == crate::graph::coupled::PairedResourceMode::Feasibility
                    && !oracle
                        .activity_rows
                        .iter()
                        .any(|entry| entry.resource == resource)
                {
                    continue;
                }
                (
                    *normalization,
                    *mode != crate::graph::coupled::PairedResourceMode::Feasibility,
                )
            }
            _ => continue,
        };
        let objective_coefficient = if cost_mode {
            kernel.effective_objective_coefficient(resource)?
        } else {
            0.
        };
        let baseline = if cost_mode
            && normalization
                == crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum
        {
            Some(kernel.paired_initial_minimum(resource, work, Some(deadline))?)
        } else {
            None
        };
        for (arc_index, paired_arc_lower) in paired_arc_lower.iter_mut().enumerate() {
            let (lower_function, upper_function) = match definition {
                CoupledResource::PairedPiecewiseLinear { arcs, .. } => {
                    let arc = arcs.get(arc_index)?;
                    (&arc.lower, &arc.upper)
                }
                CoupledResource::SlopePaired { arcs, .. } => {
                    let arc = arcs.get(arc_index)?;
                    let low = slope_function_hull(&arc.lower, work, deadline)?;
                    let high = slope_function_hull(&arc.upper, work, deadline)?;
                    let values = Interval {
                        lo: low.lo,
                        hi: high.hi,
                    };
                    add_paired_arc_contribution(
                        oracle,
                        offset,
                        duals,
                        resource,
                        normalization,
                        objective_coefficient,
                        baseline,
                        values,
                        paired_arc_lower,
                    )?;
                    continue;
                }
                _ => continue,
            };
            let lower_hull = piecewise_value_hull(lower_function, work, deadline)?;
            let upper_hull = piecewise_value_hull(upper_function, work, deadline)?;
            let values = Interval {
                lo: lower_hull.lo,
                hi: upper_hull.hi,
            };
            if values.lo > values.hi {
                return None;
            }
            add_paired_arc_contribution(
                oracle,
                offset,
                duals,
                resource,
                normalization,
                objective_coefficient,
                baseline,
                values,
                paired_arc_lower,
            )?;
        }
    }
    let hops = graph.original_max_arcs();
    for &(cut, row) in &oracle.nonlinear_rows {
        let mut per_arc = 0u64;
        for arc in 0..graph.arcs.len() {
            *work = work.checked_sub(1)?;
            if Instant::now() >= deadline {
                return None;
            }
            per_arc = per_arc.max(super::nonlinear::arc_upper(oracle, cut, arc) as u64);
        }
        let maximum = u128::try_from(hops)
            .ok()?
            .checked_mul(u128::from(per_arc))?;
        if maximum > (1u128 << 53) {
            return None;
        }
        add_master_row_lower(
            offset,
            duals,
            row,
            Interval {
                lo: 0.,
                hi: maximum as f64,
            },
            &mut lower,
        )?;
    }
    for &(cut, row) in &oracle.knapsack_rows {
        let cut = &oracle.solver.cuts.knapsack_cuts.get(cut)?.cut;
        let mut low = 0i32;
        let mut high = 0i32;
        for &value in cut.coefficients.values() {
            *work = work.checked_sub(1)?;
            if Instant::now() >= deadline {
                return None;
            }
            low = low.min(value);
            high = high.max(value);
        }
        add_master_row_lower(
            offset,
            duals,
            row,
            Interval {
                lo: f64::from(low),
                hi: f64::from(high),
            },
            &mut lower,
        )?;
    }
    for &(input, row) in &oracle.presence_rows {
        let maximum = oracle
            .solver
            .presence_rows()
            .get(input)?
            .row
            .maximum_coefficient();
        if maximum as u128 > (1u128 << 53) {
            return None;
        }
        add_master_row_lower(
            offset,
            duals,
            row,
            Interval {
                lo: 0.,
                hi: maximum as f64,
            },
            &mut lower,
        )?;
    }
    for &(row, _) in &oracle.resource_groups {
        add_master_row_lower(offset, duals, row, Interval { lo: 0., hi: 1. }, &mut lower)?;
    }
    Some((lower, paired_arc_lower))
}

fn add_master_row_lower(
    offset: usize,
    duals: &[f64],
    row: usize,
    values: Interval,
    lower: &mut f64,
) -> Option<()> {
    let dual = *duals.get(offset.checked_add(row)?)?;
    *lower = crate::core::rounding::add_down(*lower, values.scale(-dual)?.lo)?;
    Some(())
}

fn add_paired_arc_contribution<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    resource: usize,
    normalization: crate::graph::coupled::PairedCostNormalization,
    objective_coefficient: f64,
    baseline: Option<[f64; 2]>,
    values: Interval,
    lower: &mut f64,
) -> Option<()> {
    if !values.lo.is_finite() || !values.hi.is_finite() || values.lo > values.hi {
        return None;
    }
    let cost = if objective_coefficient == 0. {
        0.
    } else {
        let amount = match normalization {
            crate::graph::coupled::PairedCostNormalization::AbsoluteFinalValue => values,
            crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum => {
                let baseline = baseline?;
                Interval {
                    lo: crate::core::rounding::add_down(values.lo, -baseline[1])?,
                    hi: crate::core::rounding::add_up(values.hi, -baseline[0])?,
                }
            }
        };
        amount.scale(objective_coefficient)?.lo
    };
    *lower = crate::core::rounding::add_down(*lower, cost)?;
    for entry in oracle
        .activity_rows
        .iter()
        .filter(|entry| entry.resource == resource)
    {
        let dual = *duals.get(offset.checked_add(entry.row)?)?;
        let row_value = values.scale(entry.coefficient)?;
        *lower = crate::core::rounding::add_down(*lower, row_value.scale(-dual)?.lo)?;
    }
    Some(())
}

/// Directed enclosure of every value on a prepared raw-slope function.
/// Each segment is affine, so its extrema occur at its endpoints; interval
/// arithmetic also encloses the function's sequential binary64 evaluation.
fn slope_function_hull(
    function: &crate::graph::coupled::PreparedSlopeFunction,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let pairs = function.pairs();
    let &(initial, _) = pairs.first()?;
    let mut value = Interval::point(initial);
    let mut lower = initial;
    let mut upper = initial;
    let mut previous = pairs.first()?.1;
    for &(slope, end) in pairs.iter().skip(1) {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let span = end - previous;
        if !span.is_finite() || span <= 0. {
            return None;
        }
        let delta = Interval::point(slope).multiply(Interval::point(span))?;
        value = value.add(delta)?;
        lower = lower.min(value.lo);
        upper = upper.max(value.hi);
        previous = end;
    }
    Some(Interval {
        lo: lower,
        hi: upper,
    })
}

fn slope_charge_hull(
    functions: &[crate::graph::coupled::PreparedSlopeFunction],
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for function in functions {
        let range = slope_function_hull(function, work, deadline)?;
        hull.lo = hull.lo.min(range.lo);
        hull.hi = hull.hi.max(range.hi);
    }
    if functions.is_empty() {
        Some(Interval::point(0.))
    } else {
        Some(hull)
    }
}

/// Directed hull for encoded state costs. Knot functions clamp interpolation
/// to their segment endpoints; slope functions retain their prepared
/// sequential arithmetic, which `slope_function_hull` encloses outward.
fn encoded_charge_hull(
    functions: &[crate::graph::coupled::EncodedFunction],
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for function in functions {
        let range = match function {
            crate::graph::coupled::EncodedFunction::Knots { function } => {
                piecewise_charge_hull(std::slice::from_ref(function), work, deadline)?
            }
            crate::graph::coupled::EncodedFunction::SlopeSegments { function, .. } => {
                slope_function_hull(function, work, deadline)?
            }
        };
        hull.lo = hull.lo.min(range.lo);
        hull.hi = hull.hi.max(range.hi);
    }
    if functions.is_empty() {
        Some(Interval::point(0.))
    } else {
        Some(hull)
    }
}

/// Piecewise-linear source evaluation clamps interpolation to its segment's
/// endpoint values, so the global knot hull encloses every rounded charge.
fn piecewise_charge_hull(
    functions: &[crate::graph::coupled::PiecewiseLinearFunction],
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for function in functions {
        let values = piecewise_value_hull(function, work, deadline)?;
        hull.lo = hull.lo.min(values.lo);
        hull.hi = hull.hi.max(values.hi);
    }
    if functions.is_empty() {
        Some(Interval::point(0.))
    } else {
        Some(hull)
    }
}

fn piecewise_value_hull(
    function: &crate::graph::coupled::PiecewiseLinearFunction,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for point in &function.points {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline || !point[1].is_finite() {
            return None;
        }
        hull.lo = hull.lo.min(point[1]);
        hull.hi = hull.hi.max(point[1]);
    }
    (hull.lo.is_finite() && hull.hi.is_finite()).then_some(hull)
}

/// Sum a global per-arc charge range over all path lengths up to `hops`.
/// Zero is included at each position to relax routes that end earlier.
fn repeated_charge_hull(
    per_arc: Interval,
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let term = Interval {
        lo: per_arc.lo.min(0.),
        hi: per_arc.hi.max(0.),
    };
    let mut total = Interval::point(0.);
    for _ in 0..hops {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        total = total.add(term)?;
    }
    Some(total)
}

/// Outward lower enclosure of a cumulative/custom cost over every walk of at
/// most `hops` arcs. Increment and weight choices are relaxed independently,
/// while interval addition encloses the source program's per-arc rounding.
fn accumulated_cost_hull(
    initial: f64,
    increments: &[f64],
    weights: &[f64],
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    if increments.len() != weights.len() {
        return None;
    }
    if hops == 0 {
        return Some(Interval::point(0.));
    }
    let mut increment_hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    let mut weight_hull = increment_hull;
    for (&increment, &weight) in increments.iter().zip(weights) {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        if !increment.is_finite() || !weight.is_finite() {
            return None;
        }
        increment_hull.lo = increment_hull.lo.min(increment);
        increment_hull.hi = increment_hull.hi.max(increment);
        weight_hull.lo = weight_hull.lo.min(weight);
        weight_hull.hi = weight_hull.hi.max(weight);
    }
    if increments.is_empty() {
        return Some(Interval::point(0.));
    }
    let mut state = Interval::point(initial);
    let mut cost = Interval::point(0.);
    for _ in 0..hops {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        state = state.add(increment_hull)?;
        let term = state.multiply(weight_hull)?;
        let term = Interval {
            lo: term.lo.min(0.),
            hi: term.hi.max(0.),
        };
        cost = cost.add(term)?;
    }
    Some(cost)
}

fn suffix_table<const D: usize>(
    graph: &Graph<D>,
    arc_costs: &[f64],
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<Vec<Vec<Option<f64>>>> {
    let mut suffix = vec![vec![None; graph.vertices.len()]; hops.checked_add(1)?];
    *suffix[0].get_mut(graph.sink)? = Some(0.);
    for remaining in 1..=hops {
        if Instant::now() >= deadline {
            return None;
        }
        let (before, current) = suffix.split_at_mut(remaining);
        current[0].clone_from(&before[remaining - 1]);
        let previous = &before[remaining - 1];
        let current = &mut current[0];
        for (index, arc) in graph.arcs.iter().enumerate() {
            *work = work.checked_sub(1)?;
            let Some(tail) = previous.get(arc.head).copied().flatten() else {
                continue;
            };
            let candidate = crate::core::rounding::add_down(*arc_costs.get(index)?, tail)?;
            let value = current.get_mut(arc.tail)?;
            *value = Some(value.map_or(candidate, |old: f64| old.min(candidate)));
        }
    }
    Some(suffix)
}

fn physical_suffix_bounds<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    original_costs: Option<&OriginalVariableCosts>,
    work: &mut usize,
    deadline: Instant,
) -> Option<PhysicalSuffixBounds> {
    if !row_simple_supported(oracle) {
        return None;
    }
    let graph = &oracle.network.graph;
    if graph
        .coupled_kernel()?
        .definitions()
        .iter()
        .any(|definition| {
            !matches!(
                definition,
                CoupledResource::PairedPiecewiseLinear { .. } | CoupledResource::SlopePaired { .. }
            ) && matches!(definition, CoupledResource::Plugin(_))
        })
    {
        // Plugin costs have no directed lower bound. Do not omit them from a
        // certificate.
        return None;
    }
    if oracle.network.arc_entries.len() != graph.arcs.len() {
        return None;
    }
    *work = work.checked_sub(
        graph
            .vertices
            .len()
            .checked_add(graph.arcs.len())?
            .checked_add(oracle.route_entries.len())?
            .checked_add(oracle.network.route_entries.len())?,
    )?;
    let dual = |row: usize| -> Option<f64> {
        let value = *duals.get(row)?;
        value.is_finite().then_some(value)
    };
    let mut route_constant = 0.;
    for &(row, coefficient) in &oracle.route_entries {
        route_constant = crate::core::rounding::add_down(
            route_constant,
            crate::core::rounding::multiply_down(-dual(row)?, coefficient)?,
        )?;
    }
    for &(row, coefficient) in oracle.network.route_entries.iter() {
        route_constant = crate::core::rounding::add_down(
            route_constant,
            crate::core::rounding::multiply_down(-dual(offset.checked_add(row)?)?, coefficient)?,
        )?;
    }
    let mut arc_costs = if let Some(original) = original_costs {
        original_arc_cost_lower_bounds(oracle, original, offset, duals, work, deadline)?
    } else {
        let mut costs = Vec::with_capacity(graph.arcs.len());
        for (index, arc) in graph.arcs.iter().enumerate() {
            let mut cost = arc.cost;
            for &(row, coefficient) in oracle.network.arc_entries.get(index)? {
                *work = work.checked_sub(1)?;
                cost = crate::core::rounding::add_down(
                    cost,
                    crate::core::rounding::multiply_down(
                        -dual(offset.checked_add(row)?)?,
                        coefficient,
                    )?,
                )?;
            }
            costs.push(cost);
        }
        costs
    };
    let (other_coupled_lower, paired_arc_lower) =
        coupled_lower_bounds(oracle, offset, duals, work, deadline)?;
    let hops = graph.original_max_arcs();
    // A paired objective or activity is charged once at the terminal state.
    // Its directed per-arc envelope can therefore be attached to the final
    // physical arc entering the sink. The ordinary suffix DP then keeps the
    // terminal arc and physical path costs correlated without storing a
    // separate resource-state table.
    for (index, arc) in graph.arcs.iter().enumerate() {
        if arc.head == graph.sink {
            *work = work.checked_sub(1)?;
            if Instant::now() >= deadline {
                return None;
            }
            arc_costs[index] =
                crate::core::rounding::add_down(arc_costs[index], paired_arc_lower[index])?;
        }
    }
    let suffix = suffix_table(graph, &arc_costs, hops, work, deadline)?;
    Some(PhysicalSuffixBounds {
        suffix,
        arc_costs,
        route_constant,
        other_coupled_lower,
    })
}

fn original_arc_cost_lower_bounds<const D: usize>(
    oracle: &Oracle<D>,
    original: &OriginalVariableCosts,
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Vec<f64>> {
    let graph = &oracle.network.graph;
    let index: HashMap<_, _> = graph
        .arcs
        .iter()
        .enumerate()
        .map(|(position, arc)| (arc.id, position))
        .collect();
    let row_cost = |mut value: Interval, entries: &[(usize, f64)], work: &mut usize| {
        for &(row, coefficient) in entries {
            *work = work.checked_sub(1)?;
            if Instant::now() >= deadline || !coefficient.is_finite() {
                return None;
            }
            let dual = *duals.get(offset.checked_add(row)?)?;
            value = value.add(Interval::point(-dual).multiply(Interval::point(coefficient))?)?;
        }
        Some(value)
    };
    let mut costs = Vec::with_capacity(graph.arcs.len());
    for (position, arc) in graph.arcs.iter().enumerate() {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let saved = original.arcs.get(&arc.id)?;
        let entries = oracle.network.arc_entries.get(position)?;
        if !entries.starts_with(&saved.compiled_entries) {
            return None;
        }
        let compiled_delta = Interval::point(arc.cost).add(Interval::point(-saved.compiled))?;
        let base = Interval::point(saved.base).add(compiled_delta)?;
        let base = row_cost(base, &saved.entries, work)?;
        let additional = entries.get(saved.compiled_entries.len()..)?;
        costs.push(row_cost(base, additional, work)?);
    }
    for variable in &original.variables {
        if variable.accumulated_resource.is_some() {
            continue;
        }
        let reduced = row_cost(Interval::point(variable.cost), &variable.entries, work)?;
        let scaled = reduced.scale(1e8)?.add(Interval::point(0.5))?;
        if !scaled.lo.is_finite()
            || !scaled.hi.is_finite()
            || scaled.lo.abs() > 9_007_199_254_740_992.
            || scaled.hi.abs() > 9_007_199_254_740_992.
        {
            return None;
        }
        let rounded = Interval {
            lo: scaled.lo.floor(),
            hi: scaled.hi.floor(),
        }
        .scale(1e-8)?;
        for &(arc, association) in &variable.arcs {
            *work = work.checked_sub(1)?;
            if Instant::now() >= deadline || !association.is_finite() {
                return None;
            }
            let position = *index.get(&arc)?;
            let cost = costs.get_mut(position)?;
            *cost = cost.add(rounded.scale(association)?)?;
        }
    }
    Some(costs.into_iter().map(|cost| cost.lo).collect())
}

fn signed_scc_seed_paths<const D: usize>(
    state: &State<D>,
    oracle: &Oracle<D>,
    duals: &[f64],
    original_costs: Option<&OriginalVariableCosts>,
    work: &mut usize,
    deadline: Instant,
) -> Option<Vec<Vec<usize>>> {
    let mut graph = oracle.network.graph.as_ref().clone();
    if let Some(original) = original_costs {
        let reduced = original_arc_cost_lower_bounds(
            oracle,
            original,
            state.shared_offset,
            duals,
            work,
            deadline,
        )?;
        for (arc, cost) in graph.arcs.iter_mut().zip(reduced) {
            arc.reduced_cost = cost;
        }
    } else {
        for (index, arc) in graph.arcs.iter_mut().enumerate() {
            let mut reduced = arc.cost;
            for &(row, coefficient) in oracle.network.arc_entries.get(index)? {
                reduced = crate::core::rounding::add_down(
                    reduced,
                    crate::core::rounding::multiply_down(
                        -*duals.get(state.shared_offset.checked_add(row)?)?,
                        coefficient,
                    )?,
                )?;
            }
            arc.reduced_cost = reduced;
        }
    }
    let kernel = graph.coupled_kernel()?;
    let mut adjustments = BTreeMap::<usize, f64>::new();
    let mut paired_adjustments = BTreeMap::<usize, f64>::new();
    for entry in &oracle.activity_rows {
        if entry.coefficient == 0. {
            continue;
        }
        let definition = kernel.definitions().get(entry.resource)?;
        let dual = *duals.get(state.shared_offset.checked_add(entry.row)?)?;
        let adjustment = crate::core::rounding::multiply_down(-dual, entry.coefficient)?;
        let target = match definition {
            CoupledResource::PairedPiecewiseLinear { .. } | CoupledResource::SlopePaired { .. } => {
                &mut paired_adjustments
            }
            CoupledResource::CumulativeCost { .. } | CoupledResource::CustomResource { .. } => {
                &mut adjustments
            }
            _ => continue,
        };
        let value = target.entry(entry.resource).or_default();
        *value = crate::core::rounding::add_down(*value, adjustment)?;
    }
    if !paired_adjustments.is_empty() {
        let objective_scale = kernel.objective_scale();
        if !objective_scale.is_finite() || objective_scale == 0. {
            return None;
        }
        let mut coefficients = BTreeMap::new();
        for (resource, definition) in kernel.definitions().iter().enumerate() {
            let supported = match definition {
                CoupledResource::CustomResource { .. } | CoupledResource::CumulativeCost { .. } => {
                    true
                }
                CoupledResource::PairedPiecewiseLinear { mode, .. }
                | CoupledResource::SlopePaired { mode, .. } => {
                    *mode != crate::graph::coupled::PairedResourceMode::Feasibility
                }
                _ => false,
            };
            if supported {
                coefficients.insert(resource, kernel.objective_coefficient(resource)?);
            }
        }
        for (resource, adjustment) in paired_adjustments {
            let coefficient = coefficients.get_mut(&resource)?;
            let adjustment = adjustment / objective_scale;
            if !adjustment.is_finite() {
                return None;
            }
            *coefficient = crate::core::rounding::add_down(*coefficient, adjustment)?;
        }
        let coefficients = coefficients.into_iter().collect::<Vec<_>>();
        graph
            .set_coupled_objective_coefficients(&coefficients)
            .ok()?;
    }
    if !adjustments.is_empty() {
        graph
            .set_coupled_pricing_adjustments(
                &adjustments
                    .into_iter()
                    .filter(|(_, coefficient)| *coefficient != 0.)
                    .collect::<Vec<_>>(),
            )
            .ok()?;
    }
    let candidates = oracle
        .solver
        .signed_candidate_routes(graph, 32, deadline)
        .ok()?;
    let positions: HashMap<_, _> = oracle
        .network
        .graph
        .arcs
        .iter()
        .enumerate()
        .map(|(position, arc)| (arc.id, position))
        .collect();
    candidates
        .columns
        .into_iter()
        .map(|route| {
            route
                .arcs
                .into_iter()
                .map(|arc| positions.get(&arc).copied())
                .collect()
        })
        .collect()
}

fn branch_and_bound_price<const D: usize>(
    state: &State<D>,
    index: usize,
    duals: &[f64],
    work: &mut usize,
) -> Option<Result<(Vec<Route>, f64)>> {
    let oracle = state.oracles.get(index)?;
    let original_costs = state
        .original_variable_costs
        .iter()
        .find(|model| model.network == oracle.network.id);
    let physical = physical_suffix_bounds(
        oracle,
        state.shared_offset,
        duals,
        original_costs,
        work,
        state.deadline,
    )?;
    let graph = &oracle.network.graph;
    let hops = physical.suffix.len().checked_sub(1)?;
    // The scalar suffix remains a walk relaxation, but this proof traversal
    // only needs route-domain paths. Avoid revisiting vertices when physical
    // simplicity is declared or implied by hard groups; bounded walks that
    // permit repeats continue to explore them.
    let vertex_simple_domain = graph.route_domain() == crate::graph::RouteDomain::PhysicalSimple
        || graph.bounded_walks_are_vertex_simple();
    let prune_repeated_vertices = vertex_simple_domain && !graph.is_acyclic();
    let Some(global_tail) = physical
        .suffix
        .get(hops)?
        .get(graph.source)
        .copied()
        .flatten()
    else {
        return Some(Ok((Vec::new(), 0.)));
    };
    let Some(global_constant) =
        crate::core::rounding::add_down(physical.route_constant, physical.other_coupled_lower)
    else {
        return None;
    };
    // This suffix relaxation bounds every admissible walk. If branch
    // refinement later hits its budget, retained pricing can still use this
    // certified minimum bound alongside the source-validated columns found.
    let global_lower_bound = crate::core::rounding::add_down(global_tail, global_constant)?;
    // The universal bound closes pricing immediately when it is nonnegative;
    // candidate generation cannot produce a negative column in that case.
    if global_lower_bound >= 0. {
        return Some(Ok((Vec::new(), global_lower_bound)));
    }
    let mut outgoing = vec![Vec::new(); graph.vertices.len()];
    for (arc, data) in graph.arcs.iter().enumerate() {
        let Some(left) = work.checked_sub(1) else {
            return Some(Ok((Vec::new(), global_lower_bound)));
        };
        *work = left;
        outgoing.get_mut(data.tail)?.push(arc);
    }
    struct Frame {
        vertex: usize,
        remaining: usize,
        prefix: f64,
        path_len: usize,
        next_arc: usize,
        entered: bool,
    }
    let mut stack = vec![Frame {
        vertex: graph.source,
        remaining: hops,
        prefix: 0.,
        path_len: 0,
        next_arc: 0,
        entered: false,
    }];
    let mut path = Vec::new();
    let mut iterations = 100_000;
    let mut minimum = f64::INFINITY;
    let mut routes = Vec::new();
    let mut seen = HashSet::new();
    let mut priced_paths = HashSet::<Vec<usize>>::new();
    // Native SCC label sweeps are useful candidate generators, but their
    // minima and dominance trajectory are not proof. Re-price every seed
    // through the source-control model before the bounded-walk proof pass.
    let seed_started = Instant::now();
    let seed_deadline = seed_started
        .checked_add(state.deadline.saturating_duration_since(seed_started) / 4)
        .unwrap_or(state.deadline)
        .min(state.deadline);
    let seed_budget = *work / 4;
    let mut seed_work = seed_budget;
    let seed_iterations = iterations / 4;
    let mut remaining_seed_iterations = seed_iterations;
    if let Some(candidates) = signed_scc_seed_paths(
        state,
        oracle,
        duals,
        original_costs,
        &mut seed_work,
        seed_deadline,
    ) {
        for candidate in candidates {
            if Instant::now() >= seed_deadline {
                break;
            }
            if candidate.iter().any(|&arc| arc >= graph.arcs.len()) {
                continue;
            }
            if priced_paths.contains(&candidate) {
                continue;
            }
            if process_path(
                state,
                oracle,
                duals,
                &candidate,
                &mut seed_work,
                &mut remaining_seed_iterations,
                &mut minimum,
                &mut routes,
                &mut seen,
            )
            .is_err()
            {
                break;
            }
            priced_paths.insert(candidate);
        }
    }
    *work = work.checked_sub(seed_budget - seed_work)?;
    iterations = iterations.checked_sub(seed_iterations - remaining_seed_iterations)?;
    while !stack.is_empty() {
        if Instant::now() >= state.deadline {
            return Some(Ok((routes, global_lower_bound)));
        }
        let top = stack.len() - 1;
        if !stack[top].entered {
            let Some(left) = work.checked_sub(1) else {
                return Some(Ok((routes, global_lower_bound)));
            };
            *work = left;
            let frame = &stack[top];
            let Some(tail) = physical
                .suffix
                .get(frame.remaining)
                .and_then(|row| row.get(frame.vertex))
                .copied()
                .flatten()
            else {
                stack.pop();
                if let Some(parent) = stack.last() {
                    path.truncate(parent.path_len);
                }
                continue;
            };
            let Some(prefix_tail) = crate::core::rounding::add_down(frame.prefix, tail) else {
                return Some(Ok((routes, global_lower_bound)));
            };
            let Some(constant) = crate::core::rounding::add_down(
                physical.route_constant,
                physical.other_coupled_lower,
            ) else {
                return Some(Ok((routes, global_lower_bound)));
            };
            let Some(bound) = crate::core::rounding::add_down(prefix_tail, constant) else {
                return Some(Ok((routes, global_lower_bound)));
            };
            if bound >= 0. {
                minimum = minimum.min(bound);
                stack.pop();
                if let Some(parent) = stack.last() {
                    path.truncate(parent.path_len);
                }
                continue;
            }
            if frame.vertex == graph.sink {
                if !priced_paths.contains(&path) {
                    if process_path(
                        state,
                        oracle,
                        duals,
                        &path,
                        work,
                        &mut iterations,
                        &mut minimum,
                        &mut routes,
                        &mut seen,
                    )
                    .is_err()
                    {
                        return Some(Ok((routes, global_lower_bound)));
                    }
                    priced_paths.insert(path.clone());
                }
                stack.pop();
                if let Some(parent) = stack.last() {
                    path.truncate(parent.path_len);
                }
                continue;
            }
            if frame.remaining == 0 {
                stack.pop();
                if let Some(parent) = stack.last() {
                    path.truncate(parent.path_len);
                }
                continue;
            }
            stack[top].entered = true;
        }
        let (arc, prefix, path_len, remaining) = {
            let frame = &mut stack[top];
            let Some(&arc) = outgoing
                .get(frame.vertex)
                .and_then(|arcs| arcs.get(frame.next_arc))
            else {
                stack.pop();
                if let Some(parent) = stack.last() {
                    path.truncate(parent.path_len);
                }
                continue;
            };
            frame.next_arc += 1;
            (arc, frame.prefix, frame.path_len, frame.remaining)
        };
        let Some(left) = work.checked_sub(1) else {
            return Some(Ok((routes, global_lower_bound)));
        };
        *work = left;
        if prune_repeated_vertices {
            let Some(left) = work.checked_sub(path.len()) else {
                return Some(Ok((routes, global_lower_bound)));
            };
            *work = left;
            let head = graph.arcs[arc].head;
            if head == graph.source
                || path
                    .iter()
                    .any(|&previous| graph.arcs[previous].head == head)
            {
                continue;
            }
        }
        let Some(prefix) = crate::core::rounding::add_down(prefix, physical.arc_costs[arc]) else {
            return Some(Ok((routes, global_lower_bound)));
        };
        path.truncate(path_len);
        path.push(arc);
        stack.push(Frame {
            vertex: graph.arcs[arc].head,
            remaining: remaining - 1,
            prefix,
            path_len: path.len(),
            next_arc: 0,
            entered: false,
        });
    }
    Some(Ok((routes, if minimum.is_finite() { minimum } else { 0. })))
}

pub(super) fn price<const D: usize>(
    state: &State<D>,
    index: usize,
    duals: &[f64],
) -> Result<(Vec<Route>, f64)> {
    let oracle = &state.oracles[index];
    let mut work = 4_000_000;
    let mut iterations = 100_000;
    if let Some(result) = branch_and_bound_price(state, index, duals, &mut work) {
        return result;
    }
    let mut minimum = f64::INFINITY;
    let mut routes = vec![];
    let mut seen = HashSet::new();
    let mut visitor_error = None;
    let visited =
        bounds::columns::visit_physical_paths(oracle, &mut work, state.deadline, |arcs, work| {
            let result = process_path(
                state,
                oracle,
                duals,
                &arcs,
                work,
                &mut iterations,
                &mut minimum,
                &mut routes,
                &mut seen,
            );
            match result {
                Ok(()) => Ok(()),
                Err(error) => {
                    visitor_error = Some(error);
                    Err(crate::enumeration::PathDomainFailure::Arithmetic)
                }
            }
        });
    if let Some(error) = visitor_error {
        return Err(error);
    }
    visited.ok_or_else(|| {
        incomplete("paired source-control pricing path/control coverage incomplete")
    })?;
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
