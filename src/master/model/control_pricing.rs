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
mod additive;

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
    deadline: Instant,
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
    if n == 0 || *iterations == 0 || Instant::now() >= deadline {
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
            deadline,
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
                deadline,
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
        deadline,
    );
    let mut vertex_realization_work = vertex_realization_reserve;
    for vertex in vertices {
        if vertex_realization_work == 0 || Instant::now() >= deadline {
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
                deadline,
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
// Terminal-box LP proposals shared by path pricing and witness recovery.
mod terminal_lp;
use terminal_lp::*;
fn sorted(mut controls: Vec<PairedControl>) -> Vec<PairedControl> {
    controls.sort_by_key(|c| c.resource);
    controls
}

// Cross-path witness recovery and realization of source controls.
mod witness;
pub(super) use witness::repair_witness;
#[cfg(test)]
use witness::*;
fn hull(a: Interval, b: f64) -> Option<Interval> {
    b.is_finite().then_some(Interval {
        lo: a.lo.min(b),
        hi: a.hi.max(b),
    })
}

// Outer None is Unknown; inner None is independently proved infeasibility.
// Route-local pricing, physical minima, and column bounds.
mod path;
use path::*;
pub(super) use path::{column_bound, column_bound_using, minimum};
// Conservative suffix bounds and terminal objective envelopes.
mod suffix_bounds;
use suffix_bounds::*;
// Bounded-walk resource extension and paired prefix domains.
mod prefix_domains;
use prefix_domains::*;
// Branch-and-bound pricing over physical paths and source controls.
mod search;
pub(super) use search::price;
#[cfg(test)]
#[path = "../../../tests/unit/master/model/joint_control_tests.rs"]
mod tests;
