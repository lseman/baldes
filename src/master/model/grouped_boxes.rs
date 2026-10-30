//! Verified envelopes for original variables sharing physical arcs.
use super::*;
use crate::{core::rounding::add_down, master::preprocessing::Bounds};
mod admission;
pub(super) use admission::coupled as coupled_admission;

/// A bound on one registered original activity, not a new graph restriction.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalVariableBound {
    pub network: i32,
    pub id: i32,
    #[serde(default)]
    pub lower: f64,
    pub upper: Option<f64>,
}

/// Expanded original-variable box, also available to low-level Rust callers.
#[derive(Clone, Debug)]
pub struct GroupedVariableBound {
    pub variable: OriginalPricingVariable,
    pub lower: f64,
    pub upper: Option<f64>,
}

#[derive(Clone)]
pub(super) struct PreparedVariable {
    pub arcs: Vec<(usize, f64)>,
    pub accumulated_resource: Option<usize>,
    pub cost: f64,
    pub entries: Vec<(usize, f64)>,
    pub bounds: Bounds,
}
#[derive(Clone)]
pub(super) struct PreparedBox {
    pub variables: Vec<PreparedVariable>,
    pub caps: Vec<f64>,
}

fn add_up(a: f64, b: f64) -> Option<f64> {
    Some(-add_down(-a, -b)?)
}

fn walk_range<const D: usize>(
    graph: &Graph<D>,
    forbidden: &[bool],
    weights: &[f64],
) -> Option<Option<Bounds>> {
    let n = graph.vertices.len();
    let mut outgoing = vec![vec![]; n];
    let mut degree = vec![0usize; n];
    for (p, a) in graph.arcs.iter().enumerate() {
        if forbidden.get(p) == Some(&true) {
            continue;
        }
        outgoing[a.tail].push(p);
        degree[a.head] += 1;
    }
    let mut queue: std::collections::VecDeque<_> = (0..n).filter(|&v| degree[v] == 0).collect();
    let mut order = vec![];
    while let Some(v) = queue.pop_front() {
        order.push(v);
        for &p in &outgoing[v] {
            let head = graph.arcs[p].head;
            degree[head] -= 1;
            if degree[head] == 0 {
                queue.push_back(head);
            }
        }
    }
    let mut range = vec![None::<Bounds>; n];
    range[graph.source] = Some(Bounds {
        lower: 0.,
        upper: 0.,
    });
    let extend = |range: &mut Vec<Option<Bounds>>, p: usize, tail: Bounds| -> Option<()> {
        let a = &graph.arcs[p];
        let next = Bounds {
            lower: add_down(tail.lower, weights[p])?,
            upper: add_up(tail.upper, weights[p])?,
        };
        match &mut range[a.head] {
            Some(old) => {
                old.lower = old.lower.min(next.lower);
                old.upper = old.upper.max(next.upper);
            }
            slot @ None => *slot = Some(next),
        }
        Some(())
    };
    if order.len() == n {
        // Ignoring a tighter walk cap remains a valid relaxation of the DAG.
        for v in order {
            if let Some(tail) = range[v] {
                for &p in &outgoing[v] {
                    extend(&mut range, p, tail)?;
                }
            }
        }
    } else {
        let cap = graph.max_route_arcs().or_else(|| {
            (!graph.has_coupled_resources())
                .then(|| super::bounds::resource_hop_limit(graph, 1_000_000))
                .flatten()
        })?;
        if cap.checked_mul(graph.arcs.len().checked_add(n)?)? > 1_000_000 {
            return None;
        }
        for _ in 0..cap {
            let previous = range.clone();
            for (p, a) in graph.arcs.iter().enumerate() {
                if forbidden.get(p) == Some(&true) {
                    continue;
                }
                if let Some(tail) = previous[a.tail] {
                    extend(&mut range, p, tail)?;
                }
            }
        }
    }
    Some(range[graph.sink])
}

/// A complete scalar relaxation contains every source-domain path. Coupled
/// infeasibility can discard a path only with an independent certificate;
/// ambiguous feasibility or activity arithmetic must decline the envelope.
fn activity_range<const D: usize>(graph: &Graph<D>, resource: usize) -> Option<Option<Bounds>> {
    if activity_rows::paired(graph, resource) {
        return None;
    }
    activity_proof(graph, resource, false).or_else(|| {
        graph.validate().ok()?;
        integer_program_range(graph, resource).map(Some)
    })
}

pub(super) fn integral_activity<const D: usize>(
    graph: &Graph<D>,
    resource: usize,
) -> Option<Bounds> {
    graph.validate().ok()?;
    activity_rows::validate(
        graph,
        &AccumulatedActivityCoefficients {
            network: 0,
            resource,
            coefficient: 1.,
        },
    )
    .ok()?;
    if activity_rows::paired(graph, resource) {
        return None;
    }
    if let Some(range) = integer_program_range(graph, resource) {
        return Some(range);
    }
    activity_proof(graph, resource, true)?.or(Some(Bounds {
        lower: 0.,
        upper: 0.,
    }))
}

/// Integer data plus a walk cap prove exact integer arithmetic without visiting
/// paths. Bound every intermediate level, product and accumulated charge by
/// 2^53; otherwise retain the complete-path proof rather than trusting rounded
/// endpoint integrality. Cyclic graphs need an explicit walk cap here.
fn integer_program_range<const D: usize>(graph: &Graph<D>, resource: usize) -> Option<Bounds> {
    use crate::graph::coupled::CoupledResource;
    const EXACT: u128 = 9_007_199_254_740_992;
    let hops = graph.max_route_arcs().or_else(|| {
        graph
            .is_acyclic()
            .then(|| graph.vertices.len().saturating_sub(1))
    })? as u128;
    let (initial, increments, weights) =
        match graph.coupled_kernel()?.definitions().get(resource)? {
            CoupledResource::CustomResource {
                increments,
                weights,
                ..
            } => (0., increments, weights),
            CoupledResource::CumulativeCost {
                initial,
                increments,
                weights,
                ..
            } => (*initial, increments, weights),
            _ => return None,
        };
    let integer = |value: f64| {
        (value.is_finite() && value.fract() == 0. && value.abs() <= EXACT as f64)
            .then_some(value.abs() as u128)
    };
    let start = integer(initial)?;
    let increment = increments
        .iter()
        .try_fold(0u128, |old, &v| Some(old.max(integer(v)?)))?;
    let weight = weights
        .iter()
        .try_fold(0u128, |old, &v| Some(old.max(integer(v)?)))?;
    let level = start.checked_add(hops.checked_mul(increment)?)?;
    let product = level.checked_mul(weight)?;
    let total = hops.checked_mul(product)?;
    if level > EXACT || product > EXACT || total > EXACT {
        return None;
    }
    let positive_level = initial >= 0. && increments.iter().all(|&v| v >= 0.);
    let negative_level = initial <= 0. && increments.iter().all(|&v| v <= 0.);
    let positive_weight = weights.iter().all(|&v| v >= 0.);
    let negative_weight = weights.iter().all(|&v| v <= 0.);
    Some(Bounds {
        lower: if (positive_level && positive_weight) || (negative_level && negative_weight) {
            0.
        } else {
            -(total as f64)
        },
        upper: if (positive_level && negative_weight) || (negative_level && positive_weight) {
            0.
        } else {
            total as f64
        },
    })
}

fn visit_relaxed_activity_paths<const D: usize>(
    graph: &Graph<D>,
    work: &mut usize,
    deadline: Instant,
    mut visitor: impl FnMut(
        &[usize],
        &mut usize,
    ) -> std::result::Result<(), crate::enumeration::PathDomainFailure>,
) -> Option<()> {
    use crate::enumeration::{CompleteResourceRelaxation, PathDomainLimits};
    let mut relaxed = graph.clone();
    relaxed
        .resources
        .fill(crate::graph::ResourceKind::Disposable);
    relaxed.set_coupled_resources(&[]).ok()?;
    let allowance = *work / 2;
    let mut proof_work = work.checked_sub(allowance)?;
    let stats = CompleteResourceRelaxation::visit(
        &relaxed,
        PathDomainLimits {
            max_paths: allowance,
            max_work: allowance,
            max_stored_arcs: allowance,
            deadline: Some(deadline),
        },
        |path| visitor(path, &mut proof_work),
    )
    .ok();
    let Some(stats) = stats else {
        // A failed visitor does not expose partial traversal work. Charge the
        // reserved path budget and keep every proof failure conservative.
        *work = proof_work;
        return None;
    };
    *work = proof_work.checked_add(allowance.checked_sub(stats.work_used)?)?;
    (Instant::now() < deadline).then_some(())
}

fn activity_proof<const D: usize>(
    graph: &Graph<D>,
    resource: usize,
    integer: bool,
) -> Option<Option<Bounds>> {
    let kernel = graph.coupled_kernel()?;
    let deadline = Instant::now().checked_add(std::time::Duration::from_secs(1))?;
    // This internal proof has no caller work budget; rely on its shared
    // deadline instead of imposing another fixed work ceiling.
    let mut work = usize::MAX;
    let mut range: Option<Bounds> = None;
    visit_relaxed_activity_paths(graph, &mut work, deadline, |path, proof_work| {
        let (feasible, [lower, upper]) = kernel
            .certify_accumulated_activity(resource, path, proof_work, Some(deadline))
            .ok_or(crate::enumeration::PathDomainFailure::WorkLimit)?;
        if !feasible {
            return Ok(());
        }
        if integer
            && (lower != upper
                || lower.fract() != 0.
                || lower.abs() > 9_007_199_254_740_992.
                || kernel.accumulated_activity(resource, path).ok() != Some(lower))
        {
            return Err(crate::enumeration::PathDomainFailure::Arithmetic);
        }
        match &mut range {
            Some(old) => {
                old.lower = old.lower.min(lower);
                old.upper = old.upper.max(upper);
            }
            None => range = Some(Bounds { lower, upper }),
        }
        Ok(())
    })?;
    Some(range)
}

/// Preserve original coordinates even when the caller supplies no finite box.
/// The reference's default orthant is usable only after proving that the graph
/// activity is nonnegative; signed mappings must not silently acquire bounds.
pub(super) fn prepare_registered<'a, const D: usize>(
    definitions: &[GroupedVariableBound],
    originals: &[OriginalVariableCosts],
    rows: usize,
    graphs: impl Iterator<Item = (i32, &'a Graph<D>, &'a [bool])>,
) -> Result<HashMap<i32, PreparedBox>> {
    let graphs: Vec<_> = graphs.collect();
    let mut boxes = prepare(definitions, rows, graphs.iter().copied())?;
    let explicit: HashSet<_> = definitions
        .iter()
        .map(|b| (b.variable.network, b.variable.id))
        .collect();
    for model in originals {
        for variable in &model.variables {
            if variable.accumulated_resource.is_some() {
                continue;
            }
            if explicit.contains(&(variable.network, variable.id)) {
                continue;
            }
            let implicit = GroupedVariableBound {
                variable: variable.clone(),
                lower: 0.,
                upper: None,
            };
            // An unavailable proof declines this optimization, not the model.
            // Explicit user envelopes above retain their validation errors.
            let Ok(prepared) = prepare(&[implicit], rows, graphs.iter().copied()) else {
                continue;
            };
            for (network, incoming) in prepared {
                match boxes.entry(network) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(incoming);
                    }
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        let current = entry.get_mut();
                        for (cap, other) in current.caps.iter_mut().zip(incoming.caps) {
                            *cap = cap.min(other);
                        }
                        current.variables.extend(incoming.variables);
                    }
                }
            }
        }
    }
    Ok(boxes)
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/grouped_activity_box_tests.rs"]
mod activity_tests;

pub(super) fn prepare<'a, const D: usize>(
    definitions: &[GroupedVariableBound],
    rows: usize,
    graphs: impl Iterator<Item = (i32, &'a Graph<D>, &'a [bool])>,
) -> Result<HashMap<i32, PreparedBox>> {
    let graphs: HashMap<_, _> = graphs.map(|(id, g, f)| (id, (g, f))).collect();
    let mut boxes = HashMap::<i32, PreparedBox>::new();
    let mut seen = HashSet::new();
    for definition in definitions {
        let variable = &definition.variable;
        let &(graph, forbidden) = graphs
            .get(&variable.network)
            .ok_or_else(|| invalid("unknown grouped-variable network"))?;
        graph.validate()?;
        let upper = definition.upper.unwrap_or(f64::INFINITY);
        if !seen.insert((variable.network, variable.id))
            || (variable.accumulated_resource.is_some() && !variable.arcs.is_empty())
            || variable.id < 0
            || !definition.lower.is_finite()
            || upper.is_nan()
            || upper < definition.lower
            || !variable.cost.is_finite()
            || variable
                .entries
                .iter()
                .any(|&(r, a)| r >= rows || !a.is_finite())
        {
            return Err(invalid("invalid grouped-variable box"));
        }
        if let Some(resource) = variable.accumulated_resource {
            activity_rows::validate(
                graph,
                &AccumulatedActivityCoefficients {
                    network: variable.network,
                    resource,
                    coefficient: 1.,
                },
            )?;
        }
        let positions: HashMap<_, _> = graph
            .arcs
            .iter()
            .enumerate()
            .map(|(p, a)| (a.id, p))
            .collect();
        if positions.len() != graph.arcs.len() {
            return Err(invalid("duplicate grouped-variable arc ID"));
        }
        let setters: BTreeMap<_, _> = variable.arcs.iter().copied().collect();
        let mut weights = vec![0.; graph.arcs.len()];
        for (&id, &a) in &setters {
            let &p = positions
                .get(&id)
                .ok_or_else(|| invalid("unknown grouped-variable arc"))?;
            if !a.is_finite() {
                return Err(invalid("nonfinite grouped-variable coefficient"));
            }
            weights[p] = a;
        }
        let mut proof = Bounds {
            lower: if weights.iter().all(|&a| a >= 0.) {
                0.
            } else {
                f64::NEG_INFINITY
            },
            upper: if weights.iter().all(|&a| a <= 0.) {
                0.
            } else {
                f64::INFINITY
            },
        };
        let mut empty = false;
        match walk_range(graph, forbidden, &weights) {
            Some(Some(range)) => proof = range,
            Some(None) => empty = true,
            None => {}
        }
        if let Some(resource) = variable.accumulated_resource {
            match activity_range(graph, resource) {
                Some(Some(range)) => proof = range,
                Some(None) => empty = true,
                None => {
                    return Err(invalid(
                        "accumulated-variable envelope proof is unavailable",
                    ))
                }
            }
        }
        // Matching scalar resources supply graph-derived envelopes even when
        // cycles prevent an affordable walk expansion. Disposable waiting can
        // only raise the resource; equality axes also supply a lower envelope.
        if !graph.has_coupled_resources() {
            for r in 0..graph.resources.len() {
                if graph
                    .arcs
                    .iter()
                    .enumerate()
                    .any(|(p, a)| forbidden.get(p) != Some(&true) && weights[p] != a.consumption[r])
                {
                    continue;
                }
                let initial =
                    graph.initial[r].unwrap_or(graph.vertices[graph.source].bounds[r].lower);
                if let Some(cap) = add_up(graph.vertices[graph.sink].bounds[r].upper, -initial) {
                    proof.upper = proof.upper.min(cap);
                }
                if graph.resources[r] == crate::graph::ResourceKind::Equality {
                    if let Some(initial) = graph.root_resource(r, true) {
                        if let Some(lower) =
                            add_down(graph.vertices[graph.sink].bounds[r].lower, -initial)
                        {
                            proof.lower = proof.lower.max(lower);
                        }
                    }
                }
            }
        }
        empty |= proof.lower > proof.upper;
        if !empty && (definition.lower > proof.lower || upper < proof.upper) {
            return Err(invalid(
                "grouped-variable envelope is unproved for the graph domain",
            ));
        }
        let entry = boxes.entry(variable.network).or_insert_with(|| {
            let cap = if graph.is_acyclic() {
                1.
            } else {
                graph
                    .max_route_arcs()
                    .or_else(|| {
                        (!graph.has_coupled_resources())
                            .then(|| super::bounds::resource_hop_limit(graph, 1_000_000))
                            .flatten()
                    })
                    .filter(|&n| (n as u128) <= 9_007_199_254_740_992)
                    .map_or(f64::INFINITY, |n| n as f64)
            };
            PreparedBox {
                variables: vec![],
                caps: graph
                    .arcs
                    .iter()
                    .enumerate()
                    .map(|(p, _)| {
                        if forbidden.get(p) == Some(&true) {
                            0.
                        } else {
                            cap
                        }
                    })
                    .collect(),
            }
        });
        // For a nonnegative expression each positive occurrence is individually
        // capped by its proved group upper envelope, including overlapping groups.
        if weights.iter().all(|&a| a >= 0.) && upper.is_finite() {
            for (p, &a) in weights.iter().enumerate().filter(|(_, a)| **a > 0.) {
                let ratio = upper.max(0.) / a;
                if ratio.is_finite() {
                    let cap = if ratio == 0. {
                        0.
                    } else {
                        f64::from_bits(ratio.to_bits() + 1).floor()
                    };
                    entry.caps[p] = entry.caps[p].min(cap);
                }
            }
        }
        entry.variables.push(PreparedVariable {
            arcs: weights
                .into_iter()
                .enumerate()
                .filter(|(_, a)| *a != 0.)
                .collect(),
            accumulated_resource: variable.accumulated_resource,
            cost: variable.cost,
            entries: variable.entries.clone(),
            bounds: Bounds {
                lower: definition.lower,
                upper,
            },
        });
    }
    Ok(boxes)
}
