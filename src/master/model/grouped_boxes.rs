//! Verified envelopes for original variables sharing physical arcs.
use super::*;
use crate::{core::rounding::add_down, master::preprocessing::Bounds};

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
