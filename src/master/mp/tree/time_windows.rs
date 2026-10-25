//! Branching on how often each time window of a network vertex is selected.
//!
//! Following Li et al. (2020), each network's arcs into multi-window vertices
//! are copied once per window (`Graph::split_time_window_arcs`), so a
//! selection count is an activity row over window copies:
//! 1. arc flows are branched on summed over each input arc's copies;
//! 2. a fractional count gets `<= floor` and `>= ceil` children; the up child
//!    also forbids the vertex's other windows where at most one visit or one
//!    common start is possible (unit-bounded vertices, synchronized members);
//! 3. a synchronized member that selects several windows has one of them
//!    forbidden in one child and forced in the other.
//!
//! An extension then separates selected windows of different members that
//! cannot share a start, before the service's start domain is split.
//! Forbidden copies are removed from pricing by activity preprocessing.
use super::*;
use crate::graph::WindowArcs;
use crate::master::preprocessing::Bounds;

/// Window copies of one network.
pub(super) struct NetworkWindows {
    pub network: i32,
    pub split: WindowArcs,
}

/// Split every network on `resource`, keeping `arc_entries` aligned with the
/// copied arcs. Networks without multi-window vertices are unchanged.
#[cfg(feature = "multi-time-windows")]
pub(super) fn split<const D: usize>(
    problem: &mut Problem<D>,
    resource: usize,
) -> Result<Vec<NetworkWindows>> {
    let mut out = vec![];
    for network in &mut problem.networks {
        let positions: HashMap<_, _> = network
            .graph
            .arcs
            .iter()
            .enumerate()
            .map(|(i, a)| (a.id, i))
            .collect();
        let split = network.graph.split_time_window_arcs(resource)?;
        // Copies are appended to the graph in the order they are recorded.
        for copy in split.arcs.iter().filter(|a| a.arc != a.original) {
            let entries = network.arc_entries[positions[&copy.original]].clone();
            network.arc_entries.push(entries);
        }
        if !split.arcs.is_empty() {
            out.push(NetworkWindows {
                network: network.id,
                split,
            });
        }
    }
    Ok(out)
}

/// Rewrite a route of a split network in input arc IDs.
#[cfg(feature = "multi-time-windows")]
pub(super) fn restore(windows: &[NetworkWindows], route: &mut NetworkRoute) {
    if let Some(w) = windows.iter().find(|w| w.network == route.network) {
        w.split.restore(&mut route.route);
    }
}

fn row<const D: usize>(node: &NodeSnapshot<D>, network: i32, arcs: &[i32]) -> RowCoefficients {
    let oracle = node
        .state
        .oracles
        .iter()
        .find(|o| o.network.id == network)
        .expect("window copies belong to a tree network");
    let mut coefficients = vec![0.; oracle.network.graph.arcs.len()];
    for id in arcs {
        coefficients[oracle.index[id]] = 1.;
    }
    RowCoefficients {
        network,
        arcs: coefficients,
        per_route: 0.,
    }
}

fn oracle<const D: usize>(node: &NodeSnapshot<D>, network: i32) -> Option<&Oracle<D>> {
    node.state.oracles.iter().find(|o| o.network.id == network)
}

fn zero() -> Bounds {
    Bounds {
        lower: 0.,
        upper: 0.,
    }
}

fn fraction_score(value: f64) -> f64 {
    (value - value.floor() - 0.5).abs()
}

/// Floor/ceil children on the flow of one input arc, summed over its window
/// copies, closest to one half; None when every such flow is integral.
pub(super) fn arc_children<const D: usize>(
    node: &NodeSnapshot<D>,
    windows: &[NetworkWindows],
    activities: &HashMap<i32, Vec<f64>>,
    tolerance: f64,
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let mut best: Option<(f64, i32, Vec<i32>, f64)> = None;
    for w in windows {
        let Some(oracle) = oracle(node, w.network) else {
            continue;
        };
        let mut copies = BTreeMap::<i32, Vec<i32>>::new();
        for a in &w.split.arcs {
            copies.entry(a.original).or_default().push(a.arc);
        }
        for ids in copies.into_values().filter(|ids| ids.len() > 1) {
            let value: f64 = ids
                .iter()
                .map(|id| activities[&w.network][oracle.index[id]])
                .sum();
            if value >= 0.
                && super::branching::fractional(value, tolerance)
                && best.as_ref().map_or(true, |b| fraction_score(value) < b.0)
            {
                best = Some((fraction_score(value), w.network, ids, value));
            }
        }
    }
    let Some((_, network, ids, value)) = best else {
        return Ok(None);
    };
    let coefficients = [row(node, network, &ids)];
    Ok(Some(vec![
        node.with_activity_bounds(
            Bounds {
                lower: 0.,
                upper: value.floor(),
            },
            &coefficients,
        )?,
        node.with_activity_bounds(
            Bounds {
                lower: value.ceil(),
                upper: f64::INFINITY,
            },
            &coefficients,
        )?,
    ]))
}

/// Whether every visit of (network, vertex) shares one window in any integer
/// solution: a synchronized member (all visits share a start and a vertex's
/// windows are disjoint), or a vertex whose copies all carry coefficient one
/// in a nonnegative shared row with right-hand side at most one.
fn single_window<const D: usize>(node: &NodeSnapshot<D>, network: i32, vertex: usize) -> bool {
    if node.state.synchronized_services.iter().any(|s| {
        s.definition
            .members
            .iter()
            .any(|m| m.network == network && m.vertex == vertex)
    }) {
        return true;
    }
    let Some(oracle) = oracle(node, network) else {
        return false;
    };
    let entries = &oracle.network.arc_entries;
    let into: Vec<_> = (0..oracle.network.graph.arcs.len())
        .filter(|&a| oracle.network.graph.arcs[a].head == vertex)
        .collect();
    let Some(&first) = into.first() else {
        return false;
    };
    entries[first].iter().any(|&(r, value)| {
        let Some(shared) = node.state.shared_rows.get(r) else {
            return false;
        };
        value == 1.
            && matches!(shared.sense, RowSense::Equal | RowSense::LessEqual)
            && shared.rhs <= 1.
            && into
                .iter()
                .all(|&a| entries[a].iter().any(|&(q, v)| q == r && v == 1.))
            && node.state.oracles.iter().all(|o| {
                o.network
                    .arc_entries
                    .iter()
                    .flatten()
                    .all(|&(q, v)| q != r || v >= 0.)
                    && o.network
                        .route_entries
                        .iter()
                        .all(|&(q, v)| q != r || v >= 0.)
            })
            && node
                .state
                .master_variables
                .iter()
                .all(|m| m.entries.iter().all(|&(q, v)| q != r || v >= 0.))
    })
}

/// Children on the selection count closest to one half, or None when every
/// count is integral. `activities[network][arc]` is the arc flow.
pub(super) fn selection_children<const D: usize>(
    node: &NodeSnapshot<D>,
    windows: &[NetworkWindows],
    activities: &HashMap<i32, Vec<f64>>,
    tolerance: f64,
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let mut best: Option<(f64, i32, usize, usize, f64)> = None;
    for w in windows {
        let Some(oracle) = oracle(node, w.network) else {
            continue;
        };
        for ((vertex, window), ids) in w.split.selections() {
            let value: f64 = ids
                .iter()
                .map(|id| activities[&w.network][oracle.index[id]])
                .sum();
            if value >= 0.
                && super::branching::fractional(value, tolerance)
                && best.as_ref().map_or(true, |b| fraction_score(value) < b.0)
            {
                best = Some((fraction_score(value), w.network, vertex, window, value));
            }
        }
    }
    let Some((_, network, vertex, window, value)) = best else {
        return Ok(None);
    };
    let selections = windows
        .iter()
        .find(|w| w.network == network)
        .map(|w| w.split.selections())
        .unwrap_or_default();
    let chosen = row(node, network, &selections[&(vertex, window)]);
    let down = node.with_activity_bounds(
        Bounds {
            lower: 0.,
            upper: value.floor(),
        },
        std::slice::from_ref(&chosen),
    )?;
    let mut up = node.with_activity_bounds(
        Bounds {
            lower: value.ceil(),
            upper: f64::INFINITY,
        },
        &[chosen],
    )?;
    // Right child: W_i = {w}, where every visit must use a single window.
    let others = others(&selections, vertex, window);
    if !others.is_empty() && single_window(node, network, vertex) {
        up = up.with_activity_bounds(zero(), &[row(&up, network, &others)])?;
    }
    Ok(Some(vec![down, up]))
}

fn others(
    selections: &BTreeMap<(usize, usize), Vec<i32>>,
    vertex: usize,
    window: usize,
) -> Vec<i32> {
    selections
        .iter()
        .filter(|&(&(v, w), _)| v == vertex && w != window)
        .flat_map(|(_, ids)| ids.iter().copied())
        .collect()
}

/// Positive-quantity routes of the node's LP, including fixed routes.
fn positive<const D: usize>(output: &Output<D>, tolerance: f64) -> Vec<(&NetworkRoute, f64)> {
    output
        .routes
        .iter()
        .zip(output.values.iter().copied())
        .chain(
            output
                .snapshot
                .state
                .fixed_routes
                .iter()
                .map(|f| (&f.route, f.quantity)),
        )
        .filter(|(_, quantity)| *quantity > tolerance)
        .collect()
}

/// Reproducible pseudo-random index in 0..n (splitmix64 of `key`); the paper
/// chooses the customer and window at random.
fn pick(key: u64, n: usize) -> usize {
    let mut z = key.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) % n as u64) as usize
}

/// Second situation of Li et al. (2020): every count is integral but a
/// synchronized member selects several windows. Pick such a member and one of
/// its selected windows; forbid that window in the left child and force it in
/// the right child.
pub(super) fn multiple_window_children<const D: usize>(
    output: &Output<D>,
    windows: &[NetworkWindows],
    tolerance: f64,
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let node = &output.snapshot;
    let routes = positive(output, tolerance);
    // (network, vertex) -> selected windows.
    let mut selected = BTreeMap::<(i32, usize), Vec<usize>>::new();
    for service in &node.state.synchronized_services {
        for (route, _) in &routes {
            let Some(split) = windows.iter().find(|w| {
                w.network == route.network && w.split.resource == service.definition.resource
            }) else {
                continue;
            };
            for id in &route.route.arcs {
                let Some(copy) = split.split.arcs.iter().find(|a| a.arc == *id) else {
                    continue;
                };
                if service
                    .definition
                    .members
                    .iter()
                    .any(|m| m.network == route.network && m.vertex == copy.vertex)
                {
                    let list = selected.entry((route.network, copy.vertex)).or_default();
                    if !list.contains(&copy.window) {
                        list.push(copy.window);
                    }
                }
            }
        }
    }
    let members: Vec<_> = selected.into_iter().filter(|(_, w)| w.len() > 1).collect();
    if members.is_empty() {
        return Ok(None);
    }
    let key = members.iter().fold(members.len() as u64, |h, ((n, v), w)| {
        h.wrapping_mul(31)
            .wrapping_add(*n as u64 ^ (*v as u64) << 20 ^ (w.len() as u64) << 40)
    });
    let ((network, vertex), mut chosen) = members[pick(key, members.len())].clone();
    chosen.sort_unstable();
    let window = chosen[pick(key.rotate_left(17), chosen.len())];
    let selections = windows
        .iter()
        .find(|w| w.network == network)
        .map(|w| w.split.selections())
        .unwrap_or_default();
    let others = others(&selections, vertex, window);
    Ok(Some(vec![
        node.with_activity_bounds(
            zero(),
            &[row(node, network, &selections[&(vertex, window)])],
        )?,
        node.with_activity_bounds(zero(), &[row(node, network, &others)])?,
    ]))
}

/// Extension for members with separate window sets: two selected windows of
/// a served synchronized service that cannot share a start (disjoint within
/// the service domain) cannot both be used, so one child forbids each. None when no such
/// pair exists, leaving the start-domain split to synchronization.
pub(super) fn conflict_children<const D: usize>(
    output: &Output<D>,
    windows: &[NetworkWindows],
    tolerance: f64,
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let node = &output.snapshot;
    let routes = positive(output, tolerance);
    for service in &node.state.synchronized_services {
        let served = output
            .master_values
            .iter()
            .find(|&&(id, _)| id == service.variable())
            .map_or(0., |&(_, v)| v);
        if served <= tolerance {
            continue;
        }
        let resource = service.definition.resource;
        let [lower, upper] = service.domain.map(|t| t as f64);
        // (network, vertex, window) -> (start interval within the domain, quantity).
        let mut selected = BTreeMap::<(i32, usize, usize), (f64, f64, f64)>::new();
        for (route, quantity) in &routes {
            let (Some(oracle), Some(split)) = (
                node.state
                    .oracles
                    .iter()
                    .find(|o| o.network.id == route.network),
                windows
                    .iter()
                    .find(|w| w.network == route.network && w.split.resource == resource),
            ) else {
                continue;
            };
            let graph = &oracle.network.graph;
            for id in &route.route.arcs {
                let arc = &graph.arcs[oracle.index[id]];
                let member = service
                    .definition
                    .members
                    .iter()
                    .any(|m| m.network == route.network && m.vertex == arc.head);
                let copy = split.split.arcs.iter().find(|a| a.arc == *id);
                let (true, Some(copy), Some(window)) = (member, copy, arc.bounds[resource]) else {
                    continue;
                };
                let entry = selected
                    .entry((route.network, copy.vertex, copy.window))
                    .or_insert((window.lower.max(lower), window.upper.min(upper), 0.));
                entry.2 += quantity;
            }
        }
        let visits: Vec<_> = selected.into_iter().collect();
        let mut best: Option<(f64, usize, usize)> = None;
        for i in 0..visits.len() {
            for j in i + 1..visits.len() {
                let (a, b) = (visits[i].1, visits[j].1);
                if a.0.max(b.0) <= a.1.min(b.1) {
                    continue;
                }
                let weight = a.2.min(b.2);
                if best.map_or(true, |(w, _, _)| weight < w) {
                    best = Some((weight, i, j));
                }
            }
        }
        if let Some((_, i, j)) = best {
            let forbid = |(network, vertex, window): (i32, usize, usize)| {
                let arcs = windows
                    .iter()
                    .find(|w| w.network == network && w.split.resource == resource)
                    .and_then(|w| w.split.selections().remove(&(vertex, window)))
                    .unwrap_or_default();
                node.with_activity_bounds(zero(), &[row(node, network, &arcs)])
            };
            return Ok(Some(vec![forbid(visits[i].0)?, forbid(visits[j].0)?]));
        }
    }
    Ok(None)
}
