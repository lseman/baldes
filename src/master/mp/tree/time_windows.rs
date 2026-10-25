//! Branching on how often each time window of a network vertex is selected.
//!
//! Each network's arcs into multi-window vertices are copied once per window
//! (`Graph::split_time_window_arcs`), so a selection count is an activity row
//! over window copies. Fractional counts get floor/ceil children. Visits of a
//! synchronized service must share one start, so integral counts may still
//! select two windows that cannot meet; one of them is then forbidden in each
//! child, before the service's start domain is split.
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

/// Floor/ceil children on the selection count closest to one half, or None
/// when every count is integral. `activities[network][arc]` is the arc flow.
pub(super) fn selection_children<const D: usize>(
    node: &NodeSnapshot<D>,
    windows: &[NetworkWindows],
    activities: &HashMap<i32, Vec<f64>>,
    tolerance: f64,
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let mut best: Option<(f64, i32, Vec<i32>, f64)> = None;
    for w in windows {
        let Some(oracle) = node.state.oracles.iter().find(|o| o.network.id == w.network) else {
            continue;
        };
        let flows = &activities[&w.network];
        for arcs in w.split.selections().into_values() {
            let value: f64 = arcs.iter().map(|id| flows[oracle.index[id]]).sum();
            if value < 0. || !super::branching::fractional(value, tolerance) {
                continue;
            }
            let score = (value - value.floor() - 0.5).abs();
            if best.as_ref().map_or(true, |b| score < b.0) {
                best = Some((score, w.network, arcs, value));
            }
        }
    }
    let Some((_, network, arcs, value)) = best else {
        return Ok(None);
    };
    let coefficients = [row(node, network, &arcs)];
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

/// For a served synchronized service, two selected windows that cannot share
/// a start (disjoint within the service domain; possibly two windows of one
/// member) cannot both be used: one child forbids each. None when no such
/// pair exists, leaving the start-domain split to synchronization.
pub(super) fn synchronized_children<const D: usize>(
    output: &Output<D>,
    windows: &[NetworkWindows],
    tolerance: f64,
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let node = &output.snapshot;
    let routes: Vec<_> = output
        .routes
        .iter()
        .zip(output.values.iter().copied())
        .chain(node.state.fixed_routes.iter().map(|f| (&f.route, f.quantity)))
        .filter(|(_, quantity)| *quantity > tolerance)
        .collect();
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
            let zero = Bounds {
                lower: 0.,
                upper: 0.,
            };
            let forbid = |(network, vertex, window): (i32, usize, usize)| {
                let arcs = windows
                    .iter()
                    .find(|w| w.network == network && w.split.resource == resource)
                    .and_then(|w| w.split.selections().remove(&(vertex, window)))
                    .unwrap_or_default();
                node.with_activity_bounds(zero, &[row(node, network, &arcs)])
            };
            return Ok(Some(vec![forbid(visits[i].0)?, forbid(visits[j].0)?]));
        }
    }
    Ok(None)
}
