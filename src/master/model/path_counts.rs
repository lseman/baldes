//! Explicit branching on the number of complete paths of a registered network.
use super::model_strong::{Proposal, Restriction};
use super::*;

/// Register a network's integral path count at its own branching priority.
///
/// Requires distinct source and sink, no incoming source or outgoing sink arc,
/// and no direct source-to-sink arc. Each complete path then contributes one to
/// the count, equivalently half its source/sink boundary activity. Fixed copies
/// remain part of the original count. This registration retains its priority
/// under all branching presets and participates in branching history.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathsPerNetworkBranching {
    pub network: i32,
    pub priority: f64,
}

pub(super) fn validate<'a, const D: usize>(
    registrations: &[PathsPerNetworkBranching],
    graphs: impl Iterator<Item = (i32, &'a Graph<D>)>,
) -> Result<()> {
    if registrations.is_empty() {
        return Ok(());
    }
    let graphs: HashMap<_, _> = graphs.collect();
    let mut seen = HashSet::new();
    for registration in registrations {
        if !registration.priority.is_finite() || !seen.insert(registration.network) {
            return Err(invalid(
                "path-count branching priority or duplicate network",
            ));
        }
        let graph = graphs
            .get(&registration.network)
            .ok_or_else(|| invalid("unknown path-count branching network"))?;
        if graph.source == graph.sink
            || graph.source >= graph.vertices.len()
            || graph.sink >= graph.vertices.len()
            || graph.arcs.iter().any(|arc| {
                arc.head == graph.source
                    || arc.tail == graph.sink
                    || (arc.tail == graph.source && arc.head == graph.sink)
            })
        {
            return Err(invalid(
                "path-count branching requires an open source/sink path domain without an empty path",
            ));
        }
    }
    Ok(())
}

pub(super) fn proposals<const D: usize>(
    output: &Output<D>,
    registrations: &[PathsPerNetworkBranching],
    _tolerance: f64,
) -> Result<Vec<Proposal>> {
    if registrations.is_empty() {
        return Ok(vec![]);
    }
    validate(
        registrations,
        output
            .snapshot
            .state
            .oracles
            .iter()
            .map(|oracle| (oracle.network.id, oracle.network.graph.as_ref())),
    )?;
    let mut proposals = vec![];
    for (registration_index, registration) in registrations.iter().enumerate() {
        let value = output
            .routes
            .iter()
            .zip(&output.values)
            .filter(|(route, _)| route.network == registration.network)
            .map(|(_, &quantity)| quantity)
            .chain(
                output
                    .snapshot
                    .state
                    .fixed_routes
                    .iter()
                    .filter(|fixed| fixed.route.network == registration.network)
                    .map(|fixed| fixed.quantity),
            )
            .sum::<f64>();
        if !value.is_finite() || !(0. ..9_007_199_254_740_992.).contains(&value) {
            return Err(invalid("path-count branching activity is out of range"));
        }
        let fraction = value.fract();
        // This family's integrality precision is independent of the generic
        // component-class violation threshold and the global tree tolerance.
        if !(1e-8..=1. - 1e-8).contains(&fraction) {
            continue;
        }
        let name = format!("paths:{}", registration.network);
        proposals.push(Proposal {
            history_key: name.clone(),
            name,
            priority: registration.priority,
            value,
            restriction: Restriction::PathCount {
                network: registration.network,
            },
            history_eligible: true,
            fresh: true,
            origin: Some(super::model_strong::ProducerFamily::PathsPerNetwork(
                registration_index,
            )),
            component_admission: None,
        });
    }
    Ok(proposals)
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/path_count_tests.rs"]
mod tests;
