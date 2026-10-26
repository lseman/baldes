//! Lightweight original-domain generator registry for opt-in phased branching.
//! Full masters are forked only after native priority/quota admission.
use super::*;
use crate::master::model::model_strong::{Proposal, Restriction};

pub(super) fn collect<const D: usize>(
    output: &Output<D>,
    options: &ModelTreeOptions,
    windows: &[time_windows::NetworkWindows],
) -> Result<Vec<Proposal>> {
    let node = &output.snapshot;
    let tolerance = options.tolerance;
    let priorities = &options.branch_priorities;
    let mut proposals =
        model_strong::mapped_proposals(output, &options.mapped_variables, tolerance)?;
    for candidate in node.resource_group_candidates(tolerance)? {
        proposals.push(resource_proposal(candidate, priorities.resource));
    }
    // Native bypasses this separator when strong branching is disabled.
    if options.mapped_strong_branching.is_some() {
        if let Some(packing) = &options.packing {
            proposals.extend(packing.proposals(output, tolerance));
        }
    }
    for candidate in if options.packing.is_some() {
        vec![]
    } else {
        node.fractional_assignments(tolerance)?
    } {
        let name = format!("assignment:{}:{}", candidate.shared_row, candidate.network);
        proposals.push(Proposal {
            history_key: name.clone(),
            name,
            priority: priorities.assignment,
            value: candidate.value,
            restriction: Restriction::Assignment {
                shared_row: candidate.shared_row,
                network: candidate.network,
            },
            history_eligible: true,
            fresh: true,
        });
    }
    for variable in &node.state.master_variables {
        let Some(&(_, value)) = output
            .master_values
            .iter()
            .find(|(id, _)| *id == variable.id)
        else {
            return Err(invalid("missing original master value for branch registry"));
        };
        if variable.integer && branching::fractional(value, tolerance) {
            let name = format!("master:{}", variable.id);
            proposals.push(Proposal {
                history_key: name.clone(),
                name,
                priority: priorities.master_variable,
                value,
                restriction: Restriction::Master { id: variable.id },
                history_eligible: true,
                fresh: true,
            });
        }
    }
    let mut flows = HashMap::<i32, Vec<f64>>::new();
    for oracle in &node.state.oracles {
        let mut count = 0.;
        let mut arcs = vec![0.; oracle.network.graph.arcs.len()];
        for (route, quantity) in output
            .routes
            .iter()
            .zip(&output.values)
            .map(|(r, &q)| (r, q))
            .chain(
                node.state
                    .fixed_routes
                    .iter()
                    .map(|f| (&f.route, f.quantity)),
            )
        {
            if route.network != oracle.network.id || quantity == 0. {
                continue;
            }
            count += quantity;
            for id in &route.route.arcs {
                arcs[oracle.index[id]] += quantity;
            }
        }
        if count >= 0. && branching::fractional(count, tolerance) {
            let name = format!("multiplicity:{}", oracle.network.id);
            proposals.push(Proposal {
                history_key: name.clone(),
                name,
                priority: priorities.multiplicity,
                value: count,
                restriction: Restriction::Multiplicity {
                    network: oracle.network.id,
                },
                history_eligible: false,
                fresh: true,
            });
        }
        for (position, &value) in arcs.iter().enumerate() {
            if value < 0. || !branching::fractional(value, tolerance) {
                continue;
            }
            let id = oracle.network.graph.arcs[position].id;
            let name = format!("arc:{}:{id}", oracle.network.id);
            proposals.push(Proposal {
                history_key: name.clone(),
                name,
                priority: priorities.arc,
                value,
                restriction: Restriction::Activity {
                    network: oracle.network.id,
                    arcs: vec![(id, 1.)],
                },
                history_eligible: true,
                fresh: true,
            });
        }
        flows.insert(oracle.network.id, arcs);
    }
    for w in windows {
        let oracle = node
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == w.network)
            .ok_or_else(|| invalid("unknown window branch network"))?;
        for ((vertex, window), arcs) in w.split.selections() {
            let value = arcs
                .iter()
                .map(|id| flows[&w.network][oracle.index[id]])
                .sum::<f64>();
            if value < 0. || !branching::fractional(value, tolerance) {
                continue;
            }
            let name = format!("window:{}:{vertex}:{window}", w.network);
            proposals.push(Proposal {
                history_key: name.clone(),
                name,
                priority: priorities.window,
                value,
                restriction: Restriction::Activity {
                    network: w.network,
                    arcs: arcs.into_iter().map(|id| (id, 1.)).collect(),
                },
                history_eligible: true,
                fresh: true,
            });
        }
    }
    Ok(proposals)
}

fn resource_key(candidate: &ResourceGroupCandidate) -> String {
    let members: Vec<_> = candidate
        .members
        .iter()
        .map(|m| (m.network, &m.arcs))
        .collect();
    let boundary = candidate.threshold
        - crate::pricing::restrictions::AccumulatedResourceBranch::THRESHOLD_OFFSET;
    format!(
        "resource:{}:{}:{members:?}:{:016x}",
        candidate.packing_row,
        candidate.resource,
        if boundary == 0. {
            0
        } else {
            boundary.to_bits()
        }
    )
}

fn resource_proposal(candidate: ResourceGroupCandidate, priority: f64) -> Proposal {
    let name = format!(
        "resource:{}:{}:{:016x}",
        candidate.packing_row,
        candidate.resource,
        candidate.threshold.to_bits()
    );
    Proposal {
        history_key: resource_key(&candidate),
        name,
        priority,
        value: candidate.value,
        restriction: Restriction::Resource(candidate),
        history_eligible: true,
        fresh: true,
    }
}

/// Refresh a retained native resource generator at its saved threshold rather
/// than attributing observations of different disjunctions to one mean.
pub(in crate::master::model) fn refresh_resource<const D: usize>(
    output: &Output<D>,
    template: &ResourceGroupCandidate,
    priority: f64,
    tolerance: f64,
) -> Option<Proposal> {
    let node = &output.snapshot;
    let boundary = template.threshold
        - crate::pricing::restrictions::AccumulatedResourceBranch::THRESHOLD_OFFSET;
    if template.members.iter().all(|m| {
        node.state
            .oracles
            .iter()
            .find(|o| o.network.id == m.network)
            .is_some_and(|o| {
                o.resource_groups.iter().any(|(_, g)| {
                    g.packing_row() == template.packing_row
                        && g.resource() == template.resource
                        && g.boundary() == boundary
                        && g.arcs() == m.arcs
                })
            })
    }) {
        return None;
    }
    let mut total = 0.;
    let mut value = 0.;
    for (route, weight) in output
        .routes
        .iter()
        .zip(&output.values)
        .map(|(r, &q)| (r, q))
        .chain(
            node.state
                .fixed_routes
                .iter()
                .map(|f| (&f.route, f.quantity)),
        )
    {
        if !weight.is_finite() || weight < 0. {
            return None;
        }
        if weight == 0. {
            continue;
        }
        let Some(member) = template.members.iter().find(|m| m.network == route.network) else {
            continue;
        };
        for (position, arc) in route.route.arcs.iter().enumerate() {
            if !member.arcs.contains(arc) {
                continue;
            }
            let quantity = *route
                .route
                .resource_profile
                .get(position + 1)?
                .get(template.resource)?;
            if !quantity.is_finite() {
                return None;
            }
            total += weight;
            if quantity < boundary {
                value += weight;
            }
        }
    }
    if !total.is_finite() || total > 1. + tolerance || value <= tolerance || value >= 1. - tolerance
    {
        return None;
    }
    let mut candidate = template.clone();
    candidate.value = value;
    let mut proposal = resource_proposal(candidate, priority);
    proposal.fresh = false;
    Some(proposal)
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/model/network_proposal_tests.rs"]
mod tests;
