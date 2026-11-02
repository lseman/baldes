//! Lightweight original-domain generator registry for opt-in phased branching.
//! Full masters are forked only after native priority/quota admission.
use super::*;
use crate::master::model::model_strong::{Proposal, Restriction};

pub(super) fn multiplicity(network: i32, value: f64, priority: f64) -> Proposal {
    let name = format!("multiplicity:{network}");
    Proposal {
        history_key: name.clone(),
        name,
        priority,
        value,
        restriction: Restriction::Multiplicity { network },
        history_eligible: false,
        fresh: true,
        origin: None,
        component_admission: None,
    }
}

/// The inferred families, in the order of the former fixed cascades.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Inferred {
    Multiplicity,
    Resource,
    Assignment,
    Master,
    Arc,
    Window,
}

fn inferred(proposal: &Proposal) -> Option<Inferred> {
    Some(match &proposal.restriction {
        Restriction::Multiplicity { .. } => Inferred::Multiplicity,
        Restriction::Resource(_) => Inferred::Resource,
        Restriction::Assignment { .. } => Inferred::Assignment,
        Restriction::Master { .. } => Inferred::Master,
        Restriction::Activity { .. } if proposal.name.starts_with("arc:") => Inferred::Arc,
        Restriction::Activity { .. } if proposal.name.starts_with("window:") => Inferred::Window,
        _ => return None,
    })
}

/// `MultiplicityFirst` and `ResourceFirst` are presets over the same
/// priority dispatcher as `Phased`. Registered families (mapped variables,
/// clusters, component classes/generators, Ryan–Foster, packing) keep their
/// configured priorities.
/// Inferred families are ranked below the lowest registered priority in the
/// order of the former cascades: resource, assignment, master variable, arc
/// and window flow; route count comes first under `MultiplicityFirst` and
/// after master variables under `ResourceFirst`. `Phased` uses
/// `branch_priorities` unchanged.
fn preset_priorities(proposals: &mut [Proposal], options: &ModelTreeOptions) {
    let order: &[Inferred] = match options.branching_registry().branching {
        NetworkBranchingPolicy::Phased => return,
        NetworkBranchingPolicy::MultiplicityFirst => &[
            Inferred::Resource,
            Inferred::Assignment,
            Inferred::Master,
            Inferred::Arc,
            Inferred::Window,
        ],
        NetworkBranchingPolicy::ResourceFirst => &[
            Inferred::Resource,
            Inferred::Assignment,
            Inferred::Master,
            Inferred::Multiplicity,
            Inferred::Arc,
            Inferred::Window,
        ],
    };
    let (low, high) = super::super::family_registration::registered_priority_range(options);
    for proposal in proposals {
        let Some(family) = inferred(proposal) else {
            continue;
        };
        proposal.priority = if family == Inferred::Multiplicity
            && options.branching_registry().branching == NetworkBranchingPolicy::MultiplicityFirst
        {
            high + 1.
        } else {
            let rank = order.iter().position(|&f| f == family).unwrap();
            low - 1. - rank as f64
        };
    }
}

#[cfg(test)]
pub(super) fn collect<const D: usize>(
    output: &Output<D>,
    options: &ModelTreeOptions,
    windows: &[time_windows::NetworkWindows],
) -> Result<Vec<Proposal>> {
    collect_with_deadline(output, options, windows, None)
}

pub(super) fn collect_with_deadline<const D: usize>(
    output: &Output<D>,
    options: &ModelTreeOptions,
    windows: &[time_windows::NetworkWindows],
    deadline: Option<Instant>,
) -> Result<Vec<Proposal>> {
    let node = &output.snapshot;
    let tolerance = options.tolerance;
    let priorities = &options.branching_registry().branch_priorities;
    let mut proposals =
        super::super::family_registration::registered_proposals(output, options, deadline)?;
    proposals.extend(super::super::family_registration::path_proposals(
        output, options,
    )?);
    for candidate in node.resource_group_candidates(tolerance)? {
        proposals.push(resource_proposal(candidate, priorities.resource));
    }
    proposals.extend(super::super::family_registration::packing_proposals(
        output, options,
    ));
    for candidate in if options.branching_registry().packing.is_some() {
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
            origin: None,
            component_admission: None,
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
                origin: None,
                component_admission: None,
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
        // The default empty component sequence uses the class-violation
        // threshold, not the much smaller variable-integrality threshold.
        // Generic fallback disjunctions remain available if this registry is
        // exhausted, so declining this candidate does not certify integrality.
        let distance = (count - count.round()).abs();
        let minimum_fractionality = 0.02_f64;
        if !options
            .branching_registry()
            .paths_per_network
            .iter()
            .any(|p| p.network == oracle.network.id)
            && count >= 0.
            && branching::fractional(count, tolerance)
            && minimum_fractionality
                < distance - (distance.max(minimum_fractionality) * 1e-10 + 1e-6)
        {
            proposals.push(multiplicity(
                oracle.network.id,
                count,
                priorities.multiplicity,
            ));
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
                origin: None,
                component_admission: None,
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
                origin: None,
                component_admission: None,
            });
        }
    }
    preset_priorities(&mut proposals, options);
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
        origin: Some(super::super::model_strong::ProducerFamily::PackingResource),
        component_admission: None,
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
