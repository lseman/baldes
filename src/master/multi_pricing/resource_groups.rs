//! Owned accumulated-resource disjunctions over one shared unit packing row.
use super::*;
use crate::{
    master::branching::AccumulatedResourceGroupBranch,
    pricing::restrictions::AccumulatedResourceBranch,
};
use std::borrow::Cow;
mod profiles;

/// Physical occurrences of one shared packing group in one pricing network.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceGroupMember {
    pub network: i32,
    pub arcs: Vec<i32>,
}

pub(super) fn entries<const D: usize>(oracle: &Oracle<D>, route: &Route) -> Vec<(usize, f64)> {
    oracle
        .resource_groups
        .iter()
        .map(|(row, group)| (*row, group.coefficient(route, false)))
        .collect()
}

fn windows_graph<const D: usize>(oracle: &Oracle<D>) -> Result<(Graph<D>, HashSet<i32>)> {
    let mut graph = oracle.network.graph.clone();
    let mut blocked = HashSet::new();
    for (_, group) in &oracle.resource_groups {
        group.install(&mut graph)?;
        blocked.extend(
            group
                .arcs()
                .iter()
                .copied()
                .filter(|&a| group.additive(a) != 0.),
        );
    }
    Ok((graph, blocked))
}

pub(super) fn domain_graph<const D: usize>(oracle: &Oracle<D>) -> Result<Cow<'_, Graph<D>>> {
    if oracle.resource_groups.is_empty() {
        return Ok(Cow::Borrowed(&oracle.network.graph));
    }
    let (mut graph, blocked) = windows_graph(oracle)?;
    graph.arcs.retain(|a| !blocked.contains(&a.id));
    Ok(Cow::Owned(graph))
}

/// Residual activity-zero proofs constrain future copies, not routes already
/// moved into the fixed part. Keep them separate from schedule feasibility.
pub(super) fn residual_domain_graph<const D: usize>(
    oracle: &Oracle<D>,
) -> Result<Cow<'_, Graph<D>>> {
    let mut graph = domain_graph(oracle)?;
    if !oracle.resource_groups.is_empty() && oracle.proved_zero_arcs.iter().any(|&v| v) {
        graph.to_mut().arcs.retain(|a| {
            !oracle
                .proved_zero_arcs
                .get(oracle.index[&a.id])
                .copied()
                .unwrap_or(false)
        });
    }
    Ok(graph)
}

pub(super) fn check_load_rows<const D: usize>(
    state: &State<D>,
    network: i32,
    resource: usize,
) -> Result<()> {
    let oracle = state
        .oracles
        .iter()
        .find(|o| o.network.id == network)
        .ok_or_else(|| invalid("unknown resource group network"))?;
    if state.universe.is_none()
        && oracle.network.graph.resources.get(resource)
            == Some(&crate::graph::ResourceKind::Disposable)
        && oracle
            .resource_groups
            .iter()
            .any(|(_, g)| g.upper() && g.resource() == resource)
    {
        return Err(invalid(
            "resource-load rows on a re-timed branch axis require complete enumeration",
        ));
    }
    Ok(())
}

impl<const D: usize> NodeSnapshot<D> {
    /// Branch on one resource coordinate at every supplied occurrence of a
    /// shared packing group. The original row must prove at most one occurrence
    /// across all networks and static variables. Both children include t-1e-6.
    /// The parent is unchanged, including on validation or backend failure.
    /// Re-timing a disposable axis with resource-load rows requires a complete
    /// universe, so their original path coefficients remain unchanged.
    pub fn with_resource_group_branch(
        &self,
        packing_row: usize,
        resource: usize,
        threshold: f64,
        upper: bool,
        members: &[ResourceGroupMember],
    ) -> Result<Self> {
        let state = &self.state;
        let row = state
            .shared_rows
            .get(packing_row)
            .ok_or_else(|| invalid("shared resource group packing row"))?;
        if !matches!(row.sense, RowSense::Equal | RowSense::LessEqual)
            || row.rhs > 1.
            || !threshold.is_finite()
            || members.is_empty()
        {
            return Err(invalid(
                "resource group requires a finite threshold and a unit packing row",
            ));
        }
        // Contributions outside the selected supports must not cancel a group
        // occurrence. Fixed copies retain these original nonnegative terms.
        if state.oracles.iter().any(|o| {
            o.network
                .arc_entries
                .iter()
                .flatten()
                .chain(&o.network.route_entries)
                .any(|&(r, a)| r == packing_row && a < 0.)
        }) || state.master_variables.iter().any(|v| {
            v.entries.iter().any(|&(r, a)| {
                r == packing_row
                    && ((a > 0. && v.bounds.lower < 0.) || (a < 0. && v.bounds.upper > 0.))
            })
        }) {
            return Err(invalid(
                "shared resource group needs nonnegative original packing contributions",
            ));
        }
        let mut seen = HashSet::new();
        let mut prepared = vec![];
        for member in members {
            let oracle = state
                .oracles
                .iter()
                .find(|o| o.network.id == member.network)
                .ok_or_else(|| invalid("unknown resource group network"))?;
            if !seen.insert(member.network) {
                return Err(invalid("duplicate resource group network"));
            }
            if upper
                && oracle.network.graph.resources.get(resource)
                    == Some(&crate::graph::ResourceKind::Disposable)
                && state.universe.is_none()
                && oracle.knapsack_rows.iter().any(|&(index, _)| {
                    oracle.solver.cuts.knapsack_cuts[index].graph_resources[&member.network].0
                        == resource
                })
            {
                return Err(invalid(
                    "re-timing a resource-load row axis requires complete enumeration",
                ));
            }
            let group = AccumulatedResourceGroupBranch::new(
                packing_row,
                member.arcs.clone(),
                resource,
                threshold,
                upper,
            )?;
            group.validate_support(
                &oracle.network.graph,
                &state.shared_rows,
                &oracle.network.arc_entries,
                &oracle.network.route_entries,
            )?;
            let graph = windows_graph(oracle)?.0;
            prepared.push((member.network, group.prepare(&graph)?));
        }
        let shared = state.shared_rows.len();
        let mut node = self.with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &[],
        )?;
        for (network, group) in prepared {
            let oracle = node
                .state
                .oracles
                .iter_mut()
                .find(|o| o.network.id == network)
                .unwrap();
            for &arc in group.arcs() {
                oracle
                    .solver
                    .add_accumulated_resource_branch(AccumulatedResourceBranch {
                        arc,
                        resource,
                        threshold,
                        upper,
                    })?;
            }
            oracle.resource_groups.push((shared, group));
        }
        node = super::nonlinear::refresh_row(node, shared)?;
        if profiles::refresh(&mut node, &seen)? {
            // An original-feasible fixed path can become exactly impossible
            // under a new window even if its rounded profile passes the rule.
            node = node.with_row(
                Row {
                    sense: RowSense::LessEqual,
                    rhs: -1.,
                },
                &[],
            )?;
        }
        node.state.correction = None;
        node.state.gradient = None;
        node.warm.stabilization = None;
        Ok(node)
    }

    /// Network, appended shared row and the owned original-group restriction.
    pub fn resource_group_branches(
        &self,
    ) -> impl Iterator<Item = (i32, usize, &AccumulatedResourceGroupBranch)> {
        self.state.oracles.iter().flat_map(|o| {
            o.resource_groups
                .iter()
                .map(move |(row, g)| (o.network.id, *row, g))
        })
    }
}
