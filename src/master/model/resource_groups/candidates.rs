//! Shared packing groups use the same recovered event kernel as routing trees.
use super::*;
use crate::master::LpMaster;
use crate::{
    graph::ResourceKind,
    master::branching::resource_intervals::{
        resource_branch_fraction, select_resource_threshold, suffix_upper, ResourceInterval,
    },
};

/// A proved shared packing group and its current lower-side fractional mass.
#[derive(Clone, Debug)]
pub struct ResourceGroupCandidate {
    pub packing_row: usize,
    pub resource: usize,
    pub threshold: f64,
    pub value: f64,
    pub members: Vec<ResourceGroupMember>,
}

impl ResourceGroupCandidate {
    pub fn apply<const D: usize>(
        &self,
        node: &NodeSnapshot<D>,
        upper: bool,
    ) -> Result<NodeSnapshot<D>> {
        node.with_resource_group_branch(
            self.packing_row,
            self.resource,
            self.threshold,
            upper,
            &self.members,
        )
    }
}

impl<const D: usize> NodeSnapshot<D> {
    /// Generate disposable-resource disjunctions from the current original-
    /// feasible LP. Supports span physical arcs of every participating network;
    /// original unit packing rows prove their integer disjunction validity.
    pub fn resource_group_candidates(&self, tolerance: f64) -> Result<Vec<ResourceGroupCandidate>> {
        if !tolerance.is_finite() || tolerance <= 0. || tolerance >= 0.5 {
            return Err(invalid(
                "resource candidate tolerance must be between zero and one half",
            ));
        }
        if !self.master.solved()
            || self
                .warm
                .artificials
                .iter()
                .any(|&c| self.master.values()[c].abs() > tolerance)
        {
            return Err(invalid(
                "resource candidates require an original-feasible solved node",
            ));
        }
        let valid_rows: Vec<_> = self
            .state
            .shared_rows
            .iter()
            .enumerate()
            .map(|(r, row)| {
                matches!(row.sense, RowSense::Equal | RowSense::LessEqual)
                    && row.rhs <= 1.
                    && nonnegative_packing(&self.state, r)
            })
            .collect();
        let mut supports = BTreeMap::<(usize, usize), Vec<ResourceGroupMember>>::new();
        let mut observations = BTreeMap::<(usize, usize), Vec<ResourceInterval>>::new();
        let mut invalid = HashSet::new();
        let mut uppers = BTreeMap::<(usize, usize), f64>::new();
        for oracle in &self.state.oracles {
            let graph = domain_graph(oracle)?;
            let positions: HashMap<_, _> = graph
                .arcs
                .iter()
                .enumerate()
                .map(|(i, a)| (a.id, i))
                .collect();
            let mut rows_by_arc = HashMap::<i32, Vec<usize>>::new();
            for arc in &graph.arcs {
                for &(row, a) in &oracle.network.arc_entries[oracle.index[&arc.id]] {
                    if a >= 1. && valid_rows[row] {
                        let rows = rows_by_arc.entry(arc.id).or_default();
                        if !rows.contains(&row) {
                            rows.push(row);
                        }
                    }
                }
            }
            let routes: Vec<_> = self
                .warm
                .columns
                .iter()
                .enumerate()
                .filter_map(|(c, m)| {
                    let tag = m.as_ref()?.tag;
                    let weight = self.master.values()[c];
                    if tag < 0 || weight <= 0. {
                        return None;
                    }
                    let route = &self.state.routes[tag as usize];
                    (route.network == oracle.network.id).then_some((&route.route, weight, false))
                })
                .chain(
                    self.state
                        .fixed_routes
                        .iter()
                        .filter(|f| f.route.network == oracle.network.id && f.quantity > 0.)
                        .map(|f| (&f.route.route, f.quantity, true)),
                )
                .collect();
            for (resource, &kind) in graph.resources.iter().enumerate() {
                if kind != ResourceKind::Disposable {
                    continue;
                }
                // The recovered interval separator assumes cumulative
                // nonnegative consumption on the branched resource. Signed
                // secondary disposable axes remain valid pricing resources,
                // but are not advertised as this disjunction.
                if graph.arcs.iter().any(|arc| arc.consumption[resource] < 0.) {
                    continue;
                }
                // Both children must be supported before advertising the split.
                if self.state.universe.is_none()
                    && oracle.knapsack_rows.iter().any(|&(i, _)| {
                        oracle.solver.cuts.knapsack_cuts[i].graph_resources[&oracle.network.id].0
                            == resource
                    })
                {
                    continue;
                }
                let resource_upper = graph
                    .vertices
                    .iter()
                    .map(|v| v.bounds[resource].upper)
                    .fold(f64::NEG_INFINITY, f64::max)
                    + 2e-6;
                let mut arcs_by_row = BTreeMap::<usize, Vec<i32>>::new();
                for (&arc, rows) in &rows_by_arc {
                    for &row in rows {
                        arcs_by_row.entry(row).or_default().push(arc);
                    }
                }
                for (row, mut arcs) in arcs_by_row {
                    arcs.sort_unstable();
                    supports
                        .entry((row, resource))
                        .or_default()
                        .push(ResourceGroupMember {
                            network: oracle.network.id,
                            arcs,
                        });
                    uppers
                        .entry((row, resource))
                        .and_modify(|b| *b = b.max(resource_upper))
                        .or_insert(resource_upper);
                }
                for &(route, weight, fixed) in &routes {
                    let suffix = suffix_upper(
                        &graph,
                        route,
                        resource,
                        |id| positions.get(&id).copied(),
                        |arc| {
                            let mut w = graph.vertices[arc.head].bounds[resource];
                            if let Some(a) = arc.bounds[resource] {
                                w.lower = w.lower.max(a.lower);
                                w.upper = w.upper.min(a.upper);
                            }
                            (w.lower <= w.upper).then_some(w)
                        },
                    );
                    // Reverse occurrence order matches the recovered routing generator.
                    for (i, id) in route.arcs.iter().enumerate().rev() {
                        let Some(rows) = rows_by_arc.get(id) else {
                            continue;
                        };
                        let lower = route
                            .resource_profile
                            .get(i + 1)
                            .and_then(|p| p.get(resource))
                            .copied();
                        let upper = if fixed {
                            lower
                        } else {
                            suffix.as_ref().map(|s| s[i])
                        };
                        for &row in rows {
                            match (lower, upper) {
                                (Some(lower), Some(upper))
                                    if lower.is_finite()
                                        && upper.is_finite()
                                        && lower <= upper
                                        && weight.is_finite() =>
                                {
                                    observations.entry((row, resource)).or_default().push(
                                        ResourceInterval {
                                            lower,
                                            upper,
                                            weight,
                                        },
                                    );
                                }
                                _ => {
                                    invalid.insert((row, resource));
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut result = vec![];
        for ((packing_row, resource), members) in supports {
            if invalid.contains(&(packing_row, resource)) {
                continue;
            }
            let intervals = observations
                .remove(&(packing_row, resource))
                .unwrap_or_default();
            if intervals.iter().map(|i| i.weight).sum::<f64>() > 1. + tolerance {
                continue;
            }
            let Ok(Some(threshold)) =
                select_resource_threshold(&intervals, uppers[&(packing_row, resource)])
            else {
                continue;
            };
            let boundary = threshold - AccumulatedResourceBranch::THRESHOLD_OFFSET;
            if members.iter().all(|member| {
                self.state
                    .oracles
                    .iter()
                    .find(|o| o.network.id == member.network)
                    .unwrap()
                    .resource_groups
                    .iter()
                    .any(|(_, g)| {
                        g.packing_row() == packing_row
                            && g.resource() == resource
                            && g.boundary() == boundary
                            && g.arcs() == member.arcs
                    })
            }) {
                continue;
            }
            let value = resource_branch_fraction(&intervals, threshold);
            if value > tolerance && value < 1. - tolerance {
                result.push(ResourceGroupCandidate {
                    packing_row,
                    resource,
                    threshold,
                    value,
                    members,
                });
            }
        }
        Ok(result)
    }
}
