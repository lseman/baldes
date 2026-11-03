//! Packing-set Ryan–Foster branching on network paths.
//!
//! A branch on packing sets `a` and `b` is a master row `sum <= 0` whose
//! column coefficient is one exactly when the column's path violates the
//! branch (together: visits exactly one of the sets; apart: visits both),
//! and the same restriction is installed in every member network's pricing.
//! Columns are never capped. Candidates are pairs whose LP co-occurrence lies
//! strictly between 1e-6 and 0.999999, ranked by distance to one half with
//! ties in pair order, and the together child is the upper child.
use super::model_strong::{Proposal, Restriction};
use super::*;
use crate::pricing::restrictions::ArcSetConstraint;
use crate::pricing::restrictions::RyanFosterConstraint;
use std::collections::BTreeSet;

/// The two packing-set vertices of one network in a Ryan–Foster branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RyanFosterMember {
    pub network: i32,
    pub first: usize,
    pub second: usize,
}

#[derive(Clone, Debug)]
pub(super) struct ArcSetMember {
    network: i32,
    first: Vec<i32>,
    second: Vec<i32>,
}

/// Register packing-set Ryan–Foster branching. `sets` labels vertices with
/// packing-set indices per network; when empty, the packing-assignment
/// registration is used. When neither is present, explicit rank-one
/// incidence supplies the sets, or a single network's customer vertices are
/// used in vertex order.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RyanFosterBranching {
    pub priority: f64,
    #[serde(default)]
    pub sets: Vec<NetworkPackingAssignment>,
    #[serde(skip)]
    prepared: Prepared,
}

#[derive(Clone, Debug, Default)]
struct Prepared {
    /// Per network: vertex -> packing set.
    labels: BTreeMap<i32, Vec<Option<usize>>>,
    /// Per network: set -> its single vertex, or None if it has several.
    vertices: BTreeMap<i32, HashMap<usize, Option<usize>>>,
    arc_sets: Option<BTreeMap<i32, BTreeMap<i32, Vec<usize>>>>,
    width: usize,
}

impl RyanFosterBranching {
    pub fn new(priority: f64) -> Self {
        Self {
            priority,
            sets: vec![],
            prepared: Prepared::default(),
        }
    }

    pub(super) fn prepare<'a, const D: usize>(
        &mut self,
        graphs: impl Iterator<Item = (i32, &'a Graph<D>)>,
        packing: Option<&PackingBranching>,
        incidence: Option<&super::RankOneIncidenceConfig>,
    ) -> Result<()> {
        if !self.priority.is_finite() {
            return Err(invalid("Ryan–Foster branching requires finite priority"));
        }
        let graphs: Vec<_> = graphs.collect();
        if self.sets.is_empty() && packing.is_none_or(|p| p.networks.is_empty()) {
            if let Some(incidence) = incidence {
                let mut prepared = Prepared::default();
                let mut arc_sets = BTreeMap::new();
                if incidence.graphs.len() != graphs.len() {
                    return Err(invalid("Ryan–Foster incidence graph count"));
                }
                for &(network, physical) in &graphs {
                    let graph = incidence
                        .graphs
                        .iter()
                        .find(|g| g.id == network)
                        .ok_or_else(|| invalid("unknown Ryan–Foster incidence network"))?;
                    if graph.packing_set_count != incidence.rows.len() as i32 {
                        return Err(invalid("Ryan–Foster incidence set count"));
                    }
                    let data = crate::cuts::separation::prepare_incidence(graph)
                        .ok_or_else(|| invalid("invalid Ryan–Foster incidence"))?;
                    let mut arcs = BTreeMap::new();
                    for arc in &physical.arcs {
                        let index = usize::try_from(arc.id)
                            .map_err(|_| invalid("invalid Ryan–Foster incidence arc"))?;
                        if data.arc_tail.get(index) != Some(&(arc.tail as i32))
                            || data.arc_head.get(index) != Some(&(arc.head as i32))
                        {
                            return Err(invalid("Ryan–Foster incidence arc mismatch"));
                        }
                        arcs.insert(
                            arc.id,
                            data.arc_packing_sets[index]
                                .iter()
                                .map(|&s| s as usize)
                                .collect(),
                        );
                    }
                    arc_sets.insert(network, arcs);
                }
                prepared.width = incidence.rows.len().saturating_sub(1).to_string().len();
                prepared.arc_sets = Some(arc_sets);
                self.prepared = prepared;
                return Ok(());
            }
        }
        let registrations: Vec<(i32, Vec<(usize, usize)>)> = if !self.sets.is_empty() {
            self.sets
                .iter()
                .map(|s| (s.network, s.vertices.clone()))
                .collect()
        } else if let Some(packing) = packing.filter(|p| !p.networks.is_empty()) {
            packing
                .networks
                .iter()
                .map(|s| (s.network, s.vertices.clone()))
                .collect()
        } else {
            let [(network, graph)] = graphs[..] else {
                return Err(invalid(
                    "Ryan–Foster branching on several networks requires packing sets",
                ));
            };
            let vertices = (0..graph.vertices.len())
                .filter(|&v| v != graph.source && v != graph.sink)
                .enumerate()
                .map(|(set, v)| (v, set))
                .collect();
            vec![(network, vertices)]
        };
        let mut prepared = Prepared::default();
        let mut largest = 0;
        for (network, vertices) in registrations {
            let graph = graphs
                .iter()
                .find(|(id, _)| *id == network)
                .map(|(_, g)| *g)
                .ok_or_else(|| invalid("unknown Ryan–Foster network"))?;
            if prepared.labels.contains_key(&network) {
                return Err(invalid("duplicate Ryan–Foster network"));
            }
            let mut labels = vec![None; graph.vertices.len()];
            for &(vertex, set) in &vertices {
                if vertex >= labels.len() || vertex == graph.source || vertex == graph.sink {
                    return Err(invalid(
                        "Ryan–Foster packing sets require customer vertices",
                    ));
                }
                labels[vertex] = Some(set);
            }
            let mut singles = HashMap::new();
            for (vertex, set) in labels.iter().enumerate() {
                if let Some(set) = *set {
                    largest = largest.max(set);
                    singles
                        .entry(set)
                        .and_modify(|v: &mut Option<usize>| *v = None)
                        .or_insert(Some(vertex));
                }
            }
            prepared.labels.insert(network, labels);
            prepared.vertices.insert(network, singles);
        }
        prepared.width = largest.to_string().len();
        self.prepared = prepared;
        Ok(())
    }

    /// Network members of a pair, or None when a network holds only one of
    /// the sets or a set with several vertices.
    fn members(&self, a: usize, b: usize) -> Option<Vec<RyanFosterMember>> {
        let mut members = vec![];
        for (&network, singles) in &self.prepared.vertices {
            match (singles.get(&a), singles.get(&b)) {
                (None, None) => {}
                (Some(&Some(first)), Some(&Some(second))) => members.push(RyanFosterMember {
                    network,
                    first,
                    second,
                }),
                _ => return None,
            }
        }
        (!members.is_empty()).then_some(members)
    }

    pub(super) fn proposals<const D: usize>(&self, output: &Output<D>) -> Result<Vec<Proposal>> {
        let node = &output.snapshot;
        let mut together = BTreeMap::<(usize, usize), f64>::new();
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
            if quantity == 0. {
                continue;
            }
            let oracle = node
                .state
                .oracles
                .iter()
                .find(|o| o.network.id == route.network)
                .ok_or_else(|| invalid("unknown Ryan–Foster route network"))?;
            let sets: BTreeSet<usize> = if let Some(networks) = &self.prepared.arc_sets {
                let labels = networks
                    .get(&route.network)
                    .ok_or_else(|| invalid("unknown Ryan–Foster route network"))?;
                let mut sets = BTreeSet::new();
                for arc in &route.route.arcs {
                    sets.extend(
                        labels
                            .get(arc)
                            .ok_or_else(|| invalid("unknown Ryan–Foster route arc"))?
                            .iter()
                            .copied(),
                    );
                }
                sets
            } else {
                let Some(labels) = self.prepared.labels.get(&route.network) else {
                    continue;
                };
                path_vertices(oracle, &route.route)?
                    .filter_map(|v| labels.get(v).copied().flatten())
                    .collect()
            };
            let sets: Vec<_> = sets.into_iter().collect();
            for (i, &a) in sets.iter().enumerate() {
                for &b in &sets[i + 1..] {
                    *together.entry((a, b)).or_default() += quantity;
                }
            }
        }
        let width = self.prepared.width;
        let mut proposals = vec![];
        for ((a, b), value) in together {
            if !(value > 1e-6 && value < 0.999999) {
                continue;
            }
            let restriction = if let Some(networks) = &self.prepared.arc_sets {
                let members = networks
                    .iter()
                    .filter_map(|(&network, arcs)| {
                        let first: Vec<_> = arcs
                            .iter()
                            .filter_map(|(&arc, sets)| sets.contains(&a).then_some(arc))
                            .collect();
                        let second: Vec<_> = arcs
                            .iter()
                            .filter_map(|(&arc, sets)| sets.contains(&b).then_some(arc))
                            .collect();
                        (!first.is_empty() || !second.is_empty()).then_some(ArcSetMember {
                            network,
                            first,
                            second,
                        })
                    })
                    .collect();
                Restriction::ArcSetRyanFoster { members }
            } else {
                let Some(members) = self.members(a, b) else {
                    continue;
                };
                Restriction::RyanFoster { members }
            };
            // Zero padding keeps the name tie-break in native pair order.
            let name = format!("PS_{a:0width$}_{b:0width$}");
            proposals.push(Proposal {
                history_key: name.clone(),
                name,
                priority: self.priority,
                value,
                restriction,
                history_eligible: true,
                fresh: true,
                origin: Some(super::model_strong::ProducerFamily::PackingRyanFoster),
                component_admission: None,
            });
        }
        Ok(proposals)
    }
}

/// Source followed by every arc head: the vertices a path visits.
fn path_vertices<'a, const D: usize>(
    oracle: &'a Oracle<D>,
    route: &'a Route,
) -> Result<impl Iterator<Item = usize> + 'a> {
    let graph = &oracle.network.graph;
    let heads = route
        .arcs
        .iter()
        .map(|id| {
            oracle
                .index
                .get(id)
                .map(|&a| graph.arcs[a].head)
                .ok_or_else(|| invalid("unknown Ryan–Foster route arc"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(std::iter::once(graph.source).chain(heads))
}

pub(super) fn entries<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
) -> Result<Vec<(usize, f64)>> {
    if oracle.ryan_foster_rows.is_empty() && oracle.arc_set_branch_rows.is_empty() {
        return Ok(vec![]);
    }
    let vertices: Vec<_> = path_vertices(oracle, route)?.collect();
    let mut result: Vec<_> = oracle
        .ryan_foster_rows
        .iter()
        .filter(|(_, c)| !c.satisfied_by(vertices.iter().copied()))
        .map(|&(row, _)| (row, 1.))
        .collect();
    result.extend(
        oracle
            .arc_set_branch_rows
            .iter()
            .filter(|(_, c)| !c.satisfied_by(route.arcs.iter().copied()))
            .map(|(row, _)| (*row, 1.)),
    );
    Ok(result)
}

impl<const D: usize> NodeSnapshot<D> {
    pub(super) fn with_arc_set_branch(
        &self,
        members: &[ArcSetMember],
        together: bool,
    ) -> Result<Self> {
        if members.is_empty() {
            return Err(invalid("Ryan–Foster branch requires a network member"));
        }
        let shared = self.state.shared_rows.len();
        let mut node = self.with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &[],
        )?;
        for member in members {
            let oracle = node
                .state
                .oracles
                .iter_mut()
                .find(|o| o.network.id == member.network)
                .ok_or_else(|| invalid("unknown Ryan–Foster network"))?;
            oracle.arc_set_branch_rows.push((
                shared,
                ArcSetConstraint {
                    first: member.first.clone(),
                    second: member.second.clone(),
                    together,
                },
            ));
            oracle.solver.set_arc_set_branches(
                oracle
                    .arc_set_branch_rows
                    .iter()
                    .map(|(_, c)| c.clone())
                    .collect(),
            )?;
        }
        node = super::nonlinear::refresh_row(node, shared)?;
        node.state.correction = None;
        node.state.gradient = None;
        node.warm.stabilization = None;
        Ok(node)
    }
    /// Branch on whether two packing sets are served together by the same
    /// path (`together`) or never by the same path. Existing columns keep
    /// their bounds; violating columns get coefficient one in a new `<= 0`
    /// row, and pricing of every member network excludes violating paths.
    /// The parent is unchanged.
    pub fn with_ryan_foster(&self, members: &[RyanFosterMember], together: bool) -> Result<Self> {
        if members.is_empty() {
            return Err(invalid("Ryan–Foster branch requires a network member"));
        }
        let mut seen = HashSet::new();
        for member in members {
            if !seen.insert(member.network)
                || !self
                    .state
                    .oracles
                    .iter()
                    .any(|o| o.network.id == member.network)
            {
                return Err(invalid("unknown or duplicate Ryan–Foster network"));
            }
        }
        let shared = self.state.shared_rows.len();
        let mut node = self.with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &[],
        )?;
        for member in members {
            let oracle = node
                .state
                .oracles
                .iter_mut()
                .find(|o| o.network.id == member.network)
                .unwrap();
            oracle.ryan_foster_rows.push((
                shared,
                RyanFosterConstraint {
                    first: member.first,
                    second: member.second,
                    together,
                },
            ));
            let constraints = oracle.ryan_foster_rows.iter().map(|(_, c)| *c).collect();
            oracle.solver.set_ryan_foster(constraints)?;
        }
        node = super::nonlinear::refresh_row(node, shared)?;
        node.state.correction = None;
        node.state.gradient = None;
        node.warm.stabilization = None;
        Ok(node)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/ryan_foster_tests.rs"]
mod tests;
