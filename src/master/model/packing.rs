//! Explicit native packing labels and their assignment-branch generator.
use super::model_strong::{Proposal, Restriction};
use super::*;

/// Packing setters for a pricing configuration. The last label for a vertex
/// wins; these labels do not set elementarity or shared-row membership.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkPackingAssignment {
    pub network: i32,
    pub vertices: Vec<(usize, usize)>,
    // Native ties use configuration identity. Keep that identity through tree
    // option clones without exposing process addresses in serialized models.
    #[serde(skip)]
    identity: Arc<()>,
}

impl NetworkPackingAssignment {
    pub fn new(network: i32, vertices: Vec<(usize, usize)>) -> Self {
        Self {
            network,
            vertices,
            identity: Arc::new(()),
        }
    }
}

/// Register native packing-assignment branching. This selects Phased
/// branching and replaces inferred shared-row assignment candidates. Other
/// registered families retain their configured priorities.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackingBranching {
    pub priority: f64,
    pub networks: Vec<NetworkPackingAssignment>,
    #[serde(skip)]
    prepared: Vec<(usize, usize, MappedBranchVariable)>,
}

impl PackingBranching {
    pub fn new(priority: f64, networks: Vec<NetworkPackingAssignment>) -> Self {
        Self {
            priority,
            networks,
            prepared: vec![],
        }
    }

    pub(super) fn prepare<'a, const D: usize>(
        &mut self,
        graphs: impl Iterator<Item = (i32, &'a Graph<D>)>,
        explicit: &[MappedBranchVariable],
    ) -> Result<()> {
        if !self.priority.is_finite() {
            return Err(invalid("packing branching requires finite priority"));
        }
        let graphs: HashMap<_, _> = graphs.collect();
        let mut seen = HashSet::new();
        let mut prepared = vec![];
        for registration in &self.networks {
            let graph = graphs
                .get(&registration.network)
                .ok_or_else(|| invalid("unknown packing network"))?;
            if !seen.insert(registration.network) {
                return Err(invalid("duplicate packing network"));
            }
            if graph.source >= graph.vertices.len()
                || graph.sink >= graph.vertices.len()
                || graph
                    .arcs
                    .iter()
                    .any(|a| a.tail >= graph.vertices.len() || a.head >= graph.vertices.len())
            {
                return Err(invalid("invalid packing graph endpoints"));
            }
            let mut labels = vec![None; graph.vertices.len()];
            for &(vertex, group) in &registration.vertices {
                if vertex >= labels.len() || group > i32::MAX as usize {
                    return Err(invalid(
                        "unknown packing vertex or out-of-range packing index",
                    ));
                }
                if vertex == graph.source || vertex == graph.sink {
                    return Err(invalid("source and sink cannot belong to a packing set"));
                }
                labels[vertex] = Some(group);
            }
            let mut groups = BTreeMap::<usize, BTreeMap<i32, f64>>::new();
            for group in labels.iter().flatten() {
                groups.entry(*group).or_default();
            }
            for arc in &graph.arcs {
                for group in [labels[arc.tail], labels[arc.head]].into_iter().flatten() {
                    *groups.entry(group).or_default().entry(arc.id).or_default() += 0.5;
                }
            }
            for (group, arcs) in groups {
                let name = format!("ESA_{}_{}", registration.network, group);
                if explicit.iter().any(|v| v.name == name) {
                    return Err(invalid(
                        "explicit mapping conflicts with packing assignment name",
                    ));
                }
                if arcs.is_empty() {
                    continue;
                }
                let mapping = MappedBranchVariable {
                    name,
                    priority: self.priority,
                    networks: vec![MappedNetworkVariable {
                        network: registration.network,
                        arcs: arcs.into_iter().collect(),
                        per_route: 0.,
                    }],
                    masters: vec![],
                };
                mapping.validate(std::iter::once((registration.network, *graph)), &[])?;
                prepared.push((Arc::as_ptr(&registration.identity) as usize, group, mapping));
            }
        }
        self.prepared = prepared;
        Ok(())
    }

    // A total ordering for mixed registries: packing identity breaks packing
    // ties, while existing nonpacking names retain their relative ordering.
    pub(super) fn tie_key(&self, name: &str) -> (u8, usize, usize) {
        self.prepared
            .iter()
            .find(|(_, _, v)| v.name == name)
            .map_or((1, 0, 0), |(identity, group, _)| (0, *identity, *group))
    }

    pub(super) fn proposals<const D: usize>(
        &self,
        output: &Output<D>,
        tolerance: f64,
    ) -> Vec<Proposal> {
        let mut candidates: Vec<_> = self
            .prepared
            .iter()
            .filter_map(|(identity, group, mapping)| {
                let value = mapping.value(output)?;
                (value > tolerance && value < 1. - tolerance)
                    .then_some((*identity, *group, mapping, value))
            })
            .collect();
        candidates.sort_by(|a, b| {
            (a.3 - 0.5)
                .abs()
                .total_cmp(&(b.3 - 0.5).abs())
                .then(a.0.cmp(&b.0))
                .then(a.1.cmp(&b.1))
        });
        let mut groups = HashSet::new();
        candidates
            .into_iter()
            .filter(|(_, group, _, _)| groups.insert(*group))
            .map(|(_, _, mapping, value)| Proposal {
                name: mapping.name.clone(),
                history_key: format!("mapped:{}", mapping.name),
                priority: mapping.priority,
                value,
                restriction: Restriction::Mapped(mapping.clone()),
                history_eligible: true,
                fresh: true,
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/packing_tests.rs"]
mod tests;
