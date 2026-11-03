//! Install separated rank-one cuts from explicit graph incidence.
use super::*;
use separation::SeparatedCut;

fn vertex_sets(vertex: &VertexData, kind: i32) -> &[i32] {
    if kind == 0 {
        &vertex.packing_sets
    } else {
        &vertex.covering_sets
    }
}

fn arc_sets(arc: &ArcData, kind: i32) -> &[i32] {
    if kind == 0 {
        &arc.packing_sets
    } else {
        &arc.covering_sets
    }
}

impl CutRegistry {
    /// Register a separated rank-one cut on its physical graphs.
    ///
    /// Packing/covering membership may be attached to an arc or a vertex.
    /// Each arc and vertex may carry several distinct memberships of each kind.
    /// Graph and arc IDs are local to each graph. Graphs missing from the cut's
    /// memory contribute zero to ordinary columns but remain available for
    /// enumerated columns.
    pub fn register_separated_cut(
        &mut self,
        graphs: &[&GraphData],
        separated: &SeparatedCut,
    ) -> Result<usize> {
        let denominator = separated
            .denominator
            .checked_abs()
            .filter(|&d| (1..8).contains(&d))
            .ok_or_else(|| invalid("separated cut denominator"))?;
        let rows =
            usize::try_from(separated.num_rows).map_err(|_| invalid("separated cut row count"))?;
        if rows == 0
            || separated.set_ids.len() != rows
            || separated.numerators.len() != rows
            || !matches!(separated.kind, 0..=2)
            || !matches!(separated.memory_type, 0..=1)
            || !separated.rhs.is_finite()
            || separated.rhs.fract() != 0.
            || separated.rhs < i32::MIN as f64
            || separated.rhs > i32::MAX as f64
        {
            return Err(invalid("separated cut descriptor"));
        }
        let sign = if separated.kind == 1 { -1 } else { 1 };
        if separated.denominator != sign * denominator {
            return Err(invalid("separated cut denominator sign"));
        }
        let mut weights = BTreeMap::new();
        for (&set, &numerator) in separated.set_ids.iter().zip(&separated.numerators) {
            let Some(weight) = numerator.checked_mul(sign) else {
                return Err(invalid("separated cut numerator"));
            };
            if set < 0
                || weight <= 0
                || weight >= denominator
                || weights.insert(set, weight).is_some()
            {
                return Err(invalid("separated cut row weights"));
            }
        }
        if graphs.is_empty() {
            return Err(invalid("separated cut requires graphs"));
        }
        let mut graph_support = BTreeMap::new();
        let mut graph_arc_weights = BTreeMap::new();
        let mut graph_memories = BTreeMap::new();
        let mut graph_arc_memories = BTreeMap::new();
        let mut max_vertices = 0;
        for graph in graphs {
            if graph.id < 0 || graph_support.contains_key(&graph.id) {
                return Err(invalid("separated cut graph IDs"));
            }
            let n = graph.vertices.len();
            if n == 0
                || n > 1024
                || graph.source < 0
                || graph.sink < 0
                || graph.source as usize >= n
                || graph.sink as usize >= n
            {
                return Err(invalid("separated cut graph vertices"));
            }
            max_vertices = max_vertices.max(n);
            let mut vertices = vec![None; n];
            for vertex in &graph.vertices {
                let index =
                    usize::try_from(vertex.id).map_err(|_| invalid("separated cut vertex ID"))?;
                if index >= n || vertices[index].replace(vertex).is_some() {
                    return Err(invalid("separated cut vertex ID"));
                }
            }
            let set_count = if separated.kind == 0 {
                graph.packing_set_count
            } else {
                graph.covering_set_count
            };
            if !(0..1024).contains(&set_count) {
                return Err(invalid("separated cut graph set count"));
            }
            let valid_sets = |sets: &[i32]| sets.iter().all(|&id| (0..set_count).contains(&id));
            let mut support = vec![];
            for vertex in &graph.vertices {
                let sets = vertex_sets(vertex, separated.kind);
                if !valid_sets(sets) {
                    return Err(invalid("separated cut vertex set incidence"));
                }
                let weight = sets
                    .iter()
                    .enumerate()
                    .filter(|(index, set)| !sets[..*index].contains(set))
                    .map(|(_, set)| set)
                    .filter_map(|set| weights.get(set))
                    .try_fold(0i32, |total, &weight| total.checked_add(weight))
                    .ok_or_else(|| invalid("separated cut vertex weight overflow"))?;
                if weight != 0 {
                    support.push((vertex.id as usize, weight));
                }
            }
            support.sort_unstable();
            let mut arc_weights = BTreeMap::new();
            let mut arc_ids = BTreeSet::new();
            for arc in &graph.arcs {
                let head =
                    usize::try_from(arc.head).map_err(|_| invalid("separated cut arc endpoint"))?;
                if arc.id < 0
                    || arc.tail < 0
                    || arc.tail as usize >= n
                    || head >= n
                    || !arc_ids.insert(arc.id)
                {
                    return Err(invalid("separated cut physical arc"));
                }
                let sets = arc_sets(arc, separated.kind);
                if !valid_sets(sets) {
                    return Err(invalid("separated cut arc set incidence"));
                }
                // The separator uses the union of arc and head sets. Avoid
                // counting the same set twice when translating that union.
                let head_sets = vertex_sets(vertices[head].unwrap(), separated.kind);
                let weight = sets
                    .iter()
                    .enumerate()
                    .filter(|(index, set)| !sets[..*index].contains(set))
                    .map(|(_, set)| set)
                    .filter(|set| !head_sets.contains(set))
                    .filter_map(|set| weights.get(set))
                    .try_fold(0i32, |total, &weight| total.checked_add(weight))
                    .ok_or_else(|| invalid("separated cut arc weight overflow"))?;
                if weight != 0 {
                    arc_weights.insert(arc.id, weight);
                }
            }
            if let Some(memory) = separated.memory.get(&graph.id) {
                if separated.memory_type == 1 {
                    if !memory.is_subset(&arc_ids) {
                        return Err(invalid("separated cut arc memory"));
                    }
                    graph_arc_memories.insert(graph.id, memory.clone());
                } else {
                    let mut bits = vec![false; n];
                    for &vertex in memory {
                        let vertex = usize::try_from(vertex)
                            .map_err(|_| invalid("separated cut vertex memory"))?;
                        if vertex >= n {
                            return Err(invalid("separated cut vertex memory"));
                        }
                        bits[vertex] = true;
                    }
                    graph_memories.insert(graph.id, bits);
                }
            }
            graph_support.insert(graph.id, support);
            graph_arc_weights.insert(graph.id, arc_weights);
        }
        if separated
            .memory
            .keys()
            .any(|id| !graph_support.contains_key(id))
        {
            return Err(invalid("separated cut memory graph"));
        }
        let id = i32::try_from(self.cuts.len()).map_err(|_| invalid("separated cut ID"))?;
        let index = self.cuts.len();
        self.cuts.push(RankOneCut {
            weights: vec![0; max_vertices],
            graph_support,
            graph_pricing_support: BTreeMap::new(),
            general_kpath_support: None,
            memory: vec![],
            graph_dependent_memory: true,
            graph_memories,
            memory_kind: if separated.memory_type == 1 {
                MemoryKind::Arc
            } else {
                MemoryKind::Vertex
            },
            arc_memory: BTreeSet::new(),
            graph_arc_memories,
            arc_weights: BTreeMap::new(),
            graph_arc_weights,
            visit_once: false,
            type2: separated.kind == 2,
            denominator,
            dual: 0.,
            class: if separated.kind == 1 {
                CutClass::Covering
            } else {
                CutClass::Packing
            },
            rhs: separated.rhs as i32,
            kind: separated.class,
            id,
            ordered_support: None,
        });
        Ok(index)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/cuts/registry/incidence_registration_tests.rs"]
mod tests;
