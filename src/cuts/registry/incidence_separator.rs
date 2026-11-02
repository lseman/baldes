//! Stateful rank-one separation on explicit physical set incidence.
use super::*;
use separation::SeparatedCut;

/// Stateful separator prepared from explicit per-graph arc and vertex sets.
///
/// The graphs are owned snapshots. Callers provide the same phase and active
/// cuts they would provide to [`Separation::separate`], and retain the returned
/// logical cuts when they are needed as active cuts in a later round.
#[derive(Clone, Debug)]
pub struct IncidenceCutSeparator {
    graphs: Vec<GraphData>,
    state: Separation,
}

impl IncidenceCutSeparator {
    /// Prepare a separator using the supplied physical set incidence.
    pub fn new(params: Parameters, graphs: &[&GraphData]) -> Result<Self> {
        if graphs.is_empty()
            || !(1..=5).contains(&params.max_num_rows)
            || params.max_num_per_round <= 0
            || !(0..=3).contains(&params.memory_type)
            || !(0..=2).contains(&params.arc_memory_type)
            || !(0..=2).contains(&params.type_to_separate)
            || params.ls_num_iterations < 0
            || params.neighbourhood_size <= 0
            || params.max_num_triplets < 0
            || !params.violation_tolerance.is_finite()
            || params.violation_tolerance <= 0.
        {
            return Err(invalid("incidence separator parameters"));
        }
        let mut ids = BTreeSet::new();
        let mut incidences = Vec::with_capacity(graphs.len());
        for graph in graphs {
            if graph.id < 0
                || graph.id == i32::MAX
                || !ids.insert(graph.id)
                || graph.arcs.iter().any(|arc| arc.id == i32::MAX)
                || graph.vertices.iter().any(|vertex| vertex.id == i32::MAX)
                || graph
                    .vertices
                    .iter()
                    .any(|vertex| vertex.packing_sets.len() > 1 || vertex.covering_sets.len() > 1)
                || graph
                    .arcs
                    .iter()
                    .any(|arc| arc.packing_sets.len() > 1 || arc.covering_sets.len() > 1)
            {
                return Err(invalid("incidence separator graph"));
            }
            let incidence = separation::prepare_incidence(graph)
                .ok_or_else(|| invalid("incidence separator graph"))?;
            incidences.push((graph.id, incidence));
        }
        let mut state = Separation::new(params);
        if !state.prepare(graphs, incidences) {
            return Err(invalid("incidence separator preparation"));
        }
        Ok(Self {
            graphs: graphs.iter().map(|g| (*g).clone()).collect(),
            state,
        })
    }

    /// Current rank reached by the separator's most recent round.
    pub fn current_phase(&self) -> i32 {
        self.state.current_max_rank
    }

    /// Separate physical columns and register all resulting cuts atomically.
    ///
    /// Returns each new registry index and its logical separator descriptor.
    /// The input's active-cut list is explicit, preserving native memory
    /// inheritance without guessing which registry rows describe those sets.
    pub fn separate_and_register(
        &mut self,
        registry: &mut CutRegistry,
        input: &Input<'_>,
    ) -> Result<Vec<(usize, SeparatedCut)>> {
        if input.solutions_are_set_sequences
            || input.solutions.len() != input.values.len()
            || !matches!(input.cut_memory_type, 0..=2)
            || (input.cut_memory_type == 1 && self.state.params.memory_type > 1)
            || input.separation_phase < 1
            || input.separation_phase > self.state.params.max_num_rows
            || input
                .values
                .iter()
                .any(|&value| !value.is_finite() || value < 0.)
        {
            return Err(invalid("incidence separator input"));
        }
        for solution in &input.solutions {
            let graph = usize::try_from(solution.graph_id)
                .map_err(|_| invalid("incidence separator column graph"))?;
            let incidence = self
                .state
                .preprocessed
                .get(graph)
                .and_then(Option::as_ref)
                .ok_or_else(|| invalid("incidence separator column graph"))?;
            if solution.arc_ids.iter().any(|&arc| {
                usize::try_from(arc)
                    .ok()
                    .and_then(|id| incidence.arc_tail.get(id))
                    .is_none_or(|&tail| tail < 0)
            }) {
                return Err(invalid("incidence separator column arc"));
            }
        }
        let graph_refs: Vec<_> = self.graphs.iter().collect();
        let mut validation = CutRegistry::default();
        for active in &input.active_cuts {
            validation.register_separated_cut(&graph_refs, active)?;
        }
        let mut state = self.state.clone();
        state.vertex_cut_load.clear();
        if state.params.max_per_vertex > 0 {
            for (index, cut) in registry.cuts.iter().enumerate() {
                if registry.retired.contains(&index) {
                    continue;
                }
                for graph in &self.graphs {
                    let mut touched = BTreeSet::new();
                    for vertex in &graph.vertices {
                        if cut.pricing_weight_for_graph(graph.id, vertex.id as usize) != 0 {
                            touched.insert(vertex.id);
                        }
                    }
                    match cut.memory_kind {
                        MemoryKind::Vertex => {
                            for (vertex, &present) in cut
                                .memory_for_graph_ref(graph.id)
                                .unwrap_or(&[])
                                .iter()
                                .enumerate()
                            {
                                if present && vertex < graph.vertices.len() {
                                    touched.insert(vertex as i32);
                                }
                            }
                        }
                        MemoryKind::Arc => {
                            if let Some(memory) = cut.arc_memory_for_graph_ref(graph.id) {
                                for arc in &graph.arcs {
                                    if memory.contains(&arc.id) {
                                        touched.insert(arc.tail);
                                        touched.insert(arc.head);
                                    }
                                }
                            }
                        }
                    }
                    for vertex in touched {
                        *state.vertex_cut_load.entry((graph.id, vertex)).or_default() += 1;
                    }
                }
            }
        }
        let separated = state
            .separate(input)
            .ok_or_else(|| invalid("incidence separation failed"))?;
        let start = registry.cuts.len();
        let mut staged = CutRegistry::default();
        let mut result = Vec::with_capacity(separated.len());
        for cut in separated {
            let local = staged.register_separated_cut(&graph_refs, &cut)?;
            let index = start
                .checked_add(local)
                .ok_or_else(|| invalid("incidence cut index"))?;
            staged.cuts[local].id =
                i32::try_from(index).map_err(|_| invalid("incidence cut ID"))?;
            result.push((index, cut));
        }
        registry.cuts.extend(staged.cuts);
        self.state = state;
        Ok(result)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/cuts/registry/incidence_separator_tests.rs"]
mod tests;
