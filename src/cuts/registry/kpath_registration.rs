use super::*;

impl CutRegistry {
    /// Registers a strong k-path cut on `nodes` with the given `rhs` and `dual`.
    pub fn add_strong_k_path_cut(
        &mut self,
        vertices: usize,
        nodes: &[usize],
        rhs: i32,
        dual: f64,
    ) -> Result<usize> {
        if nodes.is_empty() || rhs <= 0 || !dual.is_finite() {
            return Err(invalid("strong k-path cut parameters"));
        }
        let mut weights = vec![0i32; vertices];
        let mut ordered = Vec::with_capacity(nodes.len());
        for &v in nodes {
            if v >= vertices {
                return Err(invalid("strong k-path cut vertex out of range"));
            }
            weights[v] = 1;
            ordered.push((v, 1));
        }
        let id = self.cuts.len() as i32;
        self.cuts.push(RankOneCut {
            weights,
            graph_support: BTreeMap::new(),
            graph_pricing_support: BTreeMap::new(),
            native_kpath_support: None,
            memory: vec![true; vertices],
            graph_dependent_memory: false,
            graph_memories: BTreeMap::new(),
            memory_kind: MemoryKind::Vertex,
            arc_memory: BTreeSet::new(),
            graph_arc_memories: BTreeMap::new(),
            arc_weights: BTreeMap::new(),
            visit_once: true,
            native_type2: false,
            denominator: 2,
            dual,
            class: CutClass::Covering,
            rhs,
            kind: -1,
            id,
            ordered_support: Some(ordered),
        });
        Ok(self.cuts.len() - 1)
    }

    /// Preserve createCut's graph-specific vertex memory.
    pub fn register_strong_k_path_cut(
        &mut self,
        vertices: usize,
        graph: i32,
        cut: &kpath::StrongKPathCut,
        dual: f64,
    ) -> Result<usize> {
        if cut.memory.iter().any(|&v| v >= vertices) {
            return Err(invalid("strong k-path memory vertex out of range"));
        }
        let index = self.add_strong_k_path_cut(vertices, &cut.customers, cut.rhs, dual)?;
        let registered = &mut self.cuts[index];
        registered.id = cut.id;
        registered.memory.fill(false);
        for &v in &cut.memory {
            registered.memory[v] = true;
        }
        registered.graph_dependent_memory = true;
        registered
            .graph_memories
            .insert(graph, registered.memory.clone());
        Ok(index)
    }

    /// Adapt original set support and graph-specific physical vertex memories
    /// to solver inputs. Every graph keeps its own support, including aliases
    /// where several physical vertices belong to one original set.
    pub fn register_native_k_path_cut(
        &mut self,
        graphs: &[kpath::KPathGraph],
        use_packing_sets: bool,
        cut: &kpath::NativeKPathCut,
        dual: f64,
    ) -> Result<usize> {
        if !dual.is_finite() || cut.arc_memory {
            return Err(invalid(
                "native k-path pricing requires finite dual and vertex memory",
            ));
        }
        let mut graph_support = BTreeMap::new();
        let mut graph_pricing_support = BTreeMap::new();
        let mut graph_memories = BTreeMap::new();
        let mut max_vertices = 0;
        for input in graphs {
            let graph = &input.graph;
            if graph.id < 0 || graph_support.contains_key(&graph.id) {
                return Err(invalid("native k-path registration graph ID"));
            }
            let mut support = vec![];
            let mut pricing_support = vec![];
            let mut vertices = 0;
            for vertex in &graph.vertices {
                let v =
                    usize::try_from(vertex.id).map_err(|_| invalid("negative k-path vertex"))?;
                vertices = vertices.max(v + 1);
                if vertex
                    .covering_sets
                    .iter()
                    .any(|&e| e >= 0 && cut.set_ids.contains(&(e as usize)))
                {
                    pricing_support.push((v, 1));
                }
                let sets = if use_packing_sets {
                    &vertex.packing_sets
                } else {
                    &vertex.covering_sets
                };
                if sets
                    .first()
                    .is_some_and(|&e| e >= 0 && cut.set_ids.contains(&(e as usize)))
                {
                    support.push((v, 1));
                }
            }
            support.sort_unstable_by_key(|&(v, _)| v);
            support.dedup();
            graph_support.insert(graph.id, support);
            pricing_support.sort_unstable_by_key(|&(v, _)| v);
            pricing_support.dedup();
            graph_pricing_support.insert(graph.id, pricing_support);
            max_vertices = max_vertices.max(vertices);
            if let Some(memory) = cut.memory.get(&graph.id) {
                let mut selected = vec![false; vertices];
                for &v in memory {
                    if v >= vertices {
                        return Err(invalid("native k-path memory vertex out of range"));
                    }
                    selected[v] = true;
                }
                graph_memories.insert(graph.id, selected);
            }
        }
        self.cuts.push(RankOneCut {
            weights: vec![0; max_vertices],
            graph_support,
            graph_pricing_support,
            native_kpath_support: Some(cut.set_ids.clone()),
            memory: vec![false; max_vertices],
            graph_dependent_memory: true,
            graph_memories,
            memory_kind: MemoryKind::Vertex,
            arc_memory: BTreeSet::new(),
            graph_arc_memories: BTreeMap::new(),
            arc_weights: BTreeMap::new(),
            visit_once: true,
            native_type2: false,
            denominator: 2,
            dual,
            class: CutClass::Covering,
            rhs: cut.rhs,
            kind: -1,
            id: cut.id,
            ordered_support: None,
        });
        Ok(self.cuts.len() - 1)
    }
}
