use super::*;

impl CutRegistry {
    /// Separates with the current parameters and registers every emitted cut.
    pub fn separate<const D: usize>(
        &mut self,
        g: &Graph<D>,
        customers: &[usize],
        columns: &[FractionalColumn],
    ) -> Result<usize> {
        let p = &self.parameters;
        let vertices = g.vertices.len();
        if p.max_rank < 1
            || p.max_rank > 8
            || p.max_cuts < 1
            || customers.len() >= 1024
            || !(p.violation_epsilon > 0.)
        {
            return Err(invalid("native rank-one separation parameters"));
        }
        let mut row_of = vec![-1i32; vertices];
        for (row, &v) in customers.iter().enumerate() {
            if v >= vertices || row_of[v] >= 0 {
                return Err(invalid("native cut customer IDs"));
            }
            row_of[v] = row as i32;
        }
        let mut graph_ids = BTreeSet::from([p.graph_id]);
        for c in columns {
            graph_ids.insert(if c.graph_id < 0 {
                p.graph_id
            } else {
                c.graph_id
            });
        }
        let mut graphs = vec![];
        let mut incidences = vec![];
        for &id in &graph_ids {
            if id < 0 {
                return Err(invalid("negative cut graph ID"));
            }
            let rows = customers.len() as i32;
            let packing = if p.covering { 0 } else { rows };
            let covering =
                if p.covering || (p.use_covering_sets_for_one_row && p.covering_sets_present) {
                    rows
                } else {
                    0
                };
            let mut graph = GraphData {
                id,
                source: g.source as i32,
                sink: g.sink as i32,
                elem_set_count: rows,
                packing_set_count: packing,
                covering_set_count: covering,
                packing_neighbourhood: vec![vec![]; packing as usize],
                ..Default::default()
            };
            for v in 0..vertices {
                let row = row_of[v];
                let mut vertex = VertexData {
                    id: v as i32,
                    ..Default::default()
                };
                if row >= 0 {
                    vertex.elem_sets = vec![row];
                    if packing > 0 {
                        vertex.packing_sets = vec![row];
                    }
                    if covering > 0 {
                        vertex.covering_sets = vec![row];
                    }
                }
                graph.vertices.push(vertex);
            }
            for a in &g.arcs {
                graph.arcs.push(ArcData {
                    id: a.id,
                    tail: a.tail as i32,
                    head: a.head as i32,
                    ..Default::default()
                });
            }
            let incidence: Incidence = separation::prepare_incidence(&graph)
                .ok_or_else(|| invalid("native cut graph incidence could not be prepared"))?;
            incidences.push((id, incidence));
            graphs.push(graph);
        }
        let mut solutions = vec![];
        for c in columns {
            if !c.value.is_finite() || c.value < 0. {
                return Err(invalid("fractional column value"));
            }
            let arcs = self.route(g, &c.vertices)?;
            solutions.push(Solution {
                graph_id: if c.graph_id < 0 {
                    p.graph_id
                } else {
                    c.graph_id
                },
                enumerated: c.enumerated,
                arc_ids: arcs.iter().map(|a| a.2).collect(),
            });
        }
        let params = Parameters {
            max_num_rows: p.max_rank,
            max_num_per_round: p.max_cuts as i32,
            memory_type: p.memory_type as i32,
            arc_memory_type: p.arc_memory_type,
            sp_dependent_memory: p.sp_dependent_memory,
            ls_num_iterations: if p.mode == CandidateMode::LocalSearch {
                p.local_search_rounds
            } else {
                0
            },
            ls_iterations_negative: p.ls_iterations_negative,
            neighbourhood_size: 16,
            max_num_triplets: if p.mode == CandidateMode::CliqueEnumeration {
                0
            } else {
                p.max_triplets.min(i32::MAX as usize) as i32
            },
            violation_tolerance: p.violation_epsilon,
            verbosity: -1,
            type_to_separate: if p.covering { 2 } else { 1 },
            use_cov_sets_for_one_row_packing: p.use_covering_sets_for_one_row,
            ..Default::default()
        };
        let mut state = Separation::new(params);
        let refs: Vec<&GraphData> = graphs.iter().collect();
        if !state.prepare(&refs, incidences) {
            return Err(invalid("native rank-one separation preparation failed"));
        }
        let input = Input {
            solutions: solutions.iter().collect(),
            values: columns.iter().map(|c| c.value).collect(),
            cut_memory_type: match p.memory_type {
                MemoryType::Arc => 1,
                MemoryType::AllVertices => 2,
                _ => 0,
            },
            solutions_are_set_sequences: false,
            separation_phase: if p.start_rank > 0 {
                p.start_rank
            } else {
                p.max_rank
            },
            active_cuts: vec![],
        };
        let generated = state
            .separate(&input)
            .ok_or_else(|| invalid("native rank-one separation failed"))?;
        self.separation_phase = state.current_max_rank;
        let added = generated.len();
        for native in generated {
            let mut cut = RankOneCut {
                weights: vec![0; vertices],
                graph_support: BTreeMap::new(),
                graph_pricing_support: BTreeMap::new(),
                native_kpath_support: None,
                memory: vec![false; vertices],
                graph_dependent_memory: true,
                graph_memories: BTreeMap::new(),
                memory_kind: if native.memory_type == 1 {
                    MemoryKind::Arc
                } else {
                    MemoryKind::Vertex
                },
                arc_memory: BTreeSet::new(),
                graph_arc_memories: BTreeMap::new(),
                arc_weights: BTreeMap::new(),
                visit_once: false,
                native_type2: native.kind == 2,
                denominator: native.denominator.abs(),
                dual: 0.,
                class: if native.kind == 1 {
                    CutClass::Covering
                } else {
                    CutClass::Packing
                },
                rhs: native.rhs as i32,
                kind: native.class,
                id: self.cuts.len() as i32,
                ordered_support: None,
            };
            let mut ordered = vec![];
            for r in 0..native.num_rows as usize {
                let vertex = customers[native.set_ids[r] as usize];
                let weight = native.numerators[r].abs();
                cut.weights[vertex] = weight;
                ordered.push((vertex, weight));
            }
            cut.ordered_support = Some(ordered);
            for (&graph, memory) in &native.memory {
                if native.memory_type == 1 {
                    cut.graph_arc_memories.insert(graph, memory.clone());
                    cut.arc_memory.extend(memory.iter().copied());
                } else {
                    let selected = cut
                        .graph_memories
                        .entry(graph)
                        .or_insert_with(|| vec![false; vertices]);
                    *selected = vec![false; vertices];
                    for &v in memory {
                        selected[v as usize] = true;
                        cut.memory[v as usize] = true;
                    }
                }
            }
            self.cuts.push(cut);
        }
        Ok(added)
    }

    /// or_rcsp_separate_cuts_default: the option-driven separator, packing
    /// then (with covering sets) covering, starting at the separation phase.
    pub fn separate_default<const D: usize>(
        &mut self,
        g: &Graph<D>,
        options: &Options,
        customers: &[usize],
        columns: &[FractionalColumn],
    ) -> Result<usize> {
        let rows = options.get("BALDESrankOneCutsMaxNumRows")?;
        let triplets = options.get("BALDESrankOneCutsMaxNumTriplets")?;
        if rows > 3. && triplets < 1. {
            return Err(invalid(
                "lm-1Rank cuts separator: separation of 4 and 5-rows cuts needs tuple based separation",
            ));
        }
        if rows > 5. {
            return Err(invalid(
                "lm-1Rank cuts separator: separation of 6-row and more cuts is not supported",
            ));
        }
        if rows < 1. {
            return Err(invalid(
                "lm-1Rank cuts separator: at least one row required",
            ));
        }
        let per_round = options.get("BALDESrankOneCutsMaxNumPerRound")?;
        if per_round < 1. {
            return Err(invalid(
                "lm-1Rank cuts separator: positive cuts per round required",
            ));
        }
        let raw_iterations = options.get("BALDESrankOneCutsLSnumIterations")?;
        let iterations = raw_iterations.abs();
        let kind = options.get("BALDESrankOneCutsTypeToSeparate")? as i32;
        if !(0..=2).contains(&kind) {
            return Err(invalid("BALDESrankOneCutsTypeToSeparate must be 0, 1 or 2"));
        }
        let policy = memory_policy(options)?;
        let p = &mut self.parameters;
        p.violation_epsilon = 0.02;
        p.max_triplets = triplets as usize;
        p.local_search_rounds = iterations.min(1e9) as i32;
        p.ls_iterations_negative = raw_iterations < 0.;
        p.max_rank = rows as i32;
        if self.separation_phase > rows as i32 {
            self.separation_phase = rows as i32;
        }
        let phase = self.separation_phase;
        let mut found = 0;
        for covering in [false, true] {
            if (!covering && kind == 2) || (covering && (kind == 1 || !self.covering_sets_present))
            {
                continue;
            }
            let p = &mut self.parameters;
            p.max_cuts = per_round as usize;
            p.mode = if iterations > 0. {
                CandidateMode::LocalSearch
            } else {
                CandidateMode::Tuple
            };
            p.covering = covering;
            p.start_rank = phase;
            p.arc_memory_type = policy.arc_memory_type;
            p.covering_sets_present = covering || self.covering_sets_present;
            p.use_covering_sets_for_one_row =
                options.get("BALDESuseCovSetsForSepOneRowPackingCuts")? != 0.;
            p.sp_dependent_memory = policy.sp_dependent;
            p.graph_id = g.graph_id.max(0);
            p.memory_type = policy.memory_type;
            found += self.separate(g, customers, columns)?;
        }
        Ok(found)
    }
}
