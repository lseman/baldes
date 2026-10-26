use super::*;

impl Separation {
    /// Empty separator with the given parameters.
    pub fn new(params: Parameters) -> Self {
        Self {
            max_graph_id: -1,
            params,
            cut_memory_type: 1,
            current_max_rank: 1,
            ..Default::default()
        }
    }

    pub fn incidence(&self, graph: i32) -> &Incidence {
        self.preprocessed[graph as usize].as_ref().unwrap()
    }

    pub(super) fn candidate(
        &mut self,
        rhs: f64,
        kind: i32,
        rows: i32,
        class: i32,
        denominator: i32,
        numerators: Vec<i32>,
        sets: Vec<i32>,
    ) -> usize {
        let id = self.round_cut_counter;
        self.round_cut_counter += 1;
        self.push_candidate(SeparatedCut {
            id,
            rhs,
            kind,
            num_rows: rows,
            class,
            denominator,
            numerators,
            set_ids: sets,
            memory_type: self.cut_memory_type,
            memory: BTreeMap::new(),
        })
    }

    pub fn push_candidate(&mut self, cut: SeparatedCut) -> usize {
        self.arena.push(cut);
        self.arena.len() - 1
    }

    pub fn insert_candidate(&mut self, cuts: &mut CandidateSet, score: f64, cut: usize) {
        let id = self.arena[cut].id;
        cuts.insert(score, id, cut);
    }

    /// Coefficient of a column in a cut (memory-filtered state walk).
    pub fn coefficient(&self, sol: &Solution, cut: &SeparatedCut) -> f64 {
        let graph = sol.graph_id;
        let mem = cut.memory.get(&graph);
        if sol.arc_ids.is_empty() {
            return 0.0;
        }
        let unfiltered = sol.enumerated;
        if mem.is_none() && !unfiltered {
            return 0.0;
        }
        let info = self.incidence(graph);
        let packing_like = (cut.kind as u32) & !2u32 == 0;
        let (mut state, mut count) = (0i32, 0i32);
        for &arc in &sol.arc_ids {
            if !unfiltered {
                let mem = mem.unwrap();
                if cut.memory_type == 1 {
                    if !mem.contains(&arc) {
                        state = 0;
                    }
                } else if cut.memory_type == 0 && !mem.contains(&info.arc_tail[arc as usize]) {
                    state = 0;
                }
            }
            let sets = if cut.kind == 0 {
                &info.arc_packing_sets[arc as usize]
            } else {
                &info.arc_covering_sets[arc as usize]
            };
            for &set in sets {
                for r in 0..cut.num_rows as usize {
                    if set == cut.set_ids[r] {
                        state += cut.numerators[r];
                        break;
                    }
                }
                if packing_like {
                    if cut.denominator <= state {
                        count += 1;
                        state -= cut.denominator;
                    }
                } else if state < 0 {
                    count += 1;
                    state -= cut.denominator;
                }
            }
        }
        count as f64
    }

    pub(super) fn raise_rank(&mut self, rank: i32) {
        let mut r = if self.current_max_rank > rank - 1 {
            self.current_max_rank
        } else {
            rank
        };
        if self.params.max_num_rows < r {
            r = self.params.max_num_rows;
        }
        self.current_max_rank = r;
    }

    /// Separates packing then covering cuts rank by rank. Returned cuts carry
    /// final ids and memories.
    pub fn separate(&mut self, input: &Input) -> Option<Vec<SeparatedCut>> {
        let mut out = vec![];
        if self.num_packing_sets == 0 && self.num_covering_sets == 0 {
            return Some(out);
        }
        if input.solutions.len() != input.values.len() {
            return None;
        }
        self.num_separate_calls += 1;
        self.round_cut_counter = 0;
        self.arena.clear();
        self.current_max_rank = input.separation_phase;
        self.cut_memory_type = input.cut_memory_type;
        if self.cut_memory_type == 1 {
            self.active_cuts = input.active_cuts.clone();
        }
        let mut columns = vec![];
        for (sol, &raw) in input.solutions.iter().zip(&input.values) {
            let value = (raw * 1e8 + 0.5).floor() / 1e8;
            let graph = sol.graph_id;
            if !input.solutions_are_set_sequences {
                if let Some(Some(info)) = self.preprocessed.get(graph as usize) {
                    columns.push(ColumnInfo::from_arcs(
                        graph,
                        info,
                        value,
                        sol.arc_ids.clone(),
                        sol.enumerated,
                        self.num_packing_sets,
                        self.num_covering_sets,
                    ));
                }
            } else {
                columns.push(ColumnInfo::from_set_sequence(
                    graph,
                    value,
                    &sol.arc_ids,
                    self.num_packing_sets,
                ));
            }
        }
        if columns.is_empty() {
            return Some(out);
        }
        let elementary: Vec<i32> = (0..columns.len() as i32)
            .filter(|&c| columns[c as usize].elementary)
            .collect();
        let mut cuts = CandidateSet::new();
        let sort_and_set_memory = !input.solutions_are_set_sequences;
        let max_per_round = self.params.max_num_per_round;
        let few = |cuts: &CandidateSet, limit: i32| (cuts.len() as i64) < limit as i64;
        if self.num_packing_sets >= 1 && self.params.type_to_separate != 2 {
            let mut tuple: Option<TupleSeparation> = None;
            let mut ls: Option<LocalSearch<false>> = None;
            self.one_row_pack(&columns, &mut cuts);
            if few(&cuts, max_per_round) {
                self.raise_rank(3);
            }
            self.insert_best(
                &columns,
                &mut cuts,
                max_per_round,
                sort_and_set_memory,
                true,
                &mut out,
            );
            if self.current_max_rank > 2 {
                self.three_rows_pack(&columns, &mut cuts);
                let limit3 = max_per_round * 3 / 2;
                if few(&cuts, limit3) {
                    self.raise_rank(4);
                }
                self.insert_best(
                    &columns,
                    &mut cuts,
                    limit3,
                    sort_and_set_memory,
                    true,
                    &mut out,
                );
                if self.current_max_rank > 3 {
                    let local_search =
                        self.params.ls_num_iterations >= 1 && !self.params.ls_iterations_negative;
                    if !local_search {
                        if self.params.max_num_triplets < 1 {
                            self.four_rows_pack(&columns, &mut cuts);
                        } else {
                            let mut t = TupleSeparation::new(self, &columns, &elementary);
                            t.four_rows_pack(self, &mut cuts);
                            tuple = Some(t);
                        }
                    } else {
                        let mut l = LocalSearch::<false>::new(self, &columns, &elementary);
                        l.run(self, &columns, &mut cuts, 4, 1.0, &[2, 1, 1, 1], 3, 0);
                        ls = Some(l);
                    }
                    if few(&cuts, max_per_round) {
                        self.raise_rank(5);
                    }
                    self.insert_best(
                        &columns,
                        &mut cuts,
                        max_per_round,
                        sort_and_set_memory,
                        true,
                        &mut out,
                    );
                    if self.current_max_rank > 4 {
                        if !local_search {
                            if self.params.max_num_triplets < 1 {
                                self.five_rows_pack(&columns, &mut cuts);
                            } else {
                                let t = tuple.as_mut().unwrap();
                                for kind in 0..2 {
                                    t.five_rows_pack(
                                        self,
                                        &mut cuts,
                                        five_or_more_row_rhs(0, kind),
                                        five_or_more_row_numerators(0, kind),
                                        five_or_more_row_denominator(0, kind),
                                        kind as i32,
                                    );
                                }
                            }
                        } else {
                            let l = ls.as_mut().unwrap();
                            for kind in 0..7 {
                                l.run(
                                    self,
                                    &columns,
                                    &mut cuts,
                                    5,
                                    five_or_more_row_rhs(0, kind),
                                    five_or_more_row_numerators(0, kind),
                                    five_or_more_row_denominator(0, kind),
                                    kind as i32,
                                );
                            }
                        }
                        if few(&cuts, max_per_round) {
                            self.raise_rank(6);
                        }
                        self.insert_best(
                            &columns,
                            &mut cuts,
                            max_per_round,
                            sort_and_set_memory,
                            true,
                            &mut out,
                        );
                        if self.current_max_rank > 5 && self.params.ls_num_iterations > 0 {
                            if self.params.ls_iterations_negative {
                                ls = Some(LocalSearch::<false>::new(self, &columns, &elementary));
                            }
                            let first = [7, 12, 18];
                            let end = [12, 18, 23];
                            for step in 0..3 {
                                let rows = 6 + step as i32;
                                if let Some(l) = ls.as_mut() {
                                    for kind in first[step]..end[step] {
                                        l.run(
                                            self,
                                            &columns,
                                            &mut cuts,
                                            rows,
                                            five_or_more_row_rhs(0, kind),
                                            five_or_more_row_numerators(0, kind),
                                            five_or_more_row_denominator(0, kind),
                                            kind as i32,
                                        );
                                    }
                                }
                                if step < 2 && few(&cuts, max_per_round) {
                                    self.raise_rank(rows + 1);
                                }
                                self.insert_best(
                                    &columns,
                                    &mut cuts,
                                    max_per_round,
                                    sort_and_set_memory,
                                    true,
                                    &mut out,
                                );
                                if step < 2
                                    && !(self.current_max_rank > rows
                                        && self.params.ls_num_iterations > 0)
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        if self.num_covering_sets > 0 && self.params.type_to_separate != 1 {
            self.one_row_cov(&columns, &mut cuts);
            if few(&cuts, max_per_round) {
                self.raise_rank(3);
            }
            self.insert_best(
                &columns,
                &mut cuts,
                max_per_round,
                sort_and_set_memory,
                true,
                &mut out,
            );
            if self.current_max_rank > 2 {
                self.three_rows_cov(&columns, &mut cuts);
                if few(&cuts, max_per_round) {
                    self.raise_rank(4);
                }
                self.insert_best(
                    &columns,
                    &mut cuts,
                    max_per_round * 3 / 2,
                    sort_and_set_memory,
                    true,
                    &mut out,
                );
                if self.current_max_rank > 3 {
                    let mut ls: Option<LocalSearch<true>> = None;
                    if self.params.ls_num_iterations > 0 {
                        let mut l = LocalSearch::<true>::new(self, &columns, &elementary);
                        l.run(self, &columns, &mut cuts, 4, 2.0, &[1, 1, 1, 1], 3, 0);
                        l.run(self, &columns, &mut cuts, 4, 3.0, &[2, 2, 2, 1], 3, 1);
                        ls = Some(l);
                    }
                    if few(&cuts, max_per_round) {
                        self.raise_rank(5);
                    }
                    self.insert_best(
                        &columns,
                        &mut cuts,
                        max_per_round,
                        sort_and_set_memory,
                        true,
                        &mut out,
                    );
                    if self.current_max_rank > 4 && self.params.ls_num_iterations > 0 {
                        let l = ls.as_mut().unwrap();
                        for kind in 0..10 {
                            l.run(
                                self,
                                &columns,
                                &mut cuts,
                                5,
                                five_or_more_row_rhs(1, kind),
                                five_or_more_row_numerators(1, kind),
                                five_or_more_row_denominator(1, kind),
                                kind as i32,
                            );
                        }
                        self.insert_best(
                            &columns,
                            &mut cuts,
                            max_per_round,
                            sort_and_set_memory,
                            true,
                            &mut out,
                        );
                    }
                }
            }
        }
        cuts.clear();
        Some(out)
    }
}
