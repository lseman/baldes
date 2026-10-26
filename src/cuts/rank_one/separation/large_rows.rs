use super::*;

impl Separation {
    /// Keeps the best `max_num` candidates, with final ids and memories.
    pub(super) fn insert_best(
        &mut self,
        columns: &[ColumnInfo],
        cuts: &mut CandidateSet,
        max_num: i32,
        sort_and_set_memory: bool,
        insert: bool,
        out: &mut Vec<SeparatedCut>,
    ) {
        if !insert {
            cuts.clear();
            return;
        }
        let limit = self.params.max_per_vertex;
        let mut loads = BTreeMap::new();
        if limit > 0 {
            loads.clone_from(&self.vertex_cut_load);
            for cut in out.iter() {
                for key in self.cut_vertices(cut) {
                    *loads.entry(key).or_default() += 1;
                }
            }
        }
        let mut added = 0;
        for (_, handle) in cuts.ordered() {
            if added >= max_num {
                continue;
            }
            let mut cut = self.arena[handle].clone();
            cut.id = self.next_cut_id;
            self.next_cut_id += 1;
            if sort_and_set_memory {
                let mut rows: Vec<(i32, i32)> = (0..cut.num_rows as usize)
                    .map(|r| (cut.set_ids[r], cut.numerators[r]))
                    .collect();
                rows.sort();
                for (r, (s, n)) in rows.into_iter().enumerate() {
                    cut.set_ids[r] = s;
                    cut.numerators[r] = n;
                }
                if !self.set_cut_memory(&mut cut, columns) {
                    continue;
                }
            }
            if limit > 0 {
                let touched = self.cut_vertices(&cut);
                if touched
                    .iter()
                    .any(|key| loads.get(key).copied().unwrap_or(0) >= limit)
                {
                    continue;
                }
                for key in touched {
                    *loads.entry(key).or_default() += 1;
                }
            }
            out.push(cut);
            added += 1;
        }
        cuts.clear();
    }

    /// Four-row packing cuts by neighbourhood enumeration.
    pub(super) fn four_rows_pack(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        for i in 0..self.num_packing_sets {
            for j in self.packing_neighbour_lists[i as usize].clone() {
                for k in self.table_100[i as usize][j as usize].clone() {
                    for l in self.table_118[i as usize][j as usize][k as usize].clone() {
                        let sets = [i, j, k, l];
                        let mut best = 0.0f64;
                        let mut best_pos = -1i32;
                        for t in 0..4 {
                            let mut lhs = 0.0;
                            for col in columns {
                                let nz = |q: i32| (col.packing_set_count[q as usize] != 0) as u32;
                                let cnt = nz(i) + nz(j) + nz(k) + nz(l) + nz(sets[t]);
                                lhs = lhs + ((cnt / 3) as f64) * col.value;
                            }
                            if lhs > best {
                                best_pos = t as i32;
                            }
                            best = if lhs > best { lhs } else { best };
                        }
                        if !(best - 1.0 > self.params.violation_tolerance) {
                            continue;
                        }
                        if !self.params.skip_subcut_checks
                            && crate::cuts::rank_one::subcuts::three_rows_pack_violated(
                                self,
                                columns,
                                &sets,
                                &self.permutations_4rows,
                            )
                        {
                            continue;
                        }
                        let mut numerators = vec![1, 1, 1, 1];
                        numerators[best_pos as usize] += 1;
                        let cut = self.candidate(1.0, 0, 4, -1, 3, numerators, sets.to_vec());
                        self.insert_candidate(cuts, rounded_score((1.0 - best) * 1e8 + 0.5), cut);
                    }
                }
            }
        }
    }

    /// Five-row packing cuts by neighbourhood enumeration.
    pub(super) fn five_rows_pack(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        let mut lhs = vec![0.0f64; self.permutations_130.len()];
        for i in 0..self.num_packing_sets {
            let neighbours = self.packing_neighbour_lists[i as usize].clone();
            let mut column_ids = vec![];
            for (c, col) in columns.iter().enumerate() {
                let pc = &col.packing_set_count;
                let mut seen = pc[i as usize] != 0;
                for &j in &neighbours {
                    if pc[j as usize] == 0 {
                        continue;
                    }
                    if seen {
                        column_ids.push(c as i32);
                        break;
                    }
                    seen = true;
                }
            }
            for &j in &neighbours {
                for k in self.table_100[i as usize][j as usize].clone() {
                    for l in self.table_118[i as usize][j as usize][k as usize].clone() {
                        for m in l + 1..self.num_packing_sets {
                            if !self.meet(i, m)
                                || !self.meet(j, m)
                                || !self.meet(k, m)
                                || !self.meet(l, m)
                            {
                                continue;
                            }
                            lhs.iter_mut().for_each(|v| *v = 0.0);
                            for &c in &column_ids {
                                let col = &columns[c as usize];
                                let pc = &col.packing_set_count;
                                let mut mask = (pc[i as usize] != 0) as usize;
                                if pc[j as usize] != 0 {
                                    mask += 2;
                                }
                                if pc[k as usize] != 0 {
                                    mask += 4;
                                }
                                if pc[l as usize] != 0 {
                                    mask += 8;
                                }
                                if pc[m as usize] != 0 {
                                    mask += 16;
                                }
                                let coefs = &self.table_160[mask];
                                let perms = &self.table_178[mask];
                                for t in 0..coefs.len() {
                                    let v = &mut lhs[perms[t] as usize];
                                    *v = (coefs[t] as f64) * col.value + *v;
                                }
                            }
                            let mut best = 0.0f64;
                            let mut best_perm = -1i64;
                            for p in 0..self.permutations_130.len() {
                                let kind = self.vector_148[p] as usize;
                                let v = lhs[p] / five_or_more_row_rhs(0, kind) - 1.0;
                                if v > best {
                                    best_perm = p as i64;
                                }
                                best = if v > best { v } else { best };
                            }
                            if !(best > self.params.violation_tolerance) {
                                continue;
                            }
                            let sets = vec![i, j, k, l, m];
                            if !self.params.skip_subcut_checks {
                                if crate::cuts::rank_one::subcuts::three_rows_pack_violated(
                                    self,
                                    columns,
                                    &sets,
                                    &self.permutations_5rows,
                                ) {
                                    continue;
                                }
                                if crate::cuts::rank_one::subcuts::four_rows_pack_violated(
                                    self,
                                    columns,
                                    &sets,
                                    &column_ids,
                                ) {
                                    continue;
                                }
                            }
                            let kind = self.vector_148[best_perm as usize] as usize;
                            let numerators =
                                self.permutations_130[best_perm as usize][..5].to_vec();
                            let id = self.round_cut_counter;
                            self.round_cut_counter += 1;
                            let cut = self.push_candidate(SeparatedCut {
                                id,
                                rhs: five_or_more_row_rhs(0, kind),
                                kind: 0,
                                num_rows: 5,
                                class: kind as i32,
                                denominator: five_or_more_row_denominator(0, kind),
                                numerators,
                                set_ids: sets,
                                memory_type: self.cut_memory_type,
                                memory: BTreeMap::new(),
                            });
                            self.insert_candidate(cuts, rounded_score(0.5 - best * 1e8), cut);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/cuts/rank_one/vertex_budget.rs"]
mod vertex_budget_tests;
