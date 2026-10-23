use super::*;

impl Separation {
    /// Unique support and memory vertices in each prepared graph. Arc memory
    /// touches both endpoints; repeated arcs/sets count once per cut.
    pub(super) fn cut_vertices(&self, cut: &NativeCut) -> BTreeSet<(i32, i32)> {
        let mut touched = BTreeSet::new();
        for (graph, info) in self.graph_info.iter().enumerate() {
            let Some(info) = info else { continue };
            let graph = graph as i32;
            let sets = if cut.kind == 0 {
                &info.packing_set_vertices
            } else {
                &info.covering_set_vertices
            };
            for &set in &cut.set_ids {
                for &v in &sets[set as usize] {
                    touched.insert((graph, v));
                }
            }
            if let Some(memory) = cut.memory.get(&graph) {
                for &id in memory {
                    if cut.memory_type == 1 {
                        let incidence = self.incidence(graph);
                        touched.insert((graph, incidence.arc_tail[id as usize]));
                        touched.insert((graph, incidence.arc_head[id as usize]));
                    } else {
                        touched.insert((graph, id));
                    }
                }
            }
        }
        touched
    }

    /// Adds a memory set to the cut (and parallel arcs for arc memory).
    pub(super) fn augment_memory(&self, cut: &mut NativeCut, graph: i32, mem: &BTreeSet<i32>) {
        if mem.is_empty() {
            return;
        }
        if self.params.sp_dependent_memory {
            let m = cut.memory.entry(graph).or_default();
            m.extend(mem.iter().copied());
            if self.cut_memory_type == 1 {
                let info = self.incidence(graph);
                let gi = self.graph_info[graph as usize].as_ref().unwrap();
                for &arc in mem {
                    let parallel = &gi.arcs_between[info.arc_tail[arc as usize] as usize]
                        [info.arc_head[arc as usize] as usize];
                    m.extend(parallel.iter().copied());
                }
            }
            return;
        }
        if self.cut_memory_type == 1 {
            let info = self.incidence(graph);
            for &arc in mem {
                let (tail, head) = (
                    info.arc_tail[arc as usize] as usize,
                    info.arc_head[arc as usize] as usize,
                );
                let graphs: Vec<i32> = cut.memory.keys().copied().collect();
                for node in graphs {
                    let Some(gi) = self.graph_info.get(node as usize).and_then(Option::as_ref)
                    else {
                        continue;
                    };
                    let parallel = gi.arcs_between[tail][head].clone();
                    cut.memory.entry(node).or_default().extend(parallel);
                }
            }
        } else {
            for m in cut.memory.values_mut() {
                m.extend(mem.iter().copied());
            }
        }
    }

    /// Builds the cut memory from the columns. False only for a one-row covering cut
    /// whose memory stayed empty.
    pub fn set_cut_memory(&self, cut: &mut NativeCut, columns: &[ColumnInfo]) -> bool {
        cut.memory.clear();
        for g in 0..=self.max_graph_id {
            if self.preprocessed[g as usize].is_some() {
                cut.memory.insert(g, BTreeSet::new());
            }
        }
        if cut.memory_type == 2 {
            cut.memory_type = 0;
            for g in 0..=self.max_graph_id {
                if let Some(gi) = &self.graph_info[g as usize] {
                    cut.memory.insert(g, gi.default_memory.clone());
                }
            }
            return true;
        }
        let n = if cut.kind == 0 {
            self.num_packing_sets
        } else {
            self.num_covering_sets
        };
        let mut coef = vec![0i32; n as usize];
        for r in 0..cut.num_rows as usize {
            coef[cut.set_ids[r] as usize] = cut.numerators[r];
        }
        for col in columns {
            if col.enumerated {
                continue;
            }
            let touched = if (cut.kind - 1) as u32 <= 1 {
                (0..cut.num_rows as usize)
                    .any(|r| col.covering_set_count[cut.set_ids[r] as usize] != 0)
            } else {
                let mut sum = 0;
                for r in 0..cut.num_rows as usize {
                    sum += col.packing_set_count[cut.set_ids[r] as usize] * cut.numerators[r];
                }
                cut.denominator <= sum
            };
            if !touched || col.arc_ids.is_empty() {
                continue;
            }
            let info = self.incidence(col.graph_id);
            let mut mem = BTreeSet::new();
            let mut state = 0;
            let mut it = 0;
            loop {
                let arc = col.arc_ids[it] as usize;
                let sets = if cut.kind == 0 {
                    &info.arc_packing_sets[arc]
                } else {
                    &info.arc_covering_sets[arc]
                };
                for &set in sets {
                    state += coef[set as usize];
                    if cut.kind == 0 {
                        if cut.denominator <= state {
                            self.augment_memory(cut, col.graph_id, &mem);
                            mem.clear();
                            state -= cut.denominator;
                        }
                    } else if state >= 0 {
                        self.augment_memory(cut, col.graph_id, &mem);
                        mem.clear();
                    } else {
                        state -= cut.denominator;
                    }
                }
                it += 1;
                if it == col.arc_ids.len() {
                    break;
                }
                if state > 0 {
                    let next = col.arc_ids[it];
                    if cut.memory_type == 1 {
                        mem.insert(next);
                    } else {
                        mem.insert(info.arc_tail[next as usize]);
                    }
                }
            }
        }
        if cut.memory_type != 1 {
            for g in 0..=self.max_graph_id {
                if self.preprocessed[g as usize].is_none() {
                    continue;
                }
                let gi = self.graph_info[g as usize].as_ref().unwrap();
                if self.params.sp_dependent_memory && cut.memory[&g].is_empty() {
                    continue;
                }
                for r in 0..cut.num_rows as usize {
                    let set = cut.set_ids[r] as usize;
                    let list = if cut.kind == 0 {
                        &gi.packing_set_vertices[set]
                    } else {
                        &gi.covering_set_vertices[set]
                    };
                    let tmp: BTreeSet<i32> = list.iter().copied().collect();
                    self.augment_memory(cut, g, &tmp);
                }
            }
        } else {
            for g in 0..=self.max_graph_id {
                let Some(info) = &self.preprocessed[g as usize] else {
                    continue;
                };
                let gi = self.graph_info[g as usize].as_ref().unwrap();
                if self.params.sp_dependent_memory && cut.memory[&g].is_empty() {
                    continue;
                }
                let current = cut.memory[&g].clone();
                for arc in current {
                    let tail = info.arc_tail[arc as usize] as usize;
                    let list = &gi.arcs_between[tail][tail];
                    let tmp: BTreeSet<i32> = list.first().copied().into_iter().collect();
                    self.augment_memory(cut, g, &tmp);
                }
                let arc_memory_type = self.params.arc_memory_type;
                if arc_memory_type == 0 {
                    continue;
                }
                let mut heads = BTreeSet::new();
                let kind = cut.kind;
                let set_arcs = |set: usize| {
                    if kind == 0 {
                        gi.packing_set_arcs[set].clone()
                    } else {
                        gi.covering_set_arcs[set].clone()
                    }
                };
                if arc_memory_type == 1 {
                    for r in 0..cut.num_rows as usize {
                        for a in set_arcs(cut.set_ids[r] as usize) {
                            heads.insert(info.arc_head[a as usize]);
                        }
                    }
                }
                let mut vertices = vec![];
                for r in 0..cut.num_rows as usize {
                    let set = cut.set_ids[r] as usize;
                    for a in set_arcs(set) {
                        if arc_memory_type != 2 && !heads.contains(&info.arc_tail[a as usize]) {
                            continue;
                        }
                        self.augment_memory(cut, g, &BTreeSet::from([a]));
                    }
                    let vs = if cut.kind == 0 {
                        &gi.packing_set_vertices[set]
                    } else {
                        &gi.covering_set_vertices[set]
                    };
                    vertices.extend_from_slice(vs);
                }
                for &u in &vertices {
                    for &v in &vertices {
                        let list = &gi.arcs_between[u as usize][v as usize];
                        let tmp: BTreeSet<i32> = list.first().copied().into_iter().collect();
                        self.augment_memory(cut, g, &tmp);
                    }
                }
            }
            for other in &self.active_cuts {
                if cut.num_rows != other.num_rows || (cut.num_rows > 4 && cut.class != other.class)
                {
                    continue;
                }
                let same = (0..cut.num_rows as usize).all(|r| {
                    cut.set_ids[r] == other.set_ids[r] && cut.numerators[r] == other.numerators[r]
                });
                if !same {
                    continue;
                }
                for (graph, memory) in cut.memory.iter_mut() {
                    if let Some(o) = other.memory.get(graph) {
                        memory.extend(o.iter().copied());
                    }
                }
            }
        }
        if cut.kind == 1 && cut.num_rows == 1 {
            return (0..=self.max_graph_id)
                .any(|g| self.preprocessed[g as usize].is_some() && !cut.memory[&g].is_empty());
        }
        true
    }
}
