use super::*;

impl Separation {
    /// Per-graph set incidence, arc lists and default memory.
    pub(super) fn prepare_graph_info(
        &self,
        g: &GraphData,
        build_arc_lists: bool,
        all_vertices_memory: bool,
    ) -> Option<GraphInfo> {
        let info = self.preprocessed.get(g.id as usize)?.as_ref()?;
        let mut gi = GraphInfo {
            num_packing_sets: g.packing_set_count,
            num_covering_sets: g.covering_set_count,
            packing_set_vertices: vec![vec![]; g.packing_set_count as usize],
            covering_set_vertices: vec![vec![]; g.covering_set_count as usize],
            packing_set_arcs: vec![vec![]; g.packing_set_count as usize],
            covering_set_arcs: vec![vec![]; g.covering_set_count as usize],
            ..Default::default()
        };
        for v in &g.vertices {
            for &p in &v.packing_sets {
                gi.packing_set_vertices[p as usize].push(v.id);
            }
            for &q in &v.covering_sets {
                gi.covering_set_vertices[q as usize].push(v.id);
            }
        }
        if all_vertices_memory {
            for v in &g.vertices {
                if v.id != g.sink && v.id != g.source {
                    gi.default_memory.insert(v.id);
                }
            }
        }
        if build_arc_lists {
            let n = (info.max_vertex_id + 1) as usize;
            gi.arcs_between = vec![vec![vec![]; n]; n];
        }
        for a in &g.arcs {
            for &p in &a.packing_sets {
                gi.packing_set_arcs[p as usize].push(a.id);
            }
            for &q in &a.covering_sets {
                gi.covering_set_arcs[q as usize].push(a.id);
            }
            if !build_arc_lists {
                continue;
            }
            let (t, h) = (a.tail as usize, a.head as usize);
            gi.arcs_between[t][h].push(a.id);
            gi.arcs_between[h][t].push(a.id);
            if a.head == g.sink && a.head != g.source {
                gi.arcs_between[g.source as usize][h].push(a.id);
            }
            if a.tail == g.source && a.tail != g.sink {
                gi.arcs_between[t][g.sink as usize].push(a.id);
            }
        }
        let includes =
            |big: &[i32], small: &[i32]| small.iter().all(|x| big.binary_search(x).is_ok());
        for v in &g.vertices {
            if !includes(&v.elem_sets, &v.packing_sets) || !includes(&v.elem_sets, &v.covering_sets)
            {
                return None;
            }
        }
        for a in &g.arcs {
            if !includes(&a.elem_sets, &a.packing_sets) || !includes(&a.elem_sets, &a.covering_sets)
            {
                return None;
            }
        }
        Some(gi)
    }

    /// Index permutations of the 4- and 5-row packing sub-cut checks.
    pub(super) fn build_packing_permut_vectors(&mut self) {
        generate_permut_rows(&[1, 1, 1, 0], &mut self.permutations_4rows);
        if self.params.max_num_rows > 4 {
            generate_permut_rows(&[1, 1, 1, 0, 0], &mut self.permutations_5rows);
            generate_permut_rows(&[1, 1, 1, 1, 0], &mut self.permutations_5rows_b);
            let mut p = vec![2, 1, 1, 1, 0];
            let mut perms = vec![];
            generate_different_permuts(&mut p, &mut perms, 0);
            self.permutations_190.resize(perms.len(), vec![0]);
            for (i, perm) in perms.iter().enumerate() {
                for j in 0..=4 {
                    match perm[j] {
                        1 => self.permutations_190[i].push(j as i32),
                        2 => self.permutations_190[i][0] = j as i32,
                        _ => {}
                    }
                }
            }
        }
    }

    /// Index permutations of the covering sub-cut checks.
    pub(super) fn build_covering_permut_vectors(&mut self) {
        let mut p = vec![1, 2, 2, 2, 0];
        let mut perms = vec![];
        generate_different_permuts(&mut p, &mut perms, 0);
        self.covering_permutations.resize(perms.len(), vec![0]);
        for (i, perm) in perms.iter().enumerate() {
            for j in 0..=4 {
                match perm[j] {
                    2 => self.covering_permutations[i].push(j as i32),
                    1 => self.covering_permutations[i][0] = j as i32,
                    _ => {}
                }
            }
        }
    }

    /// Packing-set neighbourhoods from the graphs or their distances.
    pub(super) fn calculate_packing_neighbours(&mut self, graphs: &[&GraphData]) {
        let n = self.num_packing_sets as usize;
        self.packing_neighbours = vec![[0; 16]; n];
        self.packing_distances_defined = false;
        for g in graphs {
            for p in 0..g.packing_set_count as usize {
                let Some(list) = g.packing_neighbourhood.get(p) else {
                    continue;
                };
                if list.is_empty() {
                    continue;
                }
                self.packing_distances_defined = true;
                for &e in list {
                    set_bit(&mut self.packing_neighbours[p], e as usize);
                }
            }
        }
        if !self.packing_distances_defined {
            let mut dist = vec![vec![0.0f64; n]; n];
            let mut count = 0;
            for g in graphs {
                if g.elem_set_distances.is_empty() {
                    continue;
                }
                let (c, c1) = (count as f64, (count + 1) as f64);
                for i in 0..n {
                    for j in 0..n {
                        let d = &mut dist[i][j];
                        *d = (*d * c + g.elem_set_distances[i][j]) / c1;
                    }
                }
                count += 1;
            }
            if count != 0 {
                let k = self.num_packing_sets.min(self.params.neighbourhood_size) as usize;
                for i in 0..n {
                    let mut order: Vec<(f64, usize)> = (0..n).map(|j| (dist[i][j], j)).collect();
                    order.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    if order.is_empty() {
                        continue;
                    }
                    let bits = &mut self.packing_neighbours[i];
                    set_bit(bits, order[0].1);
                    if k == 1 {
                        continue;
                    }
                    let mut t = 1;
                    while t != order.len() {
                        set_bit(bits, order[t].1);
                        t += 1;
                        if t == k {
                            break;
                        }
                    }
                }
                self.packing_distances_defined = true;
            }
            if !self.packing_distances_defined {
                return;
            }
        }
        if self.params.ls_num_iterations > 0 {
            self.neighbours_208 = vec![[0; 16]; n];
            for i in 0..n {
                for j in 0..n {
                    if i != j && bits_any(&self.packing_neighbours[i], &self.packing_neighbours[j])
                    {
                        set_bit(&mut self.neighbours_208[i], j);
                    }
                }
            }
        }
    }

    pub(super) fn meet(&self, a: i32, b: i32) -> bool {
        bits_any(
            &self.packing_neighbours[a as usize],
            &self.packing_neighbours[b as usize],
        )
    }

    /// Neighbourhood tuples and coefficient tables of 4/5-row enumeration.
    pub(super) fn build_4_and_5_row_structures(&mut self) {
        let build = self.params.max_num_triplets <= 0
            && (self.params.ls_num_iterations == 0 || self.params.ls_iterations_negative);
        if !build {
            return;
        }
        let n = self.num_packing_sets as usize;
        self.packing_neighbour_lists = vec![vec![]; n];
        self.table_100 = vec![vec![vec![]; n]; n];
        self.table_118 = vec![vec![vec![vec![]; n]; n]; n];
        let m = self.num_packing_sets;
        for i in 0..m {
            for j in i + 1..m {
                if !self.meet(i, j) {
                    continue;
                }
                self.packing_neighbour_lists[i as usize].push(j);
                for k in j + 1..m {
                    if !self.meet(i, k) || !self.meet(j, k) {
                        continue;
                    }
                    self.table_100[i as usize][j as usize].push(k);
                    for l in k + 1..m {
                        if self.meet(i, l) && self.meet(j, l) && self.meet(k, l) {
                            self.table_118[i as usize][j as usize][k as usize].push(l);
                        }
                    }
                }
            }
        }
        for kind in 0..=6 {
            let mut numerators = five_or_more_row_numerators(0, kind).to_vec();
            generate_different_permuts(&mut numerators, &mut self.permutations_130, 0);
            self.vector_148
                .resize(self.permutations_130.len(), kind as i32);
        }
        self.table_160 = vec![vec![]; 32];
        self.table_178 = vec![vec![]; 32];
        for mask in 1u32..32 {
            for (i, perm) in self.permutations_130.iter().enumerate() {
                let mut sum = 0;
                if mask & 1 != 0 {
                    sum = perm[0];
                }
                if mask & 2 != 0 {
                    sum += perm[1];
                }
                if mask & 4 != 0 {
                    sum += perm[2];
                }
                if mask & 8 != 0 {
                    sum += perm[3];
                }
                if mask >> 4 != 0 {
                    sum += perm[4];
                }
                let denominator = five_or_more_row_denominator(0, self.vector_148[i] as usize);
                if denominator <= sum {
                    self.table_160[mask as usize].push(sum / denominator);
                    self.table_178[mask as usize].push(i as i32);
                }
            }
        }
    }

    /// Prepares the separator for the graphs. Incidences are prepared by the caller.
    pub fn prepare(&mut self, graphs: &[&GraphData], incidences: Vec<(i32, Incidence)>) -> bool {
        if graphs.is_empty() {
            return false;
        }
        for g in graphs {
            self.max_graph_id = self.max_graph_id.max(g.id);
            self.num_packing_sets = self.num_packing_sets.max(g.packing_set_count);
            self.num_covering_sets = self.num_covering_sets.max(g.covering_set_count);
        }
        let size = (self.max_graph_id + 1) as usize;
        self.preprocessed = vec![None; size];
        self.graph_info = vec![None; size];
        for (id, incidence) in incidences {
            self.preprocessed[id as usize] = Some(incidence);
        }
        if graphs
            .iter()
            .any(|g| self.preprocessed[g.id as usize].is_none())
        {
            return false;
        }
        if self.params.max_num_rows > 3 {
            if self.num_packing_sets > 0 {
                self.build_packing_permut_vectors();
                self.calculate_packing_neighbours(graphs);
                if !self.packing_distances_defined && self.params.max_num_triplets <= 0 {
                    self.params.ls_num_iterations = 0;
                    self.params.max_num_rows = 4;
                    return false;
                }
                self.build_4_and_5_row_structures();
            }
            if self.num_covering_sets != 0 {
                self.build_covering_permut_vectors();
            }
        }
        if !self.params.sp_dependent_memory {
            if graphs.len() == 1 {
                self.params.sp_dependent_memory = true;
            } else {
                let first: BTreeSet<_> = graphs[0].vertices.iter().map(|v| v.id).collect();
                for g in &graphs[1..] {
                    let other: BTreeSet<_> = g.vertices.iter().map(|v| v.id).collect();
                    if first != other {
                        self.params.sp_dependent_memory = true;
                        break;
                    }
                }
            }
        }
        let build_arc_lists = (self.params.memory_type as u32) <= 1;
        let all_vertices_memory = self.params.memory_type == 3;
        for g in graphs {
            let Some(gi) = self.prepare_graph_info(g, build_arc_lists, all_vertices_memory) else {
                return false;
            };
            self.graph_info[g.id as usize] = Some(gi);
        }
        true
    }
}
