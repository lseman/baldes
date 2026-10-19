use super::*;

impl Separation {
    pub(super) fn separate(
        &mut self,
        routes: &[(Vec<i32>, f64, bool)],
    ) -> Vec<(i32, i32, Vec<usize>, BTreeSet<usize>)> {
        let n = self.n;
        if n == 0 {
            return Vec::new();
        }
        let width = (n + 2) as i32;
        self.columns.clear();
        for i in 0..n {
            self.customer_columns[i].clear();
        }
        self.num_columns = routes.len();
        for (i, (arcs, raw_val, flag)) in routes.iter().enumerate() {
            let value = (raw_val * 1e8 + 0.5).floor() / 1e8;
            let mut elements = Vec::with_capacity(arcs.len() + 1);
            elements.push(n);
            for &arc in arcs {
                let head = (arc % width) as usize;
                let e = if head >= 1 && head <= n { head - 1 } else { n };
                elements.push(e);
            }
            if arcs.is_empty() {
                continue;
            }
            let mut vertices = Vec::with_capacity(arcs.len() + 1);
            vertices.push((arcs[0] / width) as usize);
            for &arc in arcs {
                vertices.push((arc % width) as usize);
            }
            self.columns.push(ColumnInfo {
                column: i,
                value,
                flag: *flag,
                customers: elements,
                vertices,
            });
        }
        let sorted = self.generate_candidates();
        let mut out = Vec::with_capacity(sorted.len());
        for c in &sorted {
            out.push(self.create_cut(c));
            self.next_cut_id += 1;
        }
        out
    }

    pub(super) fn generate_candidates(&mut self) -> Vec<Candidate> {
        let n = self.n;
        for col in &self.columns {
            for &e in &col.customers {
                if e != n {
                    self.customer_columns[e].push(col.column);
                }
            }
        }
        let mut candidates = CandidateSet::default();
        if self.two_path {
            self.kohl_two_path_heuristic(&mut candidates);
            self.two_path_subsets.clear();
        }
        self.greedy_construction_heuristic(&mut candidates);
        self.exact_separation_procedure(&mut candidates);
        self.connected_components(&mut candidates);

        let mut sorted: Vec<Candidate> = candidates.0.into_values().collect();
        stdsort::stable_sort(&mut sorted, |a, b| {
            if a.violation > b.violation + 1e-6 {
                return true;
            }
            if b.violation - 1e-6 > a.violation {
                return false;
            }
            a.subset < b.subset
        });

        sorted.truncate(self.params.max_num_per_round.max(0) as usize);
        sorted
    }
}
