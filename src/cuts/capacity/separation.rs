use super::*;

impl Separation {
    pub(super) fn add_to_global_pool(&mut self, candidates: &[Candidate]) {
        let mut inserted = 0;
        for cut in candidates {
            if cut.heuristic == 5 {
                continue;
            }
            let mut subset = Bits1024::default();
            for i in 0..self.n {
                if cut.test(i) {
                    subset.set(i);
                }
            }
            inserted += 1;
            self.global_subsets.insert(subset);
            if inserted == self.params.max_num_per_round {
                break;
            }
        }
    }

    /// createCut: (rhs, complement type, set ids, variable weights).
    pub(super) fn create_cut(&self, c: &Candidate) -> (i32, i32, Vec<usize>, Vec<(i32, f64)>) {
        let n = self.n;
        let mut inside = vec![];
        let mut outside = vec![];
        for i in 0..n {
            if !(self.demands[i] > 1e-6) {
                continue;
            }
            if c.test(i) {
                inside.push(i);
            } else {
                outside.push(i);
            }
        }
        let mut weights = vec![];
        let mut add = |a: usize, b: usize, w: f64| {
            for &var in &self.arcs_between[a][b] {
                weights.push((var, w));
            }
        };
        let (rhs, kind);
        if !self.params.complement_form {
            rhs = 2 * c.k;
            kind = 0;
            outside.push(n);
            for &a in &inside {
                for &b in &outside {
                    add(a, b, 1.0);
                }
            }
        } else {
            kind = 1;
            let internal = |vertices: &[usize], add: &mut dyn FnMut(usize, usize, f64)| {
                for i in 0..vertices.len() {
                    for j in i + 1..vertices.len() {
                        add(vertices[i], vertices[j], 1.0);
                    }
                }
            };
            if inside.len() > self.num_positive_demand / 2 {
                rhs = self.num_positive_demand as i32 - inside.len() as i32 - c.k;
                internal(&outside, &mut add);
                for &a in &outside {
                    add(a, n, 0.5);
                }
                for &a in &inside {
                    add(a, n, -0.5);
                }
            } else {
                rhs = inside.len() as i32 - c.k;
                internal(&inside, &mut add);
            }
        }
        (rhs, kind, inside, weights)
    }

    /// separate: rounded flows from the columns, the generators in order, then
    /// stable ranking by violation. Returns (rhs, type, set ids, var weights).
    pub(super) fn separate(
        &mut self,
        columns: &[(Vec<i32>, f64)],
    ) -> Vec<(i32, i32, Vec<usize>, Vec<(i32, f64)>)> {
        self.statistics.calls += 1;
        let n = self.n;
        if n == 0 {
            return vec![];
        }
        let width = (n + 2) as i32;
        for i in 0..=n {
            self.graph.flow[i].iter_mut().for_each(|x| *x = 0.);
            self.graph.neighbours[i].clear();
            if i != n && self.demands[i] > 0. {
                self.graph.groups[i] = vec![i];
                self.graph.representative[i] = i;
            }
        }
        self.columns.clear();
        self.num_columns = columns.len();
        for (c, (arcs, raw)) in columns.iter().enumerate() {
            let value = (raw * 1e8 + 0.5).floor() / 1e8;
            let mut elements = vec![n];
            let mut previous = n;
            for &arc in arcs {
                let head = (arc % width) as usize;
                let element = if head >= 1 && head <= n && self.demands[head - 1] >= 1e-6 {
                    head - 1
                } else {
                    n
                };
                self.graph.flow[previous][element] = self.graph.flow[previous][element] + value;
                self.graph.flow[element][previous] = self.graph.flow[element][previous] + value;
                elements.push(element);
                previous = element;
            }
            if arcs.is_empty() {
                continue;
            }
            self.columns.push(Column {
                column: c as i32,
                value,
                customers: elements,
            });
        }
        for i in 0..=n {
            for j in i + 1..=n {
                if self.graph.flow[i][j] > 1e-6 || self.graph.flow[j][i] > 1e-6 {
                    self.graph.neighbours[i].push(j);
                    self.graph.neighbours[j].push(i);
                }
            }
        }
        self.graph.demands = self.demands.clone();
        let mut candidates = CandidateSet::default();
        if self.two_path {
            self.kohl_two_path(&mut candidates);
            self.two_path_subsets.clear();
        }
        self.connected_components(&mut candidates);
        let threshold = self.params.shrink_threshold;
        if threshold > 1e-6 {
            if !self.params.complement_form || threshold <= 0.999999 {
                let mut copy = self.graph.clone();
                self.shrinking(threshold, &mut candidates, &mut copy, false);
            } else {
                let mut live = self.graph.clone();
                self.shrinking(threshold, &mut candidates, &mut live, true);
                self.graph = live;
            }
        }
        self.min_cut_heuristic(&mut candidates);
        self.greedy_construction(&mut candidates);
        if self.params.use_global_pool {
            self.global_pool_greedy(&mut candidates);
        }
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
        if self.params.use_global_pool {
            self.add_to_global_pool(&sorted);
        }
        let count = (self.params.max_num_per_round.max(0) as usize).min(sorted.len());
        self.statistics.cuts_created += count;
        sorted[..count].iter().map(|c| self.create_cut(c)).collect()
    }
}
