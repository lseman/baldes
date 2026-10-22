use super::*;

impl Separation {
    pub(super) fn kohl_greedy_procedure(
        &mut self,
        tp: &mut TwoPathCandidate,
        visited: &mut std::collections::HashSet<Bits1024>,
        remaining: &mut usize,
    ) {
        if *remaining == 0 || !visited.insert(tp.customers) {
            return;
        }
        *remaining -= 1;
        // A partial search still supplies valid candidate subsets. The usual
        // violation/time-feasibility checks decide whether they yield cuts.
        if *remaining == 0 {
            self.two_path_subsets.insert(tp.customers);
            return;
        }
        if tp.size < self.params.max_kohl_set_size && self.n > 0 {
            let mut leaf = true;
            for j in 0..self.n {
                if *remaining == 0 {
                    break;
                }
                if tp.customers.test(j) || !(self.demands[j] > 1e-6) {
                    continue;
                }
                let mut added = 0.0;
                for &col in &self.customer_columns[j] {
                    if !tp.column_touched.get(col).copied().unwrap_or(false) {
                        added += self.columns[col].value;
                    }
                }
                let flow = tp.cut_weight + added;
                if !(2.0 - self.params.violation_tolerance > flow) {
                    continue;
                }
                tp.cut_weight = flow;
                tp.size += 1;
                tp.demand += self.demands[j];
                tp.customers.set(j);
                let mut pushed = 0usize;
                for i in 0..self.customer_columns[j].len() {
                    let col = self.customer_columns[j][i];
                    if tp.column_touched.get(col).copied().unwrap_or(false) {
                        continue;
                    }
                    if let Some(b) = tp.column_touched.get_mut(col) {
                        *b = true;
                    }
                    pushed += 1;
                    tp.touched_columns.push(col);
                }
                self.kohl_greedy_procedure(tp, visited, remaining);
                for _ in 0..pushed {
                    if let Some(col) = tp.touched_columns.pop() {
                        if let Some(b) = tp.column_touched.get_mut(col) {
                            *b = false;
                        }
                    }
                }
                tp.customers.reset(j);
                leaf = false;
                tp.size -= 1;
                tp.demand -= self.demands[j];
                tp.cut_weight -= added;
            }
            if !leaf {
                return;
            }
        }
        self.two_path_subsets.insert(tp.customers);
    }

    pub(super) fn kohl_two_path_heuristic(&mut self, out: &mut CandidateSet) {
        let mut elements = Vec::with_capacity(self.n);
        let mut c = Candidate::new(9, self.n);
        let mut tp = TwoPathCandidate {
            size: 0,
            cut_weight: 0.0,
            demand: 0.0,
            column_touched: vec![false; self.columns.len()],
            touched_columns: Vec::with_capacity(self.n),
            customers: Bits1024::default(),
        };
        // A subset has the same touched-column union regardless of the order
        // used to construct it. Traverse it once, preserving first-visit order.
        let mut visited = std::collections::HashSet::new();
        let mut remaining = self.params.max_kohl_states;
        self.kohl_greedy_procedure(&mut tp, &mut visited, &mut remaining);
        for set in self.two_path_subsets.iter() {
            for i in 0..self.n {
                if !set.test(i) {
                    continue;
                }
                c.set(i);
                elements.push(i as i32);
            }
            self.compute_violation(&mut c);
            if c.violation > self.params.violation_tolerance {
                out.insert(&c);
            } else if c.k <= 1 && !self.pulse.is_subset_time_feasible(&elements) {
                c.k = 2;
                c.violation = 2.0 - c.lhs;
                if c.violation > self.params.violation_tolerance {
                    out.insert(&c);
                }
            }
            c.clear();
            elements.clear();
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/cuts/kpath/two_path_tests.rs"]
mod tests;
