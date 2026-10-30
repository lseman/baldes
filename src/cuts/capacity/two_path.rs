use super::*;

// Bound the cache even when a large graph has many distinct subsets. Once
// full, continue the original search for subsets that have not been stored.
const MAX_EXACT_SUBSETS: usize = 32_768;
const EXACT_SCALE: f64 = 1_048_576.;
const EXACT_SUM_LIMIT: u64 = 1 << 48;

impl Separation {
    /// All growth/backtracking arithmetic is exact when each input is on this
    /// binary grid and their total scaled magnitude is bounded. That makes a
    /// subset's state independent of the order in which it was reached.
    fn exact_two_path_inputs(&self) -> bool {
        let mut total = 0_u64;
        for &value in self.demands.iter().chain(self.graph.flow.iter().flatten()) {
            if value == 0. && value.is_sign_negative() {
                return false;
            }
            let scaled = value.abs() * EXACT_SCALE;
            if !scaled.is_finite() || scaled.fract() != 0. || scaled > EXACT_SUM_LIMIT as f64 {
                return false;
            }
            total = match total.checked_add(scaled as u64) {
                Some(sum) if sum <= EXACT_SUM_LIMIT => sum,
                _ => return false,
            };
        }
        true
    }

    fn kohl_greedy<const MEMO: bool>(&mut self, c: &mut TwoPath, visited: &mut HashSet<Bits1024>) {
        if MEMO {
            if visited.contains(&c.customers) {
                return;
            }
            if visited.len() < MAX_EXACT_SUBSETS {
                visited.insert(c.customers);
            }
        }
        if c.size < self.params.max_kohl_set_size && self.n > 0 {
            let mut leaf = true;
            for j in 0..self.n {
                if c.customers.test(j) {
                    continue;
                }
                if !(self.demands[j] > 1e-6) {
                    continue;
                }
                if !(self.demands[j] + c.demand < self.capacity + 1e-6) {
                    continue;
                }
                if !((c.cut_weight + c.vector_30[j]) - c.vector_18[j]
                    < 4.0 - self.params.violation_tolerance)
                {
                    continue;
                }
                c.size += 1;
                c.cut_weight = (c.vector_30[j] - c.vector_18[j]) + c.cut_weight;
                c.demand = c.demand + self.demands[j];
                c.customers.set(j);
                for &nb in &self.graph.neighbours[j] {
                    if c.customers.test(nb) {
                        continue;
                    }
                    let w = self.graph.flow[j][nb];
                    c.vector_18[nb] = c.vector_18[nb] + w;
                    c.vector_30[nb] = c.vector_30[nb] - w;
                }
                self.kohl_greedy::<MEMO>(c, visited);
                for &nb in &self.graph.neighbours[j] {
                    if c.customers.test(nb) {
                        continue;
                    }
                    let w = self.graph.flow[j][nb];
                    c.vector_18[nb] = c.vector_18[nb] - w;
                    c.vector_30[nb] = c.vector_30[nb] + w;
                }
                c.customers.reset(j);
                leaf = false;
                c.demand = c.demand - self.demands[j];
                c.size -= 1;
                c.cut_weight = c.cut_weight - (c.vector_30[j] - c.vector_18[j]);
            }
            if !leaf {
                return;
            }
        }
        self.two_path_subsets.insert(c.customers);
    }

    pub(super) fn kohl_two_path(&mut self, output: &mut CandidateSet) {
        let n = self.n;
        let mut growth = TwoPath {
            size: 0,
            cut_weight: 0.,
            demand: 0.,
            vector_18: vec![0.; n + 1],
            vector_30: vec![0.; n + 1],
            customers: Bits1024::default(),
        };
        for i in 0..n {
            for &nb in &self.graph.neighbours[i] {
                growth.vector_30[i] = growth.vector_30[i] + self.graph.flow[i][nb];
            }
        }
        let mut visited = HashSet::new();
        if self.exact_two_path_inputs() {
            self.kohl_greedy::<true>(&mut growth, &mut visited);
        } else {
            self.kohl_greedy::<false>(&mut growth, &mut visited);
        }
        let leaves: Vec<Bits1024> = self.two_path_subsets.iter();
        for leaf in leaves {
            let mut cut = Candidate::new(5, n);
            let mut elements = vec![];
            for i in 0..n {
                if leaf.test(i) {
                    cut.set(i);
                    elements.push(i as i32);
                }
            }
            compute_violation(&self.graph, n, self.capacity, &mut cut, 2);
            if !(cut.violation > self.params.violation_tolerance)
                || self.pulse.is_subset_time_feasible(&elements)
            {
                continue;
            }
            cut.k = 2;
            cut.violation += 2.0;
            output.insert(&cut);
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/cuts/capacity/two_path.rs"]
mod tests;
