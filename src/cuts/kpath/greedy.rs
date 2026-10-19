use super::*;

impl Separation {
    pub(super) fn greedy_construction_heuristic(&self, out: &mut CandidateSet) {
        let n = self.n;
        let mut cand = Candidate::new(7, n);
        let mut visited = UnorderedSet::new();
        let tol = self.params.violation_tolerance;
        // C++ vector<bool> has word-aligned capacity (multiples of 64 bits);
        // index n (the "no element" marker) is inside that capacity when n % 64 != 0.
        let elem_cap = ((n + 63) / 64).max(1) * 64;
        for seed in 0..n {
            if self.demands[seed] == 0.0 {
                continue;
            }
            let seed_cols = &self.customer_columns[seed];
            if seed_cols.is_empty() {
                continue;
            }
            let mut column_in = vec![false; self.columns.len()];
            let mut element_in = vec![false; elem_cap.max(n + 1)];
            let mut current = Bits1024::default();
            current.set(seed);
            cand.clear();
            cand.set(seed);
            let mut demand = self.demands[seed];
            element_in[seed] = true;
            let mut weight = 0.0;
            for &c in seed_cols {
                weight += self.columns[c].value;
                column_in[c] = true;
            }
            let evaluate = |e: usize, demand: f64, weight: f64, column_in: &[bool]| -> f64 {
                let d_e = self.demands.get(e).copied().unwrap_or(0.0);
                let k = ((demand + d_e) / self.capacity).ceil() as i32;
                let mut w = weight;
                if let Some(cols) = self.customer_columns.get(e) {
                    for &c in cols {
                        if !column_in.get(c).copied().unwrap_or(false) {
                            w += self.columns[c].value;
                        }
                    }
                }
                (k as f64) - w
            };
            let mut cands: Vec<GreedyCandidate> = Vec::new();
            for &c0 in seed_cols {
                for &e in &self.columns[c0].customers {
                    // vb_test/vb_set on vector<bool>(n): out-of-capacity reads false and ignores writes
                    let in_bounds = e < elem_cap;
                    if in_bounds && element_in[e] {
                        continue;
                    }
                    let violation = evaluate(e, demand, weight, &column_in);
                    current.set(e);
                    let explored = visited.contains(&current);
                    current.reset(e);
                    cands.push(GreedyCandidate {
                        id: e,
                        violation,
                        delta: 0.0,
                        explored,
                    });
                    if in_bounds {
                        element_in[e] = true;
                    }
                }
            }
            while !cands.is_empty() {
                stdsort::sort(&mut cands, greedy_less);
                let best = cands.pop().unwrap();
                if best.explored {
                    break;
                }
                let e = best.id;
                current.set(e);
                visited.insert(current);
                demand += self.demands.get(e).copied().unwrap_or(0.0);
                if e < n {
                    cand.set(e);
                }
                if best.violation > tol {
                    cand.k = (demand / self.capacity).ceil() as i32;
                    cand.violation = best.violation;
                    out.insert(&cand);
                }
                if let Some(cols) = self.customer_columns.get(e) {
                    for &c in cols {
                        if column_in.get(c).copied().unwrap_or(false) {
                            continue;
                        }
                        let col = &self.columns[c];
                        weight += col.value;
                        column_in[c] = true;
                        for &e2 in &col.customers {
                            let in_bounds = e2 < elem_cap;
                            if in_bounds && element_in[e2] {
                                continue;
                            }
                            cands.push(GreedyCandidate {
                                id: e2,
                                violation: 0.0,
                                delta: 0.0,
                                explored: false,
                            });
                            if in_bounds {
                                element_in[e2] = true;
                            }
                        }
                    }
                }
                for gc in &mut cands {
                    gc.violation = evaluate(gc.id, demand, weight, &column_in);
                    current.set(gc.id);
                    gc.explored = visited.contains(&current);
                    current.reset(gc.id);
                }
            }
        }
    }
}
