use super::*;

impl Separation {
    pub(super) fn connected_components(&self, out: &mut CandidateSet) {
        let m = self.columns.len();
        let n = self.n;
        let col_cap = ((self.num_columns + 63) / 64).max(1) * 64;
        let mut column_added = vec![false; col_cap.max(self.num_columns + 1)];
        let mut comp_cols: Vec<RouteColumn> = Vec::new();
        let mut column_visited = vec![false; col_cap.max(self.num_columns + 1)];
        let mut cand = Candidate::new(6, n);
        let mut component = Bits1024::default();
        let mut queue: Vec<usize> = Vec::with_capacity(m);
        queue.push(0);
        let mut processed = 0usize;
        let mut start = 0usize;
        if !column_visited.is_empty() {
            column_visited[0] = true;
        }
        let tol = self.params.violation_tolerance;

        let rescore_and_sort = |comp_cols: &mut Vec<RouteColumn>, cand: &Candidate| {
            if comp_cols.is_empty() {
                return;
            }
            for cc in comp_cols.iter_mut() {
                for &e in &cc.customers {
                    let d = self.demands.get(e).copied().unwrap_or(0.0);
                    if d > 1e-6 && cand.test(e) {
                        cc.key -= d / self.capacity;
                    }
                }
            }
            stdsort::sort(comp_cols, |a, b| a.key > b.key);
        };

        let shrink_columns = |idx: usize,
                              count: &mut i32,
                              removed: &mut i32,
                              max_remove: i32,
                              comp_cols: &[RouteColumn],
                              cand: &mut Candidate,
                              out: &mut CandidateSet| {
            for &e in &comp_cols[idx].customers {
                let d = self.demands.get(e).copied().unwrap_or(0.0);
                if !(d > 1e-6) || !cand.test(e) {
                    continue;
                }
                if !(max_remove > *removed) || *count == 2 {
                    continue;
                }
                *count -= 1;
                *removed += 1;
                cand.reset(e);
            }
            self.compute_violation(cand);
            if cand.violation > tol {
                out.insert(cand);
            }
        };

        while !(processed >= m || start >= m) {
            if !queue.is_empty() {
                let mut count = 0i32;
                let mut demand = 0.0;
                let mut weight = 0.0;
                let mut qi = 0usize;
                while qi < queue.len() {
                    let col = &self.columns[queue[qi]];
                    comp_cols.push(RouteColumn {
                        id: col.column,
                        value: col.value,
                        customers: col.customers.clone(),
                        key: col.value,
                    });
                    weight += col.value;
                    for &e in &col.customers {
                        let d = self.demands.get(e).copied().unwrap_or(0.0);
                        if !(d > 1e-6) || cand.test(e) {
                            continue;
                        }
                        cand.set(e);
                        component.set(e);
                        demand = d + demand;
                        count += 1;
                        for &c in &self.customer_columns[e] {
                            if column_visited.get(c).copied().unwrap_or(false) {
                                continue;
                            }
                            queue.push(c);
                            if let Some(b) = column_visited.get_mut(c) {
                                *b = true;
                            }
                        }
                    }
                    qi += 1;
                }
                processed += queue.len();
                let k = (demand / self.capacity).ceil() as i32;
                cand.k = k;
                cand.violation = (k as f64) - weight;
                if cand.violation > tol {
                    out.insert(&cand);
                }
                rescore_and_sort(&mut comp_cols, &cand);
                component.flip_all();
                let max_remove = self.params.max_shrink_removals;
                if max_remove > 0 && !comp_cols.is_empty() && count > 2 {
                    let mut removed = 0i32;
                    let mut idx = 0usize;
                    loop {
                        shrink_columns(
                            idx,
                            &mut count,
                            &mut removed,
                            max_remove,
                            &comp_cols,
                            &mut cand,
                            out,
                        );
                        idx += 1;
                        if max_remove <= removed || idx >= comp_cols.len() || count <= 2 {
                            break;
                        }
                    }
                }
                cand.clear();
                comp_cols.clear();
                let mut complement = 0i32;
                for e in 0..n {
                    if !component.test(e) {
                        continue;
                    }
                    complement += 1;
                    cand.set(e);
                    for &c in &self.customer_columns[e] {
                        if column_added.get(c).copied().unwrap_or(false) {
                            continue;
                        }
                        if let Some(b) = column_added.get_mut(c) {
                            *b = true;
                        }
                        let col = &self.columns[c];
                        comp_cols.push(RouteColumn {
                            id: col.column,
                            value: col.value,
                            customers: col.customers.clone(),
                            key: col.value,
                        });
                    }
                }
                self.compute_violation(&mut cand);
                if cand.violation > tol {
                    out.insert(&cand);
                }
                rescore_and_sort(&mut comp_cols, &cand);
                if max_remove > 0 && !comp_cols.is_empty() {
                    let mut removed = 0i32;
                    let mut idx = 0usize;
                    while complement > 2 {
                        shrink_columns(
                            idx,
                            &mut complement,
                            &mut removed,
                            max_remove,
                            &comp_cols,
                            &mut cand,
                            out,
                        );
                        idx += 1;
                        if max_remove <= removed || idx >= comp_cols.len() {
                            break;
                        }
                    }
                }
                column_added.iter_mut().for_each(|b| *b = false);
                comp_cols.clear();
                cand.clear();
                component.clear();
                queue.clear();
            }
            if !column_visited.get(start).copied().unwrap_or(false) {
                queue.push(start);
                if let Some(b) = column_visited.get_mut(start) {
                    *b = true;
                }
            } else {
                start += 1;
            }
        }
    }
}
