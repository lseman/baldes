use super::*;

impl Separation {
    pub(super) fn prepare(
        demands: &[f64],
        capacity: f64,
        params: Params,
        windows: Option<&TimeWindows>,
    ) -> Result<Self> {
        if demands.iter().any(|&d| -1e-6 > d) {
            return Err(invalid("demands should be non-negative"));
        }
        if 1e-6 > capacity {
            return Err(invalid("capacity should be positive"));
        }
        let n = demands.len();
        let width = n + 2;
        let sink = n + 1;
        let mut all_demands = demands.to_vec();
        all_demands.push(0.0);

        let element = |v: usize| -> usize {
            if v >= 1 && v <= n && !(1e-6 > all_demands[v - 1]) {
                v - 1
            } else {
                n
            }
        };

        let mut pulse = PulseDigraph::new(params.num_pulse_buckets);
        if let Some(w) = windows {
            let mut ready = vec![1e12; n + 2];
            let mut due = vec![-1e12; n + 2];
            let mut dist = vec![vec![1e12; n + 2]; n + 2];

            for v in 0..width {
                let e = element(v);
                let input = if v == sink { 0 } else { v };
                ready[e] = w.ready[input];
                due[e] = w.due[input];
                if e < n && !(1e-6 > all_demands[e]) {
                    if ready[e] > w.ready[input] {
                        ready[e] = w.ready[input];
                    }
                    if w.due[input] > due[e] {
                        due[e] = w.due[input];
                    }
                } else if v == 0 {
                    if ready[n] > w.ready[input] {
                        ready[n] = w.ready[input];
                    }
                    if w.due[input] > due[n] {
                        due[n] = w.due[input];
                    }
                } else if v == sink {
                    if ready[n + 1] > w.ready[input] {
                        ready[n + 1] = w.ready[input];
                    }
                    if w.due[input] > due[n + 1] {
                        due[n + 1] = w.due[input];
                    }
                }
            }

            for tail in 0..sink {
                for head in 1..=sink {
                    if tail == head {
                        continue;
                    }
                    let weight =
                        w.service[tail] + w.travel[tail][if head == sink { 0 } else { head }];
                    let et = element(tail);
                    let eh = element(head);
                    if tail == 0 {
                        if eh != n {
                            if dist[n][eh] > weight {
                                dist[n][eh] = weight;
                            }
                        } else if head == sink && et != n {
                            if dist[et][n + 1] > weight {
                                dist[et][n + 1] = weight;
                            }
                        }
                    } else if head == sink {
                        if et != n && dist[et][n + 1] > weight {
                            dist[et][n + 1] = weight;
                        }
                    } else if eh != n && et != n && dist[et][eh] > weight {
                        dist[et][eh] = weight;
                    }
                }
            }
            pulse.set_ready_time(ready);
            pulse.set_due_date(due);
            pulse.set_distance_matrix(dist);
        }

        Ok(Self {
            params,
            n,
            demands: all_demands,
            capacity,
            two_path: windows.is_some(),
            next_cut_id: 0,
            columns: Vec::new(),
            customer_columns: vec![Vec::new(); n + 1],
            num_columns: 0,
            two_path_subsets: UnorderedSet::new(),
            pulse,
        })
    }

    pub(super) fn compute_violation(&self, c: &mut Candidate) {
        let mut used = vec![false; self.num_columns];
        let mut demand = 0.0;
        let mut weight = 0.0;
        for i in 0..self.n {
            if !c.test(i) {
                continue;
            }
            demand += self.demands[i];
            for &col in &self.customer_columns[i] {
                if used.get(col).copied().unwrap_or(false) {
                    continue;
                }
                if let Some(u) = used.get_mut(col) {
                    *u = true;
                }
                weight += self.columns[col].value;
            }
        }
        c.lhs = weight;
        let k = (demand / self.capacity).ceil() as i32;
        c.k = k;
        c.violation = (k as f64) - weight;
    }

    pub(super) fn create_cut(&self, c: &Candidate) -> (i32, i32, Vec<usize>, BTreeSet<usize>) {
        let mut elements = Vec::new();
        for i in 0..self.n {
            if c.test(i) {
                elements.push(i);
            }
        }
        let mut memory = BTreeSet::new();
        for col in &self.columns {
            if col.flag {
                continue;
            }
            let mut pending = BTreeSet::new();
            let mut inside = false;
            for j in 0..col.customers.len() {
                let e = col.customers[j];
                if e != self.n && c.test(e) {
                    if !pending.is_empty() {
                        memory.extend(pending.iter().copied());
                        pending.clear();
                    }
                    inside = true;
                } else if inside {
                    pending.insert(col.vertices[j]);
                }
            }
        }
        (self.next_cut_id, c.k, elements, memory)
    }
}
