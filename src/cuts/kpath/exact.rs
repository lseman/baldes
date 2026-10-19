use super::*;

impl Separation {
    pub(super) fn exact_separation_procedure(&self, out: &mut CandidateSet) {
        let m = self.columns.len();
        let n = self.n;
        let mut total = 0.0;
        for i in 0..n {
            total += self.demands[i];
        }
        let big = (total + total).ceil();
        let source = n + m;
        let sink = n + m + 1;
        let nodes = n + m + 2;
        let mut cap = vec![vec![0.0; nodes]; nodes];
        for c in 0..m {
            cap[source][c] = self.columns[c].value * self.capacity;
        }
        for e in 0..n {
            let node = m + e;
            cap[node][sink] = self.demands[e];
            for &c in &self.customer_columns[e] {
                cap[c][node] = big;
            }
        }
        let mut g = MinCutDigraph::new(cap);
        g.find_min_cut(source as i32, sink as i32);
        let side = g.sink_side.clone();
        let mut remaining = vec![false; nodes];
        let mut count: i32 = -1;
        if !side.is_empty() {
            for &v in &side {
                if let Some(b) = remaining.get_mut(v as usize) {
                    *b = true;
                }
            }
            count = side.len() as i32 - 1;
        }
        if let Some(b) = remaining.get_mut(sink) {
            *b = false;
        }
        let mut cand = Candidate::new(8, n);
        let mut queue: Vec<usize> = Vec::new();
        let queue_insert = |q: &mut Vec<usize>, val: usize| {
            if !q.contains(&val) {
                // std::unordered_set inserts new elements at the front of the iteration list
                q.insert(0, val);
            }
        };

        let absorb = |col: &ColumnInfo,
                      demand: &mut f64,
                      count: &mut i32,
                      cand: &mut Candidate,
                      remaining: &mut [bool],
                      queue: &mut Vec<usize>| {
            for &e in &col.customers {
                if e >= n {
                    continue;
                }
                let d = self.demands[e];
                if !(d > 1e-6) {
                    continue;
                }
                if !remaining.get(m + e).copied().unwrap_or(false) {
                    continue;
                }
                *demand = d + *demand;
                *count -= 1;
                cand.set(e);
                remaining[m + e] = false;
                for &c in &self.customer_columns[e] {
                    if remaining.get(c).copied().unwrap_or(false) {
                        queue_insert(queue, c);
                    }
                }
            }
        };

        if total - 1e-6 > g.excess[sink] && count > 0 {
            for idx in 0..(side.len().saturating_sub(1)) {
                let v = side[idx] as usize;
                if !remaining.get(v).copied().unwrap_or(false) {
                    continue;
                }
                remaining[v] = false;
                count -= 1;
                if v >= self.columns.len() {
                    if count <= 0 {
                        break;
                    }
                    continue;
                }
                let first = &self.columns[v];
                let mut weight = 0.0 + first.value;
                let mut demand = 0.0;
                absorb(
                    first,
                    &mut demand,
                    &mut count,
                    &mut cand,
                    &mut remaining,
                    &mut queue,
                );
                while !queue.is_empty() {
                    let c = queue.remove(0);
                    remaining[c] = false;
                    count -= 1;
                    let col = &self.columns[c];
                    weight += col.value;
                    absorb(
                        col,
                        &mut demand,
                        &mut count,
                        &mut cand,
                        &mut remaining,
                        &mut queue,
                    );
                }
                let k = (demand / self.capacity).ceil() as i32;
                cand.k = k;
                cand.violation = (k as f64) - weight;
                if cand.violation > 1e-6 {
                    out.insert(&cand);
                }
                cand.clear();
                if count <= 0 {
                    break;
                }
            }
        }
    }
}
