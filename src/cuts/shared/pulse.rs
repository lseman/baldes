//! Pulse search for the time feasibility of a customer set on one route, from
//! the C++ reference `native/separation/StrongKPathSeparation.h`.
use super::bits::Bits1024;

#[derive(Clone, Debug)]
pub struct PulseDigraph {
    pub num_customers: i32,
    pub num_nodes: i32,
    pub num_buckets: i32,
    pub bucket_width: f64,
    pub feasible_arc_cost: i32,
    pub source: i32,
    pub sink: i32,
    pub num_pulses: u64,
    pub ready_time: Vec<f64>,
    pub due_date: Vec<f64>,
    pub distance: Vec<Vec<f64>>,
    pub arc_cost: Vec<Vec<f64>>,
    pub bound: Vec<Vec<f64>>,
}
impl PulseDigraph {
    pub fn new(buckets: i32) -> Self {
        Self {
            num_customers: -1,
            num_nodes: -1,
            num_buckets: buckets,
            bucket_width: 0.,
            feasible_arc_cost: -1,
            source: -1,
            sink: -1,
            num_pulses: 0,
            ready_time: vec![],
            due_date: vec![],
            distance: vec![],
            arc_cost: vec![],
            bound: vec![],
        }
    }
    /// Source n, sink n+1; infeasible arcs cost 2e12; bucket width
    /// ceil((due[sink] - ready[source]) / buckets).
    fn prepare(&mut self) {
        self.source = self.num_customers;
        self.sink = self.num_customers + 1;
        let nodes = self.num_nodes.max(0) as usize;
        let fill = self.feasible_arc_cost as f64;
        self.arc_cost.resize(nodes, vec![fill; nodes]);
        let n = self.distance.len();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    self.arc_cost[i][j] = 0.;
                } else if self.distance[i][j] + self.ready_time[i] > self.due_date[j] + 1e-6 {
                    self.arc_cost[i][j] = 2e12;
                }
            }
        }
        self.bucket_width = ((self.due_date[self.sink as usize]
            - self.ready_time[self.source as usize])
            / self.num_buckets as f64)
            .ceil();
    }
    fn first(&mut self, size: usize) -> bool {
        if self.num_customers < 0 {
            self.num_nodes = size as i32;
            self.num_customers = size as i32 - 2;
            return true;
        }
        false
    }
    pub fn set_ready_time(&mut self, ready: Vec<f64>) -> bool {
        if self.first(ready.len()) {
            self.ready_time = ready;
            return true;
        }
        if ready.is_empty()
            || (!self.due_date.is_empty() && ready.len() != self.due_date.len())
            || (!self.distance.is_empty() && ready.len() != self.distance.len())
        {
            return false;
        }
        self.ready_time = ready;
        if !self.due_date.is_empty() && !self.distance.is_empty() {
            self.prepare();
        }
        true
    }
    pub fn set_due_date(&mut self, due: Vec<f64>) -> bool {
        if self.first(due.len()) {
            self.due_date = due;
            return true;
        }
        if due.is_empty()
            || (!self.ready_time.is_empty() && due.len() != self.ready_time.len())
            || (!self.distance.is_empty() && due.len() != self.distance.len())
        {
            return false;
        }
        self.due_date = due;
        if !self.ready_time.is_empty() && !self.distance.is_empty() {
            self.prepare();
        }
        true
    }
    pub fn set_distance_matrix(&mut self, distance: Vec<Vec<f64>>) -> bool {
        if self.first(distance.len()) {
            self.distance = distance;
            return true;
        }
        if distance.is_empty()
            || (!self.ready_time.is_empty() && distance.len() != self.ready_time.len())
            || (!self.due_date.is_empty() && distance.len() != self.due_date.len())
        {
            return false;
        }
        self.distance = distance;
        if !self.ready_time.is_empty() && !self.due_date.is_empty() {
            self.prepare();
        }
        true
    }
    fn bucket(&self, time: f64) -> usize {
        let bucket = (time - self.ready_time[self.source as usize] + 1e-6) / self.bucket_width;
        // Float-to-int conversion truncates, which equals floor for nonnegative
        // buckets. Preserve floor for negative values (and NaN handling).
        let bucket = if bucket >= 0. {
            bucket as i32
        } else {
            bucket.floor() as i32
        };
        bucket as usize
    }
    fn bound_at(&self, node: usize, bucket: usize) -> f64 {
        self.bound
            .get(node)
            .and_then(|row| row.get(bucket))
            .copied()
            .unwrap_or(-1e12)
    }
    /// Depth-first extension over the subset; stops after 1,000,000 pulses.
    fn pulse(&mut self, p: &mut PulsePath, subset: &[i32]) {
        self.num_pulses += 1;
        let (time, cost, i) = (p.time, p.cost, p.node as usize);
        p.visited.set(i);
        let sink = self.sink as usize;
        if self.due_date[sink] + 1e-6 > time + self.distance[i][sink] {
            let close = cost + self.arc_cost[i][sink];
            if p.best - 1e-6 > close {
                p.best = close;
            }
        }
        for &j in subset {
            let uj = j as usize;
            if p.visited.test(uj) {
                continue;
            }
            p.node = j;
            let t = (time + self.distance[i][uj]).max(self.ready_time[uj]);
            p.time = t;
            let c = cost + self.arc_cost[i][uj];
            p.cost = c;
            if !(t > self.due_date[uj] + 1e-6) {
                let b = self.bucket(t);
                if !(c + self.bound_at(uj, b) > p.best - 1e-6) {
                    p.visited.set(uj);
                    self.pulse(p, subset);
                    p.visited.reset(uj);
                }
            }
            if self.num_pulses > 1_000_000 {
                return;
            }
        }
    }
    /// Bucket bounds from the last bucket down to 1, then a pulse from the source.
    pub fn run(&mut self, subset: &[i32]) -> f64 {
        // Keep bucket storage between subset checks. Clear every row so nodes
        // outside this subset retain the same missing-bound semantics.
        self.bound
            .resize_with(self.num_customers.max(0) as usize, Vec::new);
        for row in &mut self.bound {
            row.clear();
        }
        for &i in subset {
            self.bound[i as usize].resize(self.num_buckets as usize, -1e12);
        }
        self.num_pulses = 0;
        let source = self.source as usize;
        let mut k = self.num_buckets - 1;
        while k > 0 {
            let mut stop = false;
            for &i in subset {
                let ui = i as usize;
                let mut best = if self.num_buckets - 1 == k {
                    1e12
                } else {
                    self.bound[ui][(k + 1) as usize]
                };
                let mut p = PulsePath {
                    node: i,
                    cost: 0.,
                    time: self.bucket_width * k as f64 + self.ready_time[source],
                    best,
                    visited: Bits1024::default(),
                };
                if !(p.time > self.due_date[ui] + 1e-6) {
                    let b = self.bucket(p.time);
                    if !(0.0 + self.bound_at(ui, b) > best - 1e-6) {
                        self.pulse(&mut p, subset);
                        best = p.best;
                    }
                }
                self.bound[ui][k as usize] = best;
                if self.num_pulses > 1_000_000 {
                    stop = true;
                    break;
                }
            }
            if stop || (subset.is_empty() && self.num_pulses > 1_000_000) {
                break;
            }
            k -= 1;
        }
        let mut p = PulsePath {
            node: self.source,
            cost: 0.,
            time: self.ready_time[source],
            best: 1e12,
            visited: Bits1024::default(),
        };
        self.pulse(&mut p, subset);
        p.best
    }
    /// An empty subset or a run over the pulse limit is feasible; otherwise the
    /// best cost must equal (|S| + 1) * feasible_arc_cost exactly.
    pub fn is_subset_time_feasible(&mut self, subset: &[i32]) -> bool {
        if subset.is_empty() {
            return true;
        }
        let best = self.run(subset);
        if self.num_pulses > 1_000_000 {
            return true;
        }
        ((subset.len() as i32 + 1) * self.feasible_arc_cost) as f64 == best
    }
}
#[derive(Clone, Debug)]
pub struct PulsePath {
    pub node: i32,
    pub cost: f64,
    pub time: f64,
    pub best: f64,
    pub visited: Bits1024,
}

#[cfg(test)]
#[path = "../../../tests/unit/cuts/shared/pulse_tests.rs"]
mod tests;
