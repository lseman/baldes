//! Push-relabel minimum cut with gap relabelling, from the C++ reference
//! `native/separation/StrongKPathSeparation.h` (MinCutDigraph).
use super::bits::Bits1024;

#[derive(Clone, Debug, Default)]
pub struct MinCutDigraph {
    pub source: i32,
    pub sink: i32,
    current: i32,
    arc_tail: i32,
    arc_head: i32,
    dead_source_neighbours: Vec<i32>,
    zero_row: Vec<f64>,
    pub source_side: Vec<i32>,
    in_source_side: Bits1024,
    pub sink_side: Vec<i32>,
    in_sink_side: Bits1024,
    active: Vec<i32>,
    is_active: Bits1024,
    labels: Vec<i32>,
    pub excess: Vec<f64>,
    adjacency: Vec<Vec<i32>>,
    pub capacity: Vec<Vec<f64>>,
    residual: Vec<Vec<f64>>,
    flow: Vec<Vec<f64>>,
}
fn erase_value(v: &mut Vec<i32>, x: i32) {
    if let Some(i) = v.iter().position(|&y| y == x) {
        v.remove(i);
    }
}
impl MinCutDigraph {
    /// Links `i < j` in both adjacency lists when `cap[j][i] + cap[i][j] > 1e-6`.
    pub fn new(capacity: Vec<Vec<f64>>) -> Self {
        Self::new_with_terminals(capacity, -1, -1)
    }
    /// Retain all terminal edges, including zero capacities that fractional
    /// capacity separation later forces.
    pub fn new_with_terminals(capacity: Vec<Vec<f64>>, source: i32, sink: i32) -> Self {
        let n = capacity.len();
        let mut adjacency = vec![vec![]; n];
        for i in 0..n {
            for j in i + 1..capacity[i].len() {
                if capacity[j][i] + capacity[i][j] <= 1e-6
                    && i as i32 != source
                    && j as i32 != source
                    && i as i32 != sink
                    && j as i32 != sink
                {
                    continue;
                }
                adjacency[i].push(j as i32);
                adjacency[j].push(i as i32);
            }
        }
        Self {
            adjacency,
            zero_row: vec![0.; n],
            capacity,
            ..Default::default()
        }
    }
    /// Changes only a capacity; adjacency is not rebuilt.
    pub fn set_arc_value(&mut self, tail: i32, head: i32, value: f64) {
        self.capacity[tail as usize][head as usize] = value;
    }
    fn initialize(&mut self) {
        self.residual = self.capacity.clone();
        let n = self.capacity.len();
        self.excess = vec![0.; n];
        self.flow = vec![vec![0.; n]; n];
        self.source_side.clear();
        self.in_source_side.clear();
        self.source_side.push(self.source);
        self.in_source_side.set(self.source as usize);
        self.sink_side.clear();
        self.in_sink_side.clear();
        for i in 0..n {
            self.sink_side.push(i as i32);
            self.in_sink_side.set(i);
        }
        erase_value(&mut self.sink_side, self.source);
        self.in_sink_side.reset(self.source as usize);
        self.is_active.clear();
        self.active.clear();
        let s = self.source as usize;
        for idx in 0..self.adjacency[s].len() {
            let j = self.adjacency[s][idx];
            let uj = j as usize;
            let d = self.residual[s][uj];
            self.excess[s] -= d;
            self.excess[uj] += d;
            if j == self.sink || d <= 1e-6 {
                if self.capacity[uj] == self.zero_row {
                    self.dead_source_neighbours.push(j);
                }
            } else {
                self.active.push(j);
                self.is_active.set(uj);
            }
            self.flow[s][uj] += d;
            self.flow[uj][s] -= d;
            self.residual[s][uj] -= d;
            self.residual[uj][s] = d + self.residual[uj][s];
        }
        self.labels = vec![1; n];
        self.labels[self.sink as usize] = 0;
    }
    fn admissible_arc(&mut self) {
        let u = self.current;
        let uu = u as usize;
        self.arc_tail = 0;
        self.arc_head = 0;
        let candidates: &Vec<i32> = if self.adjacency[uu].len() < self.sink_side.len() {
            &self.adjacency[uu]
        } else {
            &self.sink_side
        };
        let use_adjacency = self.adjacency[uu].len() < self.sink_side.len();
        let found = candidates.iter().copied().find(|&w| {
            let uw = w as usize;
            (!use_adjacency || self.in_sink_side.test(uw))
                && self.labels[uu] == self.labels[uw] + 1
                && self.residual[uu][uw] > 1e-6
        });
        if let Some(w) = found {
            self.arc_tail = u;
            self.arc_head = w;
        }
    }
    fn relabel(&mut self) {
        let u = self.current;
        let uu = u as usize;
        for idx in 0..self.sink_side.len() {
            let v = self.sink_side[idx];
            if v == u || self.labels[v as usize] != self.labels[uu] {
                continue;
            }
            let mut min_label = 99;
            let use_adjacency = self.adjacency[uu].len() < self.sink_side.len();
            let candidates = if use_adjacency {
                &self.adjacency[uu]
            } else {
                &self.sink_side
            };
            for &w in candidates {
                let uw = w as usize;
                if use_adjacency && !self.in_sink_side.test(uw) {
                    continue;
                }
                if !(self.residual[uu][uw] > 1e-6) {
                    continue;
                }
                let l = self.labels[uw] + 1;
                if min_label > l {
                    min_label = l;
                }
            }
            if min_label != 99 {
                self.labels[uu] = min_label;
                return;
            }
            self.source_side.push(u);
            self.in_source_side.set(uu);
            erase_value(&mut self.sink_side, u);
            self.in_sink_side.reset(uu);
            if self.is_active.test(uu) {
                erase_value(&mut self.active, u);
                self.is_active.reset(uu);
            }
            return;
        }
        let mut removed = vec![];
        for idx in 0..self.sink_side.len() {
            let v = self.sink_side[idx];
            let uv = v as usize;
            if self.labels[uv] < self.labels[uu] {
                continue;
            }
            self.source_side.push(v);
            self.in_source_side.set(uv);
            removed.push(idx);
            self.in_sink_side.reset(uv);
            if self.is_active.test(uv) {
                erase_value(&mut self.active, v);
                self.is_active.reset(uv);
            }
        }
        for (k, &idx) in removed.iter().enumerate() {
            self.sink_side.remove(idx - k);
        }
    }
    /// Discharges the first active node or relabels it; `sink_side` then holds
    /// the sink side of the cut.
    pub fn find_min_cut(&mut self, source: i32, sink: i32) {
        self.source = source;
        self.sink = sink;
        self.initialize();
        while !self.active.is_empty() {
            let mut finished = false;
            loop {
                self.current = self.active[0];
                self.admissible_arc();
                if self.arc_tail + self.arc_head < 1 {
                    break;
                }
                let uu = self.current as usize;
                let (a, b) = (self.arc_tail as usize, self.arc_head as usize);
                let res = self.residual[a][b];
                let delta = if self.excess[uu] <= res {
                    self.excess[uu]
                } else {
                    res
                };
                self.excess[a] -= delta;
                self.excess[b] += delta;
                self.flow[b][a] -= delta;
                self.flow[a][b] += delta;
                self.residual[a][b] -= delta;
                self.residual[b][a] = delta + self.residual[b][a];
                if self.excess[uu] == 0.0 {
                    self.active.remove(0);
                    self.is_active.reset(uu);
                }
                let h = self.arc_head;
                if h != self.sink && self.excess[b] > 1e-6 && !self.is_active.test(h as usize) {
                    self.active.push(h);
                    self.is_active.set(b);
                }
                let labels = &self.labels;
                self.active
                    .sort_by(|x, y| labels[*y as usize].cmp(&labels[*x as usize]));
                if self.active.is_empty() {
                    finished = true;
                    break;
                }
            }
            if finished {
                break;
            }
            self.relabel();
        }
        for v in std::mem::take(&mut self.dead_source_neighbours) {
            let uv = v as usize;
            if self.in_source_side.test(uv) {
                self.in_source_side.reset(uv);
                erase_value(&mut self.source_side, v);
            } else if v != self.sink {
                self.in_sink_side.reset(uv);
                erase_value(&mut self.sink_side, v);
            }
        }
    }
}
