//! Strong k-path cut separation, ported from the C++ reference
//! `native/separation/StrongKPathSeparation.h`, its standalone adapter
//! `native/adapters/StrongKPathApiAdapter.h`, and `api/CutOptions.h`.
pub mod preparation;
pub use preparation::{KPathGraph, KPathPreparation};

use std::collections::{BTreeMap, BTreeSet};

use super::shared::bits::Bits1024;
use super::capacity::TimeWindows;
use super::shared::mincut::MinCutDigraph;
use super::shared::pulse::PulseDigraph;
use super::shared::stdhash::UnorderedSet;
use super::shared::stdsort;
use crate::core::options::Options;
use crate::{invalid, Result};

/// Separator parameters for strong k-path cuts.
#[derive(Clone, Debug, PartialEq)]
pub struct KPathParameters {
    pub max_cuts_per_round: usize,
    pub violation_tolerance: f64,
    pub max_two_path_set_size: i32,
    pub pulse_buckets: i32,
    pub candidate_limit: usize,
    pub use_two_path: bool,
}

impl Default for KPathParameters {
    fn default() -> Self {
        Self {
            max_cuts_per_round: 100,
            violation_tolerance: 1e-6,
            max_two_path_set_size: 15,
            pulse_buckets: 10,
            candidate_limit: 1000,
            use_two_path: true,
        }
    }
}

/// Configure cuts per round from BALDESrankOneCutsMaxNumPerRound.
pub fn kpath_parameters(options: &Options, mut base: KPathParameters) -> Result<KPathParameters> {
    let per_round = options.get("BALDESrankOneCutsMaxNumPerRound")?;
    if per_round < 1.0 {
        return Err(invalid("BALDESrankOneCutsMaxNumPerRound must be positive"));
    }
    base.max_cuts_per_round = per_round as usize;
    Ok(base)
}

/// A separated strong k-path cut with depot 0 and customers 1..n.
#[derive(Clone, Debug, PartialEq)]
pub struct StrongKPathCut {
    pub customers: Vec<usize>,
    pub memory: Vec<usize>,
    pub k: i32,
    pub rhs: i32,
    pub two_path: bool,
    pub demand: f64,
    pub weight: f64,
    pub violation: f64,
    pub id: i32,
    customer_count: usize,
}

impl StrongKPathCut {
    /// Route coefficient for an elementary customer sequence (1..=n).
    pub fn coefficient(&self, route: &[usize]) -> Result<i32> {
        // The standalone adapter validates the whole sequence before invoking
        // native coefficient(), even if an earlier customer already hits S.
        if route.iter().any(|&v| v < 1 || v > self.customer_count) {
            return Err(invalid("k-path route customer"));
        }
        Ok(route
            .iter()
            .any(|v| self.customers.binary_search(v).is_ok()) as i32)
    }

    /// Limited-memory coefficient on a physical vertex sequence, optionally including terminals.
    pub fn coefficient_with_memory(&self, route: &[usize], enumerated: bool) -> Result<i32> {
        if route.iter().any(|&v| v > self.customer_count + 1) {
            return Err(invalid("k-path route customer"));
        }
        let mut inside = false;
        let mut count = 0i32;
        for &v in route {
            if self.customers.binary_search(&v).is_ok() {
                if inside {
                    continue;
                }
                if enumerated {
                    return Ok(1);
                }
                count += 1;
                inside = true;
            } else if inside {
                inside = self.memory.binary_search(&v).is_ok();
            }
        }
        Ok(count)
    }
}

#[derive(Clone, Debug)]
struct Params {
    max_num_per_round: i32,
    max_kohl_set_size: i32,
    num_pulse_buckets: i32,
    violation_tolerance: f64,
    max_shrink_removals: i32,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct Candidate {
    heuristic: i32,
    subset: Vec<bool>,
    violation: f64,
    lhs: f64,
    k: i32,
}

impl Candidate {
    fn new(heuristic: i32, n: usize) -> Self {
        Self {
            heuristic,
            subset: vec![false; n],
            violation: 0.0,
            lhs: 0.0,
            k: 0,
        }
    }

    fn clear(&mut self) {
        self.subset.iter_mut().for_each(|b| *b = false);
        self.violation = 0.0;
        self.lhs = 0.0;
        self.k = 0;
    }

    fn test(&self, i: usize) -> bool {
        self.subset.get(i).copied().unwrap_or(false)
    }

    fn set(&mut self, i: usize) {
        if let Some(b) = self.subset.get_mut(i) {
            *b = true;
        }
    }

    fn reset(&mut self, i: usize) {
        if let Some(b) = self.subset.get_mut(i) {
            *b = false;
        }
    }
}

/// KPathCandidateSet: ordered by subset; an equal subset is not replaced.
#[derive(Default)]
struct CandidateSet(BTreeMap<Vec<bool>, Candidate>);

impl CandidateSet {
    fn insert(&mut self, c: &Candidate) {
        self.0.entry(c.subset.clone()).or_insert_with(|| c.clone());
    }
}

#[derive(Clone, Debug)]
struct ColumnInfo {
    column: usize,
    value: f64,
    flag: bool,
    customers: Vec<usize>,
    vertices: Vec<usize>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct RouteColumn {
    id: usize,
    value: f64,
    customers: Vec<usize>,
    key: f64,
}

#[derive(Clone, Debug, Default)]
struct GreedyCandidate {
    id: usize,
    violation: f64,
    delta: f64,
    explored: bool,
}

struct TwoPathCandidate {
    size: i32,
    cut_weight: f64,
    demand: f64,
    column_touched: Vec<bool>,
    touched_columns: Vec<usize>,
    customers: Bits1024,
}

fn greedy_less(a: &GreedyCandidate, b: &GreedyCandidate) -> bool {
    const EPS: f64 = 1e-6;
    if a.explored != b.explored {
        return a.explored;
    }
    if b.violation - EPS > a.violation {
        return true;
    }
    if a.violation > b.violation + EPS {
        return false;
    }
    if b.delta - EPS > a.delta {
        return true;
    }
    if a.delta > b.delta + EPS {
        return false;
    }
    a.id < b.id
}

#[derive(Clone)]
struct Separation {
    params: Params,
    n: usize,
    demands: Vec<f64>,
    capacity: f64,
    two_path: bool,
    next_cut_id: i32,
    columns: Vec<ColumnInfo>,
    customer_columns: Vec<Vec<usize>>,
    num_columns: usize,
    two_path_subsets: UnorderedSet,
    pulse: PulseDigraph,
}

impl Separation {
    fn prepare(
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

    fn compute_violation(&self, c: &mut Candidate) {
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

    fn create_cut(&self, c: &Candidate) -> (i32, i32, Vec<usize>, BTreeSet<usize>) {
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

    fn kohl_greedy_procedure(&mut self, tp: &mut TwoPathCandidate) {
        if tp.size < self.params.max_kohl_set_size && self.n > 0 {
            let mut leaf = true;
            for j in 0..self.n {
                let mut added = 0.0;
                for &col in &self.customer_columns[j] {
                    if !tp.column_touched.get(col).copied().unwrap_or(false) {
                        added += self.columns[col].value;
                    }
                }
                if tp.customers.test(j) {
                    continue;
                }
                if !(self.demands[j] > 1e-6) {
                    continue;
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
                self.kohl_greedy_procedure(tp);
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

    fn kohl_two_path_heuristic(&mut self, out: &mut CandidateSet) {
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
        self.kohl_greedy_procedure(&mut tp);
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

    fn exact_separation_procedure(&self, out: &mut CandidateSet) {
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

    fn greedy_construction_heuristic(&self, out: &mut CandidateSet) {
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

    fn connected_components(&self, out: &mut CandidateSet) {
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

    fn separate(
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

    fn generate_candidates(&mut self) -> Vec<Candidate> {
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

/// Standalone strong k-path cut separator: depot 0, customers 1..n.
#[derive(Clone, Debug)]
pub struct KPathSeparator {
    pub parameters: KPathParameters,
    demands: Vec<f64>,
    capacity: f64,
    windows: Option<TimeWindows>,
    next_id: i32,
}

impl KPathSeparator {
    /// `demands` has n + 1 entries; index 0 (the depot) is set to 0.
    pub fn new(mut demands: Vec<f64>, capacity: f64) -> Result<Self> {
        if demands.len() < 2 || demands.len() > 1024 || !capacity.is_finite() || capacity <= 0.0 {
            return Err(invalid("native k-path dimensions or capacity"));
        }
        if demands.iter().any(|&d| !d.is_finite() || d < 0.0) {
            return Err(invalid("native k-path demand"));
        }
        demands[0] = 0.0;
        Ok(Self {
            parameters: KPathParameters::default(),
            demands,
            capacity,
            windows: None,
            next_id: 0,
        })
    }

    pub fn num_customers(&self) -> usize {
        self.demands.len() - 1
    }

    pub fn set_time_windows(&mut self, windows: TimeWindows) -> Result<()> {
        let n = self.demands.len();
        if windows.ready.len() != n
            || windows.due.len() != n
            || windows.service.len() != n
            || windows.travel.len() != n
            || windows.travel.iter().any(|row| row.len() != n)
        {
            return Err(invalid("time window dimensions"));
        }
        for v in 0..n {
            if !windows.ready[v].is_finite()
                || !windows.due[v].is_finite()
                || windows.ready[v] > windows.due[v]
                || !windows.service[v].is_finite()
                || windows.service[v] < 0.0
            {
                return Err(invalid("time window bounds"));
            }
            for &travel in &windows.travel[v] {
                if !travel.is_finite() || travel < 0.0 {
                    return Err(invalid("travel time"));
                }
            }
        }
        self.windows = Some(windows);
        Ok(())
    }

    pub fn has_time_windows(&self) -> bool {
        self.windows.is_some()
    }

    pub fn reset(&mut self) {
        self.next_id = 0;
    }

    pub fn set_parameter(&mut self, name: &str, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(invalid("native k-path parameter must be finite"));
        }
        match name {
            "max_cuts_per_round" => {
                if value < 1.0 || value > (i32::MAX as f64) {
                    return Err(invalid("native k-path parameters"));
                }
                self.parameters.max_cuts_per_round = value as usize;
            }
            "violation_tolerance" => {
                if !value.is_finite() || value < 0.0 {
                    return Err(invalid("native k-path parameters"));
                }
                self.parameters.violation_tolerance = value;
            }
            "max_two_path_set_size" => {
                if value < 1.0 {
                    return Err(invalid("native k-path parameters"));
                }
                self.parameters.max_two_path_set_size = value as i32;
            }
            "pulse_buckets" => {
                if value < 1.0 {
                    return Err(invalid("native k-path parameters"));
                }
                self.parameters.pulse_buckets = value as i32;
            }
            "candidate_limit" => {
                if value < 0.0 || value > (i32::MAX as f64) {
                    return Err(invalid("native k-path parameters"));
                }
                self.parameters.candidate_limit = value as usize;
            }
            "use_two_path" => {
                self.parameters.use_two_path = value != 0.0;
            }
            _ => return Err(invalid(format!("unknown k-path parameter: {name}"))),
        }
        Ok(())
    }

    pub fn get_parameter(&self, name: &str) -> Result<f64> {
        match name {
            "max_cuts_per_round" => Ok(self.parameters.max_cuts_per_round as f64),
            "violation_tolerance" => Ok(self.parameters.violation_tolerance),
            "max_two_path_set_size" => Ok(self.parameters.max_two_path_set_size as f64),
            "pulse_buckets" => Ok(self.parameters.pulse_buckets as f64),
            "candidate_limit" => Ok(self.parameters.candidate_limit as f64),
            "use_two_path" => Ok(if self.parameters.use_two_path {
                1.0
            } else {
                0.0
            }),
            _ => Err(invalid(format!("unknown k-path parameter: {name}"))),
        }
    }

    pub fn configure(&mut self, options: &Options) -> Result<()> {
        self.parameters = kpath_parameters(options, self.parameters.clone())?;
        Ok(())
    }

    pub fn separate(&mut self, columns: &[(Vec<usize>, f64)]) -> Result<Vec<StrongKPathCut>> {
        self.separate_impl(columns, &vec![false; columns.len()])
    }

    /// Native master-solution adaptation: flag_04 excludes enumerated columns
    /// from memory construction, but not from the separation score.
    pub fn separate_fractional(
        &mut self,
        columns: &[super::FractionalColumn],
    ) -> Result<Vec<StrongKPathCut>> {
        let routes: Vec<_> = columns
            .iter()
            .map(|c| (c.vertices.clone(), c.value))
            .collect();
        let flags: Vec<_> = columns.iter().map(|c| c.enumerated).collect();
        self.separate_impl(&routes, &flags)
    }

    fn separate_impl(
        &mut self,
        columns: &[(Vec<usize>, f64)],
        flags: &[bool],
    ) -> Result<Vec<StrongKPathCut>> {
        let p = &self.parameters;
        if p.max_cuts_per_round == 0
            || p.max_cuts_per_round > (i32::MAX as usize)
            || p.candidate_limit > (i32::MAX as usize)
            || p.max_two_path_set_size < 1
            || p.pulse_buckets < 1
            || !p.violation_tolerance.is_finite()
            || p.violation_tolerance < 0.0
        {
            return Err(invalid("native k-path parameters"));
        }

        let count = self.num_customers();
        let sink = count + 1;
        let width = count + 2;

        let params = Params {
            max_num_per_round: p.max_cuts_per_round as i32,
            max_kohl_set_size: p.max_two_path_set_size,
            num_pulse_buckets: p.pulse_buckets,
            violation_tolerance: p.violation_tolerance,
            max_shrink_removals: p.candidate_limit as i32,
        };

        let use_windows = if self.windows.is_some() && p.use_two_path {
            self.windows.as_ref()
        } else {
            None
        };

        let mut state =
            Separation::prepare(&self.demands[1..], self.capacity, params, use_windows)?;
        state.next_cut_id = self.next_id;

        let mut routes = Vec::with_capacity(columns.len());
        for (column, (customers, value)) in columns.iter().enumerate() {
            if !value.is_finite() || *value < 0.0 {
                return Err(invalid("column value"));
            }
            let mut arcs = Vec::with_capacity(customers.len() + 1);
            let mut previous = 0usize;
            for &v in customers {
                if v < 1 || v > count || v == previous {
                    return Err(invalid("native k-path route customer"));
                }
                arcs.push((previous * width + v) as i32);
                previous = v;
            }
            arcs.push((previous * width + sink) as i32);
            routes.push((arcs, *value, flags[column]));
        }

        let generated = state.separate(&routes);
        self.next_id = state.next_cut_id;

        let mut result = Vec::with_capacity(generated.len());
        for (id, k, set_ids, memory_set) in generated {
            let mut customers = Vec::with_capacity(set_ids.len());
            let mut demand = 0.0;
            for row in set_ids {
                customers.push(row + 1);
                demand += self.demands[row + 1];
            }
            let two_path = (k as f64) > (demand / self.capacity).ceil();
            let memory: Vec<usize> = memory_set.into_iter().collect();
            let mut cut = StrongKPathCut {
                customers,
                memory,
                k,
                rhs: k,
                two_path,
                demand,
                weight: 0.0,
                violation: 0.0,
                id,
                customer_count: count,
            };
            for (route_customers, raw_val) in columns {
                let val = (raw_val * 1e8 + 0.5).floor() / 1e8;
                let coeff = cut.coefficient(route_customers)? as f64;
                cut.weight += val * coeff;
            }
            cut.violation = (k as f64) - cut.weight;
            result.push(cut);
        }
        Ok(result)
    }
}

/// Original set IDs and graph-specific physical vertex memories. This is the
/// generic native cut, before adaptation to the standalone customer API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeKPathCut {
    pub id: i32,
    pub rhs: i32,
    pub set_ids: Vec<usize>,
    pub memory: BTreeMap<i32, BTreeSet<usize>>,
    pub arc_memory: bool,
}

/// General StrongKPathCutSeparation: uses original graph IDs and preprocessed
/// arc incidence, including covering sets and sparse physical vertex IDs.
#[derive(Clone, Debug)]
pub struct NativeKPathSeparator {
    pub parameters: KPathParameters,
    pub preparation: KPathPreparation,
    pub next_cut_id: i32,
    pub num_separate_calls: usize,
}
impl NativeKPathSeparator {
    pub fn new(use_packing_sets: bool) -> Self {
        Self {
            parameters: KPathParameters::default(),
            preparation: KPathPreparation::new(use_packing_sets, 10),
            next_cut_id: 0,
            num_separate_calls: 0,
        }
    }
    pub fn prepare(
        &mut self,
        graphs: &[KPathGraph],
        demands: &[f64],
        capacity: f64,
        two_path_resource: i32,
    ) -> Result<()> {
        self.preparation.pulse.num_buckets = self.parameters.pulse_buckets;
        self.preparation
            .prepare(graphs, demands, capacity, two_path_resource)
    }
    pub fn separate(
        &mut self,
        solutions: &[super::rank_one::separation::Solution],
        values: &[f64],
    ) -> Result<Vec<NativeKPathCut>> {
        let prep = &self.preparation;
        let n = prep.num_customers;
        if n == 0 {
            return Ok(vec![]);
        }
        if solutions.len() != values.len() {
            return Err(invalid("k-path path/value dimensions"));
        }
        self.num_separate_calls += 1;
        let p = &self.parameters;
        let params = Params {
            max_num_per_round: p.max_cuts_per_round as i32,
            max_kohl_set_size: p.max_two_path_set_size,
            num_pulse_buckets: p.pulse_buckets,
            violation_tolerance: p.violation_tolerance,
            max_shrink_removals: p.candidate_limit as i32,
        };
        let mut state = Separation::prepare(&prep.demands[..n], prep.capacity, params, None)?;
        state.pulse = prep.pulse.clone();
        state.two_path = prep.two_path_resource >= 0;
        state.num_columns = solutions.len();
        for (i, (path, &value)) in solutions.iter().zip(values).enumerate() {
            let info = usize::try_from(path.graph_id)
                .ok()
                .and_then(|g| prep.preprocessed.get(g))
                .and_then(Option::as_ref)
                .ok_or_else(|| invalid("k-path solution graph is not preprocessed"))?;
            if path.arc_ids.is_empty() {
                // The original retains input indices after skipping empty
                // paths. A hole before a later column can index past columns.
                if solutions[i + 1..].iter().any(|p| !p.arc_ids.is_empty()) {
                    return Err(invalid("native k-path empty-column index gap"));
                }
                continue;
            }
            let sets = if prep.use_packing_sets {
                &info.arc_packing_sets
            } else {
                &info.arc_covering_sets
            };
            let mut customers = vec![n];
            let mut vertices = vec![];
            for (j, &arc) in path.arc_ids.iter().enumerate() {
                let arc = usize::try_from(arc).map_err(|_| invalid("negative k-path arc ID"))?;
                let ids = sets
                    .get(arc)
                    .ok_or_else(|| invalid("k-path arc incidence"))?;
                let e = ids
                    .first()
                    .copied()
                    .filter(|&e| e >= 0 && (e as usize) < n)
                    .map_or(n, |e| e as usize);
                customers.push(e);
                if j == 0 {
                    vertices.push(
                        *info
                            .arc_tail
                            .get(arc)
                            .ok_or_else(|| invalid("k-path arc tail"))?
                            as usize,
                    );
                }
                vertices.push(
                    *info
                        .arc_head
                        .get(arc)
                        .ok_or_else(|| invalid("k-path arc head"))? as usize,
                );
            }
            state.columns.push(ColumnInfo {
                column: i,
                value: (value * 1e8 + 0.5).floor() / 1e8,
                flag: path.enumerated,
                customers,
                vertices,
            });
        }
        let candidates = state.generate_candidates();
        let mut cuts = vec![];
        for c in candidates {
            let mut memory: BTreeMap<i32, BTreeSet<usize>> = prep
                .preprocessed
                .iter()
                .enumerate()
                .filter(|(_, p)| p.is_some())
                .map(|(g, _)| (g as i32, BTreeSet::new()))
                .collect();
            for col in &state.columns {
                if col.flag {
                    continue;
                }
                let mut pending = BTreeSet::new();
                let mut inside = false;
                for (&element, &vertex) in col.customers.iter().zip(&col.vertices) {
                    if element != n && c.test(element) {
                        memory
                            .entry(solutions[col.column].graph_id)
                            .or_default()
                            .append(&mut pending);
                        inside = true;
                    } else if inside {
                        pending.insert(vertex);
                    }
                }
            }
            cuts.push(NativeKPathCut {
                id: self.next_cut_id,
                rhs: c.k,
                set_ids: (0..n).filter(|&e| c.test(e)).collect(),
                memory,
                arc_memory: false,
            });
            self.next_cut_id += 1;
        }
        self.preparation.pulse = state.pulse;
        Ok(cuts)
    }
    pub fn coefficient(
        &self,
        path: &super::rank_one::separation::Solution,
        cut: &NativeKPathCut,
    ) -> Result<i32> {
        let memory = cut.memory.get(&path.graph_id);
        if path.arc_ids.is_empty() || (memory.is_none() && !path.enumerated) {
            return Ok(0);
        }
        let info = usize::try_from(path.graph_id)
            .ok()
            .and_then(|g| self.preparation.preprocessed.get(g))
            .and_then(Option::as_ref)
            .ok_or_else(|| invalid("k-path solution graph is not preprocessed"))?;
        let sets = if self.preparation.use_packing_sets {
            &info.arc_packing_sets
        } else {
            &info.arc_covering_sets
        };
        let mut inside = false;
        let mut coefficient = 0;
        for &arc in &path.arc_ids {
            let a = usize::try_from(arc).map_err(|_| invalid("negative k-path arc ID"))?;
            let ids = sets.get(a).ok_or_else(|| invalid("k-path arc incidence"))?;
            let hit = ids.first().is_some_and(|&e| {
                e >= 0
                    && (e as usize) < self.preparation.num_customers
                    && cut.set_ids.binary_search(&(e as usize)).is_ok()
            });
            if hit {
                if inside {
                    continue;
                }
                if path.enumerated {
                    return Ok(1);
                }
                coefficient += 1;
                inside = true;
            } else if inside {
                let key = if cut.arc_memory {
                    a
                } else {
                    *info
                        .arc_head
                        .get(a)
                        .ok_or_else(|| invalid("k-path arc head"))? as usize
                };
                inside = memory.is_some_and(|m| m.contains(&key));
            }
        }
        Ok(coefficient)
    }
}
