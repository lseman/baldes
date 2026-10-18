//! Rounded capacity cut separation, from the C++ reference
//! `native/separation/RoundCapacitySeparation.h` and its standalone adapter
//! `native/adapters/RoundCapacityApiAdapter.h`. Cuts are x(delta(S)) >=
//! 2 ceil(d(S)/Q) on undirected edge flows (or the complement form), and Kohl
//! 2-path cuts when time windows are given. The subset pool persists across
//! separation calls.
use super::shared::bits::Bits1024;
use super::shared::mincut::MinCutDigraph;
use super::shared::pulse::PulseDigraph;
use super::shared::stdhash::UnorderedSet;
use super::shared::stdsort;
use crate::{invalid, Result};
use std::collections::{BTreeMap, HashSet};

/// Separator controls.
#[derive(Clone, Debug, PartialEq)]
pub struct CapacityParameters {
    pub enabled: bool,
    pub max_cuts_per_round: usize,
    pub violation_tolerance: f64,
    pub shrink_threshold: f64,
    pub candidate_limit: usize,
    pub use_global_pool: bool,
    pub use_two_path: bool,
    pub max_two_path_set_size: i32,
    pub pulse_buckets: i32,
    pub complement_form: bool,
}
impl Default for CapacityParameters {
    fn default() -> Self {
        Self {
            enabled: true,
            max_cuts_per_round: 100,
            violation_tolerance: 0.02,
            shrink_threshold: 1.0,
            candidate_limit: 1000,
            use_global_pool: true,
            use_two_path: true,
            max_two_path_set_size: 15,
            pulse_buckets: 10,
            complement_form: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapacityKind {
    Capacity,
    TwoPath,
    Complement,
}
/// A separated cut with depot 0 and customers 1..n. Edge weights are given
/// per undirected edge (a < b).
#[derive(Clone, Debug, PartialEq)]
pub struct CapacityCut {
    pub customers: Vec<usize>,
    /// Complement form: the set whose internal edges are counted.
    pub other: Vec<usize>,
    pub kind: CapacityKind,
    pub rhs: i32,
    pub k: i32,
    pub demand: f64,
    pub violation: f64,
    pub edges: BTreeMap<(usize, usize), f64>,
}
impl CapacityCut {
    pub fn greater_equal(&self) -> bool {
        self.kind != CapacityKind::Complement
    }
    pub fn edge_coefficient(&self, a: usize, b: usize) -> f64 {
        self.edges.get(&(a.min(b), a.max(b))).copied().unwrap_or(0.)
    }
    /// Route coefficient: edge weights along depot, route..., depot.
    pub fn coefficient(&self, route: &[usize]) -> i32 {
        let mut total = 0.;
        let mut previous = 0;
        for &v in route {
            total += self.edge_coefficient(previous, v);
            previous = v;
        }
        total += self.edge_coefficient(previous, 0);
        total.round() as i32
    }
}

/// Time windows of the depot (index 0) and customers; travel is (n+1)^2.
#[derive(Clone, Debug, Default)]
pub struct TimeWindows {
    pub ready: Vec<f64>,
    pub due: Vec<f64>,
    pub service: Vec<f64>,
    pub travel: Vec<Vec<f64>>,
}

/// RoundCapCutsSeparatorParameters.
#[derive(Clone, Debug)]
struct Params {
    max_num_per_round: i32,
    max_kohl_set_size: i32,
    complement_form: bool,
    violation_tolerance: f64,
    max_contracted_group_size: i32,
    contraction_selection: i32,
    greedy_selection: i32,
    shrink_threshold: f64,
    use_global_pool: bool,
    max_component_removals: i32,
}

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
            violation: 0.,
            lhs: 0.,
            k: 0,
        }
    }
    fn clear(&mut self) {
        self.subset.iter_mut().for_each(|b| *b = false);
        self.violation = 0.;
        self.lhs = 0.;
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
struct Column {
    column: i32,
    value: f64,
    customers: Vec<usize>,
}
#[derive(Clone, Debug)]
struct RouteColumn {
    customers: Vec<usize>,
    key: f64,
}
#[derive(Clone, Debug, Default)]
struct GreedyExtension {
    id: usize,
    violation: f64,
    delta: f64,
    explored: bool,
}
#[derive(Clone, Debug, Default)]
struct FracCandidate {
    field_00: i32,
    field_04: i32,
    field_08: i32,
    field_10: f64,
    nodes: Vec<i32>,
    subset: Bits1024,
}
struct TwoPath {
    size: i32,
    cut_weight: f64,
    demand: f64,
    vector_18: Vec<f64>,
    vector_30: Vec<f64>,
    customers: Bits1024,
}

/// Contraction arrays: the live separation graph or temporary copies.
#[derive(Clone)]
struct Graph {
    demands: Vec<f64>,
    flow: Vec<Vec<f64>>,
    neighbours: Vec<Vec<usize>>,
    groups: Vec<Vec<usize>>,
    representative: Vec<usize>,
}

/// RoundCapCutSeparation state.
#[derive(Clone)]
pub struct Separation {
    params: Params,
    n: usize,
    num_positive_demand: usize,
    demands: Vec<f64>,
    capacity: f64,
    two_path: bool,
    arcs_between: Vec<Vec<Vec<i32>>>,
    graph: Graph,
    global_subsets: UnorderedSet,
    two_path_subsets: UnorderedSet,
    pulse: PulseDigraph,
    columns: Vec<Column>,
    num_columns: usize,
}

fn compute_violation(g: &Graph, n: usize, capacity: f64, cut: &mut Candidate, forced_k: i32) {
    let mut inside = vec![];
    let mut outside = vec![n];
    let mut demand = 0.;
    for i in 0..n {
        if g.representative[i] != i {
            continue;
        }
        if cut.test(i) {
            demand = demand + g.demands[i];
            inside.push(i);
        } else {
            outside.push(i);
        }
    }
    let k = if forced_k < 1 {
        (demand / capacity).ceil() as i32
    } else {
        forced_k
    };
    cut.k = k;
    cut.lhs = 0.;
    for &i in &inside {
        for &j in &outside {
            cut.lhs = cut.lhs + g.flow[i][j];
        }
    }
    cut.violation = (k as f64 + k as f64) - cut.lhs;
}
fn greedy_less(a: &GreedyExtension, b: &GreedyExtension, by_delta: bool) -> bool {
    const EPS: f64 = 1e-6;
    if a.explored != b.explored {
        return a.explored;
    }
    if !by_delta {
        if b.violation - EPS > a.violation {
            return true;
        }
        if a.violation > b.violation + EPS {
            return false;
        }
    }
    if b.delta - EPS > a.delta {
        return true;
    }
    if a.delta > b.delta + EPS {
        return false;
    }
    a.id < b.id
}

impl Separation {
    /// createAndPrepareSeparation for the complete customer graph: source 0,
    /// customers 1..n, sink n+1, one master variable per directed arc
    /// (id tail * (n+2) + head). `demands` excludes the depot.
    fn prepare(demands: &[f64], capacity: f64, params: Params, buckets: i32, windows: Option<&TimeWindows>) -> Result<Self> {
        if demands.iter().any(|d| !(*d >= -1e-6)) {
            return Err(invalid("demands should be non-negative"));
        }
        if !(capacity >= 1e-6) {
            return Err(invalid("capacity should be positive"));
        }
        let n = demands.len();
        let width = n + 2;
        let sink = n + 1;
        let mut representative = vec![0; n];
        let mut num_positive_demand = 0;
        for i in 0..n {
            if demands[i] <= 0. {
                representative[i] = n;
            } else {
                num_positive_demand += 1;
            }
        }
        let mut all_demands = demands.to_vec();
        all_demands.push(0.);
        let element = |v: usize| -> usize {
            if v >= 1 && v <= n && all_demands[v - 1] >= 1e-6 {
                v - 1
            } else {
                n
            }
        };
        let mut pulse = PulseDigraph::new(buckets);
        if let Some(w) = windows {
            let mut ready = vec![1e12; n + 2];
            let mut due = vec![-1e12; n + 2];
            for v in 0..width {
                let e = element(v);
                let slot = if e == n {
                    if v == sink {
                        n + 1
                    } else if v == 0 {
                        n
                    } else {
                        continue;
                    }
                } else {
                    e
                };
                let input = if v == sink { 0 } else { v };
                if w.ready[input] < ready[slot] {
                    ready[slot] = w.ready[input];
                }
                if due[slot] < w.due[input] {
                    due[slot] = w.due[input];
                }
            }
            let mut distance = vec![vec![1e12; n + 2]; n + 2];
            for tail in 0..sink {
                for head in 1..=sink {
                    if tail == head {
                        continue;
                    }
                    let (te, he) = (element(tail), element(head));
                    let mut length = w.service[tail] + w.travel[tail][if head == sink { 0 } else { head }];
                    if length < 0. {
                        length += due[n + 1];
                    }
                    if !(te == n && he == n) {
                        let from = te;
                        let to = if he == n { n + 1 } else { he };
                        if length < distance[from][to] {
                            distance[from][to] = length;
                        }
                    }
                }
            }
            pulse.set_ready_time(ready);
            pulse.set_due_date(due);
            pulse.set_distance_matrix(distance);
        }
        let mut arcs_between = vec![vec![vec![]; n + 1]; n + 1];
        for tail in 0..sink {
            for head in 1..=sink {
                if tail == head {
                    continue;
                }
                let (te, he) = (element(tail), element(head));
                let (a, b) = (te.min(he), te.max(he));
                if a == n {
                    continue;
                }
                let id = (tail * width + head) as i32;
                arcs_between[a][b].push(id);
                arcs_between[b][a].push(id);
            }
        }
        Ok(Self {
            params,
            n,
            num_positive_demand,
            demands: all_demands,
            capacity,
            two_path: windows.is_some(),
            arcs_between,
            graph: Graph {
                demands: vec![],
                flow: vec![vec![0.; n + 1]; n + 1],
                neighbours: vec![vec![]; n + 1],
                groups: vec![vec![]; n + 1],
                representative,
            },
            global_subsets: UnorderedSet::new(),
            two_path_subsets: UnorderedSet::new(),
            pulse,
            columns: vec![],
            num_columns: 0,
        })
    }

    fn add_to_global_pool(&mut self, candidates: &[Candidate]) {
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
    fn create_cut(&self, c: &Candidate) -> (i32, i32, Vec<usize>, Vec<(i32, f64)>) {
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

    fn shrinking(&self, threshold: f64, candidates: &mut CandidateSet, g: &mut Graph, live: bool) {
        let n = self.n;
        let mut edges: Vec<(usize, usize)> = vec![];
        let mut cut = Candidate::new(1, n);
        {
            let reps = if live { &g.representative } else { &self.graph.representative };
            for i in 0..n {
                if reps[i] != i {
                    continue;
                }
                for &j in &g.neighbours[i] {
                    if j < i && reps[j] == j && threshold - 1e-6 < g.flow[j][i] {
                        edges.push((j, i));
                    }
                }
            }
        }
        while !edges.is_empty() {
            let mut selected = 0;
            if self.params.contraction_selection == 1 {
                let mut best = 0.;
                for (idx, &(x, y)) in edges.iter().enumerate() {
                    let weight = g.flow[x][y];
                    if best < weight {
                        selected = idx;
                        best = weight;
                    }
                }
            }
            let (a, b) = edges.remove(selected);
            for &m in &g.groups[a] {
                cut.set(m);
            }
            for &m in &g.groups[b] {
                cut.set(m);
            }
            if live {
                compute_violation(g, n, self.capacity, &mut cut, 0);
            } else {
                compute_violation(&self.graph, n, self.capacity, &mut cut, 0);
            }
            if cut.violation > 1e-6 {
                candidates.insert(&cut);
            }
            cut.clear();
            let neighbours = g.neighbours[b].clone();
            for &nb in &neighbours {
                if nb != a {
                    let previous = g.flow[a][nb];
                    if previous == 0.0 {
                        g.neighbours[a].push(nb);
                        g.neighbours[nb].push(a);
                    }
                    g.flow[a][nb] = g.flow[b][nb] + previous;
                    g.flow[nb][a] = g.flow[nb][a] + g.flow[b][nb];
                    let weight = g.flow[a][nb];
                    if threshold - 1e-6 < weight && previous <= threshold - 1e-6 {
                        edges.push((a, nb));
                    }
                }
                g.flow[b][nb] = 0.;
                g.flow[nb][b] = 0.;
                if let Some(pos) = g.neighbours[nb].iter().position(|&x| x == b) {
                    g.neighbours[nb].remove(pos);
                }
            }
            g.neighbours[b].clear();
            let members = std::mem::take(&mut g.groups[b]);
            for m in members {
                g.groups[a].push(m);
                g.representative[m] = a;
            }
            g.demands[a] = g.demands[a] + g.demands[b];
            g.demands[b] = 0.;
            let max_group = self.params.max_contracted_group_size as usize;
            edges.retain(|&(x, y)| {
                let reps = if live { &g.representative } else { &self.graph.representative };
                !(x == n
                    || y == n
                    || reps[x] != x
                    || reps[y] != y
                    || g.groups[x].len() + g.groups[y].len() > max_group)
            });
        }
    }

    fn kohl_greedy(&mut self, c: &mut TwoPath) {
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
                if !((c.cut_weight + c.vector_30[j]) - c.vector_18[j] < 4.0 - self.params.violation_tolerance) {
                    continue;
                }
                c.size += 1;
                c.cut_weight = (c.vector_30[j] - c.vector_18[j]) + c.cut_weight;
                c.demand = c.demand + self.demands[j];
                c.customers.set(j);
                for &nb in &self.graph.neighbours[j].clone() {
                    if c.customers.test(nb) {
                        continue;
                    }
                    let w = self.graph.flow[j][nb];
                    c.vector_18[nb] = c.vector_18[nb] + w;
                    c.vector_30[nb] = c.vector_30[nb] - w;
                }
                self.kohl_greedy(c);
                for &nb in &self.graph.neighbours[j].clone() {
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
    fn kohl_two_path(&mut self, output: &mut CandidateSet) {
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
        self.kohl_greedy(&mut growth);
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
            if !(cut.violation > self.params.violation_tolerance) || self.pulse.is_subset_time_feasible(&elements) {
                continue;
            }
            cut.k = 2;
            cut.violation += 2.0;
            output.insert(&cut);
        }
    }
    fn greedy_construction(&self, output: &mut CandidateSet) {
        let n = self.n;
        let g = &self.graph;
        let mut degree = vec![0.; n];
        for i in 0..n {
            for &nb in &g.neighbours[i] {
                degree[i] = degree[i] + g.flow[i][nb];
            }
        }
        let mut visited: HashSet<Bits1024> = HashSet::new();
        let mut cut = Candidate::new(2, n);
        for seed in 0..n {
            if g.representative[seed] != seed {
                continue;
            }
            let mut roots = Bits1024::default();
            roots.set(seed);
            cut.clear();
            for &m in &g.groups[seed] {
                cut.set(m);
            }
            let mut demand = g.demands[seed];
            let mut weight = degree[seed];
            let totals = degree.clone();
            let mut joined = g.flow[seed].clone();
            let update = |e: &mut GreedyExtension, demand: f64, weight: f64, joined: &[f64], roots: &mut Bits1024, visited: &HashSet<Bits1024>| {
                let at = e.id;
                let k = ((demand + g.demands[at]) / self.capacity).ceil() as i32;
                e.violation = (k * 2) as f64 - ((weight - (joined[at] + joined[at])) + totals[at]);
                e.delta = joined[at];
                roots.set(at);
                e.explored = visited.contains(roots);
                roots.reset(at);
            };
            let mut extensions: Vec<GreedyExtension> = vec![];
            for &nb in &g.neighbours[seed] {
                if nb == n || cut.test(nb) {
                    continue;
                }
                let mut e = GreedyExtension {
                    id: nb,
                    ..Default::default()
                };
                update(&mut e, demand, weight, &joined, &mut roots, &visited);
                extensions.push(e);
            }
            while !extensions.is_empty() {
                let by_delta = self.params.greedy_selection != 0;
                stdsort::sort(&mut extensions, |a, b| greedy_less(a, b, by_delta));
                let selected = extensions.pop().unwrap();
                if selected.explored {
                    break;
                }
                let at = selected.id;
                roots.set(at);
                visited.insert(roots);
                demand = demand + g.demands[at];
                for &m in &g.groups[at] {
                    cut.set(m);
                }
                if selected.violation > self.params.violation_tolerance {
                    cut.k = (demand / self.capacity).ceil() as i32;
                    cut.violation = selected.violation;
                    output.insert(&cut);
                }
                weight = (weight - (joined[at] + joined[at])) + totals[at];
                joined[at] = 0.;
                for &nb in &g.neighbours[at] {
                    if nb == n || cut.test(nb) {
                        continue;
                    }
                    if joined[nb] == 0.0 {
                        extensions.push(GreedyExtension {
                            id: nb,
                            ..Default::default()
                        });
                    }
                    joined[nb] = joined[nb] + g.flow[at][nb];
                }
                for e in &mut extensions {
                    update(e, demand, weight, &joined, &mut roots, &visited);
                }
            }
        }
    }
    fn global_pool_greedy(&self, output: &mut CandidateSet) {
        let n = self.n;
        let g = &self.graph;
        let score = |total: f64, crossing: f64| {
            let k = (total / self.capacity).ceil();
            (k + k) - crossing
        };
        for stored in &self.global_subsets.iter() {
            let mut subset = Bits1024::default();
            for i in 0..n {
                if stored.test(i) {
                    subset.set(g.representative[i]);
                }
            }
            let mut inside = vec![0.; n];
            let mut outside = vec![0.; n];
            let mut members: Vec<(f64, usize)> = vec![];
            let mut demand = 0.;
            let mut weight = 0.;
            for i in 0..n {
                for &nb in &g.neighbours[i] {
                    if nb != n && g.representative[nb] != nb {
                        continue;
                    }
                    let flow = g.flow[i][nb];
                    if subset.test(nb) {
                        inside[i] = inside[i] + flow;
                    } else {
                        outside[i] = outside[i] + flow;
                    }
                }
                if !subset.test(i) {
                    continue;
                }
                members.push((g.demands[i], i));
                demand = demand + g.demands[i];
                weight = weight + outside[i];
            }
            members.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut violation = score(demand, weight);
            let update_flows = |vertex: usize, added: bool, inside: &mut Vec<f64>, outside: &mut Vec<f64>| {
                for &nb in &g.neighbours[vertex] {
                    if nb == n {
                        continue;
                    }
                    let flow = g.flow[vertex][nb];
                    if added {
                        inside[nb] = inside[nb] + flow;
                        outside[nb] = outside[nb] - flow;
                    } else {
                        inside[nb] = inside[nb] - flow;
                        outside[nb] = outside[nb] + flow;
                    }
                }
            };
            for &(_, at) in &members {
                let next_demand = demand - g.demands[at];
                let next_weight = (weight - outside[at]) + inside[at];
                let next_violation = score(next_demand, next_weight);
                if !(next_violation > violation + 1e-6) {
                    continue;
                }
                subset.reset(at);
                update_flows(at, false, &mut inside, &mut outside);
                demand = next_demand;
                weight = next_weight;
                violation = next_violation;
            }
            loop {
                let mut gain = 0.;
                let (mut removed, mut added) = (-1i64, -1i64);
                for i in 0..n {
                    if g.representative[i] != i {
                        continue;
                    }
                    if subset.test(i) {
                        let next_demand = demand - g.demands[i];
                        let next_weight = inside[i] + (weight - outside[i]);
                        let mut change = score(next_demand, next_weight) - violation;
                        if gain + 1e-6 < change {
                            removed = i as i64;
                            added = -1;
                            gain = change;
                        }
                        for j in 0..n {
                            if g.representative[j] != j || subset.test(j) {
                                continue;
                            }
                            let edge = g.flow[i][j];
                            let crossing = edge + edge + ((outside[j] + next_weight) - inside[j]);
                            change = score(g.demands[j] + next_demand, crossing) - violation;
                            if gain + 1e-6 < change {
                                removed = i as i64;
                                added = j as i64;
                                gain = change;
                            }
                        }
                    } else {
                        let change = score(g.demands[i] + demand, (outside[i] + weight) - inside[i]) - violation;
                        if gain + 1e-6 < change {
                            removed = -1;
                            added = i as i64;
                            gain = change;
                        }
                    }
                }
                if !(gain > 1e-6) {
                    break;
                }
                if added != -1 {
                    let at = added as usize;
                    subset.set(at);
                    demand = demand + g.demands[at];
                    weight = (weight + outside[at]) - inside[at];
                    update_flows(at, true, &mut inside, &mut outside);
                }
                if removed != -1 {
                    let at = removed as usize;
                    subset.reset(at);
                    weight = inside[at] + (weight - outside[at]);
                    if added != -1 {
                        let edge = g.flow[added as usize][at];
                        weight = edge + edge + weight;
                    }
                    demand = demand - g.demands[at];
                    update_flows(at, false, &mut inside, &mut outside);
                }
                violation = score(demand, weight);
            }
            if !(violation > 1e-6) {
                continue;
            }
            let mut cut = Candidate::new(3, n);
            for i in 0..n {
                if subset.test(i) {
                    for &m in &g.groups[i] {
                        cut.set(m);
                    }
                }
            }
            cut.k = (demand / self.capacity).ceil() as i32;
            cut.violation = violation;
            output.insert(&cut);
        }
    }
    fn min_cut_heuristic(&self, output: &mut CandidateSet) {
        let n = self.n;
        let g = &self.graph;
        let nodes: Vec<usize> = (0..g.neighbours.len() - 1).filter(|&i| !g.neighbours[i].is_empty()).collect();
        let count = nodes.len();
        let (source, sink) = (count, count + 1);
        let width = count + 2;
        let mut capacity = vec![vec![0.; width]; width];
        let mut source_capacity = vec![];
        let mut sink_capacity = vec![];
        for (i, &vertex) in nodes.iter().enumerate() {
            let depot = g.flow[n][vertex];
            let fraction = g.demands[vertex] / self.capacity;
            let mut from = (depot - (fraction + fraction)).max(1e-6);
            let mut to = ((fraction + fraction) - depot).max(1e-6);
            if from <= 1e-6 {
                from = 0.;
            }
            if to <= 1e-6 {
                to = 0.;
            }
            source_capacity.push(from);
            sink_capacity.push(to);
            capacity[source][i] = from;
            capacity[i][sink] = to;
        }
        for i in 0..count {
            for j in i + 1..count {
                capacity[i][j] = g.flow[nodes[i]][nodes[j]];
                capacity[j][i] = g.flow[nodes[j]][nodes[i]];
            }
        }
        let mut graph = MinCutDigraph::new_with_terminals(capacity, source as i32, sink as i32);
        graph.find_min_cut(source as i32, sink as i32);
        let mut cut = Candidate::new(4, n);
        let mut candidate = FracCandidate {
            field_04: count as i32,
            ..Default::default()
        };
        let include = |node: i32, candidate: &mut FracCandidate, cut: &mut Candidate| {
            candidate.field_08 += 1;
            let vertex = nodes[node as usize];
            candidate.field_10 = g.demands[vertex] + candidate.field_10;
            candidate.nodes.push(node);
            candidate.subset.set(node as usize);
            for &m in &g.groups[vertex] {
                cut.set(m);
            }
        };
        let include_sink_side = |graph: &MinCutDigraph, candidate: &mut FracCandidate, cut: &mut Candidate| {
            for i in 0..graph.sink_side.len().saturating_sub(1) {
                include(graph.sink_side[i], candidate, cut);
            }
        };
        include_sink_side(&graph, &mut candidate, &mut cut);
        compute_violation(g, n, self.capacity, &mut cut, 0);
        if cut.violation > self.params.violation_tolerance {
            output.insert(&cut);
        }
        cut.clear();
        let mut previous: Vec<FracCandidate> = vec![];
        let mut allowed = vec![true; count];
        let mut singleton = vec![false; count];
        let mut fixed_nodes: Vec<i32> = vec![];
        let mut fixed = Bits1024::default();
        let mut second_sink: i32 = -1;
        for _pass in 0..3 {
            for seed in 0..count {
                if !allowed[seed] {
                    continue;
                }
                graph.set_arc_value(seed as i32, sink as i32, 99.0);
                candidate.field_00 = seed as i32;
                fixed_to_source(&candidate, seed as i32, &mut fixed_nodes, &mut fixed, &previous);
                if singleton[seed] {
                    second_sink_node(&candidate, &mut second_sink, seed as i32, &nodes, &fixed, &g.flow);
                    if second_sink < 0 {
                        allowed[seed] = false;
                    } else {
                        graph.set_arc_value(second_sink, sink as i32, 99.0);
                    }
                }
                for &node in &fixed_nodes {
                    graph.set_arc_value(source as i32, node, 99.0);
                }
                graph.find_min_cut(source as i32, sink as i32);
                include_sink_side(&graph, &mut candidate, &mut cut);
                loop {
                    let node = expand_subset(&mut candidate, self.capacity, &nodes, &g.demands, &fixed, &g.flow);
                    if node < 0 {
                        break;
                    }
                    for &m in &g.groups[nodes[node as usize]] {
                        cut.set(m);
                    }
                }
                if candidate.field_08 == 1 {
                    singleton[seed] = true;
                }
                compute_violation(g, n, self.capacity, &mut cut, 0);
                if cut.violation > self.params.violation_tolerance {
                    output.insert(&cut);
                }
                previous.push(candidate.clone());
                candidate.field_00 = -1;
                candidate.field_08 = 0;
                candidate.field_10 = 0.;
                candidate.nodes.clear();
                candidate.subset.clear();
                cut.clear();
                graph.set_arc_value(seed as i32, sink as i32, sink_capacity[seed]);
                if second_sink >= 0 {
                    graph.set_arc_value(second_sink, sink as i32, sink_capacity[second_sink as usize]);
                    second_sink = -1;
                }
                for &node in &fixed_nodes {
                    graph.set_arc_value(source as i32, node, source_capacity[node as usize]);
                }
                fixed_nodes.clear();
                fixed.clear();
            }
        }
    }
    fn connected_components(&self, output: &mut CandidateSet) {
        let n = self.n;
        let g = &self.graph;
        let mut touching: Vec<Vec<i32>> = vec![vec![]; n];
        for column in &self.columns {
            for &customer in &column.customers {
                if self.demands[customer] > 1e-6 {
                    touching[customer].push(column.column);
                }
            }
        }
        let mut visited = Bits1024::default();
        let mut cut = Candidate::new(0, n);
        let mut remaining = Candidate::new(0, n);
        let mut column_seen = vec![false; self.num_columns];
        let mut columns: Vec<RouteColumn> = vec![];
        let add_columns = |vertex: usize, column_seen: &mut Vec<bool>, columns: &mut Vec<RouteColumn>| {
            for &id in &touching[vertex] {
                let index = id as usize;
                if column_seen[index] {
                    continue;
                }
                column_seen[index] = true;
                let column = &self.columns[index];
                columns.push(RouteColumn {
                    customers: column.customers.clone(),
                    key: column.value,
                });
            }
        };
        let evaluate = |cut: &mut Candidate, output: &mut CandidateSet| {
            compute_violation(g, n, self.capacity, cut, 0);
            if cut.violation > self.params.violation_tolerance {
                output.insert(cut);
            }
        };
        let score_columns = |cut: &Candidate, columns: &mut Vec<RouteColumn>| {
            for column in columns.iter_mut() {
                for &vertex in &column.customers {
                    if self.demands[vertex] > 1e-6 && cut.test(vertex) {
                        column.key = column.key - self.demands[vertex] / self.capacity;
                    }
                }
            }
            stdsort::sort(columns, |a, b| a.key > b.key);
        };
        let shrink = |mut count: i32, cut: &mut Candidate, columns: &Vec<RouteColumn>, output: &mut CandidateSet| {
            let limit = self.params.max_component_removals;
            let mut removed = 0;
            if limit <= 0 {
                return;
            }
            for column in columns {
                if count <= 2 {
                    break;
                }
                for &vertex in &column.customers {
                    if self.demands[vertex] > 1e-6 && cut.test(vertex) && removed < limit && count != 2 {
                        count -= 1;
                        removed += 1;
                        cut.reset(vertex);
                    }
                }
                evaluate(cut, output);
                if removed >= limit {
                    break;
                }
            }
        };
        let reconstruct = |component: &Bits1024, cut: &mut Candidate, seen: &mut Vec<bool>, columns: &mut Vec<RouteColumn>| {
            cut.clear();
            let mut count = 0;
            for i in 0..n {
                if component.test(i) {
                    count += 1;
                    cut.set(i);
                    add_columns(i, seen, columns);
                }
            }
            count
        };
        let grow = |seed: usize,
                        keep_remaining: bool,
                        cut: &mut Candidate,
                        remaining: &mut Candidate,
                        visited: &mut Bits1024,
                        seen: &mut Vec<bool>,
                        columns: &mut Vec<RouteColumn>| {
            let mut component = Bits1024::default();
            let mut queue = vec![seed];
            component.set(seed);
            cut.set(seed);
            add_columns(seed, seen, columns);
            if keep_remaining {
                remaining.set(seed);
            }
            let mut index = 0;
            while index < queue.len() {
                let vertex = queue[index];
                visited.set(vertex);
                for &nb in &g.neighbours[vertex] {
                    if !keep_remaining && nb == n {
                        continue;
                    }
                    if visited.test(nb) {
                        continue;
                    }
                    cut.set(nb);
                    queue.push(nb);
                    visited.set(nb);
                    component.set(nb);
                    add_columns(nb, seen, columns);
                    if keep_remaining {
                        remaining.set(nb);
                    }
                }
                index += 1;
            }
            (component, queue.len() as i32)
        };
        let reset_columns = |seen: &mut Vec<bool>, columns: &mut Vec<RouteColumn>| {
            seen.iter_mut().for_each(|b| *b = false);
            columns.clear();
        };
        for &seed in &g.neighbours[n].clone() {
            if visited.test(seed) {
                continue;
            }
            let (mut component, count) = grow(seed, false, &mut cut, &mut remaining, &mut visited, &mut column_seen, &mut columns);
            score_columns(&cut, &mut columns);
            evaluate(&mut cut, output);
            for i in 0..n {
                if self.demands[i] > 1e-6 {
                    component.flip(i);
                }
            }
            shrink(count, &mut cut, &columns, output);
            reset_columns(&mut column_seen, &mut columns);
            let count = reconstruct(&component, &mut cut, &mut column_seen, &mut columns);
            evaluate(&mut cut, output);
            score_columns(&cut, &mut columns);
            shrink(count, &mut cut, &columns, output);
            reset_columns(&mut column_seen, &mut columns);
            cut.clear();
        }
        let mut components = 0;
        for seed in 0..n {
            if visited.test(seed) {
                continue;
            }
            components += 1;
            let (mut component, count) = grow(seed, true, &mut cut, &mut remaining, &mut visited, &mut column_seen, &mut columns);
            score_columns(&cut, &mut columns);
            evaluate(&mut cut, output);
            component.flip_all();
            shrink(count, &mut cut, &columns, output);
            reset_columns(&mut column_seen, &mut columns);
            let count = reconstruct(&component, &mut cut, &mut column_seen, &mut columns);
            evaluate(&mut cut, output);
            score_columns(&cut, &mut columns);
            shrink(count, &mut cut, &columns, output);
            reset_columns(&mut column_seen, &mut columns);
            cut.clear();
        }
        if components > 1 {
            let mut count = 0;
            for i in 0..n {
                if remaining.test(i) {
                    count += 1;
                    add_columns(i, &mut column_seen, &mut columns);
                }
            }
            compute_violation(g, n, self.capacity, &mut remaining, 0);
            // The remaining candidate is evaluated, but the cleared first
            // candidate is inserted.
            if remaining.violation > self.params.violation_tolerance {
                output.insert(&cut);
            }
            score_columns(&cut, &mut columns);
            shrink(count, &mut cut, &columns, output);
        }
    }

    /// separate: rounded flows from the columns, the generators in order, then
    /// stable ranking by violation. Returns (rhs, type, set ids, var weights).
    fn separate(&mut self, columns: &[(Vec<i32>, f64)]) -> Vec<(i32, i32, Vec<usize>, Vec<(i32, f64)>)> {
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
        sorted[..count].iter().map(|c| self.create_cut(c)).collect()
    }
}

/// compSecondSinkNode.
fn second_sink_node(c: &FracCandidate, out: &mut i32, node: i32, nodes: &[usize], excluded: &Bits1024, x: &[Vec<f64>]) {
    if c.field_04 <= 0 {
        return;
    }
    let mut best = 1e-6;
    let mut index = -1;
    for i in 0..c.field_04 {
        if excluded.test(i as usize) || node == i {
            continue;
        }
        let v = x[nodes[i as usize]][nodes[node as usize]];
        if v > best {
            index = i;
        }
        best = v.max(best);
    }
    if index >= 0 {
        *out = index;
    }
}
/// compFixedToSourceNodes.
fn fixed_to_source(c: &FracCandidate, node: i32, out: &mut Vec<i32>, fixed: &mut Bits1024, candidates: &[FracCandidate]) {
    for o in candidates.iter().rev() {
        if o.field_08 <= 1 || !o.subset.test(node as usize) {
            continue;
        }
        let first = o.nodes[0];
        if fixed.test(first as usize) {
            continue;
        }
        if (1..o.field_08 as usize).any(|j| fixed.test(o.nodes[j] as usize)) {
            continue;
        }
        if node != o.field_00 {
            fixed.set(o.field_00 as usize);
            continue;
        }
        if node != first {
            fixed.set(first as usize);
            continue;
        }
        if let Some(&p) = o.nodes[1..o.field_08 as usize].iter().find(|&&p| p != node) {
            fixed.set(p as usize);
        }
    }
    for i in 0..c.field_04 {
        if fixed.test(i as usize) {
            out.push(i);
        }
    }
}
/// checkExpandSubset.
fn expand_subset(c: &mut FracCandidate, q: f64, nodes: &[usize], demand: &[f64], excluded: &Bits1024, x: &[Vec<f64>]) -> i32 {
    let current_k = (c.field_10 / q).ceil() as i32;
    let mut index = -1;
    let mut best = 1e-6;
    for i in 0..c.field_04 {
        if excluded.test(i as usize) || c.subset.test(i as usize) {
            continue;
        }
        let ni = nodes[i as usize];
        let row = &x[ni];
        let (mut inside, mut out) = (0.0f64, 0.0f64);
        for j in 0..c.field_04 {
            let v = row[nodes[j as usize]];
            if c.subset.test(j as usize) {
                inside = inside + v;
            } else {
                out = out + v;
            }
        }
        let mut gain = ((demand[ni] + c.field_10) / q).ceil();
        gain = gain - current_k as f64;
        gain = gain + gain;
        out = out + x.last().unwrap()[ni];
        gain = gain - out;
        let score = inside + gain;
        if score > best {
            index = i;
        }
        best = score.max(best);
    }
    if index != -1 {
        let d = demand[nodes[index as usize]];
        c.field_08 += 1;
        c.field_10 = c.field_10 + d;
        c.nodes.push(index);
        c.subset.set(index as usize);
    }
    index
}

/// Standalone separator: depot 0, customers 1..n. State persists across
/// calls and is rebuilt when the parameters or time windows change.
#[derive(Clone)]
pub struct CapacitySeparator {
    pub parameters: CapacityParameters,
    demands: Vec<f64>,
    capacity: f64,
    windows: Option<TimeWindows>,
    state: Option<(Separation, CapacityParameters, bool)>,
    feasibility: Option<Separation>,
}
impl CapacitySeparator {
    /// `demands` has n + 1 entries; index 0 (the depot) is ignored.
    pub fn new(mut demands: Vec<f64>, capacity: f64) -> Result<Self> {
        if demands.len() < 2 || demands.len() > 1024 || !capacity.is_finite() || capacity <= 0. {
            return Err(invalid("capacity dimensions or capacity"));
        }
        if demands.iter().any(|d| !d.is_finite() || *d < 0.) {
            return Err(invalid("capacity demand"));
        }
        demands[0] = 0.;
        Ok(Self {
            parameters: CapacityParameters::default(),
            demands,
            capacity,
            windows: None,
            state: None,
            feasibility: None,
        })
    }
    pub fn num_customers(&self) -> usize {
        self.demands.len() - 1
    }
    pub fn set_time_windows(&mut self, windows: TimeWindows) -> Result<()> {
        let n = self.demands.len();
        if windows.ready.len() != n || windows.due.len() != n || windows.service.len() != n || windows.travel.len() != n
        {
            return Err(invalid("time window dimensions"));
        }
        self.windows = Some(windows);
        self.state = None;
        self.feasibility = None;
        Ok(())
    }
    pub fn reset(&mut self) {
        self.state = None;
    }
    fn build(&self, use_windows: bool) -> Result<Separation> {
        let p = &self.parameters;
            if p.max_cuts_per_round == 0
                || p.max_cuts_per_round > i32::MAX as usize
                || p.candidate_limit > i32::MAX as usize
                || p.max_two_path_set_size < 1
                || p.pulse_buckets < 1
                || !p.violation_tolerance.is_finite()
                || p.violation_tolerance < 0.
                || !p.shrink_threshold.is_finite()
            {
                return Err(invalid("capacity parameters"));
            }
            let params = Params {
                max_num_per_round: p.max_cuts_per_round as i32,
                max_kohl_set_size: p.max_two_path_set_size,
                complement_form: p.complement_form,
                violation_tolerance: p.violation_tolerance,
                max_contracted_group_size: 1_000_000,
                contraction_selection: 1,
                greedy_selection: 0,
                shrink_threshold: p.shrink_threshold,
                use_global_pool: p.use_global_pool,
                max_component_removals: p.candidate_limit as i32,
            };
            let windows = if use_windows { self.windows.as_ref() } else { None };
            Separation::prepare(&self.demands[1..], self.capacity, params, p.pulse_buckets, windows)
    }
    /// The separation state, rebuilt (pool and numbering lost) when the
    /// parameters or the time-window use changed.
    fn prepared(&mut self) -> Result<&mut Separation> {
        let use_windows = self.windows.is_some() && self.parameters.use_two_path;
        let fresh = !matches!(&self.state, Some((_, p, w)) if *p == self.parameters && *w == use_windows);
        if fresh {
            let state = self.build(use_windows)?;
            self.state = Some((state, self.parameters.clone(), use_windows));
        }
        Ok(&mut self.state.as_mut().unwrap().0)
    }
    /// Whether one route can serve the customers within the time windows.
    pub fn time_feasible(&mut self, customers: &[usize]) -> Result<bool> {
        if self.windows.is_none() {
            return Ok(true);
        }
        let n = self.num_customers();
        let subset: Vec<i32> = customers
            .iter()
            .map(|&v| {
                if v < 1 || v > n {
                    Err(invalid("capacity customer"))
                } else {
                    Ok(v as i32 - 1)
                }
            })
            .collect::<Result<_>>()?;
        if let Some((state, _, true)) = &mut self.state {
            return Ok(state.pulse.is_subset_time_feasible(&subset));
        }
        if self.feasibility.is_none() {
            self.feasibility = Some(self.build(true)?);
        }
        Ok(self.feasibility.as_mut().unwrap().pulse.is_subset_time_feasible(&subset))
    }
    /// Separates on fractional columns (customer sequences, depot implicit).
    pub fn separate(&mut self, columns: &[(Vec<usize>, f64)]) -> Result<Vec<CapacityCut>> {
        if !self.parameters.enabled {
            return Ok(vec![]);
        }
        let n = self.num_customers();
        let (sink, width) = (n + 1, n + 2);
        let mut routes = vec![];
        for (customers, value) in columns {
            if !value.is_finite() || *value < 0. {
                return Err(invalid("column value"));
            }
            let mut arcs = vec![];
            let mut previous = 0;
            for &v in customers {
                if v < 1 || v > n || v == previous {
                    return Err(invalid("capacity route customer"));
                }
                arcs.push((previous * width + v) as i32);
                previous = v;
            }
            arcs.push((previous * width + sink) as i32);
            routes.push((arcs, *value));
        }
        let capacity = self.capacity;
        let demands = self.demands.clone();
        let state = self.prepared()?;
        let generated = state.separate(&routes);
        let positive = state.num_positive_demand as i32;
        let mut result = vec![];
        for (rhs, kind, sets, weights) in generated {
            let mut cut = CapacityCut {
                customers: sets.iter().map(|s| s + 1).collect(),
                other: vec![],
                kind: CapacityKind::Capacity,
                rhs,
                k: 0,
                demand: 0.,
                violation: 0.,
                edges: BTreeMap::new(),
            };
            for &c in &cut.customers {
                cut.demand += demands[c];
            }
            for (variable, weight) in weights {
                let tail = variable as usize / width;
                let mut head = variable as usize % width;
                if head == sink {
                    head = 0;
                }
                cut.edges.insert((tail.min(head), tail.max(head)), weight);
            }
            let rounded_k = (cut.demand / capacity - 1e-9).ceil() as i32;
            if kind == 0 {
                cut.k = rhs / 2;
                cut.kind = if cut.k > rounded_k {
                    CapacityKind::TwoPath
                } else {
                    CapacityKind::Capacity
                };
            } else {
                cut.kind = CapacityKind::Complement;
                let size = cut.customers.len() as i32;
                if size > positive / 2 {
                    for v in 1..=n {
                        if demands[v] > 1e-6 && cut.customers.binary_search(&v).is_err() {
                            cut.other.push(v);
                        }
                    }
                    cut.k = positive - size - rhs;
                } else {
                    cut.other = cut.customers.clone();
                    cut.k = size - rhs;
                }
            }
            let mut lhs = 0.;
            for (customers, value) in columns {
                lhs += (value * 1e8 + 0.5).floor() / 1e8 * cut.coefficient(customers) as f64;
            }
            cut.violation = if cut.greater_equal() {
                cut.rhs as f64 - lhs
            } else {
                lhs - cut.rhs as f64
            };
            result.push(cut);
        }
        Ok(result)
    }
}
