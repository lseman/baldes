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
    /// Creation order within this separator state, retained across rounds.
    pub id: usize,
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
    statistics: CapacityStatistics,
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

/// compSecondSinkNode.
fn second_sink_node(
    c: &FracCandidate,
    out: &mut i32,
    node: i32,
    nodes: &[usize],
    excluded: &Bits1024,
    x: &[Vec<f64>],
) {
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
fn fixed_to_source(
    c: &FracCandidate,
    node: i32,
    out: &mut Vec<i32>,
    fixed: &mut Bits1024,
    candidates: &[FracCandidate],
) {
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
fn expand_subset(
    c: &mut FracCandidate,
    q: f64,
    nodes: &[usize],
    demand: &[f64],
    excluded: &Bits1024,
    x: &[Vec<f64>],
) -> i32 {
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

/// Persistent separator-state counters. Reset or a parameter rebuild starts a new state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CapacityStatistics {
    /// Successful separation rounds, including rounds producing no cuts.
    pub calls: usize,
    /// Cuts returned since this state was prepared (also the next creation ID).
    pub cuts_created: usize,
    /// Input columns retained for the most recent round.
    pub columns: usize,
    /// Candidate subsets retained across rounds.
    pub global_subsets: usize,
}

mod components;
mod greedy;
mod min_cut;
mod preparation;
mod separation;
mod shrinking;
mod two_path;

mod adapter;
