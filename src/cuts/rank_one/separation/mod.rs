//! Rank-one cut separation, ported from the C++ reference
//! `native/separation/RankOneSeparation.h`. Iteration orders, comparison
//! directions, tie breaking and unfused products are kept; Rust never contracts `a * b + c` into an FMA.
use super::local_search::LocalSearch;
use super::tuple::TupleSeparation;
use crate::cuts::shared::rbtree::CandidateSet;
use std::collections::{BTreeMap, BTreeSet};

/// Five-or-more-row numerators by class (0 packing, 1 covering) and type.
pub fn five_or_more_row_numerators(class: usize, kind: usize) -> &'static [i32] {
    const PACKING: [&[i32]; 23] = [
        &[1, 1, 1, 1, 1],
        &[1, 1, 1, 1, 1],
        &[3, 1, 1, 1, 1],
        &[3, 2, 2, 1, 1],
        &[2, 2, 1, 1, 1],
        &[3, 3, 2, 2, 1],
        &[2, 2, 2, 1, 1],
        &[3, 3, 2, 1, 1, 1],
        &[4, 2, 2, 1, 1, 1],
        &[4, 4, 2, 2, 1, 1],
        &[3, 2, 1, 1, 1, 1],
        &[4, 1, 1, 1, 1, 1],
        &[1, 1, 1, 1, 1, 1, 1],
        &[4, 4, 2, 1, 1, 1, 1],
        &[5, 2, 2, 1, 1, 1, 1],
        &[5, 5, 2, 2, 1, 1, 1],
        &[4, 2, 1, 1, 1, 1, 1],
        &[5, 1, 1, 1, 1, 1, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1],
        &[5, 5, 2, 1, 1, 1, 1, 1],
        &[6, 6, 2, 2, 1, 1, 1, 1],
        &[5, 2, 1, 1, 1, 1, 1, 1],
        &[6, 1, 1, 1, 1, 1, 1, 1],
    ];
    const COVERING: [&[i32]; 10] = [
        &[1, 1, 1, 1, 1],
        &[2, 2, 1, 1, 1],
        &[2, 2, 2, 2, 2],
        &[1, 1, 1, 1, 1],
        &[3, 2, 2, 1, 1],
        &[3, 3, 3, 3, 1],
        &[3, 3, 3, 2, 2],
        &[3, 3, 3, 1, 1],
        &[4, 2, 2, 2, 1],
        &[4, 4, 3, 3, 2],
    ];
    if class == 0 {
        PACKING[kind]
    } else {
        COVERING[kind]
    }
}
pub fn five_or_more_row_denominator(class: usize, kind: usize) -> i32 {
    const PACKING: [i32; 23] = [
        2, 3, 4, 5, 4, 4, 3, 4, 6, 5, 5, 5, 2, 5, 7, 6, 6, 6, 3, 6, 7, 7, 7,
    ];
    const COVERING: [i32; 10] = [2, 3, 3, 4, 4, 4, 4, 5, 5, 5];
    if class == 0 {
        PACKING[kind]
    } else {
        COVERING[kind]
    }
}
pub fn five_or_more_row_rhs(class: usize, kind: usize) -> f64 {
    const PACKING: [f64; 23] = [
        2., 1., 1., 1., 1., 2., 2., 2., 1., 2., 1., 1., 3., 2., 1., 2., 1., 1., 2., 2., 2., 1., 1.,
    ];
    const COVERING: [f64; 10] = [3., 3., 4., 2., 3., 4., 4., 3., 3., 4.];
    if class == 0 {
        PACKING[kind]
    } else {
        COVERING[kind]
    }
}

/// A master column as read by separation and coefficient: graph, the
/// enumerated flag (coefficients ignore the memory) and arc ids.
#[derive(Clone, Debug, Default)]
pub struct Solution {
    pub graph_id: i32,
    pub enumerated: bool,
    pub arc_ids: Vec<i32>,
}

#[derive(Clone, Debug, Default)]
pub struct VertexData {
    pub id: i32,
    pub elem_sets: Vec<i32>,
    pub packing_sets: Vec<i32>,
    pub covering_sets: Vec<i32>,
}
#[derive(Clone, Debug, Default)]
pub struct ArcData {
    pub id: i32,
    pub tail: i32,
    pub head: i32,
    pub elem_sets: Vec<i32>,
    pub packing_sets: Vec<i32>,
    pub covering_sets: Vec<i32>,
}
/// The GraphData fields rank-one separation reads.
#[derive(Clone, Debug, Default)]
pub struct GraphData {
    pub id: i32,
    pub source: i32,
    pub sink: i32,
    pub elem_set_count: i32,
    pub packing_set_count: i32,
    pub covering_set_count: i32,
    pub vertices: Vec<VertexData>,
    pub arcs: Vec<ArcData>,
    pub packing_neighbourhood: Vec<Vec<i32>>,
    pub elem_set_distances: Vec<Vec<f64>>,
}
/// PreprocessedGraphInfo incidence: each arc's sets are the sorted union of
/// its own sets and its head vertex's sets.
#[derive(Clone, Debug, Default)]
pub struct Incidence {
    pub max_arc_id: i32,
    pub max_vertex_id: i32,
    pub arc_packing_sets: Vec<Vec<i32>>,
    pub arc_covering_sets: Vec<Vec<i32>>,
    pub arc_tail: Vec<i32>,
    pub arc_head: Vec<i32>,
}
fn sorted_unique(ids: &[i32], count: i32) -> bool {
    let mut previous = -1;
    for &id in ids {
        if id <= previous || id >= count {
            return false;
        }
        previous = id;
    }
    true
}
fn union(a: &[i32], b: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else if b[j] < a[i] {
            out.push(b[j]);
            j += 1;
        } else {
            out.push(a[i]);
            i += 1;
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}
/// Cut incidence of a graph.
pub fn prepare_incidence(g: &GraphData) -> Option<Incidence> {
    if g.elem_set_count < 0
        || g.elem_set_count >= 1024
        || g.packing_set_count < 0
        || g.packing_set_count >= 1024
        || g.covering_set_count < 0
        || g.covering_set_count >= 1024
        || g.packing_set_count > g.elem_set_count
    {
        return None;
    }
    let mut vertices = BTreeMap::new();
    let mut next = Incidence {
        max_arc_id: -1,
        max_vertex_id: -1,
        ..Default::default()
    };
    for v in &g.vertices {
        if v.id < 0
            || vertices.insert(v.id, v).is_some()
            || !sorted_unique(&v.elem_sets, g.elem_set_count)
            || !sorted_unique(&v.packing_sets, g.packing_set_count)
            || !sorted_unique(&v.covering_sets, g.covering_set_count)
        {
            return None;
        }
        next.max_vertex_id = next.max_vertex_id.max(v.id);
    }
    if !vertices.contains_key(&g.source) || !vertices.contains_key(&g.sink) {
        return None;
    }
    let mut ids = BTreeSet::new();
    for a in &g.arcs {
        if a.id < 0
            || !ids.insert(a.id)
            || !vertices.contains_key(&a.tail)
            || !vertices.contains_key(&a.head)
            || !sorted_unique(&a.elem_sets, g.elem_set_count)
            || !sorted_unique(&a.packing_sets, g.packing_set_count)
            || !sorted_unique(&a.covering_sets, g.covering_set_count)
        {
            return None;
        }
        next.max_arc_id = next.max_arc_id.max(a.id);
    }
    let n = (next.max_arc_id + 1) as usize;
    next.arc_packing_sets = vec![vec![]; n];
    next.arc_covering_sets = vec![vec![]; n];
    next.arc_tail = vec![-1; n];
    next.arc_head = vec![-1; n];
    for a in &g.arcs {
        let id = a.id as usize;
        let head = vertices[&a.head];
        next.arc_packing_sets[id] = union(&head.packing_sets, &a.packing_sets);
        next.arc_covering_sets[id] = union(&head.covering_sets, &a.covering_sets);
        next.arc_tail[id] = a.tail;
        next.arc_head[id] = a.head;
    }
    Some(next)
}

/// A separated cut. `kind`: 0 packing, 1 covering,
/// 2 a third kind that rounds like packing. Memory: graph id -> vertex ids
/// (memory type 0) or arc ids (memory type 1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeparatedCut {
    pub id: i32,
    pub rhs: f64,
    pub kind: i32,
    pub num_rows: i32,
    pub class: i32,
    pub denominator: i32,
    pub numerators: Vec<i32>,
    pub set_ids: Vec<i32>,
    pub memory_type: i32,
    pub memory: BTreeMap<i32, BTreeSet<i32>>,
}

/// Rank-one separator parameters.
#[derive(Clone, Debug)]
pub struct Parameters {
    pub max_num_rows: i32,
    pub max_num_per_round: i32,
    /// Maximum cuts touching a vertex through support or memory; zero disables.
    pub max_per_vertex: usize,
    pub memory_type: i32,
    pub arc_memory_type: i32,
    pub sp_dependent_memory: bool,
    pub ls_num_iterations: i32,
    pub ls_iterations_negative: bool,
    pub neighbourhood_size: i32,
    pub max_num_triplets: i32,
    pub violation_tolerance: f64,
    pub verbosity: i32,
    pub skip_subcut_checks: bool,
    pub type_to_separate: i32,
    pub use_cov_sets_for_one_row_packing: bool,
}
impl Default for Parameters {
    fn default() -> Self {
        Self {
            max_num_rows: 0,
            max_num_per_round: 0,
            max_per_vertex: 0,
            memory_type: 0,
            arc_memory_type: 0,
            sp_dependent_memory: false,
            ls_num_iterations: 1000,
            ls_iterations_negative: false,
            neighbourhood_size: 16,
            max_num_triplets: 10000,
            violation_tolerance: 0.02,
            verbosity: -1,
            skip_subcut_checks: true,
            type_to_separate: 0,
            use_cov_sets_for_one_row_packing: false,
        }
    }
}

/// Per-column set counts used by separation.
#[derive(Clone, Debug, Default)]
pub struct ColumnInfo {
    pub graph_id: i32,
    pub value: f64,
    pub elementary: bool,
    pub enumerated: bool,
    pub arc_ids: Vec<i32>,
    pub packing_set_sequence: Vec<i32>,
    pub packing_set_count: Vec<i32>,
    pub covering_set_count: Vec<i32>,
}
impl ColumnInfo {
    /// From a route's arcs: per arc, its packing sets are counted and
    /// appended to the sequence, its covering sets counted.
    pub fn from_arcs(
        graph_id: i32,
        info: &Incidence,
        value: f64,
        arcs: Vec<i32>,
        enumerated: bool,
        n_pack: i32,
        n_cov: i32,
    ) -> Self {
        let mut c = Self {
            graph_id,
            value,
            elementary: true,
            enumerated,
            arc_ids: arcs,
            packing_set_sequence: vec![],
            packing_set_count: vec![0; n_pack as usize],
            covering_set_count: vec![0; n_cov as usize],
        };
        for &a in &c.arc_ids {
            for &p in &info.arc_packing_sets[a as usize] {
                c.packing_set_count[p as usize] += 1;
                c.packing_set_sequence.push(p);
            }
            for &q in &info.arc_covering_sets[a as usize] {
                c.covering_set_count[q as usize] += 1;
            }
        }
        c.elementary = !c.packing_set_count.iter().any(|&x| x >= 2)
            && !c.covering_set_count.iter().any(|&x| x >= 2);
        c
    }
    /// From a packing-set sequence, which is only counted.
    pub fn from_set_sequence(graph_id: i32, value: f64, sequence: &[i32], n_pack: i32) -> Self {
        let mut c = Self {
            graph_id,
            value,
            elementary: true,
            enumerated: true,
            packing_set_count: vec![0; n_pack as usize],
            ..Default::default()
        };
        for &p in sequence {
            if p < n_pack {
                c.packing_set_count[p as usize] += 1;
                if c.packing_set_count[p as usize] > 1 {
                    c.elementary = false;
                }
            }
        }
        c
    }
}

/// Per-graph set incidence and memory data.
#[derive(Clone, Debug, Default)]
pub struct GraphInfo {
    pub num_packing_sets: i32,
    pub num_covering_sets: i32,
    pub arcs_between: Vec<Vec<Vec<i32>>>,
    pub packing_set_vertices: Vec<Vec<i32>>,
    pub packing_set_arcs: Vec<Vec<i32>>,
    pub covering_set_vertices: Vec<Vec<i32>>,
    pub covering_set_arcs: Vec<Vec<i32>>,
    pub default_memory: BTreeSet<i32>,
}

pub struct Input<'a> {
    pub solutions: Vec<&'a Solution>,
    pub values: Vec<f64>,
    pub cut_memory_type: i32,
    pub solutions_are_set_sequences: bool,
    pub separation_phase: i32,
    pub active_cuts: Vec<SeparatedCut>,
}

pub type Bits = [u64; 16];
pub fn bits_any(a: &Bits, b: &Bits) -> bool {
    a.iter().zip(b).any(|(x, y)| x & y != 0)
}
pub fn bit(a: &Bits, i: usize) -> bool {
    a[i / 64] >> (i % 64) & 1 != 0
}
pub fn set_bit(a: &mut Bits, i: usize) {
    a[i / 64] |= 1 << (i % 64);
}

/// Rank-one separator state.
#[derive(Clone, Debug, Default)]
pub struct Separation {
    pub max_graph_id: i32,
    pub preprocessed: Vec<Option<Incidence>>,
    pub graph_info: Vec<Option<GraphInfo>>,
    pub params: Parameters,
    pub num_packing_sets: i32,
    pub num_covering_sets: i32,
    pub packing_distances_defined: bool,
    pub next_cut_id: i32,
    pub round_cut_counter: i32,
    pub num_separate_calls: i32,
    pub cut_memory_type: i32,
    pub current_max_rank: i32,
    pub active_cuts: Vec<SeparatedCut>,
    /// Existing registry cuts touching each (graph, vertex), excluding retired cuts.
    pub vertex_cut_load: BTreeMap<(i32, i32), usize>,
    pub packing_neighbours: Vec<Bits>,
    pub packing_neighbour_lists: Vec<Vec<i32>>,
    pub table_100: Vec<Vec<Vec<i32>>>,
    pub table_118: Vec<Vec<Vec<Vec<i32>>>>,
    pub permutations_130: Vec<Vec<i32>>,
    pub vector_148: Vec<i32>,
    pub table_160: Vec<Vec<i32>>,
    pub table_178: Vec<Vec<i32>>,
    pub permutations_190: Vec<Vec<i32>>,
    pub permutations_4rows: Vec<Vec<i32>>,
    pub permutations_5rows: Vec<Vec<i32>>,
    pub permutations_5rows_b: Vec<Vec<i32>>,
    pub covering_permutations: Vec<Vec<i32>>,
    pub neighbours_208: Vec<Bits>,
    /// Candidate arena; the candidate set holds indices into it.
    pub arena: Vec<SeparatedCut>,
}

/// All distinct permutations of a multiset, in recursive swap order.
pub fn generate_different_permuts(p: &mut Vec<i32>, out: &mut Vec<Vec<i32>>, k: usize) {
    if p.len() - 1 == k {
        out.push(p.clone());
        return;
    }
    generate_different_permuts(p, out, k + 1);
    let mut swap_with: Vec<usize> = vec![];
    for i in k + 1..p.len() {
        let v = p[i];
        if v == p[k] {
            continue;
        }
        if !swap_with.iter().any(|&j| v == p[j]) {
            swap_with.push(i);
        }
    }
    let saved = p[k];
    for j in swap_with {
        p[k] = p[j];
        p[j] = saved;
        generate_different_permuts(p, out, k + 1);
        p[j] = p[k];
    }
    p[k] = saved;
}
/// Positions of the nonzero entries of each distinct permutation.
fn generate_permut_rows(pattern: &[i32], out: &mut Vec<Vec<i32>>) {
    let mut p = pattern.to_vec();
    let mut perms = vec![];
    generate_different_permuts(&mut p, &mut perms, 0);
    out.resize(perms.len(), vec![]);
    for (i, perm) in perms.iter().enumerate() {
        for (j, &v) in perm.iter().enumerate() {
            if v != 0 {
                out[i].push(j as i32);
            }
        }
    }
}
/// Score rounding of the row separators: floor(x) / 1e8, x already + 0.5.
fn rounded_score(x: f64) -> f64 {
    x.floor() / 1e8
}

mod driver;
mod large_rows;
mod memory;
mod preparation;
mod small_rows;
