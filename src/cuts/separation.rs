//! Rank-one cut separation, ported from the C++ reference
//! `native/separation/RankOneSeparation.h`. Iteration orders, comparison
//! directions, tie breaking and unfused products are kept; Rust never contracts `a * b + c` into an FMA.
use super::local_search::LocalSearch;
use super::rbtree::CandidateSet;
use super::tuple::TupleSeparation;
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
pub struct NativeCut {
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
    pub active_cuts: Vec<NativeCut>,
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
    pub active_cuts: Vec<NativeCut>,
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
    pub arena: Vec<NativeCut>,
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

impl Separation {
    /// Empty separator with the given parameters.
    pub fn new(params: Parameters) -> Self {
        Self {
            max_graph_id: -1,
            params,
            cut_memory_type: 1,
            current_max_rank: 1,
            ..Default::default()
        }
    }
    pub fn incidence(&self, graph: i32) -> &Incidence {
        self.preprocessed[graph as usize].as_ref().unwrap()
    }
    fn candidate(
        &mut self,
        rhs: f64,
        kind: i32,
        rows: i32,
        class: i32,
        denominator: i32,
        numerators: Vec<i32>,
        sets: Vec<i32>,
    ) -> usize {
        let id = self.round_cut_counter;
        self.round_cut_counter += 1;
        self.push_candidate(NativeCut {
            id,
            rhs,
            kind,
            num_rows: rows,
            class,
            denominator,
            numerators,
            set_ids: sets,
            memory_type: self.cut_memory_type,
            memory: BTreeMap::new(),
        })
    }
    pub fn push_candidate(&mut self, cut: NativeCut) -> usize {
        self.arena.push(cut);
        self.arena.len() - 1
    }
    pub fn insert_candidate(&mut self, cuts: &mut CandidateSet, score: f64, cut: usize) {
        let id = self.arena[cut].id;
        cuts.insert(score, id, cut);
    }

    /// Per-graph set incidence, arc lists and default memory.
    fn prepare_graph_info(&self, g: &GraphData, build_arc_lists: bool, all_vertices_memory: bool) -> Option<GraphInfo> {
        let info = self.preprocessed.get(g.id as usize)?.as_ref()?;
        let mut gi = GraphInfo {
            num_packing_sets: g.packing_set_count,
            num_covering_sets: g.covering_set_count,
            packing_set_vertices: vec![vec![]; g.packing_set_count as usize],
            covering_set_vertices: vec![vec![]; g.covering_set_count as usize],
            packing_set_arcs: vec![vec![]; g.packing_set_count as usize],
            covering_set_arcs: vec![vec![]; g.covering_set_count as usize],
            ..Default::default()
        };
        for v in &g.vertices {
            for &p in &v.packing_sets {
                gi.packing_set_vertices[p as usize].push(v.id);
            }
            for &q in &v.covering_sets {
                gi.covering_set_vertices[q as usize].push(v.id);
            }
        }
        if all_vertices_memory {
            for v in &g.vertices {
                if v.id != g.sink && v.id != g.source {
                    gi.default_memory.insert(v.id);
                }
            }
        }
        if build_arc_lists {
            let n = (info.max_vertex_id + 1) as usize;
            gi.arcs_between = vec![vec![vec![]; n]; n];
        }
        for a in &g.arcs {
            for &p in &a.packing_sets {
                gi.packing_set_arcs[p as usize].push(a.id);
            }
            for &q in &a.covering_sets {
                gi.covering_set_arcs[q as usize].push(a.id);
            }
            if !build_arc_lists {
                continue;
            }
            let (t, h) = (a.tail as usize, a.head as usize);
            gi.arcs_between[t][h].push(a.id);
            gi.arcs_between[h][t].push(a.id);
            if a.head == g.sink && a.head != g.source {
                gi.arcs_between[g.source as usize][h].push(a.id);
            }
            if a.tail == g.source && a.tail != g.sink {
                gi.arcs_between[t][g.sink as usize].push(a.id);
            }
        }
        let includes = |big: &[i32], small: &[i32]| small.iter().all(|x| big.binary_search(x).is_ok());
        for v in &g.vertices {
            if !includes(&v.elem_sets, &v.packing_sets) || !includes(&v.elem_sets, &v.covering_sets) {
                return None;
            }
        }
        for a in &g.arcs {
            if !includes(&a.elem_sets, &a.packing_sets) || !includes(&a.elem_sets, &a.covering_sets) {
                return None;
            }
        }
        Some(gi)
    }
    /// Index permutations of the 4- and 5-row packing sub-cut checks.
    fn build_packing_permut_vectors(&mut self) {
        generate_permut_rows(&[1, 1, 1, 0], &mut self.permutations_4rows);
        if self.params.max_num_rows > 4 {
            generate_permut_rows(&[1, 1, 1, 0, 0], &mut self.permutations_5rows);
            generate_permut_rows(&[1, 1, 1, 1, 0], &mut self.permutations_5rows_b);
            let mut p = vec![2, 1, 1, 1, 0];
            let mut perms = vec![];
            generate_different_permuts(&mut p, &mut perms, 0);
            self.permutations_190.resize(perms.len(), vec![0]);
            for (i, perm) in perms.iter().enumerate() {
                for j in 0..=4 {
                    match perm[j] {
                        1 => self.permutations_190[i].push(j as i32),
                        2 => self.permutations_190[i][0] = j as i32,
                        _ => {}
                    }
                }
            }
        }
    }
    /// Index permutations of the covering sub-cut checks.
    fn build_covering_permut_vectors(&mut self) {
        let mut p = vec![1, 2, 2, 2, 0];
        let mut perms = vec![];
        generate_different_permuts(&mut p, &mut perms, 0);
        self.covering_permutations.resize(perms.len(), vec![0]);
        for (i, perm) in perms.iter().enumerate() {
            for j in 0..=4 {
                match perm[j] {
                    2 => self.covering_permutations[i].push(j as i32),
                    1 => self.covering_permutations[i][0] = j as i32,
                    _ => {}
                }
            }
        }
    }
    /// Packing-set neighbourhoods from the graphs or their distances.
    fn calculate_packing_neighbours(&mut self, graphs: &[&GraphData]) {
        let n = self.num_packing_sets as usize;
        self.packing_neighbours = vec![[0; 16]; n];
        self.packing_distances_defined = false;
        for g in graphs {
            for p in 0..g.packing_set_count as usize {
                let Some(list) = g.packing_neighbourhood.get(p) else {
                    continue;
                };
                if list.is_empty() {
                    continue;
                }
                self.packing_distances_defined = true;
                for &e in list {
                    set_bit(&mut self.packing_neighbours[p], e as usize);
                }
            }
        }
        if !self.packing_distances_defined {
            let mut dist = vec![vec![0.0f64; n]; n];
            let mut count = 0;
            for g in graphs {
                if g.elem_set_distances.is_empty() {
                    continue;
                }
                let (c, c1) = (count as f64, (count + 1) as f64);
                for i in 0..n {
                    for j in 0..n {
                        let d = &mut dist[i][j];
                        *d = (*d * c + g.elem_set_distances[i][j]) / c1;
                    }
                }
                count += 1;
            }
            if count != 0 {
                let k = self.num_packing_sets.min(self.params.neighbourhood_size) as usize;
                for i in 0..n {
                    let mut order: Vec<(f64, usize)> = (0..n).map(|j| (dist[i][j], j)).collect();
                    order.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    if order.is_empty() {
                        continue;
                    }
                    let bits = &mut self.packing_neighbours[i];
                    set_bit(bits, order[0].1);
                    if k == 1 {
                        continue;
                    }
                    let mut t = 1;
                    while t != order.len() {
                        set_bit(bits, order[t].1);
                        t += 1;
                        if t == k {
                            break;
                        }
                    }
                }
                self.packing_distances_defined = true;
            }
            if !self.packing_distances_defined {
                return;
            }
        }
        if self.params.ls_num_iterations > 0 {
            self.neighbours_208 = vec![[0; 16]; n];
            for i in 0..n {
                for j in 0..n {
                    if i != j && bits_any(&self.packing_neighbours[i], &self.packing_neighbours[j]) {
                        set_bit(&mut self.neighbours_208[i], j);
                    }
                }
            }
        }
    }
    fn meet(&self, a: i32, b: i32) -> bool {
        bits_any(&self.packing_neighbours[a as usize], &self.packing_neighbours[b as usize])
    }
    /// Neighbourhood tuples and coefficient tables of 4/5-row enumeration.
    fn build_4_and_5_row_structures(&mut self) {
        let build = self.params.max_num_triplets <= 0
            && (self.params.ls_num_iterations == 0 || self.params.ls_iterations_negative);
        if !build {
            return;
        }
        let n = self.num_packing_sets as usize;
        self.packing_neighbour_lists = vec![vec![]; n];
        self.table_100 = vec![vec![vec![]; n]; n];
        self.table_118 = vec![vec![vec![vec![]; n]; n]; n];
        let m = self.num_packing_sets;
        for i in 0..m {
            for j in i + 1..m {
                if !self.meet(i, j) {
                    continue;
                }
                self.packing_neighbour_lists[i as usize].push(j);
                for k in j + 1..m {
                    if !self.meet(i, k) || !self.meet(j, k) {
                        continue;
                    }
                    self.table_100[i as usize][j as usize].push(k);
                    for l in k + 1..m {
                        if self.meet(i, l) && self.meet(j, l) && self.meet(k, l) {
                            self.table_118[i as usize][j as usize][k as usize].push(l);
                        }
                    }
                }
            }
        }
        for kind in 0..=6 {
            let mut numerators = five_or_more_row_numerators(0, kind).to_vec();
            generate_different_permuts(&mut numerators, &mut self.permutations_130, 0);
            self.vector_148.resize(self.permutations_130.len(), kind as i32);
        }
        self.table_160 = vec![vec![]; 32];
        self.table_178 = vec![vec![]; 32];
        for mask in 1u32..32 {
            for (i, perm) in self.permutations_130.iter().enumerate() {
                let mut sum = 0;
                if mask & 1 != 0 {
                    sum = perm[0];
                }
                if mask & 2 != 0 {
                    sum += perm[1];
                }
                if mask & 4 != 0 {
                    sum += perm[2];
                }
                if mask & 8 != 0 {
                    sum += perm[3];
                }
                if mask >> 4 != 0 {
                    sum += perm[4];
                }
                let denominator = five_or_more_row_denominator(0, self.vector_148[i] as usize);
                if denominator <= sum {
                    self.table_160[mask as usize].push(sum / denominator);
                    self.table_178[mask as usize].push(i as i32);
                }
            }
        }
    }
    /// Prepares the separator for the graphs. Incidences are prepared by the caller.
    pub fn prepare(&mut self, graphs: &[&GraphData], incidences: Vec<(i32, Incidence)>) -> bool {
        if graphs.is_empty() {
            return false;
        }
        for g in graphs {
            self.max_graph_id = self.max_graph_id.max(g.id);
            self.num_packing_sets = self.num_packing_sets.max(g.packing_set_count);
            self.num_covering_sets = self.num_covering_sets.max(g.covering_set_count);
        }
        let size = (self.max_graph_id + 1) as usize;
        self.preprocessed = vec![None; size];
        self.graph_info = vec![None; size];
        for (id, incidence) in incidences {
            self.preprocessed[id as usize] = Some(incidence);
        }
        if graphs.iter().any(|g| self.preprocessed[g.id as usize].is_none()) {
            return false;
        }
        if self.params.max_num_rows > 3 {
            if self.num_packing_sets > 0 {
                self.build_packing_permut_vectors();
                self.calculate_packing_neighbours(graphs);
                if !self.packing_distances_defined && self.params.max_num_triplets <= 0 {
                    self.params.ls_num_iterations = 0;
                    self.params.max_num_rows = 4;
                    return false;
                }
                self.build_4_and_5_row_structures();
            }
            if self.num_covering_sets != 0 {
                self.build_covering_permut_vectors();
            }
        }
        if !self.params.sp_dependent_memory {
            if graphs.len() == 1 {
                self.params.sp_dependent_memory = true;
            } else {
                let first: BTreeSet<_> = graphs[0].vertices.iter().map(|v| v.id).collect();
                for g in &graphs[1..] {
                    let other: BTreeSet<_> = g.vertices.iter().map(|v| v.id).collect();
                    if first != other {
                        self.params.sp_dependent_memory = true;
                        break;
                    }
                }
            }
        }
        let build_arc_lists = (self.params.memory_type as u32) <= 1;
        let all_vertices_memory = self.params.memory_type == 3;
        for g in graphs {
            let Some(gi) = self.prepare_graph_info(g, build_arc_lists, all_vertices_memory) else {
                return false;
            };
            self.graph_info[g.id as usize] = Some(gi);
        }
        true
    }

    /// Coefficient of a column in a cut (memory-filtered state walk).
    pub fn coefficient(&self, sol: &Solution, cut: &NativeCut) -> f64 {
        let graph = sol.graph_id;
        let mem = cut.memory.get(&graph);
        if sol.arc_ids.is_empty() {
            return 0.0;
        }
        let unfiltered = sol.enumerated;
        if mem.is_none() && !unfiltered {
            return 0.0;
        }
        let info = self.incidence(graph);
        let packing_like = (cut.kind as u32) & !2u32 == 0;
        let (mut state, mut count) = (0i32, 0i32);
        for &arc in &sol.arc_ids {
            if !unfiltered {
                let mem = mem.unwrap();
                if cut.memory_type == 1 {
                    if !mem.contains(&arc) {
                        state = 0;
                    }
                } else if cut.memory_type == 0 && !mem.contains(&info.arc_tail[arc as usize]) {
                    state = 0;
                }
            }
            let sets = if cut.kind == 0 {
                &info.arc_packing_sets[arc as usize]
            } else {
                &info.arc_covering_sets[arc as usize]
            };
            for &set in sets {
                for r in 0..cut.num_rows as usize {
                    if set == cut.set_ids[r] {
                        state += cut.numerators[r];
                        break;
                    }
                }
                if packing_like {
                    if cut.denominator <= state {
                        count += 1;
                        state -= cut.denominator;
                    }
                } else if state < 0 {
                    count += 1;
                    state -= cut.denominator;
                }
            }
        }
        count as f64
    }

    /// One-row packing cuts.
    fn one_row_pack(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        let cov = self.params.use_cov_sets_for_one_row_packing;
        let n = if cov {
            self.num_covering_sets
        } else {
            self.num_packing_sets
        };
        let kind = if cov { 2 } else { 0 };
        for set in 0..n {
            let mut lhs = 0.0;
            for col in columns {
                let c = if cov {
                    col.covering_set_count[set as usize]
                } else {
                    col.packing_set_count[set as usize]
                };
                if c > 1 {
                    lhs = ((c >> 1) as f64) * col.value + lhs;
                }
            }
            if !(lhs > self.params.violation_tolerance) {
                continue;
            }
            let cut = self.candidate(0.0, kind, 1, -1, 2, vec![1], vec![set]);
            self.insert_candidate(cuts, rounded_score(0.5 - lhs * 1e8), cut);
        }
    }
    /// One-row covering cuts.
    fn one_row_cov(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        for set in 0..self.num_covering_sets {
            let mut lhs = 0.0;
            for col in columns {
                let c = col.covering_set_count[set as usize];
                if c > 0 {
                    lhs = (((c + 1) >> 1) as f64) * col.value + lhs;
                }
            }
            if !(1.0 - lhs > self.params.violation_tolerance) {
                continue;
            }
            let cut = self.candidate(1.0, 1, 1, -1, -2, vec![-1], vec![set]);
            self.insert_candidate(cuts, rounded_score((lhs - 1.0) * 1e8 + 0.5), cut);
        }
    }
    /// Three-row packing cuts.
    fn three_rows_pack(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        let n = self.num_packing_sets.max(0) as usize;
        let mut pv = vec![vec![0.0f64; n]; n];
        for col in columns {
            if n == 0 {
                continue;
            }
            let present: Vec<usize> = (0..n).filter(|&p| col.packing_set_count[p] != 0).collect();
            for a in 1..present.len() {
                for b in 0..a {
                    let (pa, pb) = (present[a], present[b]);
                    pv[pb][pa] += col.value;
                    pv[pa][pb] += col.value;
                }
            }
        }
        for i in 0..n {
            for j in i + 1..n {
                let vij = pv[i][j];
                if vij == 0.0 {
                    continue;
                }
                for k in j + 1..n {
                    let approx = vij + pv[i][k] + pv[j][k];
                    if !(approx - 1.0 > self.params.violation_tolerance) {
                        continue;
                    }
                    let mut lhs = 0.0;
                    for col in columns {
                        let pc = &col.packing_set_count;
                        let (a, b, c) = (pc[i] != 0, pc[j] != 0, pc[k] != 0);
                        if if a { b || c } else { b && c } {
                            lhs += col.value;
                        }
                    }
                    if !(lhs - 1.0 > self.params.violation_tolerance) {
                        continue;
                    }
                    let cut = self.candidate(
                        1.0,
                        0,
                        3,
                        -1,
                        2,
                        vec![1, 1, 1],
                        vec![i as i32, j as i32, k as i32],
                    );
                    self.insert_candidate(cuts, rounded_score((1.0 - lhs) * 1e8 + 0.5), cut);
                }
            }
        }
    }
    /// Three-row covering cuts.
    fn three_rows_cov(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        let n = self.num_covering_sets.max(0) as usize;
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    let mut lhs = 0.0;
                    for col in columns {
                        let cc = &col.covering_set_count;
                        let sum = cc[j] + cc[i] + cc[k] + 1;
                        lhs = ((sum / 2) as f64) * col.value + lhs;
                    }
                    if !(2.0 - lhs > self.params.violation_tolerance) {
                        continue;
                    }
                    let cut = self.candidate(
                        2.0,
                        1,
                        3,
                        -1,
                        -2,
                        vec![-1, -1, -1],
                        vec![i as i32, j as i32, k as i32],
                    );
                    self.insert_candidate(cuts, rounded_score((lhs - 2.0) * 1e8 + 0.5), cut);
                }
            }
        }
    }

    /// Adds a memory set to the cut (and parallel arcs for arc memory).
    fn augment_memory(&self, cut: &mut NativeCut, graph: i32, mem: &BTreeSet<i32>) {
        if mem.is_empty() {
            return;
        }
        if self.params.sp_dependent_memory {
            let m = cut.memory.entry(graph).or_default();
            m.extend(mem.iter().copied());
            if self.cut_memory_type == 1 {
                let info = self.incidence(graph);
                let gi = self.graph_info[graph as usize].as_ref().unwrap();
                for &arc in mem {
                    let parallel = &gi.arcs_between[info.arc_tail[arc as usize] as usize]
                        [info.arc_head[arc as usize] as usize];
                    m.extend(parallel.iter().copied());
                }
            }
            return;
        }
        if self.cut_memory_type == 1 {
            let info = self.incidence(graph);
            for &arc in mem {
                let (tail, head) = (info.arc_tail[arc as usize] as usize, info.arc_head[arc as usize] as usize);
                let graphs: Vec<i32> = cut.memory.keys().copied().collect();
                for node in graphs {
                    let Some(gi) = self.graph_info.get(node as usize).and_then(Option::as_ref) else {
                        continue;
                    };
                    let parallel = gi.arcs_between[tail][head].clone();
                    cut.memory.entry(node).or_default().extend(parallel);
                }
            }
        } else {
            for m in cut.memory.values_mut() {
                m.extend(mem.iter().copied());
            }
        }
    }
    /// Builds the cut memory from the columns. False only for a one-row covering cut
    /// whose memory stayed empty.
    pub fn set_cut_memory(&self, cut: &mut NativeCut, columns: &[ColumnInfo]) -> bool {
        cut.memory.clear();
        for g in 0..=self.max_graph_id {
            if self.preprocessed[g as usize].is_some() {
                cut.memory.insert(g, BTreeSet::new());
            }
        }
        if cut.memory_type == 2 {
            cut.memory_type = 0;
            for g in 0..=self.max_graph_id {
                if let Some(gi) = &self.graph_info[g as usize] {
                    cut.memory.insert(g, gi.default_memory.clone());
                }
            }
            return true;
        }
        let n = if cut.kind == 0 {
            self.num_packing_sets
        } else {
            self.num_covering_sets
        };
        let mut coef = vec![0i32; n as usize];
        for r in 0..cut.num_rows as usize {
            coef[cut.set_ids[r] as usize] = cut.numerators[r];
        }
        for col in columns {
            if col.enumerated {
                continue;
            }
            let touched = if (cut.kind - 1) as u32 <= 1 {
                (0..cut.num_rows as usize).any(|r| col.covering_set_count[cut.set_ids[r] as usize] != 0)
            } else {
                let mut sum = 0;
                for r in 0..cut.num_rows as usize {
                    sum += col.packing_set_count[cut.set_ids[r] as usize] * cut.numerators[r];
                }
                cut.denominator <= sum
            };
            if !touched || col.arc_ids.is_empty() {
                continue;
            }
            let info = self.incidence(col.graph_id);
            let mut mem = BTreeSet::new();
            let mut state = 0;
            let mut it = 0;
            loop {
                let arc = col.arc_ids[it] as usize;
                let sets = if cut.kind == 0 {
                    &info.arc_packing_sets[arc]
                } else {
                    &info.arc_covering_sets[arc]
                };
                for &set in sets {
                    state += coef[set as usize];
                    if cut.kind == 0 {
                        if cut.denominator <= state {
                            self.augment_memory(cut, col.graph_id, &mem);
                            mem.clear();
                            state -= cut.denominator;
                        }
                    } else if state >= 0 {
                        self.augment_memory(cut, col.graph_id, &mem);
                        mem.clear();
                    } else {
                        state -= cut.denominator;
                    }
                }
                it += 1;
                if it == col.arc_ids.len() {
                    break;
                }
                if state > 0 {
                    let next = col.arc_ids[it];
                    if cut.memory_type == 1 {
                        mem.insert(next);
                    } else {
                        mem.insert(info.arc_tail[next as usize]);
                    }
                }
            }
        }
        if cut.memory_type != 1 {
            for g in 0..=self.max_graph_id {
                if self.preprocessed[g as usize].is_none() {
                    continue;
                }
                let gi = self.graph_info[g as usize].as_ref().unwrap();
                if self.params.sp_dependent_memory && cut.memory[&g].is_empty() {
                    continue;
                }
                for r in 0..cut.num_rows as usize {
                    let set = cut.set_ids[r] as usize;
                    let list = if cut.kind == 0 {
                        &gi.packing_set_vertices[set]
                    } else {
                        &gi.covering_set_vertices[set]
                    };
                    let tmp: BTreeSet<i32> = list.iter().copied().collect();
                    self.augment_memory(cut, g, &tmp);
                }
            }
        } else {
            for g in 0..=self.max_graph_id {
                let Some(info) = &self.preprocessed[g as usize] else {
                    continue;
                };
                let gi = self.graph_info[g as usize].as_ref().unwrap();
                if self.params.sp_dependent_memory && cut.memory[&g].is_empty() {
                    continue;
                }
                let current = cut.memory[&g].clone();
                for arc in current {
                    let tail = info.arc_tail[arc as usize] as usize;
                    let list = &gi.arcs_between[tail][tail];
                    let tmp: BTreeSet<i32> = list.first().copied().into_iter().collect();
                    self.augment_memory(cut, g, &tmp);
                }
                let arc_memory_type = self.params.arc_memory_type;
                if arc_memory_type == 0 {
                    continue;
                }
                let mut heads = BTreeSet::new();
                let kind = cut.kind;
                let set_arcs = |set: usize| {
                    if kind == 0 {
                        gi.packing_set_arcs[set].clone()
                    } else {
                        gi.covering_set_arcs[set].clone()
                    }
                };
                if arc_memory_type == 1 {
                    for r in 0..cut.num_rows as usize {
                        for a in set_arcs(cut.set_ids[r] as usize) {
                            heads.insert(info.arc_head[a as usize]);
                        }
                    }
                }
                let mut vertices = vec![];
                for r in 0..cut.num_rows as usize {
                    let set = cut.set_ids[r] as usize;
                    for a in set_arcs(set) {
                        if arc_memory_type != 2 && !heads.contains(&info.arc_tail[a as usize]) {
                            continue;
                        }
                        self.augment_memory(cut, g, &BTreeSet::from([a]));
                    }
                    let vs = if cut.kind == 0 {
                        &gi.packing_set_vertices[set]
                    } else {
                        &gi.covering_set_vertices[set]
                    };
                    vertices.extend_from_slice(vs);
                }
                for &u in &vertices {
                    for &v in &vertices {
                        let list = &gi.arcs_between[u as usize][v as usize];
                        let tmp: BTreeSet<i32> = list.first().copied().into_iter().collect();
                        self.augment_memory(cut, g, &tmp);
                    }
                }
            }
            for other in &self.active_cuts {
                if cut.num_rows != other.num_rows || (cut.num_rows > 4 && cut.class != other.class) {
                    continue;
                }
                let same = (0..cut.num_rows as usize)
                    .all(|r| cut.set_ids[r] == other.set_ids[r] && cut.numerators[r] == other.numerators[r]);
                if !same {
                    continue;
                }
                for (graph, memory) in cut.memory.iter_mut() {
                    if let Some(o) = other.memory.get(graph) {
                        memory.extend(o.iter().copied());
                    }
                }
            }
        }
        if cut.kind == 1 && cut.num_rows == 1 {
            return (0..=self.max_graph_id)
                .any(|g| self.preprocessed[g as usize].is_some() && !cut.memory[&g].is_empty());
        }
        true
    }
    /// Keeps the best `max_num` candidates, with final ids and memories.
    fn insert_best(
        &mut self,
        columns: &[ColumnInfo],
        cuts: &mut CandidateSet,
        max_num: i32,
        sort_and_set_memory: bool,
        insert: bool,
        out: &mut Vec<NativeCut>,
    ) {
        if !insert {
            cuts.clear();
            return;
        }
        let mut added = 0;
        for (_, handle) in cuts.ordered() {
            if added >= max_num {
                continue;
            }
            let mut cut = self.arena[handle].clone();
            cut.id = self.next_cut_id;
            self.next_cut_id += 1;
            if sort_and_set_memory {
                let mut rows: Vec<(i32, i32)> = (0..cut.num_rows as usize)
                    .map(|r| (cut.set_ids[r], cut.numerators[r]))
                    .collect();
                rows.sort();
                for (r, (s, n)) in rows.into_iter().enumerate() {
                    cut.set_ids[r] = s;
                    cut.numerators[r] = n;
                }
                if !self.set_cut_memory(&mut cut, columns) {
                    continue;
                }
            }
            out.push(cut);
            added += 1;
        }
        cuts.clear();
    }

    /// Four-row packing cuts by neighbourhood enumeration.
    fn four_rows_pack(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        for i in 0..self.num_packing_sets {
            for j in self.packing_neighbour_lists[i as usize].clone() {
                for k in self.table_100[i as usize][j as usize].clone() {
                    for l in self.table_118[i as usize][j as usize][k as usize].clone() {
                        let sets = [i, j, k, l];
                        let mut best = 0.0f64;
                        let mut best_pos = -1i32;
                        for t in 0..4 {
                            let mut lhs = 0.0;
                            for col in columns {
                                let nz = |q: i32| (col.packing_set_count[q as usize] != 0) as u32;
                                let cnt = nz(i) + nz(j) + nz(k) + nz(l) + nz(sets[t]);
                                lhs = lhs + ((cnt / 3) as f64) * col.value;
                            }
                            if lhs > best {
                                best_pos = t as i32;
                            }
                            best = if lhs > best { lhs } else { best };
                        }
                        if !(best - 1.0 > self.params.violation_tolerance) {
                            continue;
                        }
                        if !self.params.skip_subcut_checks
                            && super::subcuts::three_rows_pack_violated(self, columns, &sets, &self.permutations_4rows)
                        {
                            continue;
                        }
                        let mut numerators = vec![1, 1, 1, 1];
                        numerators[best_pos as usize] += 1;
                        let cut = self.candidate(1.0, 0, 4, -1, 3, numerators, sets.to_vec());
                        self.insert_candidate(cuts, rounded_score((1.0 - best) * 1e8 + 0.5), cut);
                    }
                }
            }
        }
    }
    /// Five-row packing cuts by neighbourhood enumeration.
    fn five_rows_pack(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        let mut lhs = vec![0.0f64; self.permutations_130.len()];
        for i in 0..self.num_packing_sets {
            let neighbours = self.packing_neighbour_lists[i as usize].clone();
            let mut column_ids = vec![];
            for (c, col) in columns.iter().enumerate() {
                let pc = &col.packing_set_count;
                let mut seen = pc[i as usize] != 0;
                for &j in &neighbours {
                    if pc[j as usize] == 0 {
                        continue;
                    }
                    if seen {
                        column_ids.push(c as i32);
                        break;
                    }
                    seen = true;
                }
            }
            for &j in &neighbours {
                for k in self.table_100[i as usize][j as usize].clone() {
                    for l in self.table_118[i as usize][j as usize][k as usize].clone() {
                        for m in l + 1..self.num_packing_sets {
                            if !self.meet(i, m) || !self.meet(j, m) || !self.meet(k, m) || !self.meet(l, m) {
                                continue;
                            }
                            lhs.iter_mut().for_each(|v| *v = 0.0);
                            for &c in &column_ids {
                                let col = &columns[c as usize];
                                let pc = &col.packing_set_count;
                                let mut mask = (pc[i as usize] != 0) as usize;
                                if pc[j as usize] != 0 {
                                    mask += 2;
                                }
                                if pc[k as usize] != 0 {
                                    mask += 4;
                                }
                                if pc[l as usize] != 0 {
                                    mask += 8;
                                }
                                if pc[m as usize] != 0 {
                                    mask += 16;
                                }
                                let coefs = &self.table_160[mask];
                                let perms = &self.table_178[mask];
                                for t in 0..coefs.len() {
                                    let v = &mut lhs[perms[t] as usize];
                                    *v = (coefs[t] as f64) * col.value + *v;
                                }
                            }
                            let mut best = 0.0f64;
                            let mut best_perm = -1i64;
                            for p in 0..self.permutations_130.len() {
                                let kind = self.vector_148[p] as usize;
                                let v = lhs[p] / five_or_more_row_rhs(0, kind) - 1.0;
                                if v > best {
                                    best_perm = p as i64;
                                }
                                best = if v > best { v } else { best };
                            }
                            if !(best > self.params.violation_tolerance) {
                                continue;
                            }
                            let sets = vec![i, j, k, l, m];
                            if !self.params.skip_subcut_checks {
                                if super::subcuts::three_rows_pack_violated(self, columns, &sets, &self.permutations_5rows) {
                                    continue;
                                }
                                if super::subcuts::four_rows_pack_violated(self, columns, &sets, &column_ids) {
                                    continue;
                                }
                            }
                            let kind = self.vector_148[best_perm as usize] as usize;
                            let numerators = self.permutations_130[best_perm as usize][..5].to_vec();
                            let id = self.round_cut_counter;
                            self.round_cut_counter += 1;
                            let cut = self.push_candidate(NativeCut {
                                id,
                                rhs: five_or_more_row_rhs(0, kind),
                                kind: 0,
                                num_rows: 5,
                                class: kind as i32,
                                denominator: five_or_more_row_denominator(0, kind),
                                numerators,
                                set_ids: sets,
                                memory_type: self.cut_memory_type,
                                memory: BTreeMap::new(),
                            });
                            self.insert_candidate(cuts, rounded_score(0.5 - best * 1e8), cut);
                        }
                    }
                }
            }
        }
    }
    fn raise_rank(&mut self, rank: i32) {
        let mut r = if self.current_max_rank > rank - 1 {
            self.current_max_rank
        } else {
            rank
        };
        if self.params.max_num_rows < r {
            r = self.params.max_num_rows;
        }
        self.current_max_rank = r;
    }

    /// Separates packing then covering cuts rank by rank. Returned cuts carry
    /// final ids and memories.
    pub fn separate(&mut self, input: &Input) -> Option<Vec<NativeCut>> {
        let mut out = vec![];
        if self.num_packing_sets == 0 && self.num_covering_sets == 0 {
            return Some(out);
        }
        if input.solutions.len() != input.values.len() {
            return None;
        }
        self.num_separate_calls += 1;
        self.round_cut_counter = 0;
        self.arena.clear();
        self.current_max_rank = input.separation_phase;
        self.cut_memory_type = input.cut_memory_type;
        if self.cut_memory_type == 1 {
            self.active_cuts = input.active_cuts.clone();
        }
        let mut columns = vec![];
        for (sol, &raw) in input.solutions.iter().zip(&input.values) {
            let value = (raw * 1e8 + 0.5).floor() / 1e8;
            let graph = sol.graph_id;
            if !input.solutions_are_set_sequences {
                if let Some(Some(info)) = self.preprocessed.get(graph as usize) {
                    columns.push(ColumnInfo::from_arcs(
                        graph,
                        info,
                        value,
                        sol.arc_ids.clone(),
                        sol.enumerated,
                        self.num_packing_sets,
                        self.num_covering_sets,
                    ));
                }
            } else {
                columns.push(ColumnInfo::from_set_sequence(
                    graph,
                    value,
                    &sol.arc_ids,
                    self.num_packing_sets,
                ));
            }
        }
        if columns.is_empty() {
            return Some(out);
        }
        let elementary: Vec<i32> = (0..columns.len() as i32)
            .filter(|&c| columns[c as usize].elementary)
            .collect();
        let mut cuts = CandidateSet::new();
        let sort_and_set_memory = !input.solutions_are_set_sequences;
        let max_per_round = self.params.max_num_per_round;
        let few = |cuts: &CandidateSet, limit: i32| (cuts.len() as i64) < limit as i64;
        if self.num_packing_sets >= 1 && self.params.type_to_separate != 2 {
            let mut tuple: Option<TupleSeparation> = None;
            let mut ls: Option<LocalSearch<false>> = None;
            self.one_row_pack(&columns, &mut cuts);
            if few(&cuts, max_per_round) {
                self.raise_rank(3);
            }
            self.insert_best(&columns, &mut cuts, max_per_round, sort_and_set_memory, true, &mut out);
            if self.current_max_rank > 2 {
                self.three_rows_pack(&columns, &mut cuts);
                let limit3 = max_per_round * 3 / 2;
                if few(&cuts, limit3) {
                    self.raise_rank(4);
                }
                self.insert_best(&columns, &mut cuts, limit3, sort_and_set_memory, true, &mut out);
                if self.current_max_rank > 3 {
                    let local_search = self.params.ls_num_iterations >= 1 && !self.params.ls_iterations_negative;
                    if !local_search {
                        if self.params.max_num_triplets < 1 {
                            self.four_rows_pack(&columns, &mut cuts);
                        } else {
                            let mut t = TupleSeparation::new(self, &columns, &elementary);
                            t.four_rows_pack(self, &mut cuts);
                            tuple = Some(t);
                        }
                    } else {
                        let mut l = LocalSearch::<false>::new(self, &columns, &elementary);
                        l.run(self, &columns, &mut cuts, 4, 1.0, &[2, 1, 1, 1], 3, 0);
                        ls = Some(l);
                    }
                    if few(&cuts, max_per_round) {
                        self.raise_rank(5);
                    }
                    self.insert_best(&columns, &mut cuts, max_per_round, sort_and_set_memory, true, &mut out);
                    if self.current_max_rank > 4 {
                        if !local_search {
                            if self.params.max_num_triplets < 1 {
                                self.five_rows_pack(&columns, &mut cuts);
                            } else {
                                let t = tuple.as_mut().unwrap();
                                for kind in 0..2 {
                                    t.five_rows_pack(
                                        self,
                                        &mut cuts,
                                        five_or_more_row_rhs(0, kind),
                                        five_or_more_row_numerators(0, kind),
                                        five_or_more_row_denominator(0, kind),
                                        kind as i32,
                                    );
                                }
                            }
                        } else {
                            let l = ls.as_mut().unwrap();
                            for kind in 0..7 {
                                l.run(
                                    self,
                                    &columns,
                                    &mut cuts,
                                    5,
                                    five_or_more_row_rhs(0, kind),
                                    five_or_more_row_numerators(0, kind),
                                    five_or_more_row_denominator(0, kind),
                                    kind as i32,
                                );
                            }
                        }
                        if few(&cuts, max_per_round) {
                            self.raise_rank(6);
                        }
                        self.insert_best(&columns, &mut cuts, max_per_round, sort_and_set_memory, true, &mut out);
                        if self.current_max_rank > 5 && self.params.ls_num_iterations > 0 {
                            if self.params.ls_iterations_negative {
                                ls = Some(LocalSearch::<false>::new(self, &columns, &elementary));
                            }
                            let first = [7, 12, 18];
                            let end = [12, 18, 23];
                            for step in 0..3 {
                                let rows = 6 + step as i32;
                                if let Some(l) = ls.as_mut() {
                                    for kind in first[step]..end[step] {
                                        l.run(
                                            self,
                                            &columns,
                                            &mut cuts,
                                            rows,
                                            five_or_more_row_rhs(0, kind),
                                            five_or_more_row_numerators(0, kind),
                                            five_or_more_row_denominator(0, kind),
                                            kind as i32,
                                        );
                                    }
                                }
                                if step < 2 && few(&cuts, max_per_round) {
                                    self.raise_rank(rows + 1);
                                }
                                self.insert_best(&columns, &mut cuts, max_per_round, sort_and_set_memory, true, &mut out);
                                if step < 2 && !(self.current_max_rank > rows && self.params.ls_num_iterations > 0) {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        if self.num_covering_sets > 0 && self.params.type_to_separate != 1 {
            self.one_row_cov(&columns, &mut cuts);
            if few(&cuts, max_per_round) {
                self.raise_rank(3);
            }
            self.insert_best(&columns, &mut cuts, max_per_round, sort_and_set_memory, true, &mut out);
            if self.current_max_rank > 2 {
                self.three_rows_cov(&columns, &mut cuts);
                if few(&cuts, max_per_round) {
                    self.raise_rank(4);
                }
                self.insert_best(&columns, &mut cuts, max_per_round * 3 / 2, sort_and_set_memory, true, &mut out);
                if self.current_max_rank > 3 {
                    let mut ls: Option<LocalSearch<true>> = None;
                    if self.params.ls_num_iterations > 0 {
                        let mut l = LocalSearch::<true>::new(self, &columns, &elementary);
                        l.run(self, &columns, &mut cuts, 4, 2.0, &[1, 1, 1, 1], 3, 0);
                        l.run(self, &columns, &mut cuts, 4, 3.0, &[2, 2, 2, 1], 3, 1);
                        ls = Some(l);
                    }
                    if few(&cuts, max_per_round) {
                        self.raise_rank(5);
                    }
                    self.insert_best(&columns, &mut cuts, max_per_round, sort_and_set_memory, true, &mut out);
                    if self.current_max_rank > 4 && self.params.ls_num_iterations > 0 {
                        let l = ls.as_mut().unwrap();
                        for kind in 0..10 {
                            l.run(
                                self,
                                &columns,
                                &mut cuts,
                                5,
                                five_or_more_row_rhs(1, kind),
                                five_or_more_row_numerators(1, kind),
                                five_or_more_row_denominator(1, kind),
                                kind as i32,
                            );
                        }
                        self.insert_best(&columns, &mut cuts, max_per_round, sort_and_set_memory, true, &mut out);
                    }
                }
            }
        }
        cuts.clear();
        Some(out)
    }
}
