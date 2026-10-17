//! Limited-memory rank-one cuts: the cut registry of the C API, the native
//! separator (`separation`, `local_search`, `tuple`) and its adapter, and the
//! route coefficient walk. Ported from the C++ reference adapters
//! `native/adapters/RankOneApiAdapter.h` and `src/api/NativeCutCoefficient.h`.
pub mod bits;
pub mod capacity;
pub mod local_search;
pub mod mincut;
pub mod pulse;
pub mod rbtree;
pub mod separation;
pub mod state;
pub mod stdhash;
pub mod stdsort;
pub mod subcuts;
pub mod tuple;

use crate::core::options::Options;
use crate::graph::Graph;
use crate::{invalid, Result};
use separation::{ArcData, GraphData, Incidence, Input, Parameters, Separation, Solution, VertexData};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutClass {
    Packing,
    Covering,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryKind {
    Vertex,
    Arc,
}
/// RCSPrankOneCutsMemoryType: 0 vertex, 1 arc, 2 default copy, 3 all vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryType {
    Vertex = 0,
    Arc = 1,
    Default = 2,
    AllVertices = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateMode {
    CliqueEnumeration = 0,
    Tuple = 1,
    LocalSearch = 2,
}

/// One registered rank-one cut (openrscp::RankOneCut). Weights and memory are
/// indexed by physical vertex; memories may depend on the graph id.
#[derive(Clone, Debug, PartialEq)]
pub struct RankOneCut {
    pub weights: Vec<i32>,
    pub memory: Vec<bool>,
    pub graph_dependent_memory: bool,
    pub graph_memories: BTreeMap<i32, Vec<bool>>,
    pub memory_kind: MemoryKind,
    pub arc_memory: BTreeSet<i32>,
    pub graph_arc_memories: BTreeMap<i32, BTreeSet<i32>>,
    pub arc_weights: BTreeMap<i32, i32>,
    pub native_type2: bool,
    pub denominator: i32,
    pub dual: f64,
    pub class: CutClass,
    pub rhs: i32,
    pub kind: i32,
    pub id: i32,
    /// (vertex, weight) in native row order.
    pub ordered_support: Option<Vec<(usize, i32)>>,
}
impl RankOneCut {
    pub fn memory_for_graph(&self, graph: i32) -> Vec<bool> {
        if !self.graph_dependent_memory {
            return self.memory.clone();
        }
        self.graph_memories
            .get(&graph)
            .cloned()
            .unwrap_or_else(|| vec![false; self.weights.len()])
    }
    pub fn arc_memory_for_graph(&self, graph: i32) -> BTreeSet<i32> {
        if !self.graph_dependent_memory {
            return self.arc_memory.clone();
        }
        self.graph_arc_memories.get(&graph).cloned().unwrap_or_default()
    }
}

/// A fractional master column for separation: customer sequence (source and
/// sink are added), value, and the enumerated flag.
#[derive(Clone, Debug)]
pub struct FractionalColumn {
    pub vertices: Vec<usize>,
    pub value: f64,
    pub graph_id: i32,
    pub enumerated: bool,
}

/// CutSeparationParameters of the adapter.
#[derive(Clone, Debug)]
pub struct SeparationParameters {
    pub max_cuts: usize,
    pub max_triplets: usize,
    pub max_rank: i32,
    pub start_rank: i32,
    pub local_search_rounds: i32,
    pub graph_id: i32,
    pub arc_memory_type: i32,
    pub violation_epsilon: f64,
    pub covering: bool,
    pub sp_dependent_memory: bool,
    pub ls_iterations_negative: bool,
    pub use_covering_sets_for_one_row: bool,
    pub covering_sets_present: bool,
    pub mode: CandidateMode,
    pub memory_type: MemoryType,
}
impl Default for SeparationParameters {
    fn default() -> Self {
        Self {
            max_cuts: 20,
            max_triplets: 10000,
            max_rank: 5,
            start_rank: 0,
            local_search_rounds: 10,
            graph_id: 0,
            arc_memory_type: 2,
            violation_epsilon: 1e-6,
            covering: false,
            sp_dependent_memory: false,
            ls_iterations_negative: false,
            use_covering_sets_for_one_row: false,
            covering_sets_present: false,
            mode: CandidateMode::CliqueEnumeration,
            memory_type: MemoryType::Vertex,
        }
    }
}

/// rank_one_memory_policy.
#[derive(Clone, Copy, Debug)]
pub struct MemoryPolicy {
    pub memory_type: MemoryType,
    pub arc_memory_type: i32,
    pub sp_dependent: bool,
}
pub fn memory_policy(options: &Options) -> Result<MemoryPolicy> {
    let memory_type = match options.get("RCSPrankOneCutsMemoryType")? as i32 {
        0 => MemoryType::Vertex,
        1 => MemoryType::Arc,
        2 => MemoryType::Default,
        3 => MemoryType::AllVertices,
        _ => return Err(invalid("RCSPrankOneCutsMemoryType must be 0, 1, 2 or 3")),
    };
    Ok(MemoryPolicy {
        memory_type,
        arc_memory_type: options.get("RCSPrankOneCutsArcMemoryType")? as i32,
        sp_dependent: options.get("RCSPrankOneCutsSpDependentMemory")? != 0.,
    })
}

/// Separator adapter plus the registry it fills.
#[derive(Clone, Debug)]
pub struct CutRegistry {
    pub cuts: Vec<RankOneCut>,
    pub retired: BTreeSet<usize>,
    /// Cut indices in pricing order (set_pricing_cut_order); the rest follow.
    pub pricing_order: Vec<usize>,
    pub parameters: SeparationParameters,
    pub separation_phase: i32,
    pub covering_sets_present: bool,
    /// Endpoints -> (tail, head, smallest physical arc ID), for the graph
    /// with this many arcs.
    ends: std::cell::RefCell<(usize, std::collections::HashMap<(usize, usize), (usize, usize, i32)>)>,
}
impl Default for CutRegistry {
    fn default() -> Self {
        Self {
            cuts: vec![],
            retired: BTreeSet::new(),
            pricing_order: vec![],
            parameters: SeparationParameters::default(),
            separation_phase: 1,
            covering_sets_present: false,
            ends: Default::default(),
        }
    }
}

fn physical_arcs<'a>(g: &'a Graph) -> BTreeMap<(usize, usize), &'a crate::graph::Arc> {
    let mut by_ends: BTreeMap<(usize, usize), &crate::graph::Arc> = BTreeMap::new();
    for a in &g.arcs {
        let slot = by_ends.entry((a.tail, a.head)).or_insert(a);
        if a.id < slot.id {
            *slot = a;
        }
    }
    by_ends
}
/// Physical arcs of a customer sequence; source and sink are added.
pub fn route_arcs<'a>(g: &'a Graph, vertices: &[usize]) -> Result<Vec<&'a crate::graph::Arc>> {
    let by_ends = physical_arcs(g);
    let mut route = vertices.to_vec();
    if route.is_empty() {
        return Ok(vec![]);
    }
    if route[0] != g.source {
        route.insert(0, g.source);
    }
    if *route.last().unwrap() != g.sink {
        route.push(g.sink);
    }
    route
        .windows(2)
        .map(|w| {
            by_ends
                .get(&(w[0], w[1]))
                .copied()
                .ok_or_else(|| invalid("cut route has no physical arc"))
        })
        .collect()
}

impl CutRegistry {
    pub fn reset_separation_phase(&mut self) {
        self.separation_phase = 1;
    }
    pub fn increase_separation_phase(&mut self) {
        let phase = if self.separation_phase == 1 {
            3
        } else {
            self.separation_phase + 1
        };
        self.separation_phase = self.parameters.max_rank.min(phase);
    }
    /// Separates with the current parameters and registers every emitted cut.
    pub fn separate(&mut self, g: &Graph, customers: &[usize], columns: &[FractionalColumn]) -> Result<usize> {
        let p = &self.parameters;
        let vertices = g.vertices.len();
        if p.max_rank < 1 || p.max_rank > 8 || p.max_cuts < 1 || customers.len() >= 1024 || !(p.violation_epsilon > 0.)
        {
            return Err(invalid("native rank-one separation parameters"));
        }
        let mut row_of = vec![-1i32; vertices];
        for (row, &v) in customers.iter().enumerate() {
            if v >= vertices || row_of[v] >= 0 {
                return Err(invalid("native cut customer IDs"));
            }
            row_of[v] = row as i32;
        }
        let mut graph_ids = BTreeSet::from([p.graph_id]);
        for c in columns {
            graph_ids.insert(if c.graph_id < 0 { p.graph_id } else { c.graph_id });
        }
        let mut graphs = vec![];
        let mut incidences = vec![];
        for &id in &graph_ids {
            if id < 0 {
                return Err(invalid("negative cut graph ID"));
            }
            let rows = customers.len() as i32;
            let packing = if p.covering { 0 } else { rows };
            let covering = if p.covering || (p.use_covering_sets_for_one_row && p.covering_sets_present) {
                rows
            } else {
                0
            };
            let mut graph = GraphData {
                id,
                source: g.source as i32,
                sink: g.sink as i32,
                elem_set_count: rows,
                packing_set_count: packing,
                covering_set_count: covering,
                packing_neighbourhood: vec![vec![]; packing as usize],
                ..Default::default()
            };
            for v in 0..vertices {
                let row = row_of[v];
                let mut vertex = VertexData {
                    id: v as i32,
                    ..Default::default()
                };
                if row >= 0 {
                    vertex.elem_sets = vec![row];
                    if packing > 0 {
                        vertex.packing_sets = vec![row];
                    }
                    if covering > 0 {
                        vertex.covering_sets = vec![row];
                    }
                }
                graph.vertices.push(vertex);
            }
            for a in &g.arcs {
                graph.arcs.push(ArcData {
                    id: a.id,
                    tail: a.tail as i32,
                    head: a.head as i32,
                    ..Default::default()
                });
            }
            let incidence: Incidence = separation::prepare_incidence(&graph)
                .ok_or_else(|| invalid("native cut graph incidence could not be prepared"))?;
            incidences.push((id, incidence));
            graphs.push(graph);
        }
        let mut solutions = vec![];
        for c in columns {
            if !c.value.is_finite() || c.value < 0. {
                return Err(invalid("fractional column value"));
            }
            let arcs = self.route(g, &c.vertices)?;
            solutions.push(Solution {
                graph_id: if c.graph_id < 0 { p.graph_id } else { c.graph_id },
                enumerated: c.enumerated,
                arc_ids: arcs.iter().map(|a| a.2).collect(),
            });
        }
        let params = Parameters {
            max_num_rows: p.max_rank,
            max_num_per_round: p.max_cuts as i32,
            memory_type: p.memory_type as i32,
            arc_memory_type: p.arc_memory_type,
            sp_dependent_memory: p.sp_dependent_memory,
            ls_num_iterations: if p.mode == CandidateMode::LocalSearch {
                p.local_search_rounds
            } else {
                0
            },
            ls_iterations_negative: p.ls_iterations_negative,
            neighbourhood_size: 16,
            max_num_triplets: if p.mode == CandidateMode::CliqueEnumeration {
                0
            } else {
                p.max_triplets.min(i32::MAX as usize) as i32
            },
            violation_tolerance: p.violation_epsilon,
            verbosity: -1,
            type_to_separate: if p.covering { 2 } else { 1 },
            use_cov_sets_for_one_row_packing: p.use_covering_sets_for_one_row,
            ..Default::default()
        };
        let mut state = Separation::new(params);
        let refs: Vec<&GraphData> = graphs.iter().collect();
        if !state.prepare(&refs, incidences) {
            return Err(invalid("native rank-one separation preparation failed"));
        }
        let input = Input {
            solutions: solutions.iter().collect(),
            values: columns.iter().map(|c| c.value).collect(),
            cut_memory_type: match p.memory_type {
                MemoryType::Arc => 1,
                MemoryType::AllVertices => 2,
                _ => 0,
            },
            solutions_are_set_sequences: false,
            separation_phase: if p.start_rank > 0 { p.start_rank } else { p.max_rank },
            active_cuts: vec![],
        };
        let generated = state
            .separate(&input)
            .ok_or_else(|| invalid("native rank-one separation failed"))?;
        self.separation_phase = state.current_max_rank;
        let added = generated.len();
        for native in generated {
            let mut cut = RankOneCut {
                weights: vec![0; vertices],
                memory: vec![false; vertices],
                graph_dependent_memory: true,
                graph_memories: BTreeMap::new(),
                memory_kind: if native.memory_type == 1 {
                    MemoryKind::Arc
                } else {
                    MemoryKind::Vertex
                },
                arc_memory: BTreeSet::new(),
                graph_arc_memories: BTreeMap::new(),
                arc_weights: BTreeMap::new(),
                native_type2: native.kind == 2,
                denominator: native.denominator.abs(),
                dual: 0.,
                class: if native.kind == 1 {
                    CutClass::Covering
                } else {
                    CutClass::Packing
                },
                rhs: native.rhs as i32,
                kind: native.class,
                id: self.cuts.len() as i32,
                ordered_support: None,
            };
            let mut ordered = vec![];
            for r in 0..native.num_rows as usize {
                let vertex = customers[native.set_ids[r] as usize];
                let weight = native.numerators[r].abs();
                cut.weights[vertex] = weight;
                ordered.push((vertex, weight));
            }
            cut.ordered_support = Some(ordered);
            for (&graph, memory) in &native.memory {
                if native.memory_type == 1 {
                    cut.graph_arc_memories.insert(graph, memory.clone());
                    cut.arc_memory.extend(memory.iter().copied());
                } else {
                    let selected = cut.graph_memories.entry(graph).or_insert_with(|| vec![false; vertices]);
                    *selected = vec![false; vertices];
                    for &v in memory {
                        selected[v as usize] = true;
                        cut.memory[v as usize] = true;
                    }
                }
            }
            self.cuts.push(cut);
        }
        Ok(added)
    }
    /// or_rcsp_separate_cuts_default: the option-driven separator, packing
    /// then (with covering sets) covering, starting at the separation phase.
    pub fn separate_default(
        &mut self,
        g: &Graph,
        options: &Options,
        customers: &[usize],
        columns: &[FractionalColumn],
    ) -> Result<usize> {
        let rows = options.get("RCSPrankOneCutsMaxNumRows")?;
        let triplets = options.get("RCSPrankOneCutsMaxNumTriplets")?;
        if rows > 3. && triplets < 1. {
            return Err(invalid(
                "lm-1Rank cuts separator: separation of 4 and 5-rows cuts needs tuple based separation",
            ));
        }
        if rows > 5. {
            return Err(invalid("lm-1Rank cuts separator: separation of 6-row and more cuts is not supported"));
        }
        if rows < 1. {
            return Err(invalid("lm-1Rank cuts separator: at least one row required"));
        }
        let per_round = options.get("RCSPrankOneCutsMaxNumPerRound")?;
        if per_round < 1. {
            return Err(invalid("lm-1Rank cuts separator: positive cuts per round required"));
        }
        let raw_iterations = options.get("RCSPrankOneCutsLSnumIterations")?;
        let iterations = raw_iterations.abs();
        let kind = options.get("RCSPrankOneCutsTypeToSeparate")? as i32;
        if !(0..=2).contains(&kind) {
            return Err(invalid("RCSPrankOneCutsTypeToSeparate must be 0, 1 or 2"));
        }
        let policy = memory_policy(options)?;
        let p = &mut self.parameters;
        p.violation_epsilon = 0.02;
        p.max_triplets = triplets as usize;
        p.local_search_rounds = iterations.min(1e9) as i32;
        p.ls_iterations_negative = raw_iterations < 0.;
        p.max_rank = rows as i32;
        if self.separation_phase > rows as i32 {
            self.separation_phase = rows as i32;
        }
        let phase = self.separation_phase;
        let mut found = 0;
        for covering in [false, true] {
            if (!covering && kind == 2) || (covering && (kind == 1 || !self.covering_sets_present)) {
                continue;
            }
            let p = &mut self.parameters;
            p.max_cuts = per_round as usize;
            p.mode = if iterations > 0. {
                CandidateMode::LocalSearch
            } else {
                CandidateMode::Tuple
            };
            p.covering = covering;
            p.start_rank = phase;
            p.arc_memory_type = policy.arc_memory_type;
            p.covering_sets_present = covering || self.covering_sets_present;
            p.use_covering_sets_for_one_row = options.get("RCSPuseCovSetsForSepOneRowPackingCuts")? != 0.;
            p.sp_dependent_memory = policy.sp_dependent;
            p.graph_id = g.graph_id.max(0);
            p.memory_type = policy.memory_type;
            found += self.separate(g, customers, columns)?;
        }
        Ok(found)
    }
    /// or_rcsp_cut_separation_phase: (phase, at maximum).
    pub fn separation_phase(&self, options: &Options) -> Result<(i32, bool)> {
        let rows = options.get("RCSPrankOneCutsMaxNumRows")? as i32;
        let phase = self.separation_phase.min(rows);
        Ok((phase, phase >= rows))
    }
    pub fn increase_phase(&mut self, options: &Options) -> Result<i32> {
        self.parameters.max_rank = options.get("RCSPrankOneCutsMaxNumRows")? as i32;
        self.increase_separation_phase();
        Ok(self.separation_phase)
    }
    pub fn retire(&mut self, cut: usize) {
        if let Some(c) = self.cuts.get_mut(cut) {
            c.dual = 0.;
        }
        self.retired.insert(cut);
    }
    pub fn is_covering(&self, cut: usize) -> bool {
        self.cuts[cut].class == CutClass::Covering
    }
    /// Physical (tail, head, arc ID) route of a customer sequence; source and
    /// sink are added. Parallel arcs resolve to the smallest ID.
    pub fn route(&self, g: &Graph, vertices: &[usize]) -> Result<Vec<(usize, usize, i32)>> {
        let mut ends = self.ends.borrow_mut();
        if ends.0 != g.arcs.len() || ends.1.is_empty() {
            ends.1.clear();
            for a in &g.arcs {
                let slot = ends.1.entry((a.tail, a.head)).or_insert((a.tail, a.head, a.id));
                if a.id < slot.2 {
                    slot.2 = a.id;
                }
            }
            ends.0 = g.arcs.len();
        }
        if vertices.is_empty() {
            return Ok(vec![]);
        }
        let mut route = Vec::with_capacity(vertices.len() + 2);
        if vertices[0] != g.source {
            route.push(g.source);
        }
        route.extend_from_slice(vertices);
        if *route.last().unwrap() != g.sink {
            route.push(g.sink);
        }
        route
            .windows(2)
            .map(|w| {
                ends.1
                    .get(&(w[0], w[1]))
                    .copied()
                    .ok_or_else(|| invalid("cut route has no physical arc"))
            })
            .collect()
    }
    /// or_rcsp_cut_coefficient_ex for a customer sequence.
    pub fn coefficient(&self, g: &Graph, cut: usize, vertices: &[usize], enumerated: bool) -> Result<i32> {
        let route = self.route(g, vertices)?;
        coefficient(&self.cuts[cut], g.graph_id.max(0), &route, enumerated)
    }
    pub fn route_coefficient(&self, g: &Graph, cut: usize, route: &[(usize, usize, i32)], enumerated: bool) -> Result<i32> {
        coefficient(&self.cuts[cut], g.graph_id.max(0), route, enumerated)
    }
}

/// Coefficient of a route in a cut, walked on the route's own incidence. Each arc's sets are its head (with support)
/// and its arc row (with arc support); the state resets where the memory
/// filter fails unless the column is enumerated.
pub fn coefficient(input: &RankOneCut, graph: i32, route: &[(usize, usize, i32)], enumerated: bool) -> Result<i32> {
    if route.is_empty() {
        return Ok(0);
    }
    let kind = if input.native_type2 {
        2
    } else if input.class == CutClass::Covering {
        1
    } else {
        0
    };
    let sign = if kind == 1 { -1 } else { 1 };
    let denominator = sign * input.denominator;
    let arc_memory = input.memory_kind == MemoryKind::Arc;
    let present = !input.graph_dependent_memory
        || if arc_memory {
            input.graph_arc_memories.contains_key(&graph)
        } else {
            input.graph_memories.contains_key(&graph)
        };
    if !present && !enumerated {
        return Ok(0);
    }
    let vertex_memory = if arc_memory || !present {
        None
    } else if input.graph_dependent_memory {
        input.graph_memories.get(&graph)
    } else {
        Some(&input.memory)
    };
    let arc_set = if arc_memory && present {
        Some(input.arc_memory_for_graph(graph))
    } else {
        None
    };
    let packing_like = kind & !2 == 0;
    let (mut state, mut count) = (0i32, 0i32);
    let mut previous: Option<usize> = None;
    for &(tail, head, id) in route {
        if previous.is_some_and(|p| p != tail) {
            return Err(invalid("cut physical route is not contiguous"));
        }
        previous = Some(head);
        if !enumerated {
            let kept = match (&vertex_memory, &arc_set) {
                (Some(m), _) => m.get(tail).copied().unwrap_or(false),
                (None, Some(a)) => a.contains(&id),
                _ => true,
            };
            if !kept {
                state = 0;
            }
        }
        let mut step = |increment: i32| {
            state += increment;
            if packing_like {
                if denominator <= state {
                    count += 1;
                    state -= denominator;
                }
            } else if state < 0 {
                count += 1;
                state -= denominator;
            }
        };
        let w = input.weights.get(head).copied().unwrap_or(0);
        if w != 0 {
            step(sign * w);
        }
        if let Some(&aw) = (!input.arc_weights.is_empty())
            .then(|| input.arc_weights.get(&id))
            .flatten()
        {
            if aw != 0 {
                step(sign * aw);
            }
        }
    }
    Ok(count)
}
