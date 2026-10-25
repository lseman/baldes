//! Cut registry, graph adapters, memory policies and route coefficients.
//! Ported from `native/adapters/RankOneApiAdapter.h` and
//! `src/api/NativeCutCoefficient.h`.
use super::{knapsack, kpath, rank_one::separation};
use crate::core::options::Options;
use crate::graph::Graph;
use crate::{invalid, Result};
use separation::{
    ArcData, GraphData, Incidence, Input, Parameters, Separation, Solution, VertexData,
};
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
/// BALDESrankOneCutsMemoryType: 0 vertex, 1 arc, 2 default copy, 3 all vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryType {
    Vertex = 0,
    Arc = 1,
    Default = 2,
    AllVertices = 3,
}
impl MemoryType {
    /// Compile-time policy for generated cuts. Explicit full-vertex memory
    /// modes retain their meaning; manually registered rows are not converted.
    pub(crate) fn for_build(self) -> Self {
        if cfg!(feature = "rank-one-node-memory") && self == Self::Arc {
            Self::Vertex
        } else {
            self
        }
    }
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
    /// Graph-specific physical support; empty retains the legacy vertex mapping.
    pub graph_support: BTreeMap<i32, Vec<(usize, i32)>>,
    /// Native k-path pricing reads covering incidence regardless of the
    /// packing/covering universe selected by separation and coefficients.
    pub graph_pricing_support: BTreeMap<i32, Vec<(usize, i32)>>,
    /// Full-memory native grouping uses original set IDs, before expansion.
    pub native_kpath_support: Option<Vec<usize>>,
    pub memory: Vec<bool>,
    pub graph_dependent_memory: bool,
    pub graph_memories: BTreeMap<i32, Vec<bool>>,
    pub memory_kind: MemoryKind,
    pub arc_memory: BTreeSet<i32>,
    pub graph_arc_memories: BTreeMap<i32, BTreeSet<i32>>,
    pub arc_weights: BTreeMap<i32, i32>,
    pub visit_once: bool,
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
    pub fn weight_for_graph(&self, graph: i32, vertex: usize) -> i32 {
        if self.graph_support.is_empty() {
            return self.weights.get(vertex).copied().unwrap_or(0);
        }
        self.graph_support
            .get(&graph)
            .and_then(|support| {
                support
                    .binary_search_by_key(&vertex, |&(v, _)| v)
                    .ok()
                    .map(|i| support[i].1)
            })
            .unwrap_or(0)
    }
    pub fn ordered_support_for_graph(&self, graph: i32) -> Option<&Vec<(usize, i32)>> {
        if self.graph_support.is_empty() {
            self.ordered_support.as_ref()
        } else {
            self.graph_support.get(&graph)
        }
    }

    pub fn pricing_weight_for_graph(&self, graph: i32, vertex: usize) -> i32 {
        if self.graph_pricing_support.is_empty() {
            return self.weight_for_graph(graph, vertex);
        }
        self.graph_pricing_support
            .get(&graph)
            .and_then(|support| {
                support
                    .binary_search_by_key(&vertex, |&(v, _)| v)
                    .ok()
                    .map(|i| support[i].1)
            })
            .unwrap_or(0)
    }
    pub fn pricing_ordered_support_for_graph(&self, graph: i32) -> Option<&Vec<(usize, i32)>> {
        if self.graph_pricing_support.is_empty() {
            self.ordered_support_for_graph(graph)
        } else {
            self.graph_pricing_support.get(&graph)
        }
    }
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
        self.graph_arc_memories
            .get(&graph)
            .cloned()
            .unwrap_or_default()
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
    /// Maximum cuts touching a vertex through support or memory; zero disables.
    pub max_per_vertex: usize,
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
            max_per_vertex: 0,
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
    let memory_type = match options.get("BALDESrankOneCutsMemoryType")? as i32 {
        0 => MemoryType::Vertex,
        1 => MemoryType::Arc,
        2 => MemoryType::Default,
        3 => MemoryType::AllVertices,
        _ => return Err(invalid("BALDESrankOneCutsMemoryType must be 0, 1, 2 or 3")),
    };
    Ok(MemoryPolicy {
        memory_type: memory_type.for_build(),
        arc_memory_type: options.get("BALDESrankOneCutsArcMemoryType")? as i32,
        sp_dependent: options.get("BALDESrankOneCutsSpDependentMemory")? != 0.,
    })
}

/// Separator adapter plus the registry it fills.
#[derive(Clone, Debug)]
pub struct CutRegistry {
    pub cuts: Vec<RankOneCut>,
    pub knapsack_cuts: Vec<knapsack::ResourceKnapsackCut>,
    pub retired: BTreeSet<usize>,
    /// Cut indices in pricing order (set_pricing_cut_order); the rest follow.
    pub pricing_order: Vec<usize>,
    /// Price nonexact cuts with native 1e-8 dual rounding after filtering
    /// their raw duals; otherwise only strong k-path duals are rounded.
    pub round_duals: bool,
    pub parameters: SeparationParameters,
    pub separation_phase: i32,
    pub covering_sets_present: bool,
    /// Endpoints -> (tail, head, smallest physical arc ID), for the graph
    /// with this many arcs.
    ends: std::cell::RefCell<(
        usize,
        std::collections::HashMap<(usize, usize), (usize, usize, i32)>,
    )>,
}

fn physical_arcs<'a, const D: usize>(
    g: &'a Graph<D>,
) -> BTreeMap<(usize, usize), &'a crate::graph::Arc<D>> {
    let mut by_ends: BTreeMap<(usize, usize), &crate::graph::Arc<D>> = BTreeMap::new();
    for a in &g.arcs {
        let slot = by_ends.entry((a.tail, a.head)).or_insert(a);
        if a.id < slot.id {
            *slot = a;
        }
    }
    by_ends
}
/// Physical arcs of a customer sequence; source and sink are added.
pub fn route_arcs<'a, const D: usize>(
    g: &'a Graph<D>,
    vertices: &[usize],
) -> Result<Vec<&'a crate::graph::Arc<D>>> {
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

/// Coefficient of a route in a cut, walked on the route's own incidence. Each arc's sets are its head (with support)
/// and its arc row (with arc support); the state resets where the memory
/// filter fails unless the column is enumerated.
pub fn coefficient(
    input: &RankOneCut,
    graph: i32,
    route: &[(usize, usize, i32)],
    enumerated: bool,
) -> Result<i32> {
    if route.is_empty() {
        return Ok(0);
    }
    // The usual rank-one packing row needs only dense vertex weights and
    // vertex memory. Keep the general arc/support policies out of its loop.
    if !input.visit_once
        && (input.native_type2 || input.class == CutClass::Packing)
        && input.memory_kind == MemoryKind::Vertex
        && input.graph_support.is_empty()
        && input.arc_weights.is_empty()
    {
        let memory = if input.graph_dependent_memory {
            input.graph_memories.get(&graph)
        } else {
            Some(&input.memory)
        };
        if !enumerated && memory.is_none() {
            return Ok(0);
        }
        let (mut state, mut count) = (0, 0);
        let mut previous = None;
        for &(tail, head, _) in route {
            if previous.is_some_and(|p| p != tail) {
                return Err(invalid("cut physical route is not contiguous"));
            }
            previous = Some(head);
            if !enumerated && !memory.and_then(|m| m.get(tail)).copied().unwrap_or(false) {
                state = 0;
            }
            let weight = input.weights.get(head).copied().unwrap_or(0);
            if weight != 0 {
                state += weight;
                if input.denominator <= state {
                    count += 1;
                    state -= input.denominator;
                }
            }
        }
        return Ok(count);
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
        if input.graph_dependent_memory {
            input.graph_arc_memories.get(&graph)
        } else {
            Some(&input.arc_memory)
        }
    } else {
        None
    };
    if input.visit_once {
        if !enumerated && input.graph_dependent_memory && !input.graph_memories.contains_key(&graph)
        {
            return Ok(0);
        }
        if !input.arc_weights.is_empty() {
            return Err(invalid("native strong k-path requires vertex support"));
        }
        let mut inside = false;
        let mut count = 0i32;
        let mut previous: Option<usize> = None;
        for &(tail, head, id) in route {
            if previous.is_some_and(|p| p != tail) {
                return Err(invalid("cut physical route is not contiguous"));
            }
            previous = Some(head);
            let in_support = input.weight_for_graph(graph, head) != 0;
            if in_support {
                if !inside {
                    if enumerated {
                        return Ok(1);
                    }
                    inside = true;
                    count += 1;
                }
            } else if inside && !enumerated {
                let kept = match (&vertex_memory, &arc_set) {
                    (Some(m), _) => m.get(head).copied().unwrap_or(false),
                    (None, Some(a)) => a.contains(&id),
                    _ => false,
                };
                if !kept {
                    inside = false;
                }
            }
        }
        return Ok(count);
    }
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
        let w = input.weight_for_graph(graph, head);
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

#[cfg(test)]
#[path = "../../../tests/unit/cuts/registry/coefficient_tests.rs"]
mod coefficient_tests;

mod kpath_registration;
mod lifecycle;
mod rank_one;
mod routes;

impl Default for CutRegistry {
    fn default() -> Self {
        Self {
            cuts: vec![],
            knapsack_cuts: vec![],
            retired: BTreeSet::new(),
            pricing_order: vec![],
            round_duals: false,
            parameters: SeparationParameters::default(),
            separation_phase: 1,
            covering_sets_present: false,
            ends: Default::default(),
        }
    }
}
