//! Labeling-side cut state of ResourceType<SetBasedCutsResources>, from the
//! C++ reference `native/cuts/CutState.h`, `CutPreparation.h` and
//! `enumeration/FullMemoryPreparation.h`. Limited-memory labels carry twenty
//! packed words and an additional cost; enumeration labels sixty words.
//! Vertex tables are indexed by physical vertex, arc tables by the arc's
//! position in `Graph::arcs`. Strong k-path cuts are not ported.
use super::{CutClass, CutRegistry, MemoryKind};
use crate::graph::Graph;
use std::collections::{BTreeMap, BTreeSet};

pub const LIMITED_WORDS: usize = 20;
pub const FULL_WORDS: usize = 60;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ArcTransition {
    pub mask: u32,
    pub denominator: i32,
    pub source_word: i32,
    pub source_shift: i32,
    pub target_word: i32,
    pub target_shift: i32,
    pub increment: i32,
    pub covering: bool,
    pub overflow_cost: f64,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DominanceDescriptor {
    pub mask: u32,
    pub denominator: i32,
    pub word: i32,
    pub shift: i32,
    pub subtract_cost: bool,
    pub cost: f64,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VertexSupport {
    pub mask: u32,
    pub denominator: i32,
    pub word: i32,
    pub shift: i32,
    pub increment: i32,
    pub covering: bool,
    pub cost: f64,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VertexInfo {
    pub support: Vec<VertexSupport>,
    pub full_memory_support: Vec<VertexSupport>,
    pub dominance: Vec<DominanceDescriptor>,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArcInfo {
    pub ordinary: Vec<ArcTransition>,
    pub full_memory: Vec<ArcTransition>,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CutResourceInfo {
    /// Minus the duals of k-path cuts supported at the vertex (zero here).
    pub vertex_cut_potential: Vec<f64>,
    pub vertices: Vec<VertexInfo>,
    pub forward: Vec<ArcInfo>,
    pub backward: Vec<ArcInfo>,
}

pub fn arithmetic_shift_word(word: u32, shift: i32) -> u32 {
    let count = (shift as u32) & 31;
    let mut result = word >> count;
    if count != 0 && word & 0x8000_0000 != 0 {
        result |= u32::MAX << (32 - count);
    }
    result
}
fn field(word: u32, shift: i32, mask: u32) -> i32 {
    (arithmetic_shift_word(word, shift) & mask) as i32
}

/// Overflow arithmetic of the transitions and vertex supports: packing
/// pays r / d when r >= d; covering pays (r + 1) / d + 1 when r < 0, with a
/// negative denominator. Returns (coefficient, remainder).
fn overflow(remainder: i32, denominator: i32, covering: bool) -> (i32, i32) {
    if covering {
        if remainder < 0 {
            let coefficient = (remainder + 1) / denominator + 1;
            return (
                coefficient,
                (remainder as u32).wrapping_sub((denominator as u32).wrapping_mul(coefficient as u32)) as i32,
            );
        }
        (0, remainder)
    } else if remainder >= denominator {
        (remainder / denominator, remainder % denominator)
    } else {
        (0, remainder)
    }
}

/// Limited-memory transition along an arc. Output words start from
/// zero; costs accumulate in descriptor order.
pub fn change_along_arc(words: &mut [u32; LIMITED_WORDS], additional: &mut f64, transitions: &[ArcTransition]) {
    let mut result = [0u32; LIMITED_WORDS];
    for t in transitions {
        if t.source_word < 0 && t.increment == 0 {
            continue;
        }
        let mut remainder = if t.source_word < 0 {
            0
        } else {
            field(words[t.source_word as usize], t.source_shift, t.mask)
        };
        if t.increment != 0 {
            remainder = (remainder as u32).wrapping_add(t.increment as u32) as i32;
            let (coefficient, rest) = overflow(remainder, t.denominator, t.covering);
            remainder = rest;
            if coefficient != 0 {
                let charge = coefficient as f64 * t.overflow_cost;
                *additional = *additional + charge;
            }
        }
        if remainder > 0 && t.target_word >= 0 {
            result[t.target_word as usize] |= (remainder as u32) << ((t.target_shift as u32) & 31);
        }
    }
    *words = result;
}
/// Vertex supports added in place.
pub fn apply_vertex_support(words: &mut [u32], additional: &mut f64, support: &[VertexSupport]) {
    for f in support {
        let original = words[f.word as usize];
        let remainder = (field(original, f.shift, f.mask) as u32).wrapping_add(f.increment as u32) as i32;
        let (coefficient, remainder) = overflow(remainder, f.denominator, f.covering);
        if coefficient != 0 {
            let charge = coefficient as f64 * f.cost;
            *additional = charge + *additional;
        }
        let shift = (f.shift as u32) & 31;
        words[f.word as usize] = (original & !(f.mask << shift)) | ((remainder as u32) << shift);
    }
}
/// Full-memory transitions: the persistent words are
/// edited in descriptor order; a negative destination clears the source.
pub fn change_full_memory_along_arc(words: &mut [u32; FULL_WORDS], additional: &mut f64, transitions: &[ArcTransition]) {
    for t in transitions {
        let original = words[t.source_word as usize];
        let mut remainder = field(original, t.source_shift, t.mask);
        if t.increment != 0 {
            remainder = (remainder as u32).wrapping_add(t.increment as u32) as i32;
            let (coefficient, rest) = overflow(remainder, t.denominator, t.covering);
            remainder = rest;
            if coefficient != 0 {
                let charge = coefficient as f64 * t.overflow_cost;
                *additional = charge + *additional;
            }
        }
        if t.target_word < 0 {
            words[t.source_word as usize] = original & !(t.mask << ((t.source_shift as u32) & 31));
        } else {
            let shift = (t.target_shift as u32) & 31;
            let target = &mut words[t.target_word as usize];
            *target = (*target & !(t.mask << shift)) | ((remainder as u32) << shift);
        }
    }
}

/// Cut-field dominance. Signed field comparison; the gap is
/// updated in descriptor order with an early exit.
pub fn packed_dominate(descriptors: &[DominanceDescriptor], a: &[u32], b: &[u32], gap: &mut f64) -> bool {
    for f in descriptors {
        let left = field(a[f.word as usize], f.shift, f.mask);
        let right = field(b[f.word as usize], f.shift, f.mask);
        if left <= right {
            continue;
        }
        if f.subtract_cost {
            *gap -= f.cost;
        } else {
            *gap += f.cost;
        }
        if *gap >= 1e-6 {
            return false;
        }
    }
    !(*gap >= 1e-6)
}
/// Cut part of a join at `vertex`: additional costs, then each field whose
/// summed remainders reach its denominator.
pub fn concatenation_cost(
    info: &CutResourceInfo,
    vertex: usize,
    forward: (&[u32], f64),
    backward: (&[u32], f64),
    cost: f64,
) -> f64 {
    let potential = info.vertex_cut_potential.get(vertex).copied().unwrap_or(0.0);
    let mut total = (forward.1 + backward.1) + potential;
    total = total + cost;
    let mut cost = total;
    let Some(v) = info.vertices.get(vertex) else {
        return cost;
    };
    for f in &v.dominance {
        let a = field(forward.0[f.word as usize], f.shift, f.mask);
        let b = field(backward.0[f.word as usize], f.shift, f.mask);
        let sum = (a as u32).wrapping_add(b as u32) as i32;
        if f.subtract_cost {
            if -f.denominator <= sum {
                cost = cost - f.cost;
            }
        } else if f.denominator <= sum {
            cost = f.cost + cost;
        }
    }
    cost
}

/// A limited-memory rank-one cut restated at vertex level.
#[derive(Clone, Debug, PartialEq)]
pub struct LimitedInput {
    pub covering: bool,
    pub type2: bool,
    pub arc_memory: bool,
    pub denominator: i32,
    pub dual: f64,
    pub vertex_coefficient: Vec<i32>,
    /// Arc ID -> coefficient, nonzero entries only.
    pub arc_coefficient: BTreeMap<i32, i32>,
    pub memory_vertices: Vec<usize>,
    pub memory_arcs: BTreeSet<i32>,
}
/// FullMemoryRankOneInput.
#[derive(Clone, Debug, PartialEq)]
pub struct FullInput {
    pub ordered_support: Vec<(usize, i32)>,
    pub denominator: i32,
    pub covering: bool,
    pub dual: f64,
    pub vertex_coefficient: Vec<i32>,
    pub arc_coefficient: BTreeMap<i32, i32>,
}

const BITMASK: [u32; 8] = [0, 1, 1, 3, 3, 7, 7, 7];
const DENOMINATOR_BITS: [i32; 8] = [0, 1, 1, 2, 2, 3, 3, 3];
fn lim_memory_bits(denominator: i32) -> i32 {
    if (0..8).contains(&denominator) {
        DENOMINATOR_BITS[denominator as usize]
    } else if (-8..0).contains(&denominator) {
        BITMASK[(8 + denominator) as usize] as i32
    } else {
        panic!("limited-memory denominator")
    }
}
fn lim_memory_mask(denominator: i32) -> u32 {
    BITMASK[denominator.unsigned_abs() as usize]
}

/// Per cut and vertex the bit position, and the owning
/// cut of every dual-carrying record for dual-only updates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LimitedPreparation {
    pub positions: Vec<Vec<i32>>,
    pub dominance_cut: Vec<Vec<usize>>,
    pub support_cut: Vec<Vec<usize>>,
    pub transition_cut: [Vec<Vec<usize>>; 2],
}

/// Vertex fields, dominance descriptors and supports. False when a vertex needs more
/// than twenty words.
pub fn prepare_vertices(
    info: &mut CutResourceInfo,
    prep: &mut LimitedPreparation,
    g: &Graph,
    cuts: &[LimitedInput],
) -> bool {
    let n = g.vertices.len();
    prep.positions.clear();
    info.vertices.resize(n, VertexInfo::default());
    for v in &mut info.vertices {
        v.dominance.clear();
        v.support.clear();
    }
    prep.dominance_cut = vec![vec![]; n];
    prep.support_cut = vec![vec![]; n];
    let mut next = vec![0i32; n];
    for (index, cut) in cuts.iter().enumerate() {
        let mask = lim_memory_mask(cut.denominator);
        let bits = lim_memory_bits(cut.denominator);
        let mut position = vec![-1i32; n];
        let mut allocate = |vertex: usize, position: &mut Vec<i32>| {
            let mut at = next[vertex];
            if at % 32 + bits > 31 {
                at = (at / 32) * 32 + 32;
            }
            next[vertex] = at;
            position[vertex] = at;
            next[vertex] += bits;
        };
        if !cut.arc_memory {
            for &vertex in &cut.memory_vertices {
                allocate(vertex, &mut position);
            }
        } else {
            let mut incoming = vec![false; n];
            let mut outgoing = vec![false; n];
            for a in &g.arcs {
                if cut.memory_arcs.contains(&a.id) {
                    outgoing[a.tail] = true;
                    incoming[a.head] = true;
                }
            }
            for vertex in 0..n {
                if (incoming[vertex] && outgoing[vertex]) || cut.vertex_coefficient[vertex] != 0 {
                    allocate(vertex, &mut position);
                }
            }
        }
        for vertex in 0..n {
            let at = position[vertex];
            if at < 0 {
                continue;
            }
            info.vertices[vertex].dominance.push(DominanceDescriptor {
                mask,
                denominator: cut.denominator,
                word: at >> 5,
                shift: at & 31,
                subtract_cost: cut.covering,
                cost: cut.dual,
            });
            prep.dominance_cut[vertex].push(index);
            if cut.vertex_coefficient[vertex] > 0 {
                info.vertices[vertex].support.push(VertexSupport {
                    mask,
                    denominator: cut.denominator,
                    word: at >> 5,
                    shift: at & 31,
                    increment: cut.vertex_coefficient[vertex],
                    covering: cut.covering,
                    cost: cut.dual,
                });
                prep.support_cut[vertex].push(index);
            }
        }
        prep.positions.push(position);
    }
    info.vertex_cut_potential = vec![0.0; n];
    next.iter().all(|&used| {
        let words = if used <= 0 { 1 } else { (used - 1) / 32 + 1 };
        words <= LIMITED_WORDS as i32
    })
}
/// Arc transitions of one direction.
pub fn prepare_arcs(
    info: &mut CutResourceInfo,
    prep: &mut LimitedPreparation,
    g: &Graph,
    cuts: &[LimitedInput],
    forward: bool,
) {
    let tables = if forward {
        &mut info.forward
    } else {
        &mut info.backward
    };
    tables.resize(g.arcs.len(), ArcInfo::default());
    let owners = &mut prep.transition_cut[if forward { 0 } else { 1 }];
    owners.resize(g.arcs.len(), vec![]);
    for (t, o) in tables.iter_mut().zip(owners.iter_mut()) {
        t.ordinary.clear();
        o.clear();
    }
    let index_of: BTreeMap<i32, usize> = g.arcs.iter().enumerate().map(|(i, a)| (a.id, i)).collect();
    // Arcs entering each vertex (forward; leaving it backward), ordered by
    // the other end and then graph order.
    let n = g.vertices.len();
    let mut into: Vec<Vec<usize>> = vec![vec![]; n];
    let mut order: Vec<usize> = (0..g.arcs.len()).collect();
    order.sort_by_key(|&i| if forward { g.arcs[i].tail } else { g.arcs[i].head });
    for i in order {
        let to = if forward { g.arcs[i].head } else { g.arcs[i].tail };
        into[to].push(i);
    }
    for (index, cut) in cuts.iter().enumerate() {
        let position = &prep.positions[index];
        let mask = lim_memory_mask(cut.denominator);
        let coordinates = |at: i32| if at < 0 { (-1, -1) } else { (at >> 5, at % 32) };
        let mut push = |arc: usize, source: i32, target: i32, increment: i32| {
            let (sw, ss) = coordinates(source);
            let (tw, ts) = coordinates(target);
            tables[arc].ordinary.push(ArcTransition {
                mask,
                denominator: cut.denominator,
                source_word: sw,
                source_shift: ss,
                target_word: tw,
                target_shift: ts,
                increment,
                covering: cut.covering,
                overflow_cost: cut.dual,
            });
            owners[arc].push(index);
        };
        let coefficient = |id: i32| cut.arc_coefficient.get(&id).copied().unwrap_or(0);
        if !cut.arc_memory {
            for &memory in &cut.memory_vertices {
                let target = position[memory];
                for &i in &into[memory] {
                    let a = &g.arcs[i];
                    let increment = coefficient(a.id);
                    let source = position[if forward { a.tail } else { a.head }];
                    if source < 0 && increment == 0 {
                        continue;
                    }
                    push(i, source, target, increment);
                }
            }
        } else {
            for &id in &cut.memory_arcs {
                let Some(&i) = index_of.get(&id) else {
                    continue;
                };
                let a = &g.arcs[i];
                let increment = coefficient(id);
                let source = position[if forward { a.tail } else { a.head }];
                let target = position[if forward { a.head } else { a.tail }];
                if increment == 0 && (source < 0 || target < 0) {
                    continue;
                }
                push(i, source, target, increment);
            }
        }
        for (&id, &increment) in &cut.arc_coefficient {
            let Some(&i) = index_of.get(&id) else {
                continue;
            };
            let a = &g.arcs[i];
            if increment < 1 || position[a.head] >= 0 || position[a.tail] >= 0 {
                continue;
            }
            push(i, -1, -1, increment);
        }
    }
}
/// Dual-only change: every cost copied from a dual takes the new dual.
pub fn update_duals(info: &mut CutResourceInfo, prep: &LimitedPreparation, cuts: &[LimitedInput], forward: bool, backward: bool) {
    for (v, vertex) in info.vertices.iter_mut().enumerate() {
        if v >= prep.dominance_cut.len() {
            break;
        }
        for (i, d) in vertex.dominance.iter_mut().enumerate() {
            d.cost = cuts[prep.dominance_cut[v][i]].dual;
        }
        for (i, s) in vertex.support.iter_mut().enumerate() {
            s.cost = cuts[prep.support_cut[v][i]].dual;
        }
    }
    let n = info.vertex_cut_potential.len();
    info.vertex_cut_potential = vec![0.0; n];
    for (direction, enabled) in [(0, forward), (1, backward)] {
        if !enabled {
            continue;
        }
        let tables = if direction == 0 {
            &mut info.forward
        } else {
            &mut info.backward
        };
        for (arc, owners) in prep.transition_cut[direction].iter().enumerate() {
            if arc >= tables.len() {
                break;
            }
            for (i, t) in tables[arc].ordinary.iter_mut().enumerate() {
                t.overflow_cost = cuts[owners[i]].dual;
            }
        }
    }
}

/// FullMemoryField.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FullField {
    pub mask: u32,
    pub denominator: i32,
    pub word: i32,
    pub shift: i32,
    pub covering: bool,
    pub index: i32,
    pub cost: f64,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FullPreparation {
    pub fields: Vec<FullField>,
    pub row_to_field: Vec<i32>,
}
/// Full-memory enumeration tables. Rows with the same ordered
/// support share one field whose cost is the sum of their duals.
pub fn prepare_full_memory(
    info: &mut CutResourceInfo,
    prep: &mut FullPreparation,
    g: &Graph,
    cuts: &[FullInput],
    backward: bool,
) -> bool {
    let vertices = g.vertices.len();
    let mut groups: BTreeMap<Vec<(usize, i32)>, Vec<usize>> = BTreeMap::new();
    for (row, cut) in cuts.iter().enumerate() {
        groups.entry(cut.ordered_support.clone()).or_default().push(row);
    }
    let mut next = FullPreparation {
        fields: vec![],
        row_to_field: vec![-1; cuts.len()],
    };
    let mut vertex_tables: Vec<Vec<VertexSupport>> = vec![vec![]; vertices];
    let mut forward_tables: Vec<Vec<ArcTransition>> = vec![vec![]; g.arcs.len()];
    let mut backward_tables: Vec<Vec<ArcTransition>> = vec![vec![]; g.arcs.len()];
    let mut position = 0i32;
    for rows in groups.values() {
        let cut = &cuts[rows[0]];
        let mut cost = 0.0;
        for &row in rows {
            cost = cost + cuts[row].dual;
        }
        let bits = lim_memory_bits(cut.denominator);
        if position % 32 + bits > 31 {
            position = (position / 32 + 1) * 32;
        }
        let f = FullField {
            mask: lim_memory_mask(cut.denominator),
            denominator: cut.denominator,
            word: position / 32,
            shift: position % 32,
            covering: cut.covering,
            index: next.fields.len() as i32,
            cost,
        };
        position += bits;
        next.fields.push(f);
        if position > (FULL_WORDS * 32) as i32 {
            return false;
        }
        for &row in rows {
            next.row_to_field[row] = f.index;
        }
        for vertex in 0..vertices {
            let increment = cut.vertex_coefficient[vertex];
            if increment == 0 {
                continue;
            }
            vertex_tables[vertex].push(VertexSupport {
                mask: f.mask,
                denominator: f.denominator,
                word: f.word,
                shift: f.shift,
                increment,
                covering: f.covering,
                cost: f.cost,
            });
        }
        for (i, a) in g.arcs.iter().enumerate() {
            let Some(&increment) = cut.arc_coefficient.get(&a.id) else {
                continue;
            };
            let t = ArcTransition {
                mask: f.mask,
                denominator: f.denominator,
                source_word: f.word,
                source_shift: f.shift,
                target_word: f.word,
                target_shift: f.shift,
                increment,
                covering: f.covering,
                overflow_cost: f.cost,
            };
            forward_tables[i].push(t);
            if backward {
                backward_tables[i].push(t);
            }
        }
    }
    info.vertices.resize(vertices, VertexInfo::default());
    if info.vertex_cut_potential.len() < vertices {
        info.vertex_cut_potential.resize(vertices, 0.0);
    }
    for (v, table) in vertex_tables.into_iter().enumerate() {
        info.vertices[v].full_memory_support = table;
    }
    let arcs = g.arcs.len();
    if info.forward.len() < arcs {
        info.forward.resize(arcs, ArcInfo::default());
    }
    if info.backward.len() < arcs {
        info.backward.resize(arcs, ArcInfo::default());
    }
    for t in &mut info.forward {
        t.full_memory.clear();
    }
    for t in &mut info.backward {
        t.full_memory.clear();
    }
    for (arc, table) in forward_tables.into_iter().enumerate() {
        info.forward[arc].full_memory = table;
    }
    if backward {
        for (arc, table) in backward_tables.into_iter().enumerate() {
            info.backward[arc].full_memory = table;
        }
    }
    *prep = next;
    true
}
/// Full-memory join cost: every field participates.
pub fn full_memory_concatenation_cost(
    info: &CutResourceInfo,
    prep: &FullPreparation,
    vertex: usize,
    forward: (&[u32], f64),
    backward: (&[u32], f64),
    cost: f64,
) -> f64 {
    let potential = info.vertex_cut_potential.get(vertex).copied().unwrap_or(0.0);
    let total = (forward.1 + backward.1) + potential;
    let mut cost = total + cost;
    for f in &prep.fields {
        let index = f.word as usize;
        let a = arithmetic_shift_word(forward.0[index], f.shift) & f.mask;
        let b = arithmetic_shift_word(backward.0[index], f.shift) & f.mask;
        let sum = a.wrapping_add(b) as i32;
        if f.covering {
            if -f.denominator <= sum {
                cost = cost - f.cost;
            }
        } else if f.denominator <= sum {
            cost = f.cost + cost;
        }
    }
    cost
}
/// project_enumeration_label: limited-memory words of a full-memory state at
/// a vertex.
pub fn project_words(
    limited: &LimitedPreparation,
    full: &FullPreparation,
    vertex: usize,
    words: &[u32; FULL_WORDS],
) -> [u32; LIMITED_WORDS] {
    let mut result = [0u32; LIMITED_WORDS];
    for (row, positions) in limited.positions.iter().enumerate() {
        let target = positions[vertex];
        if target < 0 {
            continue;
        }
        let f = &full.fields[full.row_to_field[row] as usize];
        let value = arithmetic_shift_word(words[f.word as usize], f.shift) & f.mask;
        result[(target / 32) as usize] |= value << ((target as u32) % 32);
    }
    result
}
/// Rank-one costs of an
/// enumerated path from physical incidence (no memory), the byte-wide
/// accumulator included.
pub fn enumerated_path_cut_cost(cuts: &[LimitedInput], vertices: &[usize], arc_ids: &[i32], mut cost: f64) -> f64 {
    for cut in cuts {
        let mut sum: u8 = 0;
        let mut add = |value: i32| {
            let v = if cut.covering && !cut.type2 { -value } else { value };
            sum = (sum as i32 + v) as u8;
        };
        for &v in vertices {
            add(cut.vertex_coefficient[v]);
        }
        for id in arc_ids {
            if let Some(&c) = cut.arc_coefficient.get(id) {
                add(c);
            }
        }
        let coefficient = if cut.covering && !cut.type2 {
            if sum != 0 {
                (sum as i32 - 1) / (-cut.denominator) + 1
            } else {
                0
            }
        } else {
            sum as i32 / cut.denominator
        };
        if coefficient != 0 {
            cost = cost + coefficient as f64 * cut.dual;
        }
    }
    cost
}

/// build_cut_inputs: active registry cuts in pricing order. Zero-dual and
/// retired cuts allocate no label state. The cost of a cut is minus its
/// master dual.
pub fn build_inputs(registry: &CutRegistry, g: &Graph) -> (Vec<LimitedInput>, Vec<FullInput>) {
    let graph = g.graph_id.max(0);
    let vertices = g.vertices.len();
    let mut order = registry.pricing_order.clone();
    let mut ordered = vec![false; registry.cuts.len()];
    for &i in &order {
        ordered[i] = true;
    }
    order.extend((0..registry.cuts.len()).filter(|&i| !ordered[i]));
    let mut limited = vec![];
    let mut full = vec![];
    for index in order {
        let cut = &registry.cuts[index];
        if registry.retired.contains(&index) || cut.dual.abs() <= 1e-6 {
            continue;
        }
        let memory = cut.memory_for_graph(graph);
        let covering = cut.class == CutClass::Covering;
        let sign = if covering { -1 } else { 1 };
        let coefficient: Vec<i32> = (0..vertices).map(|v| cut.weights[v] * sign).collect();
        let arc_coefficient: BTreeMap<i32, i32> = cut.arc_weights.iter().map(|(&a, &w)| (a, w * sign)).collect();
        limited.push(LimitedInput {
            covering,
            type2: cut.native_type2,
            arc_memory: cut.memory_kind == MemoryKind::Arc,
            denominator: sign * cut.denominator,
            dual: -cut.dual,
            vertex_coefficient: coefficient.clone(),
            arc_coefficient: arc_coefficient.clone(),
            memory_vertices: (0..vertices).filter(|&v| memory[v]).collect(),
            memory_arcs: cut.arc_memory_for_graph(graph),
        });
        let mut ordered_support = match &cut.ordered_support {
            Some(s) => s.clone(),
            None => {
                let mut s: Vec<(usize, i32)> = (0..vertices)
                    .filter(|&v| cut.weights[v] != 0)
                    .map(|v| (v, cut.weights[v]))
                    .collect();
                s.extend(cut.arc_weights.iter().map(|(&a, &w)| (vertices + a as usize, w)));
                s
            }
        };
        for entry in &mut ordered_support {
            entry.1 *= sign;
        }
        full.push(FullInput {
            ordered_support,
            denominator: sign * cut.denominator,
            covering,
            dual: -cut.dual,
            vertex_coefficient: coefficient,
            arc_coefficient,
        });
    }
    (limited, full)
}
