//! Labeling-side cut state of ResourceType<SetBasedCutsResources>, from the
//! C++ reference `native/cuts/CutState.h`, `CutPreparation.h` and
//! `enumeration/FullMemoryPreparation.h`. Limited-memory labels carry twenty
//! packed words and an additional cost; enumeration labels sixty words.
//! Vertex tables are indexed by physical vertex, arc tables by the arc's
//! position in `Graph::arcs`.
use crate::cuts::{CutClass, CutRegistry, MemoryKind};
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
    /// Minus the duals of k-path cuts supported at the vertex.
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
#[inline(always)]
fn padded_word(words: &[u32], index: usize) -> u32 {
    match words.get(index) {
        Some(&word) => word,
        None => {
            assert!(index < LIMITED_WORDS, "cut word exceeds the logical state");
            0
        }
    }
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
                (remainder as u32)
                    .wrapping_sub((denominator as u32).wrapping_mul(coefficient as u32))
                    as i32,
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
pub fn change_along_arc(
    words: &mut [u32; LIMITED_WORDS],
    additional: &mut f64,
    transitions: &[ArcTransition],
) {
    *words = changed_along_arc(words, additional, transitions);
}
/// Transition from a retained read-only state into a fresh full scratch state.
pub(crate) fn changed_along_arc(
    words: &[u32],
    additional: &mut f64,
    transitions: &[ArcTransition],
) -> [u32; LIMITED_WORDS] {
    let mut result = [0u32; LIMITED_WORDS];
    for t in transitions {
        if t.source_word < 0 && t.increment == 0 {
            continue;
        }
        let mut remainder = if t.source_word < 0 {
            0
        } else {
            field(
                padded_word(words, t.source_word as usize),
                t.source_shift,
                t.mask,
            )
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
    result
}
/// Vertex supports added in place.
pub fn apply_vertex_support(words: &mut [u32], additional: &mut f64, support: &[VertexSupport]) {
    for f in support {
        let original = words[f.word as usize];
        let remainder =
            (field(original, f.shift, f.mask) as u32).wrapping_add(f.increment as u32) as i32;
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
pub fn change_full_memory_along_arc(
    words: &mut [u32; FULL_WORDS],
    additional: &mut f64,
    transitions: &[ArcTransition],
) {
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
/// updated in descriptor order with an early exit. Compact limited-memory
/// prefixes have an implicit zero suffix up to the twenty-word limit.
pub fn packed_dominate(
    descriptors: &[DominanceDescriptor],
    a: &[u32],
    b: &[u32],
    gap: &mut f64,
) -> bool {
    for f in descriptors {
        let left = field(padded_word(a, f.word as usize), f.shift, f.mask);
        let right = field(padded_word(b, f.word as usize), f.shift, f.mask);
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
    let potential = info
        .vertex_cut_potential
        .get(vertex)
        .copied()
        .unwrap_or(0.0);
    let mut total = (forward.1 + backward.1) + potential;
    total = total + cost;
    let mut cost = total;
    let Some(v) = info.vertices.get(vertex) else {
        return cost;
    };
    for f in &v.dominance {
        let a = field(padded_word(forward.0, f.word as usize), f.shift, f.mask);
        let b = field(padded_word(backward.0, f.word as usize), f.shift, f.mask);
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
/// A limited-memory strong k-path cut restated at vertex level.
#[derive(Clone, Debug, PartialEq)]
pub struct LimitedKPathInput {
    pub dual: f64,
    pub arc_memory: bool,
    pub support: Vec<bool>,
    pub memory_vertices: Vec<usize>,
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
/// FullMemoryKPathInput.
#[derive(Clone, Debug, PartialEq)]
pub struct FullKPathInput {
    pub ordered_support: Vec<usize>,
    pub dual: f64,
    pub vertex_support: Vec<bool>,
}

const BITMASK: [u32; 8] = [0, 1, 1, 3, 3, 7, 7, 7];
const DENOMINATOR_BITS: [i32; 8] = [0, 1, 1, 2, 2, 3, 3, 3];

/// build_cut_inputs: active registry cuts in pricing order. Zero-dual and
/// retired cuts allocate no label state. The cost of a cut is minus its
/// master dual.
pub fn build_inputs<const D: usize>(
    registry: &CutRegistry,
    g: &Graph<D>,
) -> (
    Vec<LimitedInput>,
    Vec<LimitedKPathInput>,
    Vec<FullInput>,
    Vec<FullKPathInput>,
) {
    let graph = g.graph_id.max(0);
    let vertices = g.vertices.len();
    let mut order = registry.pricing_order.clone();
    let mut ordered = vec![false; registry.cuts.len()];
    for &i in &order {
        ordered[i] = true;
    }
    order.extend((0..registry.cuts.len()).filter(|&i| !ordered[i]));
    let mut limited = vec![];
    let mut kpath = vec![];
    let mut full = vec![];
    let mut full_kpath = vec![];
    for index in order {
        let cut = &registry.cuts[index];
        if registry.retired.contains(&index) || cut.dual.abs() <= 1e-6 {
            continue;
        }
        let dual = if cut.visit_once {
            (-cut.dual * 1e8 + 0.5).floor() / 1e8
        } else {
            -cut.dual
        };
        let weights: Vec<i32> = (0..vertices)
            .map(|v| cut.pricing_weight_for_graph(graph, v))
            .collect();
        let ordered_support = cut.pricing_ordered_support_for_graph(graph);
        if cut.visit_once
            && cut.native_kpath_support.is_none()
            && cut.graph_dependent_memory
            && !cut.graph_memories.contains_key(&graph)
        {
            // coefficient zero. Enumerated paths still use full incidence.
            let support: Vec<bool> = (0..vertices).map(|v| weights[v] != 0).collect();
            full_kpath.push(FullKPathInput {
                ordered_support: cut.native_kpath_support.clone().unwrap_or_else(|| {
                    ordered_support.map_or_else(
                        || (0..vertices).filter(|&v| support[v]).collect(),
                        |s| s.iter().map(|&(v, _)| v).collect(),
                    )
                }),
                dual,
                vertex_support: support,
            });
            continue;
        }
        let memory = cut.memory_for_graph(graph);
        if cut.visit_once {
            let support: Vec<bool> = (0..vertices).map(|v| weights[v] != 0).collect();
            let memory_vertices: Vec<usize> = (0..vertices)
                .filter(|&v| {
                    memory.get(v).copied().unwrap_or(false)
                        && (cut.native_kpath_support.is_some() || !support[v])
                })
                .collect();
            kpath.push(LimitedKPathInput {
                dual,
                arc_memory: false,
                support: support.clone(),
                memory_vertices,
            });
            let ordered_support =
                cut.native_kpath_support
                    .clone()
                    .unwrap_or_else(|| match ordered_support {
                        Some(s) => s.iter().map(|&(v, _)| v).collect(),
                        None => (0..vertices).filter(|&v| support[v]).collect(),
                    });
            full_kpath.push(FullKPathInput {
                ordered_support,
                dual,
                vertex_support: support,
            });
            continue;
        }
        let covering = cut.class == CutClass::Covering;
        let sign = if covering { -1 } else { 1 };
        let coefficient: Vec<i32> = (0..vertices).map(|v| weights[v] * sign).collect();
        let arc_coefficient: BTreeMap<i32, i32> = cut
            .arc_weights
            .iter()
            .map(|(&a, &w)| (a, w * sign))
            .collect();
        limited.push(LimitedInput {
            covering,
            type2: cut.native_type2,
            arc_memory: cut.memory_kind == MemoryKind::Arc,
            denominator: sign * cut.denominator,
            dual,
            vertex_coefficient: coefficient.clone(),
            arc_coefficient: arc_coefficient.clone(),
            memory_vertices: (0..vertices)
                .filter(|&v| memory.get(v).copied().unwrap_or(false))
                .collect(),
            memory_arcs: cut.arc_memory_for_graph(graph),
        });
        let mut ordered_support = match ordered_support {
            Some(s) => s.clone(),
            None => {
                let mut s: Vec<(usize, i32)> = (0..vertices)
                    .filter(|&v| weights[v] != 0)
                    .map(|v| (v, weights[v]))
                    .collect();
                s.extend(
                    cut.arc_weights
                        .iter()
                        .map(|(&a, &w)| (vertices + a as usize, w)),
                );
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
            dual,
            vertex_coefficient: coefficient,
            arc_coefficient,
        });
    }
    (limited, kpath, full, full_kpath)
}

#[cfg(test)]
mod compact_prefix_tests {
    use super::*;
    #[test]
    fn omitted_words_match_zero_extended_states_in_all_readers() {
        let prefix = [1, 0];
        let mut padded = [0; LIMITED_WORDS];
        padded[..2].copy_from_slice(&prefix);
        let mut other = [0; LIMITED_WORDS];
        other[0] = 2;
        other[18] = 2;
        let descriptors = vec![
            DominanceDescriptor {
                word: 0,
                mask: 3,
                denominator: 3,
                cost: 0.25,
                ..Default::default()
            },
            DominanceDescriptor {
                word: 18,
                mask: 3,
                denominator: 3,
                cost: 1.25,
                ..Default::default()
            },
        ];
        for reverse in [false, true] {
            for initial in [-2., -0.75, -0.1] {
                let mut full_gap = initial;
                let mut short_gap = initial;
                let (a, b, sa, sb): (&[u32], &[u32], &[u32], &[u32]) = if reverse {
                    (&other, &padded, &other, &prefix)
                } else {
                    (&padded, &other, &prefix, &other)
                };
                assert_eq!(
                    packed_dominate(&descriptors, a, b, &mut full_gap),
                    packed_dominate(&descriptors, sa, sb, &mut short_gap)
                );
                assert_eq!(full_gap.to_bits(), short_gap.to_bits());
            }
        }
        let mut info = CutResourceInfo::default();
        info.vertices.resize_with(1, VertexInfo::default);
        info.vertices[0].dominance = descriptors;
        assert_eq!(
            concatenation_cost(&info, 0, (&padded, 0.1), (&other, 0.2), -4.).to_bits(),
            concatenation_cost(&info, 0, (&prefix, 0.1), (&other, 0.2), -4.).to_bits()
        );
        let transitions = [
            ArcTransition {
                source_word: 18,
                target_word: 19,
                mask: 1,
                denominator: 2,
                increment: 1,
                overflow_cost: 0.5,
                ..Default::default()
            },
            ArcTransition {
                source_word: 0,
                target_word: 0,
                mask: 1,
                denominator: 2,
                increment: 1,
                overflow_cost: 0.5,
                ..Default::default()
            },
        ];
        let mut full_cost = -0.;
        let mut short_cost = -0.;
        let expected = changed_along_arc(&padded, &mut full_cost, &transitions);
        let actual = changed_along_arc(&prefix, &mut short_cost, &transitions);
        assert_eq!(actual, expected);
        assert_eq!(actual[19], 1);
        assert_eq!(full_cost.to_bits(), short_cost.to_bits());
    }
}

mod full;
pub use full::*;

mod limited;
pub use limited::*;

use limited::{lim_memory_bits, lim_memory_mask};
