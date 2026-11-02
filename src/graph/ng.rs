//! Remembered element sets, arc NG information and NG transitions.
//! The highest native local coordinate is reserved, so each vertex holds at most
//! `NG_MEMORY_CAPACITY` live elements: 15 by default, 31 with `ng-memory-31`
//! and 63 with `ng-memory-63` (the larger size wins when both are enabled).
use super::Graph;
use crate::{invalid, Result};
use std::collections::BTreeSet;
mod sets;
use sets::{CompactSet, ElementSet};
/// Local NG memory word of a label.
#[cfg(feature = "ng-memory-63")]
pub type NgMask = u64;
/// Local NG memory word of a label.
#[cfg(all(feature = "ng-memory-31", not(feature = "ng-memory-63")))]
pub type NgMask = u32;
/// Local NG memory word of a label.
#[cfg(not(any(feature = "ng-memory-31", feature = "ng-memory-63")))]
pub type NgMask = u16;
/// Live elements per vertex; the highest bit of `NgMask` is reserved.
pub const NG_MEMORY_CAPACITY: usize = NgMask::BITS as usize - 1;
const RESERVED: NgMask = 1 << NG_MEMORY_CAPACITY;
#[derive(Clone, Debug, Default)]
pub struct Transition {
    pub forbidden: NgMask,
    pub base: NgMask,
    pub transfer: Vec<(u8, u8)>,
}
impl Transition {
    pub fn extend(&self, before: NgMask, vertex_bits: NgMask) -> Option<NgMask> {
        if before & self.forbidden != 0 {
            return None;
        }
        let mut next = self.base;
        for &(source, target) in &self.transfer {
            let target: NgMask = 1 << target;
            next = (next & !target)
                | if before & (1 << source) != 0 {
                    target
                } else {
                    0
                };
        }
        Some((next | vertex_bits) & !RESERVED)
    }
}
#[derive(Clone, Debug)]
pub struct PreparedNg {
    pub vertex_bits: Vec<NgMask>,
    pub local_elements: Vec<Vec<usize>>,
    pub forward: Vec<Transition>,
    pub backward: Vec<Transition>,
}
impl PreparedNg {
    pub fn new<const D: usize>(g: &Graph<D>) -> Result<Self> {
        g.validate()?;
        let n = g.vertices.len();
        if g.max_route_arcs().is_some() && !g.hard_walk_groups() {
            return Ok(Self {
                vertex_bits: vec![0; n],
                local_elements: vec![vec![]; n],
                forward: vec![Transition::default(); g.arcs.len()],
                backward: vec![Transition::default(); g.arcs.len()],
            });
        }
        if g.element_count <= CompactSet::CAPACITY {
            Self::prepare::<D, CompactSet>(g)
        } else {
            Self::prepare::<D, BTreeSet<usize>>(g)
        }
    }

    fn prepare<const D: usize, S: ElementSet>(g: &Graph<D>) -> Result<Self> {
        let n = g.vertices.len();
        let e = g.element_count;
        let checked = |values: &[usize]| -> Result<S> {
            if values.iter().any(|&id| id >= e) {
                return Err(invalid("NG element out of range"));
            }
            Ok(values.iter().copied().collect())
        };
        let mut member = Vec::with_capacity(n);
        for v in &g.vertices {
            let mut m = checked(&v.memory)?;
            m.extend_set(&checked(&v.elements)?);
            if g.elementary || g.hard_walk_groups() {
                m.extend_domain(e);
            }
            member.push(m);
        }
        let mut incoming = vec![S::default(); n];
        let mut outgoing = incoming.clone();
        let mut remembered = Vec::with_capacity(g.arcs.len());
        for a in &g.arcs {
            let mut keep = member[a.tail].intersect_set(&member[a.head]);
            keep.extend_set(&checked(&a.memory)?);
            keep.extend_set(&checked(&a.elements)?);
            outgoing[a.tail].extend_set(&keep);
            incoming[a.head].extend_set(&keep);
            remembered.push(keep);
        }
        let mut local = Vec::with_capacity(n);
        let mut vertex_bits = Vec::with_capacity(n);
        for v in 0..n {
            let mut own = member[v].clone();
            own.extend_intersection(&incoming[v], &outgoing[v]);
            if own.cardinality() > NG_MEMORY_CAPACITY {
                return Err(invalid(format!(
                    "native NG local memory exceeds {NG_MEMORY_CAPACITY} elements"
                )));
            }
            let mut positions = vec![None; e];
            for (i, id) in own.elements().enumerate() {
                positions[id] = Some(i as u8);
            }
            let bits = g.vertices[v]
                .elements
                .iter()
                .fold(0 as NgMask, |bits, &id| {
                    bits | (1 << positions[id].unwrap())
                });
            local.push(positions);
            vertex_bits.push(bits);
        }
        let build = |backward: bool| {
            g.arcs
                .iter()
                .zip(&remembered)
                .map(|(a, keep)| {
                    let (from, to) = if backward {
                        (a.head, a.tail)
                    } else {
                        (a.tail, a.head)
                    };
                    let mut t = Transition::default();
                    let mut seen_at_target: NgMask = 0;
                    let mut remember = |id: usize| {
                        if let Some(bit) = local[from][id] {
                            t.forbidden |= 1 << bit;
                        }
                        if let Some(bit) = local[to][id] {
                            seen_at_target |= 1 << bit;
                        }
                    };
                    let seen = g.vertices[to].elements.iter().chain(&a.elements);
                    // Long input slices can contain many duplicate elements.
                    // Deduplicate those before replaying local coordinates.
                    if g.vertices[to]
                        .elements
                        .len()
                        .saturating_add(a.elements.len())
                        > 256
                    {
                        let unique: S = seen.copied().collect();
                        for id in unique.elements() {
                            remember(id);
                        }
                    } else {
                        for &id in seen {
                            remember(id);
                        }
                    }
                    for &id in &a.elements {
                        if let Some(bit) = local[to][id] {
                            t.base |= 1 << bit;
                        }
                    }
                    for id in keep.elements() {
                        if let (Some(s), Some(tgt)) = (local[from][id], local[to][id]) {
                            if seen_at_target & (1 << tgt) == 0 {
                                t.transfer.push((s, tgt));
                            }
                        }
                    }
                    t
                })
                .collect()
        };
        let local_elements = local
            .iter()
            .map(|positions| {
                positions
                    .iter()
                    .enumerate()
                    .filter_map(|(e, p)| p.map(|_| e))
                    .collect()
            })
            .collect();
        Ok(Self {
            local_elements,
            forward: build(false),
            backward: build(true),
            vertex_bits,
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/graph/ng_preparation_tests.rs"]
mod tests;
