//! Remembered element sets, arc NG information and NG transitions.
//! Native local coordinate 15 is reserved; at most 15 live elements per vertex.
use super::Graph;
use crate::{invalid, Result};
use std::collections::BTreeSet;
#[derive(Clone, Debug, Default)]
pub struct Transition {
    pub forbidden: u16,
    pub base: u16,
    pub transfer: Vec<(u8, u8)>,
}
impl Transition {
    pub fn extend(&self, before: u16, vertex_bits: u16) -> Option<u16> {
        if before & self.forbidden != 0 {
            return None;
        }
        let mut next = self.base;
        for &(source, target) in &self.transfer {
            let target = 1u16 << target;
            next = (next & !target)
                | if before & (1u16 << source) != 0 {
                    target
                } else {
                    0
                };
        }
        Some((next | vertex_bits) & !(1 << 15))
    }
}
#[derive(Debug)]
pub struct PreparedNg {
    pub vertex_bits: Vec<u16>,
    pub local_elements: Vec<Vec<usize>>,
    pub forward: Vec<Transition>,
    pub backward: Vec<Transition>,
}
impl PreparedNg {
    pub fn new(g: &Graph) -> Result<Self> {
        g.validate()?;
        let n = g.vertices.len();
        let e = g.element_count;
        let checked = |values: &[usize]| -> Result<BTreeSet<usize>> {
            if values.iter().any(|&id| id >= e) {
                return Err(invalid("NG element out of range"));
            }
            Ok(values.iter().copied().collect())
        };
        let mut member = Vec::with_capacity(n);
        for v in &g.vertices {
            let mut m = checked(&v.memory)?;
            m.extend(checked(&v.elements)?);
            if g.elementary {
                m.extend(0..e);
            }
            member.push(m);
        }
        let mut incoming = vec![BTreeSet::new(); n];
        let mut outgoing = incoming.clone();
        let mut remembered = Vec::with_capacity(g.arcs.len());
        for a in &g.arcs {
            let mut keep: BTreeSet<_> = member[a.tail]
                .intersection(&member[a.head])
                .copied()
                .collect();
            keep.extend(checked(&a.memory)?);
            keep.extend(checked(&a.elements)?);
            outgoing[a.tail].extend(&keep);
            incoming[a.head].extend(&keep);
            remembered.push(keep);
        }
        let mut local = Vec::with_capacity(n);
        let mut vertex_bits = Vec::with_capacity(n);
        for v in 0..n {
            let mut own = member[v].clone();
            own.extend(incoming[v].intersection(&outgoing[v]));
            if own.len() > 15 {
                return Err(invalid("native NG local memory exceeds fifteen elements"));
            }
            let mut positions = vec![None; e];
            for (i, &id) in own.iter().enumerate() {
                positions[id] = Some(i as u8);
            }
            let bits = g.vertices[v]
                .elements
                .iter()
                .fold(0, |bits, &id| bits | (1 << positions[id].unwrap()));
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
                    let seen: BTreeSet<_> = g.vertices[to]
                        .elements
                        .iter()
                        .chain(&a.elements)
                        .copied()
                        .collect();
                    let mut t = Transition::default();
                    for &id in &seen {
                        if let Some(bit) = local[from][id] {
                            t.forbidden |= 1 << bit;
                        }
                    }
                    for &id in &a.elements {
                        if let Some(bit) = local[to][id] {
                            t.base |= 1 << bit;
                        }
                    }
                    for &id in keep {
                        if seen.contains(&id) {
                            continue;
                        }
                        if let (Some(s), Some(tgt)) = (local[from][id], local[to][id]) {
                            t.transfer.push((s, tgt));
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
