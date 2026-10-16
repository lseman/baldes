//! Dynamic NG cycle strengthening, preserving native path/cycle priority.
use super::{ng::PreparedNg, Graph};
use crate::{invalid, Result};
use std::collections::BTreeSet;
#[derive(Clone, Debug)]
pub struct FractionalPath {
    pub arcs: Vec<i32>,
    pub value: f64,
}
#[derive(Clone, Debug)]
pub struct Cycle {
    pub arcs: Vec<usize>,
    pub element: usize,
    pub begin: usize,
    pub end: usize,
    pub covered: bool,
}
#[derive(Clone, Debug)]
pub struct Controls {
    pub mode: i32,
    pub max_vertex_size: usize,
    pub max_average_size: usize,
    pub max_columns: usize,
    pub max_cycle_size: i32,
    pub mirror: bool,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            mode: 1,
            max_vertex_size: 15,
            max_average_size: 15,
            max_columns: 100,
            max_cycle_size: 5,
            mirror: false,
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct Counts {
    pub columns_forbidden: usize,
    pub phase_one: usize,
    pub phase_two: usize,
    pub non_elementary: usize,
    pub elementary: usize,
}
pub fn cycles(g: &Graph, path: &[i32]) -> Result<Vec<Cycle>> {
    let mut last = vec![None; g.element_count];
    let physical: Vec<_> = path
        .iter()
        .map(|&id| {
            g.arcs
                .iter()
                .position(|a| a.id == id)
                .ok_or_else(|| invalid("unknown NG path arc"))
        })
        .collect::<Result<_>>()?;
    let mut cycles = vec![];
    for (position, &id) in physical.iter().enumerate() {
        let a = &g.arcs[id];
        for &e in &g.vertices[a.tail].elements {
            last[e] = Some(position);
        }
        let entered: BTreeSet<_> = g.vertices[a.head]
            .elements
            .iter()
            .chain(&a.elements)
            .copied()
            .collect();
        for e in entered {
            if let Some(previous) = last[e] {
                if previous != position {
                    cycles.push(Cycle {
                        arcs: physical[previous..=position].to_vec(),
                        element: e,
                        begin: previous,
                        end: position,
                        covered: false,
                    });
                }
            }
            last[e] = Some(position);
        }
    }
    Ok(cycles)
}
pub fn strengthen(g: &mut Graph, paths: &[FractionalPath], controls: &Controls) -> Result<Counts> {
    g.validate()?;
    let prepared = PreparedNg::new(g)?;
    let mut counts = Counts::default();
    let mut neighbourhood: Vec<BTreeSet<usize>> = prepared
        .local_elements
        .iter()
        .map(|row| row.iter().copied().collect())
        .collect();
    if controls.mode == 1 {
        let smallest = neighbourhood
            .iter()
            .map(BTreeSet::len)
            .min()
            .unwrap_or(controls.max_vertex_size);
        let total: usize = neighbourhood.iter().map(BTreeSet::len).sum();
        if smallest >= controls.max_vertex_size
            || total / g.vertices.len() >= controls.max_average_size
        {
            return Ok(counts);
        }
    }
    let original = g.clone();
    let mut order: Vec<_> = (0..paths.len()).collect();
    order.sort_by(|&a, &b| {
        let x = &paths[a];
        let y = &paths[b];
        if x.value > y.value + 1e-6 {
            std::cmp::Ordering::Less
        } else if x.value < y.value - 1e-6 {
            std::cmp::Ordering::Greater
        } else {
            x.arcs.cmp(&y.arcs)
        }
    });
    for index in order {
        let mut cycles = cycles(g, &paths[index].arcs)?;
        if cycles.is_empty() {
            counts.elementary += 1;
            continue;
        }
        counts.non_elementary += 1;
        if counts.columns_forbidden >= controls.max_columns {
            continue;
        }
        cycles.sort_by_key(|c| c.arcs.len());
        let mut forbidden = 0;
        let mut selected_phase = 0;
        for phase in 1..=2 {
            for c in 0..cycles.len() {
                if phase == 1 && (controls.max_cycle_size as usize) <= cycles[c].arcs.len() {
                    break;
                }
                let covered = (0..c).any(|i| {
                    cycles[c].begin <= cycles[i].begin
                        && cycles[i].end <= cycles[c].end
                        && cycles[i].covered
                });
                if covered {
                    cycles[c].covered = true;
                }
                if cycles[c].covered {
                    continue;
                }
                let cycle = &cycles[c];
                let e = cycle.element;
                let mut members: BTreeSet<_> =
                    cycle.arcs.iter().skip(1).map(|&a| g.arcs[a].tail).collect();
                let first = &g.arcs[cycle.arcs[0]];
                let last = &g.arcs[*cycle.arcs.last().unwrap()];
                if g.vertices[first.tail].elements.contains(&e) {
                    members.insert(first.tail);
                }
                if g.vertices[last.head].elements.contains(&e) {
                    members.insert(last.head);
                }
                if members.iter().any(|&v| {
                    neighbourhood[v].len() == controls.max_vertex_size
                        && !neighbourhood[v].contains(&e)
                }) {
                    continue;
                }
                for &v in &members {
                    neighbourhood[v].insert(e);
                    if controls.mode == 1 && !g.vertices[v].memory.contains(&e) {
                        g.vertices[v].memory.push(e);
                    }
                }
                if controls.mode != 1 {
                    for &id in &cycle.arcs {
                        if !g.arcs[id].memory.contains(&e) {
                            g.arcs[id].memory.push(e);
                        }
                        if controls.mirror {
                            let a = &g.arcs[id];
                            let Some(reverse) = g
                                .arcs
                                .iter()
                                .position(|b| b.tail == a.head && b.head == a.tail)
                            else {
                                *g = original;
                                return Err(invalid("NG mirror requires reverse arcs"));
                            };
                            if !g.arcs[reverse].memory.contains(&e) {
                                g.arcs[reverse].memory.push(e);
                            }
                        }
                    }
                }
                cycles[c].covered = true;
                forbidden += 1;
                selected_phase = phase;
                if phase == 2 {
                    break;
                }
            }
            if forbidden > 0 {
                break;
            }
        }
        if forbidden > 0 {
            counts.columns_forbidden += 1;
            if selected_phase == 1 {
                counts.phase_one += 1;
            } else {
                counts.phase_two += 1;
            }
        }
    }
    if PreparedNg::new(g).is_err() {
        *g = original;
        return Ok(Counts {
            columns_forbidden: 0,
            ..counts
        });
    }
    Ok(counts)
}
/// NG memories from a distance matrix: stable distance/id ties, and each
/// physical traversal stops at the first previously populated destination.
pub fn initialize_distances(g: &mut Graph, distances: &[f64], size: usize) -> Result<bool> {
    let n = g.element_count;
    if distances.len()
        != n.checked_mul(n)
            .ok_or_else(|| invalid("NG distance dimension overflow"))?
        || distances.iter().any(|x| x.is_nan())
    {
        return Err(invalid("NG set distance dimensions/value"));
    }
    if n == 0 {
        return Ok(false);
    }
    let mut neighbourhood = vec![];
    for e in 0..n {
        let mut ids: Vec<_> = (0..n).collect();
        ids.sort_by(|&a, &b| {
            distances[e * n + a]
                .total_cmp(&distances[e * n + b])
                .then(a.cmp(&b))
        });
        ids.truncate(size.min(n));
        ids.sort_unstable();
        neighbourhood.push(ids);
    }
    let mut changed = false;
    for vertex in &mut g.vertices {
        if !vertex.memory.is_empty() {
            break;
        }
        let mut memory = BTreeSet::new();
        for &e in &vertex.elements {
            memory.extend(&neighbourhood[e]);
        }
        vertex.memory = memory.into_iter().collect();
        changed |= !vertex.memory.is_empty();
    }
    for arc in &mut g.arcs {
        if !arc.memory.is_empty() {
            break;
        }
        let mut memory = BTreeSet::new();
        for &e in &arc.elements {
            memory.extend(&neighbourhood[e]);
        }
        arc.memory = memory.into_iter().collect();
        changed |= !arc.memory.is_empty();
    }
    Ok(changed)
}
