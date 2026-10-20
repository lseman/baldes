use super::*;

pub(super) fn lim_memory_bits(denominator: i32) -> i32 {
    if (0..8).contains(&denominator) {
        DENOMINATOR_BITS[denominator as usize]
    } else if (-8..0).contains(&denominator) {
        BITMASK[(8 + denominator) as usize] as i32
    } else {
        panic!("limited-memory denominator")
    }
}
pub(super) fn lim_memory_mask(denominator: i32) -> u32 {
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
pub fn prepare_vertices<const D: usize>(
    info: &mut CutResourceInfo,
    prep: &mut LimitedPreparation,
    g: &Graph<D>,
    cuts: &[LimitedInput],
    kpath: &[LimitedKPathInput],
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
    let mut index = 0usize;
    for cut in cuts {
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
        index += 1;
    }
    for cut in kpath {
        if cut.arc_memory {
            return false;
        }
        let mut position = vec![-1i32; n];
        for &vertex in &cut.memory_vertices {
            let at = next[vertex];
            next[vertex] += 1;
            position[vertex] = at;
            info.vertices[vertex].dominance.push(DominanceDescriptor {
                mask: 1,
                denominator: -2,
                word: at >> 5,
                shift: at & 31,
                subtract_cost: true,
                cost: cut.dual,
            });
            prep.dominance_cut[vertex].push(index);
        }
        prep.positions.push(position);
        index += 1;
    }
    info.vertex_cut_potential = vec![0.0; n];
    for vertex in 0..n {
        for cut in kpath {
            if cut.support.get(vertex).copied().unwrap_or(false) {
                info.vertex_cut_potential[vertex] -= cut.dual;
            }
        }
    }
    next.iter().all(|&used| {
        let words = if used <= 0 { 1 } else { (used - 1) / 32 + 1 };
        words <= LIMITED_WORDS as i32
    })
}
/// Arc transitions of one direction.
pub fn prepare_arcs<const D: usize>(
    info: &mut CutResourceInfo,
    prep: &mut LimitedPreparation,
    g: &Graph<D>,
    cuts: &[LimitedInput],
    kpath: &[LimitedKPathInput],
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
    let index_of: BTreeMap<i32, usize> =
        g.arcs.iter().enumerate().map(|(i, a)| (a.id, i)).collect();
    // Arcs entering each vertex (forward; leaving it backward), ordered by
    // the other end and then graph order.
    let n = g.vertices.len();
    let mut into: Vec<Vec<usize>> = vec![vec![]; n];
    let mut order: Vec<usize> = (0..g.arcs.len()).collect();
    order.sort_by_key(|&i| {
        if forward {
            g.arcs[i].tail
        } else {
            g.arcs[i].head
        }
    });
    for i in order {
        let to = if forward {
            g.arcs[i].head
        } else {
            g.arcs[i].tail
        };
        into[to].push(i);
    }
    let coordinates = |at: i32| if at < 0 { (-1, -1) } else { (at >> 5, at % 32) };
    let mut push = |arc: usize,
                    owner: usize,
                    mask: u32,
                    denominator: i32,
                    source: i32,
                    target: i32,
                    increment: i32,
                    covering: bool,
                    cost: f64| {
        let (sw, ss) = coordinates(source);
        let (tw, ts) = coordinates(target);
        tables[arc].ordinary.push(ArcTransition {
            mask,
            denominator,
            source_word: sw,
            source_shift: ss,
            target_word: tw,
            target_shift: ts,
            increment,
            covering,
            overflow_cost: cost,
        });
        owners[arc].push(owner);
    };
    let mut cut_index = 0usize;
    for cut in cuts {
        let owner = cut_index;
        let position = &prep.positions[cut_index];
        cut_index += 1;
        let mask = lim_memory_mask(cut.denominator);
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
                    push(
                        i,
                        owner,
                        mask,
                        cut.denominator,
                        source,
                        target,
                        increment,
                        cut.covering,
                        cut.dual,
                    );
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
                push(
                    i,
                    owner,
                    mask,
                    cut.denominator,
                    source,
                    target,
                    increment,
                    cut.covering,
                    cut.dual,
                );
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
            push(
                i,
                owner,
                mask,
                cut.denominator,
                -1,
                -1,
                increment,
                cut.covering,
                cut.dual,
            );
        }
    }
    for cut in kpath {
        let owner = cut_index;
        let position = &prep.positions[cut_index];
        cut_index += 1;
        for &memory in &cut.memory_vertices {
            let target = position[memory];
            for &i in &into[memory] {
                let a = &g.arcs[i];
                let other = if forward { a.tail } else { a.head };
                let source = position[other];
                if source < 0 && !cut.support[other] {
                    continue;
                }
                push(
                    i,
                    owner,
                    1,
                    -2,
                    source,
                    target,
                    if source < 0 { 1 } else { 0 },
                    true,
                    cut.dual,
                );
            }
        }
        for support in 0..n {
            if !cut.support[support] {
                continue;
            }
            for &i in &into[support] {
                let a = &g.arcs[i];
                let other = if forward { a.tail } else { a.head };
                if cut.support[other] {
                    continue;
                }
                push(i, owner, 1, -2, position[other], -1, -1, true, cut.dual);
            }
        }
    }
}
/// Dual-only change: every cost copied from a dual takes the new dual.
pub fn update_duals(
    info: &mut CutResourceInfo,
    prep: &LimitedPreparation,
    cuts: &[LimitedInput],
    kpath: &[LimitedKPathInput],
    forward: bool,
    backward: bool,
) {
    let dual = |cut: usize| {
        if cut < cuts.len() {
            cuts[cut].dual
        } else {
            kpath[cut - cuts.len()].dual
        }
    };
    for (v, vertex) in info.vertices.iter_mut().enumerate() {
        if v >= prep.dominance_cut.len() {
            break;
        }
        for (i, d) in vertex.dominance.iter_mut().enumerate() {
            d.cost = dual(prep.dominance_cut[v][i]);
        }
        for (i, s) in vertex.support.iter_mut().enumerate() {
            s.cost = dual(prep.support_cut[v][i]);
        }
    }
    let n = info.vertex_cut_potential.len();
    info.vertex_cut_potential = vec![0.0; n];
    for vertex in 0..n {
        for cut in kpath {
            if cut.support.get(vertex).copied().unwrap_or(false) {
                info.vertex_cut_potential[vertex] -= cut.dual;
            }
        }
    }
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
                t.overflow_cost = dual(owners[i]);
            }
        }
    }
}
