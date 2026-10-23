use super::*;

/// Full-memory cut field.
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
/// Full-memory enumeration tables. Rows with the same ordered support,
/// denominator and rounding direction share one field and sum their duals.
pub fn prepare_full_memory<const D: usize>(
    info: &mut CutResourceInfo,
    prep: &mut FullPreparation,
    g: &Graph<D>,
    cuts: &[FullInput],
    kpath: &[FullKPathInput],
    backward: bool,
) -> bool {
    let vertices = g.vertices.len();
    let mut groups: BTreeMap<_, Vec<usize>> = BTreeMap::new();
    let mut kpath_groups: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    for (row, cut) in cuts.iter().enumerate() {
        groups
            .entry((cut.ordered_support.clone(), cut.denominator, cut.covering))
            .or_default()
            .push(row);
    }
    for (row, cut) in kpath.iter().enumerate() {
        kpath_groups
            .entry(cut.ordered_support.clone())
            .or_default()
            .push(row);
    }
    let mut next = FullPreparation {
        fields: vec![],
        row_to_field: vec![-1; cuts.len() + kpath.len()],
    };
    let mut vertex_tables: Vec<Vec<VertexSupport>> = vec![vec![]; vertices];
    let mut forward_tables: Vec<Vec<ArcTransition>> = vec![vec![]; g.arcs.len()];
    let mut backward_tables: Vec<Vec<ArcTransition>> = vec![vec![]; g.arcs.len()];
    let mut position = 0i32;
    let transition =
        |f: &FullField, increment: i32, target_word: i32, target_shift: i32| ArcTransition {
            mask: f.mask,
            denominator: f.denominator,
            source_word: f.word,
            source_shift: f.shift,
            target_word,
            target_shift,
            increment,
            covering: f.covering,
            overflow_cost: f.cost,
        };
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
            let t = transition(&f, increment, f.word, f.shift);
            forward_tables[i].push(t);
            if backward {
                backward_tables[i].push(t);
            }
        }
    }
    for rows in kpath_groups.values() {
        let cut = &kpath[rows[0]];
        let mut cost = 0.0;
        for &row in rows {
            cost = cost + kpath[row].dual;
        }
        let f = FullField {
            mask: 1,
            denominator: -2,
            word: position / 32,
            shift: position % 32,
            covering: true,
            index: next.fields.len() as i32,
            cost,
        };
        position += 1;
        next.fields.push(f);
        if position > (FULL_WORDS * 32) as i32 {
            return false;
        }
        for &row in rows {
            next.row_to_field[cuts.len() + row] = f.index;
        }
        for (i, a) in g.arcs.iter().enumerate() {
            let add = |direction: bool, tables: &mut Vec<Vec<ArcTransition>>| {
                let from = cut.vertex_support[if direction { a.tail } else { a.head }];
                let to = cut.vertex_support[if direction { a.head } else { a.tail }];
                if !from && to {
                    tables[i].push(transition(&f, -1, -1, -1));
                } else if from && !to {
                    // Keep the visited bit in this cut's field. The word
                    // index is not its bit offset within that word.
                    tables[i].push(transition(&f, 1, f.word, f.shift));
                }
            };
            add(true, &mut forward_tables);
            if backward {
                add(false, &mut backward_tables);
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
    let potential = info
        .vertex_cut_potential
        .get(vertex)
        .copied()
        .unwrap_or(0.0);
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
/// Rank-one and strong k-path costs of an
/// enumerated path from physical incidence (no memory), the byte-wide
/// accumulator included.
pub fn enumerated_path_cut_cost(
    cuts: &[LimitedInput],
    kpath: &[LimitedKPathInput],
    vertices: &[usize],
    arc_ids: &[i32],
    mut cost: f64,
) -> f64 {
    for cut in cuts {
        let mut sum: u8 = 0;
        let mut add = |value: i32| {
            let v = if cut.covering && !cut.type2 {
                -value
            } else {
                value
            };
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
    for cut in kpath {
        if vertices
            .iter()
            .any(|&v| cut.support.get(v).copied().unwrap_or(false))
        {
            cost = cost + cut.dual;
        }
    }
    cost
}
