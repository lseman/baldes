//! Product of the load relaxation and a projection of the NG automaton.
//! Every true NG continuation projects to a walk here. Unselected NG bits,
//! other resources, and future nonnegative packing penalties are relaxed.
use super::*;

const MAX_SUBSET_SIZE: usize = 6;
const MAX_TABLE_CELLS: usize = 4 * 1024 * 1024;
const MAX_TRANSITIONS: usize = 1_000_000;

#[derive(Clone, Debug)]
pub(super) struct SubsetCompletion {
    // Dense selected-subset bit -> local NG bit at each physical vertex.
    bits: Vec<Vec<(usize, NgMask)>>,
    vertex_bits: Vec<NgMask>,
    size: usize,
    offsets: Vec<usize>,
    width: usize,
    prefix: Vec<f64>,
    suffix: Vec<f64>,
}
impl SubsetCompletion {
    pub(super) fn prepare<const D: usize>(
        g: &Graph<D>,
        ng: &PreparedNg,
        resource: usize,
        bidirectional: bool,
        requested: usize,
        started: Instant,
        limit: std::time::Duration,
    ) -> Option<Self> {
        let mut size = requested.min(MAX_SUBSET_SIZE).min(g.element_count);
        while size > 0 {
            let states = 1usize << size;
            let cells = g
                .vertices
                .len()
                .checked_mul(states)?
                .checked_mul(LOAD_LEVELS + 2)?
                .checked_mul(if bidirectional { 2 } else { 1 })?;
            if cells <= MAX_TABLE_CELLS && g.arcs.len().checked_mul(states)? <= MAX_TRANSITIONS {
                break;
            }
            size -= 1;
        }
        if size == 0 {
            return None;
        }
        let mut present = vec![false; g.element_count];
        for &e in ng.local_elements.iter().flatten() {
            present[e] = true;
        }
        // A reduced-cost/resource score selects the customers offering the
        // largest possible improvement. Selection affects strength, not safety.
        let mut score = vec![0f64; g.element_count];
        for arc in &g.arcs {
            let reward = (-arc.reduced_cost).max(0.) / arc.consumption[resource].max(1e-9);
            for &e in g.vertices[arc.head].elements.iter().chain(&arc.elements) {
                score[e] = score[e].max(reward);
            }
        }
        let mut selected: Vec<_> = (0..g.element_count)
            .filter(|&e| present[e] && score[e] > 0.)
            .collect();
        selected.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
        selected.truncate(size);
        size = selected.len();
        if size == 0 {
            return None;
        }
        let states = 1usize << size;
        let bits: Vec<Vec<(usize, NgMask)>> = ng
            .local_elements
            .iter()
            .map(|local| {
                selected
                    .iter()
                    .filter_map(|e| local.iter().position(|id| id == e))
                    .enumerate()
                    .map(|(dense, bit)| (1 << dense, 1 << bit))
                    .collect()
            })
            .collect();
        // Only subsets representable at this vertex need a state. The same
        // selected element may have a different local coordinate elsewhere;
        // PreparedNg's transfer map carries it between the dense state spaces.
        let mut offsets = Vec::with_capacity(g.vertices.len() + 1);
        offsets.push(0);
        for map in &bits {
            offsets.push(offsets.last().copied().unwrap() + (1usize << map.len()));
        }
        let width = *offsets.last()?;
        let mut encoding = vec![0 as NgMask; width];
        for (v, map) in bits.iter().enumerate() {
            for mask in 0..1usize << map.len() {
                for &(global, local) in map {
                    if mask & global != 0 {
                        encoding[offsets[v] + mask] |= local;
                    }
                }
            }
        }
        let scale = LOAD_LEVELS as f64
            / g.vertices
                .iter()
                .map(|v| v.bounds[resource].upper)
                .fold(0., f64::max);
        let build = |forward: bool| -> Option<Vec<f64>> {
            let mut edges = Vec::with_capacity(g.arcs.len() * states);
            for (a, arc) in g.arcs.iter().enumerate() {
                if a % 128 == 0 && started.elapsed() >= limit {
                    return None;
                }
                let (from, to, transition) = if forward {
                    (arc.tail, arc.head, &ng.forward[a])
                } else {
                    (arc.head, arc.tail, &ng.backward[a])
                };
                let units = multiply_down(arc.consumption[resource], scale)?
                    .floor()
                    .max(0.) as usize;
                for mask in 0..1usize << bits[from].len() {
                    let Some(next) =
                        transition.extend(encoding[offsets[from] + mask], ng.vertex_bits[to])
                    else {
                        continue;
                    };
                    let next_mask = Self::project(&bits[to], next);
                    edges.push((
                        offsets[from] + mask,
                        offsets[to] + next_mask,
                        units,
                        arc.reduced_cost,
                    ));
                }
            }
            let terminal = if forward { g.sink } else { g.source };
            let terminals: Vec<_> = (offsets[terminal]..offsets[terminal + 1]).collect();
            LoadCompletion::table(width, &terminals, &edges, false, started, limit)
        };
        let suffix = build(true)?;
        let prefix = if bidirectional { build(false)? } else { vec![] };
        Some(Self {
            bits,
            vertex_bits: ng.vertex_bits.clone(),
            size,
            offsets,
            width,
            prefix,
            suffix,
        })
    }

    fn project(bits: &[(usize, NgMask)], memory: NgMask) -> usize {
        bits.iter().fold(0, |mask, &(global, local)| {
            mask | if memory & local != 0 { global } else { 0 }
        })
    }

    pub(super) fn bound<const D: usize>(
        &self,
        label: &StoredLabel<D>,
        forward: bool,
        level: usize,
    ) -> f64 {
        // The sweep tests completion before its vertex step marks the current
        // vertex's elements, so include those bits explicitly in this query.
        let memory = label.ng | self.vertex_bits[label.vertex];
        let mask = Self::project(&self.bits[label.vertex], memory);
        let table = if forward { &self.suffix } else { &self.prefix };
        table[level * self.width + self.offsets[label.vertex] + mask]
    }

    pub(super) fn size(&self) -> usize {
        self.size
    }
}
