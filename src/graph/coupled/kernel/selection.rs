//! Directional selection masks and per-vertex forgetting memory.
use super::*;

/// Vertex masks and whole-resource memory, independently for each selection.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PreparedSelection {
    heads: Vec<usize>,
    tails: Vec<usize>,
    masks: Vec<u64>,
    initial: u64,
    source: usize,
    sink: usize,
    relaxed: bool,
    remembered: Vec<bool>,
}
impl PreparedSelection {
    pub(super) fn new<const D: usize>(
        g: &Graph<D>,
        vertices: &[u64],
        initial: u64,
        relaxed: bool,
    ) -> Self {
        Self {
            heads: g.arcs.iter().map(|a| a.head).collect(),
            tails: g.arcs.iter().map(|a| a.tail).collect(),
            masks: if vertices.is_empty() {
                vec![u64::MAX; g.vertices.len()]
            } else {
                vertices.to_vec()
            },
            initial,
            source: g.source,
            sink: g.sink,
            relaxed,
            remembered: vec![!relaxed; g.vertices.len()],
        }
    }
    fn at_vertex(&self, domain: u64, vertex: usize) -> u64 {
        if self.remembered[vertex] {
            domain & self.masks[vertex]
        } else {
            // Forget all earlier eliminations in this resource's value slice.
            self.initial
        }
    }
    pub(super) fn root(&self, forward: bool) -> u64 {
        self.at_vertex(self.initial, if forward { self.source } else { self.sink })
    }
    /// Check the arriving vertex before forgetting. Backward arcs arrive at
    /// their tail. Arc incompatibilities are always checked before a reset.
    #[inline]
    pub(super) fn enter(&self, domain: u64, arc: usize, forward: bool) -> Option<u64> {
        let vertex = if forward {
            self.heads[arc]
        } else {
            self.tails[arc]
        };
        if domain & self.masks[vertex] == 0 {
            return None;
        }
        Some(self.at_vertex(domain, vertex))
    }
}

impl CoupledKernel {
    /// Whether any selection resource has relaxed vertex masks.
    pub(crate) fn selection_relaxed(&self) -> bool {
        self.selection_memory
            .iter()
            .flatten()
            .any(|memory| memory.relaxed)
    }

    /// The exact selection rule: every resource keeps a value along `arcs`
    /// through the source mask, every arc mask and every head's mask.
    pub(crate) fn selection_proper(&self, arcs: &[usize]) -> bool {
        self.selection_masks().all(|masks| {
            arcs.iter()
                .try_fold(masks.start, |domain, &arc| {
                    let next = domain & masks.arc(arc)?;
                    (next != 0).then_some(next)
                })
                .is_some()
        })
    }

    /// Remember every vertex, including both endpoints, of each minimal
    /// violating vertex interval. Stored domains then retain the contradiction
    /// in either direction and across every forward/backward split.
    pub(crate) fn strengthen_selection(&mut self, arcs: &[usize]) -> bool {
        let windows: Vec<(usize, Vec<usize>)> = self
            .selection_masks()
            .enumerate()
            .filter(|(r, _)| {
                self.selection_memory[*r]
                    .as_ref()
                    .is_some_and(|m| m.relaxed)
            })
            .map(|(r, masks)| (r, violating_vertices(&masks, arcs)))
            .collect();
        let mut changed = false;
        for (r, window) in windows {
            let relaxed = self.selection_memory[r].as_mut().expect("relaxed resource");
            for vertex in window {
                changed |= !std::mem::replace(&mut relaxed.remembered[vertex], true);
            }
        }
        if changed {
            // Caches keyed by this kernel must not survive the new relaxation.
            self.identity = std::sync::Arc::new(());
        }
        changed
    }

    /// The reference's tightening schedule for one improper path. Each
    /// relaxed resource contributes at most one window: the minimal one ending
    /// at the path's first violation (findViolatingSubPaths). Windows no
    /// longer than `max_length` are remembered; when none is, the shortest one
    /// (first on ties) is, so the path is always excluded afterwards
    /// (strengthenCurrentRelaxation). Window length counts the vertex
    /// positions between the failing items, as the reference does.
    pub(crate) fn strengthen_selection_first_windows(
        &mut self,
        arcs: &[usize],
        max_length: i64,
    ) -> bool {
        let windows: Vec<(usize, FirstWindow)> = self
            .selection_masks()
            .enumerate()
            .filter(|(r, _)| {
                self.selection_memory[*r]
                    .as_ref()
                    .is_some_and(|m| m.relaxed)
            })
            .filter_map(|(r, masks)| Some((r, first_violating_window(&masks, arcs)?)))
            .collect();
        let fits = |window: &FirstWindow| window.length as i64 <= max_length;
        let fallback = if windows.iter().any(|(_, window)| fits(window)) {
            None
        } else {
            // min_by_key keeps the first of equal lengths.
            windows
                .iter()
                .enumerate()
                .min_by_key(|(_, (_, window))| window.length)
                .map(|(index, _)| index)
        };
        let mut changed = false;
        for (index, (r, window)) in windows.iter().enumerate() {
            if !fits(window) && fallback != Some(index) {
                continue;
            }
            let relaxed = self.selection_memory[*r]
                .as_mut()
                .expect("relaxed resource");
            for &vertex in &window.vertices {
                changed |= !std::mem::replace(&mut relaxed.remembered[vertex], true);
            }
        }
        if changed {
            self.identity = std::sync::Arc::new(());
        }
        changed
    }

    fn selection_masks(&self) -> impl Iterator<Item = ExactMasks<'_>> {
        let heads = &self.selection_heads;
        self.definitions.iter().filter_map(move |r| match r {
            CoupledResource::Selection {
                initial,
                allowed,
                vertices,
                ..
            } => Some(ExactMasks {
                initial: *initial,
                source: self.selection_source,
                start: initial
                    & vertices
                        .get(self.selection_source)
                        .copied()
                        .unwrap_or(u64::MAX),
                allowed,
                vertices,
                heads,
            }),
            _ => None,
        })
    }
}

/// Exact masks of one selection resource.
struct ExactMasks<'a> {
    initial: u64,
    source: usize,
    start: u64,
    allowed: &'a [u64],
    vertices: &'a [u64],
    heads: &'a [usize],
}
impl ExactMasks<'_> {
    fn arc(&self, arc: usize) -> Option<u64> {
        let head = *self.heads.get(arc)?;
        Some(self.allowed.get(arc)? & self.vertices.get(head).copied().unwrap_or(u64::MAX))
    }
}

/// Minimal empty intervals of visited vertices and their intervening arcs.
/// Interior windows start from the resource universe, without reapplying the
/// source vertex mask. Remember both endpoints even when an arc causes failure.
fn violating_vertices(masks: &ExactMasks<'_>, arcs: &[usize]) -> Vec<usize> {
    let mut vertices = vec![masks.source];
    for &arc in arcs {
        let Some(&head) = masks.heads.get(arc) else {
            return vec![];
        };
        if masks.allowed.get(arc).is_none() {
            return vec![];
        }
        vertices.push(head);
    }
    let vertex_mask = |position: usize| {
        masks
            .vertices
            .get(vertices[position])
            .copied()
            .unwrap_or(u64::MAX)
    };
    let mut remembered = vec![];
    for end in 0..vertices.len() {
        let mut domain = masks.initial & vertex_mask(end);
        let mut start = end;
        while domain != 0 && start > 0 {
            start -= 1;
            domain &= masks.allowed[arcs[start]] & vertex_mask(start);
        }
        if domain != 0 {
            continue;
        }
        let mut shorter = masks.initial & vertex_mask(start);
        for i in start..end.saturating_sub(1) {
            shorter &= masks.allowed[arcs[i]] & vertex_mask(i + 1);
        }
        if start == end || shorter != 0 {
            remembered.extend_from_slice(&vertices[start..=end]);
        }
    }
    remembered
}

/// A minimal violating window: its vertices, both endpoints included, and its
/// length in the reference's convention.
struct FirstWindow {
    vertices: Vec<usize>,
    length: usize,
}

/// The path alternates vertex and arc masks: item 2p is vertex position p and
/// item 2i+1 is arc i. Scan forward to the first item that empties the
/// domain, then backward from it to the latest start that still empties it.
/// An arc item stands at its tail position for the length, while the window
/// remembers its head as well. Interior windows start from the universe.
fn first_violating_window(masks: &ExactMasks<'_>, arcs: &[usize]) -> Option<FirstWindow> {
    let mut vertices = vec![masks.source];
    for &arc in arcs {
        masks.allowed.get(arc)?;
        vertices.push(*masks.heads.get(arc)?);
    }
    let mask = |item: usize| {
        if item % 2 == 0 {
            masks
                .vertices
                .get(vertices[item / 2])
                .copied()
                .unwrap_or(u64::MAX)
        } else {
            masks.allowed[arcs[item / 2]]
        }
    };
    let mut domain = masks.initial;
    let failure = (0..=2 * arcs.len()).find(|&item| {
        domain &= mask(item);
        domain == 0
    })?;
    let mut domain = masks.initial;
    let start = (0..=failure).rev().find(|&item| {
        domain &= mask(item);
        domain == 0
    })?;
    Some(FirstWindow {
        vertices: vertices[start / 2..=failure.div_ceil(2)].to_vec(),
        length: failure / 2 - start / 2,
    })
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/selection/tests.rs"]
mod tests;
