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

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(relaxed: bool) -> Graph<2> {
        let mut graph = Graph::new(4, 1).unwrap();
        for vertex in 0..4 {
            graph.set_bounds(vertex, 0, 0., 3.).unwrap();
        }
        for (tail, head, cost) in [(0, 1, 1.), (1, 2, -1.), (2, 3, 1.), (1, 3, 1.), (0, 2, 1.)] {
            let arc = graph.add_arc(tail, head, cost).unwrap();
            graph.arc_mut(arc).unwrap().consumption[0] = 1.;
        }
        graph
            .set_coupled_resources(&[CoupledResource::Selection {
                initial: 3,
                allowed: vec![3; 5],
                vertices: vec![3, 1, 2, 3],
                relaxed,
            }])
            .unwrap();
        graph
    }

    #[test]
    fn vertex_mask_tightening_excludes_improper_routes_and_terminates() {
        let exact_graph = graph(false);
        let relaxed_graph = graph(true);
        let exact = exact_graph.coupled_kernel().unwrap();
        let mut relaxed = relaxed_graph.coupled_kernel().unwrap().clone();
        let bad = [0, 1, 2];
        assert_eq!(exact.replay(&bad).unwrap(), None);
        assert_eq!(relaxed.replay(&bad).unwrap(), Some(0.));
        assert!(!relaxed.selection_proper(&bad));
        assert!(relaxed.strengthen_selection(&bad));
        assert_eq!(relaxed.replay(&bad).unwrap(), None);
        assert!(!relaxed.strengthen_selection(&bad));
        for route in [&[0, 3][..], &[4, 2][..]] {
            assert!(relaxed.selection_proper(route));
            assert_eq!(relaxed.replay(route).unwrap(), exact.replay(route).unwrap());
        }
    }

    #[test]
    fn forgetting_restores_prior_arc_eliminations_and_tightening_remembers_both_endpoints() {
        let mut graph = graph(true);
        let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
        let CoupledResource::Selection {
            allowed, vertices, ..
        } = &mut definitions[0]
        else {
            unreachable!()
        };
        vertices.fill(3);
        allowed[0] = 1;
        allowed[1] = 2;
        graph.set_coupled_resources(&definitions).unwrap();
        let mut kernel = graph.coupled_kernel().unwrap().clone();
        let route = [0, 1, 2];
        assert_eq!(kernel.replay(&route).unwrap(), Some(0.));
        assert!(!kernel.selection_proper(&route));
        assert!(kernel.strengthen_selection(&route));
        assert_eq!(
            kernel.selection_memory[0].as_ref().unwrap().remembered,
            [true, true, true, false]
        );
        assert_eq!(kernel.replay(&route).unwrap(), None);
        assert!(!kernel.strengthen_selection(&route));
        for split in 0..=route.len() {
            assert!(!accepts_split(&kernel, &route, split));
        }
    }

    fn follow(
        kernel: &CoupledKernel,
        arcs: &[usize],
        forward: bool,
    ) -> Option<Box<CoupledLabelState>> {
        let mut state = kernel.root(forward);
        let mut workspace = CoupledWorkspace::default();
        let path: Vec<_> = if forward {
            arcs.to_vec()
        } else {
            arcs.iter().rev().copied().collect()
        };
        for arc in path {
            state = kernel
                .extend_internal(&state, arc, forward, false, false, &mut workspace)
                .unwrap()?
                .0;
        }
        Some(state)
    }

    fn accepts_split(kernel: &CoupledKernel, arcs: &[usize], split: usize) -> bool {
        match (
            follow(kernel, &arcs[..split], true),
            follow(kernel, &arcs[split..], false),
        ) {
            (Some(f), Some(b)) => {
                let ordinary = kernel.join(&f, &b).unwrap().is_some();
                assert_eq!(
                    ordinary,
                    kernel.join_resource_objective(&f, &b).unwrap().is_some()
                );
                ordinary
            }
            _ => false,
        }
    }

    #[test]
    fn directional_vertex_masks_and_roots_match_native_primitive_cases() {
        let g = graph(true);
        let mut memory = PreparedSelection::new(&g, &[1, 2, 3, 2], 3, true);
        assert_eq!(memory.root(true), 3);
        assert_eq!(memory.root(false), 3);
        // Arc 0 goes from vertex 0 (mask 1) to vertex 1 (mask 2).
        assert_eq!(memory.enter(1, 0, true), None);
        assert_eq!(memory.enter(1, 0, false), Some(3));
        assert_eq!(memory.enter(2, 0, true), Some(3));
        memory.remembered.fill(true);
        assert_eq!(memory.root(true), 1);
        assert_eq!(memory.root(false), 2);
        assert_eq!(memory.enter(1, 0, false), Some(1));
        assert_eq!(memory.enter(3, 0, true), Some(2));
    }

    #[test]
    fn arc_only_relaxation_also_forgets_at_unrestricted_vertices() {
        let mut g = graph(true);
        let mut allowed = vec![3; 5];
        allowed[0] = 1;
        allowed[1] = 2;
        g.set_coupled_resources(&[CoupledResource::Selection {
            initial: 3,
            allowed,
            vertices: vec![],
            relaxed: true,
        }])
        .unwrap();
        let mut kernel = g.coupled_kernel().unwrap().clone();
        assert!(kernel.selection_relaxed());
        assert_eq!(kernel.replay(&[0, 1, 2]).unwrap(), Some(0.));
        assert!(kernel.strengthen_selection(&[0, 1, 2]));
        for split in 0..=3 {
            assert!(!accepts_split(&kernel, &[0, 1, 2], split));
        }
    }

    #[test]
    fn interior_windows_do_not_reapply_source_masks_and_memory_is_per_resource() {
        let mut g = graph(true);
        let definitions = [
            CoupledResource::Selection {
                initial: 3,
                allowed: vec![3; 5],
                vertices: vec![1, 3, 2, 3],
                relaxed: true,
            },
            CoupledResource::Selection {
                initial: 3,
                allowed: vec![3; 5],
                vertices: vec![3; 4],
                relaxed: true,
            },
        ];
        g.set_coupled_resources(&definitions).unwrap();
        let mut kernel = g.coupled_kernel().unwrap().clone();
        assert!(kernel.strengthen_selection(&[0, 1, 2]));
        assert_eq!(
            kernel.selection_memory[0].as_ref().unwrap().remembered,
            [true, true, true, false]
        );
        assert_eq!(
            kernel.selection_memory[1].as_ref().unwrap().remembered,
            [false; 4]
        );
        for split in 0..=3 {
            assert!(!accepts_split(&kernel, &[0, 1, 2], split));
        }
    }

    #[test]
    fn exhaustive_masks_preserve_exact_routes_and_tightening_rejects_bad_routes_at_every_split() {
        let mut g = graph(true);
        let route = [0, 1, 2];
        // Exhaust all two-value vertex masks and two conflicting arc masks.
        for bits in 0..1024usize {
            let vertices = vec![
                1 + (bits & 1) as u64,
                ((bits >> 1) & 3) as u64,
                ((bits >> 3) & 3) as u64,
                ((bits >> 5) & 3) as u64,
            ];
            let mut allowed = vec![3; 5];
            allowed[0] = ((bits >> 7) & 3) as u64;
            allowed[1] = ((bits >> 9) & 1) as u64 + 1;
            // Independent exact complete-route intersection.
            let proper = vertices
                .iter()
                .chain(allowed[..3].iter())
                .fold(3, |a, &b| a & b)
                != 0;
            g.set_coupled_resources(&[CoupledResource::Selection {
                initial: 3,
                allowed,
                vertices,
                relaxed: true,
            }])
            .unwrap();
            let original = g.coupled_kernel().unwrap().clone();
            let mut exact_definitions = original.definitions().to_vec();
            let CoupledResource::Selection { relaxed, .. } = &mut exact_definitions[0] else {
                unreachable!()
            };
            *relaxed = false;
            g.set_coupled_resources(&exact_definitions).unwrap();
            let exact = g.coupled_kernel().unwrap();
            let mut tightened = original.clone();
            if !proper {
                tightened.strengthen_selection(&route);
            }
            for split in 0..=3 {
                assert_eq!(
                    accepts_split(exact, &route, split),
                    proper,
                    "exact {bits}/{split}"
                );
                if proper {
                    assert!(accepts_split(&original, &route, split), "{bits}/{split}");
                }
                assert_eq!(
                    accepts_split(&tightened, &route, split),
                    proper,
                    "{bits}/{split}"
                );
            }
            // Snapshots keep their original per-vertex memory.
            assert!(original.selection_memory[0]
                .as_ref()
                .unwrap()
                .remembered
                .iter()
                .all(|&v| !v));
        }
    }

    #[cfg(feature = "clp")]
    #[test]
    fn relaxed_vertex_selection_matches_exact_master_and_never_accepts_improper_incumbent() {
        use crate::master::root::{self, RootOptions, RootProblem};
        for direction in [0., 1., 2.] {
            for (relaxed, arc_only) in [(false, false), (true, false), (false, true), (true, true)]
            {
                let mut graph = graph(relaxed);
                if arc_only {
                    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
                    let CoupledResource::Selection {
                        allowed, vertices, ..
                    } = &mut definitions[0]
                    else {
                        unreachable!()
                    };
                    vertices.clear();
                    allowed[0] = 1;
                    allowed[1] = 2;
                    graph.set_coupled_resources(&definitions).unwrap();
                }
                let kernel = graph.coupled_kernel().unwrap().clone();
                let problem = RootProblem::set_partitioning(graph, 2.).unwrap();
                let mut options = RootOptions::default();
                options.cg.stabilize = false;
                options
                    .pricing_options
                    .push(("BALDESuseBidirectionalSearch".into(), direction));
                let output = root::solve(problem, options).unwrap();
                assert!((output.master.lp - 4.).abs() < 1e-8, "{output:?}");
                assert!((output.incumbent - 4.).abs() < 1e-8, "{output:?}");
                for (route, value) in output.route_values() {
                    if value > 1e-9 {
                        let arcs: Vec<_> = route.arcs.iter().map(|&arc| arc as usize).collect();
                        assert!(kernel.selection_proper(&arcs));
                    }
                }
            }
        }
    }
}
