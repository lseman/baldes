//! Relaxed selection memory: check-only vertex masks that the master narrows.
use super::*;

/// One relaxed selection resource. Each arc carries its head's vertex mask;
/// the head narrows the domain only once it is remembered.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct RelaxedSelection {
    heads: Vec<usize>,
    masks: Vec<u64>,
    remembered: Vec<bool>,
}
impl RelaxedSelection {
    pub(super) fn new<const D: usize>(g: &Graph<D>, vertices: &[u64]) -> Self {
        Self {
            heads: g.arcs.iter().map(|a| a.head).collect(),
            masks: g.arcs.iter().map(|a| vertices[a.head]).collect(),
            remembered: vec![false; g.vertices.len()],
        }
    }
    /// The domain after entering `arc`'s head, or None when it misses the
    /// head's mask. A forgotten head checks without narrowing.
    #[inline]
    pub(super) fn enter(&self, domain: u64, arc: usize) -> Option<u64> {
        let kept = domain & self.masks[arc];
        if kept == 0 {
            return None;
        }
        Some(if self.remembered[self.heads[arc]] {
            kept
        } else {
            domain
        })
    }
}

impl CoupledKernel {
    /// Whether any selection resource has relaxed vertex masks.
    pub(crate) fn selection_relaxed(&self) -> bool {
        self.selection_memory.iter().any(Option::is_some)
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

    /// Remember the heads of each relaxed resource's violating windows in
    /// `arcs`. A window [i, j] is shortest: its masks intersect to empty and
    /// no proper subwindow does. With its heads remembered (arc masks always
    /// narrow), forward and backward extension reject it, and a join that
    /// splits it intersects to empty, so the relaxation excludes this route
    /// (findViolatingSubPaths / forbidSubPath). Returns whether any vertex
    /// started narrowing.
    pub(crate) fn strengthen_selection(&mut self, arcs: &[usize]) -> bool {
        let windows: Vec<(usize, Vec<usize>)> = self
            .selection_masks()
            .enumerate()
            .filter(|(r, _)| self.selection_memory[*r].is_some())
            .map(|(r, masks)| (r, violating_arcs(&masks, arcs)))
            .collect();
        let mut changed = false;
        for (r, window) in windows {
            let relaxed = self.selection_memory[r].as_mut().expect("relaxed resource");
            for arc in window {
                let head = relaxed.heads[arc];
                changed |= !std::mem::replace(&mut relaxed.remembered[head], true);
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

/// Every arc of each shortest empty window [i, j] of exact masks.
fn violating_arcs(masks: &ExactMasks<'_>, arcs: &[usize]) -> Vec<usize> {
    let Some(exact) = arcs
        .iter()
        .map(|&arc| masks.arc(arc))
        .collect::<Option<Vec<_>>>()
    else {
        return vec![];
    };
    let mut window = vec![];
    for end in 0..exact.len() {
        // The latest start whose window up to `end` is already empty.
        let mut domain = masks.start & exact[end];
        let mut start = end;
        while domain != 0 && start > 0 {
            start -= 1;
            domain &= exact[start];
        }
        if domain != 0 {
            continue;
        }
        // Shortest only if the window without its last arc is nonempty.
        let shorter = exact[start..end].iter().fold(masks.start, |d, &m| d & m);
        if start == end || shorter != 0 {
            window.extend_from_slice(&arcs[start..=end]);
        }
    }
    window
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
    fn relaxed_selection_keeps_arc_masks_exact() {
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
        assert_eq!(
            graph.coupled_kernel().unwrap().replay(&[0, 1, 2]).unwrap(),
            None
        );
    }

    #[cfg(feature = "clp")]
    #[test]
    fn relaxed_vertex_selection_matches_exact_master_and_never_accepts_improper_incumbent() {
        use crate::master::root::{self, RootOptions, RootProblem};
        for direction in [0., 1., 2.] {
            for relaxed in [false, true] {
                let graph = graph(relaxed);
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
