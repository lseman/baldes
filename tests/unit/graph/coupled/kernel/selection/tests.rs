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

fn remembered(kernel: &CoupledKernel, resource: usize) -> Vec<usize> {
    let memory = kernel.selection_memory[resource].as_ref().unwrap();
    (0..memory.remembered.len())
        .filter(|&v| memory.remembered[v])
        .collect()
}

#[test]
fn first_window_schedule_caps_length_and_falls_back_to_the_shortest() {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    let selection = |vertices: Vec<u64>| CoupledResource::Selection {
        initial: 3,
        allowed: vec![3; 3],
        vertices,
        relaxed: true,
    };
    // Resource 0 first fails at vertex 2; its window {1, 2} has length 1.
    // Resource 1 first fails at vertex 3; its window {0..3} has length 3.
    graph
        .set_coupled_resources(&[selection(vec![3, 1, 2, 3]), selection(vec![1, 3, 3, 2])])
        .unwrap();
    let route = [0, 1, 2];
    for (cap, short, long) in [
        (0, vec![1, 2], vec![]),
        (1, vec![1, 2], vec![]),
        (3, vec![1, 2], vec![0, 1, 2, 3]),
    ] {
        let mut kernel = graph.coupled_kernel().unwrap().clone();
        assert!(kernel.strengthen_selection_first_windows(&route, cap));
        assert_eq!(remembered(&kernel, 0), short, "cap {cap}");
        assert_eq!(remembered(&kernel, 1), long, "cap {cap}");
        // One remembered window already excludes the improper route.
        assert!(kernel.replay(&route).unwrap().is_none());
        assert!(!kernel.strengthen_selection_first_windows(&route, cap));
    }
}

#[test]
fn first_window_arc_failure_has_zero_length_and_remembers_the_head() {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::Selection {
            initial: 3,
            allowed: vec![3, 0, 3],
            vertices: vec![],
            relaxed: true,
        }])
        .unwrap();
    let mut kernel = graph.coupled_kernel().unwrap().clone();
    assert!(kernel.strengthen_selection_first_windows(&[0, 1, 2], 0));
    assert_eq!(remembered(&kernel, 0), vec![1, 2]);
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

fn follow(kernel: &CoupledKernel, arcs: &[usize], forward: bool) -> Option<Box<CoupledLabelState>> {
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
    use crate::master::routing::{self, RootOptions, RoutingContext};
    for direction in [0., 1., 2.] {
        for (relaxed, arc_only) in [(false, false), (true, false), (false, true), (true, true)] {
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
            let problem = RoutingContext::set_partitioning(graph, 2.).unwrap();
            let mut options = RootOptions::default();
            options.cg.stabilize = false;
            options
                .pricing_options
                .push(("BALDESuseBidirectionalSearch".into(), direction));
            let output = routing::solve(problem, options).unwrap();
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
