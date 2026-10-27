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
fn individually_empty_arc_needs_no_additional_selection_memory() {
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
    assert!(!kernel.strengthen_selection_first_windows(&[0, 1, 2], 0));
    assert!(remembered(&kernel, 0).is_empty());
    for split in 0..=3 {
        assert!(!accepts_split(&kernel, &[0, 1, 2], split));
    }
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

#[test]
fn packed_word_boundaries_and_directional_forgetting_match_metasolver_observations() {
    let fixture = include_str!("../../../../../fixtures/coupled/metasolver-selection-packed.txt");
    let mut observations = 0;
    let mut largest_word_count = 0;
    let mut largest_descriptor_count = 0;
    for line in fixture.lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<_> = line.split_whitespace().collect();
        let forward = fields[1] == "1";
        let relaxed = fields[2] == "1";
        let count: usize = fields[3].parse().unwrap();
        let words: usize = fields[4].parse().unwrap();
        let success = fields[5] == "1";
        assert_eq!(fields.len(), 6 + count * 6);
        let mut graph = Graph::<2>::new(5, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        graph.add_arc(2, 4, 0.).unwrap();
        let target_vertex = if forward { 2 } else { 1 };
        let prefix_arc = if forward { 0 } else { 2 };
        let target_arc = 1;
        let mut before = Vec::new();
        let mut after = Vec::new();
        let mut remembered = Vec::new();
        let definitions: Vec<_> = fields[6..]
            .chunks_exact(6)
            .map(|resource| {
                let width: u32 = resource[0].parse().unwrap();
                let universe = u64::MAX >> (64 - width);
                let domain = u64::from_str_radix(resource[1], 16).unwrap();
                let arc = u64::from_str_radix(resource[2], 16).unwrap();
                let vertex = u64::from_str_radix(resource[3], 16).unwrap();
                remembered.push(resource[4] == "1");
                after.push(u64::from_str_radix(resource[5], 16).unwrap());
                before.push(domain);
                let mut allowed = vec![universe; 3];
                allowed[prefix_arc] = domain;
                allowed[target_arc] = arc;
                let mut vertices = vec![universe; 5];
                vertices[target_vertex] = vertex;
                CoupledResource::Selection {
                    initial: universe,
                    allowed,
                    vertices,
                    relaxed,
                }
            })
            .collect();
        graph.set_coupled_resources(&definitions).unwrap();
        let mut kernel = graph.coupled_kernel().unwrap().clone();
        for (resource, &keep) in remembered.iter().enumerate() {
            let memory = kernel.selection_memory[resource].as_mut().unwrap();
            memory.remembered.fill(true);
            memory.remembered[target_vertex] = keep;
        }
        let root = kernel.root(forward);
        let state = kernel
            .extend(&root, prefix_arc, forward)
            .unwrap()
            .unwrap()
            .0;
        assert_eq!(
            state.selection_domains(),
            before,
            "{} forward={forward}",
            fields[0]
        );
        let actual = kernel.extend(&state, target_arc, forward).unwrap();
        assert_eq!(actual.is_some(), success, "{} forward={forward}", fields[0]);
        if let Some((state, cost)) = actual {
            assert_eq!(cost, 0.);
            assert_eq!(
                state.selection_domains(),
                after,
                "{} forward={forward}",
                fields[0]
            );
        }
        observations += 1;
        largest_word_count = largest_word_count.max(words);
        largest_descriptor_count = largest_descriptor_count.max(count);
    }
    assert_eq!(observations, 648);
    assert_eq!(largest_word_count, 16);
    assert_eq!(largest_descriptor_count, 40);
}

#[cfg(feature = "clp")]
#[test]
fn registered_selection_preparation_matches_actual_masks_and_path_domains() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/coupled/metasolver-selection-preparation.json"
    ))
    .unwrap();
    assert_eq!(
        corpus["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let mut prepared_models = 0;
    let mut rejected_models = 0;
    let mut transitions = 0;
    let mut path_splits = 0;
    for case in corpus["cases"].as_array().unwrap() {
        if !case["valid"].as_bool().unwrap() {
            // These exceed the reference's value/packed-storage limits. They
            // do not imply a matching BALDES descriptor-count restriction.
            assert_eq!(case["preparation"]["return"], 0);
            rejected_models += 1;
            continue;
        }
        assert_eq!(case["preparation"]["return"], 1);
        let state = &case["preparation"]["state"];
        assert_eq!(state["parameters"]["relaxed"], 0);
        let mut graph = Graph::<2>::new(case["vertices"].as_u64().unwrap() as usize, 1).unwrap();
        for vertex in 0..graph.vertices.len() {
            graph.set_bounds(vertex, 0, 0., 4.).unwrap();
        }
        for arc in case["arcs"].as_array().unwrap() {
            let id = graph
                .add_arc(
                    arc["tail"].as_u64().unwrap() as usize,
                    arc["head"].as_u64().unwrap() as usize,
                    arc["cost"].as_f64().unwrap(),
                )
                .unwrap();
            graph.arc_mut(id).unwrap().consumption[0] = 1.;
        }
        let definitions: Vec<_> = case["selection"]
            .as_array()
            .unwrap()
            .iter()
            .map(|resource| {
                let size = resource["values"].as_u64().unwrap();
                let universe = u64::MAX >> (64 - size);
                CoupledResource::Selection {
                    initial: universe,
                    vertices: resource["vertices"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|m| m.as_u64().unwrap())
                        .collect(),
                    allowed: resource["allowed"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|m| m.as_u64().unwrap())
                        .collect(),
                    relaxed: false,
                }
            })
            .collect();
        graph.set_coupled_resources(&definitions).unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let descriptors = state["descriptors"].as_array().unwrap();
        let mapping: Vec<_> = state["vertex_mapping"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["original"].as_u64().unwrap() as usize)
            .collect();
        let mut slots = Vec::new();
        for descriptor in descriptors {
            let resource = case["selection"]
                .as_array()
                .unwrap()
                .iter()
                .position(|r| r["id"] == descriptor["id"])
                .unwrap();
            let word = descriptor["word"].as_u64().unwrap() as usize;
            let offset = descriptor["offset"].as_u64().unwrap() as u32;
            let mask = descriptor["mask"].as_u64().unwrap();
            let memory = kernel.selection_memory[resource].as_ref().unwrap();
            for (internal, &original) in mapping.iter().enumerate() {
                let actual =
                    (state["vertex_masks"][internal][word].as_u64().unwrap() & mask) >> offset;
                assert_eq!(
                    memory.masks[original], actual,
                    "{} resource={resource} vertex={original}",
                    case["name"]
                );
                assert_eq!(
                    state["forget_masks"][internal][word].as_u64().unwrap() & mask,
                    0
                );
                assert!(memory.remembered[original]);
            }
            slots.push((resource, word, offset, mask));
        }
        for forward in [true, false] {
            let root = kernel.root(forward);
            for arc in state["arcs"].as_array().unwrap() {
                let physical = arc["id"].as_u64().unwrap() as usize;
                let arrival = arc[if forward { "head" } else { "tail" }].as_u64().unwrap() as usize;
                let mut expected = vec![0; slots.len()];
                for &(resource, word, offset, mask) in &slots {
                    expected[resource] = ((arc["masks"][word].as_u64().unwrap()
                        & state["vertex_masks"][arrival][word].as_u64().unwrap())
                        & mask)
                        >> offset;
                }
                let actual = kernel.extend(&root, physical, forward).unwrap();
                assert_eq!(
                    actual.is_some(),
                    expected.iter().all(|&d| d != 0),
                    "{} arc={physical} forward={forward}",
                    case["name"]
                );
                if let Some((state, cost)) = actual {
                    assert_eq!(state.selection_domains(), expected);
                    assert_eq!(cost, 0.);
                }
                transitions += 1;
            }
        }
        let mut optimum: f64 = 0.;
        for path in [&[0, 1, 2][..], &[3][..]] {
            let feasible = kernel.selection_proper(path);
            for split in 0..=path.len() {
                assert_eq!(
                    accepts_split(kernel, path, split),
                    feasible,
                    "{} split={split}",
                    case["name"]
                );
                path_splits += 1;
            }
            if feasible {
                optimum = optimum.min(path.iter().map(|&a| graph.arcs[a].cost).sum());
            }
        }
        assert_eq!(
            optimum,
            case["reference_optimum"].as_f64().unwrap(),
            "{}",
            case["name"]
        );
        prepared_models += 1;
    }
    assert_eq!(prepared_models, 58);
    assert_eq!(rejected_models, 6);
    assert_eq!(transitions, 464);
    assert_eq!(path_splits, 348);
}

#[test]
fn selection_first_windows_match_enabled_reference_controls_at_every_join() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/coupled/metasolver-selection-strengthening.json"
    ))
    .unwrap();
    let mut splits = 0;
    for case in corpus["cases"].as_array().unwrap() {
        let number = |v: &serde_json::Value| v.as_u64().unwrap() as usize;
        let mut graph = Graph::<2>::new(number(&case["vertices"]), 1).unwrap();
        for arc in case["arcs"].as_array().unwrap() {
            graph.add_arc(number(&arc[0]), number(&arc[1]), 0.).unwrap();
        }
        let resources: Vec<_> = case["selection"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                let masks = |key: &str| {
                    r[key]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|m| m.as_u64().unwrap())
                        .collect()
                };
                CoupledResource::Selection {
                    initial: (1 << number(&r["values"])) - 1,
                    allowed: masks("allowed"),
                    vertices: masks("vertices"),
                    relaxed: true,
                }
            })
            .collect();
        graph.set_coupled_resources(&resources).unwrap();
        let mut kernel = graph.coupled_kernel().unwrap().clone();
        let paths: Vec<Vec<usize>> = case["paths"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["arcs"].as_array().unwrap().iter().map(number).collect())
            .collect();
        for (index, path) in paths.iter().enumerate() {
            let expected = case["reference"]["windows"][index].as_array().unwrap();
            let actual: Vec<_> = kernel
                .selection_masks()
                .enumerate()
                .filter_map(|(r, masks)| {
                    let window = first_violating_window(&masks, path)?;
                    let vertices: Vec<_> = std::iter::once(0)
                        .chain(path.iter().map(|&a| graph.arcs[a].head))
                        .collect();
                    let first = vertices
                        .iter()
                        .position(|v| *v == window.vertices[0])
                        .unwrap();
                    Some(serde_json::json!([r, first, first + window.length]))
                })
                .collect();
            assert_eq!(&actual, expected, "{} path={index}", case["name"]);
        }
        if case["limit"].as_i64().unwrap() >= 0 {
            for path in &paths {
                kernel.strengthen_selection_first_windows(path, case["cap"].as_i64().unwrap());
            }
        }
        for r in 0..resources.len() {
            let expected: Vec<_> = case["reference"]["remembered"][r]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect();
            assert_eq!(
                remembered(&kernel, r),
                expected,
                "{} resource={r}",
                case["name"]
            );
        }
        for path in paths {
            for split in 0..=path.len() {
                if case["limit"].as_i64().unwrap() >= 0 {
                    assert_eq!(
                        accepts_split(&kernel, &path, split),
                        kernel.selection_proper(&path),
                        "{} split={split}",
                        case["name"]
                    );
                }
                splits += 1;
            }
        }
    }
    assert_eq!(splits, 663);
}

#[test]
fn first_window_memory_preserves_endpoint_constraints_at_every_join() {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    for encoded in 0..3_usize.pow(7) {
        let mut digits = encoded;
        let mut mask = || {
            let value = (digits % 3 + 1) as u64;
            digits /= 3;
            value
        };
        let vertices = (0..4).map(|_| mask()).collect();
        let allowed = (0..3).map(|_| mask()).collect();
        graph
            .set_coupled_resources(&[CoupledResource::Selection {
                initial: 3,
                allowed,
                vertices,
                relaxed: true,
            }])
            .unwrap();
        let mut kernel = graph.coupled_kernel().unwrap().clone();
        let path = [0, 1, 2];
        let proper = kernel.selection_proper(&path);
        kernel.strengthen_selection_first_windows(&path, 0);
        for split in 0..=3 {
            assert_eq!(
                accepts_split(&kernel, &path, split),
                proper,
                "masks={encoded} split={split}"
            );
        }
    }
}
