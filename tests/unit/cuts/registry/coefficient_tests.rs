use super::*;

#[test]
fn dense_packing_matches_graph_support_for_walks_and_memory_policies() {
    let mut dense = RankOneCut {
        weights: vec![0, 1, -1, 4],
        graph_support: BTreeMap::new(),
        graph_pricing_support: BTreeMap::new(),
        native_kpath_support: None,
        memory: vec![],
        graph_dependent_memory: false,
        graph_memories: BTreeMap::new(),
        memory_kind: MemoryKind::Vertex,
        arc_memory: BTreeSet::new(),
        graph_arc_memories: BTreeMap::new(),
        arc_weights: BTreeMap::new(),
        visit_once: false,
        native_type2: false,
        denominator: 2,
        dual: 0.0,
        class: CutClass::Packing,
        rhs: 0,
        kind: -1,
        id: 0,
        ordered_support: None,
    };
    // A sparse graph-specific support forces the general implementation.
    // Include repeats, out-of-range vertices, signed/large weights, and
    // memory breaks; a coefficient increments once per nonzero weight.
    for denominator in [1, 2, 3] {
        dense.denominator = denominator;
        for mask in 0..16 {
            dense.memory = (0..4).map(|v| mask & (1 << v) != 0).collect();
            for policy in 0..3 {
                dense.graph_dependent_memory = policy != 0;
                dense.graph_memories.clear();
                if policy == 1 {
                    dense.graph_memories.insert(7, dense.memory.clone());
                }
                for native in [false, true] {
                    dense.native_type2 = native;
                    dense.class = if native {
                        CutClass::Covering
                    } else {
                        CutClass::Packing
                    };
                    let mut sparse = dense.clone();
                    sparse
                        .graph_support
                        .insert(7, dense.weights.iter().copied().enumerate().collect());
                    for code in 0..3125usize {
                        let mut digits = code;
                        let mut tail = 0;
                        let mut route = vec![];
                        for id in 0..5 {
                            let head = digits % 5;
                            digits /= 5;
                            route.push((tail, head, id));
                            tail = head;
                        }
                        for enumerated in [false, true] {
                            assert_eq!(
                                coefficient(&dense, 7, &route, enumerated).unwrap(),
                                coefficient(&sparse, 7, &route, enumerated).unwrap()
                            );
                        }
                    }
                    assert_eq!(coefficient(&dense, 7, &[], false).unwrap(), 0);
                    for enumerated in [false, true] {
                        let broken = [(0, 1, 0), (2, 3, 1)];
                        assert_eq!(
                            coefficient(&dense, 7, &broken, enumerated).map_err(|e| e.to_string()),
                            coefficient(&sparse, 7, &broken, enumerated).map_err(|e| e.to_string())
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn rank_one_rounds_inherit_only_active_arc_memories() {
    for graph_dependent in [false, true] {
        for retired in [false, true] {
            let mut graph = Graph::<2>::new(6, 1).unwrap();
            graph.graph_id = 7;
            for tail in 0..6 {
                for head in 0..6 {
                    if tail != head {
                        graph.add_arc(tail, head, 1.0).unwrap();
                    }
                }
            }
            let remembered = graph
                .arcs
                .iter()
                .find(|a| a.tail == 4 && a.head == 2)
                .unwrap()
                .id;
            let mut registry = CutRegistry::default();
            registry.parameters.max_rank = 3;
            registry.parameters.graph_id = 7;
            registry.parameters.memory_type = MemoryType::Arc;
            let mut columns: Vec<_> = [vec![1, 4, 2], vec![1, 3], vec![2, 3]]
                .into_iter()
                .map(|vertices| FractionalColumn {
                    vertices,
                    value: 0.5,
                    graph_id: 7,
                    enumerated: false,
                })
                .collect();
            // Physical vertex IDs differ from the separator's row IDs.
            let customers = [3, 1, 2];
            assert_eq!(registry.separate(&graph, &customers, &columns).unwrap(), 1);
            assert!(registry.cuts[0]
                .arc_memory_for_graph(7)
                .contains(&remembered));
            registry.cuts[0].graph_dependent_memory = graph_dependent;
            if retired {
                registry.retire(0);
            }
            let previous = registry.cuts[0].clone();
            columns[0].vertices = vec![1, 2];
            assert_eq!(registry.separate(&graph, &[2, 3, 1], &columns).unwrap(), 1);
            assert_eq!(registry.cuts[0], previous);
            assert_eq!(registry.cuts[1].weights, previous.weights);
            assert_eq!(
                registry.cuts[1]
                    .arc_memory_for_graph(7)
                    .contains(&remembered),
                !retired
            );
            assert_eq!(registry.coefficient(&graph, 1, &[1, 2], false).unwrap(), 1);
            assert_eq!(
                registry.coefficient(&graph, 1, &[1, 4, 2], false).unwrap(),
                i32::from(!retired)
            );
        }
    }
}
