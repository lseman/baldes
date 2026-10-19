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
