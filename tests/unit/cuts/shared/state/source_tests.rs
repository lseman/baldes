use super::*;

fn cut(id: i32, visit_once: bool) -> RankOneCut {
    RankOneCut {
        weights: vec![0, 1, 2, 1, 0, 0],
        graph_support: BTreeMap::new(),
        graph_pricing_support: BTreeMap::new(),
        native_kpath_support: None,
        memory: vec![true; 6],
        graph_dependent_memory: false,
        graph_memories: BTreeMap::new(),
        memory_kind: MemoryKind::Arc,
        arc_memory: [3].into_iter().collect(),
        graph_arc_memories: BTreeMap::new(),
        arc_weights: [(3, 1)].into_iter().collect(),
        visit_once,
        native_type2: false,
        denominator: 3,
        dual: -1.234567891,
        class: CutClass::Packing,
        rhs: 1,
        kind: 0,
        id,
        ordered_support: Some(vec![(2, 2), (1, 1), (3, 1)]),
    }
}

fn fixture() -> (CutRegistry, Graph<2>) {
    let mut registry = CutRegistry::default();
    registry.cuts = vec![cut(0, false), cut(1, true), cut(2, true), cut(3, true)];
    registry.cuts[2].native_kpath_support = Some(vec![2, 1]);
    registry.cuts[3].graph_dependent_memory = true;
    // A duplicated cut must contribute twice in exactly this order.
    registry.pricing_order = vec![2, 0, 2, 3];
    (registry, Graph::<2>::new(6, 0).unwrap())
}

fn assert_inputs(a: &Inputs, b: &Inputs) {
    macro_rules! compare {
        ($lane:tt) => {
            assert_eq!(a.$lane.len(), b.$lane.len());
            for (actual, expected) in a.$lane.iter().zip(&b.$lane) {
                assert!(actual.same_structure(expected));
                if expected.dual.is_nan() {
                    assert!(actual.dual.is_nan());
                } else {
                    assert_eq!(actual.dual.to_bits(), expected.dual.to_bits());
                }
            }
        };
    }
    compare!(0);
    compare!(1);
    compare!(2);
    compare!(3);
}

#[test]
fn cached_prices_match_fresh_inputs_across_activation_rounding_and_duplicate_order() {
    let (initial, graph) = fixture();
    for exact in [false, true] {
        let mut registry = initial.clone();
        let (mut inputs, mut source) = build_inputs_cached(&registry, &graph, exact);
        assert_eq!(
            [
                inputs.0.len(),
                inputs.1.len(),
                inputs.2.len(),
                inputs.3.len()
            ],
            [1, 3, 1, 4]
        );
        assert_eq!(inputs.3[0].ordered_support, vec![2, 1]);
        assert_eq!(inputs.3[1].ordered_support, vec![2, 1]);
        let epsilon = 1e-6_f64;
        let near = f64::from_bits(epsilon.to_bits() + 1);
        for index in 0..registry.cuts.len() {
            for dual in [
                -2.345678901,
                near,
                epsilon,
                -epsilon,
                0.,
                -0.,
                1e-9,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                -0.75,
            ] {
                registry.cuts[index].dual = dual;
                if source.matches(&registry, &graph, exact) {
                    let pointers = [
                        inputs.0.as_ptr() as usize,
                        inputs.1.as_ptr() as usize,
                        inputs.2.as_ptr() as usize,
                        inputs.3.as_ptr() as usize,
                    ];
                    assert!(source.refresh_duals(&registry, &mut inputs).is_some());
                    assert_eq!(
                        pointers,
                        [
                            inputs.0.as_ptr() as usize,
                            inputs.1.as_ptr() as usize,
                            inputs.2.as_ptr() as usize,
                            inputs.3.as_ptr() as usize
                        ]
                    );
                } else {
                    (inputs, source) = build_inputs_cached(&registry, &graph, exact);
                }
                let expected = build_inputs_with_duals(&registry, &graph, exact);
                assert_inputs(&inputs, &expected);
            }
        }
        // NaN remains active and unequal to itself, just as fresh PartialEq.
        registry.cuts[0].dual = f64::NAN;
        if !source.matches(&registry, &graph, exact) {
            (inputs, source) = build_inputs_cached(&registry, &graph, exact);
        }
        assert_eq!(source.refresh_duals(&registry, &mut inputs), Some(true));
        assert_eq!(source.refresh_duals(&registry, &mut inputs), Some(true));
    }
}

#[test]
fn every_nondual_cut_field_invalidates_the_source_snapshot() {
    let (registry, graph) = fixture();
    let (_, source) = build_inputs_cached(&registry, &graph, false);
    let edits: [fn(&mut RankOneCut); 19] = [
        |c| c.weights.push(1),
        |c| {
            c.graph_support.insert(0, vec![(1, 2)]);
        },
        |c| {
            c.graph_pricing_support.insert(0, vec![(1, 2)]);
        },
        |c| c.native_kpath_support = Some(vec![1]),
        |c| c.memory[0] = !c.memory[0],
        |c| c.graph_dependent_memory = !c.graph_dependent_memory,
        |c| {
            c.graph_memories.insert(0, vec![true]);
        },
        |c| c.memory_kind = MemoryKind::Vertex,
        |c| {
            c.arc_memory.insert(8);
        },
        |c| {
            c.graph_arc_memories.insert(0, [2].into_iter().collect());
        },
        |c| {
            c.arc_weights.insert(9, 2);
        },
        |c| c.visit_once = !c.visit_once,
        |c| c.native_type2 = !c.native_type2,
        |c| c.denominator += 1,
        |c| c.class = CutClass::Covering,
        |c| c.rhs += 1,
        |c| c.kind += 1,
        |c| c.id += 1,
        |c| c.ordered_support = Some(vec![(3, 2)]),
    ];
    for edit in edits {
        let mut changed = registry.clone();
        edit(&mut changed.cuts[0]);
        assert!(!source.matches(&changed, &graph, false));
    }
}

#[test]
fn order_retirement_graph_extent_and_exact_mode_invalidate_the_source_snapshot() {
    let (mut registry, mut graph) = fixture();
    let (_, source) = build_inputs_cached(&registry, &graph, false);
    assert!(!source.matches(&registry, &graph, true));
    registry.pricing_order.reverse();
    assert!(!source.matches(&registry, &graph, false));
    registry.pricing_order.reverse();
    registry.retired.insert(0);
    assert!(!source.matches(&registry, &graph, false));
    registry.retired.clear();
    graph.graph_id = 7;
    assert!(!source.matches(&registry, &graph, false));
    graph.graph_id = -7; // The builder normalizes all negative graph IDs to zero.
    assert!(source.matches(&registry, &graph, false));
    graph.vertices.push(graph.vertices[0].clone());
    assert!(!source.matches(&registry, &graph, false));
    graph.vertices.pop();
    registry.cuts.push(cut(4, false));
    assert!(!source.matches(&registry, &graph, false));
}

#[test]
fn inactive_source_shapes_are_not_cloned_and_latest_shape_is_built_on_activation() {
    let (mut registry, graph) = fixture();
    registry.cuts[0].dual = 0.;
    registry.retired.insert(1);
    for exact in [false, true] {
        let (mut inputs, source) = build_inputs_cached(&registry, &graph, exact);
        assert_eq!(source.cuts.len(), 2);
        let mut changed = registry.clone();
        changed.cuts[0].weights[1] = 4;
        changed.cuts[0].memory[2] = false;
        changed.cuts[1].ordered_support = Some(vec![(1, 7)]);
        assert!(source.matches(&changed, &graph, exact));
        assert_eq!(source.refresh_duals(&changed, &mut inputs), Some(false));
        let fresh = if exact {
            build_inputs_exact(&changed, &graph)
        } else {
            build_inputs(&changed, &graph)
        };
        assert_eq!(inputs, fresh);
        changed.cuts[0].dual = -2.;
        assert!(!source.matches(&changed, &graph, exact));
        let (activated, next_source) = build_inputs_cached(&changed, &graph, exact);
        assert!(next_source.matches(&changed, &graph, exact));
        let fresh = if exact {
            build_inputs_exact(&changed, &graph)
        } else {
            build_inputs(&changed, &graph)
        };
        assert_eq!(activated, fresh);
        changed.retired.clear();
        assert!(!next_source.matches(&changed, &graph, exact));
    }
}
