use super::*;

#[test]
fn full_memory_rank_one_rows_with_different_denominators_remain_distinct() {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for vertex in 0..3 {
        graph.add_arc(vertex, vertex + 1, 0.).unwrap();
    }
    let inputs: Vec<_> = [(2, 2.), (3, 3.)]
        .into_iter()
        .map(|(denominator, dual)| FullInput {
            ordered_support: vec![(1, 1), (2, 1)],
            denominator,
            covering: false,
            dual,
            vertex_coefficient: vec![0, 1, 1, 0],
            arc_coefficient: Default::default(),
        })
        .collect();
    let mut info = CutResourceInfo::default();
    let mut prep = FullPreparation::default();
    assert!(prepare_full_memory(
        &mut info,
        &mut prep,
        &graph,
        &inputs,
        &[],
        true
    ));
    // floor(2/2)*2 + floor(2/3)*3 = 2. Combining the two duals
    // into a denominator-2 field would incorrectly charge 5.
    for forward in [false, true] {
        let mut words = [0; FULL_WORDS];
        let mut cost = 0.;
        for step in 0..4 {
            let vertex = if forward { step } else { 3 - step };
            apply_vertex_support(
                &mut words,
                &mut cost,
                &info.vertices[vertex].full_memory_support,
            );
        }
        assert_eq!(cost, 2.);
    }
    // Identical rows can still share their state and add their duals.
    let mut same_rows = inputs.clone();
    same_rows[1].denominator = 2;
    assert!(prepare_full_memory(
        &mut info,
        &mut prep,
        &graph,
        &same_rows,
        &[],
        true
    ));
    assert_eq!(prep.fields.len(), 1);
    assert_eq!(prep.fields[0].cost, 5.);
}

#[test]
fn full_memory_kpath_fields_do_not_overwrite_other_cuts() {
    for last_support in [2, 33] {
        let outside = last_support + 1;
        let sink = outside + 1;
        let mut graph = Graph::<2>::new(sink + 1, 0).unwrap();
        let path = [0, last_support, outside, 1, sink];
        for edge in path.windows(2) {
            graph.add_arc(edge[0], edge[1], 0.).unwrap();
        }
        let inputs: Vec<_> = (1..=last_support)
            .map(|vertex| {
                let mut support = vec![false; sink + 1];
                support[vertex] = true;
                FullKPathInput {
                    ordered_support: vec![vertex],
                    dual: -(vertex as f64),
                    vertex_support: support,
                }
            })
            .collect();
        let mut info = CutResourceInfo::default();
        let mut prep = FullPreparation::default();
        assert!(prepare_full_memory(
            &mut info,
            &mut prep,
            &graph,
            &[],
            &inputs,
            true
        ));
        // Both traversal directions must count each visited support once,
        // independently of unrelated cuts and of packed-word boundaries.
        for forward in [false, true] {
            let mut words = [0; FULL_WORDS];
            let mut cost = 0.;
            let indices: Vec<_> = if forward {
                (0..graph.arcs.len()).collect()
            } else {
                (0..graph.arcs.len()).rev().collect()
            };
            for index in indices {
                let table = if forward {
                    &info.forward[index]
                } else {
                    &info.backward[index]
                };
                change_full_memory_along_arc(&mut words, &mut cost, &table.full_memory);
            }
            assert_eq!(cost, -(last_support as f64) - 1.);
        }
    }
}

#[test]
fn machine_kpath_exit_layout_preserves_recovered_destination_shift() {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    graph.add_arc(2, 3, 0.).unwrap();
    let inputs: Vec<_> = [1, 2]
        .into_iter()
        .map(|vertex| {
            let mut support = vec![false; 4];
            support[vertex] = true;
            FullKPathInput {
                ordered_support: vec![vertex],
                dual: -1.,
                vertex_support: support,
            }
        })
        .collect();
    let mut source = CutResourceInfo::default();
    let mut machine = CutResourceInfo::default();
    assert!(prepare_full_memory(&mut source, &mut FullPreparation::default(), &graph, &[], &inputs, true));
    assert!(prepare_full_memory_machine(&mut machine, &mut FullPreparation::default(), &graph, &[], &inputs, true));
    let source_exit = &source.forward[0].full_memory[0];
    let machine_exit = &machine.forward[0].full_memory[0];
    assert_eq!((source_exit.source_shift, source_exit.target_shift), (1, 1));
    assert_eq!((machine_exit.source_shift, machine_exit.target_shift), (1, 0));
    // The pinned helper produces these packed words in both directions for
    // the corresponding descriptor; its additional cost remains zero.
    for (initial, expected) in [(0, 1), (1, 1), (2, 2), (3, 2)] {
        let mut words = [0; FULL_WORDS];
        let mut additional = 0.;
        words[0] = initial;
        change_full_memory_along_arc(&mut words, &mut additional, &[*machine_exit]);
        assert_eq!((words[0], additional), (expected, 0.));
    }
}

#[test]
fn omitted_words_match_zero_extended_states_in_all_readers() {
    let prefix = [1, 0];
    let mut padded = [0; LIMITED_WORDS];
    padded[..2].copy_from_slice(&prefix);
    let mut other = [0; LIMITED_WORDS];
    other[0] = 2;
    other[18] = 2;
    let descriptors = vec![
        DominanceDescriptor {
            word: 0,
            mask: 3,
            denominator: 3,
            cost: 0.25,
            ..Default::default()
        },
        DominanceDescriptor {
            word: 18,
            mask: 3,
            denominator: 3,
            cost: 1.25,
            ..Default::default()
        },
    ];
    for reverse in [false, true] {
        for initial in [-2., -0.75, -0.1] {
            let mut full_gap = initial;
            let mut short_gap = initial;
            let (a, b, sa, sb): (&[u32], &[u32], &[u32], &[u32]) = if reverse {
                (&other, &padded, &other, &prefix)
            } else {
                (&padded, &other, &prefix, &other)
            };
            assert_eq!(
                packed_dominate(&descriptors, a, b, &mut full_gap),
                packed_dominate(&descriptors, sa, sb, &mut short_gap)
            );
            assert_eq!(full_gap.to_bits(), short_gap.to_bits());
        }
    }
    let mut info = CutResourceInfo::default();
    info.vertices.resize_with(1, VertexInfo::default);
    info.vertices[0].dominance = descriptors;
    assert_eq!(
        concatenation_cost(&info, 0, (&padded, 0.1), (&other, 0.2), -4.).to_bits(),
        concatenation_cost(&info, 0, (&prefix, 0.1), (&other, 0.2), -4.).to_bits()
    );
    let transitions = [
        ArcTransition {
            source_word: 18,
            target_word: 19,
            mask: 1,
            denominator: 2,
            increment: 1,
            overflow_cost: 0.5,
            ..Default::default()
        },
        ArcTransition {
            source_word: 0,
            target_word: 0,
            mask: 1,
            denominator: 2,
            increment: 1,
            overflow_cost: 0.5,
            ..Default::default()
        },
    ];
    let mut full_cost = -0.;
    let mut short_cost = -0.;
    let expected = changed_along_arc(&padded, &mut full_cost, &transitions);
    let actual = changed_along_arc(&prefix, &mut short_cost, &transitions);
    assert_eq!(actual, expected);
    assert_eq!(actual[19], 1);
    assert_eq!(full_cost.to_bits(), short_cost.to_bits());
}
#[test]
fn structural_comparison_ignores_only_duals() {
    use super::*;
    let limited = LimitedInput {
        covering: false,
        type2: false,
        arc_memory: true,
        denominator: 3,
        dual: 1.,
        vertex_coefficient: vec![1, 0, 2],
        arc_coefficient: [(7, 1)].into_iter().collect(),
        memory_vertices: vec![0, 2],
        memory_arcs: [4].into_iter().collect(),
    };
    let mut other = limited.clone();
    other.dual = f64::NAN;
    assert!(limited.same_structure(&other));
    other.denominator = 2;
    assert!(!limited.same_structure(&other));
    other = limited.clone();
    other.memory_arcs.insert(8);
    assert!(!limited.same_structure(&other));
    let kpath = LimitedKPathInput {
        dual: 2.,
        arc_memory: false,
        support: vec![true, false],
        memory_vertices: vec![1],
    };
    let mut other = kpath.clone();
    other.dual = -0.;
    assert!(kpath.same_structure(&other));
    other.arc_memory = true;
    assert!(!kpath.same_structure(&other));
    let full = FullInput {
        dual: 1.,
        ordered_support: vec![(1, 2)],
        denominator: 3,
        covering: false,
        vertex_coefficient: vec![0, 2],
        arc_coefficient: BTreeMap::new(),
    };
    let mut other = full.clone();
    other.dual = -2.;
    assert!(full.same_structure(&other));
    other.covering = true;
    assert!(!full.same_structure(&other));
    let full_kpath = FullKPathInput {
        dual: 2.,
        ordered_support: vec![0],
        vertex_support: vec![true],
    };
    let mut other = full_kpath.clone();
    other.dual = 3.;
    assert!(full_kpath.same_structure(&other));
    other.ordered_support.push(1);
    assert!(!full_kpath.same_structure(&other));
}
