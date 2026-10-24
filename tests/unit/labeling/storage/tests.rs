use super::*;

fn check_scan<const PREFIX: bool, const FORWARD: bool, const RESOURCES: usize, const NG: bool>() {
    // A unique qualifying label at every position exercises short scans,
    // vector block boundaries, and resuming after a previous candidate.
    const LEN: usize = 25;
    for use_ng in [false, true] {
        if use_ng && !NG {
            continue;
        }
        for matching in 0..=LEN {
            let mut columns = HotColumns::default();
            for i in 0..LEN {
                let failing = if PREFIX == FORWARD { 2. } else { 0. };
                let resources = if i == matching || use_ng {
                    [1.; 2]
                } else {
                    [failing; 2]
                };
                columns.push(HotLabel {
                    key: 0.,
                    cost: 0.,
                    additional_cost: 0.,
                    resources: resources.into(),
                    hops: 1,
                    ng: if !use_ng || i == matching {
                        1
                    } else if PREFIX {
                        2
                    } else {
                        0
                    },
                });
            }
            for from in [0, 1, 7, 8, 9, 15, 16, 24, LEN] {
                let expected = if matching >= from { matching } else { LEN };
                assert_eq!(
                    columns.next_candidate::<PREFIX, FORWARD, RESOURCES, NG>(from, 1., 1., 1., 1),
                    expected,
                    "prefix={PREFIX} forward={FORWARD} resources={RESOURCES} ng={NG} use_ng={use_ng} matching={matching} from={from}"
                );
            }
        }
    }
    if PREFIX {
        let mut columns = HotColumns::default();
        for _ in 0..LEN {
            columns.push(HotLabel {
                key: 2.,
                cost: 0.,
                additional_cost: 0.,
                resources: [3.; 2].into(),
                hops: 1,
                ng: 2,
            });
        }
        // A key above the prefix threshold stops the scan independently
        // of resource and NG dominance.
        assert_eq!(
            columns.next_candidate::<PREFIX, FORWARD, RESOURCES, NG>(0, 1., 1., 1., 1),
            0
        );
    }
}

#[test]
fn dominance_scan_preserves_first_candidate_across_block_boundaries() {
    macro_rules! check {
        ($p:literal, $f:literal) => {
            check_scan::<$p, $f, 1, false>();
            check_scan::<$p, $f, 1, true>();
            check_scan::<$p, $f, 2, false>();
            check_scan::<$p, $f, 2, true>();
        };
    }
    check!(true, true);
    check!(true, false);
    check!(false, true);
    check!(false, false);
}
#[test]
fn backward_disposable_dominance_subtracts_before_comparing() {
    let graph = Graph::<2>::new(3, 1).unwrap();
    let a = HotLabel {
        key: -1.,
        cost: -1.,
        additional_cost: 0.,
        resources: [-EPSILON, 0.].into(),
        hops: 1,
        ng: 0,
    };
    let b = HotLabel {
        resources: [f64::from_bits(1), 0.].into(),
        ..a
    };
    assert!(LabelStorage::dominates_hot(
        &graph,
        &a,
        &b,
        false,
        InsertionTest::Resources,
        false,
        &[],
        || panic!("unexpected cut state")
    ));
    let mut prefix = HotColumns::default();
    let mut suffix = HotColumns::default();
    for i in 0..9 {
        prefix.push(if i == 8 {
            a.clone()
        } else {
            HotLabel {
                resources: [-100., 0.].into(),
                ..a
            }
        });
        suffix.push(if i == 8 {
            b.clone()
        } else {
            HotLabel {
                resources: [100., 0.].into(),
                ..b
            }
        });
    }
    assert_eq!(
        prefix.next_candidate::<true, false, 1, false>(0, 0., b.resources[0], 0., 0),
        8
    );
    assert_eq!(
        suffix.next_candidate::<false, false, 1, false>(0, 0., a.resources[0], 0., 0),
        8
    );
}

#[test]
fn equality_dominance_retains_binary_interval_endpoints() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    graph.resources[0] = ResourceKind::Equality;
    let make = |resource, cost| StoredLabel {
        id: 0,
        local_id: 0,
        bucket: BucketId {
            vertex: 1,
            bucket: 0,
        },
        vertex: 1,
        cost,
        original_cost: cost,
        resources: [resource, 0.].into(),
        ng: 0,
        branch_state: None,
        presence_state: None,
        coupled_state: None,
        parent: None,
        arc: None,
        hops: 1,
        expanded: true,
        cut_words: [0; LIMITED_WORDS],
        additional_cost: 0.,
        half_potential: 0.,
    };
    for forward in [true, false] {
        for test in [
            InsertionTest::Full,
            InsertionTest::ResourcesAndCost,
            InsertionTest::ResourcesAndNg,
            InsertionTest::Resources,
        ] {
            // These three pairs are accepted by the binary, but rejected
            // by abs(a-b) <= EPSILON because subtraction rounds differently.
            for (a, b) in [
                (1., 1. - EPSILON),
                (1e6, 1e6 - EPSILON),
                (1e6, 1e6 + EPSILON),
            ] {
                let mut storage = LabelStorage::new([1, 1, 1]);
                let mut stats = Statistics::default();
                assert!(storage.insert(
                    &graph,
                    make(a, -1.),
                    forward,
                    test,
                    10,
                    false,
                    &[],
                    &mut stats
                ));
                assert!(
                    !storage.insert(
                        &graph,
                        make(b, 0.),
                        forward,
                        test,
                        10,
                        false,
                        &[],
                        &mut stats
                    ),
                    "forward={forward} test={test:?} a={a} b={b}"
                );
                assert_eq!(storage.buckets[1][0].active.len(), 1);
                assert_eq!(storage.buckets[1][0].next_id, 1);

                // Exercise removal as well as rejection, and the archive
                // path for already expanded resource-only predecessors.
                let mut storage = LabelStorage::new([1, 1, 1]);
                assert!(storage.insert(
                    &graph,
                    make(b, 0.),
                    forward,
                    test,
                    10,
                    false,
                    &[],
                    &mut stats
                ));
                assert!(storage.insert(
                    &graph,
                    make(a, -1.),
                    forward,
                    test,
                    10,
                    false,
                    &[],
                    &mut stats
                ));
                assert_eq!(storage.buckets[1][0].active.len(), 1);
                let archived = matches!(
                    test,
                    InsertionTest::ResourcesAndCost | InsertionTest::Resources
                );
                assert_eq!(storage.buckets[1][0].archived.len(), usize::from(archived));
            }
        }
    }
}

#[test]
fn empty_cut_dominance_does_not_load_cold_words() {
    let g = Graph::<2>::new(3, 1).unwrap();
    let mut a = HotLabel {
        key: 0.,
        cost: -2.,
        additional_cost: 0.,
        resources: [0.; 2].into(),
        hops: 1,
        ng: 0,
    };
    let b = HotLabel {
        cost: -1.,
        ..a.clone()
    };
    for cost in [-2., -1., -1. + EPSILON, f64::NAN] {
        a.cost = cost;
        let mut gap = (a.cost - b.cost) + (a.additional_cost - b.additional_cost);
        let expected = if gap >= EPSILON {
            false
        } else {
            packed_dominate(&[], &[0; LIMITED_WORDS], &[0; LIMITED_WORDS], &mut gap)
        };
        assert_eq!(
            LabelStorage::dominates_hot(
                &g,
                &a,
                &b,
                true,
                InsertionTest::Full,
                false,
                &[],
                || panic!("empty cuts accessed cold state")
            ),
            expected
        );
    }
    a.cost = -2.;
    let descriptors = [DominanceDescriptor {
        mask: 1,
        word: 0,
        cost: 2.,
        ..Default::default()
    }];
    let mut left = [0; LIMITED_WORDS];
    left[0] = 1;
    let right = [0; LIMITED_WORDS];
    let called = std::cell::Cell::new(false);
    assert!(!LabelStorage::dominates_hot(
        &g,
        &a,
        &b,
        true,
        InsertionTest::Full,
        false,
        &descriptors,
        || {
            called.set(true);
            (&left, &right)
        }
    ));
    assert!(called.get());
}

#[test]
fn owned_insertion_moves_branch_state_and_borrowed_insertion_shares_on_acceptance() {
    let graph = Graph::<2>::new(3, 1).unwrap();
    let make = || StoredLabel {
        id: 0,
        local_id: 0,
        bucket: BucketId {
            vertex: 1,
            bucket: 0,
        },
        vertex: 1,
        cost: -1.,
        original_cost: 2.,
        resources: [0.; 2].into(),
        ng: 0,
        branch_state: Some(std::sync::Arc::new([5; 16])),
        presence_state: None,
        coupled_state: None,
        parent: None,
        arc: None,
        hops: 1,
        expanded: false,
        cut_words: [0; LIMITED_WORDS],
        additional_cost: 0.,
        half_potential: 0.,
    };
    let mut owned = LabelStorage::new([1, 1, 1]);
    owned.labels.set_word_limit(2);
    owned.labels.set_mode(false, false);
    let label = make();
    let pointer = label.branch_state.as_deref().unwrap().as_ptr();
    assert!(owned.insert(
        &graph,
        label,
        true,
        InsertionTest::Full,
        10,
        false,
        &[],
        &mut Statistics::default()
    ));
    assert_eq!(
        owned
            .labels
            .view(0)
            .branch_state()
            .as_deref()
            .unwrap()
            .as_ptr(),
        pointer
    );
    let mut borrowed = LabelStorage::new([1, 1, 1]);
    borrowed.labels.set_word_limit(2);
    borrowed.labels.set_mode(false, false);
    let label = make();
    let pointer = label.branch_state.as_deref().unwrap().as_ptr();
    assert!(borrowed.insert_ref(
        &graph,
        &label,
        HotLabel::of(&label),
        true,
        InsertionTest::Full,
        10,
        false,
        &[],
        &mut Statistics::default()
    ));
    let state = borrowed.labels.view(0).branch_state();
    assert_eq!(state, &label.branch_state);
    assert_eq!(state.as_deref().unwrap().as_ptr(), pointer);
    assert_eq!(label.branch_state.as_deref().unwrap().as_ptr(), pointer);
}

#[test]
fn general_dominance_checks_all_fixed_array_coordinates() {
    let g = Graph::<3>::new(3, 3).unwrap();
    let a = StoredLabel {
        id: 0,
        local_id: 0,
        bucket: BucketId {
            vertex: 1,
            bucket: 0,
        },
        vertex: 1,
        cost: -9.,
        original_cost: -9.,
        resources: vec![0., 0., 9.].into(),
        ng: 0,
        branch_state: None,
        presence_state: None,
        coupled_state: None,
        parent: None,
        arc: None,
        hops: 1,
        expanded: false,
        cut_words: [0; LIMITED_WORDS],
        additional_cost: 0.,
        half_potential: 0.,
    };
    let b = StoredLabel {
        cost: -5.,
        resources: vec![0., 0., 2.].into(),
        ..a.clone()
    };
    assert!(!LabelStorage::dominates(
        &g,
        &a,
        &b,
        true,
        InsertionTest::Full,
        false,
        &[]
    ));
    assert!(LabelStorage::dominates(
        &g,
        &a,
        &b,
        false,
        InsertionTest::Full,
        false,
        &[]
    ));
    assert!(!LabelStorage::dominates_hot(
        &g,
        &HotLabel::of(&a),
        &HotLabel::of(&b),
        true,
        InsertionTest::Full,
        false,
        &[],
        || panic!("resource comparison must reject before reading cut words")
    ));
    assert!(LabelStorage::dominates_hot(
        &g,
        &HotLabel::of(&a),
        &HotLabel::of(&b),
        false,
        InsertionTest::Full,
        false,
        &[],
        || panic!("an empty cut list does not need cut words")
    ));
}

#[test]
fn presence_states_survive_arena_promotion_and_prevent_false_dominance() {
    use crate::cuts::soft_conflicts::{Mode, PresenceRow};
    use crate::labeling::arena::OrdinaryLabel;
    let graph = Graph::<2>::new(3, 1).unwrap();
    let row = PresenceRow::new(3, vec![(0, 2)], Mode::AnyPair).unwrap();
    let mut a = row.empty_state();
    let mut b = row.empty_state();
    row.visit(&mut a, 0).unwrap();
    row.visit(&mut b, 1).unwrap();
    assert_eq!(row.coefficient(&a).unwrap(), row.coefficient(&b).unwrap());
    let make = |state, cost| StoredLabel {
        id: 0,
        local_id: 0,
        bucket: BucketId {
            vertex: 1,
            bucket: 0,
        },
        vertex: 1,
        cost,
        original_cost: 0.,
        resources: [0.; 2].into(),
        ng: 0,
        branch_state: None,
        presence_state: Some(vec![state].into()),
        coupled_state: None,
        parent: None,
        arc: None,
        hops: 1,
        expanded: false,
        cut_words: [0; LIMITED_WORDS],
        additional_cost: 0.,
        half_potential: 0.,
    };
    for forward in [false, true] {
        for test in [
            InsertionTest::Full,
            InsertionTest::ResourcesAndCost,
            InsertionTest::ResourcesAndNg,
            InsertionTest::Resources,
        ] {
            let mut storage = LabelStorage::new([1, 1, 1]);
            storage.labels.set_mode(true, false);
            let mut stats = Statistics::default();
            let first = make(a.clone(), 0.);
            let ptr = first.presence_state.as_ref().unwrap().as_ptr();
            let hot = HotLabel::of(&first);
            assert!(storage.insert_label::<false, OrdinaryLabel<2>, _>(
                &graph,
                first,
                hot,
                forward,
                test,
                10,
                false,
                &[],
                &mut stats
            ));
            assert_eq!(
                storage
                    .labels
                    .view(0)
                    .presence_state()
                    .as_ref()
                    .unwrap()
                    .as_ptr(),
                ptr
            );
            let other = make(b.clone(), -1.);
            assert!(!LabelStorage::dominates(
                &graph,
                &other,
                &make(a.clone(), 0.),
                forward,
                test,
                false,
                &[]
            ));
            let hot = HotLabel::of(&other);
            assert!(storage.insert_label::<false, OrdinaryLabel<2>, _>(
                &graph,
                other,
                hot,
                forward,
                test,
                10,
                false,
                &[],
                &mut stats
            ));
            assert_eq!(storage.buckets[1][0].active.len(), 2);
            let duplicate = make(b.clone(), 1.);
            let hot = HotLabel::of(&duplicate);
            assert!(!storage.insert_label::<false, OrdinaryLabel<2>, _>(
                &graph,
                duplicate,
                hot,
                forward,
                test,
                10,
                false,
                &[],
                &mut stats
            ));
            assert_eq!(storage.buckets[1][0].active.len(), 2);
            assert_eq!(
                storage.labels.view(0).presence_state(),
                &Some(vec![a.clone()].into())
            );
        }
    }
}

#[test]
fn coupled_continuation_metadata_uses_physical_vertex_in_a_reordered_bucket() {
    use crate::graph::coupled::{CoupledResource, PiecewiseLinearFunction};
    use crate::graph::Window;
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head, cost) in [(0, 1, -0.5), (0, 1, 0.), (1, 3, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::IntervalCost {
            initial: Window::new(0., 1.).unwrap(),
            bounds: Window::new(0., 1.).unwrap(),
            increments: vec![0.; 3],
            functions: vec![
                PiecewiseLinearFunction::new(vec![[0., 0.], [1., 1.]]).unwrap(),
                PiecewiseLinearFunction::new(vec![[0., 0.], [1., 0.]]).unwrap(),
                PiecewiseLinearFunction::new(vec![[1., 0.], [2., 0.]]).unwrap(),
            ],
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let make = |arc: usize| {
        let (mut state, _) = kernel.extend(&root, arc, true).unwrap().unwrap();
        let (cost, original_cost) = state
            .account_extension(&root, graph.arcs[arc].reduced_cost, graph.arcs[arc].cost)
            .unwrap();
        StoredLabel {
            id: 0,
            local_id: 0,
            // Internal slot 2 represents physical vertex 1. Physical vertex 2
            // is isolated and would incorrectly certify a frozen continuation.
            bucket: BucketId {
                vertex: 2,
                bucket: 0,
            },
            vertex: 1,
            cost,
            original_cost,
            resources: [0.; 2].into(),
            ng: 0,
            branch_state: None,
            presence_state: None,
            coupled_state: Some(state),
            parent: None,
            arc: Some(arc),
            hops: 1,
            expanded: false,
            cut_words: [0; LIMITED_WORDS],
            additional_cost: 0.,
            half_potential: 0.,
        }
    };
    for order in [[0, 1], [1, 0]] {
        let mut storage = LabelStorage::new([1; 4]);
        let mut stats = Statistics::default();
        for arc in order {
            assert!(storage.insert(
                &graph,
                make(arc),
                true,
                InsertionTest::Full,
                10,
                false,
                &[],
                &mut stats
            ));
        }
        // At x=1 the initially cheaper rising front finishes at +0.5;
        // the initially more expensive flat front finishes at zero.
        assert_eq!(storage.buckets[2][0].active.len(), 2);
    }
}

#[test]
fn hot_column_compaction_preserves_all_fields_and_allocations() {
    for len in 0..=9 {
        let original: Vec<_> = (0..len).map(|i| (i * 7) % 13).collect();
        for mask in 0..(1 << len) {
            let mut columns = HotColumns::<2>::default();
            for i in 0..len {
                let x = i as f64;
                columns.push(HotLabel {
                    key: x,
                    cost: -x,
                    additional_cost: x + 0.5,
                    resources: [x + 1., x + 2.],
                    hops: i * 3,
                    ng: (i * 5) as u16,
                });
            }
            let selected: Vec<_> = (0..len).filter(|&i| mask & (1 << i) != 0).collect();
            let expected: Vec<_> = selected.iter().map(|&i| columns.get(i)).collect();
            let retained: Vec<_> = selected.iter().map(|&i| original[i]).collect();
            let capacities = |c: &HotColumns<2>| {
                [
                    c.key.capacity(),
                    c.cost.capacity(),
                    c.additional_cost.capacity(),
                    c.resources[0].capacity(),
                    c.resources[1].capacity(),
                    c.hops.capacity(),
                    c.ng.capacity(),
                ]
            };
            let before = capacities(&columns);
            columns.retain_subsequence(&original, &retained);
            assert_eq!(columns.key.len(), expected.len());
            assert_eq!(capacities(&columns), before);
            for (i, expected) in expected.iter().enumerate() {
                let actual = columns.get(i);
                assert_eq!(actual.key, expected.key);
                assert_eq!(actual.cost.to_bits(), expected.cost.to_bits());
                assert_eq!(actual.additional_cost, expected.additional_cost);
                assert_eq!(actual.resources, expected.resources);
                assert_eq!(actual.hops, expected.hops);
                assert_eq!(actual.ng, expected.ng);
            }
        }
    }
}
