use super::*;

fn label(id: usize) -> StoredLabel {
    StoredLabel {
        id,
        local_id: id as u64 + 7,
        bucket: BucketId {
            vertex: 2,
            bucket: 3,
        },
        vertex: 5,
        cost: -4.5,
        original_cost: 6.,
        resources: [1.5, 2.5],
        ng: 0x15,
        branch_state: None,
        parent: id.checked_sub(1),
        arc: Some(id + 1),
        hops: id + 1,
        expanded: false,
        cut_words: [0; LIMITED_WORDS],
        additional_cost: 0.,
        half_potential: 0.,
    }
}
fn assert_same(a: &StoredLabel, b: &StoredLabel) {
    assert_eq!(
        (a.id, a.local_id, a.bucket, a.vertex),
        (b.id, b.local_id, b.bucket, b.vertex)
    );
    assert_eq!(a.cost.to_bits(), b.cost.to_bits());
    assert_eq!(a.original_cost.to_bits(), b.original_cost.to_bits());
    assert_eq!(a.resources.map(f64::to_bits), b.resources.map(f64::to_bits));
    assert_eq!(
        (a.ng, a.parent, a.arc, a.hops, a.expanded),
        (b.ng, b.parent, b.arc, b.hops, b.expanded)
    );
    assert_eq!(a.branch_state, b.branch_state);
    assert_eq!(a.cut_words, b.cut_words);
    assert_eq!(a.additional_cost.to_bits(), b.additional_cost.to_bits());
    assert_eq!(a.half_potential.to_bits(), b.half_potential.to_bits());
}

#[test]
fn a_typed_sweep_can_promote_when_new_high_words_appear() {
    use crate::cuts::state::LimitedInput;
    use crate::labeling::engine::{EngineControls, LabelingEngine};
    use crate::pricing::phases::Phase;
    let mut g = crate::Graph::new(4, 1).unwrap();
    for vertex in 0..4 {
        g.set_bounds(vertex, 0, 0., 3.).unwrap();
    }
    for (u, v) in [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)] {
        let arc = g.add_arc(u, v, -2.).unwrap();
        g.arc_mut(arc).unwrap().consumption[0] = if v == 3 { 0. } else { 1. };
    }
    let input = LimitedInput {
        covering: false,
        type2: false,
        arc_memory: false,
        denominator: 2,
        dual: 1. / 64.,
        vertex_coefficient: vec![0, 1, 1, 0],
        arc_coefficient: Default::default(),
        memory_vertices: vec![0, 1, 2, 3],
        memory_arcs: Default::default(),
    };
    let run = |words| {
        let mut engine = LabelingEngine::prepare(
            &g,
            EngineControls {
                bidirectional: false,
                ..Default::default()
            },
            Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            },
        )
        .unwrap();
        assert!(engine.load_cuts(&g, vec![input.clone(); 600], vec![]));
        engine.forward.labels.set_word_limit(words);
        let output = engine.solve(&g, &crate::Controls::default()).unwrap();
        let mut routes: Vec<_> = output
            .columns
            .iter()
            .map(|r| {
                (
                    r.nodes.clone(),
                    r.arcs.clone(),
                    r.cost.to_bits(),
                    r.original_cost.to_bits(),
                )
            })
            .collect();
        routes.sort();
        (
            output.minimum_reduced_cost.to_bits(),
            routes,
            matches!(engine.forward.labels.records, Records::Full(_)),
        )
    };
    let full = run(LIMITED_WORDS);
    let compact = run(2);
    assert_eq!((full.0, full.1), (compact.0, compact.1));
    assert!(compact.2);
}

#[test]
fn sized_cut_records_preserve_float_bits_and_promote_on_high_words() {
    assert_eq!(std::mem::size_of::<ExtendedLabel<2>>(), 96);
    assert_eq!(std::mem::size_of::<ExtendedLabel<4>>(), 104);
    assert_eq!(std::mem::size_of::<ExtendedLabel<8>>(), 120);
    for (words, bytes) in [(2, 96), (4, 104), (8, 120)] {
        let mut arena = LabelArena::default();
        arena.set_word_limit(words);
        arena.set_mode(false, false);
        let mut originals = Vec::new();
        for id in 0..20 {
            let mut l = label(id);
            l.cut_words[words - 1] = id as u32 + 1;
            l.additional_cost = -0.;
            l.half_potential = f64::from_bits(1);
            l.branch_state = Some(Box::new([id as u32; 16]));
            arena.push(l.clone());
            assert_same(&l, &arena.view(id).to_owned());
            assert_eq!(l.key().to_bits(), arena.view(id).key().to_bits());
            originals.push(l);
        }
        assert_eq!(arena.allocated_bytes(), arena.capacity() * bytes);
        let _ = &arena[0];
        let cloned = arena.clone();
        assert_eq!(arena.expand(&[0, 1]), 2);
        assert!(arena[0].expanded);
        arena.unexpand_where(|_| true);
        for (id, l) in originals.iter().enumerate() {
            assert_same(l, &cloned.view(id).to_owned());
        }
        let mut wide = label(20);
        wide.cut_words[words] = 99;
        arena.push(wide.clone());
        assert!(matches!(arena.records, Records::Full(_)));
        assert_same(&wide, &arena.view(20).to_owned());
        for (id, l) in originals.iter().enumerate() {
            assert_same(l, &arena.view(id).to_owned());
        }
        arena.clear();
        arena.set_word_limit(words);
        arena.set_mode(false, false);
        arena.push(label(0));
        let capacity = arena.capacity();
        arena.clear();
        arena.set_mode(false, false);
        assert_eq!(arena.capacity(), capacity);
        arena.set_word_limit(LIMITED_WORDS);
        arena.set_mode(false, false);
        assert!(matches!(arena.records, Records::Extended(_)));
    }
}

#[test]
fn branched_records_omit_cut_payload_and_promote_losslessly() {
    assert_eq!(std::mem::size_of::<BranchedLabel>(), 72);
    let mut arena = LabelArena::default();
    arena.set_mode(false, true);
    for id in 0..20 {
        let mut l = label(id);
        l.branch_state = Some(Box::new([id as u32; 16]));
        arena.push(l.clone());
        assert_same(&l, &arena.view(id).to_owned());
    }
    assert_eq!(arena.allocated_bytes(), arena.capacity() * 72);
    assert!(matches!(arena.records, Records::Branched(_)));
    assert_eq!(arena.expand(&[0, 1]), 2);
    assert!(arena[0].expanded);
    arena.unexpand_where(|_| true);
    assert!(!arena.view(0).expanded());
    let mut l = label(20);
    l.cut_words[4] = 17;
    l.branch_state = Some(Box::new([99; 16]));
    arena.push(l.clone());
    assert!(matches!(arena.records, Records::Full(_)));
    assert_same(&l, &arena.view(20).to_owned());
    for id in 0..20 {
        assert_eq!(
            arena.view(id).branch_state(),
            &Some(Box::new([id as u32; 16]))
        );
    }
    arena.clear();
    arena.set_mode(false, true);
    arena.push(label(0));
    let capacity = arena.capacity();
    arena.clear();
    arena.set_mode(false, true);
    assert_eq!(arena.capacity(), capacity);
    let mut l = label(0);
    l.additional_cost = -0.;
    arena.push(l.clone());
    assert!(matches!(arena.records, Records::Full(_)));
    assert_same(&l, &arena.view(0).to_owned());
}

#[test]
fn extended_records_preserve_payloads_inspection_and_fallback() {
    assert_eq!(std::mem::size_of::<ExtendedLabel>(), 168);
    assert_eq!(std::mem::size_of::<StoredLabel>(), 224);
    let mut arena = LabelArena::default();
    arena.set_ordinary(false);
    let mut originals = Vec::new();
    for id in 0..40 {
        let mut l = label(id);
        l.cut_words = std::array::from_fn(|i| (i * 17 + id) as u32);
        l.additional_cost = if id % 2 == 0 { -0. } else { -2.5 };
        l.half_potential = f64::from_bits(1);
        l.resources[1] = f64::from_bits(0x7ff8_0000_0000_0123);
        l.branch_state = Some(Box::new([id as u32; 16]));
        let ptr = l.branch_state.as_deref().unwrap().as_ptr();
        arena.push(l.clone());
        assert_same(&l, &arena.view(id).to_owned());
        assert_eq!(arena.view(id).key().to_bits(), l.key().to_bits());
        // Submission transfers the branch allocation instead of cloning it.
        let mut moved = l.clone();
        let moved_ptr = moved.branch_state.as_deref().unwrap().as_ptr();
        let packed = ExtendedLabel::<LIMITED_WORDS>::pack(&mut moved, id).unwrap();
        assert_eq!(packed.branch_state.as_deref().unwrap().as_ptr(), moved_ptr);
        assert_ne!(ptr, moved_ptr);
        originals.push(l);
    }
    assert!(matches!(arena.records, Records::Extended(_)));
    assert!(arena.inspected.get().is_none());
    assert_eq!(arena.allocated_bytes(), arena.capacity() * 168);
    assert_same(&originals[0], &arena[0]);
    assert_eq!(arena.expand(&[1, 2]), 2);
    assert!(arena.inspected.get().is_none());
    assert!(arena[1].expanded);
    arena.unexpand_where(|_| true);
    assert!(!arena.view(1).expanded());
    let cloned = arena.clone();
    assert!(cloned.inspected.get().is_none());
    for (id, l) in originals.iter().enumerate() {
        assert_same(l, &cloned.view(id).to_owned());
    }
    // Promote after inspection: both existing payloads and the wide value survive.
    let _ = &arena[0];
    let mut wide = label(40);
    wide.local_id = u64::MAX;
    arena.push(wide.clone());
    assert!(matches!(arena.records, Records::Full(_)));
    for (id, l) in originals.iter().enumerate() {
        assert_same(l, &arena.view(id).to_owned());
    }
    assert_same(&wide, &arena.view(40).to_owned());
    let mut cloned = cloned;
    cloned[0].cost = -100.;
    assert!(matches!(cloned.records, Records::Full(_)));
    assert_eq!(cloned.view(0).cost(), -100.);
    arena.clear();
    arena.set_ordinary(false);
    arena.push(label(0));
    let capacity = arena.capacity();
    arena.clear();
    arena.set_ordinary(false);
    assert_eq!(arena.capacity(), capacity);
    assert_eq!(arena.allocated_bytes(), capacity * 168);
}

#[test]
fn ordinary_records_are_64_bytes_and_preserve_float_bits_and_ancestry() {
    assert_eq!(std::mem::size_of::<OrdinaryLabel>(), 64);
    let mut arena = LabelArena::default();
    arena.set_ordinary(true);
    for id in 0..40 {
        let mut l = label(id);
        l.cost = if id % 2 == 0 { -0. } else { f64::from_bits(1) };
        l.resources[1] = f64::from_bits(0x7ff8_0000_0000_0123);
        arena.push(l.clone());
        assert_same(&l, &arena.view(id).to_owned());
        assert_eq!(arena.view(id).key().to_bits(), l.key().to_bits());
    }
    assert!(matches!(arena.records, Records::Ordinary(_)));
    assert!(arena.inspected.get().is_none());
    assert_eq!(arena.allocated_bytes(), arena.capacity() * 64);
}
#[test]
fn inspection_mutation_and_clone_preserve_the_vector_view() {
    let mut arena = LabelArena::default();
    arena.set_ordinary(true);
    arena.push(label(0));
    arena.push(label(1));
    assert_same(&label(0), &arena[0]);
    assert!(arena.inspected.get().is_some());
    let cloned = arena.clone();
    assert!(cloned.inspected.get().is_none());
    assert_eq!(arena.expand(&[1]), 1);
    assert!(arena.inspected.get().is_none());
    assert!(arena[1].expanded);
    assert!(!cloned[1].expanded);
    assert_eq!(arena.expand(&[1]), 0);
    arena[0].cost = -12.;
    assert!(matches!(arena.records, Records::Full(_)));
    assert_eq!(arena.view(0).cost(), -12.);
    assert!(arena.view(1).expanded());
    arena.unexpand_where(|_| true);
    assert!(!arena[1].expanded);
    let collected: Vec<_> = arena.into_iter().collect();
    assert_eq!(collected[0].cost, -12.);
}
#[test]
fn unsupported_payloads_and_identifiers_promote_without_losing_state() {
    let mut cases = Vec::new();
    let mut l = label(1);
    l.cut_words[3] = 7;
    cases.push(l);
    let mut l = label(1);
    l.additional_cost = -0.;
    cases.push(l);
    let mut l = label(1);
    l.additional_cost = -2.;
    cases.push(l);
    let mut l = label(1);
    l.half_potential = 3.;
    cases.push(l);
    let mut l = label(1);
    l.branch_state = Some(Box::new([5; 16]));
    cases.push(l);
    let mut l = label(1);
    l.local_id = u64::MAX;
    cases.push(l);
    let mut l = label(1);
    l.parent = Some(u32::MAX as usize);
    cases.push(l);
    let mut l = label(1);
    l.arc = Some(u32::MAX as usize);
    cases.push(l);
    let mut l = label(1);
    l.id = 99;
    cases.push(l);
    if usize::BITS > 32 {
        let mut l = label(1);
        l.vertex = usize::MAX;
        cases.push(l);
        let mut l = label(1);
        l.bucket.bucket = usize::MAX;
        cases.push(l);
        let mut l = label(1);
        l.bucket.vertex = usize::MAX;
        cases.push(l);
        let mut l = label(1);
        l.hops = usize::MAX;
        cases.push(l);
    }
    for l in cases {
        let mut arena = LabelArena::default();
        arena.set_ordinary(true);
        arena.push(label(0));
        arena.push(l.clone());
        assert!(matches!(arena.records, Records::Full(_)), "{l:?}");
        assert_same(&label(0), &arena.view(0).to_owned());
        assert_same(&l, &arena.view(1).to_owned());
    }
}
#[test]
fn reset_reuses_compact_capacity_and_allows_switching_to_full_records() {
    let mut arena = LabelArena::default();
    arena.set_ordinary(true);
    for id in 0..20 {
        arena.push(label(id));
    }
    let capacity = arena.capacity();
    assert_eq!(arena.expand(&[0, 1, 2]), 3);
    arena.unexpand_where(|_| true);
    assert!(!arena.view(2).expanded());
    arena.clear();
    arena.set_ordinary(true);
    assert_eq!(arena.capacity(), capacity);
    assert_eq!(arena.allocated_bytes(), capacity * 64);
    arena.push(label(0));
    assert_same(&label(0), &arena.view(0).to_owned());
    arena.clear();
    arena.set_ordinary(false);
    arena.push(label(0));
    assert!(matches!(arena.records, Records::Extended(_)));
}

#[test]
fn all_record_combinations_can_complete_fix_and_enumerate_together() {
    use crate::pricing::Solver;
    for (forward_kind, backward_kind) in (0..7).flat_map(|f| (0..7).map(move |b| (f, b))) {
        let mut solver = Solver::new(4, 1).unwrap();
        for v in 0..4 {
            solver.graph.set_bounds(v, 0, 0., 3.).unwrap();
        }
        for (u, v, cost) in [
            (0, 1, -4.),
            (1, 3, -4.),
            (0, 2, -5.),
            (2, 3, -4.),
            (1, 2, -2.),
        ] {
            let arc = solver.graph.add_arc(u, v, cost).unwrap();
            solver.graph.arc_mut(arc).unwrap().consumption[0] = if v == 3 { 0. } else { 1. };
        }
        assert_eq!(
            solver
                .price_phase(0, 150, false)
                .unwrap()
                .minimum_reduced_cost,
            -10.
        );
        let engine = solver.engine.as_mut().unwrap();
        assert!(matches!(
            engine.forward.labels.records,
            Records::Ordinary(_)
        ));
        assert!(matches!(
            engine.backward.labels.records,
            Records::Ordinary(_)
        ));
        for (storage, kind) in [
            (&mut engine.forward.labels, forward_kind),
            (&mut engine.backward.labels, backward_kind),
        ] {
            match kind {
                0 => {}
                1 => {
                    let labels: Vec<_> = (0..storage.len())
                        .map(|id| storage.view(id).to_owned())
                        .collect();
                    storage.clear();
                    storage.set_word_limit(LIMITED_WORDS);
                    storage.set_ordinary(false);
                    for label in labels {
                        storage.push(label);
                    }
                    assert!(matches!(storage.records, Records::Extended(_)));
                }
                2 => {
                    let labels: Vec<_> = (0..storage.len())
                        .map(|id| storage.view(id).to_owned())
                        .collect();
                    storage.clear();
                    storage.set_mode(false, true);
                    for label in labels {
                        storage.push(label);
                    }
                    assert!(matches!(storage.records, Records::Branched(_)));
                }
                3..=5 => {
                    let labels: Vec<_> = (0..storage.len())
                        .map(|id| storage.view(id).to_owned())
                        .collect();
                    storage.clear();
                    storage.set_word_limit([2, 4, 8][kind - 3]);
                    storage.set_mode(false, false);
                    for label in labels {
                        storage.push(label);
                    }
                }
                _ => {
                    storage.reserve(0);
                    assert!(matches!(storage.records, Records::Full(_)));
                }
            }
        }
        solver.run_fixing_enumeration(0, 0., 20.).unwrap();
        let mut costs: Vec<_> = solver
            .enumerate(20.)
            .unwrap()
            .iter()
            .map(|r| r.cost)
            .collect();
        costs.sort_by(f64::total_cmp);
        assert_eq!(costs, [-10., -9., -8.]);
        assert_eq!(
            solver
                .price_phase(0, 150, false)
                .unwrap()
                .minimum_reduced_cost,
            -10.
        );
    }
}
