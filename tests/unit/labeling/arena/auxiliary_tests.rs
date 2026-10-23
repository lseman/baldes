use super::*;
use crate::cuts::soft_conflicts::{Mode, PresenceRow};
use crate::graph::coupled::CoupledResource;

fn label() -> StoredLabel<2> {
    StoredLabel {
        id: 0,
        local_id: 7,
        bucket: BucketId {
            vertex: 1,
            bucket: 0,
        },
        vertex: 1,
        cost: -3.,
        original_cost: 8.,
        resources: [1., -0.].into(),
        ng: 2,
        branch_state: None,
        presence_state: None,
        coupled_state: None,
        parent: None,
        arc: Some(0),
        hops: 1,
        expanded: false,
        cut_words: [0; LIMITED_WORDS],
        additional_cost: 0.,
        half_potential: 0.,
    }
}
#[test]
fn auxiliary_payloads_move_without_wide_promotion_and_survive_inspection() {
    let mut graph = crate::Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::Selection {
            initial: 3,
            allowed: vec![3],
        }])
        .unwrap();
    for words in [2, 4, 8, LIMITED_WORDS] {
        let mut arena = LabelArena::default();
        arena.set_word_limit(words);
        arena.set_resource_mode(false, false, true);
        let mut input = label();
        input.cut_words[words - 1] = 19;
        input.additional_cost = -0.;
        input.branch_state = Some(Box::new([3u32; 16]));
        let row = PresenceRow::new(2, vec![(0, 1)], Mode::AnyPair).unwrap();
        input.presence_state = Some(vec![row.empty_state()].into_boxed_slice());
        input.coupled_state = Some(graph.coupled_kernel().unwrap().root(true));
        let address = input.coupled_state.as_deref().unwrap() as *const _;
        arena.push(input);
        assert!(arena.kind() != RecordKind::Full);
        assert_eq!(arena.view(0).coupled_state().unwrap() as *const _, address);
        assert_eq!(arena.view(0).cut_words()[words - 1], 19);
        assert_eq!(arena[0].resources[1].to_bits(), (-0f64).to_bits());
        assert_eq!(arena[0].additional_cost.to_bits(), (-0f64).to_bits());
        assert!(arena[0].presence_state.is_some());
        assert_eq!(arena.expand(&[0]), 1);
        assert!(arena[0].expanded);
        let mut borrowed = LabelArena::default();
        borrowed.set_word_limit(words);
        borrowed.set_resource_mode(false, false, true);
        borrowed.push_borrowed_for::<StoredLabel<2>>(&arena[0], 23);
        assert!(borrowed.kind() != RecordKind::Full);
        assert_eq!(
            borrowed.view(0).coupled_state(),
            arena.view(0).coupled_state()
        );
        assert_ne!(
            borrowed.view(0).coupled_state().unwrap() as *const _,
            address
        );
        assert_eq!(borrowed[0].local_id, 23);
        assert_eq!(borrowed[0].branch_state, arena[0].branch_state);
        assert_eq!(borrowed[0].presence_state, arena[0].presence_state);
        let copied = arena.clone();
        assert_eq!(
            copied.view(0).coupled_state(),
            arena.view(0).coupled_state()
        );
        arena[0].local_id = u64::MAX;
        assert!(arena.kind() == RecordKind::Full);
        assert_eq!(arena.view(0).to_owned().local_id, u64::MAX);
        assert_eq!(copied.view(0).to_owned().local_id, 7);
    }
    assert!(std::mem::size_of::<AuxiliaryLabel<2, 2>>() < std::mem::size_of::<StoredLabel<2>>());
}
#[test]
fn retained_borrowed_metadata_overrides_ids_and_preserves_full_fallback() {
    for (words, ordinary, branched) in [
        (0, true, false),
        (2, false, false),
        (4, false, false),
        (8, false, false),
        (LIMITED_WORDS, false, false),
        (0, false, true),
    ] {
        let mut arena = LabelArena::default();
        arena.set_word_limit(words);
        arena.set_mode(ordinary, branched);
        let mut input = label();
        input.id = 999;
        if words > 0 {
            input.cut_words[words - 1] = 13;
        }
        if branched {
            input.branch_state = Some(Box::new([5u32; 16]));
        }
        arena.push_borrowed_for::<StoredLabel<2>>(&input, 11);
        assert!(arena.kind() != RecordKind::Full);
        let stored = arena.view(0).to_owned();
        assert_eq!((stored.id, stored.local_id), (0, 11));
        assert_eq!(stored.cut_words, input.cut_words);
        assert_eq!(stored.branch_state, input.branch_state);
        assert_eq!(input.id, 999);
        assert_eq!(input.local_id, 7);
        arena.push_borrowed_for::<StoredLabel<2>>(&input, u64::MAX);
        assert!(arena.kind() == RecordKind::Full);
        assert_eq!(arena.view(1).to_owned().local_id, u64::MAX);
        assert_eq!(arena.view(0).to_owned().local_id, 11);
    }
    let mut arena = LabelArena::default();
    arena.set_mode(true, false);
    let mut input = label();
    input.additional_cost = -0.;
    arena.push_borrowed_for::<OrdinaryLabel<2>>(&input, 0);
    assert!(arena.kind() == RecordKind::Full);
    assert_eq!(arena.view(0).additional_cost().to_bits(), (-0f64).to_bits());
}
