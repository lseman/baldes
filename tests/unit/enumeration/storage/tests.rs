use super::*;

#[test]
fn resource_dominance_preserves_binary_boundaries_during_enumeration() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    graph.resources[0] = ResourceKind::Equality;
    let make = |resource, cost| Label {
        id: 0,
        vertex: 1,
        resources: [resource, 0.].into(),
        reduced_cost: cost,
        original_cost: cost,
        visited: [0; 16],
        element_hash: 0,
        parent: None,
        arc: None,
        hops: 1,
        cut_words: [0; FULL_WORDS],
        cut_cost: 0.,
        presence_state: None,
        coupled_state: None,
    };
    for forward in [true, false] {
        for (a, b) in [
            (1., 1. - EPSILON),
            (1e6, 1e6 - EPSILON),
            (1e6, 1e6 + EPSILON),
        ] {
            let mut storage = Storage::new(1).unwrap();
            assert!(storage.insert(&graph, make(a, -1.), forward));
            assert!(!storage.insert(&graph, make(b, 0.), forward));
            assert_eq!(storage.pending, 1);
            let mut storage = Storage::new(1).unwrap();
            assert!(storage.insert(&graph, make(b, 0.), forward));
            assert!(storage.insert(&graph, make(a, -1.), forward));
            assert_eq!(storage.pending, 1);
            assert_eq!(storage.live, [false, true]);
        }
        // The immediately adjacent value outside the upper endpoint
        // must survive: tolerance must not be enlarged by the fix.
        let edge: f64 = 1. + EPSILON;
        let outside = f64::from_bits(edge.to_bits() + 1);
        let mut storage = Storage::new(1).unwrap();
        assert!(storage.insert(&graph, make(1., -1.), forward));
        assert!(storage.insert(&graph, make(outside, 0.), forward));
        assert_eq!(storage.pending, 2);
    }
    graph.resources[0] = ResourceKind::Disposable;
    let mut storage = Storage::new(1).unwrap();
    assert!(storage.insert(&graph, make(-EPSILON, -1.), false));
    assert!(!storage.insert(&graph, make(f64::from_bits(1), 0.), false));
    assert_eq!(storage.pending, 1);
}

#[test]
fn dominance_removal_preserves_survivor_order_and_pending_after_pop() {
    let graph = Graph::<1>::new(3, 1).unwrap();
    let make = |resource, cost| Label {
        id: 0,
        vertex: 1,
        resources: [resource].into(),
        reduced_cost: cost,
        original_cost: cost,
        visited: [0; 16],
        element_hash: 0,
        parent: None,
        arc: None,
        hops: 1,
        cut_words: [0; FULL_WORDS],
        cut_cost: 0.,
        presence_state: None,
        coupled_state: None,
    };
    let mut storage = Storage::new(1).unwrap();
    // Cheaper labels consume more resource, so these are incomparable.
    for (resource, cost) in [(3., 1.), (2., 2.), (1., 3.)] {
        assert!(storage.insert(&graph, make(resource, cost), true));
    }
    assert_eq!(storage.pop().unwrap().id, 0);
    assert_eq!(storage.pending, 2);
    // Dominate both a popped label and a pending label, leaving label 2.
    assert!(storage.insert(&graph, make(2., 0.), true));
    assert_eq!(storage.partitions[0], [3, 2]);
    assert_eq!(storage.live, [false, false, true, true]);
    assert_eq!(storage.pending, 2);
    assert!(!storage.insert(&graph, make(3., 4.), true));
    assert_eq!(storage.partitions[0], [3, 2]);
    assert_eq!(storage.pending, 2);
    assert_eq!(storage.pop().unwrap().id, 2);
    assert_eq!(storage.pop().unwrap().id, 3);
    assert!(storage.pop().is_none());
    assert_eq!(storage.pending, 0);
}
