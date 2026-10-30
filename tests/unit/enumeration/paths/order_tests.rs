use super::*;

#[test]
fn tolerance_order_uses_first_greater_candidate_even_when_costs_are_not_partitioned() {
    for preserve in [false, true] {
        let mut storage = PathStorage::new(1).unwrap();
        storage.preserve_arc_paths = preserve;
        for (id, multiple) in [0., 2., 1., 1., 0.].into_iter().enumerate() {
            let mut visited = [0; 16];
            visited[0] = 1 << id;
            assert!(storage.insert(Path {
                arcs: vec![id],
                element_hash: 0,
                visited,
                original_cost: multiple * EPSILON,
                reduced_cost: 0.,
            }));
        }
        let order: Vec<_> = storage
            .ordered_paths()
            .iter()
            .map(|path| path.arcs[0])
            .collect();
        assert_eq!(order, [0, 4, 1, 2, 3]);
    }
}

#[test]
fn replacement_keeps_the_native_partition_order_without_replaying_survivors() {
    let mut storage = PathStorage::new(1).unwrap();
    for (id, mask, multiple) in [(0, 0, 2.), (1, 1, 1.), (2, 2, 0.), (3, 0, 0.), (4, 0, 0.)] {
        let mut visited = [0; 16];
        visited[0] = 1 << mask;
        let inserted = storage.insert(Path {
            arcs: vec![id],
            element_hash: 0,
            visited,
            original_cost: multiple * EPSILON,
            reduced_cost: 0.,
        });
        assert_eq!(inserted, id != 4);
    }
    let order: Vec<_> = storage
        .ordered_paths()
        .iter()
        .map(|path| path.arcs[0])
        .collect();
    assert_eq!(order, [2, 3, 1]);
    let mut child = storage.clone();
    child.retain(|path| path.arcs[0] != 2);
    let retained: Vec<_> = child
        .ordered_paths()
        .iter()
        .map(|path| path.arcs[0])
        .collect();
    assert_eq!(retained, [3, 1]);
    assert_eq!(storage.ordered_paths().len(), 3);
}

#[test]
fn duplicate_scan_uses_export_partition_when_storage_buckets_are_coalesced() {
    for count in [1, 8] {
        let mut storage = PathStorage::new(count).unwrap();
        for (id, mask, multiple) in [(0, 0, 2.), (1, 0, 1.), (2, 0, 1.), (3, 1, 1.), (4, 1, 0.)] {
            let mut visited = [0; 16];
            visited[0] = 1 << mask;
            let inserted = storage.insert(Path {
                arcs: vec![id],
                element_hash: mask,
                visited,
                original_cost: multiple * EPSILON,
                reduced_cost: 0.,
            });
            assert_eq!(inserted, id == 0 || id == 3);
        }
        let order: Vec<_> = storage
            .ordered_paths()
            .iter()
            .map(|path| path.arcs[0])
            .collect();
        assert_eq!(order, [0, 3]);
    }
}

#[test]
fn export_order_survives_partition_tuning_replacement_retention_and_forks() {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for (tail, head, cost) in [(0, 1, 1.), (1, 3, 1.), (0, 2, 2.), (2, 3, 2.), (1, 2, 1.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    for count in [1, 8] {
        let mut storage = PathStorage::new(count).unwrap();
        let make = |arcs: Vec<usize>, hash, visited, cost| {
            let mut bits = [0; 16];
            bits[0] = visited;
            Path {
                arcs,
                element_hash: hash,
                visited: bits,
                original_cost: cost,
                reduced_cost: cost,
            }
        };
        storage.insert(make(vec![0, 1], 1, 2, 2.));
        storage.insert(make(vec![2, 3], 0, 1, 4.));
        storage.insert(make(vec![0, 4, 3], 1, 3, 4.));
        let paths = |s: &PathStorage| {
            s.routes(&graph)
                .unwrap()
                .into_iter()
                .map(|r| r.arcs)
                .collect::<Vec<_>>()
        };
        assert_eq!(paths(&storage), [vec![0, 1], vec![0, 4, 3], vec![2, 3]]);
        let mut child = storage.clone();
        child.retain(|p| p.arcs != [0, 1]);
        assert_eq!(paths(&child), [vec![0, 4, 3], vec![2, 3]]);
        assert!(child.insert(make(vec![0, 4, 3], 1, 3, 3.5)));
        assert_eq!(paths(&child), [vec![0, 4, 3], vec![2, 3]]);
        assert_eq!(paths(&storage).len(), 3);
        child.inspect(&graph, 3., &[], &[]).unwrap();
        assert!(paths(&child).is_empty());
        assert!(child.path_order.is_empty());
    }
}

#[test]
fn pricing_preserves_canonical_ties_and_reconstructs_only_selected_paths() {
    let mut graph = Graph::<1>::new(4, 1).unwrap();
    for (tail, head) in [(0, 1), (1, 3), (0, 2), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    let mut storage = PathStorage::new(1).unwrap();
    storage.preserve_arc_paths = true;
    let make = |arcs, hash, cost| Path {
        arcs,
        element_hash: hash,
        visited: [0; 16],
        original_cost: 0.,
        reduced_cost: cost,
    };
    storage.insert(make(vec![0, 1], 1, -0.));
    storage.insert(make(vec![2, 3], 0, 0.));
    let (routes, minimum) = storage.priced_routes(&graph, 1, false, 0.).unwrap();
    assert_eq!(routes[0].arcs, [0, 1]);
    assert!(minimum.is_sign_negative());

    // Deliberately make the unselected path impossible to reconstruct. Ranking
    // can still return the valid selected route without replaying that path.
    graph.arcs[2].consumption[0] = 2e5;
    assert!(storage.routes(&graph).is_err());
    assert_eq!(
        storage.priced_routes(&graph, 1, false, 0.).unwrap().0.len(),
        1
    );
    assert!(storage
        .priced_routes(&graph, 0, false, 0.)
        .unwrap()
        .0
        .is_empty());
    assert!(storage.priced_routes(&graph, 2, false, 0.).is_err());
}

#[test]
fn preserved_identity_index_is_shared_until_membership_changes() {
    let mut graph = Graph::<1>::new(4, 1).unwrap();
    for (tail, head) in [(0, 1), (1, 3), (0, 2), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    let make = |arcs| Path {
        arcs,
        element_hash: 0,
        visited: [0; 16],
        original_cost: 0.,
        reduced_cost: 0.,
    };
    let mut parent = PathStorage::new(1).unwrap();
    parent.preserve_arc_paths = true;
    parent.insert(make(vec![0, 1]));
    parent.insert(make(vec![2, 3]));
    let mut child = parent.clone();
    assert!(!child.insert(make(vec![0, 1])));
    child.inspect(&graph, f64::INFINITY, &[], &[]).unwrap();
    child.retain(|_| true);
    assert!(Arc::ptr_eq(&child.path_order, &parent.path_order));
    assert!(Arc::ptr_eq(&child.export_rows, &parent.export_rows));
    child.retain(|p| p.arcs != [0, 1]);
    assert!(!Arc::ptr_eq(&child.path_order, &parent.path_order));
    assert!(!Arc::ptr_eq(&child.export_rows, &parent.export_rows));
    assert_eq!(parent.len, 2);
    assert_eq!(child.len, 1);
    let retained: &[usize] = &[2, 3];
    assert!(Arc::ptr_eq(
        parent.path_order.get_key_value(retained).unwrap().0,
        child.path_order.get_key_value(retained).unwrap().0
    ));
    assert!(child.insert(make(vec![0, 1])));
    assert!(!child.insert(make(vec![0, 1])));
    assert_eq!(child.routes(&graph).unwrap().len(), 2);
}
