use super::SolutionPool;

#[test]
fn representatives_obey_total_column_limit() {
    for total in 0..4 {
        let mut pool = SolutionPool::new(6, total, 30);
        for group in 0..6 {
            pool.insert(group, -(group as f64), 0., ());
            assert!(pool.len() <= total);
        }
        let costs: Vec<_> = pool.into_sorted().into_iter().map(|c| c.cost).collect();
        assert_eq!(
            costs,
            (0..total).map(|i| -5. + i as f64).collect::<Vec<_>>()
        );
    }
}

#[test]
fn signed_zero_costs_keep_candidate_identity_order() {
    let mut pool = SolutionPool::new(2, 4, 2);
    pool.insert(0, 0., 0., ());
    pool.insert(1, -0., 0., ());
    pool.insert(0, -0., 0., ());
    assert_eq!(pool.iter().map(|c| c.id).collect::<Vec<_>>(), vec![0, 1, 2]);
    assert_eq!(
        pool.into_sorted()
            .into_iter()
            .map(|c| c.id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}

#[test]
fn borrowed_iteration_merges_representatives_and_other_candidates() {
    let mut pool = SolutionPool::new(3, 8, 2);
    for (group, cost) in [(0, 3.), (1, 1.), (0, 2.), (2, 4.), (1, 5.), (2, 0.)] {
        pool.insert(group, cost, 0., ());
    }
    let borrowed: Vec<_> = pool.iter().map(|c| (c.cost, c.id)).collect();
    let owned: Vec<_> = pool
        .into_sorted()
        .into_iter()
        .map(|c| (c.cost, c.id))
        .collect();
    assert_eq!(borrowed, owned);
    assert_eq!(
        borrowed.iter().map(|c| c.0).collect::<Vec<_>>(),
        vec![0., 1., 2., 3., 4., 5.]
    );
}
