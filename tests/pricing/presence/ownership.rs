use super::*;

#[test]
fn solver_owned_presence_survives_rebuilds_and_isolates_child_updates() {
    for symmetric in [false, true] {
        let (g, events) = fixtures(symmetric);
        for mode in [Mode::PairCount, Mode::AnyPair] {
            for bidirectional in [false, true] {
                let mut parent = owned_solver(g.clone(), bidirectional);
                let rows = input(&g, &events, mode, 0.7);
                parent.set_presence_rows(rows.clone()).unwrap();
                assert!(parent.has_cuts());
                assert!(parent.engine.is_none());
                let expected = exhaustive(&g, &events, mode, 0.7);
                assert!(
                    (parent
                        .price_phase(0, 150, false)
                        .unwrap()
                        .minimum_reduced_cost
                        - expected)
                        .abs()
                        < 1e-10
                );
                let labels = parent.engine.as_ref().unwrap().forward.labels.len();
                parent.set_presence_rows(rows.clone()).unwrap();
                assert_eq!(parent.engine.as_ref().unwrap().forward.labels.len(), labels);
                let mut invalid = rows.clone();
                invalid[0].arc_events.clear();
                assert!(parent.set_presence_rows(invalid).is_err());
                assert_eq!(parent.presence_rows(), rows.as_slice());
                assert!(parent.engine.is_some());
                parent
                    .set_option("BALDESuseBidirectionalSearch", f64::from(!bidirectional))
                    .unwrap();
                assert!(parent.engine.is_none());
                assert!(
                    (parent
                        .price_phase(0, 150, false)
                        .unwrap()
                        .minimum_reduced_cost
                        - expected)
                        .abs()
                        < 1e-10
                );
                parent.restore_arcs();
                assert_eq!(parent.presence_rows(), rows.as_slice());
                assert!(
                    (parent
                        .price_phase(0, 150, false)
                        .unwrap()
                        .minimum_reduced_cost
                        - expected)
                        .abs()
                        < 1e-10
                );
                let mut child = parent.fork_for_node();
                assert_eq!(child.presence_rows(), rows.as_slice());
                child
                    .set_presence_rows(input(&g, &events, mode, -0.625))
                    .unwrap();
                assert!(child.engine.is_none());
                let child_expected = exhaustive(&g, &events, mode, -0.625);
                assert!(
                    (child
                        .price_phase(0, 150, false)
                        .unwrap()
                        .minimum_reduced_cost
                        - child_expected)
                        .abs()
                        < 1e-10
                );
                assert_eq!(parent.presence_rows(), rows.as_slice());
                assert!(
                    (parent
                        .price_phase(0, 150, false)
                        .unwrap()
                        .minimum_reduced_cost
                        - expected)
                        .abs()
                        < 1e-10
                );
                child.set_presence_rows(vec![]).unwrap();
                assert!(!child.has_cuts());
                let base = exhaustive(&g, &events, mode, 0.) - 0.2;
                assert!(
                    (child
                        .price_phase(0, 150, false)
                        .unwrap()
                        .minimum_reduced_cost
                        - base)
                        .abs()
                        < 1e-10
                );
            }
        }
    }
}

#[test]
fn owned_presence_updates_keep_branches_and_physical_exclusions() {
    use baldes::pricing::restrictions::RyanFosterConstraint;
    let (g, events) = fixtures(false);
    let branch = RyanFosterConstraint {
        first: 1,
        second: 2,
        together: false,
    };
    let mut parent = owned_solver(g.clone(), true);
    parent.set_ryan_foster(vec![branch]).unwrap();
    parent.set_forbidden_arcs(vec![7]).unwrap();
    for dual in [0.7, -0.625] {
        parent
            .set_presence_rows(input(&g, &events, Mode::PairCount, dual))
            .unwrap();
        let mut child = parent.fork_for_node();
        assert_eq!(child.ryan_foster_constraints(), &[branch]);
        let result = child.price_phase(0, 150, false).unwrap();
        assert!(!result.columns.is_empty());
        for route in result.columns {
            assert!(!route.arcs.contains(&g.arcs[7].id));
            assert!(branch.satisfied_by(route.nodes.iter().map(|&v| v as usize)));
            assert!(branch.elementary_on_pair(route.nodes.iter().map(|&v| v as usize)));
        }
    }
}

#[test]
fn owned_presence_reaches_raw_functor_rebuilds() {
    let (mut g, events) = fixtures(false);
    let raw_costs = vec![0.123456789; g.arcs.len()];
    for (i, a) in g.arcs.iter_mut().enumerate() {
        a.variables = vec![(i, 1.)];
    }
    let mut solver = owned_solver(g.clone(), true);
    solver
        .set_presence_rows(input(&g, &events, Mode::PairCount, 0.7))
        .unwrap();
    solver.cache_variable_costs(&raw_costs).unwrap();
    for (a, cost) in g.arcs.iter_mut().zip(&raw_costs) {
        a.reduced_cost = a.cost + cost;
    }
    let expected = exhaustive(&g, &events, Mode::PairCount, 0.7);
    let raw = solver.price_functor_phase(0, 150).unwrap();
    assert!((raw.minimum_reduced_cost - expected).abs() < 1e-10);
}

#[test]
fn installing_owned_presence_invalidates_only_the_childs_retained_universe() {
    let mut g = Graph::<2>::new(4, 1).unwrap();
    for v in 0..4 {
        g.set_bounds(v, 0, 0., 4.).unwrap();
    }
    for v in 0..3 {
        let id = g.add_arc(v, v + 1, 0.).unwrap();
        g.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    let events = vec![vec![0], vec![1], vec![]];
    let mut parent = owned_solver(g.clone(), true);
    assert_eq!(parent.enumerate(10.).unwrap().len(), 1);
    assert!(parent.enumeration.as_ref().unwrap().completed);
    let mut child = parent.fork_for_node();
    child
        .set_presence_rows(input(&g, &events, Mode::PairCount, 0.7))
        .unwrap();
    assert!(child.enumeration.is_none());
    assert!(child.engine.is_none());
    assert!(parent.enumeration.as_ref().unwrap().completed);
    assert_eq!(
        parent
            .price_phase(0, 150, false)
            .unwrap()
            .minimum_reduced_cost,
        0.
    );
    assert!(
        (child
            .price_phase(0, 150, false)
            .unwrap()
            .minimum_reduced_cost
            + 1.2)
            .abs()
            < 1e-12
    );
}
