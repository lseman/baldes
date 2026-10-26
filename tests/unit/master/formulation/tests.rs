use super::*;

#[test]
fn network_quantities_and_shared_rows_have_one_physical_layout() {
    let rows = [Row {
        sense: RowSense::Equal,
        rhs: 3.,
    }];
    let mut built = build(
        &rows,
        &[
            Window::new(1., 2.).unwrap(),
            Window {
                lower: 0.,
                upper: f64::INFINITY,
            },
        ],
        &Backend::default(),
    )
    .unwrap();
    assert_eq!(built.shared_offset, 2);
    assert_eq!(built.quantity_rows, vec![vec![(0, 1.), (1, 1.)], vec![]]);
    assert_eq!(built.master.rhs, vec![2., 1., 3.]);
    // A route in network 0 is bounded by its quantity rows; the second
    // network is uncapped and supplies the rest of the shared demand.
    built
        .master
        .add_column(1., &[(0, 1.), (1, 1.), (2, 1.)])
        .unwrap();
    built.master.add_column(2., &[(2, 1.)]).unwrap();
    assert!(built.master.solve());
    assert!((built.master.objective() - 4.).abs() < 1e-7);
}

#[test]
fn routing_and_explicit_network_rows_construct_the_same_master() {
    let shared = [Row {
        sense: RowSense::Equal,
        rhs: 1.,
    }];
    let mut routing = build(
        &[
            Row {
                sense: RowSense::LessEqual,
                rhs: 2.,
            },
            shared[0].clone(),
        ],
        &[],
        &Backend::default(),
    )
    .unwrap();
    let mut network = build(
        &shared,
        &[Window::new(0., 2.).unwrap()],
        &Backend::default(),
    )
    .unwrap();
    assert_eq!(routing.master.rhs, network.master.rhs);
    assert_eq!(routing.master.senses, network.master.senses);
    for built in [&mut routing, &mut network] {
        built.master.add_column(3., &[(0, 1.), (1, 1.)]).unwrap();
        assert!(built.master.solve());
        assert_eq!(built.master.objective(), 3.);
    }
}

#[test]
fn dynamic_rows_preserve_assigned_ids_and_reject_stale_prefixes() {
    let mut built = build(&[], &[], &Backend::default()).unwrap();
    built.master.add_column(1., &[]).unwrap();
    let mut pending = PendingRows::new(0);
    assert_eq!(pending.push(RowSense::GreaterEqual, 2., vec![(0, 1.)]), 0);
    assert_eq!(pending.push(RowSense::LessEqual, 5., vec![(0, 1.)]), 1);
    pending.install(&mut built.master).unwrap();
    assert!(pending.install(&mut built.master).is_err());
    assert_eq!(built.master.rhs.len(), 2);
    assert!(built.master.solve());
    assert_eq!(built.master.objective(), 2.);
}

#[test]
fn routing_registration_and_general_solver_consume_the_same_canonical_model() {
    let mut graph = crate::Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 2.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let registration = crate::master::model::Problem::routing(graph, 1.).unwrap();
    assert_eq!(registration.model.networks.len(), 1);
    for bound_mode in [
        crate::master::cg::BoundMode::Numerical,
        crate::master::cg::BoundMode::Certified,
    ] {
        let mut root = crate::master::routing::RootOptions::default();
        root.bound_mode = bound_mode;
        root.cg.stabilize = false;
        root.enumerate = false;
        let routing = crate::master::routing::solve(registration.clone(), root).unwrap();
        let mut options = crate::master::model::Options::default();
        options.cg.bound_mode = bound_mode;
        options.cg.stabilize = false;
        let general = crate::master::model::solve(registration.model.clone(), options).unwrap();
        assert_eq!(routing.master.lp, 2.);
        assert_eq!(routing.master.lp, general.master.lp);
        assert_eq!(routing.master.bound, general.master.bound);
    }
}
