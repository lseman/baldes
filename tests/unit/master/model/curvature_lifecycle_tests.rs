use super::*;

fn node() -> NodeSnapshot<2> {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let mut options = Options::default();
    options.cg.time_limit = std::time::Duration::ZERO;
    options.preprocessing.enabled = false;
    solve(
        Problem {
            rows: vec![],
            networks: vec![Network {
                id: 7,
                graph,
                multiplicity: Window::new(0., 1.).unwrap(),
                arc_entries: vec![vec![]; 2],
                route_entries: vec![],
                activity_entries: vec![],
            }],
        },
        options,
    )
    .unwrap()
    .snapshot
}

#[test]
fn explicit_dynamic_rows_and_ordinary_children_keep_physical_classification_independent() {
    let parent = node();
    let original_rows = parent.master.rhs.len();
    assert!(parent
        .state
        .curvature_rows
        .iter()
        .all(|r| r.active && r.class == cg::CurvatureRowClass::Excluded));
    let dynamic = parent
        .with_dynamic_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 10.,
            },
            &[],
        )
        .unwrap();
    let ordinary = dynamic
        .with_row(
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 0.,
            },
            &[],
        )
        .unwrap();
    assert_eq!(parent.state.curvature_rows.len(), original_rows);
    assert_eq!(dynamic.state.curvature_rows.len(), original_rows + 1);
    assert_eq!(
        ordinary.state.curvature_rows[original_rows].class,
        cg::CurvatureRowClass::Dynamic
    );
    assert_eq!(
        ordinary.state.curvature_rows[original_rows + 1].class,
        cg::CurvatureRowClass::Ordinary
    );
    assert!(ordinary.warm.proximal.is_none());
    let count = ordinary.master.rhs.len();
    let controller = ColumnGenerationController {
        objective_cutoff: f64::INFINITY,
        state: Rc::new(RefCell::new(ordinary.state)),
        request_initial_enumeration: false,
        search_inactive_columns: false,
        artificial_penalty: None,
        enumeration_options: None,
        enumeration_checkpoint: None,
        removed_columns: Vec::new(),
        trust_preprocessing_infeasibility: false,
        allow_objective_fixing: false,
        checkpoints: CheckpointStore::default(),
    };
    let rows = cg::Controller::<ClpMaster>::curvature_rows(&controller, count + 1);
    assert_eq!(rows[original_rows].class, cg::CurvatureRowClass::Dynamic);
    assert_eq!(
        rows[original_rows + 1].class,
        cg::CurvatureRowClass::Ordinary
    );
    assert_eq!(rows[count].class, cg::CurvatureRowClass::Excluded);
    assert!(!rows[count].active);
}

#[test]
fn plain_fork_preserves_row_roles_without_sharing_mutable_classification() {
    let parent = node();
    let mut child = parent.fork().unwrap();
    child.state.curvature_rows[0].class = cg::CurvatureRowClass::Dynamic;
    assert_eq!(
        parent.state.curvature_rows[0].class,
        cg::CurvatureRowClass::Excluded
    );
    assert!(!child.state.scheduler.complete);
}

#[test]
fn consumed_trial_fork_matches_reusable_snapshot_solve_and_preserves_parent() {
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let mut options = Options::default();
        options.cg.bound_mode = mode;
        options.cg.stabilize = false;
        options.cg.artificial_cost = 100.;
        options.preprocessing.enabled = false;
        let parent = solve(problem.clone(), options.clone()).unwrap();
        let child = parent
            .snapshot
            .with_multiplicity(
                42,
                crate::master::preprocessing::Bounds {
                    lower: 1.,
                    upper: 3.,
                },
            )
            .unwrap();
        let child_rhs = child.master.rhs.clone();
        let expected = child.solve(options.clone()).unwrap();
        let actual = child.fork().unwrap().solve_forked(options.clone()).unwrap();
        assert_eq!(actual.master.status, expected.master.status);
        assert_eq!(actual.master.lp, expected.master.lp);
        assert_eq!(
            actual.master.numerical_bound,
            expected.master.numerical_bound
        );
        assert_eq!(
            actual.master.certified_bound,
            expected.master.certified_bound
        );
        assert_eq!(actual.master.iterations, expected.master.iterations);
        assert_eq!(actual.master.pricing_calls, expected.master.pricing_calls);
        assert_eq!(actual.values, expected.values);
        let routes = |out: &Output<2>| {
            out.routes
                .iter()
                .map(|r| (r.network, r.route.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(routes(&actual), routes(&expected));
        assert_eq!(
            parent.snapshot.solve(options.clone()).unwrap().master.lp,
            parent.master.lp
        );
        assert_eq!(child.master.rhs, child_rhs);
        assert_eq!(child.solve(options).unwrap().master.lp, expected.master.lp);
    }
}

#[test]
fn assignment_scoring_retains_fixed_contributions_with_registry_tag_collisions() {
    let mut problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    problem.rows.push(Row {
        sense: RowSense::Equal,
        rhs: 1.,
    });
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    problem.networks.push(Network {
        id: 99,
        graph,
        multiplicity: Window::new(0., 1.).unwrap(),
        arc_entries: vec![vec![(3, 1.)]],
        route_entries: vec![],
        activity_entries: vec![],
    });
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options.preprocessing.enabled = false;
    let parent = solve(problem, options.clone()).unwrap();
    let assignments = |node: &NodeSnapshot<2>| {
        node.fractional_assignments(1e-6)
            .unwrap()
            .into_iter()
            .map(|a| (a.shared_row, a.network, a.value))
            .collect::<Vec<_>>()
    };
    let expected = assignments(&parent.snapshot);
    assert!(!expected.is_empty());
    let tag = parent.routes.iter().position(|r| r.network == 99).unwrap();
    let fixed = parent
        .snapshot
        .fix_route(tag, 1.)
        .unwrap()
        .solve(options)
        .unwrap();
    assert_eq!(fixed.snapshot.state.fixed_routes.len(), 1);
    assert_eq!(assignments(&fixed.snapshot), expected);
    assert_eq!(assignments(&parent.snapshot), expected);
}
