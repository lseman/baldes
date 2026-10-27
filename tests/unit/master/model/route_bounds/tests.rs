use super::*;
use crate::master::cg::BoundMode;

fn fixture(rhs: f64, cheap_cap: f64, costs: &[f64]) -> Problem<2> {
    let mut graph = Graph::<2>::new(costs.len() + 2, 1).unwrap();
    let sink = costs.len() + 1;
    let mut arc_entries = vec![];
    for (i, &cost) in costs.iter().enumerate() {
        graph.add_arc(0, i + 1, cost).unwrap();
        graph.add_arc(i + 1, sink, 0.).unwrap();
        let mut entries = vec![(0, 1.)];
        if i == 0 {
            entries.push((1, 1.));
        }
        if i == costs.len() - 1 {
            entries.push((2, 1.));
        }
        arc_entries.push(entries);
        arc_entries.push(vec![]);
    }
    Problem {
        rows: vec![
            Row {
                sense: RowSense::Equal,
                rhs,
            },
            Row {
                sense: RowSense::LessEqual,
                rhs: cheap_cap,
            },
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.25,
            },
        ],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 10.).unwrap(),
            arc_entries,
            route_entries: vec![],
            activity_entries: vec![],
        }],
    }
}
fn options(mode: BoundMode) -> Options {
    let mut options = Options::default();
    options.cg.bound_mode = mode;
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options.enumeration = Some(EnumerationOptions::default());
    options
}
#[test]
fn coupled_objective_survives_full_domain_caps_and_directed_compensation() {
    use crate::graph::coupled::CoupledResource;
    let mut problem = fixture(1., 0.5, &[0., 10., 20.]);
    problem.networks[0]
        .graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![2., 0., 2., 0., 2., 0.],
            weights: vec![1., 0., 1., 0., 1., 0.],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    let root = solve(problem, options(BoundMode::Certified)).unwrap();
    assert_eq!(root.master.status, cg::Status::Optimal, "{:?}", root.master);
    assert_eq!(root.master.lp, 7.);
    assert!(root.snapshot.enumeration_report().complete);
    let mut duals = vec![0.; root.snapshot.state.rhs.len()];
    duals[root.snapshot.state.shared_offset] = 12.;
    assert_eq!(certificate(&root.snapshot.state, &duals), Some(-5.));
    let bound =
        super::super::bounds::certificate(&root.snapshot.state, &root.snapshot.master, &duals)
            .unwrap();
    assert_eq!(bound, 7.);
}
#[test]
fn positive_column_caps_have_a_tight_directed_box_certificate() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let result = solve(fixture(1., 0.5, &[0., 10., 20.]), options(mode)).unwrap();
        assert_eq!(
            result.master.status,
            cg::Status::Optimal,
            "{:?}",
            result.master
        );
        assert!((result.master.lp - 5.).abs() < 1e-9);
        let cheap = result
            .routes
            .iter()
            .position(|r| r.route.arcs[0] == 0)
            .unwrap();
        assert_eq!(upper(&result.snapshot.state, cheap), Some(0.5));
        assert!(result
            .snapshot
            .preprocessing_report()
            .route_upper_bounds
            .iter()
            .any(|&(tag, cap)| tag == cheap && cap == 0.5));
        let mut duals = vec![0.; result.snapshot.state.rhs.len()];
        duals[result.snapshot.state.shared_offset] = 10.;
        let bound = super::super::bounds::certificate(
            &result.snapshot.state,
            &result.snapshot.master,
            &duals,
        )
        .unwrap();
        assert!((bound - 5.).abs() < 1e-9, "{bound}");
        // The cheap column has reduced cost -10 at its upper bound .5.
        // Cheapest-route * network multiplicity would give 10-100=-90.
        assert_eq!(certificate(&result.snapshot.state, &duals), Some(-5.));
    }
}
#[test]
fn cleanup_and_future_regeneration_restore_the_same_positive_cap() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut configured = options(mode);
        configured.cg.column_cleanup_threshold = 2.;
        configured.cg.insert_non_negative = false;
        configured.cg.column_cleanup_ratio = 0.75;
        let root = solve(fixture(1., 0.5, &[0., 10., 20., 30.]), configured.clone()).unwrap();
        assert_eq!(root.master.status, cg::Status::Optimal, "{:?}", root.master);
        let target = root
            .routes
            .iter()
            .position(|r| r.route.arcs[0] == 6)
            .unwrap();
        assert!(!root
            .snapshot
            .warm
            .columns
            .iter()
            .flatten()
            .any(|c| c.tag == target as i32));
        let mut coefficients = vec![0.; 8];
        coefficients[6] = 1.;
        let child = root
            .snapshot
            .with_row(
                Row {
                    sense: RowSense::GreaterEqual,
                    rhs: 0.25,
                },
                &[RowCoefficients {
                    network: 7,
                    arcs: coefficients,
                    per_route: 0.,
                }],
            )
            .unwrap();
        let result = child.solve(configured).unwrap();
        assert_eq!(
            result.master.status,
            cg::Status::Optimal,
            "{:?}",
            result.master
        );
        assert!((result.master.lp - 10.).abs() < 1e-8);
        let column = result
            .snapshot
            .warm
            .columns
            .iter()
            .position(|c| c.as_ref().is_some_and(|c| c.tag == target as i32))
            .unwrap();
        assert_eq!(result.snapshot.master.upper[column], 0.25);
        assert_eq!(result.values[target], 0.25);
        assert!((root.snapshot.solve(options(mode)).unwrap().master.lp - 5.).abs() < 1e-8);
    }
}
#[test]
fn route_fixing_reduces_its_cap_even_when_propagation_is_disabled() {
    // A third unused route has cap .25; the two used routes supply two units.
    let root = solve(
        fixture(2., 1., &[1., 10., 20.]),
        options(BoundMode::Certified),
    )
    .unwrap();
    let cheap = root
        .routes
        .iter()
        .position(|r| r.route.arcs[0] == 0)
        .unwrap();
    assert_eq!(upper(&root.snapshot.state, cheap), Some(1.));
    assert!(root.snapshot.fix_route(cheap, 2.).is_err());
    let child = root.snapshot.fix_route(cheap, 1.).unwrap();
    assert_eq!(upper(&child.state, cheap), Some(0.));
    let mut configured = options(BoundMode::Certified);
    configured.preprocessing.enabled = false;
    let result = child.solve(configured).unwrap();
    assert_eq!(
        result.master.status,
        cg::Status::Optimal,
        "{:?}",
        result.master
    );
    assert!((result.master.lp - 11.).abs() < 1e-8);
    assert_eq!(result.values[cheap], 0.);
    assert!(
        (root
            .snapshot
            .solve(options(BoundMode::Certified))
            .unwrap()
            .master
            .lp
            - 11.)
            .abs()
            < 1e-8
    );
}
#[test]
fn physical_path_aliases_cannot_restore_an_independent_column_cap() {
    let mut configured = options(BoundMode::Certified);
    configured.enumeration = None;
    let root = solve(fixture(1., 0.5, &[0., 10., 20.]), configured).unwrap();
    let mut node = root.snapshot.fork().unwrap();
    let canonical = node
        .state
        .routes
        .iter()
        .position(|r| r.route.arcs[0] == 0)
        .unwrap();
    let mut alias = node.state.routes[canonical].clone();
    // A different stored waiting profile can share physical arcs. Full
    // elementary transfer owns one canonical variable for this path.
    Arc::make_mut(&mut alias).route.resource_profile[1][0] = 1.;
    Arc::make_mut(&mut alias).route.resource_profile[2][0] = 1.;
    Arc::make_mut(&mut alias).route.resources[0] = 1.;
    let tag = node.state.routes.len();
    node.state
        .tags
        .insert((alias.network, alias.route.identity()), tag as i32);
    let coefficients = entries(
        &node.state.oracles[0],
        &alias.route,
        node.state.shared_offset,
    )
    .unwrap();
    let column = node
        .master
        .add_column(alias.route.original_cost, &coefficients)
        .unwrap();
    let metadata = cg::Column {
        cost: alias.route.original_cost,
        reduced_cost: 0.,
        entries: coefficients,
        acceptable: true,
        tag: tag as i32,
        identity: format!("alias:{tag}"),
    };
    node.warm.columns.push(Some(metadata.clone()));
    node.warm.generated_pool.push(metadata);
    node.state.routes.push(alias);
    let complete = node.enumerate(&EnumerationOptions::default()).unwrap();
    assert_eq!(complete.master.upper[column], 0.);
    assert_eq!(complete.state.route_upper_bounds.get(&tag), Some(&0.));
    assert!(!complete.state.universe.as_ref().unwrap().contains(&tag));
    let result = complete.solve(options(BoundMode::Certified)).unwrap();
    assert_eq!(
        result.master.status,
        cg::Status::Optimal,
        "{:?}",
        result.master
    );
    assert!((result.master.lp - 5.).abs() < 1e-8);
    assert_eq!(result.values[tag], 0.);
}
#[cfg(feature = "highs")]
#[test]
fn complete_mip_restores_a_retired_route_needed_for_integer_feasibility() {
    let members: [&[usize]; 5] = [&[0, 1], &[1, 2], &[0, 2], &[0, 1, 2], &[]];
    let mut graph = Graph::<2>::new(7, 1).unwrap();
    let mut arc_entries = vec![];
    for (i, (&cost, rows)) in [1., 1., 1., 2., 0.].iter().zip(members).enumerate() {
        graph.add_arc(0, i + 1, cost).unwrap();
        graph.add_arc(i + 1, 6, 0.).unwrap();
        arc_entries.push(rows.iter().map(|&r| (r, 1.)).collect());
        arc_entries.push(vec![]);
    }
    let problem = Problem {
        rows: (0..3)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 10.).unwrap(),
            arc_entries,
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut configured = options(BoundMode::Certified);
    configured.cg.column_cleanup_threshold = 3.;
    configured.cg.column_cleanup_ratio = 0.8;
    configured.cg.insert_non_negative = false;
    let root = solve(problem, configured).unwrap();
    assert_eq!(root.master.status, cg::Status::Optimal, "{:?}", root.master);
    assert_eq!(root.master.lp, 1.5);
    let triple = root
        .routes
        .iter()
        .position(|r| r.route.arcs[0] == 6)
        .unwrap();
    assert!(!root.master.column_tags.contains(&(triple as i32)));
    let original_columns = root.snapshot.master.costs.len();
    // Three pair routes cover an odd number of rows only fractionally.
    // The empty dummy cannot help. The retired triple is the only integer
    // covering solution, proving optimum two by exhaustive alternatives.
    let result = root
        .snapshot
        .solve_enumerated_mip(&Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(
        result.mip.status,
        crate::master::solvers::mip::MipStatus::Optimal
    );
    assert_eq!(result.mip.solution.unwrap().objective, 2.);
    assert!(result
        .route_values
        .iter()
        .any(|r| r.route.route.arcs[0] == 6 && r.quantity == 1.));
    assert_eq!(root.snapshot.master.costs.len(), original_columns);
    assert!(!root
        .snapshot
        .warm
        .columns
        .iter()
        .flatten()
        .any(|c| c.tag == triple as i32));
}
#[test]
fn activity_caps_preserve_the_lp_unless_integer_projection_is_requested() {
    use crate::master::preprocessing::Bounds;
    let problem: Problem<2> = super::super::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem()
    .unwrap();
    let root = solve(problem, options(BoundMode::Certified)).unwrap();
    for (cap, lp) in [(0.5, 1.5), (0.25, 1.75)] {
        let mut arcs = vec![0.; 10];
        arcs[0] = 1.;
        let child = root
            .snapshot
            .with_activity_bounds(
                Bounds {
                    lower: f64::NEG_INFINITY,
                    upper: cap,
                },
                &[RowCoefficients {
                    network: 7,
                    arcs,
                    per_route: 0.,
                }],
            )
            .unwrap();
        let continuous = child.solve(options(BoundMode::Certified)).unwrap();
        assert_eq!(
            continuous.master.status,
            cg::Status::Optimal,
            "{:?}",
            continuous.master
        );
        assert!((continuous.master.lp - lp).abs() < 1e-8);
        let mut integer = options(BoundMode::Certified);
        integer.preprocessing.integer_domains = true;
        let result = child.solve(integer).unwrap();
        assert_eq!(
            result.master.status,
            cg::Status::Optimal,
            "{:?}",
            result.master
        );
        assert_eq!(result.master.lp, 2.);
    }
    assert_eq!(
        root.snapshot
            .solve(options(BoundMode::Certified))
            .unwrap()
            .master
            .lp,
        1.5
    );
}
