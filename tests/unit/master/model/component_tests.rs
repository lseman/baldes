use super::*;
use crate::master::preprocessing::Bounds;
#[cfg(feature = "highs")]
use crate::master::solvers::mip::MipOptions;

fn fixture() -> (Output<2>, Options, ComponentClassBranching) {
    fixture_with_coefficient(1.)
}

fn fixture_with_coefficient(
    cheap_coefficient: f64,
) -> (Output<2>, Options, ComponentClassBranching) {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (a, b, c) in [(0, 1, 1.), (1, 3, 0.), (0, 2, 5.), (2, 3, 0.)] {
        graph.add_arc(a, b, c).unwrap();
    }
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, cheap_coefficient)], vec![], vec![(0, 1.)], vec![]],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.preprocessing.enabled = false;
    options.gap_enumeration = false;
    let class = ComponentClassBranching {
        name: "expensive".into(),
        priority: 1.,
        minimum_fractionality: 0.02,
        network: 7,
        coordinates: vec![ComponentCoordinate {
            id: 99,
            arc_values: vec![None, None, Some(1), None],
        }],
        sequence: ComponentSequence {
            bounds: vec![ComponentBound {
                coordinate: 99,
                threshold: 1.,
                sense: ComponentSense::GreaterEqual,
            }],
        },
    };
    (solve(problem, options.clone()).unwrap(), options, class)
}

#[test]
fn component_class_prices_future_columns_and_replays_fixed_and_enumerated_routes() {
    let (root, options, class) = fixture();
    assert_eq!(root.master.lp, 1.);
    let child = root
        .snapshot
        .with_component_bounds(
            &class,
            Bounds {
                lower: 1.,
                upper: f64::INFINITY,
            },
        )
        .unwrap();
    let solved = child.solve(options.clone()).unwrap();
    assert_eq!(solved.master.lp, 5.);
    assert!(solved.master.certified_bound <= 5.);
    // A degenerate LP basis may put the price on a loaded route's upper cap.
    // At this explicit dual point, joint class/path replay must recover 5;
    // independent [0,1] class envelopes alone yield only 1.
    let mut duals = vec![0.; solved.snapshot.master.rhs.len()];
    let offset = solved.snapshot.state.shared_offset;
    duals[offset] = 1.;
    duals[offset + 1] = 4.;
    let bound =
        bounds::certificate(&solved.snapshot.state, &solved.snapshot.master, &duals).unwrap();
    assert!(bound >= 5. - 1e-7 && bound <= 5., "{bound}");
    let tag = solved
        .routes
        .iter()
        .position(|r| r.route.arcs == [2, 3])
        .unwrap();
    let fixed = solved.snapshot.fix_route(tag, 1.).unwrap();
    assert_eq!(fixed.solve(options.clone()).unwrap().master.lp, 5.);
    let complete = child.enumerate(&EnumerationOptions::default()).unwrap();
    let complete = complete.solve(options).unwrap();
    assert_eq!(complete.master.lp, 5.);
    #[cfg(feature = "highs")]
    assert_eq!(
        complete
            .snapshot
            .solve_enumerated_mip(&MipOptions::default())
            .unwrap()
            .unwrap()
            .mip
            .solution
            .unwrap()
            .objective,
        5.
    );
    assert_eq!(
        root.snapshot.solve(Options::default()).unwrap().master.lp,
        1.
    );
}

#[test]
fn explicit_component_generator_branches_on_class_quantity() {
    let (root, options, mut class) = fixture_with_coefficient(2.);
    assert_eq!(root.master.lp, 0.5);
    class.coordinates[0].arc_values = vec![Some(1), None, None, None];
    class.priority = 1e9;
    let proposal = class.proposal(&root, 1e-7).unwrap().unwrap();
    assert_eq!(proposal.value, 0.5);
    assert!(matches!(
        proposal.restriction,
        model_strong::Restriction::Component(_)
    ));
    let mut tree = ModelTreeOptions::default();
    tree.node = options;
    tree.branching_registry.branching = NetworkBranchingPolicy::Phased;
    tree.branching_registry.component_classes = vec![class];
    let solved = root.snapshot.solve_tree(tree).unwrap();
    assert_eq!(solved.incumbent, 5.);
}

#[test]
fn empty_class_counts_complete_routes_without_reading_unused_coordinates() {
    let (root, options, mut class) = fixture_with_coefficient(2.);
    class.sequence.bounds.clear();
    // Each individual input is valid; their unused path sum exceeds binary64's
    // exact integer domain. Native empty membership never reads that sum.
    class.coordinates[0].arc_values = vec![Some(1_i64 << 53); 4];
    let coefficients = class.coefficients(&root.snapshot).unwrap();
    let pricing = crate::cuts::soft_conflicts::PresencePricing::new(
        4,
        vec![crate::cuts::soft_conflicts::PricingInput {
            row: coefficients.row,
            arc_events: coefficients.arc_events,
            dual: 0.,
        }],
    )
    .unwrap();
    assert_eq!(pricing.route_coefficients(&[0]).unwrap(), [0]);
    assert_eq!(pricing.route_coefficients(&[0, 1]).unwrap(), [1]);
    assert_eq!(pricing.route_coefficients(&[2, 3]).unwrap(), [1]);
    let child = root
        .snapshot
        .with_component_bounds(
            &class,
            Bounds {
                lower: 1.,
                upper: 1.,
            },
        )
        .unwrap();
    let solved = child.solve(options.clone()).unwrap();
    assert_eq!(solved.master.lp, 5.);
    let tag = solved
        .snapshot
        .state
        .routes
        .iter()
        .position(|route| route.route.arcs == [2, 3])
        .unwrap();
    let fixed = solved.snapshot.fix_route(tag, 1.).unwrap();
    assert_eq!(fixed.solve(options).unwrap().master.lp, 5.);
    assert!(root.snapshot.state.component_branches.is_empty());
}

#[test]
fn empty_class_keeps_coordinate_definition_validation_even_without_bounds() {
    let (root, _, mut class) = fixture();
    class.sequence.bounds.clear();
    let unrestricted = Bounds {
        lower: f64::NEG_INFINITY,
        upper: f64::INFINITY,
    };
    let valid = class.clone();
    class.coordinates[0].arc_values[0] = Some(i64::MAX);
    assert!(root
        .snapshot
        .with_component_bounds(&class, unrestricted)
        .is_err());
    class = valid.clone();
    class.coordinates[0].arc_values.pop();
    assert!(root
        .snapshot
        .with_component_bounds(&class, unrestricted)
        .is_err());
    class = valid.clone();
    class.coordinates.push(class.coordinates[0].clone());
    assert!(root
        .snapshot
        .with_component_bounds(&class, unrestricted)
        .is_err());
    class = valid;
    class.name.clear();
    assert!(root
        .snapshot
        .with_component_bounds(&class, unrestricted)
        .is_err());
}

#[test]
fn projected_component_rows_match_one_thousand_original_membership_points() {
    #[derive(serde::Deserialize)]
    struct Predicate {
        coordinate: usize,
        sense: String,
        threshold_bits: u64,
    }
    #[derive(serde::Deserialize)]
    struct Case {
        predicates: Vec<Predicate>,
        values: Vec<Option<i64>>,
        satisfied: bool,
    }
    #[derive(serde::Deserialize)]
    struct Capture {
        coordinate_ids: Vec<usize>,
        cases: Vec<Case>,
    }
    let capture: Capture =
        serde_json::from_str(include_str!("component_membership_fixtures.json")).unwrap();
    assert_eq!(capture.cases.len(), 1000);
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    for (index, case) in capture.cases.into_iter().enumerate() {
        let mut coordinates = vec![ComponentCoordinate {
            id: 99,
            arc_values: vec![Some(1_i64 << 53); 2],
        }];
        // Definition order differs from native numeric-ID order.
        for identifier in [11, 7, 2] {
            let position = capture
                .coordinate_ids
                .iter()
                .position(|&id| id == identifier)
                .unwrap();
            coordinates.push(ComponentCoordinate {
                id: identifier,
                arc_values: vec![case.values[position], None],
            });
        }
        let class = ComponentClassBranching {
            name: "membership".into(),
            priority: 1.,
            minimum_fractionality: 0.02,
            network: 7,
            coordinates,
            sequence: ComponentSequence {
                bounds: case
                    .predicates
                    .into_iter()
                    .map(|bound| ComponentBound {
                        coordinate: bound.coordinate,
                        threshold: f64::from_bits(bound.threshold_bits),
                        sense: if bound.sense == "G" {
                            ComponentSense::GreaterEqual
                        } else {
                            ComponentSense::LessEqual
                        },
                    })
                    .collect(),
            },
        };
        let coefficients = class.for_graph(&graph).unwrap();
        let pricing = crate::cuts::soft_conflicts::PresencePricing::new(
            2,
            vec![crate::cuts::soft_conflicts::PricingInput {
                row: coefficients.row,
                arc_events: coefficients.arc_events,
                dual: 0.,
            }],
        )
        .unwrap();
        assert_eq!(
            pricing.route_coefficients(&[0]).unwrap(),
            [0],
            "partial case {index}"
        );
        assert_eq!(
            pricing.route_coefficients(&[0, 1]).unwrap(),
            [usize::from(case.satisfied)],
            "native case {index}"
        );
    }
}

#[test]
fn projected_coordinates_preserve_repetition_cancellation_and_used_overflow() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let mut class = ComponentClassBranching {
        name: "signed".into(),
        priority: 1.,
        minimum_fractionality: 0.02,
        network: 7,
        coordinates: vec![
            ComponentCoordinate {
                id: 99,
                arc_values: vec![Some(1_i64 << 53); 3],
            },
            ComponentCoordinate {
                id: 10,
                arc_values: vec![Some(3), Some(-1), Some(-2)],
            },
            ComponentCoordinate {
                id: 2,
                arc_values: vec![Some(0), None, None],
            },
        ],
        sequence: ComponentSequence {
            bounds: vec![
                ComponentBound {
                    coordinate: 10,
                    threshold: 0.,
                    sense: ComponentSense::GreaterEqual,
                },
                ComponentBound {
                    coordinate: 10,
                    threshold: 1.,
                    sense: ComponentSense::LessEqual,
                },
                ComponentBound {
                    coordinate: 2,
                    threshold: 0.,
                    sense: ComponentSense::GreaterEqual,
                },
            ],
        },
    };
    let replay = |class: &ComponentClassBranching| {
        let coefficients = class.for_graph(&graph).unwrap();
        crate::cuts::soft_conflicts::PresencePricing::new(
            3,
            vec![crate::cuts::soft_conflicts::PricingInput {
                row: coefficients.row,
                arc_events: coefficients.arc_events,
                dual: 0.,
            }],
        )
        .unwrap()
    };
    let pricing = replay(&class);
    assert_eq!(pricing.route_coefficients(&[0, 2]).unwrap(), [1]);
    assert_eq!(pricing.route_coefficients(&[0, 1, 2]).unwrap(), [1]);
    assert_eq!(pricing.route_coefficients(&[0, 1, 1, 2]).unwrap(), [0]);
    class.coordinates[1].arc_values = vec![Some(1_i64 << 53); 3];
    assert!(replay(&class).route_coefficients(&[0, 2]).is_err());
    class.coordinates[0].arc_values[0] = Some(i64::MAX);
    assert!(
        class.for_graph(&graph).is_err(),
        "unused definitions remain validated"
    );
}
