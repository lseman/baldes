use super::*;
use baldes::graph::coupled::CoupledResource;
use baldes::master::model::{AccumulatedActivityCoefficients, NetworkPricingSchedule};

fn problem(cumulative: bool, ids: &[i32]) -> Problem<2> {
    let networks = ids
        .iter()
        .enumerate()
        .map(|(order, &id)| {
            let permutation = if order % 2 == 0 {
                [0, 1, 2, 3]
            } else {
                [2, 0, 3, 1]
            };
            let mut graph = Graph::<2>::new(4, 1).unwrap();
            graph.source = permutation[0];
            graph.sink = permutation[3];
            graph.elementary = true;
            graph.vertices[graph.source].elements.clear();
            graph.vertices[graph.sink].elements.clear();
            for vertex in 0..4 {
                graph.set_bounds(vertex, 0, 0., 2.).unwrap();
            }
            for (tail, head, cost) in [(0, 1, -20.), (1, 3, 0.), (0, 2, -20.), (2, 3, 0.)] {
                let arc = graph
                    .add_arc(permutation[tail], permutation[head], cost)
                    .unwrap();
                graph.arc_mut(arc).unwrap().consumption[0] = 1.;
            }
            let resource = if cumulative {
                CoupledResource::CumulativeCost {
                    initial: 0.,
                    increments: vec![2., 1., 3., 1.],
                    weights: vec![2., 2., 1., 1.],
                    sink_upper: None,
                    source_upper: None,
                }
            } else {
                CoupledResource::CustomResource {
                    increments: vec![1.; 4],
                    weights: vec![3., 4., 0., 1.],
                    forward_bound: None,
                    backward_bound: None,
                }
            };
            graph
                .set_coupled_resources(&[
                    CoupledResource::Selection {
                        initial: 3,
                        allowed: vec![3; 4],
                        vertices: vec![],
                        relaxed: false,
                    },
                    resource,
                ])
                .unwrap();
            Network {
                id,
                graph,
                multiplicity: Window::new(0., 1.).unwrap(),
                arc_entries: vec![vec![]; 4],
                route_entries: vec![],
                activity_entries: vec![],
            }
        })
        .collect();
    Problem {
        rows: vec![],
        networks,
    }
}
fn configured(mode: BoundMode, schedule: NetworkPricingSchedule, direction: f64) -> Options {
    let mut options = super::options();
    options.cg.bound_mode = mode;
    options.network_pricing_schedule = schedule;
    options.pricing_options = vec![
        ("BALDESmaxNumOfColsPerExactIteration".into(), 1.),
        ("BALDESuseBidirectionalSearch".into(), direction),
    ];
    options
}
fn activity(network: i32, coefficient: f64) -> AccumulatedActivityCoefficients {
    AccumulatedActivityCoefficients {
        network,
        resource: 1,
        coefficient,
    }
}

#[test]
fn activity_rows_match_independent_registered_control_lps_in_all_pricing_modes() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-derived-activity.json"
    ))
    .unwrap();
    let mut compared = 0;
    for case in corpus["cases"].as_array().unwrap() {
        if case["objective_resource_coefficient"] != 1 {
            continue;
        }
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for schedule in [
                NetworkPricingSchedule::FullSweep,
                NetworkPricingSchedule::Cyclic,
                NetworkPricingSchedule::Sorted,
                NetworkPricingSchedule::PromisingFirst,
                NetworkPricingSchedule::UnvisitedFirst,
            ] {
                for direction in [0., 1., 2.] {
                    let options = configured(mode, schedule, direction);
                    let parent = model::solve(
                        problem(case["kind"] == "cumulative", &[42]),
                        options.clone(),
                    )
                    .unwrap();
                    // Only the initial best path is exported, so the changed
                    // row must cause fresh pricing of the missing path.
                    assert!(parent.routes.iter().all(|route| route.route.arcs == [2, 3]));
                    let output = if case["row"].is_null() {
                        parent.snapshot.solve(options).unwrap()
                    } else {
                        parent
                            .snapshot
                            .with_accumulated_activity_row(
                                Row {
                                    sense: RowSense::GreaterEqual,
                                    rhs: case["row"]["lower"].as_f64().unwrap(),
                                },
                                &[activity(42, 1.)],
                                &[RowCoefficients {
                                    network: 42,
                                    arcs: vec![
                                        case["row"]["coefficient_on_first_path_arc"]
                                            .as_f64()
                                            .unwrap(),
                                        0.,
                                        0.,
                                        0.,
                                    ],
                                    per_route: 0.,
                                }],
                            )
                            .unwrap()
                            .solve(options)
                            .unwrap()
                    };
                    let expected = case["oracle"]["objective_float"].as_f64().unwrap();
                    optimal(&output, expected);
                    if mode == BoundMode::Certified {
                        let mut pieces = case["oracle"]["objective"].as_str().unwrap().split('/');
                        let numerator: f64 = pieces.next().unwrap().parse().unwrap();
                        let denominator = pieces
                            .next()
                            .map(|part| part.parse::<f64>().unwrap())
                            .unwrap_or(1.);
                        let product = output.master.bound * denominator;
                        let residual = output.master.bound.mul_add(denominator, -product);
                        assert!(
                            product < numerator || (product == numerator && residual <= 0.),
                            "{} {:?}",
                            case["name"],
                            output.master
                        );
                    }
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, 180);
}

#[test]
fn shared_activity_rows_cover_multiple_networks_and_permuted_terminals() {
    for cumulative in [false, true] {
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for direction in [0., 1., 2.] {
                let options = configured(mode, NetworkPricingSchedule::Cyclic, direction);
                let parent = model::solve(problem(cumulative, &[7, 42]), options.clone()).unwrap();
                let row = Row {
                    sense: RowSense::GreaterEqual,
                    rhs: if cumulative { 15. } else { 5. },
                };
                let output = parent
                    .snapshot
                    .with_accumulated_activity_row(row, &[activity(42, 1.), activity(7, 1.)], &[])
                    .unwrap()
                    .solve(options)
                    .unwrap();
                optimal(&output, if cumulative { -25. } else { -35. });
            }
        }
    }
}

#[test]
fn signed_repeated_activity_terms_replay_one_original_row() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let options = configured(mode, NetworkPricingSchedule::FullSweep, 1.);
        let parent = model::solve(problem(false, &[42]), options.clone()).unwrap();
        let output = parent
            .snapshot
            .with_accumulated_activity_row(
                Row {
                    sense: RowSense::LessEqual,
                    rhs: -5.,
                },
                &[activity(42, -0.5), activity(42, -0.5)],
                &[],
            )
            .unwrap()
            .solve(options)
            .unwrap();
        optimal(&output, -15.);
    }
}

#[test]
fn linear_activity_cancellation_and_fixed_route_rhs_are_refreshed_once() {
    let options = configured(BoundMode::Certified, NetworkPricingSchedule::FullSweep, 1.);
    let parent = model::solve(problem(false, &[42]), options.clone()).unwrap();
    let fixed = parent.snapshot.fix_route(0, 1.).unwrap();
    let cancelled = fixed
        .with_accumulated_activity_row(
            Row {
                sense: RowSense::Equal,
                rhs: 0.,
            },
            &[activity(42, 1.)],
            &[RowCoefficients {
                network: 42,
                arcs: vec![-11., 0., -2., 0.],
                per_route: 0.,
            }],
        )
        .unwrap();
    optimal(&cancelled.solve(options.clone()).unwrap(), -18.);
    let impossible = fixed
        .with_accumulated_activity_row(
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 5.,
            },
            &[activity(42, 1.)],
            &[],
        )
        .unwrap()
        .solve(options.clone())
        .unwrap();
    assert_eq!(impossible.master.status, Status::Infeasible);
    optimal(&parent.snapshot.solve(options).unwrap(), -18.);
}

#[test]
fn activity_rows_survive_tree_children_and_validate_original_integer_witnesses() {
    use baldes::master::tree::{ModelTreeOptions, TreeStatus};
    for cumulative in [false, true] {
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            let options = configured(mode, NetworkPricingSchedule::FullSweep, 1.);
            let parent = model::solve(problem(cumulative, &[42]), options.clone()).unwrap();
            let node = parent
                .snapshot
                .with_accumulated_activity_row(
                    Row {
                        sense: RowSense::GreaterEqual,
                        rhs: if cumulative { 8. } else { 5. },
                    },
                    &[activity(42, 1.)],
                    &[],
                )
                .unwrap();
            let tree = ModelTreeOptions {
                node: options,
                ..Default::default()
            };
            let output = node.solve_tree(tree).unwrap();
            assert_eq!(output.status, TreeStatus::Optimal, "{output:?}");
            assert_eq!(output.incumbent, if cumulative { -10. } else { -9. });
            assert_eq!(output.open_nodes, 0);
        }
    }
}

#[test]
fn activity_arc_branch_children_close_the_original_dual_domain() {
    let options = configured(BoundMode::Certified, NetworkPricingSchedule::FullSweep, 1.);
    let parent = model::solve(problem(false, &[42]), options.clone()).unwrap();
    let node = parent
        .snapshot
        .with_accumulated_activity_row(
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 5.,
            },
            &[activity(42, 1.)],
            &[],
        )
        .unwrap();
    let child = node
        .with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &[RowCoefficients {
                network: 42,
                arcs: vec![1., 0., 0., 0.],
                per_route: 0.,
            }],
        )
        .unwrap()
        .solve(options.clone())
        .unwrap();
    assert_eq!(
        child.master.status,
        Status::Infeasible,
        "{:?} {:?}",
        child.master,
        child.duals
    );
    assert!(child
        .master
        .feasibility
        .iter()
        .any(|report| report.certified_infeasible
            && report.lower_bound.is_some_and(|bound| bound > 0.)));
    let child = node
        .with_row(
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.,
            },
            &[RowCoefficients {
                network: 42,
                arcs: vec![1., 0., 0., 0.],
                per_route: 0.,
            }],
        )
        .unwrap()
        .solve(options)
        .unwrap();
    optimal(&child, -9.);
}

#[test]
fn imported_activity_rows_survive_new_columns_and_open_session_rebuilds() {
    use baldes::master::preprocessing::Bounds;
    use model::imported::{self, Column, ColumnBatch, ImportOptions, Subproblem};
    use model::MasterVariable;
    let bounds = |lower, upper| Bounds { lower, upper };
    let column = |id, cost, value| Column {
        id,
        cost,
        bounds: bounds(0., 1.),
        integer: true,
        entries: vec![(99, value)],
    };
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let options = configured(mode, NetworkPricingSchedule::FullSweep, 1.);
        let root = imported::solve(
            problem(false, &[42]),
            vec![MasterVariable {
                id: 99,
                cost: 0.,
                bounds: bounds(0., 3.),
                integer: false,
                entries: vec![],
            }],
            vec![Subproblem {
                id: 77,
                variables: vec![99],
                multiplicity: bounds(0., 1.),
                columns: vec![column(0, 100., 1.)],
            }],
            options.clone(),
        )
        .unwrap();
        let parent = root.snapshot().unwrap();
        let row = Row {
            sense: RowSense::GreaterEqual,
            rhs: 5.,
        };
        let child = parent
            .with_accumulated_activity_row(row.clone(), &[activity(42, 1.)], &[(99, 1.)])
            .unwrap();
        optimal(&child.solve(options.clone()).unwrap().output, -15.);
        let expanded = child
            .solve_with_columns(77, vec![column(1, 0., 3.)], options.clone())
            .unwrap();
        optimal(&expanded.output, -18.);
        assert_eq!(expanded.values.original, vec![(99, 3.)]);
        let open = parent
            .begin_imports(&[77])
            .unwrap()
            .with_accumulated_activity_row(row, &[activity(42, 1.)], &[(99, 1.)])
            .unwrap();
        let streamed = open
            .collect(&ImportOptions::default(), |_, _| {
                Ok(ColumnBatch {
                    columns: vec![column(1, 0., 3.)],
                    complete: true,
                })
            })
            .unwrap()
            .session
            .solve(options.clone())
            .unwrap();
        optimal(&streamed.output, -18.);
        assert_eq!(streamed.values.original, vec![(99, 3.)]);
        optimal(&parent.solve(options).unwrap().output, -18.);
    }
}
