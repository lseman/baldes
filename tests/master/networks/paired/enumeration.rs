use super::*;

#[test]
fn fractional_invariant_controls_enumerate_from_priced_and_cold_snapshots() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-objective-mapping.json"
    ))
    .unwrap();
    for points in [false, true] {
        for cost in [-1., 1.] {
            for cold in [false, true] {
                let mut source = corpus["cases"][0]["model"].clone();
                for arc in source["networks"][0]["coupled_resources"][0]["arcs"]
                    .as_array_mut()
                    .unwrap()
                {
                    arc["increment"]["pairs"][0][0] = 0.5.into();
                }
                if points {
                    source["networks"][0] =
                        control_point_source_model(source["networks"][0].clone());
                }
                source["rows"] = serde_json::json!([{"sense":"ge","rhs":0.5}]);
                source["pricing_variables"] = serde_json::json!([{"network":42,"id":901,"cost":cost,"accumulated_resource":0,"entries":[[0,1.]]}]);
                let (problem, registry) = model::format::Model::from_json(&source.to_string())
                    .unwrap()
                    .compile::<2>()
                    .unwrap();
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = BoundMode::Certified;
                registry.configure_pricing(&mut options);
                let mut initial = options.clone();
                if cold {
                    initial.cg.time_limit = std::time::Duration::ZERO;
                }
                let parent = model::solve(problem, initial).unwrap();
                let complete = parent
                    .snapshot
                    .enumerate(&model::EnumerationOptions::default())
                    .unwrap();
                assert!(
                    complete.enumeration_report().complete,
                    "{:?}",
                    complete.enumeration_report()
                );
                let solved = complete.solve(options).unwrap();
                let expected = -20. + cost;
                assert_eq!(
                    solved.master.status,
                    Status::Optimal,
                    "{:#?}",
                    solved.master
                );
                assert!((solved.master.lp - expected).abs() < 1e-7);
                assert!(
                    solved.master.bound <= expected + 1e-10
                        && expected - solved.master.bound < 1e-7
                );
                assert_eq!(registry.activity(42, 901, &solved).unwrap().lower, 1.);
            }
        }
    }
}

#[test]
fn complete_invariant_universes_accept_new_paired_rows_and_preserve_fixed_copies() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-objective-mapping.json"
    ))
    .unwrap();
    for cost in [-1., 1.] {
        let mut source = corpus["cases"][0]["model"].clone();
        source["pricing_variables"][0]["cost"] = cost.into();
        let (problem, registry) = model::format::Model::from_json(&source.to_string())
            .unwrap()
            .compile::<2>()
            .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.bound_mode = BoundMode::Certified;
        registry.configure_pricing(&mut options);
        let parent = model::solve(problem, options.clone()).unwrap();
        let complete = parent
            .snapshot
            .enumerate(&model::EnumerationOptions::default())
            .unwrap();
        assert!(complete.enumeration_report().complete);
        let activities = [model::AccumulatedActivityCoefficients {
            network: 42,
            resource: 0,
            coefficient: 1.,
        }];
        let child = complete
            .with_accumulated_activity_row(
                Row {
                    sense: RowSense::GreaterEqual,
                    rhs: 2.,
                },
                &activities,
                &[],
            )
            .unwrap()
            .solve(options.clone())
            .unwrap();
        assert_eq!(child.master.status, Status::Optimal, "{:#?}", child.master);
        let expected = -20. + 2. * cost;
        assert!((child.master.lp - expected).abs() < 1e-7);
        assert!(child.snapshot.enumeration_report().complete);
        let index = child
            .routes
            .iter()
            .position(|r| r.route.arcs.len() == 2)
            .unwrap();
        let fixed = child.snapshot.fix_route(index, 1.).unwrap();
        let extra = fixed
            .with_accumulated_activity_row(
                Row {
                    sense: RowSense::Equal,
                    rhs: 2.,
                },
                &activities,
                &[],
            )
            .unwrap()
            .solve(options.clone())
            .unwrap();
        assert_eq!(extra.master.status, Status::Optimal, "{:#?}", extra.master);
        assert!((extra.master.lp - expected).abs() < 1e-7);
        let impossible = complete
            .with_accumulated_activity_row(
                Row {
                    sense: RowSense::GreaterEqual,
                    rhs: 3.,
                },
                &activities,
                &[],
            )
            .unwrap()
            .solve(options)
            .unwrap();
        assert_eq!(
            impossible.master.status,
            Status::Infeasible,
            "{:#?}",
            impossible.master
        );
    }
}

#[test]
fn partial_control_enumeration_is_atomic_across_networks_and_limits() {
    let fixed: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-objective-mapping.json"
    ))
    .unwrap();
    let variable: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = fixed["cases"][0]["model"].clone();
    let mut second = variable["cases"][0]["model"]["networks"][0].clone();
    second["id"] = 7.into();
    second["front_program_domain"] = false.into();
    second["coupled_resources"][0]["policy"] = "strict_domain".into();
    source["networks"].as_array_mut().unwrap().push(second);
    source["rows"] = serde_json::json!([{"sense":"ge","rhs":1.}]);
    source["pricing_variables"] = serde_json::json!([
        {"network":42,"id":901,"cost":-1.,"accumulated_resource":0,"entries":[[0,1.]]},
        {"network":7,"id":901,"cost":1.,"accumulated_resource":0,"entries":[[0,1.]]}
    ]);
    let (problem, registry) = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.bound_mode = BoundMode::Certified;
    registry.configure_pricing(&mut options);
    let parent = model::solve(problem, options.clone()).unwrap();
    assert!((parent.master.lp + 39.).abs() < 1e-7);
    for limits in [
        model::EnumerationOptions::default(),
        model::EnumerationOptions {
            max_work: 0,
            ..Default::default()
        },
        model::EnumerationOptions {
            max_routes: 1,
            ..Default::default()
        },
        model::EnumerationOptions {
            time_limit: std::time::Duration::ZERO,
            ..Default::default()
        },
    ] {
        let partial = parent.snapshot.enumerate(&limits).unwrap();
        let report = partial.enumeration_report();
        assert!(
            !report.complete && report.transferred == 0 && report.failure.is_some(),
            "{report:?}"
        );
        let resumed = partial.solve(options.clone()).unwrap();
        assert_eq!(
            resumed.master.status,
            Status::Optimal,
            "{:#?}",
            resumed.master
        );
        assert!((resumed.master.lp + 39.).abs() < 1e-7);
    }
    assert_eq!(parent.snapshot.enumeration_report().attempts, 0);
}

#[test]
fn mixed_invariant_networks_share_full_enumeration_under_every_schedule() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-objective-mapping.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    let mut second = control_point_source_model(corpus["cases"][4]["model"]["networks"][0].clone());
    second["id"] = 7.into();
    source["networks"].as_array_mut().unwrap().push(second);
    source["rows"] = serde_json::json!([{"sense":"ge","rhs":3.}]);
    source["pricing_variables"] = serde_json::json!([
        {"network":42,"id":901,"cost":-1.,"accumulated_resource":0,"entries":[[0,1.]]},
        {"network":7,"id":901,"cost":2.,"accumulated_resource":0,"entries":[[0,1.]]}
    ]);
    for schedule in [
        model::NetworkPricingSchedule::FullSweep,
        model::NetworkPricingSchedule::Cyclic,
        model::NetworkPricingSchedule::Sorted,
        model::NetworkPricingSchedule::PromisingFirst,
        model::NetworkPricingSchedule::UnvisitedFirst,
    ] {
        let (problem, registry) = model::format::Model::from_json(&source.to_string())
            .unwrap()
            .compile::<2>()
            .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.bound_mode = BoundMode::Certified;
        options.network_pricing_schedule = schedule;
        registry.configure_pricing(&mut options);
        let parent = model::solve(problem, options.clone()).unwrap();
        let complete = parent
            .snapshot
            .enumerate(&model::EnumerationOptions::default())
            .unwrap();
        let report = complete.enumeration_report();
        assert!(report.complete && report.routes == 6, "{report:?}");
        assert_eq!(report.networks, [(42, 3), (7, 3)]);
        let solved = complete.solve(options).unwrap();
        assert_eq!(
            solved.master.status,
            Status::Optimal,
            "{:#?}",
            solved.master
        );
        assert!((solved.master.lp + 38.).abs() < 1e-7);
        assert!(solved.master.bound <= -38. + 1e-10 && -38. - solved.master.bound < 1e-7);
        for network in [42, 7] {
            assert_eq!(registry.activity(network, 901, &solved).unwrap().lower, 2.);
        }
    }
}

#[cfg(feature = "highs")]
#[test]
fn uncertain_activity_encoding_cannot_turn_a_feasible_source_domain_into_mip_infeasibility() {
    use baldes::graph::coupled::CoupledResource;
    use baldes::master::solvers::mip::{MipOptions, MipStatus};
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (a, b) in [(0, 1), (1, 2), (2, 3)] {
        graph.add_arc(a, b, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![1., 2f64.powi(-53), -1.],
            weights: vec![0., 0., 1.],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    graph
        .set_coupled_objective_coefficients(&[(0, 0.)])
        .unwrap();
    let (_, source) = graph
        .coupled_kernel()
        .unwrap()
        .certify_accumulated_activity(0, &[0, 1, 2], &mut 1_000_000, None)
        .unwrap();
    assert!(source[0] != source[1] && source[0] <= 2f64.powi(-53) && source[1] >= 2f64.powi(-53));
    // Exact source activity is2^-53 per route;2^53 copies satisfy the row.
    // Rounded replay stores zero, making the finite backend matrix infeasible.
    let copies = 9_007_199_254_740_992.;
    assert_eq!(2f64.powi(-53) * copies, 1.);
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::GreaterEqual,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 42,
            graph,
            multiplicity: Window::new(0., copies).unwrap(),
            arc_entries: vec![vec![]; 3],
            route_entries: vec![],
            activity_entries: vec![model::AccumulatedActivityEntry {
                resource: 0,
                row: 0,
                coefficient: 1.,
            }],
        }],
    };
    let mut options = Options::default();
    options.cg.time_limit = std::time::Duration::ZERO;
    let cold = model::solve(problem, options).unwrap();
    let complete = cold
        .snapshot
        .enumerate(&model::EnumerationOptions::default())
        .unwrap();
    assert!(
        complete.enumeration_report().complete,
        "{:?}",
        complete.enumeration_report()
    );
    let mip = complete
        .solve_enumerated_mip(&MipOptions::default())
        .unwrap()
        .unwrap();
    assert_eq!(mip.mip.status, MipStatus::Incomplete, "{:?}", mip.mip);
    assert_eq!(mip.mip.bound, f64::NEG_INFINITY);
    assert!(mip.mip.solution.is_none());
}

#[test]
fn signed_invariant_paired_rows_preserve_cutoff_improvements() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-objective-mapping.json"
    ))
    .unwrap();
    for case in [0, 4] {
        for cost in [-1., 1.] {
            let mut source = corpus["cases"][case]["model"].clone();
            source["rows"] = serde_json::json!([{"sense":"ge","rhs":1.}]);
            source["pricing_variables"][0]["cost"] = cost.into();
            source["pricing_variables"][0]["entries"] = serde_json::json!([[0, 1.]]);
            let (problem, registry) = model::format::Model::from_json(&source.to_string())
                .unwrap()
                .compile::<2>()
                .unwrap();
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = BoundMode::Certified;
            registry.configure_pricing(&mut options);
            let root = model::solve(problem, options.clone()).unwrap();
            let expected = -20. + 2. * cost;
            let view = root
                .snapshot
                .enumerate_below(expected + 0.5, &model::EnumerationOptions::default())
                .unwrap();
            assert!(view.report().complete);
            assert!(view.routes().any(|r| r.route.arcs.len() == 2));
            #[cfg(feature = "highs")]
            {
                use baldes::master::solvers::mip::{MipOptions, MipStatus};
                let mip = view.solve_mip(&MipOptions::default()).unwrap();
                assert_eq!(mip.mip.status, MipStatus::Optimal, "{:?}", mip.mip);
                assert!((mip.mip.solution.unwrap().objective - expected).abs() < 1e-7);
                let strict = root
                    .snapshot
                    .enumerate_below(expected, &model::EnumerationOptions::default())
                    .unwrap();
                assert_eq!(
                    strict.solve_mip(&MipOptions::default()).unwrap().mip.status,
                    MipStatus::ObjectiveCutoff
                );
            }
            assert!(
                (root.snapshot.solve(options).unwrap().master.lp - root.master.lp).abs() < 1e-7
            );
        }
    }
}

#[test]
fn negative_relative_paired_objectives_use_the_global_initial_minimum() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for case in [0, 18] {
        for points in [false, true] {
            for mode in [BoundMode::Numerical, BoundMode::Certified] {
                for coefficient in [-1., -2.] {
                    let mut source = corpus["cases"][case]["model"].clone();
                    source["networks"][0]["front_program_domain"] = false.into();
                    source["networks"][0]["coupled_resources"][0]["policy"] =
                        "strict_domain".into();
                    source["networks"][0]["coupled_resources"][0]["initial"]["pairs"][0][0] =
                        2.into();
                    source["networks"][0]["coupled_resources"][0]["initial"]["pairs"][1][0] =
                        1.into();
                    if points {
                        source["networks"][0] =
                            control_point_source_model(source["networks"][0].clone());
                    }
                    source["networks"][0]["coupled_resources"][0]["normalization"] =
                        "final_minus_initial_minimum".into();
                    let mut problem = model::format::Model::from_json(&source.to_string())
                        .unwrap()
                        .problem::<2>()
                        .unwrap();
                    problem.networks[0]
                        .graph
                        .set_coupled_objective_coefficients(&[(0, coefficient)])
                        .unwrap();
                    assert!(problem.networks[0]
                        .graph
                        .clone()
                        .set_front_program_domain(true)
                        .is_err());
                    let kernel = problem.networks[0].graph.coupled_kernel().unwrap();
                    let controls = [baldes::graph::coupled::PairedControl {
                        resource: 0,
                        base: vec![0.5, 1.5, 2.5],
                    }];
                    let selected = kernel
                        .replay_with_controls(&[1, 2], &controls)
                        .unwrap()
                        .unwrap();
                    assert_eq!(selected.paired_values, [(0, 6.5)]);
                    assert_eq!(selected.cost, coefficient * (6.5 - 2.));
                    // The chosen initial value is2.5, but normalization uses
                    // the global initial minimum2, independently of this witness.
                    let proof = kernel
                        .certify_with_controls(&[1, 2], &controls, &mut 1_000_000, None)
                        .unwrap();
                    assert!(proof.0 && proof.1[0] <= selected.cost && proof.1[1] >= selected.cost);
                    assert!(kernel
                        .certify_with_controls(&[1, 2], &[], &mut 1_000_000, None)
                        .is_none());
                    let mut options = Options::default();
                    options.cg.stabilize = false;
                    options.cg.bound_mode = mode;
                    let output = model::solve(problem, options.clone()).unwrap();
                    let expected = -20. + coefficient * (8. - 2.);
                    assert_eq!(
                        output.master.status,
                        Status::Optimal,
                        "{:#?}",
                        output.master
                    );
                    assert!(
                        (output.master.lp - expected).abs() < 1e-7,
                        "{:#?}",
                        output.master
                    );
                    assert!(
                        output.master.bound <= expected + 1e-10
                            && expected - output.master.bound < 1e-7
                    );
                    let index = output.values.iter().position(|&value| value > 0.5).unwrap();
                    assert_eq!(output.routes[index].route.coupled_values, [(0, 8.)]);
                    let fixed = output
                        .snapshot
                        .fix_route(index, 1.)
                        .unwrap()
                        .solve(options.clone())
                        .unwrap();
                    assert_eq!(fixed.master.status, Status::Optimal);
                    assert!((fixed.master.lp - expected).abs() < 1e-7);
                    assert!(
                        (output.snapshot.solve(options).unwrap().master.lp - expected).abs() < 1e-7
                    );
                    let mut tree_options = baldes::master::tree::ModelTreeOptions::default();
                    tree_options.node.cg.stabilize = false;
                    tree_options.node.cg.bound_mode = mode;
                    let tree = output.snapshot.solve_tree(tree_options).unwrap();
                    assert_eq!(
                        tree.status,
                        baldes::master::tree::TreeStatus::Optimal,
                        "{tree:?}"
                    );
                    assert!((tree.incumbent - expected).abs() < 1e-7);
                }
            }
        }
    }
}

#[test]
fn monotone_paired_rows_enumerate_only_proved_dominating_endpoints() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for points in [false, true] {
        for (cost, sense, coefficient, rhs, gamma) in [
            (-1., "ge", 1., 5., 8.),
            (-1., "le", -1., -5., 8.),
            (1., "le", 1., 4., 3.),
            (1., "ge", -1., -4., 3.),
            (0., "ge", 1., 5., 8.),
            (0., "le", 1., 4., 3.),
        ] {
            let mut source = corpus["cases"][0]["model"].clone();
            source["networks"][0]["front_program_domain"] = false.into();
            source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
            if points {
                source["networks"][0] = control_point_source_model(source["networks"][0].clone());
            }
            source["rows"] = serde_json::json!([{"sense":sense,"rhs":rhs}]);
            source["pricing_variables"] = serde_json::json!([
                {"network":42,"id":901,"cost":cost,"accumulated_resource":0,"entries":[[0,coefficient]]}
            ]);
            let (problem, registry) = model::format::Model::from_json(&source.to_string())
                .unwrap()
                .compile::<2>()
                .unwrap();
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = BoundMode::Certified;
            registry.configure_pricing(&mut options);
            let root = model::solve(problem, options.clone()).unwrap();
            let expected = -20. + cost * gamma;
            assert!(
                (root.master.lp - expected).abs() < 1e-7,
                "{:#?}",
                root.master
            );
            let complete = root
                .snapshot
                .enumerate(&model::EnumerationOptions::default())
                .unwrap();
            assert!(
                complete.enumeration_report().complete,
                "{points} {cost} {sense}: {:?}",
                complete.enumeration_report()
            );
            let solved = complete.solve(options).unwrap();
            assert_eq!(solved.master.status, Status::Optimal);
            assert!((solved.master.lp - expected).abs() < 1e-7);
            assert!(
                solved.master.bound <= expected + 1e-10 && expected - solved.master.bound < 1e-7
            );
            let index = solved.values.iter().position(|&x| x > 0.5).unwrap();
            assert_eq!(solved.routes[index].route.coupled_values, [(0, gamma)]);
            #[cfg(feature = "highs")]
            {
                use baldes::master::solvers::mip::{MipOptions, MipStatus};
                let mip = solved
                    .snapshot
                    .solve_enumerated_mip(&MipOptions::default())
                    .unwrap()
                    .unwrap();
                assert_eq!(mip.mip.status, MipStatus::Optimal, "{:?}", mip.mip);
                assert!((mip.mip.solution.unwrap().objective - expected).abs() < 1e-7);
                let cutoff = root
                    .snapshot
                    .enumerate_below(expected + 0.5, &model::EnumerationOptions::default())
                    .unwrap();
                assert_eq!(
                    cutoff.solve_mip(&MipOptions::default()).unwrap().mip.status,
                    MipStatus::Optimal
                );
            }
        }
    }
}

#[test]
fn equality_and_opposed_paired_rows_keep_varying_control_transfer_incomplete() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for (cost, rows, expected) in [
        (0., serde_json::json!([{"sense":"eq","rhs":4.}]), -20.),
        (
            0.,
            serde_json::json!([{"sense":"ge","rhs":4.},{"sense":"le","rhs":5.}]),
            -20.,
        ),
        (1., serde_json::json!([{"sense":"ge","rhs":4.}]), -16.),
        (-1., serde_json::json!([{"sense":"le","rhs":4.}]), -24.),
    ] {
        let mut source = corpus["cases"][0]["model"].clone();
        source["networks"][0]["front_program_domain"] = false.into();
        source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
        let entries: Vec<_> = (0..rows.as_array().unwrap().len())
            .map(|r| (r, 1.))
            .collect();
        source["rows"] = rows;
        source["pricing_variables"] = serde_json::json!([
            {"network":42,"id":901,"cost":cost,"accumulated_resource":0,"entries":entries}
        ]);
        let (problem, registry) = model::format::Model::from_json(&source.to_string())
            .unwrap()
            .compile::<2>()
            .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        registry.configure_pricing(&mut options);
        let root = model::solve(problem, options.clone()).unwrap();
        assert_eq!(root.master.status, Status::Optimal);
        assert!((root.master.lp - expected).abs() < 1e-7);
        let attempt = root
            .snapshot
            .enumerate(&model::EnumerationOptions::default())
            .unwrap();
        assert!(!attempt.enumeration_report().complete);
        assert!((attempt.solve(options).unwrap().master.lp - expected).abs() < 1e-7);
    }
}

#[test]
fn compatible_activity_rows_reuse_endpoint_universes_and_fixed_copies() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for points in [false, true] {
        for (cost, sense, coefficient, rhs, gamma) in [
            (-1., RowSense::GreaterEqual, 1., 14., 8.),
            (1., RowSense::LessEqual, 1., 7., 3.),
        ] {
            let mut source = corpus["cases"][0]["model"].clone();
            source["networks"][0]["front_program_domain"] = false.into();
            source["networks"][0]["multiplicity"] = serde_json::json!([0., 2.]);
            source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
            if points {
                source["networks"][0] = control_point_source_model(source["networks"][0].clone());
            }
            source["pricing_variables"] = serde_json::json!([
                {"network":42,"id":901,"cost":cost,"accumulated_resource":0}
            ]);
            let (problem, registry) = model::format::Model::from_json(&source.to_string())
                .unwrap()
                .compile::<2>()
                .unwrap();
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = BoundMode::Certified;
            registry.configure_pricing(&mut options);
            let root = model::solve(problem, options.clone()).unwrap();
            let complete = root
                .snapshot
                .enumerate(&model::EnumerationOptions::default())
                .unwrap();
            assert!(complete.enumeration_report().complete);
            let expected = 2. * (-20. + cost * gamma);
            let terms = [model::AccumulatedActivityCoefficients {
                network: 42,
                resource: 0,
                coefficient,
            }];
            let row = Row { sense, rhs };
            let child = complete
                .with_accumulated_activity_row(row.clone(), &terms, &[])
                .unwrap()
                .solve(options.clone())
                .unwrap();
            assert_eq!(child.master.status, Status::Optimal);
            assert!((child.master.lp - expected).abs() < 1e-7);
            assert!(child.snapshot.enumeration_report().complete);
            let index = child.values.iter().position(|&q| q > 0.5).unwrap();
            let fixed = child.snapshot.fix_route(index, 1.).unwrap();
            let again = fixed
                .with_accumulated_activity_row(row.clone(), &terms, &[])
                .unwrap()
                .solve(options.clone())
                .unwrap();
            assert!((again.master.lp - expected).abs() < 1e-7);
            assert!(again.snapshot.enumeration_report().complete);
            assert!((complete.solve(options.clone()).unwrap().master.lp - expected).abs() < 1e-7);
            let equality = Row {
                sense: RowSense::Equal,
                rhs: 2. * gamma,
            };
            assert!(complete
                .with_accumulated_activity_row(equality, &terms, &[])
                .is_err());
            let opposite = Row {
                sense: if cost < 0. {
                    RowSense::LessEqual
                } else {
                    RowSense::GreaterEqual
                },
                rhs: 8.,
            };
            assert!(complete
                .with_accumulated_activity_row(opposite, &terms, &[])
                .is_err());
            let mixed = [
                terms[0].clone(),
                model::AccumulatedActivityCoefficients {
                    network: 42,
                    resource: 0,
                    coefficient: -0.25,
                },
            ];
            assert!(complete
                .with_accumulated_activity_row(row, &mixed, &[])
                .is_err());
            #[cfg(feature = "highs")]
            {
                use baldes::master::solvers::mip::{MipOptions, MipStatus};
                let mip = again
                    .snapshot
                    .solve_enumerated_mip(&MipOptions::default())
                    .unwrap()
                    .unwrap();
                assert_eq!(mip.mip.status, MipStatus::Optimal, "{:?}", mip.mip);
                assert!((mip.mip.solution.unwrap().objective - expected).abs() < 1e-7);
            }
        }
    }
}

#[test]
fn joint_paired_rows_recover_interior_integer_witnesses_in_the_shared_pricing_path() {
    use baldes::master::tree::{ModelTreeOptions, TreeStatus};
    let base: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/joint-interior-model.json"
    ))
    .unwrap();
    for representation in 0..4 {
        let mut source = base.clone();
        let points = control_point_source_model(source["networks"][0].clone());
        for r in 0..2 {
            if representation & (1 << r) != 0 {
                source["networks"][0]["coupled_resources"][r] =
                    points["coupled_resources"][r].clone();
            }
        }
        for quantity in [1., 2.] {
            source["networks"][0]["multiplicity"] = serde_json::json!([quantity, quantity]);
            for shape in 0..3 {
                let mut input = source.clone();
                if shape == 1 {
                    input["networks"] = input["networks"][0].clone();
                } else if shape == 2 {
                    input["networks"]
                        .as_array_mut()
                        .unwrap()
                        .push(serde_json::json!({
                            "id":7,"vertices":2,"source":0,"sink":1,"multiplicity":[1,1],
                            "arcs":[{"tail":0,"head":1,"cost":2.}]
                        }));
                }
                for mode in [BoundMode::Numerical, BoundMode::Certified] {
                    for schedule in [
                        model::NetworkPricingSchedule::FullSweep,
                        model::NetworkPricingSchedule::Cyclic,
                        model::NetworkPricingSchedule::Sorted,
                        model::NetworkPricingSchedule::PromisingFirst,
                        model::NetworkPricingSchedule::UnvisitedFirst,
                    ] {
                        let (problem, registry) =
                            model::format::Model::from_json(&input.to_string())
                                .unwrap()
                                .compile::<2>()
                                .unwrap();
                        let mut options = ModelTreeOptions::default();
                        options.node.cg.stabilize = false;
                        options.node.cg.bound_mode = mode;
                        options.node.network_pricing_schedule = schedule;
                        registry.configure_tree(&mut options).unwrap();
                        let solved = baldes::master::tree::solve(problem, options).unwrap();
                        assert_eq!(
                            solved.status,
                            TreeStatus::Optimal,
                            "{representation} {shape} {mode:?} {schedule:?}"
                        );
                        let expected = -10. * quantity + if shape == 2 { 2. } else { 0. };
                        assert_eq!(solved.incumbent, expected);
                        assert!(solved.bound <= expected && solved.bound > expected - 1e-8);
                        // Integer copies may use distinct controls at the same
                        // optimum. The shared rows constrain their weighted
                        // totals, not a homogeneous per-copy control.
                        let primary: Vec<_> = solved
                            .routes
                            .iter()
                            .filter(|r| r.route.network == 42)
                            .collect();
                        assert_eq!(primary.iter().map(|r| r.quantity).sum::<f64>(), quantity);
                        for resource in 0..2 {
                            let total: f64 = primary
                                .iter()
                                .map(|r| {
                                    r.quantity
                                        * r.route
                                            .route
                                            .coupled_values
                                            .iter()
                                            .find(|(index, _)| *index == resource)
                                            .unwrap()
                                            .1
                                })
                                .sum();
                            assert_eq!(total, 4.);
                        }
                        if quantity == 1. {
                            assert_eq!(primary.len(), 1);
                            assert_eq!(
                                primary[0].route.route.coupled_values,
                                vec![(0, 4.), (1, 4.)]
                            );
                        }
                        assert_eq!(
                            solved
                                .routes
                                .iter()
                                .filter(|r| r.route.network == 7)
                                .map(|r| r.quantity)
                                .sum::<f64>(),
                            if shape == 2 { 1. } else { 0. }
                        );
                    }
                }
            }
        }
    }
}
