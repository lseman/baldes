use super::*;

#[test]
fn paired_original_variables_use_selected_columns_and_absolute_nonnegative_objectives() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for case_index in [0, 18] {
        for coefficient in [0_f64, 0.5, 1., 2., 5., 8., 1e-308] {
            for mode in [BoundMode::Numerical, BoundMode::Certified] {
                for direction in [0., 1., 2.] {
                    let mut source = corpus["cases"][case_index]["model"].clone();
                    source["pricing_variables"] = serde_json::json!([{
                        "network":42,"id":901,"cost":coefficient,"accumulated_resource":0
                    }]);
                    let (problem, registry) = model::format::Model::from_json(&source.to_string())
                        .unwrap()
                        .compile::<2>()
                        .unwrap();
                    let options = || {
                        let mut options = Options::default();
                        options.cg.stabilize = false;
                        options.cg.bound_mode = mode;
                        options.pricing_options =
                            vec![("BALDESuseBidirectionalSearch".into(), direction)];
                        registry.configure_pricing(&mut options);
                        options
                    };
                    let output = model::solve(problem, options()).unwrap();
                    let expected = (-20. + 3. * coefficient).min(0.);
                    assert_eq!(output.master.status, Status::Optimal);
                    assert_eq!(output.master.lp, expected);
                    assert!(
                        output.master.bound <= expected && expected - output.master.bound < 1e-7
                    );
                    let activity = registry.activity(42, 901, &output).unwrap();
                    let gamma = if expected < 0. { 3. } else { 0. };
                    assert_eq!((activity.lower, activity.upper), (gamma, gamma));
                    let result = model::format::SolveResult::root(&output, 0.)
                        .with_original_variables(&registry)
                        .unwrap();
                    assert_eq!(result.original_variables[0].lower, gamma);
                    let resumed = output.snapshot.solve(options()).unwrap();
                    assert_eq!(registry.activity(42, 901, &resumed).unwrap(), activity);
                    if expected < 0. {
                        let route = output
                            .routes
                            .iter()
                            .zip(&output.values)
                            .find(|(_, q)| **q > 0.)
                            .unwrap()
                            .0;
                        assert!(registry
                            .activity_from_paths(42, 901, [(42, route.route.arcs.as_slice(), 1.)])
                            .is_err());
                        let wrong = [(0, 4.)];
                        assert!(registry
                            .activity_from_columns(
                                42,
                                901,
                                [model::OriginalColumn {
                                    network: 42,
                                    arcs: &route.route.arcs,
                                    coupled_controls: None,
                                    coupled_values: Some(&wrong),
                                    quantity: 1.
                                }]
                            )
                            .is_err());
                    }
                }
            }
        }
    }
}

#[test]
fn paired_original_variables_reject_unimplemented_control_changes() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for variable in [
        serde_json::json!({"network":42,"id":901,"cost":-1.,"accumulated_resource":0}),
        serde_json::json!({"network":42,"id":901,"cost":1.,"accumulated_resource":0,"integer":true}),
    ] {
        let mut source = corpus["cases"][0]["model"].clone();
        source["pricing_variables"] = serde_json::json!([variable]);
        assert!(model::format::Model::from_json(&source.to_string())
            .unwrap()
            .compile::<2>()
            .is_err());
    }
    let mut source = corpus["cases"][0]["model"].clone();
    source["rows"] = serde_json::json!([{"sense":"le","rhs":10.}]);
    source["pricing_variables"] = serde_json::json!([{
        "network":42,"id":901,"cost":1.,"accumulated_resource":0,"entries":[[0,1.]]
    }]);
    let error = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap_err()
        .to_string();
    assert!(error.contains("source model"));
}

#[test]
fn paired_original_identities_remain_network_scoped_under_every_schedule() {
    use model::NetworkPricingSchedule;
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    let mut second = source["networks"][0].clone();
    second["id"] = serde_json::json!(7);
    second["coupled_resources"][0]["initial"]["pairs"][0][0] = serde_json::json!(2.);
    source["networks"].as_array_mut().unwrap().push(second);
    source["pricing_variables"] = serde_json::json!([
        {"network":42,"id":901,"cost":0.5,"accumulated_resource":0},
        {"network":7,"id":901,"cost":2.,"accumulated_resource":0}
    ]);
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for schedule in [
            NetworkPricingSchedule::FullSweep,
            NetworkPricingSchedule::Cyclic,
            NetworkPricingSchedule::Sorted,
            NetworkPricingSchedule::PromisingFirst,
            NetworkPricingSchedule::UnvisitedFirst,
        ] {
            let (problem, registry) = model::format::Model::from_json(&source.to_string())
                .unwrap()
                .compile::<2>()
                .unwrap();
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = mode;
            options.network_pricing_schedule = schedule;
            registry.configure_pricing(&mut options);
            let output = model::solve(problem, options).unwrap();
            assert_eq!(output.master.status, Status::Optimal);
            assert_eq!(output.master.lp, -28.5);
            for network in [7, 42] {
                let value = registry.activity(network, 901, &output).unwrap();
                let expected = if network == 7 { 5. } else { 3. };
                assert_eq!((value.lower, value.upper), (expected, expected));
            }
        }
    }
}

#[test]
fn paired_original_weight_changes_the_selected_route_and_reported_value() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for (coefficient, expected, gamma) in [(0.5, -19., 4.), (2., -14., 3.)] {
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for direction in [0., 1., 2.] {
                let mut source = corpus["cases"][0]["model"].clone();
                source["networks"][0]["arcs"][3]["cost"] = serde_json::json!(-21.);
                for index in [3, 4] {
                    let pairs = source["networks"][0]["coupled_resources"][0]["arcs"][index]
                        ["increment"]["pairs"]
                        .as_array_mut()
                        .unwrap();
                    pairs[0][0] = serde_json::json!(2.);
                    for pair in pairs.iter_mut().skip(1) {
                        pair[0] = serde_json::json!(0.);
                    }
                }
                source["pricing_variables"] = serde_json::json!([{
                    "network":42,"id":901,"cost":coefficient,"accumulated_resource":0
                }]);
                let (problem, registry) = model::format::Model::from_json(&source.to_string())
                    .unwrap()
                    .compile::<2>()
                    .unwrap();
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = mode;
                options.pricing_options = vec![("BALDESuseBidirectionalSearch".into(), direction)];
                registry.configure_pricing(&mut options);
                let output = model::solve(problem, options).unwrap();
                assert_eq!(output.master.status, Status::Optimal);
                assert_eq!(output.master.lp, expected);
                let value = registry.activity(42, 901, &output).unwrap();
                assert_eq!((value.lower, value.upper), (gamma, gamma));
            }
        }
    }
}

#[test]
fn control_point_absolute_original_bindings_keep_nonzero_initial_values() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    // Exact integral encodings: convert independent slope segments to knots.
    fn knots(function: &serde_json::Value) -> serde_json::Value {
        let pairs = function["pairs"].as_array().unwrap();
        let (mut y, mut x) = (pairs[0][0].as_f64().unwrap(), pairs[0][1].as_f64().unwrap());
        let mut points = vec![serde_json::json!([x, y])];
        for pair in &pairs[1..] {
            let end = pair[1].as_f64().unwrap();
            y += pair[0].as_f64().unwrap() * (end - x);
            x = end;
            points.push(serde_json::json!([x, y]));
        }
        serde_json::json!({"points":points})
    }
    for case_index in [36, 39] {
        for coefficient in [0_f64, 0.5, 1., 2., 8., 1e-308] {
            for mode in [BoundMode::Numerical, BoundMode::Certified] {
                for direction in [0., 1., 2.] {
                    let mut source = corpus["cases"][case_index]["model"].clone();
                    source["networks"][0]["front_program_domain"] = serde_json::json!(false);
                    let resource = &mut source["networks"][0]["coupled_resources"][0];
                    resource["initial"] = knots(&resource["initial"]);
                    for arc in resource["arcs"].as_array_mut().unwrap() {
                        for key in ["arrival", "increment", "lower", "upper"] {
                            arc[key] = knots(&arc[key]);
                        }
                    }
                    resource["kind"] = serde_json::json!("paired_piecewise_linear");
                    resource.as_object_mut().unwrap().remove("policy");
                    source["pricing_variables"] = serde_json::json!([{
                        "network":42,"id":901,"cost":coefficient,"accumulated_resource":0
                    }]);
                    let (problem, registry) = model::format::Model::from_json(&source.to_string())
                        .unwrap()
                        .compile::<2>()
                        .unwrap();
                    let mut options = Options::default();
                    options.cg.stabilize = false;
                    options.cg.bound_mode = mode;
                    options.pricing_options =
                        vec![("BALDESuseBidirectionalSearch".into(), direction)];
                    registry.configure_pricing(&mut options);
                    let output = model::solve(problem, options.clone()).unwrap();
                    let expected = (-20. + 5. * coefficient).min(0.);
                    assert_eq!(output.master.status, Status::Optimal);
                    assert_eq!(output.master.lp, expected);
                    assert!(
                        output.master.bound <= expected && expected - output.master.bound < 1e-7
                    );
                    let gamma = if expected < 0. { 5. } else { 0. };
                    let activity = registry.activity(42, 901, &output).unwrap();
                    assert_eq!((activity.lower, activity.upper), (gamma, gamma));
                    assert_eq!(
                        registry
                            .activity(42, 901, &output.snapshot.solve(options).unwrap())
                            .unwrap(),
                        activity
                    );
                    let formatted = model::format::SolveResult::root(&output, 0.)
                        .with_original_variables(&registry)
                        .unwrap();
                    assert_eq!(formatted.original_variables[0].lower, gamma);
                }
            }
        }
    }
}

#[test]
fn controlled_routes_keep_identity_costs_and_original_values_through_pool_resume_and_fixing() {
    use baldes::graph::coupled::PairedControl;
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for control_points in [false, true] {
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for direction in [0., 1., 2.] {
                let mut source = corpus["cases"][0]["model"].clone();
                source["networks"][0]["front_program_domain"] = false.into();
                source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
                let mut second = corpus["cases"][18]["model"]["networks"][0].clone();
                second["id"] = 7.into();
                second["front_program_domain"] = false.into();
                second["coupled_resources"][0]["policy"] = "strict_domain".into();
                if control_points {
                    source["networks"][0] =
                        control_point_source_model(source["networks"][0].clone());
                    second = control_point_source_model(second);
                }
                source["networks"].as_array_mut().unwrap().push(second);
                source["pricing_variables"] = serde_json::json!([
                    {"network":42,"id":901,"cost":1.,"accumulated_resource":0},
                    {"network":7,"id":901,"cost":2.,"accumulated_resource":0}
                ]);
                let (problem, registry) = model::format::Model::from_json(&source.to_string())
                    .unwrap()
                    .compile::<2>()
                    .unwrap();
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = mode;
                options.pricing_options = vec![("BALDESuseBidirectionalSearch".into(), direction)];
                registry.configure_pricing(&mut options);
                let root = model::solve(problem, options.clone()).unwrap();
                optimal(&root, -31.);
                let original = root
                    .routes
                    .iter()
                    .position(|route| route.network == 42 && route.route.arcs.len() == 2)
                    .unwrap();
                let generated = root
                    .snapshot
                    .optimize_route_control(
                        original,
                        0,
                        false,
                        Some(Window::new(4., 4.).unwrap()),
                        Default::default(),
                    )
                    .unwrap();
                assert!(generated.complete);
                assert_eq!(generated.dependent, Some(4.));
                assert!(generated.lower_bound.unwrap() <= 4.);
                let control = generated.control.unwrap();
                let node = root
                    .snapshot
                    .with_route_controls(original, vec![control.clone()])
                    .unwrap();
                let result = node.solve(options.clone()).unwrap();
                optimal(&result, -31.);
                let controlled = result
                    .routes
                    .iter()
                    .position(|route| {
                        route.network == 42 && route.route.coupled_controls == [control.clone()]
                    })
                    .unwrap();
                assert_eq!(
                    result.routes[controlled].route.coupled_values,
                    vec![(0, 4.)]
                );
                assert_eq!(result.routes[controlled].route.original_cost, -16.);
                assert_ne!(controlled, original);
                if direction == 1. && mode == BoundMode::Certified {
                    let complete = result
                        .snapshot
                        .enumerate(&model::EnumerationOptions::default())
                        .unwrap();
                    assert!(complete.enumeration_report().complete);
                    assert!(complete
                        .with_route_controls(original, vec![control.clone()])
                        .is_err());
                    assert!(complete.fix_route(controlled, 1.).is_err());
                }
                let fixed = result
                    .snapshot
                    .fix_route(controlled, 1.)
                    .unwrap()
                    .solve(options.clone())
                    .unwrap();
                optimal(&fixed, -30.);
                let activity = registry.activity(42, 901, &fixed).unwrap();
                assert_eq!((activity.lower, activity.upper), (4., 4.));
                let other = registry.activity(7, 901, &fixed).unwrap();
                assert_eq!((other.lower, other.upper), (3., 3.));
                let formatted = model::format::SolveResult::root(&fixed, 0.)
                    .with_original_variables(&registry)
                    .unwrap();
                assert_eq!(
                    formatted.fixed_routes[0].coupled_controls,
                    [control.clone()]
                );
                assert_eq!(formatted.fixed_routes[0].cost, -16.);
                if direction == 1. {
                    let mut tree_options = baldes::master::tree::ModelTreeOptions::default();
                    tree_options.node = options.clone();
                    let tree = fixed.snapshot.solve_tree(tree_options).unwrap();
                    assert_eq!(tree.status, baldes::master::tree::TreeStatus::Optimal);
                    assert_eq!(tree.incumbent, -30.);
                }
                let resumed = fixed.snapshot.solve(options.clone()).unwrap();
                optimal(&resumed, -30.);
                assert_eq!(registry.activity(42, 901, &resumed).unwrap(), activity);
                // Equivalent source witnesses reuse the same model column.
                let alternative = PairedControl {
                    resource: 0,
                    base: if control.base == [0.5, 1.5, 2.5] {
                        vec![0., 2., 3.]
                    } else {
                        vec![0.5, 1.5, 2.5]
                    },
                };
                let more = result
                    .snapshot
                    .with_route_controls(original, vec![alternative.clone()])
                    .unwrap()
                    .solve(options.clone())
                    .unwrap();
                let variants = more
                    .routes
                    .iter()
                    .filter(|route| route.network == 42 && !route.route.coupled_controls.is_empty())
                    .count();
                assert_eq!(variants, 1);
                let duplicate = more
                    .snapshot
                    .with_route_controls(
                        original,
                        vec![PairedControl {
                            resource: 0,
                            base: vec![-0., 2., 3.],
                        }],
                    )
                    .unwrap()
                    .solve(options.clone())
                    .unwrap();
                assert_eq!(duplicate.routes.len(), more.routes.len());
                assert!(root
                    .snapshot
                    .with_route_controls(
                        original,
                        vec![PairedControl {
                            resource: 0,
                            base: vec![0.5, 1., 2.]
                        }]
                    )
                    .is_err());
            }
        }
    }
}

#[test]
fn paired_shared_rows_price_interior_and_reversed_terminal_controls() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for case in [0, 18] {
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for (sense, rhs, cost, expected) in [
                ("ge", 4., 1., -16.),
                ("ge", 7., 1., -13.),
                ("le", 8., 1., -17.),
                ("ge", 4., 0., -20.),
            ] {
                let mut source = corpus["cases"][case]["model"].clone();
                source["networks"][0]["front_program_domain"] = false.into();
                source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
                source["rows"] = serde_json::json!([{"sense":sense,"rhs":rhs}]);
                source["pricing_variables"] = serde_json::json!([{
                    "network":42,"id":901,"cost":cost,"accumulated_resource":0,"entries":[[0,1.]]
                }]);
                let (problem, registry) = model::format::Model::from_json(&source.to_string())
                    .unwrap()
                    .compile::<2>()
                    .unwrap();
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = mode;
                registry.configure_pricing(&mut options);
                let output = model::solve(problem, options.clone()).unwrap();
                assert_eq!(
                    output.master.status,
                    Status::Optimal,
                    "case={case} {sense}{rhs} {:#?}",
                    output.master
                );
                assert!(
                    (output.master.lp - expected).abs() < 1e-7,
                    "{:#?}",
                    output.master
                );
                assert!(
                    output.master.bound <= expected + 1e-10
                        && expected - output.master.bound < 1e-7,
                    "{:#?}",
                    output.master
                );
                let gamma = registry.activity(42, 901, &output).unwrap();
                if sense == "ge" {
                    assert!(gamma.lower >= rhs - 1e-7, "{gamma:?}");
                }
                let resumed = output.snapshot.solve(options).unwrap();
                assert!((resumed.master.lp - expected).abs() < 1e-7);
            }
        }
    }
}

#[test]
fn negative_paired_objectives_share_scheduling_and_rows_with_positive_networks() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for schedule in [
        model::NetworkPricingSchedule::FullSweep,
        model::NetworkPricingSchedule::Cyclic,
        model::NetworkPricingSchedule::Sorted,
        model::NetworkPricingSchedule::PromisingFirst,
        model::NetworkPricingSchedule::UnvisitedFirst,
    ] {
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for row in [false, true] {
                let mut source = corpus["cases"][0]["model"].clone();
                source["networks"][0]["front_program_domain"] = false.into();
                source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
                let mut second = control_point_source_model(source["networks"][0].clone());
                second["id"] = 7.into();
                source["networks"].as_array_mut().unwrap().push(second);
                let entries = if row {
                    source["rows"] = serde_json::json!([{"sense":"le","rhs":9.}]);
                    serde_json::json!([[0, 1.]])
                } else {
                    serde_json::json!([])
                };
                source["pricing_variables"] = serde_json::json!([
                    {"network":42,"id":901,"cost":-2.,"accumulated_resource":0,"entries":entries},
                    {"network":7,"id":901,"cost":0.5,"accumulated_resource":0,"entries":entries}
                ]);
                let (problem, registry) = model::format::Model::from_json(&source.to_string())
                    .unwrap()
                    .compile::<2>()
                    .unwrap();
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = mode;
                options.network_pricing_schedule = schedule;
                registry.configure_pricing(&mut options);
                let result = model::solve(problem, options).unwrap();
                let expected = if row { -50.5 } else { -54.5 };
                assert_eq!(
                    result.master.status,
                    Status::Optimal,
                    "{:#?}",
                    result.master
                );
                assert!(
                    (result.master.lp - expected).abs() < 1e-7,
                    "{:#?}",
                    result.master
                );
                assert!(
                    result.master.bound <= expected + 1e-10
                        && expected - result.master.bound < 1e-7
                );
                let first = registry.activity(42, 901, &result).unwrap();
                let second = registry.activity(7, 901, &result).unwrap();
                assert!(
                    (first.lower - if row { 6. } else { 8. }).abs() < 1e-7
                        && (second.lower - 3.).abs() < 1e-7,
                    "{first:?} {second:?}"
                );
            }
        }
    }
}

#[test]
fn negative_paired_costs_keep_deterministic_activity_certificates_and_fixed_row_contributions() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    source["networks"][0]["front_program_domain"] = false.into();
    source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
    source["networks"][0]["coupled_resources"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({
            "kind":"custom_resource","increments":[1.,1.,1.,1.,1.],"weights":[1.,1.,1.,1.,1.]
        }));
    source["rows"] = serde_json::json!([{"sense":"ge","rhs":3.}]);
    source["pricing_variables"] = serde_json::json!([
        {"network":42,"id":901,"cost":-1.,"accumulated_resource":0},
        {"network":42,"id":902,"cost":0.,"accumulated_resource":1,"entries":[[0,1.]]}
    ]);
    let (problem, registry) = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.bound_mode = BoundMode::Certified;
    registry.configure_pricing(&mut options);
    let result = model::solve(problem, options.clone()).unwrap();
    assert_eq!(
        result.master.status,
        Status::Optimal,
        "{:#?}",
        result.master
    );
    assert!((result.master.lp + 28.).abs() < 1e-7);
    let index = result
        .routes
        .iter()
        .position(|r| r.route.arcs.len() == 2 && r.route.coupled_values == vec![(0, 8.)])
        .unwrap();
    let fixed = result
        .snapshot
        .fix_route(index, 1.)
        .unwrap()
        .solve(options)
        .unwrap();
    assert_eq!(fixed.master.status, Status::Optimal, "{:#?}", fixed.master);
    assert!((fixed.master.lp + 28.).abs() < 1e-7 && (fixed.master.bound + 28.).abs() < 1e-7);
}

#[test]
fn paired_objective_mapping_corrects_all_eight_retained_modern_registrations() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-objective-mapping.json"
    ))
    .unwrap();
    assert_eq!(
        corpus["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    for case in corpus["cases"].as_array().unwrap() {
        let expected = case["independent_objective"].as_f64().unwrap();
        let coefficient = case["coefficient"].as_f64().unwrap();
        if coefficient == -1. {
            assert_eq!(case["reference_root"], -18.);
            assert_eq!(expected, -22.);
        }
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for points in [false, true] {
                let mut source = case["model"].clone();
                if points {
                    source["networks"][0] =
                        control_point_source_model(source["networks"][0].clone());
                }
                let (problem, registry) = model::format::Model::from_json(&source.to_string())
                    .unwrap()
                    .compile::<2>()
                    .unwrap();
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = mode;
                registry.configure_pricing(&mut options);
                let result = model::solve(problem, options).unwrap();
                assert_eq!(
                    result.master.status,
                    Status::Optimal,
                    "{:#?}",
                    result.master
                );
                assert!(
                    (result.master.lp - expected).abs() < 1e-7,
                    "{:#?}",
                    result.master
                );
                assert!(
                    result.master.bound <= expected + 1e-10
                        && expected - result.master.bound < 1e-7,
                    "{:#?}",
                    result.master
                );
                let gamma = registry.activity(42, 901, &result).unwrap();
                assert_eq!((gamma.lower, gamma.upper), (2., 2.));
            }
        }
    }
}

#[test]
fn nonconstant_source_pricing_matches_unit_reference_and_signed_control_oracles() {
    let base: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-nonconstant-source-model.json"
    ))
    .unwrap();
    // A least-waiting two-arc path has base x,x+1,x+2 and Gamma=3+2*x,
    // bounded above by 8. Witnesses x=0,2.5,.5 attain the three optima;
    // free waiting also permits other schedules with the same final Gamma.
    for (coefficient, upper, expected, gamma) in [
        (1., None, -17., 3.),
        (-1., None, -28., 8.),
        (-1., Some(4.), -24., 4.),
    ] {
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for knots in [false, true] {
                let mut source = base.clone();
                source["pricing_variables"][0]["cost"] = coefficient.into();
                if let Some(upper) = upper {
                    source["rows"] = serde_json::json!([{"sense":"le","rhs":upper}]);
                    source["networks"][0]["activity_entries"] = serde_json::json!([[0, 0, 1.]]);
                }
                if knots {
                    source["networks"][0] =
                        control_point_source_model(source["networks"][0].clone());
                }
                let (problem, registry) = model::format::Model::from_json(&source.to_string())
                    .unwrap()
                    .compile::<2>()
                    .unwrap();
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = mode;
                registry.configure_pricing(&mut options);
                let root = model::solve(problem, options).unwrap();
                assert_eq!(root.master.status, Status::Optimal);
                assert!((root.master.lp - expected).abs() < 1e-7, "{root:?}");
                assert!(
                    root.master.bound <= expected + 1e-10 && expected - root.master.bound < 1e-7
                );
                let activity = registry.activity(42, 901, &root).unwrap();
                assert_eq!((activity.lower, activity.upper), (gamma, gamma));
                if coefficient == -1. && upper.is_none() {
                    for (route, value) in root
                        .routes
                        .iter()
                        .zip(&root.values)
                        .filter(|(_, value)| **value > 0.)
                    {
                        assert!(*value > 0.);
                        let base = &route.route.coupled_controls[0].base;
                        assert!(base[1] >= base[0] + 1. && base[2] >= base[1] + 1.);
                        assert_eq!(2. + base[0] + base[1], 8.);
                        assert_eq!(route.route.coupled_values, vec![(0, 8.)]);
                    }
                }
            }
        }
    }
}

#[test]
fn paired_variable_boxes_cover_all_controls_including_negative_objectives() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for points in [false, true] {
        for cost in [-1., 0., 1., 2.] {
            let mut source = corpus["cases"][0]["model"].clone();
            source["networks"][0]["front_program_domain"] = false.into();
            source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
            if points {
                source["networks"][0] = control_point_source_model(source["networks"][0].clone());
            }
            source["pricing_variables"] =
                serde_json::json!([{"network":42,"id":901,"cost":cost,"accumulated_resource":0}]);
            let (problem, registry) = model::format::Model::from_json(&source.to_string())
                .unwrap()
                .compile::<2>()
                .unwrap();
            let configure = |upper| {
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = BoundMode::Certified;
                registry
                    .configure_boxes(
                        &mut options,
                        &[model::OriginalVariableBound {
                            network: 42,
                            id: 901,
                            lower: 0.,
                            upper: Some(upper),
                        }],
                    )
                    .unwrap();
                options
            };
            let output = model::solve(problem.clone(), configure(8.)).unwrap();
            assert_eq!(
                output.master.status,
                Status::Optimal,
                "{:#?}",
                output.master
            );
            let expected = -20. + cost * if cost < 0. { 8. } else { 3. };
            assert!((output.master.lp - expected).abs() < 1e-7);
            assert!(
                output.master.bound <= expected + 1e-10 && expected - output.master.bound < 1e-7
            );
            // Minimum-only columns would incorrectly validate the upper3 box.
            assert!(model::solve(problem.clone(), configure(3.)).is_err());
            let resumed = output.snapshot.solve(configure(8.)).unwrap();
            assert!((resumed.master.lp - expected).abs() < 1e-7);
        }
    }
}

#[test]
fn paired_integer_bindings_use_shared_mapped_branch_rows_and_tree_replay() {
    use baldes::master::{
        preprocessing::Bounds,
        tree::{ModelTreeOptions, TreeStatus},
    };
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-objective-mapping.json"
    ))
    .unwrap();
    for case in [0, 4] {
        for points in [false, true] {
            let mut source = corpus["cases"][case]["model"].clone();
            if points {
                source["networks"][0] = control_point_source_model(source["networks"][0].clone());
            }
            for (p, arc) in source["networks"][0]["arcs"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                arc["cost"] = if p == 0 {
                    100.
                } else if p == 1 || p == 3 {
                    20.
                } else {
                    0.
                }
                .into();
                if p == 1 || p == 3 {
                    arc["entries"] = serde_json::json!([[0, 1.]]);
                }
            }
            source["rows"] = serde_json::json!([{"sense":"ge","rhs":0.25}]);
            source["pricing_variables"] = serde_json::json!([{"network":42,"id":901,"cost":0.,"accumulated_resource":0,"integer":true}]);
            let (problem, registry) = model::format::Model::from_json(&source.to_string())
                .unwrap()
                .compile::<2>()
                .unwrap();
            let branch = &registry.branch_variables()[0];
            assert_eq!(branch.name, "X_42_901");
            assert_eq!(branch.networks[0].activities, [(0, 1.)]);
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = BoundMode::Certified;
            registry.configure_pricing(&mut options);
            let parent = model::solve(problem.clone(), options.clone()).unwrap();
            assert_eq!(
                parent.master.status,
                Status::Optimal,
                "{:#?}",
                parent.master
            );
            assert!((parent.master.lp - 5.).abs() < 1e-7);
            let activity = registry.activity(42, 901, &parent).unwrap();
            assert!((activity.lower - 0.5).abs() < 1e-7);
            let down = branch
                .apply(
                    &parent.snapshot,
                    Bounds {
                        lower: f64::NEG_INFINITY,
                        upper: 0.,
                    },
                )
                .unwrap()
                .solve(options.clone())
                .unwrap();
            assert_eq!(down.master.status, Status::Infeasible, "{:#?}", down.master);
            let up = branch
                .apply(
                    &parent.snapshot,
                    Bounds {
                        lower: 1.,
                        upper: f64::INFINITY,
                    },
                )
                .unwrap()
                .solve(options.clone())
                .unwrap();
            assert_eq!(up.master.status, Status::Optimal, "{:#?}", up.master);
            assert!((up.master.lp - 10.).abs() < 1e-7);
            let mut tree = ModelTreeOptions::default();
            tree.branching = baldes::master::tree::NetworkBranchingPolicy::ResourceFirst;
            tree.node = options.clone();
            registry.configure_tree(&mut tree).unwrap();
            let solved = parent.snapshot.solve_tree(tree).unwrap();
            assert_eq!(solved.status, TreeStatus::Optimal, "{solved:?}");
            assert!((solved.incumbent - 20.).abs() < 1e-7);
            assert!(
                solved
                    .mapped_branch_evaluations
                    .iter()
                    .any(|evaluation| evaluation.name == "X_42_901"),
                "{solved:?}"
            );
            let resumed = parent.snapshot.solve(options).unwrap();
            assert!((resumed.master.lp - 5.).abs() < 1e-7);
        }
    }
}

#[test]
fn paired_integer_registration_rejects_continuous_and_fractional_terminal_activities() {
    for fractional in [false, true] {
        let file = if fractional {
            include_str!("../../../fixtures/coupled/metasolver-paired-objective-mapping.json")
        } else {
            include_str!("../../../fixtures/coupled/metasolver-paired-preparation.json")
        };
        let corpus: serde_json::Value = serde_json::from_str(file).unwrap();
        for points in [false, true] {
            let mut source = corpus["cases"][0]["model"].clone();
            source["networks"][0]["front_program_domain"] = false.into();
            source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
            if fractional {
                for arc in source["networks"][0]["coupled_resources"][0]["arcs"]
                    .as_array_mut()
                    .unwrap()
                {
                    arc["increment"]["pairs"][0][0] = 0.5.into();
                }
            }
            if points {
                source["networks"][0] = control_point_source_model(source["networks"][0].clone());
            }
            source["pricing_variables"] = serde_json::json!([{"network":42,"id":901,"cost":0.,"accumulated_resource":0,"integer":true}]);
            let error = model::format::Model::from_json(&source.to_string())
                .unwrap()
                .compile::<2>()
                .unwrap_err()
                .to_string();
            assert!(error.contains("exactly integer"), "{error}");
        }
    }
}

#[test]
fn invariant_paired_rows_and_signed_costs_have_complete_controlled_universes() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-objective-mapping.json"
    ))
    .unwrap();
    for case in [0, 4] {
        for points in [false, true] {
            for mode in [BoundMode::Numerical, BoundMode::Certified] {
                for cost in [-1., 1.] {
                    let mut source = corpus["cases"][case]["model"].clone();
                    if points {
                        source["networks"][0] =
                            control_point_source_model(source["networks"][0].clone());
                    }
                    source["rows"] = serde_json::json!([{"sense":"ge","rhs":1.}]);
                    source["pricing_variables"] = serde_json::json!([{"network":42,"id":901,"cost":cost,"accumulated_resource":0,"entries":[[0,1.]]}]);
                    let (problem, registry) = model::format::Model::from_json(&source.to_string())
                        .unwrap()
                        .compile::<2>()
                        .unwrap();
                    let mut options = Options::default();
                    options.cg.stabilize = false;
                    options.cg.bound_mode = mode;
                    registry.configure_pricing(&mut options);
                    let parent = model::solve(problem, options.clone()).unwrap();
                    let expected = -20. + 2. * cost;
                    let complete = parent
                        .snapshot
                        .enumerate(&model::EnumerationOptions::default())
                        .unwrap();
                    assert!(
                        complete.enumeration_report().complete,
                        "{:?}",
                        complete.enumeration_report()
                    );
                    assert_eq!(complete.enumeration_report().routes, 3);
                    let result = complete.solve(options.clone()).unwrap();
                    assert_eq!(
                        result.master.status,
                        Status::Optimal,
                        "{:#?}",
                        result.master
                    );
                    assert!((result.master.lp - expected).abs() < 1e-7);
                    assert!(
                        result.master.bound <= expected + 1e-10
                            && expected - result.master.bound < 1e-7
                    );
                    assert_eq!(registry.activity(42, 901, &result).unwrap().lower, 2.);
                    assert!(result
                        .routes
                        .iter()
                        .filter(|r| r.route.arcs.len() == 2)
                        .all(|r| !r.route.coupled_controls.is_empty()));
                    let index = result
                        .routes
                        .iter()
                        .position(|r| r.route.arcs.len() == 2)
                        .unwrap();
                    let fixed = result
                        .snapshot
                        .fix_route(index, 1.)
                        .unwrap()
                        .solve(options.clone())
                        .unwrap();
                    assert_eq!(fixed.master.status, Status::Optimal, "{:#?}", fixed.master);
                    assert!((fixed.master.lp - expected).abs() < 1e-7);
                    #[cfg(feature = "highs")]
                    {
                        use baldes::master::solvers::mip::{MipOptions, MipStatus};
                        let mip = result
                            .snapshot
                            .solve_enumerated_mip(&MipOptions::default())
                            .unwrap()
                            .unwrap();
                        assert_eq!(mip.mip.status, MipStatus::Optimal, "{:?}", mip.mip);
                        assert!((mip.mip.solution.unwrap().objective - expected).abs() < 1e-7);
                        assert!(mip.route_values.iter().all(|fixed| !fixed
                            .route
                            .route
                            .coupled_controls
                            .is_empty()));
                    }
                    let repeated = result
                        .snapshot
                        .enumerate(&model::EnumerationOptions::default())
                        .unwrap();
                    assert_eq!(repeated.enumeration_report().attempts, 1);
                    assert!(!parent.snapshot.enumeration_report().complete);
                }
            }
        }
    }
}

#[test]
fn relative_and_absolute_negative_networks_share_rows_and_every_pricing_schedule() {
    use baldes::graph::coupled::{CoupledResource, PairedCostNormalization};
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    source["networks"][0]["front_program_domain"] = false.into();
    let definition = &mut source["networks"][0]["coupled_resources"][0];
    definition["policy"] = "strict_domain".into();
    definition["normalization"] = "final_minus_initial_minimum".into();
    definition["initial"]["pairs"][0][0] = 2.into();
    definition["initial"]["pairs"][1][0] = 1.into();
    let mut problem = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .problem::<2>()
        .unwrap();
    problem.networks[0]
        .graph
        .set_coupled_objective_coefficients(&[(0, -1.)])
        .unwrap();
    let mut other = problem.networks[0].clone();
    other.id = 7;
    let mut definitions = other.graph.coupled_kernel().unwrap().definitions().to_vec();
    match &mut definitions[0] {
        CoupledResource::SlopePaired { normalization, .. } => {
            *normalization = PairedCostNormalization::AbsoluteFinalValue
        }
        _ => unreachable!(),
    }
    other.graph.set_coupled_resources(&definitions).unwrap();
    other
        .graph
        .set_coupled_objective_coefficients(&[(0, -2.)])
        .unwrap();
    problem.networks.push(other);
    problem.rows = vec![Row {
        sense: RowSense::LessEqual,
        rhs: 1.5,
    }];
    for network in &mut problem.networks {
        network.route_entries = vec![(0, 1.)];
    }
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for schedule in [
            model::NetworkPricingSchedule::FullSweep,
            model::NetworkPricingSchedule::Cyclic,
            model::NetworkPricingSchedule::Sorted,
            model::NetworkPricingSchedule::PromisingFirst,
            model::NetworkPricingSchedule::UnvisitedFirst,
        ] {
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = mode;
            options.network_pricing_schedule = schedule;
            let output = model::solve(problem.clone(), options.clone()).unwrap();
            assert_eq!(
                output.master.status,
                Status::Optimal,
                "{schedule:?} {:#?}",
                output.master
            );
            assert!((output.master.lp + 49.).abs() < 1e-7);
            assert!(output.master.bound <= -49. + 1e-10 && -49. - output.master.bound < 1e-7);
            let mut tree_options = baldes::master::tree::ModelTreeOptions::default();
            tree_options.node = options;
            let tree = output.snapshot.solve_tree(tree_options).unwrap();
            assert_eq!(
                tree.status,
                baldes::master::tree::TreeStatus::Optimal,
                "{tree:?}"
            );
            assert!((tree.incumbent + 36.).abs() < 1e-7);
        }
    }
}

#[test]
fn explicit_graph_costs_survive_variable_compilation_and_reject_duplicate_ownership() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    source["networks"][0]["front_program_domain"] = false.into();
    source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
    let paired = source["networks"][0]["coupled_resources"][0].clone();
    source["networks"][0]["coupled_resources"]
        .as_array_mut()
        .unwrap()
        .push(paired);
    source["networks"][0]["coupled_objective_coefficients"] = serde_json::json!([[1, -1.]]);
    source["pricing_variables"] = serde_json::json!([
        {"network":42,"id":901,"cost":1.,"accumulated_resource":0}
    ]);
    let (problem, registry) = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    assert_eq!(
        problem.networks[0]
            .graph
            .coupled_kernel()
            .unwrap()
            .objective_coefficient(1),
        Some(-1.)
    );
    let mut options = Options::default();
    options.cg.stabilize = false;
    registry.configure_pricing(&mut options);
    let root = model::solve(problem, options).unwrap();
    assert_eq!(root.master.status, Status::Optimal, "{:#?}", root.master);
    assert!((root.master.lp + 25.).abs() < 1e-7);
    assert!(root.master.bound <= -25. + 1e-10 && -25. - root.master.bound < 1e-7);
    for cost in [1., -1.] {
        source["networks"][0]["coupled_objective_coefficients"] = serde_json::json!([[0, cost]]);
        let error = model::format::Model::from_json(&source.to_string())
            .unwrap()
            .compile::<2>()
            .err()
            .unwrap();
        assert!(error.to_string().contains("not both"), "{error}");
    }
}

#[test]
fn source_control_cycle_ignores_dead_negative_branch_and_retains_bounded_optimum() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-objective-mapping.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    source["pricing_variables"] = serde_json::json!([]);
    let network = &mut source["networks"][0];
    network["vertices"] = 4.into();
    network["sink"] = 3.into();
    network["elementary"] = false.into();
    network["route_domain"] = serde_json::json!({"kind":"bounded_walks","max_arcs":4});
    network["arcs"] = serde_json::json!([
        {"tail":0,"head":1,"cost":-20.},
        {"tail":1,"head":1,"cost":0.},
        {"tail":1,"head":3,"cost":0.},
        {"tail":1,"head":2,"cost":-100.},
        {"tail":2,"head":2,"cost":-100.}
    ]);
    let arc = network["coupled_resources"][0]["arcs"][0].clone();
    network["coupled_resources"][0]["arcs"] =
        serde_json::json!([arc.clone(), arc.clone(), arc.clone(), arc.clone(), arc]);
    network["coupled_objective_coefficients"] = serde_json::json!([[0, -1.]]);
    let problem = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .problem::<2>()
        .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.bound_mode = BoundMode::Certified;
    let root = model::solve(problem, options).unwrap();
    assert_eq!(root.master.status, Status::Optimal, "{:#?}", root.master);
    assert!((root.master.lp + 24.).abs() < 1e-7);
    assert!(root.master.bound <= -24. + 1e-10 && -24. - root.master.bound < 1e-7);
    assert!(root
        .routes
        .iter()
        .any(|route| route.route.arcs.len() == 4 && route.route.coupled_values == [(0, 4.)]));
}
