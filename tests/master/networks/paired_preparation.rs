//! Actual registered modern resource factory outputs and independent path oracles.
use super::*;

#[test]
fn registered_fictive_selection_factories_match_reference_and_path_oracle() {
    assert_eq!(
        check_registered_cases(include_str!(
            "../../fixtures/coupled/metasolver-fictive-preparation.json"
        )),
        288
    );
}

#[test]
fn registered_pickup_delivery_factory_models_match_reference_and_load_oracle() {
    assert_eq!(
        check_registered_cases(include_str!(
            "../../fixtures/coupled/metasolver-spd-preparation.json"
        )),
        180
    );
}

fn check_registered_cases(source: &str) -> usize {
    let corpus: serde_json::Value = serde_json::from_str(source).unwrap();
    assert_eq!(
        corpus["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let mut compared = 0;
    for case in corpus["cases"].as_array().unwrap() {
        if let Some(waiting) = case["registration"]["functions"]["waiting"].as_f64() {
            for arc in case["model"]["networks"][0]["coupled_resources"][0]["arcs"]
                .as_array()
                .unwrap()
            {
                assert_eq!(arc["waiting"].as_i64().unwrap(), waiting.trunc() as i64);
            }
        }
        let expected = case["independent_objective"].as_f64().unwrap();
        if case["reference"]["outcome"] == "reference_reconstruction_defect" {
            assert_eq!(case["prepared"]["mode"], 4);
            assert_eq!(case["reference"]["exit_code"], 1);
            let problem = model::format::Model::from_json(&case["model"].to_string())
                .unwrap()
                .problem::<2>()
                .unwrap();
            let proof = problem.networks[0]
                .graph
                .coupled_kernel()
                .unwrap()
                .certify_front_program(&[1, 2], &mut 1_000_000, None)
                .unwrap_or_else(|| panic!("unknown certificate {}", case["name"]));
            assert!(proof.feasible);
            assert_eq!(proof.objective, [0., 0.]);
        } else {
            assert_eq!(
                expected,
                case["reference"]["bcRecRootLpVal"].as_f64().unwrap()
            );
        }
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for direction in [0., 1., 2.] {
                for singular in [false, true] {
                    let mut source = case["model"].clone();
                    if singular {
                        source["networks"] = source["networks"][0].clone();
                    }
                    let problem = model::format::Model::from_json(&source.to_string())
                        .unwrap()
                        .problem::<2>()
                        .unwrap();
                    let mut options = Options::default();
                    options.cg.stabilize = false;
                    options.cg.bound_mode = mode;
                    options.pricing_options =
                        vec![("BALDESuseBidirectionalSearch".into(), direction)];
                    let output = model::solve(problem, options).unwrap();
                    assert_eq!(
                        output.master.status,
                        Status::Optimal,
                        "{} mode={mode:?} direction={direction} {:#?}",
                        case["name"],
                        output.master
                    );
                    assert_eq!(output.master.lp, expected, "{}", case["name"]);
                    assert!(
                        output.master.bound <= expected && expected - output.master.bound < 1e-7
                    );
                    compared += 1;
                }
            }
        }
    }
    compared
}

#[test]
fn registered_paired_factory_models_match_reference_and_independent_optima() {
    assert_eq!(
        check_registered_cases(include_str!(
            "../../fixtures/coupled/metasolver-paired-preparation.json"
        )),
        504
    );
}

#[test]
fn nonlinear_paired_factories_and_waiting_conversion_match_actual_reference() {
    assert_eq!(
        check_registered_cases(include_str!(
            "../../fixtures/coupled/metasolver-paired-pwl-preparation.json"
        )),
        432
    );
}

#[test]
fn feasibility_factories_preserve_constraints_without_charging_dependent_values() {
    assert_eq!(
        check_registered_cases(include_str!(
            "../../fixtures/coupled/metasolver-paired-feasibility-preparation.json"
        )),
        96
    );
}

#[test]
fn registered_paired_factories_share_multiple_network_pricing_and_tree_bounds() {
    use baldes::master::{
        model::NetworkPricingSchedule,
        tree::{ModelTreeOptions, TreeStatus},
    };
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let cases = corpus["cases"].as_array().unwrap();
    let mut roots = 0;
    let mut trees = 0;
    for i in 0..6 {
        let mut source = cases[i]["model"].clone();
        let mut second = cases[i + 18]["model"]["networks"][0].clone();
        second["id"] = 7.into();
        source["networks"].as_array_mut().unwrap().push(second);
        let expected = cases[i]["independent_objective"].as_f64().unwrap()
            + cases[i + 18]["independent_objective"].as_f64().unwrap();
        let problem = model::format::Model::from_json(&source.to_string())
            .unwrap()
            .problem::<2>()
            .unwrap();
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for schedule in [
                NetworkPricingSchedule::FullSweep,
                NetworkPricingSchedule::Cyclic,
                NetworkPricingSchedule::Sorted,
                NetworkPricingSchedule::PromisingFirst,
                NetworkPricingSchedule::UnvisitedFirst,
            ] {
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = mode;
                options.network_pricing_schedule = schedule;
                optimal(&model::solve(problem.clone(), options).unwrap(), expected);
                roots += 1;
            }
            let mut options = ModelTreeOptions::default();
            options.node.cg.stabilize = false;
            options.node.cg.bound_mode = mode;
            let output = baldes::master::tree::solve(problem.clone(), options).unwrap();
            assert_eq!(output.status, TreeStatus::Optimal);
            assert_eq!(output.incumbent, expected);
            assert!(output.bound <= expected && expected - output.bound < 1e-7);
            trees += 1;
        }
    }
    assert_eq!((roots, trees), (60, 12));
}

#[test]
fn feasibility_and_cost_factories_share_network_schedules_and_tree_orchestration() {
    use baldes::master::{
        model::NetworkPricingSchedule,
        tree::{ModelTreeOptions, TreeStatus},
    };
    let feasibility: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-feasibility-preparation.json"
    ))
    .unwrap();
    let cost: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = feasibility["cases"][6]["model"].clone();
    let mut second = cost["cases"][0]["model"]["networks"][0].clone();
    second["id"] = 7.into();
    source["networks"].as_array_mut().unwrap().push(second);
    let problem = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .problem::<2>()
        .unwrap();
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for schedule in [
            NetworkPricingSchedule::FullSweep,
            NetworkPricingSchedule::Cyclic,
            NetworkPricingSchedule::Sorted,
            NetworkPricingSchedule::PromisingFirst,
            NetworkPricingSchedule::UnvisitedFirst,
        ] {
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = mode;
            options.network_pricing_schedule = schedule;
            optimal(&model::solve(problem.clone(), options).unwrap(), -37.);
        }
        let mut options = ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.node.cg.bound_mode = mode;
        let output = baldes::master::tree::solve(problem.clone(), options).unwrap();
        assert_eq!(output.status, TreeStatus::Optimal);
        assert_eq!(output.incumbent, -37.);
        assert_eq!(output.bound, -37.);
    }
}

#[test]
fn selected_paired_values_survive_pricing_phase_changes_and_snapshot_reuse() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for direction in [0., 1., 2.] {
        let problem = model::format::Model::from_json(&corpus["cases"][0]["model"].to_string())
            .unwrap()
            .problem::<2>()
            .unwrap();
        let options = || {
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.pricing_options = vec![("BALDESuseBidirectionalSearch".into(), direction)];
            options
        };
        let output = model::solve(problem, options()).unwrap();
        let resumed = output.snapshot.solve(options()).unwrap();
        for solved in [&output, &resumed] {
            assert_eq!(solved.master.lp, -17.);
            assert!(!solved.routes.is_empty());
            for route in &solved.routes {
                let expected = if route.route.arcs.len() == 1 { 1. } else { 3. };
                assert_eq!(route.route.coupled_values, vec![(0, expected)]);
            }
        }
    }
}

#[test]
fn paired_original_variables_use_selected_columns_and_absolute_nonnegative_objectives() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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

fn control_point_source_model(mut source: serde_json::Value) -> serde_json::Value {
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
    for resource in source["coupled_resources"].as_array_mut().unwrap() {
        resource["initial"] = knots(&resource["initial"]);
        for arc in resource["arcs"].as_array_mut().unwrap() {
            for key in ["arrival", "increment", "lower", "upper"] {
                arc[key] = knots(&arc[key]);
            }
        }
        resource["kind"] = "paired_piecewise_linear".into();
        resource.as_object_mut().unwrap().remove("policy");
    }
    source
}

#[test]
fn controlled_routes_keep_identity_costs_and_original_values_through_pool_resume_and_fixing() {
    use baldes::graph::coupled::PairedControl;
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
fn paired_mixed_reference_row_uses_a_proved_control_instead_of_the_invalid_reference_bound() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for case in [0, 18] {
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            let mut source = corpus["cases"][case]["model"].clone();
            let network = &mut source["networks"][0];
            network["front_program_domain"] = false.into();
            network["coupled_resources"][0]["policy"] = "strict_domain".into();
            // Reference network: first two-arc path is positions 1/2;
            // second path is 3/4. Only the first receives the additive term.
            network["arcs"][1]["cost"] = 0.into();
            network["arcs"][2]["cost"] = 0.into();
            network["arcs"][1]["entries"] = serde_json::json!([[0, 10.]]);
            source["rows"] = serde_json::json!([{"sense":"ge","rhs":4.}]);
            source["pricing_variables"] = serde_json::json!([{
                "network":42,"id":901,"cost":1.,"accumulated_resource":0,"entries":[[0,1.]]
            }]);
            let (problem, registry) = model::format::Model::from_json(&source.to_string())
                .unwrap()
                .compile::<2>()
                .unwrap();
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = mode;
            registry.configure_pricing(&mut options);
            let output = model::solve(problem, options).unwrap();
            assert_eq!(
                output.master.status,
                Status::Optimal,
                "{:#?}",
                output.master
            );
            assert!(
                (output.master.lp + 16.).abs() < 1e-7,
                "{:#?}",
                output.master
            );
            assert!(
                output.master.bound <= -16. + 1e-10 && -16. - output.master.bound < 1e-7,
                "{:#?}",
                output.master
            );
            assert!(output
                .routes
                .iter()
                .any(|r| r.route.coupled_values == vec![(0, 4.)]));
        }
    }
}

#[test]
fn paired_rows_share_network_schedules_and_keep_continuous_domains_out_of_enumeration() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
            let mut source = corpus["cases"][0]["model"].clone();
            source["networks"][0]["front_program_domain"] = false.into();
            source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
            let mut second = source["networks"][0].clone();
            second["id"] = 7.into();
            source["networks"].as_array_mut().unwrap().push(second);
            source["rows"] = serde_json::json!([{"sense":"ge","rhs":8.}]);
            source["pricing_variables"] = serde_json::json!([
                {"network":42,"id":901,"cost":1.,"accumulated_resource":0,"entries":[[0,1.]]},
                {"network":7,"id":901,"cost":2.,"accumulated_resource":0,"entries":[[0,1.]]}
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
            let output = model::solve(problem, options.clone()).unwrap();
            assert_eq!(
                output.master.status,
                Status::Optimal,
                "{:#?}",
                output.master
            );
            assert!(
                (output.master.lp + 29.).abs() < 1e-7,
                "{:#?}",
                output.master
            );
            assert!(output.master.bound <= -29. + 1e-10 && -29. - output.master.bound < 1e-7);
            let a = registry.activity(42, 901, &output).unwrap();
            let b = registry.activity(7, 901, &output).unwrap();
            assert!(
                (a.lower - 5.).abs() < 1e-7 && (b.lower - 3.).abs() < 1e-7,
                "{a:?} {b:?}"
            );
            let enumerated = output
                .snapshot
                .enumerate(&model::EnumerationOptions::default())
                .unwrap();
            assert!(!enumerated.enumeration_report().complete);
        }
    }
}

#[test]
fn paired_rows_refresh_existing_columns_but_reject_previously_completed_universes() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    source["networks"][0]["front_program_domain"] = false.into();
    source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
    source["pricing_variables"] =
        serde_json::json!([{"network":42,"id":901,"cost":1.,"accumulated_resource":0}]);
    let (problem, registry) = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.bound_mode = BoundMode::Certified;
    registry.configure_pricing(&mut options);
    let root = model::solve(problem, options.clone()).unwrap();
    let coefficients = [model::AccumulatedActivityCoefficients {
        network: 42,
        resource: 0,
        coefficient: 1.,
    }];
    let row = Row {
        sense: RowSense::GreaterEqual,
        rhs: 4.,
    };
    let child = root
        .snapshot
        .with_accumulated_activity_row(row.clone(), &coefficients, &[])
        .unwrap()
        .solve(options)
        .unwrap();
    assert_eq!(child.master.status, Status::Optimal);
    assert!((child.master.lp + 16.).abs() < 1e-7);
    for (candidate, route) in child
        .routes
        .iter()
        .enumerate()
        .filter(|(_, r)| r.route.coupled_values == vec![(0, 4.)])
    {
        let fixed = child
            .snapshot
            .fix_route(candidate, 1.)
            .unwrap()
            .solve(Options::default())
            .unwrap();
        assert_eq!(fixed.master.status, Status::Optimal, "{:#?}", fixed.master);
        assert!(
            (fixed.master.lp - route.route.original_cost).abs() < 1e-7,
            "{:#?}",
            fixed.master
        );
        assert!((fixed.master.bound - fixed.master.lp).abs() < 1e-7);
    }
    let enumerated = root
        .snapshot
        .enumerate(&model::EnumerationOptions::default())
        .unwrap();
    assert!(enumerated.enumeration_report().complete);
    assert!(enumerated
        .with_accumulated_activity_row(row, &coefficients, &[])
        .is_err());
}

#[test]
fn paired_shared_rows_keep_multiple_source_symbols_and_repeated_signed_coefficients() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for knot_first in [false, true] {
        let mut source = corpus["cases"][0]["model"].clone();
        source["networks"][0]["front_program_domain"] = false.into();
        source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
        let knot = control_point_source_model(source["networks"][0].clone())["coupled_resources"]
            [0]
        .clone();
        source["networks"][0]["coupled_resources"]
            .as_array_mut()
            .unwrap()
            .push(knot);
        if knot_first {
            source["networks"][0]["coupled_resources"]
                .as_array_mut()
                .unwrap()
                .swap(0, 1);
        }
        source["rows"] = serde_json::json!([{"sense":"ge","rhs":7.}]);
        // Deliberately split one coefficient with opposite-signed entries.
        source["pricing_variables"] = serde_json::json!([
            {"network":42,"id":901,"cost":1.,"accumulated_resource":0,"entries":[[0,2.],[0,-1.]]},
            {"network":42,"id":902,"cost":2.,"accumulated_resource":1,"entries":[[0,1.]]}
        ]);
        let (problem, registry) = model::format::Model::from_json(&source.to_string())
            .unwrap()
            .compile::<2>()
            .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.bound_mode = BoundMode::Certified;
        registry.configure_pricing(&mut options);
        let output = model::solve(problem, options).unwrap();
        assert_eq!(
            output.master.status,
            Status::Optimal,
            "{:#?}",
            output.master
        );
        assert!(
            (output.master.lp + 10.).abs() < 1e-7,
            "{:#?}",
            output.master
        );
        assert!(output.master.bound <= -10. + 1e-10 && -10. - output.master.bound < 1e-7);
        let first = registry.activity(42, 901, &output).unwrap();
        let second = registry.activity(42, 902, &output).unwrap();
        assert!(
            (first.lower - 4.).abs() < 1e-7 && (second.lower - 3.).abs() < 1e-7,
            "{first:?} {second:?}"
        );
    }
}

#[test]
fn negative_paired_original_objectives_select_maximum_controls_and_valid_bounds() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for case in [0, 18] {
        for points in [false, true] {
            for mode in [BoundMode::Numerical, BoundMode::Certified] {
                for (cost, upper, gamma) in [(-1., None, 8.), (-2., None, 8.), (-0.5, Some(4.), 4.)]
                {
                    let mut source = corpus["cases"][case]["model"].clone();
                    source["networks"][0]["front_program_domain"] = false.into();
                    source["networks"][0]["coupled_resources"][0]["policy"] =
                        "strict_domain".into();
                    if points {
                        source["networks"][0] =
                            control_point_source_model(source["networks"][0].clone());
                    }
                    let variable = if let Some(rhs) = upper {
                        source["rows"] = serde_json::json!([{"sense":"le","rhs":rhs}]);
                        serde_json::json!({"network":42,"id":901,"cost":cost,"accumulated_resource":0,"entries":[[0,1.]]})
                    } else {
                        serde_json::json!({"network":42,"id":901,"cost":cost,"accumulated_resource":0})
                    };
                    source["pricing_variables"] = serde_json::json!([variable]);
                    let (problem, registry) = model::format::Model::from_json(&source.to_string())
                        .unwrap()
                        .compile::<2>()
                        .unwrap();
                    let mut options = Options::default();
                    options.cg.stabilize = false;
                    options.cg.bound_mode = mode;
                    registry.configure_pricing(&mut options);
                    let output = model::solve(problem, options.clone()).unwrap();
                    let expected = -20. + cost * gamma;
                    assert_eq!(
                        output.master.status,
                        Status::Optimal,
                        "case={case} points={points} cost={cost} {:#?}",
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
                    let activity = registry.activity(42, 901, &output).unwrap();
                    assert!(
                        (activity.lower - gamma).abs() < 1e-7
                            && (activity.upper - gamma).abs() < 1e-7,
                        "{activity:?}"
                    );
                    let resumed = output.snapshot.solve(options).unwrap();
                    assert!((resumed.master.lp - expected).abs() < 1e-7);
                    if upper.is_none() {
                        let mut tree_options = baldes::master::tree::ModelTreeOptions::default();
                        tree_options.node.cg.stabilize = false;
                        registry.configure_tree(&mut tree_options).unwrap();
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
}

#[test]
fn negative_paired_objectives_share_scheduling_and_rows_with_positive_networks() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/metasolver-paired-objective-mapping.json"
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
fn paired_variable_boxes_cover_all_controls_including_negative_objectives() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/metasolver-paired-objective-mapping.json"
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
            include_str!("../../fixtures/coupled/metasolver-paired-objective-mapping.json")
        } else {
            include_str!("../../fixtures/coupled/metasolver-paired-preparation.json")
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
        "../../fixtures/coupled/metasolver-paired-objective-mapping.json"
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
fn fractional_invariant_controls_enumerate_from_priced_and_cold_snapshots() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-objective-mapping.json"
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
        "../../fixtures/coupled/metasolver-paired-objective-mapping.json"
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
        "../../fixtures/coupled/metasolver-paired-objective-mapping.json"
    ))
    .unwrap();
    let variable: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/metasolver-paired-objective-mapping.json"
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
        "../../fixtures/coupled/metasolver-paired-objective-mapping.json"
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
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
fn relative_and_absolute_negative_networks_share_rows_and_every_pricing_schedule() {
    use baldes::graph::coupled::{CoupledResource, PairedCostNormalization};
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
fn source_controls_cover_repeated_vertices_up_to_the_declared_walk_cap() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-objective-mapping.json"
    ))
    .unwrap();
    for points in [false, true] {
        for cap in [2, 3, 4] {
            let mut source = corpus["cases"][0]["model"].clone();
            source["pricing_variables"] = serde_json::json!([]);
            let network = &mut source["networks"][0];
            network["vertices"] = 3.into();
            network["sink"] = 2.into();
            network["elementary"] = false.into();
            network["route_domain"] = serde_json::json!({"kind":"bounded_walks","max_arcs":cap});
            network["arcs"] = serde_json::json!([
                {"tail":0,"head":1,"cost":-20.},
                {"tail":1,"head":1,"cost":0.},
                {"tail":1,"head":2,"cost":0.}
            ]);
            let arc = network["coupled_resources"][0]["arcs"][0].clone();
            network["coupled_resources"][0]["arcs"] =
                serde_json::json!([arc.clone(), arc.clone(), arc]);
            network["coupled_objective_coefficients"] = serde_json::json!([[0, -1.]]);
            if points {
                *network = control_point_source_model(network.clone());
            }
            let problem = model::format::Model::from_json(&source.to_string())
                .unwrap()
                .problem::<2>()
                .unwrap();
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = BoundMode::Certified;
            let root = model::solve(problem, options.clone()).unwrap();
            let expected = -20. - cap as f64;
            assert_eq!(
                root.master.status,
                Status::Optimal,
                "{points} {cap}: {:#?}",
                root.master
            );
            assert!((root.master.lp - expected).abs() < 1e-7);
            assert!(root.master.bound <= expected + 1e-10 && expected - root.master.bound < 1e-7);
            assert!(root
                .routes
                .iter()
                .any(|r| r.route.arcs.len() == cap && r.route.coupled_values == [(0, cap as f64)]));
            let complete = root
                .snapshot
                .enumerate(&model::EnumerationOptions::default())
                .unwrap();
            assert!(
                complete.enumeration_report().complete,
                "{:?}",
                complete.enumeration_report()
            );
            assert_eq!(complete.enumeration_report().routes, cap - 1);
            #[cfg(feature = "highs")]
            {
                use baldes::master::solvers::mip::{MipOptions, MipStatus};
                let mip = complete
                    .solve_enumerated_mip(&MipOptions::default())
                    .unwrap()
                    .unwrap();
                assert_eq!(mip.mip.status, MipStatus::Optimal, "{:?}", mip.mip);
                assert!((mip.mip.solution.unwrap().objective - expected).abs() < 1e-7);
                let cutoff = root
                    .snapshot
                    .enumerate_below(expected + 0.5, &model::EnumerationOptions::default())
                    .unwrap();
                let mip = cutoff.solve_mip(&MipOptions::default()).unwrap();
                assert_eq!(mip.mip.status, MipStatus::Optimal, "{:?}", mip.mip);
                assert!((mip.mip.solution.unwrap().objective - expected).abs() < 1e-7);
            }
            let resumed = complete.solve(options).unwrap();
            assert!((resumed.master.lp - expected).abs() < 1e-7);
        }
    }
}

#[test]
fn objective_only_varying_controls_enumerate_attained_dominating_endpoints() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for points in [false, true] {
        for relative in [false, true] {
            for bind_variable in [false, true] {
                if relative && bind_variable {
                    continue;
                }
                let mut source = corpus["cases"][0]["model"].clone();
                source["networks"][0]["front_program_domain"] = false.into();
                source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
                if relative {
                    source["networks"][0]["coupled_resources"][0]["normalization"] =
                        "final_minus_initial_minimum".into();
                    source["networks"][0]["coupled_resources"][0]["initial"]["pairs"][0][0] =
                        2.into();
                    source["networks"][0]["coupled_resources"][0]["initial"]["pairs"][1][0] =
                        1.into();
                }
                if points {
                    source["networks"][0] =
                        control_point_source_model(source["networks"][0].clone());
                }
                if bind_variable {
                    source["pricing_variables"] = serde_json::json!([
                        {"network":42,"id":901,"cost":-1.,"accumulated_resource":0}
                    ]);
                } else {
                    source["networks"][0]["coupled_objective_coefficients"] =
                        serde_json::json!([[0, -1.]]);
                }
                let (problem, registry) = model::format::Model::from_json(&source.to_string())
                    .unwrap()
                    .compile::<2>()
                    .unwrap();
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = BoundMode::Certified;
                registry.configure_pricing(&mut options);
                let root = model::solve(problem, options.clone()).unwrap();
                let expected = if relative { -26. } else { -28. };
                let complete = root
                    .snapshot
                    .enumerate(&model::EnumerationOptions::default())
                    .unwrap();
                assert!(
                    complete.enumeration_report().complete,
                    "{:?}",
                    complete.enumeration_report()
                );
                assert_eq!(complete.enumeration_report().routes, 3);
                let solved = complete.solve(options.clone()).unwrap();
                assert_eq!(solved.master.status, Status::Optimal);
                assert!((solved.master.lp - expected).abs() < 1e-7);
                let index = solved.values.iter().position(|&x| x > 0.5).unwrap();
                assert_eq!(solved.routes[index].route.coupled_values, [(0, 8.)]);
                assert!(
                    (solved
                        .snapshot
                        .fix_route(index, 1.)
                        .unwrap()
                        .solve(options.clone())
                        .unwrap()
                        .master
                        .lp
                        - expected)
                        .abs()
                        < 1e-7
                );
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
                    let mip = cutoff.solve_mip(&MipOptions::default()).unwrap();
                    assert_eq!(mip.mip.status, MipStatus::Optimal, "{:?}", mip.mip);
                }
                if !relative {
                    let row = Row {
                        sense: RowSense::LessEqual,
                        rhs: 4.,
                    };
                    let terms = [model::AccumulatedActivityCoefficients {
                        network: 42,
                        resource: 0,
                        coefficient: 1.,
                    }];
                    assert!(solved
                        .snapshot
                        .with_accumulated_activity_row(row.clone(), &terms, &[])
                        .is_err());
                    let child = root
                        .snapshot
                        .with_accumulated_activity_row(row, &terms, &[])
                        .unwrap()
                        .solve(options)
                        .unwrap();
                    assert!((child.master.lp + 24.).abs() < 1e-7);
                    assert!(
                        !child
                            .snapshot
                            .enumerate(&model::EnumerationOptions::default())
                            .unwrap()
                            .enumeration_report()
                            .complete
                    );
                }
            }
        }
    }
}

#[test]
fn monotone_paired_rows_enumerate_only_proved_dominating_endpoints() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../fixtures/coupled/joint-interior-model.json"
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
                        assert_eq!(solved.routes.len(), if shape == 2 { 2 } else { 1 });
                        let primary = solved
                            .routes
                            .iter()
                            .find(|r| r.route.network == 42)
                            .unwrap();
                        assert_eq!(primary.quantity, quantity);
                        assert_eq!(
                            primary.route.route.coupled_values,
                            vec![(0, 4. / quantity), (1, 4. / quantity)]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn repaired_paired_witnesses_return_the_joint_schedule_across_snapshots() {
    use baldes::master::{
        model::{SynchronizedService, SynchronizedServiceMember},
        tree::{ModelTreeOptions, TreeStatus},
    };
    let base: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/joint-synchronized-model.json"
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
            for mode in [BoundMode::Numerical, BoundMode::Certified] {
                let (problem, registry) = model::format::Model::from_json(&source.to_string())
                    .unwrap()
                    .compile::<2>()
                    .unwrap();
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = mode;
                options.synchronized_services = vec![SynchronizedService {
                    id: 17,
                    resource: 0,
                    vehicles_required: quantity as u32 + 1,
                    reward: 1.,
                    start: [0, 10],
                    members: vec![
                        SynchronizedServiceMember {
                            network: 42,
                            vertex: 1,
                        },
                        SynchronizedServiceMember {
                            network: 7,
                            vertex: 1,
                        },
                    ],
                }];
                registry.configure_pricing(&mut options);
                let root = model::solve(problem, options.clone()).unwrap();
                for start in [None, Some((8, 9))] {
                    let snapshot = match start {
                        Some((lo, hi)) => {
                            root.snapshot.with_synchronized_start(17, lo, hi).unwrap()
                        }
                        None => root.snapshot.with_synchronized_start(17, 0, 10).unwrap(),
                    };
                    let mut tree = ModelTreeOptions::default();
                    tree.node = options.clone();
                    registry.configure_tree(&mut tree).unwrap();
                    let result = snapshot.solve_tree(tree).unwrap();
                    assert_eq!(
                        result.status,
                        TreeStatus::Optimal,
                        "{representation} {quantity} {mode:?}"
                    );
                    assert_eq!(result.incumbent, -10. * quantity + 1.);
                    for fixed in &result.routes {
                        assert_eq!(
                            fixed.route.route.resource_profile[1][0],
                            if start.is_some() { 8. } else { 5. }
                        );
                    }
                    let primary = result
                        .routes
                        .iter()
                        .find(|r| r.route.network == 42)
                        .unwrap();
                    assert_eq!(
                        primary.route.route.coupled_values,
                        vec![(0, 4. / quantity), (1, 4. / quantity)]
                    );
                    assert_eq!(primary.quantity, quantity);
                }
                // The narrowed child must not rewrite the parent's cached profiles.
                assert_eq!(
                    root.snapshot.solve(options).unwrap().master.status,
                    Status::Optimal
                );
            }
        }
    }
}

#[test]
fn cross_network_and_cross_path_rows_recover_joint_source_controls() {
    use baldes::master::tree::{ModelTreeOptions, TreeStatus};
    for (cross_network, fixture) in [
        (
            true,
            include_str!("../../fixtures/coupled/cross-network-interior-model.json"),
        ),
        (
            false,
            include_str!("../../fixtures/coupled/cross-path-interior-model.json"),
        ),
    ] {
        let base: serde_json::Value = serde_json::from_str(fixture).unwrap();
        for representation in 0..4 {
            let mut source = base.clone();
            if cross_network {
                for i in 0..2 {
                    if representation & (1 << i) != 0 {
                        source["networks"][i] =
                            control_point_source_model(source["networks"][i].clone());
                    }
                }
            } else {
                let points = control_point_source_model(source["networks"][0].clone());
                for i in 0..2 {
                    if representation & (1 << i) != 0 {
                        source["networks"][0]["coupled_resources"][i] =
                            points["coupled_resources"][i].clone();
                    }
                }
            }
            for counts in [[1., 1.], [2., 2.], [2., 1.], [3., 1.], [3., 3.]] {
                if cross_network {
                    for i in 0..2 {
                        source["networks"][i]["multiplicity"] =
                            serde_json::json!([counts[i], counts[i]]);
                    }
                } else {
                    source["networks"][0]["multiplicity"] =
                        serde_json::json!([counts[0] + counts[1], counts[0] + counts[1]]);
                    for i in 0..2 {
                        source["rows"][i]["rhs"] = counts[i].into();
                    }
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
                            model::format::Model::from_json(&source.to_string())
                                .unwrap()
                                .compile::<2>()
                                .unwrap();
                        let mut options = ModelTreeOptions::default();
                        options.node.cg.stabilize = false;
                        options.node.cg.bound_mode = mode;
                        options.node.network_pricing_schedule = schedule;
                        registry.configure_tree(&mut options).unwrap();
                        let tree = baldes::master::tree::solve(problem, options).unwrap();
                        let expected = -10. * counts[0] + 2. * counts[1];
                        assert_eq!(
                            tree.status,
                            TreeStatus::Optimal,
                            "{cross_network} {representation} {counts:?} {mode:?} {schedule:?}"
                        );
                        assert_eq!(tree.incumbent, expected);
                        assert!(tree.bound <= expected && tree.bound > expected - 1e-7);
                        let mut quantities = [0., 0.];
                        let mut gamma = [0., 0.];
                        for fixed in &tree.routes {
                            let second = if cross_network {
                                fixed.route.network == 7
                            } else {
                                fixed.route.route.arcs == vec![1]
                            };
                            let i = usize::from(second);
                            assert!(fixed.quantity > 0. && fixed.quantity.fract() == 0.);
                            quantities[i] += fixed.quantity;
                            let resource = if cross_network { 0 } else { i };
                            gamma[i] += fixed.quantity
                                * fixed
                                    .route
                                    .route
                                    .coupled_values
                                    .iter()
                                    .find(|(r, _)| *r == resource)
                                    .unwrap()
                                    .1;
                            if !cross_network {
                                assert_eq!(
                                    fixed
                                        .route
                                        .route
                                        .coupled_values
                                        .iter()
                                        .find(|(r, _)| *r == 1 - i)
                                        .unwrap()
                                        .1,
                                    0.
                                );
                            }
                        }
                        assert_eq!(quantities, counts);
                        assert_eq!(gamma, [4., 4.]);
                    }
                }
            }
        }
    }
}

#[test]
fn narrow_dyadic_domains_recover_integral_copy_blocks() {
    use baldes::master::{
        model,
        tree::{ModelTreeOptions, TreeStatus},
    };
    let positive: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/dyadic-copy-model.json"
    ))
    .unwrap();
    for sign in [1., -1.] {
        let mut source = positive.clone();
        if sign < 0. {
            source["rows"][0]["rhs"] = (-8.).into();
            for network in source["networks"].as_array_mut().unwrap() {
                let resource = &mut network["coupled_resources"][0];
                resource["initial"]["pairs"][0][0] = (-1.25).into();
                resource["initial"]["pairs"][1][0] = (-1.).into();
                for arc in resource["arcs"].as_array_mut().unwrap() {
                    arc["lower"]["pairs"][0][0] = (-8.).into();
                    arc["upper"]["pairs"][0][0] = 0.into();
                }
            }
        }
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            let (problem, registry) = model::format::Model::from_json(&source.to_string())
                .unwrap()
                .compile::<2>()
                .unwrap();
            let mut options = ModelTreeOptions::default();
            options.node.cg.stabilize = false;
            options.node.cg.bound_mode = mode;
            registry.configure_tree(&mut options).unwrap();
            let tree = baldes::master::tree::solve(problem, options).unwrap();
            assert_eq!(tree.status, TreeStatus::Optimal);
            assert_eq!(tree.incumbent, -24.);
            let mut quantities = [0.; 2];
            let mut gamma = [0.; 2];
            for fixed in tree.routes {
                let i = usize::from(fixed.route.network == 7);
                assert!(fixed.quantity > 0. && fixed.quantity.fract() == 0.);
                quantities[i] += fixed.quantity;
                gamma[i] += fixed.quantity
                    * fixed
                        .route
                        .route
                        .coupled_values
                        .iter()
                        .find(|(r, _)| *r == 0)
                        .unwrap()
                        .1;
            }
            assert_eq!(quantities, [3., 3.]);
            assert_eq!(gamma, [4. * sign; 2]);
        }
    }
}

#[test]
fn narrow_dyadic_domains_recover_exact_three_copy_gamma_totals() {
    use baldes::master::tree::{ModelTreeOptions, TreeStatus};

    let source = include_str!("../../fixtures/coupled/dyadic-narrow-interior-model.json");
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let (problem, registry) = model::format::Model::from_json(source)
            .unwrap()
            .compile::<2>()
            .unwrap();
        let mut options = ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.node.cg.bound_mode = mode;
        registry.configure_tree(&mut options).unwrap();
        let tree = baldes::master::tree::solve(problem, options).unwrap();
        assert_eq!(tree.status, TreeStatus::Optimal, "{tree:?}");
        assert_eq!(tree.incumbent, -24.);
        assert!(tree.bound <= -24. && tree.bound > -24. - 1e-7);
        let mut quantities = [0., 0.];
        let mut totals = [0., 0.];
        let mut controls = [
            std::collections::BTreeSet::new(),
            std::collections::BTreeSet::new(),
        ];
        for fixed in &tree.routes {
            let index = usize::from(fixed.route.network == 7);
            let gamma = fixed.route.route.coupled_values[0].1;
            assert!((0.125..=0.15625).contains(&gamma));
            assert!(fixed.quantity > 0. && fixed.quantity.fract() == 0.);
            quantities[index] += fixed.quantity;
            totals[index] += fixed.quantity * gamma;
            controls[index].insert(gamma.to_bits());
        }
        assert_eq!(quantities, [3., 3.]);
        assert!((totals[0] - 0.40625).abs() <= f64::EPSILON);
        assert!((totals[1] - 0.40625).abs() <= f64::EPSILON);
        assert!(controls.iter().all(|values| values.len() > 1));
    }
}

#[test]
fn narrow_fractional_domains_recover_exact_three_copy_gamma_totals() {
    use baldes::master::tree::{ModelTreeOptions, TreeStatus};

    let mut source: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/dyadic-narrow-interior-model.json"
    ))
    .unwrap();
    let expected_gamma = 1.0333333333333333_f64 - 1.;
    let expected_total = expected_gamma * 3.;
    source["rows"][0]["rhs"] = (expected_total * 2.).into();
    for network in source["networks"].as_array_mut().unwrap() {
        network["coupled_resources"][0]["initial_base"]["lower"] = 1.03.into();
        network["coupled_resources"][0]["initial_base"]["upper"] = 1.04.into();
    }
    let source = source.to_string();
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let (problem, registry) = model::format::Model::from_json(&source)
            .unwrap()
            .compile::<2>()
            .unwrap();
        let mut options = ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.node.cg.bound_mode = mode;
        registry.configure_tree(&mut options).unwrap();
        let tree = baldes::master::tree::solve(problem, options).unwrap();
        assert_eq!(tree.status, TreeStatus::Optimal, "{tree:?}");
        assert_eq!(tree.incumbent, -24.);
        let mut quantities = [0., 0.];
        let mut totals = [0., 0.];
        let mut controls = [
            std::collections::BTreeSet::new(),
            std::collections::BTreeSet::new(),
        ];
        for fixed in &tree.routes {
            let index = usize::from(fixed.route.network == 7);
            let gamma = fixed.route.route.coupled_values[0].1;
            assert!((0.03..=0.04).contains(&gamma));
            quantities[index] += fixed.quantity;
            totals[index] += fixed.quantity * gamma;
            controls[index].insert(gamma.to_bits());
        }
        assert_eq!(quantities, [3., 3.]);
        assert_eq!(totals, [expected_total, expected_total]);
        assert!(controls
            .iter()
            .flatten()
            .any(|&bits| f64::from_bits(bits) == expected_gamma));
    }
}
