use super::*;

#[test]
fn paired_owner_costs_use_selected_columns_in_an_explicit_complete_universe() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for case_index in [0, 18] {
        for coefficient in [-1_f64, 0., 0.5, 1., 2., 5., 8., 1e-308] {
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
                        options.enumeration = Some(Default::default());
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
fn paired_owner_registration_accepts_signed_costs_rows_and_integer_metadata() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    source["rows"] = serde_json::json!([{"sense":"le","rhs":10.}]);
    source["pricing_variables"] = serde_json::json!([{
        "network":42,"id":901,"cost":-1.,"accumulated_resource":0,"entries":[[0,1.]]
    }]);
    let (problem, _) = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    assert_eq!(
        problem.networks[0]
            .graph
            .coupled_kernel()
            .unwrap()
            .objective_coefficient(0),
        Some(-1.)
    );
    assert_eq!(problem.networks[0].activity_entries.len(), 1);
    source["pricing_variables"][0]["integer"] = true.into();
    let (_, registry) = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    assert!(registry.variables()[0].integer);
    assert!(registry.branch_variables().is_empty());
    source["networks"][0]["coupled_resources"][0]["mode"] = "feasibility".into();
    source["pricing_variables"][0]["cost"] = 0.into();
    let error = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap_err()
        .to_string();
    assert!(error.contains("feasibility-only paired"), "{error}");
}

#[test]
fn paired_integer_registration_retains_fractional_selected_activity() {
    // Captured I registration: integer-all-fractional/paired-result.json.
    // The owner finishes at -19.5 although the associated activity is .5.
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    for arc in source["networks"][0]["paired_front_programs"][0]["arcs"]
        .as_array_mut()
        .unwrap()
    {
        arc["dependent"]["function"]["pairs"] = serde_json::json!([[0.25, 0.], [0., 8.]]);
    }
    source["pricing_variables"] = serde_json::json!([{
        "network":42,"id":901,"cost":1.,"accumulated_resource":0,"integer":true
    }]);
    source["networks"][0]["paired_owner_registration_bounds"] =
        serde_json::json!([[0, [0., 100.]]]);
    let (problem, registry) = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    assert!(registry.variables()[0].integer);
    assert!(registry.branch_variables().is_empty());
    let mut options = Options::default();
    options.cg.stabilize = false;
    let output = model::solve(problem, options).unwrap();
    assert!(
        (output.master.lp + 19.5).abs() < 1e-8,
        "{:?}",
        output.master
    );
    let activity = registry.activity(42, 901, &output).unwrap();
    assert!(activity.lower <= 0.5 && activity.upper >= 0.5);
}

#[test]
fn direct_routing_signed_owner_objective_never_uses_unit_raw_pricing_as_a_bound() {
    use baldes::master::routing::{self, RootOptions, RoutingContext};
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    for arc in source["networks"][0]["paired_front_programs"][0]["arcs"]
        .as_array_mut()
        .unwrap()
    {
        arc["dependent"]["function"]["pairs"] = serde_json::json!([[1., 0.], [0., 8.]]);
    }
    let (problem, _) = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut graph = problem.networks[0].graph.clone();
        graph
            .set_coupled_objective_coefficients(&[(0, -1.)])
            .unwrap();
        let mut context = RoutingContext::set_partitioning(graph, 1.).unwrap();
        // Optional path selection, including the unused interior vertex.
        for row in &mut context.model.rows {
            row.sense = RowSense::LessEqual;
        }
        let mut options = RootOptions::default();
        options.enumerate = false;
        options.cg.stabilize = false;
        options.bound_mode = mode;
        let output = routing::solve(context, options).unwrap();
        assert!((output.master.lp + 22.).abs() < 1e-8, "{:?}", output.master);
        assert_eq!(output.master.status, Status::Stalled);
        assert_eq!(output.master.numerical_bound, f64::NEG_INFINITY);
        assert_eq!(output.master.certified_bound, f64::NEG_INFINITY);
    }
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
    second["paired_front_programs"][0]["root"]["function"]["pairs"][0][0] = serde_json::json!(2.);
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
            options.enumeration = Some(Default::default());
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
                    let pairs = source["networks"][0]["paired_front_programs"][0]["arcs"][index]
                        ["dependent"]["function"]["pairs"]
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
                options.enumeration = Some(Default::default());
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
fn paired_owner_cost_mapping_matches_registered_selected_columns_after_explicit_enumeration() {
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
        let compiled = model::format::Model::from_json(&case["model"].to_string())
            .unwrap()
            .compile::<2>();
        if coefficient < 0. {
            assert_eq!(case["reference_root"], -18.);
            assert_eq!(expected, -22.);
        }
        if let Some(incumbent) = case["reference_incumbent"].as_f64() {
            assert_eq!(incumbent, expected);
        }
        let (problem, registry) = compiled.unwrap();
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = mode;
            options.enumeration = Some(Default::default());
            registry.configure_pricing(&mut options);
            let result = model::solve(problem.clone(), options).unwrap();
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
                result.master.bound <= expected + 1e-10 && expected - result.master.bound < 1e-7,
                "{:#?}",
                result.master
            );
            let gamma = registry.activity(42, 901, &result).unwrap();
            assert_eq!((gamma.lower, gamma.upper), (2., 2.));
        }
    }
}

#[test]
fn signed_owner_cost_preserves_raw_paired_pricing_and_restricted_master_cost_separately() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-objective-mapping.json"
    ))
    .unwrap();
    let case = corpus["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["coefficient"] == -1.)
        .unwrap();
    let (problem, registry) = model::format::Model::from_json(&case["model"].to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    let graph = &problem.networks[0].graph;
    let mut raw =
        baldes::pricing::Solver::<2>::new(graph.vertices.len(), graph.resources.len()).unwrap();
    raw.graph = graph.clone();
    assert_eq!(
        raw.price_phase(0, 20, false).unwrap().minimum_reduced_cost,
        -18.
    );
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.max_iterations = 12;
    // Owner metadata lives in the compiled model, so correctness does not
    // depend on installing the optional cached-pricing decomposition.
    let output = model::solve(problem, options).unwrap();
    assert_eq!(output.master.lp, -22.);
    assert_eq!(output.master.status, Status::Stalled);
    assert_eq!(output.master.certified_bound, f64::NEG_INFINITY);
    assert_eq!(output.master.numerical_bound, f64::NEG_INFINITY);
    let gamma = registry.activity(42, 901, &output).unwrap();
    assert_eq!((gamma.lower, gamma.upper), (2., 2.));
    let route = output
        .routes
        .iter()
        .zip(&output.values)
        .find(|(_, value)| **value > 0.5)
        .unwrap()
        .0;
    assert_eq!(route.route.coupled_values, [(0, 2.)]);
    assert_eq!(route.route.original_cost, -22.);
}

#[test]
fn paired_owner_rows_replay_recorded_selected_columns_without_a_source_control_projection() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-owner-rows.json"
    ))
    .unwrap();
    for case in corpus["cases"].as_array().unwrap() {
        if case["case"] == "singleton-lower" {
            assert_eq!(
                case["native_outcome"],
                "adapter_rejects_propagated_lower_bound"
            );
            let (problem, _) = model::format::Model::from_json(&case["model"].to_string())
                .unwrap()
                .compile::<2>()
                .unwrap();
            let error = model::solve(problem, Options::default())
                .err()
                .expect("owner callback must reject the propagated bound")
                .to_string();
            assert!(
                error.contains("non-zero lower bounds on variables associated to arcs"),
                "{error}"
            );
            continue;
        }
        let (problem, registry) = model::format::Model::from_json(&case["model"].to_string())
            .unwrap()
            .compile::<2>()
            .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.max_iterations = 12;
        registry.configure_pricing(&mut options);
        let output = model::solve(problem, options).unwrap();
        assert_eq!(
            output.master.lp,
            case["native_restricted_master_lp"].as_f64().unwrap(),
            "{case}"
        );
        let expected = if case["case"] == "baseline" || case["case"] == "nonbinding-upper" {
            Status::Optimal
        } else {
            Status::Stalled
        };
        assert_eq!(output.master.status, expected);
        if expected == Status::Stalled {
            assert_eq!(output.master.certified_bound, f64::NEG_INFINITY);
            assert_eq!(output.master.numerical_bound, f64::NEG_INFINITY);
        }
        for (route, quantity) in output
            .routes
            .iter()
            .zip(&output.values)
            .filter(|(_, q)| **q > 1e-8)
        {
            assert!(*quantity > 0.);
            assert!(
                case["native_selected_columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|column| {
                        column["arcs"] == serde_json::json!(route.route.arcs)
                            && column["associations"][0]["value"] == route.route.coupled_values[0].1
                    }),
                "{route:?}"
            );
        }
    }
}

#[test]
fn paired_owner_bound_propagation_keeps_mixed_signed_rows_and_callback_tolerance() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    // Fresh owner-bound-propagation captures use explicit gamma[0,100].
    // Bounds are registration state, never substituted for a shared row.
    for (sense, rhs, gamma, arc, rejected, expected_upper) in [
        ("ge", 4., 1., 0., true, 100.),
        ("ge", 4., 1., 10., false, 100.),
        ("ge", 4., 1., -10., true, 100.),
        ("le", -4., -1., 0., true, 100.),
        ("ge", -4., -1., 0., false, 4.),
        ("eq", 4., 1., 10., false, 4.),
        ("ge", 1e-6, 1., 0., false, 100.),
        ("ge", 3e-6, 1., 0., true, 100.),
    ] {
        let mut source = corpus["cases"][0]["model"].clone();
        source["rows"] = serde_json::json!([{"sense":sense,"rhs":rhs}]);
        source["networks"][0]["paired_owner_registration_bounds"] =
            serde_json::json!([[0, [0., 100.]]]);
        source["networks"][0]["arcs"][1]["entries"] = serde_json::json!([[0, arc]]);
        source["pricing_variables"] = serde_json::json!([{
            "network":42,"id":901,"cost":1.,"accumulated_resource":0,"entries":[[0,gamma]]
        }]);
        let (problem, _) = model::format::Model::from_json(&source.to_string())
            .unwrap()
            .compile::<2>()
            .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.max_iterations = 12;
        let result = model::solve(problem, options);
        if rejected {
            let error = result
                .err()
                .expect("owner callback must reject the propagated bound")
                .to_string();
            assert!(
                error.contains("non-zero lower bounds on variables associated to arcs"),
                "{sense} {rhs} {gamma} {arc}: {error}"
            );
        } else {
            let output = result.unwrap();
            let bounds = output
                .snapshot
                .preprocessing_report()
                .paired_owner_bounds
                .iter()
                .find(|&&(network, resource, _)| network == 42 && resource == 0)
                .unwrap()
                .2;
            assert!(
                (bounds.upper - expected_upper).abs() < 1e-8,
                "{sense} {rhs} {gamma} {arc}: {bounds:?}"
            );
            assert!(bounds.lower <= 2e-6, "{bounds:?}");
        }
    }
}

#[test]
fn direct_routing_paired_row_uses_selected_activity_and_rejects_fractional_route_witness() {
    use baldes::master::routing::{self, RootOptions, RoutingContext};
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let (problem, _) = model::format::Model::from_json(&corpus["cases"][0]["model"].to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    let mut graph = problem.networks[0].graph.clone();
    graph
        .set_paired_owner_registration_bounds(&[(0, [0., 100.])])
        .unwrap();
    let mut context = RoutingContext::set_partitioning(graph, 1.).unwrap();
    for row in &mut context.model.rows {
        row.sense = RowSense::LessEqual;
    }
    // Vertex1 is unused, so its pre-existing row has no physical arc terms.
    // Replace it with gamma<=2, the captured owner-bound-upper-row2 case.
    context.model.rows[1].rhs = 2.;
    context.model.networks[0]
        .activity_entries
        .push(model::AccumulatedActivityEntry {
            row: 1,
            resource: 0,
            coefficient: 1.,
        });
    let mut options = RootOptions::default();
    options.enumerate = false;
    options.cg.stabilize = false;
    options.cg.max_iterations = 12;
    let output = routing::solve(context, options).unwrap();
    // Captured native RMP=-11.333333; native owner terminates with a fatal
    // duality gap. Source output keeps that RMP without claiming its optimum.
    assert!(
        (output.master.lp + 34. / 3.).abs() < 1e-7,
        "{:?}",
        output.master
    );
    assert_eq!(output.master.status, Status::Stalled);
    assert!(
        output.incumbent >= -1e-8,
        "a fractional selected route is not an integer witness"
    );
    assert_eq!(output.master.numerical_bound, f64::NEG_INFINITY);
}

#[test]
fn paired_owner_registration_upper_is_not_a_shared_row_or_a_path_window() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][0]["model"].clone();
    source["networks"][0]["paired_owner_registration_bounds"] = serde_json::json!([[0, [0., 2.]]]);
    source["pricing_variables"] = serde_json::json!([{
        "network":42,"id":901,"cost":1.,"accumulated_resource":0
    }]);
    let (problem, registry) = model::format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    assert_eq!(
        problem.networks[0]
            .graph
            .coupled_kernel()
            .unwrap()
            .paired_owner_registration_bounds(),
        &[(0, [0., 2.])]
    );
    let mut options = Options::default();
    options.cg.stabilize = false;
    let output = model::solve(problem, options).unwrap();
    // Native actual Mlp/PB and public printed incumbent are -17/gamma3,
    // despite current owner UB2 at fillRCSPInput. A row gamma<=2 differs.
    assert!((output.master.lp + 17.).abs() < 1e-8, "{:?}", output.master);
    let activity = registry.activity(42, 901, &output).unwrap();
    assert_eq!((activity.lower, activity.upper), (3., 3.));
    assert_eq!(
        output.snapshot.preprocessing_report().paired_owner_bounds[0]
            .2
            .upper,
        2.
    );
}

#[test]
fn paired_owner_local_and_aggregate_bounds_survive_multiplicity_and_fixed_routes() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for (multiplicity, local_lower) in [(1., 0.), (2., -4.)] {
        let mut source = corpus["cases"][0]["model"].clone();
        source["networks"][0]["multiplicity"] = serde_json::json!([0., multiplicity]);
        source["networks"][0]["paired_owner_registration_bounds"] =
            serde_json::json!([[0, [-4., 100.]]]);
        source["pricing_variables"] = serde_json::json!([{
            "network":42,"id":901,"cost":1.,"accumulated_resource":0
        }]);
        let (problem, _) = model::format::Model::from_json(&source.to_string())
            .unwrap()
            .compile::<2>()
            .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        let output = model::solve(problem, options.clone()).unwrap();
        let report = output.snapshot.preprocessing_report();
        assert_eq!(report.paired_owner_bounds[0].2.lower, local_lower);
        assert_eq!(report.paired_owner_aggregate_bounds[0].2.lower, 0.);
        if multiplicity == 2. {
            let tag = output.values.iter().position(|&value| value > 0.).unwrap();
            let fixed = output
                .snapshot
                .fix_route(tag, 1.)
                .unwrap()
                .solve(options)
                .unwrap();
            let report = fixed.snapshot.preprocessing_report();
            assert_eq!(report.paired_owner_aggregate_bounds[0].2.upper, 97.);
            // Remaining single-copy domain converts residual global UB97 with
            // signed local lower: effective minimum1 leaves the same local UB.
            assert_eq!(report.paired_owner_bounds[0].2.upper, 97.);
            assert_eq!(report.paired_owner_bounds[0].2.lower, 0.);
        }
    }
}
