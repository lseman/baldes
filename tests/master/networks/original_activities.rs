use super::*;
use model::{format::Model, NetworkPricingSchedule};

fn input(case: &serde_json::Value, singular: bool) -> serde_json::Value {
    let mut value = case["model"].clone();
    if singular {
        value["networks"] = value["networks"][0].clone();
    }
    value
}
fn configured(mode: BoundMode, schedule: NetworkPricingSchedule, direction: f64) -> Options {
    let mut options = super::options();
    options.cg.bound_mode = mode;
    options.network_pricing_schedule = schedule;
    options.pricing_options = vec![
        ("BALDESuseBidirectionalSearch".into(), direction),
        ("BALDESmaxNumOfColsPerExactIteration".into(), 1.),
    ];
    options
}
fn fraction(value: &serde_json::Value) -> f64 {
    let mut parts = value.as_str().unwrap().split('/');
    parts.next().unwrap().parse::<f64>().unwrap()
        / parts
            .next()
            .map(|p| p.parse::<f64>().unwrap())
            .unwrap_or(1.)
}

#[test]
fn registered_accumulated_variables_match_all_eight_independent_lp_controls() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/metasolver_registered_activities.json"
    ))
    .unwrap();
    let mut compared = 0;
    for case in corpus["cases"].as_array().unwrap() {
        let (problem, registry) = Model::from_json(&input(case, false).to_string())
            .unwrap()
            .compile::<2>()
            .unwrap();
        assert_eq!(
            problem.networks[0]
                .graph
                .coupled_kernel()
                .unwrap()
                .objective_coefficient(1),
            case["objective_resource_coefficient"].as_f64()
        );
        assert_eq!(
            problem.networks[0].activity_entries.len(),
            usize::from(!case["row"].is_null())
        );
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for schedule in [
                NetworkPricingSchedule::FullSweep,
                NetworkPricingSchedule::Cyclic,
                NetworkPricingSchedule::Sorted,
                NetworkPricingSchedule::PromisingFirst,
                NetworkPricingSchedule::UnvisitedFirst,
            ] {
                for direction in [0., 1., 2.] {
                    let mut options = configured(mode, schedule, direction);
                    registry.configure_pricing(&mut options);
                    let output = model::solve(problem.clone(), options).unwrap();
                    optimal(&output, case["oracle"]["objective_float"].as_f64().unwrap());
                    let theta = fraction(&case["oracle"]["fraction_first_path"]);
                    let expected = theta * case["activity_values_by_path"][0].as_f64().unwrap()
                        + (1. - theta) * case["activity_values_by_path"][1].as_f64().unwrap();
                    let activity = registry.activity(42, 37, &output).unwrap();
                    assert!(
                        (activity.lower - expected).abs() < 1e-7
                            && (activity.upper - expected).abs() < 1e-7,
                        "{} {:?}",
                        case["name"],
                        activity
                    );
                    if mode == BoundMode::Certified {
                        let mut parts = case["oracle"]["objective"].as_str().unwrap().split('/');
                        let n = parts.next().unwrap().parse::<f64>().unwrap();
                        let d = parts
                            .next()
                            .map(|s| s.parse::<f64>().unwrap())
                            .unwrap_or(1.);
                        let value = output.master.bound * d;
                        let error = output.master.bound.mul_add(d, -value);
                        assert!(
                            value < n || (value == n && error <= 0.),
                            "{} {:?}",
                            case["name"],
                            output.master
                        );
                    }
                    let result = model::format::SolveResult::root(&output, 0.)
                        .with_original_variables(&registry)
                        .unwrap();
                    let named = result
                        .original_variables
                        .iter()
                        .find(|v| v.network == 42 && v.id == 37)
                        .unwrap();
                    assert_eq!((named.lower, named.upper), (activity.lower, activity.upper));
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, 240);
}

#[test]
fn model_problem_retains_activity_rows_without_a_pricing_registry_sidecar() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/metasolver_registered_activities.json"
    ))
    .unwrap();
    for case in corpus["cases"].as_array().unwrap() {
        for singular in [false, true] {
            let problem = Model::from_json(&input(case, singular).to_string())
                .unwrap()
                .problem::<2>()
                .unwrap();
            let output = model::solve(
                problem,
                configured(BoundMode::Certified, NetworkPricingSchedule::FullSweep, 1.),
            )
            .unwrap();
            optimal(&output, case["oracle"]["objective_float"].as_f64().unwrap());
        }
    }
}

#[test]
fn activity_variable_identity_is_scoped_by_network_and_survives_fresh_compilation() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/metasolver_registered_activities.json"
    ))
    .unwrap();
    let case = corpus["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "custom-pure-row")
        .unwrap();
    let mut source = input(case, true);
    let mut second = source["networks"].clone();
    second["id"] = 7.into();
    source["networks"] = serde_json::json!([source["networks"].clone(), second]);
    source["pricing_variables"][2]["cost"] = 2.into();
    let copies = source["pricing_variables"].as_array().unwrap().clone();
    for mut variable in copies {
        variable["network"] = 7.into();
        if variable["id"] == 37 {
            variable["cost"] = 1.into();
        }
        source["pricing_variables"]
            .as_array_mut()
            .unwrap()
            .push(variable);
    }
    let (problem, registry) = Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    let mut options = configured(BoundMode::Certified, NetworkPricingSchedule::Cyclic, 1.);
    registry.configure_pricing(&mut options);
    let output = model::solve(problem, options).unwrap();
    optimal(&output, -33.);
    for (network, expected) in [(7, 3.), (42, 2.)] {
        let value = registry.activity(network, 37, &output).unwrap();
        assert!((value.lower - expected).abs() < 1e-7 && (value.upper - expected).abs() < 1e-7);
    }
}

#[test]
fn zero_and_signed_original_costs_preserve_intrinsic_shared_row_activity() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/metasolver_registered_activities.json"
    ))
    .unwrap();
    let case = corpus["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "custom-pure-row")
        .unwrap();
    for coefficient in [0., -2.] {
        let mut source = input(case, false);
        for variable in source["pricing_variables"].as_array_mut().unwrap() {
            if variable["id"] == 37 {
                variable["cost"] = coefficient.into();
            } else if variable["id"] == 103 {
                // Remove the zero-objective tie without changing the activity.
                variable["cost"] = (-19.).into();
            }
        }
        let (problem, registry) = Model::from_json(&source.to_string())
            .unwrap()
            .compile::<2>()
            .unwrap();
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for direction in [0., 1., 2.] {
                let mut options = configured(mode, NetworkPricingSchedule::FullSweep, direction);
                registry.configure_pricing(&mut options);
                let output = model::solve(problem.clone(), options).unwrap();
                optimal(&output, -20. + coefficient * 11.);
                let value = registry.activity(42, 37, &output).unwrap();
                assert!((value.lower - 11.).abs() < 1e-7);
                assert!((value.upper - 11.).abs() < 1e-7);
            }
        }
    }
}

#[test]
fn accumulated_boxes_prove_the_graph_domain_independently_of_master_rows_and_costs() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/metasolver_registered_activities.json"
    ))
    .unwrap();
    for case in corpus["cases"].as_array().unwrap() {
        let (problem, registry) = Model::from_json(&input(case, false).to_string())
            .unwrap()
            .compile::<2>()
            .unwrap();
        let values = case["activity_values_by_path"].as_array().unwrap();
        let lower = values
            .iter()
            .map(|v| v.as_f64().unwrap())
            .fold(f64::INFINITY, f64::min);
        let upper = values
            .iter()
            .map(|v| v.as_f64().unwrap())
            .fold(f64::NEG_INFINITY, f64::max);
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            let mut options = configured(mode, NetworkPricingSchedule::FullSweep, 1.);
            registry
                .configure_boxes(
                    &mut options,
                    &[model::OriginalVariableBound {
                        network: 42,
                        id: 37,
                        lower,
                        upper: Some(upper),
                    }],
                )
                .unwrap();
            let output = model::solve(problem.clone(), options).unwrap();
            optimal(&output, case["oracle"]["objective_float"].as_f64().unwrap());
        }
        // Shared-row restrictions are not graph restrictions. Even when the
        // master needs gamma >= 5, the graph still has its gamma=2 route.
        for (lower, upper) in [(lower + 1., upper), (lower, upper - 1.)] {
            let mut options =
                configured(BoundMode::Certified, NetworkPricingSchedule::FullSweep, 1.);
            registry
                .configure_boxes(
                    &mut options,
                    &[model::OriginalVariableBound {
                        network: 42,
                        id: 37,
                        lower,
                        upper: Some(upper),
                    }],
                )
                .unwrap();
            let error = model::solve(problem.clone(), options).err().unwrap();
            assert!(
                error.to_string().contains("envelope is unproved"),
                "{error}"
            );
        }
    }
}

#[test]
fn integer_accumulated_registration_uses_shared_mapped_branch_rows() {
    use baldes::master::preprocessing::Bounds;
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/metasolver_registered_activities.json"
    ))
    .unwrap();
    let case = corpus["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "custom-mixed-row")
        .unwrap();
    let mut source = input(case, false);
    source["pricing_variables"][2]["integer"] = true.into();
    let (problem, registry) = Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    let branch = &registry.branch_variables()[0];
    assert_eq!(branch.name, "X_42_37");
    assert_eq!(branch.networks[0].activities, [(1, 1.)]);
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut options = configured(mode, NetworkPricingSchedule::FullSweep, 1.);
        registry.configure_pricing(&mut options);
        let parent = model::solve(problem.clone(), options.clone()).unwrap();
        optimal(&parent, -279. / 17.);
        let row_mapping = model::MappedBranchVariable::from_shared_row(
            "original-shared-row",
            0.,
            &parent.snapshot,
            0,
            &[42],
            &[],
        )
        .unwrap();
        assert_eq!(row_mapping.networks[0].activities, [(1, 1.)]);
        let row_conflict = row_mapping
            .apply(
                &parent.snapshot,
                Bounds {
                    lower: f64::NEG_INFINITY,
                    upper: 4.,
                },
            )
            .unwrap();
        assert_eq!(
            row_conflict.solve(options.clone()).unwrap().master.status,
            Status::Infeasible
        );
        let down = branch
            .apply(
                &parent.snapshot,
                Bounds {
                    lower: f64::NEG_INFINITY,
                    upper: 3.,
                },
            )
            .unwrap();
        optimal(&down.solve(options.clone()).unwrap(), -4.5);
        let impossible = branch
            .apply(
                &parent.snapshot,
                Bounds {
                    lower: f64::NEG_INFINITY,
                    upper: 2.,
                },
            )
            .unwrap();
        assert_eq!(
            impossible.solve(options.clone()).unwrap().master.status,
            Status::Infeasible
        );
        let up = branch
            .apply(
                &parent.snapshot,
                Bounds {
                    lower: 4.,
                    upper: f64::INFINITY,
                },
            )
            .unwrap();
        let result = up.solve(options.clone()).unwrap();
        optimal(&result, -16.);
        let value = registry.activity(42, 37, &result).unwrap();
        assert!((value.lower - 4.).abs() < 1e-7 && (value.upper - 4.).abs() < 1e-7);
        optimal(&parent.snapshot.solve(options).unwrap(), parent.master.lp);
    }
}

#[test]
fn integer_accumulated_registration_rejects_fractional_intrinsic_paths() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/metasolver_registered_activities.json"
    ))
    .unwrap();
    let case = corpus["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "custom-mixed-row")
        .unwrap();
    let mut source = input(case, false);
    source["pricing_variables"][2]["integer"] = true.into();
    source["networks"][0]["coupled_resources"][1]["weights"][0] = 0.5.into();
    let error = Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .err()
        .unwrap();
    assert!(error.to_string().contains("exactly integer"), "{error}");
}

#[test]
fn activity_rows_have_scoped_cutoff_enumeration_without_pruning_improving_integer_routes() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/metasolver_registered_activities.json"
    ))
    .unwrap();
    for case in corpus["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["name"].as_str().unwrap().contains("row"))
    {
        let expected = if case["name"].as_str().unwrap().starts_with("custom") {
            -9.
        } else {
            -10.
        };
        let (problem, registry) = Model::from_json(&case["model"].to_string())
            .unwrap()
            .compile::<2>()
            .unwrap();
        let mut options = configured(BoundMode::Certified, NetworkPricingSchedule::FullSweep, 1.);
        registry.configure_pricing(&mut options);
        let root = model::solve(problem, options.clone()).unwrap();
        let view = root
            .snapshot
            .enumerate_below(expected + 0.5, &model::EnumerationOptions::default())
            .unwrap();
        assert!(view.report().complete && view.report().coverage_cutoff == Some(expected + 0.5));
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
        let resumed = root.snapshot.solve(options).unwrap();
        assert!((resumed.master.lp - root.master.lp).abs() < 1e-7);
        assert!(!resumed.snapshot.enumeration_report().complete);
    }
}
