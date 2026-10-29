use super::*;

#[test]
fn paired_mixed_reference_row_uses_a_proved_control_instead_of_the_invalid_reference_bound() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
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
