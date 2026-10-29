use super::*;

#[test]
fn source_controls_cover_repeated_vertices_up_to_the_declared_walk_cap() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-objective-mapping.json"
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
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
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
fn source_control_cycles_match_hop_oracles_and_large_stable_shared_row_bounds() {
    let base: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/paired-cyclic-pricing-model.json"
    ))
    .unwrap();
    for dependent_increment in [0.5, 1.] {
        for cap in [4, 8, 16, 32, 64, 1_000_000] {
            for shared_row in [false, true] {
                if cap > 64 && !shared_row {
                    continue;
                }
                let mut input = base.clone();
                let network = &mut input["networks"][0];
                network["front_program_domain"] = false.into();
                network["route_domain"] =
                    serde_json::json!({"kind":"bounded_walks","max_arcs":cap});
                network["resources"][0]["bounds"] = serde_json::json!(vec![[0, cap]; 5]);
                network["coupled_objective_coefficients"] =
                    serde_json::json!([[0, if shared_row { 1. } else { -1. }]]);
                for arc in network["arcs"].as_array_mut().unwrap() {
                    arc["consumption"] = serde_json::json!([1.]);
                }
                let paired = &mut network["coupled_resources"][0];
                paired["policy"] = "strict_domain".into();
                paired["initial_base"]["upper"] = cap.into();
                paired["initial"]["pairs"] = serde_json::json!([[0, 0], [0, cap]]);
                for arc in paired["arcs"].as_array_mut().unwrap() {
                    arc["arrival"]["pairs"] = serde_json::json!([[1, 0], [1, cap]]);
                    arc["increment"]["pairs"] =
                        serde_json::json!([[dependent_increment, 0], [0, cap]]);
                    arc["lower"]["pairs"] = serde_json::json!([[0, 0], [0, cap]]);
                    arc["upper"]["pairs"] = serde_json::json!([[cap, 0], [0, cap]]);
                }
                if shared_row {
                    network["activity_entries"] = serde_json::json!([[0, 0, 1.]]);
                    input["rows"] = serde_json::json!([{"sense":"le","rhs":cap}]);
                }
                for mode in [BoundMode::Numerical, BoundMode::Certified] {
                    let problem = model::format::Model::from_json(&input.to_string())
                        .unwrap()
                        .problem::<2>()
                        .unwrap();
                    let mut options = Options::default();
                    options.cg.stabilize = false;
                    options.cg.artificial_cost = 100.;
                    options.cg.bound_mode = mode;
                    let root = model::solve(problem, options).unwrap();
                    // Every arc adds one base unit and a fixed dependent increment. The two
                    // starts permit every length 2..=cap, all with physical cost
                    // -20. Hence positive charges select 2 and negative ones cap.
                    let hops = if shared_row { 2 } else { cap };
                    let expected =
                        -20. + dependent_increment * if shared_row { 2. } else { -(cap as f64) };
                    assert_eq!(
                        root.master.status,
                        Status::Optimal,
                        "cap={cap} shared={shared_row} {mode:?}"
                    );
                    assert_eq!(root.master.lp, expected);
                    assert!(root.master.bound <= expected);
                    assert!(expected - root.master.bound < 1e-7);
                    let selected: Vec<_> = root
                        .routes
                        .iter()
                        .zip(&root.values)
                        .filter(|(_, value)| **value > 0.)
                        .collect();
                    assert_eq!(selected.len(), 1);
                    assert_eq!(*selected[0].1, 1.);
                    assert_eq!(selected[0].0.route.arcs.len(), hops as usize);
                    assert_eq!(
                        selected[0].0.route.coupled_values,
                        vec![(0, hops as f64 * dependent_increment)]
                    );
                    assert_eq!(
                        selected[0].0.route.coupled_controls[0].base,
                        (0..=hops).map(|value| value as f64).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn cyclic_source_controls_price_an_active_shared_row_against_a_two_column_lp_oracle() {
    let source = include_str!("../../../fixtures/coupled/paired-cyclic-active-row-model.json");
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let problem = model::format::Model::from_json(source)
            .unwrap()
            .problem::<2>()
            .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.artificial_cost = 100.;
        options.cg.bound_mode = mode;
        let root = model::solve(problem, options).unwrap();
        // Longer routes increase both cost and the <= row activity and are
        // dominated. For the two shortest starts the exact LP is
        // min -17*x-18*y, x+y<=1, -x+2*y<=0, x,y>=0.
        // Its optimum is x=2/3, y=1/3, objective -52/3.
        assert_eq!(root.master.status, Status::Optimal);
        assert!((root.master.lp + 52. / 3.).abs() < 1e-10);
        assert!((root.master.bound + 52. / 3.).abs() < 1e-7);
        assert!((root.duals[1] + 1. / 3.).abs() < 1e-10);
        let mut quantities = [0., 0.];
        let mut activity = 0.;
        for (route, &value) in root
            .routes
            .iter()
            .zip(&root.values)
            .filter(|(_, value)| **value > 0.)
        {
            let first = route.route.arcs[0];
            assert!(first == 1 || first == 3);
            quantities[usize::from(first == 3)] += value;
            let gamma = route.route.coupled_values[0].1;
            assert_eq!(gamma, if first == 1 { 3. } else { 2. });
            activity += value * (gamma - if first == 1 { 4. } else { 0. });
        }
        assert!((quantities[0] - 2. / 3.).abs() < 1e-10);
        assert!((quantities[1] - 1. / 3.).abs() < 1e-10);
        assert!(activity.abs() < 1e-10);
    }
}

#[test]
fn source_control_bound_specialization_preserves_fractional_costs_and_active_lower_floors() {
    let base: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/paired-cyclic-pricing-model.json"
    ))
    .unwrap();
    for (coefficient, lower, gamma) in [(-0.5, 0., 4.), (-1., 5., 8.)] {
        let mut input = base.clone();
        let network = &mut input["networks"][0];
        network["front_program_domain"] = false.into();
        network["route_domain"] = serde_json::json!({"kind":"bounded_walks","max_arcs":4});
        network["resources"][0]["bounds"] = serde_json::json!(vec![[0, 4]; 5]);
        network["coupled_objective_coefficients"] = serde_json::json!([[0, coefficient]]);
        for arc in network["arcs"].as_array_mut().unwrap() {
            arc["consumption"] = serde_json::json!([1.]);
        }
        let paired = &mut network["coupled_resources"][0];
        paired["policy"] = "strict_domain".into();
        for arc in paired["arcs"].as_array_mut().unwrap() {
            arc["arrival"]["pairs"] = serde_json::json!([[1, 0], [1, 8]]);
            arc["lower"]["pairs"] = serde_json::json!([[lower, 0], [0, 8]]);
        }
        let problem = model::format::Model::from_json(&input.to_string())
            .unwrap()
            .problem::<2>()
            .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.bound_mode = BoundMode::Certified;
        let root = model::solve(problem, options).unwrap();
        let expected = -20. + coefficient * gamma;
        assert_eq!(root.master.status, Status::Optimal);
        assert_eq!(root.master.lp, expected);
        assert!(root.master.bound <= expected);
        assert!(expected - root.master.bound < 1e-7);
        for (route, _) in root
            .routes
            .iter()
            .zip(&root.values)
            .filter(|(_, value)| **value > 0.)
        {
            assert_eq!(route.route.arcs.len(), 4);
            assert_eq!(route.route.coupled_values, vec![(0, gamma)]);
        }
    }
}

#[test]
fn constant_source_bounds_preserve_knot_and_slope_objective_normalizations() {
    let base: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/paired-cyclic-pricing-model.json"
    ))
    .unwrap();
    for knots in [false, true] {
        for relative in [false, true] {
            let mut input = base.clone();
            let network = &mut input["networks"][0];
            network["front_program_domain"] = false.into();
            network["route_domain"] = serde_json::json!({"kind":"bounded_walks","max_arcs":8});
            network["coupled_objective_coefficients"] = serde_json::json!([[0, -2.]]);
            for arc in network["arcs"].as_array_mut().unwrap() {
                arc["consumption"] = serde_json::json!([1.]);
            }
            let paired = &mut network["coupled_resources"][0];
            paired["policy"] = "strict_domain".into();
            paired["initial"]["pairs"] = serde_json::json!([[5, 0], [0, 8]]);
            paired["normalization"] = if relative {
                "final_minus_initial_minimum"
            } else {
                "absolute_final_value"
            }
            .into();
            for arc in paired["arcs"].as_array_mut().unwrap() {
                arc["arrival"]["pairs"] = serde_json::json!([[1, 0], [1, 8]]);
                arc["upper"]["pairs"] = serde_json::json!([[13, 0], [0, 8]]);
            }
            if knots {
                input["networks"][0] = control_point_source_model(input["networks"][0].clone());
            }
            let problem = model::format::Model::from_json(&input.to_string())
                .unwrap()
                .problem::<2>()
                .unwrap();
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = BoundMode::Certified;
            let root = model::solve(problem, options).unwrap();
            // All feasible paths have Gamma=5+h; h<=8. The negative weight
            // selects h=8, with the initial five removed only in relative mode.
            let expected = if relative { -36. } else { -46. };
            assert_eq!(root.master.status, Status::Optimal);
            assert_eq!(root.master.lp, expected);
            assert!(root.master.bound <= expected);
            assert!(expected - root.master.bound < 1e-7);
            for (route, _) in root
                .routes
                .iter()
                .zip(&root.values)
                .filter(|(_, value)| **value > 0.)
            {
                assert_eq!(route.route.arcs.len(), 8);
                assert_eq!(route.route.coupled_values, vec![(0, 13.)]);
            }
        }
    }
}
