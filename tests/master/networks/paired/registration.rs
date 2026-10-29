use super::*;

#[test]
fn registered_fictive_selection_factories_match_reference_and_path_oracle() {
    assert_eq!(
        check_registered_cases(include_str!(
            "../../../fixtures/coupled/metasolver-fictive-preparation.json"
        )),
        288
    );
}

#[test]
fn registered_pickup_delivery_factory_models_match_reference_and_load_oracle() {
    assert_eq!(
        check_registered_cases(include_str!(
            "../../../fixtures/coupled/metasolver-spd-preparation.json"
        )),
        180
    );
}

#[test]
fn registered_paired_factory_models_match_reference_and_independent_optima() {
    assert_eq!(
        check_registered_cases(include_str!(
            "../../../fixtures/coupled/metasolver-paired-preparation.json"
        )),
        504
    );
}

#[test]
fn nonlinear_paired_factories_and_waiting_conversion_match_actual_reference() {
    assert_eq!(
        check_registered_cases(include_str!(
            "../../../fixtures/coupled/metasolver-paired-pwl-preparation.json"
        )),
        432
    );
}

#[test]
fn feasibility_factories_preserve_constraints_without_charging_dependent_values() {
    assert_eq!(
        check_registered_cases(include_str!(
            "../../../fixtures/coupled/metasolver-paired-feasibility-preparation.json"
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
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../../fixtures/coupled/metasolver-paired-feasibility-preparation.json"
    ))
    .unwrap();
    let cost: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
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
        "../../../fixtures/coupled/metasolver-paired-preparation.json"
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
