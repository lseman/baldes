//! Full registered reference models with mixed resource kinds and identities.
use super::*;
use baldes::master::{
    model::NetworkPricingSchedule,
    tree::{ModelTreeOptions, TreeStatus},
};

fn controls() -> Vec<serde_json::Value> {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/metasolver_registered_networks.json"
    ))
    .unwrap();
    assert_eq!(
        corpus["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let controls = corpus["cases"].as_array().unwrap().clone();
    assert_eq!(controls.len(), 48);
    let strong: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/metasolver_registered_strong.json"
    ))
    .unwrap();
    assert_eq!(strong["reference_sha256"], corpus["reference_sha256"]);
    assert_eq!(strong["cases"].as_array().unwrap().len(), 6);
    controls
        .into_iter()
        .chain(strong["cases"].as_array().unwrap().iter().cloned())
        .collect()
}
fn schedules() -> [(NetworkPricingSchedule, &'static str); 5] {
    [
        (NetworkPricingSchedule::FullSweep, "full-sweep"),
        (NetworkPricingSchedule::Cyclic, "cyclic"),
        (NetworkPricingSchedule::Sorted, "sorted"),
        (NetworkPricingSchedule::PromisingFirst, "promising-first"),
        (NetworkPricingSchedule::UnvisitedFirst, "unvisited-first"),
    ]
}
fn parse(input: &serde_json::Value) -> Problem<2> {
    model::format::Model::from_json(&input.to_string())
        .unwrap()
        .problem()
        .unwrap()
}

#[test]
fn registered_mixed_networks_match_reference_roots_trees_and_singular_collections() {
    let mut comparisons = 0;
    let mut singular_comparisons = 0;
    let mut matched_tree_counts = 0;
    for case in controls() {
        let tree = case["tree"].as_bool().unwrap_or(false);
        let expected = case["expected"].as_f64().unwrap();
        // These retained optima are dyadic. The reference runner verifies
        // rational primal/dual certificates before publishing the fixture.
        let numerator = case["expected_exact"][0].as_i64().unwrap();
        let denominator = case["expected_exact"][1].as_u64().unwrap();
        assert!(denominator.is_power_of_two());
        assert_eq!(expected * denominator as f64, numerator as f64);
        let name = case["name"].as_str().unwrap();
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for (schedule, policy) in schedules() {
                let mut options = Options::default();
                options.cg.stabilize = false;
                options.cg.bound_mode = mode;
                options.network_pricing_schedule = schedule;
                let single = if case["model"]["networks"].as_array().unwrap().len() == 1 {
                    let mut one = case["model"].clone();
                    one["networks"] = one["networks"][0].clone();
                    Some(one)
                } else {
                    None
                };
                if tree {
                    let mut configured = ModelTreeOptions::default();
                    configured.node = options;
                    if case["strong_branching"] == true {
                        configured.branching = baldes::master::tree::NetworkBranchingPolicy::Phased;
                    }
                    let output =
                        baldes::master::tree::solve(parse(&case["model"]), configured.clone())
                            .unwrap();
                    assert_eq!(
                        output.status,
                        TreeStatus::Optimal,
                        "{name} mode={mode:?} policy={policy}"
                    );
                    assert!((output.incumbent - expected).abs() < 1e-8, "{name}");
                    assert!(
                        output.bound <= expected && expected - output.bound < 1e-8,
                        "{name}"
                    );
                    assert_eq!(output.open_nodes, 0);
                    // Sorted is an extra heuristic. Six cyclic reference
                    // tree controls time out; they do not supply a count.
                    if policy != "sorted" {
                        if let Some(count) = case["reference"][policy]["processed_nodes"].as_u64() {
                            assert_eq!(
                                output.processed_nodes, count as usize,
                                "{name} mode={mode:?} policy={policy}"
                            );
                            matched_tree_counts += 1;
                        }
                    }
                    if let Some(single) = single {
                        let singular =
                            baldes::master::tree::solve(parse(&single), configured).unwrap();
                        assert_eq!(singular.status, output.status);
                        assert_eq!(singular.incumbent, output.incumbent);
                        assert_eq!(singular.bound, output.bound);
                        assert_eq!(singular.oracle_calls, output.oracle_calls);
                        assert_eq!(singular.processed_nodes, output.processed_nodes);
                        singular_comparisons += 1;
                    }
                } else {
                    let output = model::solve(parse(&case["model"]), options.clone()).unwrap();
                    assert_eq!(
                        output.master.status,
                        Status::Optimal,
                        "{name} mode={mode:?} policy={policy}"
                    );
                    assert!((output.master.lp - expected).abs() < 1e-8, "{name}");
                    assert!(
                        output.master.bound <= expected && expected - output.master.bound < 1e-8,
                        "{name}"
                    );
                    if let Some(single) = single {
                        let singular = model::solve(parse(&single), options).unwrap();
                        assert_eq!(singular.master.status, output.master.status);
                        assert_eq!(singular.master.lp, output.master.lp);
                        assert_eq!(singular.master.bound, output.master.bound);
                        assert_eq!(singular.oracle_calls, output.oracle_calls);
                        assert_eq!(singular.master.iterations, output.master.iterations);
                        singular_comparisons += 1;
                    }
                }
                comparisons += 1;
            }
        }
    }
    assert_eq!(comparisons, 540);
    assert_eq!(singular_comparisons, 180);
    assert_eq!(matched_tree_counts, 126);
}

#[test]
fn exact_selection_constraints_close_the_original_cost_bound() {
    let model: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/multi_network_metasolver_selection.json"
    ))
    .unwrap();
    // The apparent -20 chain has no admissible selection value. The direct
    // path is -10 in the actual registered reference model and source domain.
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for (schedule, _) in schedules() {
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = mode;
            options.network_pricing_schedule = schedule;
            let output = model::solve(parse(&model), options).unwrap();
            assert_eq!(
                output.master.status,
                Status::Optimal,
                "mode={mode:?} schedule={schedule:?}"
            );
            assert_eq!(output.master.lp, -10.);
            assert_eq!(output.master.bound, -10.);
        }
    }
}
