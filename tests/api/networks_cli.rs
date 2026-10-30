#![cfg(feature = "clp")]
use serde_json::{json, Value};
use std::{path::PathBuf, process::Command};
fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/networks/triangle.json")
}
fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_baldes"))
}
fn solve(flags: &[&str]) -> Value {
    let output = command()
        .arg("solve")
        .arg(fixture())
        .args(flags)
        .args(["--json", "-"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn signed_secondary_disposable_resource_is_supported_through_json_cli() {
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/multi_network_native_signed_secondary_resource.json");
    let output = command()
        .arg("solve")
        .arg(model)
        .args(["--no-stabilization", "--json", "-"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["status"], "Optimal");
    assert_eq!(result["lp"], -1.);
    assert_eq!(result["bound"], -1.);
    assert_eq!(result["oracle_calls"][0], json!([7, 3]));
    let route = result["routes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|route| route["value"].as_f64().unwrap_or_default() > 0.5)
        .unwrap();
    assert_eq!(route["resource_profile"][0][1], 5.);
    assert_eq!(route["resource_profile"][1][1], 3.);
    assert_eq!(route["resource_profile"][2][1], 7.);
}

#[test]
fn network_schedule_flags_keep_root_and_tree_objectives() {
    for policy in [
        "full-sweep",
        "cyclic",
        "sorted",
        "promising-first",
        "unvisited-first",
    ] {
        let root = solve(&["--network-pricing-schedule", policy]);
        assert_eq!(root["status"], "Optimal");
        assert_eq!(root["lp"], 1.5);
        let tree = solve(&["--tree", "--network-pricing-schedule", policy]);
        assert_eq!(tree["status"], "Optimal");
        assert_eq!(tree["incumbent"], 2.);
    }
    assert!(!command()
        .arg("solve")
        .arg(fixture())
        .args(["--network-pricing-schedule", "unknown"])
        .output()
        .unwrap()
        .status
        .success());
}

#[test]
fn proximal_flags_use_original_objective_and_validate_configuration() {
    for flag in [
        "--automatic-curvature",
        "--automatic-curvature-five-piece",
        "--proximal-finite-box-step",
        "--proximal-explicit-three-piece",
        "--proximal-explicit-five-piece",
        "--proximal-explicit-multipoint",
    ] {
        let root = solve(&[flag]);
        assert_eq!(root["status"], "Optimal");
        assert_eq!(root["lp"], 1.5);
        let tree = solve(&[flag, "--tree"]);
        assert_eq!(tree["incumbent"], 2.);
    }
    for flags in [
        vec!["--proximal-width", "0.5"],
        vec!["--proximal-width", "0.5", "--proximal-penalty", "1"],
    ] {
        let result = solve(&flags);
        assert_eq!(result["status"], "Optimal");
        assert_eq!(result["lp"], 1.5);
        assert!(result["bound"].as_f64().unwrap() <= 1.5);
    }
    for flags in [
        vec!["--proximal-penalty", "1"],
        vec!["--proximal-width", "-1"],
        vec!["--network-unpromising-limit", "0"],
    ] {
        assert!(!command()
            .arg("solve")
            .arg(fixture())
            .args(flags)
            .output()
            .unwrap()
            .status
            .success());
    }
}
#[test]
fn root_and_assignment_tree_use_all_networks() {
    for mode in ["numerical", "certified"] {
        let root = solve(&["--bound-mode", mode, "--no-stabilization"]);
        assert_eq!(root["status"], "Optimal");
        assert_eq!(root["mode"], "root");
        assert!((root["lp"].as_f64().unwrap() - 1.5).abs() < 1e-8);
        assert_eq!(root["bound_mode"], mode);
        assert_eq!(root["oracle_calls"].as_array().unwrap().len(), 2);
        assert_eq!(root["oracle_calls"][0][0], 7);
        assert_eq!(root["oracle_calls"][1][0], 42);
        for network in [7, 42] {
            assert!(root["routes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["network"] == network && r["value"].as_f64().unwrap() > 0.));
        }
        let tree = solve(&["--tree", "--bound-mode", mode]);
        assert_eq!(tree["status"], "Optimal");
        assert_eq!(tree["incumbent"], 2.);
        assert_eq!(tree["bound"], 2.);
        assert_eq!(tree["open_nodes"], 0);
        assert_eq!(tree["nodes"], 6);
        assert_eq!(tree["tree_processed_nodes"], 7);
        assert!(tree["pricing_calls"].as_u64().unwrap() > 0);
        assert!(tree["oracle_calls"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c[1].as_u64().unwrap() > 0));
        assert!(tree["lp"].is_null());
        assert!(tree["routes"].as_array().unwrap().iter().all(|r| r["value"]
            .as_f64()
            .unwrap()
            .fract()
            == 0.));
    }
}
#[test]
fn budgets_keep_unknown_bounds_and_open_nodes_explicit() {
    let zero = solve(&["--tree", "--nodes", "0"]);
    assert_eq!(zero["status"], "NodeLimit");
    assert!(zero["incumbent"].is_null() && zero["bound"].is_null());
    assert_eq!(zero["open_nodes"], 1);
    let one = solve(&["--tree", "--nodes", "1"]);
    assert_eq!(one["status"], "NodeLimit");
    assert_eq!(one["bound"], 1.5);
    assert_eq!(one["open_nodes"], 2);
    assert!(one["incumbent"].is_null());
    assert_eq!(solve(&["--tree", "--time", "0"])["status"], "TimeLimit");
}

#[test]
fn native_node_limit_boundaries_match_all_four_retained_controls() {
    let observations: Value = serde_json::from_str(include_str!(
        "../fixtures/multi_network_native_node_limits.json"
    ))
    .unwrap();
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/multi_network_native_tree.json");
    for case in observations["cases"].as_array().unwrap() {
        let output = command()
            .arg("solve")
            .arg(&model)
            .args([
                "--tree",
                "--network-branching",
                "multiplicity-first",
                "--proximal-explicit-five-piece",
                "--network-tree-search",
                case["policy"].as_str().unwrap(),
                "--nodes",
                &case["limit"].to_string(),
                "--json",
                "-",
            ])
            .output()
            .unwrap();
        assert!(output.status.success(), "{:?}", output.stderr);
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["tree_processed_nodes"], case["processed"], "{case}");
        assert_eq!(result["nodes"], case["evaluated"], "{case}");
        assert_eq!(result["status"], case["status"], "{case}");
        assert_eq!(result["incumbent"], case["incumbent"], "{case}");
        if case["bound"].is_null() {
            assert!(result["bound"].is_null(), "{case}");
        } else {
            assert!(
                (result["bound"].as_f64().unwrap() - case["bound"].as_f64().unwrap()).abs() < 1e-8,
                "{case}"
            );
        }
    }
}

#[test]
fn explicit_network_family_order_preserves_the_integer_optimum() {
    for policy in ["resource-first", "multiplicity-first"] {
        let tree = solve(&[
            "--tree",
            "--network-branching",
            policy,
            "--no-stabilization",
        ]);
        assert_eq!(tree["status"], "Optimal");
        assert_eq!(tree["incumbent"], 2.);
        assert_eq!(tree["bound"], 2.);
        assert_eq!(tree["open_nodes"], 0);
        assert!(tree["oracle_calls"]
            .as_array()
            .unwrap()
            .iter()
            .all(|calls| calls[1].as_u64().unwrap() > 0));
    }
    assert!(!command()
        .arg("solve")
        .arg(fixture())
        .args(["--network-branching", "unknown"])
        .output()
        .unwrap()
        .status
        .success());
}
#[test]
fn writes_json_files_and_rejects_unknown_options() {
    let path =
        std::env::temp_dir().join(format!("baldes-network-output-{}.json", std::process::id()));
    let output = command()
        .arg("solve")
        .arg(fixture())
        .arg("--json")
        .arg(&path)
        .output()
        .unwrap();
    assert!(output.status.success());
    let result: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(result["lp"], 1.5);
    assert!(String::from_utf8_lossy(&output.stdout).contains("mode=root status=Optimal"));
    std::fs::remove_file(path).unwrap();
    for args in [
        vec!["--rank1"],
        vec!["--iterations", "0"],
        vec!["--time", "NaN"],
        vec!["--bound-mode", "invalid"],
        vec!["--option", "unknown=1"],
        vec!["--json"],
    ] {
        let output = command()
            .arg("solve")
            .arg(fixture())
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(!output.stderr.is_empty());
    }
    assert!(command()
        .args(["solve", "--help"])
        .output()
        .unwrap()
        .status
        .success());
}
#[test]
fn model_validation_rejects_bad_ids_rows_resources_and_unknown_fields() {
    let text = std::fs::read_to_string(fixture()).unwrap();
    let original: Value = serde_json::from_str(&text).unwrap();
    let mut invalid = vec![];
    let mut value = original.clone();
    value["networks"][1]["id"] = json!(7);
    invalid.push(value);
    let mut value = original.clone();
    value["networks"][0]["arcs"][0]["entries"] = json!([[99, 1]]);
    invalid.push(value);
    let mut value = original.clone();
    value["networks"][0]["arcs"][0]["consumption"] = json!([0, 0]);
    invalid.push(value);
    let mut value = original.clone();
    value["networkz"] = json!([]);
    invalid.push(value);
    let mut value = original.clone();
    value["networks"][0]["sink"] = json!(999);
    invalid.push(value);
    let path = std::env::temp_dir().join(format!(
        "baldes-invalid-network-{}.json",
        std::process::id()
    ));
    for value in invalid {
        std::fs::write(&path, value.to_string()).unwrap();
        let output = command().arg("solve").arg(&path).output().unwrap();
        assert!(!output.status.success(), "{value}");
        assert!(!output.stderr.is_empty());
        let zero = command()
            .arg("solve")
            .arg(&path)
            .args(["--tree", "--nodes", "0"])
            .output()
            .unwrap();
        assert!(!zero.status.success(), "{value}");
    }
    std::fs::write(&path, "{not-json").unwrap();
    assert!(!command()
        .arg("solve")
        .arg(&path)
        .output()
        .unwrap()
        .status
        .success());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn enumeration_reporting_is_shared_with_python_and_partial_transfer_falls_back() {
    let complete = solve(&["--enumerate"]);
    assert_eq!(complete["status"], "Optimal");
    assert_eq!(complete["enumeration_reports"][0]["complete"], true);
    assert_eq!(complete["enumeration_reports"][0]["routes"], 9);
    let partial = solve(&["--enumerate", "--enumeration-route-limit", "5"]);
    assert_eq!(partial["status"], "Optimal");
    assert_eq!(partial["enumeration_reports"][0]["complete"], false);
    assert_eq!(partial["enumeration_reports"][0]["transferred"], 0);
    let output = command()
        .arg("solve")
        .arg(fixture())
        .arg("--enumerated-mip")
        .output()
        .unwrap();
    assert!(!output.status.success()); // --tree is required.
    #[cfg(not(feature = "highs"))]
    assert!(!command()
        .arg("solve")
        .arg(fixture())
        .args(["--tree", "--enumerated-mip"])
        .output()
        .unwrap()
        .status
        .success());
    #[cfg(feature = "highs")]
    {
        let mip = solve(&["--tree", "--enumerated-mip"]);
        assert_eq!(mip["status"], "Optimal");
        assert_eq!(mip["incumbent"], 2.);
        assert_eq!(mip["nodes"], 1);
        assert_eq!(mip["enumerated_mip_calls"], 1);
    }
}

#[cfg(feature = "highs")]
#[test]
fn cutoff_pool_is_explicit_and_cannot_claim_original_infeasibility() {
    let tree = solve(&[
        "--tree",
        "--enumerated-mip",
        "--mip-cutoff",
        "1.75",
        "--enumeration-route-limit",
        "3",
        "--nodes",
        "1",
        "--no-stabilization",
    ]);
    assert_eq!(tree["status"], "NodeLimit");
    assert!(tree["incumbent"].is_null());
    assert!(tree["open_nodes"].as_u64().unwrap() > 0);
    assert_eq!(tree["enumerated_mip_calls"], 1);
    assert!(tree["enumeration_reports"]
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r["complete"] == true && r["coverage_cutoff"] == 1.75));
    for flags in [
        vec!["--mip-cutoff", "3"],
        vec!["--tree", "--enumerated-mip", "--mip-cutoff", "NaN"],
    ] {
        assert!(!command()
            .arg("solve")
            .arg(fixture())
            .args(flags)
            .output()
            .unwrap()
            .status
            .success());
    }
}

#[test]
fn resource_group_files_restrict_both_pricing_networks_and_tree_descendants() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for tree in [false, true] {
        for enumerate in [false, true] {
            let mut command = command();
            command
                .arg("solve")
                .arg(root.join("examples/networks/resource_groups.json"))
                .arg("--resource-branches")
                .arg(root.join("examples/networks/resource_group_upper.json"))
                .args([
                    "--bound-mode",
                    "certified",
                    "--no-stabilization",
                    "--json",
                    "-",
                ]);
            if tree {
                command.arg("--tree");
            }
            if enumerate {
                command.arg("--enumerate");
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let result: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(result["status"], "Optimal");
            assert_eq!(result["bound"], 4.);
            assert_eq!(result[if tree { "incumbent" } else { "lp" }], 4.);
            for route in result["routes"].as_array().unwrap() {
                if route["value"].as_f64().unwrap() <= 1e-6 {
                    continue;
                }
                let arcs = route["arcs"].as_array().unwrap();
                if arcs.contains(&json!(0)) || arcs.contains(&json!(2)) {
                    assert!(route["resource_profile"][1][0].as_f64().unwrap() >= 1.5 - 1e-6);
                }
            }
        }
    }
}

#[test]
fn native_depth_budgets_and_registered_aggregate_branches_preserve_objectives() {
    let root = solve(&[
        "--automatic-curvature",
        "--stabilization-max-tree-depth",
        "0",
        "--network-promising-limit",
        "2",
    ]);
    assert_eq!(root["lp"], 1.5);
    let mappings =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/networks/mapped_variables.json");
    for strong in [true, false] {
        let mut flags = vec![
            "--tree",
            "--network-branching",
            "phased",
            "--mapped-variables",
            mappings.to_str().unwrap(),
        ];
        if !strong {
            flags.push("--no-mapped-strong-branching");
        }
        let tree = solve(&flags);
        assert_eq!(tree["status"], "Optimal");
        assert_eq!(tree["incumbent"], 2.);
        assert_eq!(
            tree["strong_branching_trials"].as_u64().unwrap() > 0,
            strong
        );
    }
    assert!(!command()
        .arg("solve")
        .arg(fixture())
        .args(["--network-promising-limit", "0"])
        .output()
        .unwrap()
        .status
        .success());
}

#[test]
fn native_phased_registry_evaluates_original_fallback_candidates() {
    let tree = solve(&["--tree", "--network-branching", "phased"]);
    assert_eq!(tree["status"], "Optimal");
    assert_eq!(tree["incumbent"], 2.);
    assert!(tree["strong_branching_trials"].as_u64().unwrap() > 0);
}

#[test]
fn native_phased_trial_budget_and_prepared_children_match_retained_controls() {
    let cases: Value = serde_json::from_str(include_str!(
        "../fixtures/multi_network_native_strong_limits.json"
    ))
    .unwrap();
    for bound_mode in ["numerical", "certified"] {
        for case in cases.as_array().unwrap() {
            let limit = case["limit"].as_u64().unwrap().to_string();
            let tree = solve(&[
                "--tree",
                "--network-branching",
                "phased",
                "--nodes",
                &limit,
                "--bound-mode",
                bound_mode,
            ]);
            assert_eq!(
                tree["tree_processed_nodes"], case["processed"],
                "{bound_mode} {case}"
            );
            assert_eq!(tree["incumbent"], case["incumbent"], "{bound_mode} {case}");
            assert_eq!(tree["status"], case["status"], "{bound_mode} {case}");
            if let Some(bound) = case["bound"].as_f64() {
                assert!(
                    (tree["bound"].as_f64().unwrap() - bound).abs() < 1e-8,
                    "{bound_mode} {case}: {tree}"
                );
            } else {
                assert!(tree["bound"].is_null());
            }
            assert_eq!(
                tree["strong_branching_trials"], case["trials"],
                "{bound_mode} {case}"
            );
        }
    }
}

#[test]
fn native_depth_first_search_separates_processed_and_evaluated_nodes() {
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/multi_network_native_tree.json");
    let run = |selection: &str, open_limit: Option<&str>, rounding: Option<&str>| {
        let output = command()
            .arg("solve")
            .arg(&model)
            .args([
                "--tree",
                "--network-branching",
                "multiplicity-first",
                "--network-tree-search",
                selection,
                "--proximal-explicit-five-piece",
                "--json",
                "-",
            ])
            .args(
                rounding
                    .into_iter()
                    .flat_map(|mode| ["--network-strategy2-rounding", mode]),
            )
            .args(
                open_limit
                    .into_iter()
                    .flat_map(|limit| ["--network-open-node-limit", limit]),
            )
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice::<Value>(&output.stdout).unwrap()
    };
    let native = run("default", None, None);
    let overflow = run("default", Some("1"), None);
    let all_overflow = run("default", Some("0"), None);
    let best_bound = run("best-bound", None, None);
    let strategy2 = run("bound-priority", None, None);
    let strategy3 = run("depth-bound-priority", None, None);
    let strategy2_unrounded = run("bound-priority", None, Some("unrounded"));
    let strategy2_guarded = run("bound-priority", None, Some("guarded-tolerance"));
    let strategy2_exact_ceil = run("bound-priority", None, Some("exact-ceil"));
    assert_eq!(native["status"], "Optimal");
    assert_eq!(best_bound["status"], "Optimal");
    assert_eq!(native["incumbent"], 2.);
    assert_eq!(best_bound["incumbent"], 2.);
    assert_eq!(strategy2_exact_ceil["status"], "Optimal");
    assert_eq!(strategy2_exact_ceil["incumbent"], 2.);
    assert_eq!(overflow["status"], "Optimal");
    assert_eq!(overflow["incumbent"], 2.);
    assert!(
        overflow["tree_processed_nodes"].as_u64().unwrap() >= overflow["nodes"].as_u64().unwrap()
    );
    assert_eq!(all_overflow["status"], "Incomplete");
    assert!(all_overflow["incumbent"].is_null());
    assert_eq!(all_overflow["bound"], 1.5);
    assert_eq!(all_overflow["open_nodes"], 2);
    assert_eq!(all_overflow["tree_processed_nodes"], 1);
    for policy in ["best-bound", "bound-priority", "depth-bound-priority"] {
        let stopped = run(policy, Some("0"), None);
        assert_eq!(stopped["status"], "Incomplete");
        assert!(stopped["incumbent"].is_null());
        assert_eq!(stopped["bound"], 1.5);
        assert_eq!(stopped["open_nodes"], 2);
        assert_eq!(stopped["tree_processed_nodes"], 1);
    }
    assert_eq!(native["tree_processed_nodes"], 7);
    assert!(native["tree_processed_nodes"].as_u64().unwrap() > native["nodes"].as_u64().unwrap());
    assert_eq!(best_bound["nodes"], 5);
    assert!(
        best_bound["tree_processed_nodes"].as_u64().unwrap()
            >= best_bound["nodes"].as_u64().unwrap()
    );
    assert_eq!(strategy2["status"], "Optimal");
    assert_eq!(strategy2["incumbent"], 2.);
    assert_eq!(strategy3["status"], "Optimal");
    assert_eq!(strategy3["incumbent"], 2.);
    assert_eq!(strategy3["tree_processed_nodes"], 7);
    assert_eq!(strategy2["tree_processed_nodes"], 5);
    assert_eq!(strategy2["nodes"], 5);
    assert_eq!(strategy2_unrounded["tree_processed_nodes"], 5);
    assert_eq!(strategy2_guarded["tree_processed_nodes"], 7);
    assert_eq!(strategy2_guarded["nodes"], 6);
    let model: Value =
        serde_json::from_slice(&std::fs::read(&model).expect("read native comparison fixture"))
            .unwrap();
    for result in [&native, &best_bound, &strategy2, &strategy3] {
        let mut coverage = vec![0.; 3];
        let mut objective = 0.;
        for route in result["routes"].as_array().unwrap() {
            let network = model["networks"]
                .as_array()
                .unwrap()
                .iter()
                .find(|network| network["id"] == route["network"])
                .unwrap();
            let quantity = route["value"].as_f64().unwrap();
            objective += quantity * route["cost"].as_f64().unwrap();
            for arc_id in route["arcs"].as_array().unwrap() {
                if let Some(entries) =
                    network["arcs"][arc_id.as_u64().unwrap() as usize]["entries"].as_array()
                {
                    for entry in entries {
                        let row = entry[0].as_u64().unwrap() as usize;
                        coverage[row] += quantity * entry[1].as_f64().unwrap();
                    }
                }
            }
        }
        assert_eq!(objective, 2.);
        assert_eq!(coverage, vec![1., 1., 1.]);
    }
}

#[test]
fn explicitly_declared_front_program_domain_survives_original_cost_recovery() {
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/coupled/native_machine_network.json");
    let output = command()
        .arg("solve")
        .arg(model)
        .args([
            "--tree",
            "--bound-mode",
            "certified",
            "--no-stabilization",
            "--json",
            "-",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["status"], "Optimal");
    assert_eq!(result["incumbent"], 1003.);
    assert_eq!(result["bound"], 1003.);
}

#[test]
fn bounded_walk_json_preserves_repeated_visits_and_negative_cycle_cost() {
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/coupled/bounded_walk_network.json");
    let output = command()
        .arg("solve")
        .arg(model)
        .args([
            "--tree",
            "--bound-mode",
            "certified",
            "--no-stabilization",
            "--json",
            "-",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["status"], "Optimal");
    assert_eq!(result["incumbent"], -3.);
    assert_eq!(result["bound"], -3.);
}

#[test]
fn native_scoring_gap_controls_are_finite_nonnegative_and_keep_original_optimum() {
    for gap in ["0", "0.000001", "0.4"] {
        let tree = solve(&[
            "--tree",
            "--network-branching",
            "phased",
            "--network-scoring-gap-tolerance",
            gap,
        ]);
        assert_eq!(tree["status"], "Optimal");
        assert_eq!(tree["incumbent"], 2.);
        assert_eq!(tree["bound"], 2.);
    }
    for gap in ["-1", "NaN", "inf"] {
        let output = command()
            .arg("solve")
            .arg(fixture())
            .args(["--tree", "--network-scoring-gap-tolerance", gap])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("scoring gap tolerance"));
    }
}

#[test]
fn exact_infeasible_trial_is_not_resolved_or_left_open_at_native_budget_boundaries() {
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/multi_network_native_infeasible_trial.json");
    let cases: Value = serde_json::from_str(include_str!(
        "../fixtures/multi_network_native_infeasible_trial_limits.json"
    ))
    .unwrap();
    for mode in ["numerical", "certified"] {
        for case in cases.as_array().unwrap() {
            let limit = case["limit"].as_u64().unwrap().to_string();
            let output = command()
                .arg("solve")
                .arg(&model)
                .args([
                    "--tree",
                    "--network-branching",
                    "phased",
                    "--network-tree-search",
                    case["policy"].as_str().unwrap(),
                    "--nodes",
                    &limit,
                    "--bound-mode",
                    mode,
                    "--json",
                    "-",
                ])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let result: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(result["status"], case["status"], "{mode} {case}: {result}");
            assert_eq!(
                result["tree_processed_nodes"], case["processed"],
                "{mode} {case}"
            );
            assert_eq!(result["incumbent"], case["incumbent"], "{mode} {case}");
            if let Some(bound) = case["bound"].as_f64() {
                assert!(
                    (result["bound"].as_f64().unwrap() - bound).abs() < 1e-8,
                    "{mode} {case}: {result}"
                );
            } else {
                assert!(result["bound"].is_null());
            }
            if case["limit"].as_u64().unwrap() > 0 {
                assert_eq!(result["nodes"], 1, "{mode} {case}");
            }
        }
    }
}

#[test]
fn mapped_phase_continuation_separates_native_screens_from_processed_exact_nodes() {
    let mappings = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/multi_network_native_phased_mappings.json");
    for mode in ["numerical", "certified"] {
        let output = command()
            .arg("solve")
            .arg(fixture())
            .args([
                "--tree",
                "--network-branching",
                "resource-first",
                "--mapped-variables",
            ])
            .arg(&mappings)
            .args(["--bound-mode", mode, "--no-stabilization", "--json", "-"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["status"], "Optimal");
        assert_eq!(result["incumbent"], 2.);
        assert_eq!(result["bound"], 2.);
        // Native assignment control: 4 heuristic screens + 2 exact trials;
        // NodeTreat=7 includes the root, NodeProc=3 excludes the screens.
        assert_eq!(result["strong_branching_trials"], 6);
        assert_eq!(result["tree_processed_nodes"], 3);
        assert_eq!(result["nodes"], 1);
        assert_eq!(result["iterations"], 18);
        let calls: u64 = result["oracle_calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry[1].as_u64().unwrap())
            .sum();
        // This fixture deliberately puts each whole assignment on incoming
        // arcs. Nonnegative box admission leaves only two physical entries;
        // native accounting36 is not a count of actual solver invocations.
        // The incident-arc paired control has its own integration regression.
        assert_eq!(calls, 2);
    }
}

#[test]
fn shared_inactive_tree_paths_keep_both_original_bound_modes() {
    for mode in ["numerical", "certified"] {
        let result = solve(&[
            "--tree",
            "--network-share-inactive-columns",
            "--bound-mode",
            mode,
        ]);
        assert_eq!(result["status"], "Optimal");
        assert_eq!(result["incumbent"], 2.);
        assert!((result["bound"].as_f64().unwrap() - 2.).abs() < 1e-8);
    }
    let output = command()
        .arg("solve")
        .arg(fixture())
        .args(["--network-share-inactive-columns", "--json", "-"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires tree=true"));
}

#[test]
fn native_packing_file_registers_original_assignment_branching() {
    let packing = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/multi_network_native_packing.json");
    for mode in ["numerical", "certified"] {
        let result = solve(&[
            "--tree",
            "--packing",
            packing.to_str().unwrap(),
            "--no-stabilization",
            "--bound-mode",
            mode,
        ]);
        assert_eq!(result["status"], "Optimal");
        assert_eq!(result["incumbent"], 2.);
        assert_eq!(result["bound"], 2.);
        assert_eq!(result["tree_processed_nodes"], 3);
        assert_eq!(result["strong_branching_trials"], 6);
    }
    let invalid = command()
        .arg("solve")
        .arg(fixture())
        .args(["--packing", packing.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("requires tree=true"));
}

#[test]
fn named_native_branch_direction_controls_trial_order_through_cli() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let packing = fixtures.join("multi_network_native_packing.json");
    let directions = fixtures.join("multi_network_native_branch_directions.json");
    let result = solve(&[
        "--tree",
        "--no-stabilization",
        "--packing",
        packing.to_str().unwrap(),
        "--network-branch-directions",
        directions.to_str().unwrap(),
    ]);
    assert_eq!(result["status"], "Optimal");
    assert_eq!(result["incumbent"], 2.);
    assert_eq!(result["bound"], 2.);
    let evaluations = result["mapped_branch_evaluations"].as_array().unwrap();
    assert!(!evaluations.is_empty());
    let mut observed = std::collections::HashSet::new();
    for evaluation in evaluations {
        let key = (
            evaluation["name"].as_str().unwrap(),
            evaluation["stage"].as_u64().unwrap(),
        );
        if observed.insert(key) {
            assert_eq!(evaluation["side"], 0, "lower side must be evaluated first");
        }
    }
    let root = command()
        .arg("solve")
        .arg(fixture())
        .args(["--network-branch-directions", directions.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!root.status.success());
    assert!(String::from_utf8_lossy(&root.stderr).contains("requires tree=true"));
}

#[test]
fn original_variables_compile_once_and_report_root_and_tree_activities() {
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/multi_network_original_variables.json");
    for tree in [false, true] {
        let mut cmd = command();
        cmd.arg("solve")
            .arg(&model)
            .args(["--no-stabilization", "--json", "-"]);
        if tree {
            cmd.arg("--tree");
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value[if tree { "incumbent" } else { "lp" }], -1.);
        assert_eq!(
            value["original_variables"],
            json!([{"network":7,"id":0,"lower":1.,"upper":1.}])
        );
    }
}

#[test]
fn original_variables_preserve_original_cost_space_with_implicit_or_explicit_boxes() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for mode in ["numerical", "certified"] {
        for explicit in [false, true] {
            let mut cli = command();
            cli.arg("solve")
                .arg(fixtures.join("multi_network_original_cancellation.json"));
            if explicit {
                cli.arg("--original-variable-boxes")
                    .arg(fixtures.join("multi_network_original_cancellation_boxes.json"));
            }
            let output = cli
                .args(["--no-stabilization", "--bound-mode", mode, "--json", "-"])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let result: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(result["lp"], 0.);
            assert_eq!(result["bound"], 0.);
            assert_eq!(result["oracle_calls"], json!([[7, 3]]));
        }
    }
}

#[test]
fn solve_normalizes_scalar_and_array_pricing_and_removes_old_command_aliases() {
    let network = json!({"id":7,"vertices":3,"multiplicity":[0,1],"arcs":[{"tail":0,"head":1,"cost":-2},{"tail":1,"head":2,"cost":0}]});
    let path =
        std::env::temp_dir().join(format!("baldes-pricing-shape-{}.json", std::process::id()));
    let mut results = vec![];
    for input in [
        network.clone(),
        json!([network.clone()]),
        json!({"networks":network.clone()}),
        json!({"networks":[network]}),
    ] {
        std::fs::write(&path, input.to_string()).unwrap();
        let output = command()
            .arg("solve")
            .arg(&path)
            .args(["--no-stabilization", "--json", "-"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let data: Value = serde_json::from_slice(&output.stdout).unwrap();
        results.push((
            data["lp"].clone(),
            data["bound"].clone(),
            data["oracle_calls"].clone(),
            data["routes"].clone(),
        ));
    }
    std::fs::remove_file(path).unwrap();
    assert!(results.windows(2).all(|r| r[0] == r[1]));
    assert_eq!(results[0].0, json!(-2.));
    for command_name in ["multi-pricing", "networks", "vrptw", "cvrp"] {
        let output = command().args([command_name, "--help"]).output().unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("unknown command"));
    }
}

#[test]
fn registered_accumulated_variables_share_cli_object_array_root_and_tree_paths() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../fixtures/metasolver_registered_activities.json"
    ))
    .unwrap();
    let directory =
        std::env::temp_dir().join(format!("baldes-activity-vars-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let mut compared = 0;
    for case in corpus["cases"].as_array().unwrap() {
        let activities = case["activity_values_by_path"].as_array().unwrap();
        let lower = activities
            .iter()
            .map(|v| v.as_f64().unwrap())
            .fold(f64::INFINITY, f64::min);
        let upper = activities
            .iter()
            .map(|v| v.as_f64().unwrap())
            .fold(f64::NEG_INFINITY, f64::max);
        let boxes = directory.join("activity-boxes.json");
        std::fs::write(
            &boxes,
            serde_json::to_vec(&serde_json::json!([
                {"network": 42, "id": 37, "lower": lower, "upper": upper}
            ]))
            .unwrap(),
        )
        .unwrap();
        for singular in [false, true] {
            let mut model = case["model"].clone();
            model["pricing_variables"][2]["integer"] = true.into();
            if singular {
                model["networks"] = model["networks"][0].clone();
            }
            let path = directory.join(format!(
                "{}-{singular}.json",
                case["name"].as_str().unwrap()
            ));
            std::fs::write(&path, serde_json::to_vec(&model).unwrap()).unwrap();
            for mode in ["numerical", "certified"] {
                for tree in [false, true] {
                    let mut cmd = command();
                    cmd.arg("solve").arg(&path).args([
                        "--no-stabilization",
                        "--bound-mode",
                        mode,
                        "--json",
                        "-",
                    ]);
                    cmd.arg("--original-variable-boxes").arg(&boxes);
                    if tree {
                        cmd.arg("--tree");
                    }
                    let run = cmd.output().unwrap();
                    assert!(
                        run.status.success(),
                        "{}",
                        String::from_utf8_lossy(&run.stderr)
                    );
                    let value: Value = serde_json::from_slice(&run.stdout).unwrap();
                    assert_eq!(value["status"], "Optimal", "{} {value}", case["name"]);
                    let expected = if tree && !case["row"].is_null() {
                        if case["kind"] == "custom" {
                            -9.
                        } else {
                            -10.
                        }
                    } else {
                        case["oracle"]["objective_float"].as_f64().unwrap()
                    };
                    assert!(
                        (value[if tree { "incumbent" } else { "lp" }]
                            .as_f64()
                            .unwrap()
                            - expected)
                            .abs()
                            < 1e-7
                    );
                    let activity = value["original_variables"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|v| v["network"] == 42 && v["id"] == 37)
                        .unwrap();
                    let expected_activity = if tree && !case["row"].is_null() {
                        case["activity_values_by_path"][0].as_f64().unwrap()
                    } else {
                        let mut parts = case["oracle"]["fraction_first_path"]
                            .as_str()
                            .unwrap()
                            .split('/');
                        let theta = parts.next().unwrap().parse::<f64>().unwrap()
                            / parts
                                .next()
                                .map(|p| p.parse::<f64>().unwrap())
                                .unwrap_or(1.);
                        theta * case["activity_values_by_path"][0].as_f64().unwrap()
                            + (1. - theta) * case["activity_values_by_path"][1].as_f64().unwrap()
                    };
                    assert!((activity["lower"].as_f64().unwrap() - expected_activity).abs() < 1e-7);
                    assert!((activity["upper"].as_f64().unwrap() - expected_activity).abs() < 1e-7);
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, 64);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn paired_front_signed_owner_cost_is_reported_without_an_uncertified_bound_by_cli() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut model = corpus["cases"][0]["model"].clone();
    model["networks"][0]["coupled_objective_coefficients"] = json!([[0, -1.]]);
    let path = std::env::temp_dir().join(format!(
        "baldes-negative-paired-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, model.to_string()).unwrap();
    let output = command()
        .arg("solve")
        .arg(&path)
        .args(["--no-stabilization", "--json", "-"])
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["status"], "Stalled");
    assert_eq!(result["lp"], -23.);
    assert!(result["bound"].is_null());
    let route = result["routes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|route| route["value"].as_f64().unwrap() > 0.)
        .unwrap();
    assert_eq!(route["cost"], -23.);
    assert_eq!(route["coupled_values"], json!([[0, 3.]]));
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("raw pricing cannot certify this owner optimum"));
}
