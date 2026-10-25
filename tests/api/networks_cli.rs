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
        .arg("multi-pricing")
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
fn network_schedule_flags_keep_root_and_tree_objectives() {
    for policy in [
        "full-sweep",
        "cyclic",
        "sorted",
        "native-promising-first",
        "native-unvisited-first",
    ] {
        let root = solve(&["--network-pricing-schedule", policy]);
        assert_eq!(root["status"], "Optimal");
        assert_eq!(root["lp"], 1.5);
        let tree = solve(&["--tree", "--network-pricing-schedule", policy]);
        assert_eq!(tree["status"], "Optimal");
        assert_eq!(tree["incumbent"], 2.);
    }
    assert!(!command()
        .arg("multi-pricing")
        .arg(fixture())
        .args(["--network-pricing-schedule", "unknown"])
        .output()
        .unwrap()
        .status
        .success());
}

#[test]
fn proximal_flags_use_original_objective_and_validate_configuration() {
    for flag in ["--automatic-curvature", "--automatic-curvature-five-piece"] {
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
            .arg("multi-pricing")
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
        assert_eq!(tree["nodes"], 3);
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
        .arg("networks")
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
        .arg("networks")
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
            .arg("networks")
            .arg(fixture())
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(!output.stderr.is_empty());
    }
    assert!(command()
        .args(["networks", "--help"])
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
        let output = command().arg("networks").arg(&path).output().unwrap();
        assert!(!output.status.success(), "{value}");
        assert!(!output.stderr.is_empty());
        let zero = command()
            .arg("networks")
            .arg(&path)
            .args(["--tree", "--nodes", "0"])
            .output()
            .unwrap();
        assert!(!zero.status.success(), "{value}");
    }
    std::fs::write(&path, "{not-json").unwrap();
    assert!(!command()
        .arg("networks")
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
        .arg("networks")
        .arg(fixture())
        .arg("--enumerated-mip")
        .output()
        .unwrap();
    assert!(!output.status.success()); // --tree is required.
    #[cfg(not(feature = "highs"))]
    assert!(!command()
        .arg("networks")
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
            .arg("multi-pricing")
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
                .arg("multi-pricing")
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
