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
        .arg("networks")
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
