#![cfg(feature = "clp")]

use std::process::Command;

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_baldes"))
}

#[test]
fn help_and_invalid_instance_types() {
    for args in [
        vec!["--help"],
        vec!["vrptw", "--help"],
        vec!["cvrp", "--help"],
        vec!["hgs", "--help"],
    ] {
        let output = command().args(args).output().unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("baldes <vrptw|cvrp|hgs>"));
    }
    for args in [vec![], vec!["evrp"], vec!["vrptw"], vec!["cvrp"], vec!["hgs"]] {
        let output = command().args(args).output().unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("Usage:"));
    }
}

#[test]
fn cvrp_ignores_time_windows_while_vrptw_enforces_them() {
    let path = std::env::temp_dir().join(format!("baldes-cli-{}.txt", std::process::id()));
    // Each customer is reachable individually, but the time windows prevent a
    // combined VRPTW route. A CVRP route can visit both within capacity.
    std::fs::write(
        &path,
        "tiny\nVEHICLE\nNUMBER CAPACITY\n2 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 1.1 0\n2 0 1 1 0 1.1 0\n",
    )
    .unwrap();
    for (kind, expected) in [("vrptw", 4.), ("cvrp", 2. + 2_f64.sqrt())] {
        let output = command()
            .arg(kind)
            .arg(&path)
            .args(["--no-enumeration", "--no-stabilization"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        let lp: f64 = stdout
            .split_whitespace()
            .find_map(|word| word.strip_prefix("lp="))
            .expect("root objective")
            .parse()
            .unwrap();
        assert!((lp - expected).abs() < 1e-8, "{kind}: {stdout}");
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn hgs_respects_time_windows() {
    let path = std::env::temp_dir().join(format!("baldes-cli-hgs-{}.txt", std::process::id()));
    // Same instance: the time windows force one route per customer.
    std::fs::write(
        &path,
        "tiny\nVEHICLE\nNUMBER CAPACITY\n2 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 1.1 0\n2 0 1 1 0 1.1 0\n",
    )
    .unwrap();
    let output = command().arg("hgs").arg(&path).args(["--time", "0.2"]).output().unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("status=Found cost=4.00"), "{stdout}");
}

#[test]
fn tree_and_kohl_controls_reach_integer_solve_and_reject_cvrp_time_resource() {
    let output = command().args(["vrptw", "instances/R101.txt", "--customers", "10",
        "--tree", "--two-path-cuts", "--time", "10"]).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("status=Optimal incumbent=269.533140900"), "{stdout}");
    let output = command().args(["cvrp", "instances/R101.txt", "--two-path-cuts"]).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("require vrptw time windows"));
}

#[test]
fn tree_branching_mode_is_explicit_and_validated() {
    let path=std::env::temp_dir().join(format!("baldes-branching-cli-{}.txt",std::process::id()));
    std::fs::write(&path,"triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    for mode in ["native","vehicles","ryan-foster","arc"] {
        let output=command().arg("vrptw").arg(&path).args(["--tree","--branching",mode,"--no-stabilization"]).output().unwrap();
        assert!(output.status.success(),"{mode}: {}",String::from_utf8_lossy(&output.stderr));
        assert!(String::from_utf8_lossy(&output.stdout).contains("Optimal"));
    }
    let output=command().arg("vrptw").arg(&path).args(["--tree","--branching","invalid"]).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown branching mode"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn enumerated_mip_flag_requires_tree_and_uses_the_available_backend() {
    let path=std::env::temp_dir().join(format!("baldes-enumerated-mip-{}.txt",std::process::id()));
    std::fs::write(&path,"triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let invalid=command().arg("vrptw").arg(&path).arg("--enumerated-mip").output().unwrap();
    assert!(!invalid.status.success());assert!(String::from_utf8_lossy(&invalid.stderr).contains("requires --tree"));
    let json=path.with_extension("json");
    let result=command().arg("vrptw").arg(&path).args(["--tree","--enumerated-mip","--no-strong-branching","--no-diving","--no-mip-heuristic","--no-stabilization","--json"]).arg(&json).output().unwrap();
    if cfg!(feature="highs") {
        assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
        assert!(std::fs::read_to_string(&json).unwrap().contains("\"enumerated_mip_calls\":1"));
    } else {
        assert!(!result.status.success());assert!(String::from_utf8_lossy(&result.stderr).contains("requires the highs feature"));
    }
    let _=std::fs::remove_file(json);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn canonical_and_legacy_option_names_reach_the_same_root_and_tree_controls() {
    for prefix in ["BALDES", "RCSP"] {
        for tree in [false, true] {
            let mut args = vec!["vrptw".to_owned(), "instances/R101.txt".to_owned(),
                "--customers".to_owned(), "4".to_owned(), "--time".to_owned(), "10".to_owned(),
                "--option".to_owned(), format!("{prefix}labelDominanceStrategy=1"),
                "--option".to_owned(), format!("{prefix}hardTimeThresholdInPricing=1")];
            if tree { args.push("--tree".to_owned()); }
            let output = command().args(&args).output().unwrap();
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            assert!(String::from_utf8_lossy(&output.stdout).contains("status=Optimal"));
            args.extend(["--option".to_owned(), format!("{prefix}labelDominanceStrategy=2")]);
            let invalid = command().args(&args).output().unwrap();
            assert!(!invalid.status.success());
            assert!(String::from_utf8_lossy(&invalid.stderr).contains("unsupported label dominance strategy"));
        }
    }
}
