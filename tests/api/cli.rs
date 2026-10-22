#![cfg(feature = "clp")]

use std::process::Command;

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_baldes"))
}

#[test]
fn rounding_defaults_to_tenths_and_none_preserves_euclidean_distances() {
    let file = std::env::temp_dir().join(format!("baldes-rounding-cli-{}.txt", std::process::id()));
    let json = file.with_extension("json");
    std::fs::write(&file, "rounding\nVEHICLE\nNUMBER CAPACITY\n2 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n").unwrap();
    for (kind, flags, key) in [
        ("ce", vec![], "lp_bound"),
        ("pstep", vec!["--p", "2"], "lp_bound"),
        ("hgs", vec!["--nb-iter", "1", "--time", "0.05"], "cost"),
    ] {
        for (mode, expected) in [
            (None, 3.4),
            (Some("tenth"), 3.4),
            (Some("none"), 2. + 2_f64.sqrt()),
        ] {
            let mut cmd = command();
            cmd.arg(kind)
                .arg(&file)
                .args(&flags)
                .arg("--json")
                .arg(&json);
            if let Some(mode) = mode {
                cmd.args(["--rounding", mode]);
            }
            let output = cmd.output().unwrap();
            assert!(
                output.status.success(),
                "{kind}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let report: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
            let cost = report[key].as_f64().expect("solution cost");
            assert!((cost - expected).abs() < 1e-7, "{kind} {mode:?}: {report}");
        }
    }
    std::fs::remove_file(file).unwrap();
    std::fs::remove_file(json).unwrap();
}

#[test]
fn ce_command_solves_route_lp_and_reports_limits() {
    let file = std::env::temp_dir().join(format!("baldes-ce-cli-{}.txt", std::process::id()));
    let json = file.with_extension("json");
    std::fs::write(&file, "ce-cli\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 1 0 1 0 100 0\n3 1 0 1 0 100 0\n").unwrap();
    let output = command()
        .arg("ce")
        .arg(&file)
        .arg("--json")
        .arg(&json)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    assert_eq!(report["status"], "Converged");
    assert_eq!(report["relaxation"], "route_lp");
    assert!((report["lp_bound"].as_f64().unwrap() - 3.0).abs() < 1e-6);
    assert_eq!(report["max_vehicles"], 3.0);
    assert!(!report["routes"].as_array().unwrap().is_empty());
    assert!(String::from_utf8_lossy(&output.stdout).contains("lower_bound=3.000000"));

    let output = command()
        .arg("ce")
        .arg(&file)
        .args([
            "--customers",
            "2",
            "--rounding",
            "nearest",
            "--unlimited-fleet",
            "--json",
        ])
        .arg(&json)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    assert_eq!(report["customers"], 2);
    assert_eq!(report["status"], "Converged");
    assert!(report["max_vehicles"].is_null());
    assert!((report["lp_bound"].as_f64().unwrap() - 2.0).abs() < 1e-6);

    // The stronger initial diagram already solves the three-customer fixture.
    // This four-customer case still needs refinement and exercises its budget.
    let budget_file = file.with_extension("budget.txt");
    std::fs::write(&budget_file,"ce-limit\nVEHICLE\nNUMBER CAPACITY\n4 3\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 10000 0\n1 12 13 1 0 10000 0\n2 1 8 1 0 10000 0\n3 16 15 1 0 10000 0\n4 12 9 1 0 10000 0\n").unwrap();
    for (flag, value, status) in [
        ("--time", "0", "TimeLimit"),
        ("--iterations", "0", "IterationLimit"),
        ("--nodes", "18", "DiagramLimit"),
    ] {
        let output = command()
            .arg("ce")
            .arg(if status == "DiagramLimit" {
                &budget_file
            } else {
                &file
            })
            .args([flag, value, "--json"])
            .arg(&json)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
        assert_eq!(report["status"], status);
        assert!(report["lp_bound"].is_null());
        assert!(report["routes"].as_array().unwrap().is_empty());
        if status == "DiagramLimit" {
            assert!(report["refinements"].as_u64().unwrap() > 0);
        }
    }
    std::fs::remove_file(budget_file).unwrap();
    std::fs::remove_file(file).unwrap();
    std::fs::remove_file(json).unwrap();
}

#[test]
fn ce_help_and_option_validation() {
    let output = command().args(["ce", "--help"]).output().unwrap();
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("Usage: baldes ce INSTANCE"));
    assert!(help.contains("--unlimited-fleet"));
    let output = command().arg("--help").output().unwrap();
    assert!(String::from_utf8_lossy(&output.stdout).contains("ce"));
    for args in [
        vec!["ce"],
        vec!["ce", "missing", "--time"],
        vec!["ce", "missing", "--time", "NaN"],
        vec!["ce", "missing", "--time", "-1"],
        vec!["ce", "missing", "--nodes", "bad"],
        vec!["ce", "missing", "--unknown"],
        vec!["ce", "missing", "--rounding", "bad"],
        vec!["ce", "missing", "--log-interval", "0"],
    ] {
        let output = command().args(&args).output().unwrap();
        assert!(!output.status.success(), "{args:?}");
        assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
    }
}

#[test]
fn ce_vrptw_windows_and_progress_logging() {
    let file = std::env::temp_dir().join(format!("baldes-ce-vrptw-{}.txt", std::process::id()));
    let json = file.with_extension("json");
    // Same location and simultaneous windows: service duration prevents a
    // shared route, although the corresponding CVRP can use just one vehicle.
    std::fs::write(&file, "ce-vrptw\nVEHICLE\nNUMBER CAPACITY\n2 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 1 1 1\n2 1 0 1 1 1 1\n").unwrap();
    let mut expected_stdout = None;
    for flags in [
        vec![],
        vec!["--log"],
        vec!["--log-interval", "2"],
        vec!["-v"],
        vec!["--verbose"],
    ] {
        let output = command()
            .arg("ce")
            .arg(&file)
            .args(&flags)
            .arg("--json")
            .arg(&json)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
        assert_eq!(report["status"], "Converged");
        assert!((report["lp_bound"].as_f64().unwrap() - 4.0).abs() < 1e-6);
        assert_eq!(report["routes"].as_array().unwrap().len(), 2);
        // Remove nondeterministic wall time before comparing stdout.
        let stdout = String::from_utf8_lossy(&output.stdout)
            .split(" elapsed=")
            .next()
            .unwrap()
            .to_owned();
        if let Some(expected) = &expected_stdout {
            assert_eq!(&stdout, expected);
        } else {
            expected_stdout = Some(stdout);
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        if flags.is_empty() {
            assert!(stderr.is_empty(), "{stderr}");
        } else {
            assert!(
                stderr.contains("Lower bound") && stderr.contains("Infeasible"),
                "{stderr}"
            );
            assert!(stderr.contains("CE Converged"), "{stderr}");
            let rows: Vec<_> = stderr
                .lines()
                .filter(|line| {
                    line.split_whitespace()
                        .next()
                        .is_some_and(|v| v.parse::<usize>().is_ok())
                })
                .collect();
            assert!(!rows.is_empty());
            assert_eq!(
                rows.last()
                    .unwrap()
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap(),
                report["iterations"].as_u64().unwrap()
            );
        }
    }
    let output = command()
        .arg("ce")
        .arg(&file)
        .args(["--time", "0", "--log"])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("CE TimeLimit"));
    std::fs::remove_file(file).unwrap();
    std::fs::remove_file(json).unwrap();
}

#[test]
fn fenchel_cli_separates_and_rejects_noninteger_demands() {
    let file = std::env::temp_dir().join(format!("baldes-fenchel-cli-{}.txt", std::process::id()));
    let json = file.with_extension("json");
    let text = "fenchel-cli\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n";
    std::fs::write(&file, text).unwrap();
    let output = command()
        .arg("cvrp")
        .arg(&file)
        .args(["--fenchel-cuts", "--log", "--rounding", "none", "--json"])
        .arg(&json)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    assert_eq!(report["status"], "Optimal");
    assert!(report["cuts"].as_u64().unwrap() > 0);
    let log = String::from_utf8_lossy(&output.stderr);
    let header = log.lines().find(|line| line.contains("RMP obj")).unwrap();
    assert!(header.contains("Fenchel"), "{log}");
    assert!(
        !header.contains("Knapsack") && !header.contains("Rank1") && !header.contains("Capacity"),
        "{log}"
    );
    let last = log
        .lines()
        .filter(|line| {
            line.split_whitespace()
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .is_some()
        })
        .last()
        .unwrap();
    let fields: Vec<_> = last.split_whitespace().collect();
    assert_eq!(
        fields[8].parse::<u64>().unwrap(),
        report["cuts"].as_u64().unwrap(),
        "{log}"
    );
    assert!((report["bound"].as_f64().unwrap() - (4.0 + 2.0_f64.sqrt())).abs() < 1e-7);
    std::fs::write(&file, text.replace("1 1 0 1 0", "1 1 0 0.5 0")).unwrap();
    let output = command()
        .arg("cvrp")
        .arg(&file)
        .arg("--fenchel-cuts")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("integer demands"));
    assert!(!command()
        .arg("hgs")
        .arg(&file)
        .arg("--fenchel-cuts")
        .output()
        .unwrap()
        .status
        .success());
    std::fs::remove_file(file).unwrap();
    std::fs::remove_file(json).unwrap();
}

#[test]
fn pstep_parallel_command_and_validation() {
    let output = command().args(["pstep", "--help"]).output().unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("--threads"));
    let instance = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/instances/cvrplib/P-n16-k8.vrp"
    );
    let json = std::env::temp_dir().join(format!("baldes-pstep-cli-{}.json", std::process::id()));
    let output = command()
        .args([
            "pstep",
            instance,
            "--p",
            "2",
            "--threads",
            "3",
            "--enumerate",
            "--json",
        ])
        .arg(&json)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    assert_eq!(report["status"], "Converged");
    assert_eq!(report["pricing_workers"], 3);
    assert!(
        (report["lp_bound"].as_f64().unwrap() - report["enumerated_bound"].as_f64().unwrap()).abs()
            < 1e-6
    );
    std::fs::remove_file(&json).unwrap();
    let output = command()
        .args(["pstep", instance, "--iterations", "0", "--json"])
        .arg(&json)
        .output()
        .unwrap();
    assert!(output.status.success());
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    assert_eq!(report["status"], "IterationLimit");
    assert!(report["lp_bound"].is_null());
    std::fs::remove_file(&json).unwrap();
    for flags in [
        vec!["--threads", "0"],
        vec!["--p", "0"],
        vec!["--time", "-1"],
        vec!["--p"],
        vec!["--unknown"],
    ] {
        assert!(!command()
            .args(["pstep", instance])
            .args(flags)
            .output()
            .unwrap()
            .status
            .success());
    }
}

#[test]
fn help_and_invalid_instance_types() {
    for args in [
        vec!["--help"],
        vec!["solve", "--help"],
        vec!["vrptw", "--help"],
        vec!["cvrp", "--help"],
        vec!["hgs", "--help"],
    ] {
        let output = command().args(args).output().unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("baldes <solve|hgs|ce>"));
    }
    for args in [
        vec![],
        vec!["evrp"],
        vec!["solve"],
        vec!["vrptw"],
        vec!["cvrp"],
        vec!["hgs"],
    ] {
        let output = command().args(args).output().unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("Usage:"));
    }
}

#[test]
fn bound_mode_is_explicit_in_root_and_tree_json() {
    let path = std::env::temp_dir().join(format!("baldes-bound-mode-{}.txt", std::process::id()));
    std::fs::write(&path, "tiny\nVEHICLE\nNUMBER CAPACITY\n2 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n").unwrap();
    for tree in [false, true] {
        for mode in [None, Some("numerical"), Some("certified")] {
            let json = path.with_extension(format!("{tree}-{}.json", mode.unwrap_or("default")));
            let mut cmd = command();
            cmd.arg("vrptw")
                .arg(&path)
                .args(["--no-stabilization", "--no-enumeration", "--json"])
                .arg(&json);
            if tree {
                cmd.args(["--tree", "--no-strong-branching", "--no-diving"]);
            }
            if let Some(mode) = mode {
                cmd.args(["--bound-mode", mode]);
            }
            let result = cmd.output().unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let data = std::fs::read_to_string(&json).unwrap();
            let expected = mode.unwrap_or("numerical");
            assert!(
                data.contains(&format!("\"bound_mode\":\"{expected}\"")),
                "{data}"
            );
            assert!(
                String::from_utf8_lossy(&result.stdout).contains(&format!("bound_mode={expected}"))
            );
            assert!(data.contains("\"numerical_bound\":"), "{data}");
            if expected == "numerical" {
                assert!(data.contains("\"certified_bound\":null"), "{data}");
            } else {
                assert!(!data.contains("\"certified_bound\":null"), "{data}");
            }
            std::fs::remove_file(json).unwrap();
        }
    }
    for flags in [vec!["--bound-mode"], vec!["--bound-mode", "invalid"]] {
        let result = command()
            .arg("vrptw")
            .arg(&path)
            .args(flags)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("bound mode"));
    }
    std::fs::remove_file(path).unwrap();
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
    for (kind, flags, expected) in [
        ("vrptw", vec![], 4.),
        ("cvrp", vec![], 3.4),
        ("cvrp", vec!["--rounding", "tenth"], 3.4),
        ("cvrp", vec!["--rounding", "none"], 2. + 2_f64.sqrt()),
        ("solve", vec![], 4.),
        ("solve", vec!["--ignore-time-windows"], 3.4),
        (
            "solve",
            vec!["--ignore-time-windows", "--rounding", "none"],
            2. + 2_f64.sqrt(),
        ),
    ] {
        let output = command()
            .arg(kind)
            .arg(&path)
            .args(flags)
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
    let output = command()
        .arg("hgs")
        .arg(&path)
        .args(["--time", "0.2"])
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("status=Found cost=4.00"), "{stdout}");
}

#[test]
fn tree_and_kohl_controls_reach_integer_solve_and_reject_cvrp_time_resource() {
    let output = command()
        .args([
            "vrptw",
            "instances/R101.txt",
            "--customers",
            "10",
            "--tree",
            "--two-path-cuts",
            "--rounding",
            "none",
            "--time",
            "10",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("status=Optimal incumbent=269.533140900"),
        "{stdout}"
    );
    let output = command()
        .args(["cvrp", "instances/R101.txt", "--two-path-cuts"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("require vrptw time windows"));
}

#[test]
fn tree_branching_mode_is_explicit_and_validated() {
    let path =
        std::env::temp_dir().join(format!("baldes-branching-cli-{}.txt", std::process::id()));
    std::fs::write(&path,"triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    for mode in [
        "native",
        "vehicles",
        "ryan-foster",
        "arc",
        "cluster",
        "resource",
        "hybrid",
    ] {
        let output = command()
            .arg("vrptw")
            .arg(&path)
            .args(["--tree", "--branching", mode, "--no-stabilization"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("Optimal"));
    }
    let json = path.with_extension("no-history.json");
    let output = command()
        .arg("vrptw")
        .arg(&path)
        .args(["--tree", "--no-branch-history", "--json"])
        .arg(&json)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    assert!(report["branch_evaluations"].as_array().unwrap().is_empty());
    assert_eq!(report["branch_history_ranked"], 0);
    std::fs::remove_file(json).unwrap();
    let output = command()
        .arg("vrptw")
        .arg(&path)
        .args(["--tree", "--branching", "invalid"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown branching mode"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn enumerated_mip_flag_requires_tree_and_uses_the_available_backend() {
    let path =
        std::env::temp_dir().join(format!("baldes-enumerated-mip-{}.txt", std::process::id()));
    std::fs::write(&path,"triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let invalid = command()
        .arg("vrptw")
        .arg(&path)
        .arg("--enumerated-mip")
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("requires --tree"));
    let json = path.with_extension("json");
    // A finite witness from the heuristic triggers automatic enumeration.
    let result = command()
        .arg("vrptw")
        .arg(&path)
        .args([
            "--tree",
            "--enumerated-mip",
            "--no-strong-branching",
            "--no-diving",
            "--no-stabilization",
            "--json",
        ])
        .arg(&json)
        .output()
        .unwrap();
    if cfg!(feature = "highs") {
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(std::fs::read_to_string(&json)
            .unwrap()
            .contains("\"enumerated_mip_calls\":1"));
    } else {
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("requires the highs feature"));
    }
    let _ = std::fs::remove_file(json);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn canonical_and_legacy_option_names_reach_the_same_root_and_tree_controls() {
    for prefix in ["BALDES", "RCSP"] {
        for tree in [false, true] {
            let mut args = vec![
                "vrptw".to_owned(),
                "instances/R101.txt".to_owned(),
                "--customers".to_owned(),
                "4".to_owned(),
                "--time".to_owned(),
                "10".to_owned(),
                "--option".to_owned(),
                format!("{prefix}labelDominanceStrategy=1"),
                "--option".to_owned(),
                format!("{prefix}hardTimeThresholdInPricing=1"),
            ];
            if tree {
                args.push("--tree".to_owned());
            }
            let output = command().args(&args).output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("status=Optimal"));
            args.extend([
                "--option".to_owned(),
                format!("{prefix}labelDominanceStrategy=2"),
            ]);
            let invalid = command().args(&args).output().unwrap();
            assert!(!invalid.status.success());
            assert!(String::from_utf8_lossy(&invalid.stderr)
                .contains("unsupported label dominance strategy"));
        }
    }
}

#[test]
fn hgs_warm_start_and_ipm_flags_reach_the_routing_master() {
    let path = std::env::temp_dir().join(format!("baldes-ipm-hgs-{}.txt", std::process::id()));
    std::fs::write(&path,"tiny\nVEHICLE\nNUMBER CAPACITY\n2 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n").unwrap();
    for columns in ["best", "all", "1"] {
        let output = command()
            .arg("vrptw")
            .arg(&path)
            .args([
                "--master",
                "ipm",
                "--rounding",
                "none",
                "--hgs-time",
                "0.03",
                "--hgs-columns",
                columns,
                "--no-enumeration",
                "--time",
                "5",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        let lp: f64 = stdout
            .split_whitespace()
            .find_map(|w| w.strip_prefix("lp="))
            .unwrap()
            .parse()
            .unwrap();
        assert!((lp - (2. + 2_f64.sqrt())).abs() < 1e-6, "{stdout}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("HGS seed: columns="));
    }
    for flags in [
        vec!["--master", "wrong"],
        vec!["--hgs-columns", "wrong"],
        vec!["--hgs-columns", "0"],
        vec!["--hgs-time", "0"],
        vec!["--hgs-time", "NaN"],
    ] {
        assert!(!command()
            .arg("vrptw")
            .arg(&path)
            .args(flags)
            .output()
            .unwrap()
            .status
            .success());
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn cg_progress_is_opt_in_periodic_and_separate_from_results() {
    let base = [
        "vrptw",
        "instances/R101.txt",
        "--customers",
        "5",
        "--no-enumeration",
        "--no-stabilization",
        "--time",
        "5",
    ];
    let quiet = command().args(base).output().unwrap();
    assert!(quiet.status.success());
    assert!(!String::from_utf8_lossy(&quiet.stderr).contains("COLUMN GENERATION"));
    let logged = command()
        .args(base)
        .args(["--log-interval", "2"])
        .output()
        .unwrap();
    assert!(
        logged.status.success(),
        "{}",
        String::from_utf8_lossy(&logged.stderr)
    );
    let log = String::from_utf8_lossy(&logged.stderr);
    assert!(log.contains("BALDES | COLUMN GENERATION"));
    assert!(log.contains("RMP obj") && log.contains("Lower bound") && log.contains("CG Optimal"));
    assert!(log.contains("Bucket step"));
    for line in log.lines().filter(|line| {
        line.split_whitespace()
            .next()
            .is_some_and(|s| s.parse::<usize>().is_ok())
    }) {
        let cells: Vec<_> = line.split_whitespace().collect();
        assert!(
            cells[cells.len() - 3]
                .split('/')
                .all(|s| s.parse::<f64>().is_ok()),
            "{line}"
        );
    }
    assert!(!String::from_utf8_lossy(&logged.stdout).contains("COLUMN GENERATION"));
    let iterations: Vec<usize> = log
        .lines()
        .filter_map(|line| line.split_whitespace().next()?.parse().ok())
        .collect();
    assert!(iterations.len() >= 3, "{log}");
    assert_eq!(iterations[0], 1);
    assert!(
        iterations[1..iterations.len() - 1]
            .iter()
            .all(|n| n % 2 == 0),
        "{log}"
    );
    let lp = |bytes: &[u8]| {
        String::from_utf8_lossy(bytes)
            .split_whitespace()
            .find_map(|s| s.strip_prefix("lp="))
            .unwrap()
            .parse::<f64>()
            .unwrap()
    };
    assert_eq!(lp(&logged.stdout), lp(&quiet.stdout));
    for alias in ["--verbose", "--log", "-v"] {
        let output = command().args(base).arg(alias).output().unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("COLUMN GENERATION"));
    }
    assert!(!command()
        .args(base)
        .args(["--log-interval", "0"])
        .output()
        .unwrap()
        .status
        .success());
}

#[test]
fn highs_master_selection_respects_the_build_features() {
    let result = command()
        .args([
            "vrptw",
            "instances/R101.txt",
            "--customers",
            "5",
            "--master",
            "highs",
            "--no-enumeration",
            "--time",
            "5",
            "--log",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    if cfg!(feature = "highs") {
        assert!(result.status.success(), "{stderr}");
        assert!(stderr.contains("Master: Highs"), "{stderr}");
        assert!(String::from_utf8_lossy(&result.stdout).contains("status=Optimal"));
    } else {
        assert!(!result.status.success());
        assert!(stderr.contains("highs build feature"), "{stderr}");
    }
}

#[test]
fn hgs_solver_log_precedes_column_generation_and_preserves_stdout() {
    let output = command()
        .args([
            "vrptw",
            "instances/R101.txt",
            "--customers",
            "5",
            "--hgs-time",
            "0.05",
            "--no-enumeration",
            "--time",
            "5",
            "--log",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let log = String::from_utf8_lossy(&output.stderr);
    let hgs = log
        .find("BALDES | HYBRID GENETIC SEARCH")
        .expect("HGS banner");
    let finish = log.find("HGS Feasible").expect("HGS summary");
    let cg = log.find("BALDES | COLUMN GENERATION").expect("CG banner");
    assert!(hgs < finish && finish < cg, "{log}");
    assert!(
        log.contains("Incumbent") && log.contains("Best infeas"),
        "{log}"
    );
    assert!(!log.contains("| It "), "{log}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("instance="));
    assert!(!stdout.contains("HYBRID GENETIC SEARCH"));

    let quiet = command()
        .args([
            "hgs",
            "instances/R101.txt",
            "--customers",
            "5",
            "--time",
            "0.05",
        ])
        .output()
        .unwrap();
    assert!(quiet.status.success());
    assert!(!String::from_utf8_lossy(&quiet.stderr).contains("HYBRID GENETIC SEARCH"));

    let standalone = command()
        .args([
            "hgs",
            "instances/R101.txt",
            "--customers",
            "5",
            "--time",
            "0.05",
            "--log",
        ])
        .output()
        .unwrap();
    assert!(standalone.status.success());
    assert!(String::from_utf8_lossy(&standalone.stderr).contains("HGS Feasible"));
    assert!(!String::from_utf8_lossy(&standalone.stdout).contains("Best infeas"));
}

#[test]
fn clp_runtime_policy_modes_are_validated() {
    for (flag, value) in [
        ("--clp-scaling", "invalid"),
        ("--clp-work-areas", "invalid"),
    ] {
        let output = command()
            .args(["vrptw", "instances/C101.txt", flag, value])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("Clp"));
    }
    let path = std::env::temp_dir().join(format!("baldes-clp-policy-{}.txt", std::process::id()));
    std::fs::write(&path, "tiny\nVEHICLE\nNUMBER CAPACITY\n2 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n").unwrap();
    for scaling in ["auto", "geometric", "equilibrium", "off"] {
        for areas in ["retain", "discard"] {
            let output = command()
                .arg("vrptw")
                .arg(&path)
                .args([
                    "--clp-scaling",
                    scaling,
                    "--clp-work-areas",
                    areas,
                    "--no-enumeration",
                ])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("status=Optimal"));
        }
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn cg_progress_shows_only_enabled_cut_families() {
    for (flags, expected) in [
        (vec![], vec![]),
        (
            vec!["--cuts"],
            vec!["Rank1", "Capacity", "2Path", "KPath", "Knapsack", "Fenchel"],
        ),
        (vec!["--rank1"], vec!["Rank1"]),
        (vec!["--capacity-cuts"], vec!["Capacity"]),
        (vec!["--two-path-cuts"], vec!["Capacity", "2Path"]),
        (
            vec!["--rank1", "--capacity-cuts"],
            vec!["Rank1", "Capacity"],
        ),
    ] {
        let output = command()
            .args([
                "vrptw",
                "instances/R101.txt",
                "--customers",
                "5",
                "--no-enumeration",
                "--no-stabilization",
                "--log-interval",
                "2",
                "--time",
                "5",
            ])
            .args(&flags)
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{flags:?}: {log}");
        let header = log.lines().find(|line| line.contains("RMP obj")).unwrap();
        for family in ["Rank1", "Capacity", "2Path", "KPath", "Knapsack", "Fenchel"] {
            assert_eq!(
                header.split_whitespace().any(|s| s == family),
                expected.contains(&family),
                "{flags:?}: {header}"
            );
        }
        let rows: Vec<_> = log
            .lines()
            .filter(|line| {
                line.split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<usize>().ok())
                    .is_some()
            })
            .collect();
        assert!(!rows.is_empty(), "{log}");
        for row in rows {
            let fields: Vec<_> = row.split_whitespace().collect();
            assert_eq!(fields.len(), 11 + expected.len(), "{row}");
            for field in &fields[8..8 + expected.len()] {
                assert!(field.parse::<usize>().is_ok(), "{row}");
            }
        }
        assert!(!String::from_utf8_lossy(&output.stdout).contains("COLUMN GENERATION"));
    }
}

#[test]
fn solve_detects_cvrplib_and_rejects_time_window_cuts() {
    let path = std::env::temp_dir().join(format!("baldes-solve-cvrp-{}.data", std::process::id()));
    std::fs::write(&path, "NAME : tiny\nTYPE : CVRP\nDIMENSION : 3\nCAPACITY : 2\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n1 0 0\n2 1 0\n3 0 1\nDEMAND_SECTION\n1 0\n2 1\n3 1\nDEPOT_SECTION\n1\n-1\nEOF\n").unwrap();
    for tree in [false, true] {
        let mut cmd = command();
        cmd.arg("solve")
            .arg(&path)
            .args(["--no-enumeration", "--no-stabilization", "--time", "5"]);
        if tree {
            cmd.arg("--tree");
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("status=Optimal"), "{stdout}");
        assert!(
            stdout.contains(if tree { "incumbent=3" } else { "lp=3" }),
            "{stdout}"
        );
    }
    let output = command()
        .arg("solve")
        .arg(&path)
        .arg("--two-path-cuts")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("time windows"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn all_cuts_supports_capacity_only_models_and_rejects_hgs() {
    for flags in [vec!["cvrp"], vec!["solve", "--ignore-time-windows"]] {
        let mut cmd = command();
        cmd.arg(flags[0]).arg("instances/R101.txt");
        cmd.args(&flags[1..]);
        let output = cmd
            .args([
                "--cuts",
                "--customers",
                "5",
                "--no-enumeration",
                "--no-stabilization",
                "--log",
                "--time",
                "5",
            ])
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{flags:?}: {log}");
        let header = log.lines().find(|line| line.contains("RMP obj")).unwrap();
        for family in ["Rank1", "Capacity", "KPath", "Knapsack", "Fenchel"] {
            assert!(header.split_whitespace().any(|s| s == family), "{header}");
        }
        assert!(!header.split_whitespace().any(|s| s == "2Path"), "{header}");
    }
    let output = command()
        .args(["hgs", "instances/R101.txt", "--cuts", "--customers", "5"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--cuts requires a routing solve"));
    let output = command().args(["solve", "--help"]).output().unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("--cuts "));
}

#[test]
fn deluxing_flag_enables_enumerated_mip_and_checks_dependencies() {
    let path = std::env::temp_dir().join(format!("baldes-deluxing-{}.txt", std::process::id()));
    std::fs::write(&path,"triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    for (flags, message) in [
        (vec!["--deluxing"], "--deluxing requires --tree"),
        (
            vec!["--tree", "--deluxing", "--no-enumeration"],
            "conflicts with --no-enumeration",
        ),
        (
            vec!["--tree", "--no-enumeration", "--deluxing"],
            "conflicts with --no-enumeration",
        ),
    ] {
        let out = command()
            .arg("vrptw")
            .arg(&path)
            .args(flags)
            .output()
            .unwrap();
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains(message));
    }
    let json = path.with_extension("json");
    let out = command()
        .arg("vrptw")
        .arg(&path)
        .args([
            "--tree",
            "--deluxing",
            "--rounding",
            "none",
            "--no-strong-branching",
            "--no-diving",
            "--no-stabilization",
            "--json",
        ])
        .arg(&json)
        .output()
        .unwrap();
    if cfg!(feature = "highs") {
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
        assert_eq!(report["status"], "Optimal");
        assert_eq!(report["enumerated_mip_calls"], 1);
        assert!((report["incumbent"].as_f64().unwrap() - (4. + 2f64.sqrt())).abs() < 1e-6);
    } else {
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("requires the highs feature"));
    }
    let _ = std::fs::remove_file(json);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn routing_bucket_steps_are_applied_and_validated() {
    let file = std::env::temp_dir().join(format!("baldes-steps-{}.txt", std::process::id()));
    std::fs::write(&file, "steps\nVEHICLE\nNUMBER CAPACITY\n2 10\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 2 0 1 0 100 0\n").unwrap();
    for (value, expected) in [("7", "7/7"), ("7,3", "7/3")] {
        let output = command()
            .arg("solve")
            .arg(&file)
            .args(["--steps", value, "--log-interval", "1", "--no-enumeration"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let log = String::from_utf8_lossy(&output.stderr);
        assert!(log.contains(expected), "{log}");
    }
    for value in ["-1", "NaN", "inf", "1,2,3", "", "1,", "bad"] {
        let output = command()
            .args(["solve", "missing", "--steps", value])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{value}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("bucket steps"));
    }
    let output = command()
        .args(["solve", "missing", "--steps"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing bucket steps"));
    std::fs::remove_file(file).unwrap();
}
