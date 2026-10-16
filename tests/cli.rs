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
    ] {
        let output = command().args(args).output().unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("baldes <vrptw|cvrp>"));
    }
    for args in [vec![], vec!["evrp"], vec!["vrptw"], vec!["cvrp"]] {
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
