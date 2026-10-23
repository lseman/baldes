//! Check the exact native header used by Clp against an independent scalar loop.
use std::{fs, path::PathBuf, process::Command, time::SystemTime};

#[test]
fn paired_sparse_products_match_scalar_arithmetic_and_support_fallback() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let unique = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("baldes-clp-matrix-{}-{unique}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let compiler = std::env::var_os("CXX").unwrap_or_else(|| "c++".into());
    for scalar in [false, true] {
        let binary = directory.join(if scalar { "scalar" } else { "paired" });
        let mut command = Command::new(&compiler);
        command.args(["-std=c++14", "-O2", "-ffp-contract=off"]);
        if scalar {
            command.arg("-DBALDES_CLP_FORCE_SCALAR=1");
        }
        let output = command
            .arg("-I")
            .arg(root.join("src/master/solvers/clp/native"))
            .arg(root.join("tests/unit/master/solvers/clp/matrix_kernels.cpp"))
            .arg("-o")
            .arg(&binary)
            .output()
            .expect("the Clp native-kernel test requires the C++ compiler used by the build");
        assert!(
            output.status.success(),
            "native-kernel test compilation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let output = Command::new(&binary).output().unwrap();
        assert!(
            output.status.success(),
            "native-kernel oracle failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    fs::remove_dir_all(directory).unwrap();
}
