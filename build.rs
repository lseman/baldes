//! Only the optional LP backend links C++; all solver logic stays Rust.
//! `--features clp` links the static Clp/CoinUtils archives found in
//! BALDES_CLP_LIB_DIR (OPENRSCP_CLP_LIB_DIR is accepted for existing builds).
use std::{env, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-env-changed=BALDES_CLP_LIB_DIR");
    println!("cargo:rerun-if-env-changed=OPENRSCP_CLP_LIB_DIR");
    if env::var_os("CARGO_FEATURE_CLP").is_none() {
        return;
    }
    let dir: PathBuf = env::var_os("BALDES_CLP_LIB_DIR")
        .or_else(|| env::var_os("OPENRSCP_CLP_LIB_DIR"))
        .map(PathBuf::from)
        .expect("--features clp needs BALDES_CLP_LIB_DIR, the directory holding libopenrscp_clp_solver.a and libopenrscp_coinutils.a");
    for name in ["openrscp_clp_solver", "openrscp_coinutils"] {
        assert!(
            dir.join(format!("lib{name}.a")).exists(),
            "Clp archives missing: set BALDES_CLP_LIB_DIR to the COIN-OR build directory"
        );
    }
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static=openrscp_clp_solver");
    println!("cargo:rustc-link-lib=static=openrscp_coinutils");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
}
