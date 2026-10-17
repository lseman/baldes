//! Only the optional LP backend links C++; all solver logic stays Rust.
//! `--features clp` compiles Clp 1.17.10 and CoinUtils 2.11 from the vendor/
//! submodules, as OpenRSCP's cmake/CoinClp.cmake does.
//! `--features highs` links against HiGHS via pkg-config or a vendored copy.
fn main() {
    #[cfg(feature = "clp")]
    {
        clp::build();
        println!("cargo:rustc-link-lib=dylib=m");
    }
    #[cfg(feature = "highs")]
    {
        highs::link();
    }
}

#[cfg(feature = "clp")]
mod clp {
    use std::{env, fs, path::Path, path::PathBuf};
    /// Drivers, the optional ABC solver, AMPL and third-party Cholesky
    /// backends; IdiSolve.cpp stays (the body of Idiot::IdiSolve).
    fn excluded(stem: &str) -> bool {
        stem.starts_with("Abc")
            || stem.starts_with("CoinAbc")
            || [
                "ClpMain", "ClpSolver", "unitTest", "CbcOrClpParam", "Clp_ampl", "MyEventHandler",
                "MyMessageHandler", "ClpCholeskyMumps", "ClpCholeskyPardiso", "ClpCholeskyTaucs",
                "ClpCholeskyUfl", "ClpCholeskyWssmp", "ClpCholeskyWssmpKKT",
            ]
            .contains(&stem)
    }
    /// The bundled Clp of the the solver reference differs from 1.17.10 in two
    /// places.
    const CORRECTIONS: [(&str, &str, &str); 2] = [
        (
            // CoinFactorization::goSparse: 300 < rows < 10000, threshold2 = rows / 2.
            "CoinFactorization4.cpp",
            "        sparseThreshold_ = CoinMin(numberRows_ / 6, 500);\n        sparseThreshold2_ = numberRows_ >> 2;",
            "        sparseThreshold_ = CoinMin(numberRows_ / 6, 500);\n        sparseThreshold2_ = numberRows_ >> 1;",
        ),
        (
            // ClpSimplexPrimal::primalRow: the average step uses |theta|.
            "ClpSimplexPrimal.cpp",
            "double theta1 = CoinMax(theta_, 1.0e-12);",
            "double theta1 = CoinMax(fabs(theta_), 1.0e-12);",
        ),
    ];
    fn sources(dir: &Path, patched: &Path) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = fs::read_dir(dir)
            .unwrap_or_else(|_| panic!("{} missing: run `git submodule update --init`", dir.display()))
            .map(|entry| entry.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "cpp"))
            .filter(|p| !excluded(p.file_stem().unwrap().to_str().unwrap()))
            .collect();
        files.sort();
        for file in &mut files {
            let name = file.file_name().unwrap().to_str().unwrap();
            if name == "Clp_C_Interface.cpp" {
                // Keep the opaque C wrapper in its own translation unit;
                // upstream Clp implements the deep model/basis copy.
                let mut text = fs::read_to_string(&*file).unwrap();
                text.push_str(r#"
extern "C" Clp_Simplex *baldes_clp_clone(const Clp_Simplex *source) {
  Clp_Simplex *copy = NULL;
  try {
    copy = Clp_newModel();
    *copy->model_ = *source->model_;
    return copy;
  } catch (...) {
    if (copy) Clp_deleteModel(copy);
    return NULL;
  }
}
"#);
                let copy = patched.join(name);
                fs::write(&copy, text).unwrap();
                *file = copy;
                continue;
            }
            if let Some((_, old, new)) = CORRECTIONS.iter().find(|(n, _, _)| *n == name) {
                let text = fs::read_to_string(&*file).unwrap();
                assert_eq!(text.matches(old).count(), 1, "unsupported {name}: expected one correction site");
                let copy = patched.join(name);
                fs::write(&copy, text.replace(old, new)).unwrap();
                *file = copy;
            }
        }
        files
    }
    /// The flags of OpenRSCP's build: -O2 -DNDEBUG -std=c++14 -w, the
    /// Autotools stand-in config.h first on the include path.
    fn library(name: &str, files: &[PathBuf], includes: &[&Path]) {
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .std("c++14")
            .opt_level(2)
            .pic(true)
            .warnings(false)
            .flag_if_supported("-w")
            .define("NDEBUG", None);
        for define in ["CLP_BUILD", "COINUTILS_BUILD", "HAVE_CONFIG_H"] {
            build.define(define, None);
        }
        for define in ["HAVE_CFLOAT", "HAVE_CMATH", "HAVE_MATH_H"] {
            build.define(define, "1");
        }
        build.includes(includes).files(files).compile(name);
    }
    pub fn build() {
        let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let config = root.join("vendor/coin");
        let coinutils = root.join("vendor/CoinUtils/CoinUtils/src");
        let clp = root.join("vendor/Clp/Clp/src");
        for path in [&config, &coinutils, &clp] {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        let patched = PathBuf::from(env::var("OUT_DIR").unwrap()).join("patched");
        fs::create_dir_all(&patched).unwrap();
        let coinutils_files = sources(&coinutils, &patched);
        let clp_files = sources(&clp, &patched);
        // Clp before CoinUtils on the link line: the solver needs CoinUtils.
        library("baldes_clp_solver", &clp_files, &[&config, &coinutils, &clp]);
        library("baldes_coinutils", &coinutils_files, &[&config, &coinutils]);
    }
}

#[cfg(feature = "highs")]
mod highs {
    use std::{env, path::PathBuf};

    /// Link the HiGHS shared library: HIGHS_DIR (holding lib/ with
    /// libhighs), then a copy in vendor/highs/, then pkg-config, then the
    /// default linker path.
    pub fn link() {
        println!("cargo:rerun-if-env-changed=HIGHS_DIR");
        let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let vendored = root.join("vendor/highs");
        let dir = env::var_os("HIGHS_DIR").map(PathBuf::from).or_else(|| vendored.is_dir().then_some(vendored));
        if let Some(dir) = dir {
            println!("cargo:rustc-link-search=native={}", dir.join("lib").display());
            println!("cargo:rustc-link-lib=dylib=highs");
            return;
        }
        // pkg-config emits the search path and library itself.
        if pkg_config::Config::new().probe("highs").is_ok() {
            return;
        }
        println!("cargo:rustc-link-lib=dylib=highs");
    }
}
