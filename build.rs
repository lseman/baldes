//! Only the optional LP backend links C++; all solver logic stays Rust.
//! `--features clp` compiles Clp 1.17.10 and CoinUtils 2.11 from the vendor/
//! submodules, as OpenRSCP's cmake/CoinClp.cmake does.
//! `--features highs` links against HiGHS via pkg-config or a vendored copy.
fn main() {
    println!("cargo:rerun-if-env-changed=BALDES_RESOURCE_DIMENSIONS");
    let dimensions = std::env::var("BALDES_RESOURCE_DIMENSIONS")
        .unwrap_or_else(|_| "2".into())
        .parse::<usize>()
        .expect("BALDES_RESOURCE_DIMENSIONS must be a usize");
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    std::fs::write(
        output.join("resource_dimensions.rs"),
        format!("pub const DEFAULT_RESOURCE_DIMENSIONS: usize = {dimensions};\n"),
    )
    .unwrap();
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
                "ClpMain",
                "ClpSolver",
                "unitTest",
                "CbcOrClpParam",
                "Clp_ampl",
                "MyEventHandler",
                "MyMessageHandler",
                "ClpCholeskyMumps",
                "ClpCholeskyPardiso",
                "ClpCholeskyTaucs",
                "ClpCholeskyUfl",
                "ClpCholeskyWssmp",
                "ClpCholeskyWssmpKKT",
            ]
            .contains(&stem)
    }
    /// Numerical corrections applied to the bundled Clp 1.17.10 sources.
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
            .unwrap_or_else(|_| {
                panic!(
                    "{} missing: run `git submodule update --init`",
                    dir.display()
                )
            })
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
// Preserve allocation metadata without casting a matrix to a derived object.
struct BaldesPackedCapacity : CoinPackedMatrix {
  static CoinBigIndex elements(const CoinPackedMatrix &matrix) {
    return matrix.* &BaldesPackedCapacity::maxSize_;
  }
};
extern "C" Clp_Simplex *baldes_clp_clone(const Clp_Simplex *source) {
  Clp_Simplex *copy = NULL;
  try {
    copy = Clp_newModel();
    *copy->model_ = *source->model_;
    // Upstream cloning compacts packed matrices. Layout and spare capacity
    // affect which numerical kernel runs after a later column insertion.
    ClpPackedMatrix *destination = dynamic_cast<ClpPackedMatrix *>(copy->model_->clpMatrix());
    const ClpPackedMatrix *original = dynamic_cast<const ClpPackedMatrix *>(source->model_->clpMatrix());
    if (destination && original) {
      const CoinPackedMatrix &matrix = *original->getPackedMatrix();
      CoinPackedMatrix retained(matrix.isColOrdered(), 0.0, 0.0);
      retained.reserve(matrix.getMaxMajorDim(), BaldesPackedCapacity::elements(matrix));
      retained.copyReuseArrays(matrix);
      destination->getPackedMatrix()->swap(retained);
      destination->checkGaps();
    }
    return copy;
  } catch (...) {
    if (copy) Clp_deleteModel(copy);
    return NULL;
  }
}
#include <algorithm>
#include "CoinTime.hpp"
// Deadline setters add the current clock, including setDblParam. Preserve
// absolute deadlines through a base member pointer, without a derived cast.
struct BaldesClpDeadlines : ClpModel {
  static double *parameters(ClpModel &model) {
    return model.* &BaldesClpDeadlines::dblParam_;
  }
};
struct BaldesScopedSolveLimits {
  ClpSimplex &model;
  int iterations;
  double cpu, wall;
  explicit BaldesScopedSolveLimits(ClpSimplex &m, int limit, double seconds)
    : model(m), iterations(m.maximumIterations()) {
    double *parameters = BaldesClpDeadlines::parameters(model);
    cpu = parameters[ClpMaxSeconds];
    wall = parameters[ClpMaxWallSeconds];
    const double deadline = CoinWallclockTime() + seconds;
    parameters[ClpMaxWallSeconds] = wall >= 0.0 ? std::min(wall, deadline) : deadline;
    model.setMaximumIterations(std::min(iterations, limit));
  }
  ~BaldesScopedSolveLimits() {
    model.setMaximumIterations(iterations);
    double *parameters = BaldesClpDeadlines::parameters(model);
    parameters[ClpMaxSeconds] = cpu;
    parameters[ClpMaxWallSeconds] = wall;
  }
  BaldesScopedSolveLimits(const BaldesScopedSolveLimits &) = delete;
  BaldesScopedSolveLimits &operator=(const BaldesScopedSolveLimits &) = delete;
};
extern "C" int baldes_clp_dual_limited(Clp_Simplex *model, int iterations, double seconds, int keepWorkAreas) {
  if (!model || iterations < 0 || !std::isfinite(seconds) || seconds < 0.0 || (keepWorkAreas != 0 && keepWorkAreas != 1)) return -1;
  try {
    BaldesScopedSolveLimits limits(*model->model_, iterations, seconds);
    // Bit 1 retains work areas. Rebuild/init stays enabled; do not force
    // factorization reuse (bit 2) or skip initialization (bit 4).
    const int iterationLimit = model->model_->maximumIterations();
    model->model_->dual(0, keepWorkAreas);
    // Dual fake bounds can report unbounded on a bounded LP with very large
    // RHS values and artificial columns. Cross-check with primal simplex;
    // never impose a finite cap on an original infinite column domain.
    if (model->model_->status() == 2) {
      const int used = std::max(0, model->model_->numberIterations());
      if (used >= iterationLimit || model->model_->hitMaximumIterations()) {
        model->model_->setProblemStatus(3);
      } else {
        model->model_->setMaximumIterations(iterationLimit - used);
        model->model_->setNumberIterations(0);
        model->model_->primal(0, 0);
        const long long total = static_cast<long long>(used)
          + std::max(0, model->model_->numberIterations());
        model->model_->setNumberIterations(static_cast<int>(
          std::min<long long>(iterationLimit, total)));
      }
    }
    return 0;
  } catch (...) {
    model->model_->setProblemStatus(4);
    return -1;
  }
}
extern "C" void baldes_clp_solve_limits(Clp_Simplex *model, double *values) {
  const double *parameters = BaldesClpDeadlines::parameters(*model->model_);
  values[0] = model->model_->maximumIterations();
  values[1] = parameters[ClpMaxSeconds];
  values[2] = parameters[ClpMaxWallSeconds];
}
"#);
                let copy = patched.join(name);
                fs::write(&copy, text).unwrap();
                *file = copy;
                continue;
            }
            if name == "ClpPackedMatrix.cpp" {
                // A column stamp replaces the second per-column walk that
                // clears duplicate marks. Keep all coefficient validation and
                // duplicate compression in their original order.
                let mut text = fs::read_to_string(&*file).unwrap();
                for (old, new) in [
                    ("if (mark[iRow] == -1)", "if (mark[iRow] != iColumn)"),
                    ("mark[iRow] = j;", "mark[iRow] = iColumn;"),
                    ("      //clear mark\n      for (j = columnStart[iColumn];\n           j < columnStart[iColumn] + columnLength[iColumn]; j++) {\n        int iRow = row[j];\n        mark[iRow] = -1;\n      }\n", ""),
                ] {
                    assert_eq!(text.matches(old).count(), 1,
                        "unsupported {name}: expected one duplicate-mark site");
                    text = text.replace(old, new);
                }
                // Preserve the subtraction/addition order within each dot
                // product while sharing the matrix traversal and loads.
                let begin = text.find("int ClpPackedMatrix::transposeTimes2(").unwrap();
                let end = begin
                    + text[begin..]
                        .find("\nvoid ClpPackedMatrix::subsetTimes2(")
                        .unwrap();
                let mut method = text[begin..end].to_string();
                for indent in ["              ", "          "] {
                    let old = format!("{indent}double modification = 0.0;\n{indent}for (j = start; j < end; j++) {{\n{indent}  int iRow = row[j];\n{indent}  modification += piWeight[iRow] * elementByColumn[j];\n{indent}}}\n");
                    assert_eq!(
                        method.matches(&old).count(),
                        2,
                        "unsupported transpose weight loops"
                    );
                    method = method.replace(&old, "");
                }
                assert_eq!(method.matches("double value = 0.0;").count(), 4);
                method = method.replace(
                    "double value = 0.0;",
                    "double value = 0.0, modification = 0.0;",
                );
                for (indent, suffix, count) in [
                    ("              ", "", 2),
                    ("          ", "", 1),
                    ("          ", " * rowScale[iRow]", 1),
                ] {
                    let old = format!("\n{indent}value -= pi[iRow] * elementByColumn[j]{suffix};");
                    let new = format!(
                        "{old}\n{indent}modification += piWeight[iRow] * elementByColumn[j];"
                    );
                    assert_eq!(method.matches(&old).count(), count);
                    method = method.replace(&old, &new);
                }
                text.replace_range(begin..end, &method);
                let copy = patched.join(name);
                fs::write(&copy, text).unwrap();
                *file = copy;
                continue;
            }
            if let Some((_, old, new)) = CORRECTIONS.iter().find(|(n, _, _)| *n == name) {
                let text = fs::read_to_string(&*file).unwrap();
                assert_eq!(
                    text.matches(old).count(),
                    1,
                    "unsupported {name}: expected one correction site"
                );
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
        library(
            "baldes_clp_solver",
            &clp_files,
            &[&config, &coinutils, &clp],
        );
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
        let dir = env::var_os("HIGHS_DIR")
            .map(PathBuf::from)
            .or_else(|| vendored.is_dir().then_some(vendored));
        if let Some(dir) = dir {
            println!(
                "cargo:rustc-link-search=native={}",
                dir.join("lib").display()
            );
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
