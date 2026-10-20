//! Reproducible sparse KKT benchmark: cargo run --release --example ipm_sparse_bench.
use baldes::ipm::{SolverType, SparseMatrix, SparseSolver};
use std::{hint::black_box, time::Instant};

fn main() {
    let repeats: usize = std::env::args()
        .nth(1)
        .map(|v| v.parse().unwrap())
        .unwrap_or(20);
    assert!(repeats > 0);
    for (name, m, n, width) in [
        ("diagonal", 2048, 2048, 1),
        ("banded", 256, 1024, 3),
        ("overlapping", 128, 512, 8),
    ] {
        let mut entries = Vec::new();
        for j in 0..n {
            for k in 0..width {
                entries.push(((j + k) % m, j, 1. / (k + 1) as f64));
            }
        }
        let a = SparseMatrix::from_triplets(m, n, &entries).unwrap();
        let expected: Vec<_> = (0..n + m).map(|i| (i % 17) as f64 / 17. - 0.5).collect();
        let rp = vec![0.1; n];
        let rd = vec![0.2; m];
        for kind in [SolverType::HiPO, SolverType::NormalEquations] {
            let start = Instant::now();
            let mut solver = SparseSolver::new(a.clone(), kind).unwrap();
            let setup = start.elapsed();
            let mut factor_ns = Vec::new();
            let mut solve_ns = Vec::new();
            let mut max_error = 0.0_f64;
            // One warmup followed by measured refactorizations with changing diagonals.
            for r in 0..=repeats {
                let theta = vec![1. + r as f64 / repeats as f64; n];
                let mut rhs = a.transpose_multiply(&expected[n..]).unwrap();
                for j in 0..n {
                    rhs[j] -= (theta[j] + rp[j]) * expected[j];
                }
                rhs.extend(
                    a.multiply(&expected[..n])
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, v)| v + rd[i] * expected[n + i]),
                );
                let start = Instant::now();
                solver.factorize(black_box(&theta), &rp, &rd).unwrap();
                let factor = start.elapsed().as_nanos();
                let start = Instant::now();
                let x = black_box(solver.solve(black_box(&rhs)).unwrap());
                let solve = start.elapsed().as_nanos();
                for (x, e) in x.iter().zip(&expected) {
                    max_error = max_error.max((x - e).abs());
                }
                if r > 0 {
                    factor_ns.push(factor);
                    solve_ns.push(solve);
                }
            }
            assert!(max_error < 1e-8, "{name} {kind:?}: {max_error}");
            factor_ns.sort_unstable();
            solve_ns.sort_unstable();
            println!("{name} {kind:?} rows={m} cols={n} nnz={} setup_ms={:.3} factor_us={:.3} solve_us={:.3} max_error={max_error:.3e}",
                a.nnz(), setup.as_secs_f64() * 1000., factor_ns[repeats / 2] as f64 / 1000., solve_ns[repeats / 2] as f64 / 1000.);
        }
    }
}
