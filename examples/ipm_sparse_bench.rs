//! Reproducible sparse KKT benchmark: cargo run --release --example ipm_sparse_bench.
use baldes::extensions::ipm::{Parallel, ParallelType, SolverType, SparseMatrix, SparseSolver};
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
        if selected(name) {
            measure(name, a, repeats);
        }
    }
    // Captured routing master: exercise the actual row/column sparsity as well
    // as synthetic patterns. The KKT witness below is independent of its LP optimum.
    let mut lines = include_str!("../tests/ipm/fixtures/r201_master.txt").lines();
    let shape: Vec<usize> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .take(2)
        .map(|v| v.parse().unwrap())
        .collect();
    lines.next(); // RHS
    lines.next(); // Row senses
    let mut entries = Vec::new();
    for (j, line) in lines.enumerate() {
        for value in line.split_whitespace().skip(3) {
            let (i, v) = value.split_once(':').unwrap();
            entries.push((i.parse().unwrap(), j, v.parse().unwrap()));
        }
    }
    if selected("r201-master") {
        measure(
            "r201-master",
            SparseMatrix::from_triplets(shape[0], shape[1], &entries).unwrap(),
            repeats,
        );
    }
    for (name, sizes) in [
        ("dense-front", vec![512]),
        ("forest", vec![128; 4]),
        ("unbalanced", vec![32, 64, 128, 256]),
    ] {
        if !selected(name) {
            continue;
        }
        let m: usize = sizes.iter().sum();
        let mut entries: Vec<_> = (0..m).map(|i| (i, i, 1.)).collect();
        let mut offset = 0;
        for (block, size) in sizes.iter().enumerate() {
            for i in offset..offset + size {
                entries.push((i, m + block, 0.25 + (i % 7) as f64 * 0.01));
            }
            offset += size;
        }
        measure(
            name,
            SparseMatrix::from_triplets(m, m + sizes.len(), &entries).unwrap(),
            repeats,
        );
    }
    if selected("deep-forest") {
        // Different depths and pivot-block sizes expose cross-level overlap.
        // Identity columns keep the KKT witness well conditioned; bridge
        // columns form row cliques between adjacent blocks in each component.
        let components = [
            vec![16, 32, 64, 96],
            vec![24, 48, 72, 112],
            vec![32, 64, 96, 128],
            vec![16, 48, 80, 160],
        ];
        let m = components.iter().flatten().sum::<usize>();
        let mut entries: Vec<_> = (0..m).map(|i| (i, i, 1.)).collect();
        let mut column = m;
        let mut offset = 0;
        for blocks in components {
            for pair in blocks.windows(2) {
                for i in offset..offset + pair[0] + pair[1] {
                    entries.push((i, column, 0.1 + (i % 7) as f64 * 0.005));
                }
                column += 1;
                offset += pair[0];
            }
            offset += blocks.last().unwrap();
        }
        measure(
            "deep-forest",
            SparseMatrix::from_triplets(m, column, &entries).unwrap(),
            repeats,
        );
    }
}

fn selected(name: &str) -> bool {
    std::env::args()
        .nth(4)
        .map_or(true, |filter| filter == "all" || filter == name)
}

fn measure(name: &str, a: SparseMatrix, repeats: usize) {
    // Optional arguments after repeats: threads and off/tree/node/both/choose.
    let threads = std::env::args()
        .nth(2)
        .map(|v| v.parse().unwrap())
        .unwrap_or(1);
    let mode = std::env::args().nth(3).unwrap_or_else(|| "off".into());
    let (parallel, parallel_type) = match mode.as_str() {
        "off" => (Parallel::Off, ParallelType::Both),
        "choose" => (Parallel::Choose, ParallelType::Both),
        "tree" => (Parallel::On, ParallelType::Tree),
        "node" => (Parallel::On, ParallelType::Node),
        "both" => (Parallel::On, ParallelType::Both),
        _ => panic!("expected off/tree/node/both/choose"),
    };
    let (m, n) = (a.rows(), a.cols());
    let expected: Vec<_> = (0..n + m).map(|i| (i % 17) as f64 / 17. - 0.5).collect();
    let rp = vec![0.1; n];
    let rd = vec![0.2; m];
    for kind in [SolverType::HiPO, SolverType::NormalEquations] {
        let start = Instant::now();
        let mut solver =
            SparseSolver::with_parallel(a.clone(), kind, parallel, parallel_type, threads).unwrap();
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
        println!("{name} {kind:?} mode={mode} threads={threads} rows={m} cols={n} nnz={} setup_ms={:.3} factor_us={:.3} solve_us={:.3} max_error={max_error:.3e} plan={:?}",
                a.nnz(), setup.as_secs_f64() * 1000., factor_ns[repeats / 2] as f64 / 1000., solve_ns[repeats / 2] as f64 / 1000., solver.factorization_plan());
    }
}
