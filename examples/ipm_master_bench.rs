//! Captured R201 LP: measure Newton iterations independently of column generation.
use baldes::extensions::ipm::{solve, ModelData, Options, Sense, Solver, SolverType, SparseMatrix, Status};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let repeats: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(3);
    assert!(repeats > 0);
    let backend = match args.next().as_deref().unwrap_or("hipo") {
        "hipo" => SolverType::HiPO,
        "normal" => SolverType::NormalEquations,
        other => panic!("unknown backend: {other}"),
    };
    let mode = args.next().unwrap_or_default();
    let text = if mode == "c202-sequence" {
        include_str!("../tests/ipm/fixtures/c202_rank1_master.txt")
    } else {
        include_str!("../tests/ipm/fixtures/r201_master.txt")
    };
    let model = fixture(text);
    if mode.ends_with("sequence") {
        sequence(&model, backend, repeats);
        return;
    }
    for gap_tolerance in [Some(0.5), None] {
        let mut times = Vec::new();
        let mut iterations = Vec::new();
        for rep in 0..=repeats {
            let start = Instant::now();
            let solution = solve(
                &model,
                &Options {
                    tolerance: 1e-9,
                    gap_tolerance,
                    solver: backend,
                    ..Options::default()
                },
            )
            .unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            assert_eq!(solution.status, Status::Optimal);
            assert!(solution.primal_residual <= 1e-9);
            assert!(solution.dual_residual <= 1e-9);
            assert!(solution.relative_gap <= gap_tolerance.unwrap_or(1e-9));
            assert!((solution.objective - 937.6319384978376).abs() < 2e-6);
            if rep > 0 {
                times.push(elapsed);
                iterations.push(solution.iterations);
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{backend:?} gap={gap_tolerance:?} median_ms={:.3} newton_iterations={iterations:?}",
            times[repeats / 2] * 1000.
        );
    }
}

fn fixture(text: &str) -> ModelData {
    let mut lines = text.lines();
    let mut shape = lines.next().unwrap().split_whitespace();
    let m = shape.next().unwrap().parse().unwrap();
    let n = shape.next().unwrap().parse().unwrap();
    let b = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let sense = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| match s {
            "0" => Sense::Equal,
            "1" => Sense::LessEqual,
            "2" => Sense::GreaterEqual,
            _ => panic!("invalid row sense"),
        })
        .collect();
    let (mut c, mut lower, mut upper, mut entries) = (vec![], vec![], vec![], vec![]);
    for (j, line) in lines.enumerate() {
        let mut fields = line.split_whitespace();
        c.push(fields.next().unwrap().parse().unwrap());
        lower.push(fields.next().unwrap().parse().unwrap());
        upper.push(fields.next().unwrap().parse().unwrap());
        for item in fields {
            let (i, value) = item.split_once(':').unwrap();
            entries.push((i.parse().unwrap(), j, value.parse().unwrap()));
        }
    }
    assert_eq!(c.len(), n);
    ModelData {
        a: SparseMatrix::from_triplets(m, n, &entries).unwrap(),
        b,
        c,
        lower,
        upper,
        sense,
    }
}

/// Compare the same eight growing masters, excluding matrix construction.
fn sequence(full: &ModelData, backend: SolverType, repeats: usize) {
    let mut models = Vec::new();
    for step in 0..8 {
        let columns = full.c.len() - (7 - step) * 30;
        let entries: Vec<_> = (0..columns)
            .flat_map(|j| full.a.column(j).map(move |(i, v)| (i, j, v)))
            .collect();
        models.push(ModelData {
            a: SparseMatrix::from_triplets(full.b.len(), columns, &entries).unwrap(),
            b: full.b.clone(),
            c: full.c[..columns].to_vec(),
            lower: full.lower[..columns].to_vec(),
            upper: full.upper[..columns].to_vec(),
            sense: full.sense.clone(),
        });
    }
    let options = Options {
        tolerance: 1e-9,
        solver: backend,
        ..Options::default()
    };
    let references: Vec<_> = models
        .iter()
        .map(|m| solve(m, &options).unwrap().objective)
        .collect();
    for persistent in [false, true] {
        let mut times = Vec::new();
        let mut statistics = Default::default();
        let mut iterations = 0;
        for rep in 0..=repeats {
            let mut solver = Solver::default();
            let started = Instant::now();
            iterations = 0;
            for (model, expected) in models.iter().zip(&references) {
                let solution = if persistent {
                    solver.solve(model, &options)
                } else {
                    solve(model, &options)
                }
                .unwrap();
                assert_eq!(solution.status, Status::Optimal);
                assert!(solution.primal_residual <= 1e-9);
                assert!(solution.dual_residual <= 1e-9);
                assert!(solution.relative_gap <= 1e-9);
                assert!((solution.objective - expected).abs() < 2e-6);
                iterations += solution.iterations;
            }
            let elapsed = started.elapsed().as_secs_f64();
            statistics = solver.statistics();
            if rep > 0 {
                times.push(elapsed);
            }
        }
        times.sort_by(f64::total_cmp);
        println!("sequence persistent={persistent} backend={backend:?} median_ms={:.3} newton_iterations={iterations} stats={statistics:?}", times[repeats / 2] * 1000.);
    }
}
