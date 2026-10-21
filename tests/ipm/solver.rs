use baldes::extensions::ipm::{
    solve, HiPOLDLT, ModelData, NormalEqLDLT, Options, Sense, Solver, SolverType, SparseMatrix, Status,
};

fn model(
    a: &[&[f64]],
    b: &[f64],
    c: &[f64],
    lower: &[f64],
    upper: &[f64],
    sense: &[Sense],
) -> ModelData {
    let entries: Vec<_> = a
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().map(move |(j, &v)| (i, j, v)))
        .collect();
    ModelData {
        a: SparseMatrix::from_triplets(b.len(), c.len(), &entries).unwrap(),
        b: b.to_vec(),
        c: c.to_vec(),
        lower: lower.to_vec(),
        upper: upper.to_vec(),
        sense: sense.to_vec(),
    }
}
fn check(m: &ModelData, objective: f64) {
    for solver in [
        SolverType::HiPO,
        SolverType::NormalEquations,
        SolverType::Auto,
    ] {
        for scaling_iterations in [0, 5] {
            let s = solve(
                m,
                &Options {
                    solver,
                    scaling_iterations,
                    ..Options::default()
                },
            )
            .unwrap();
            assert_eq!(s.status, Status::Optimal, "{s:?}");
            assert!(
                (s.objective - objective).abs() < 1e-6 * (1. + objective.abs()),
                "{s:?}"
            );
            assert!(s.primal_residual < 1e-8, "{s:?}");
        }
    }
}
#[test]
fn mixed_rows() {
    use Sense::*;
    check(
        &model(
            &[&[1., 1.], &[1., 0.]],
            &[3., 1.],
            &[-1., -2.],
            &[0., 0.],
            &[f64::INFINITY; 2],
            &[LessEqual, GreaterEqual],
        ),
        -5.,
    );
    check(
        &model(
            &[&[1., 1.]],
            &[3.],
            &[1., 2.],
            &[0., 0.],
            &[f64::INFINITY; 2],
            &[Equal],
        ),
        3.,
    );
}
#[test]
fn variable_transformations() {
    use Sense::*;
    let inf = f64::INFINITY;
    check(
        &model(
            &[&[1., 1., 1., 1.]],
            &[8.],
            &[1., 0., 3., 0.],
            &[-inf, -inf, 2., 4.],
            &[inf, 3., 5., 4.],
            &[Equal],
        ),
        5.,
    );
    check(
        &model(
            &[&[0., 0., 0.]],
            &[1.],
            &[0., 0., 1.],
            &[-inf, -inf, 0.],
            &[inf, inf, inf],
            &[LessEqual],
        ),
        0.,
    );
}
#[test]
fn finite_bounds_and_empty_rows() {
    check(
        &model(&[], &[], &[-1., 2.], &[2., -3.], &[5., 4.], &[]),
        -11.,
    );
    check(
        &model(
            &[&[1., 1.]],
            &[3.],
            &[-1., -2.],
            &[0., 0.],
            &[2., 2.],
            &[Sense::LessEqual],
        ),
        -5.,
    );
    check(&model(&[], &[], &[2., -3.], &[4., 2.], &[4., 2.], &[]), 2.);
    check(&model(&[&[]], &[0.], &[], &[], &[], &[Sense::Equal]), 0.);
}
#[test]
fn multiple_free_variables_with_zero_cost() {
    check(
        &model(
            &[&[1., 0., 0.], &[0., 1., 0.], &[0., 0., 1.]],
            &[-2., 3., 7.],
            &[0., 0., 1.],
            &[f64::NEG_INFINITY; 3],
            &[f64::INFINITY; 3],
            &[Sense::Equal; 3],
        ),
        7.,
    );
}
#[test]
fn kkt_backends_repeated_factorization() {
    let a = SparseMatrix::from_triplets(
        3,
        4,
        &[
            (0, 0, 2.),
            (1, 0, -1.),
            (1, 1, 3.),
            (2, 1, 1.),
            (0, 2, 1.),
            (2, 2, 4.),
            (1, 3, 2.),
        ],
    )
    .unwrap();
    let expected = vec![1., -2., 3., 4., -1., 0.5, 2.];
    let mut hi = HiPOLDLT::new(a.clone()).unwrap();
    let mut ne = NormalEqLDLT::new(a.clone()).unwrap();
    for scale in [1., 1e-4, 100.] {
        let theta = vec![scale; 4];
        let rp = vec![0.1; 4];
        let rd = vec![0.2; 3];
        let mut rhs: Vec<_> = a
            .transpose_multiply(&expected[4..])
            .unwrap()
            .iter()
            .enumerate()
            .map(|(j, v)| v - (theta[j] + rp[j]) * expected[j])
            .collect();
        rhs.extend(
            a.multiply(&expected[..4])
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, v)| v + rd[i] * expected[4 + i]),
        );
        hi.factorize(&theta, &rp, &rd).unwrap();
        ne.factorize(&theta, &rp, &rd).unwrap();
        let x = hi.solve_with_refinement(&rhs).unwrap().0;
        let (mut z, y) = ne.solve(&rhs[..4], &rhs[4..]).unwrap();
        z.extend(y);
        for solution in [x, z] {
            for (x, e) in solution.iter().zip(&expected) {
                assert!((x - e).abs() < 1e-8, "{solution:?}");
            }
        }
    }
    assert!(hi.factorize(&[], &[], &[]).is_err());
    assert!(hi.solve(&expected).is_err());
    assert!(hi.solve_with_refinement(&expected).is_err());
    // Failed updates invalidate the factor; a valid update must recover using
    // fresh diagonals in the cached sparse matrix.
    hi.factorize(&[2.; 4], &[0.1; 4], &[0.2; 3]).unwrap();
    let mut rhs = a.transpose_multiply(&expected[4..]).unwrap();
    for j in 0..4 {
        rhs[j] -= 2.1 * expected[j];
    }
    rhs.extend(
        a.multiply(&expected[..4])
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.2 * expected[4 + i]),
    );
    for (x, e) in hi.solve_with_refinement(&rhs).unwrap().0.iter().zip(&expected) {
        assert!((x - e).abs() < 1e-8);
    }
}
#[test]
fn invalid_inputs_and_iteration_limit() {
    let mut m = model(&[&[1.]], &[1.], &[1.], &[0.], &[2.], &[Sense::Equal]);
    assert_eq!(
        solve(
            &m,
            &Options {
                max_iterations: 0,
                ..Options::default()
            }
        )
        .unwrap()
        .status,
        Status::IterationLimit
    );
    m.lower[0] = 3.;
    assert!(solve(&m, &Options::default()).is_err());
    assert!(SparseMatrix::from_csr(1, 1, &[0, 2], &[0], &[1.]).is_err());
    assert!(SparseMatrix::from_triplets(1, 1, &[(2, 0, 1.)]).is_err());
}
#[test]
fn certificates() {
    for solver in [SolverType::HiPO, SolverType::NormalEquations] {
        let m = model(
            &[&[1.]],
            &[-1.],
            &[1.],
            &[0.],
            &[f64::INFINITY],
            &[Sense::Equal],
        );
        let s = solve(
            &m,
            &Options {
                solver,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(s.status, Status::Infeasible, "{s:?}");
        assert!(s.certificate.is_some());
        let m = model(
            &[&[0.]],
            &[0.],
            &[-1.],
            &[0.],
            &[f64::INFINITY],
            &[Sense::Equal],
        );
        let s = solve(
            &m,
            &Options {
                solver,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(s.status, Status::Unbounded, "{s:?}");
        assert!(s.certificate.is_some());
    }
}

#[test]
fn cpp_shifted_bound_regression() {
    check(
        &model(
            &[&[1., 1.]],
            &[0.],
            &[-1., 0.],
            &[-2., -10.],
            &[2., 10.],
            &[Sense::Equal],
        ),
        -2.,
    );
    check(
        &model(&[&[1.]], &[1.], &[-1.], &[0.], &[2.], &[Sense::LessEqual]),
        -1.,
    );
    check(
        &model(&[&[1.]], &[1.], &[1.], &[0.], &[2.], &[Sense::GreaterEqual]),
        1.,
    );
}

struct Random(u64);
impl Random {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 32) as f64) / (u32::MAX as f64)
    }
}

#[test]
fn original_dual_signs_and_equilibration() {
    for (sense, cost, expected_dual) in
        [(Sense::GreaterEqual, 1., 1.), (Sense::LessEqual, -1., -1.)]
    {
        let m = model(&[&[1.]], &[2.], &[cost], &[0.], &[5.], &[sense]);
        let s = solve(&m, &Options::default()).unwrap();
        assert_eq!(s.status, Status::Optimal);
        assert!((s.dual[0] - expected_dual).abs() < 1e-7, "{s:?}");
    }
    check(
        &model(
            &[&[1e-6, 1e6]],
            &[1.],
            &[1., 1.],
            &[0., 0.],
            &[f64::INFINITY; 2],
            &[Sense::Equal],
        ),
        1e-6,
    );
}

#[test]
fn csr_canonicalization_and_fill_estimate() {
    let a = SparseMatrix::from_csr(2, 2, &[0, 3, 4], &[1, 0, 1, 1], &[2., 3., -2., 4.]).unwrap();
    assert_eq!(a.nnz(), 2);
    assert_eq!(a.multiply(&[2., 3.]).unwrap(), vec![6., 12.]);
    // A single dense column produces a dense 4x4 normal matrix.
    let a = SparseMatrix::from_triplets(4, 1, &[(0, 0, 1.), (1, 0, 1.), (2, 0, 1.), (3, 0, 1.)])
        .unwrap();
    let fill = baldes::extensions::ipm::estimate_fill(&a);
    assert_eq!(fill.normal_eq_nnz, 10);
    assert_eq!(fill.augmented_nnz, 9);
    assert_eq!(
        baldes::extensions::ipm::SparseSolver::new(a, SolverType::Auto)
            .unwrap()
            .active_solver_type(),
        SolverType::HiPO
    );
}

// Construct LPs with known primal/dual KKT witnesses, so expected objectives
// do not depend on either implementation of the interior-point algorithm.
#[test]
fn seeded_lps_with_known_kkt_witnesses() {
    let mut rng = Random(42);
    for case in 0..30 {
        let n = 5 + case % 11;
        let m = 2 + case % 5;
        let mut entries = vec![];
        for i in 0..m {
            for j in 0..n {
                let v = 2. * rng.next() - 1.;
                if rng.next() > 0.35 {
                    entries.push((i, j, v));
                }
            }
        }
        let a = SparseMatrix::from_triplets(m, n, &entries).unwrap();
        let lower: Vec<_> = (0..n).map(|_| -2. * rng.next()).collect();
        let upper: Vec<_> = lower.iter().map(|l| l + 1. + 3. * rng.next()).collect();
        let x: Vec<_> = (0..n)
            .map(|j| match j % 3 {
                0 => lower[j],
                1 => upper[j],
                _ => 0.5 * (lower[j] + upper[j]),
            })
            .collect();
        let y: Vec<_> = (0..m).map(|_| 2. * rng.next() - 1.).collect();
        let mut c = a.transpose_multiply(&y).unwrap();
        for (j, cj) in c.iter_mut().enumerate() {
            *cj += match j % 3 {
                0 => 0.5 + rng.next(),
                1 => -0.5 - rng.next(),
                _ => 0.,
            };
        }
        let objective: f64 = c.iter().zip(&x).map(|(c, x)| c * x).sum();
        let b = a.multiply(&x).unwrap();
        let lp = ModelData {
            a,
            b,
            c,
            lower,
            upper,
            sense: vec![Sense::Equal; m],
        };
        check(&lp, objective);
    }
}

#[test]
fn seeded_sparse_kkt_residuals() {
    let mut rng = Random(57);
    for case in 0..40 {
        let n = 1 + case % 17;
        let m = 1 + case % 11;
        let mut entries = vec![];
        for j in 0..n {
            for i in 0..m {
                let v = 2. * rng.next() - 1.;
                if rng.next() > 0.6 {
                    entries.push((i, j, v));
                }
            }
        }
        let a = SparseMatrix::from_triplets(m, n, &entries).unwrap();
        let expected: Vec<_> = (0..n + m).map(|_| 2. * rng.next() - 1.).collect();
        let theta: Vec<_> = (0..n).map(|_| 0.1 + 10. * rng.next()).collect();
        let rp = vec![0.1; n];
        let rd = vec![0.2; m];
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
        for kind in [SolverType::HiPO, SolverType::NormalEquations] {
            let mut s = baldes::extensions::ipm::SparseSolver::new(a.clone(), kind).unwrap();
            s.factorize(&theta, &rp, &rd).unwrap();
            let result = s.solve(&rhs).unwrap();
            for (r, e) in result.iter().zip(&expected) {
                assert!((r - e).abs() < 1e-8, "case {case}: {result:?}");
            }
        }
    }
}

#[test]
fn cooperative_time_limit_is_distinct_from_iteration_exhaustion() {
    let m = model(&[&[1.]], &[1.], &[1.], &[0.], &[2.], &[Sense::Equal]);
    let result = solve(&m, &Options {
        time_limit: Some(std::time::Duration::ZERO), ..Default::default()
    }).unwrap();
    assert_eq!(result.status, Status::TimeLimit);
    assert_eq!(result.iterations, 0);
}

#[test]
fn relaxed_optimality_keeps_primal_and_dual_feasibility_tight() {
    // Separate cost and feasibility scales so relaxed and strict gap targets
    // remain distinguishable even with faster adaptive centering.
    let m = model(&[&[1., 1., 1.]], &[1.], &[1., 2., 8e4], &[0.; 3],
        &[f64::INFINITY; 3], &[Sense::Equal]);
    let tight = solve(&m, &Options { tolerance: 1e-9, ..Default::default() }).unwrap();
    let loose = solve(&m, &Options {
        tolerance: 1e-9, gap_tolerance: Some(0.5), ..Default::default()
    }).unwrap();
    assert_eq!(loose.status, Status::Optimal);
    assert!(loose.primal_residual <= 1e-9, "{loose:?}");
    assert!(loose.dual_residual <= 1e-9, "{loose:?}");
    assert!(loose.relative_gap <= 0.5);
    assert!(loose.iterations < tight.iterations, "loose={loose:?}, tight={tight:?}");
    assert!(loose.relative_gap > 1e-9, "{loose:?}");
    for invalid in [0., -1., 1.1, f64::NAN, f64::INFINITY] {
        assert!(solve(&m, &Options {
            gap_tolerance: Some(invalid), ..Default::default()
        }).is_err());
    }
}

fn captured_master(text: &str) -> (ModelData, f64) {
    let mut lines = text.lines();
    let mut header = lines.next().unwrap().split_whitespace();
    let m: usize = header.next().unwrap().parse().unwrap();
    let n: usize = header.next().unwrap().parse().unwrap();
    let gap: f64 = header.next().unwrap().parse().unwrap();
    let b = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
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
    let mut entries = Vec::new();
    let mut c = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for (j, line) in lines.enumerate() {
        let mut fields = line.split_whitespace();
        c.push(fields.next().unwrap().parse().unwrap());
        lower.push(fields.next().unwrap().parse().unwrap());
        upper.push(fields.next().unwrap().parse().unwrap());
        for field in fields {
            let (i, value) = field.split_once(':').unwrap();
            entries.push((i.parse().unwrap(), j, value.parse().unwrap()));
        }
    }
    assert_eq!(c.len(), n);
    let lp = ModelData {
        a: SparseMatrix::from_triplets(m, n, &entries).unwrap(),
        b,
        c,
        lower,
        upper,
        sense,
    };
    (lp, gap)
}

/// Exact master captured immediately before the R201 strict-interior failure.
/// The simplex reference objective is recorded in fixtures/README.md.
#[test]
fn r201_master_with_wide_bounds_converges() {
    let (lp, gap) = captured_master(include_str!("fixtures/r201_master.txt"));
    for gap_tolerance in [Some(gap), None] {
        let s = solve(
            &lp,
            &Options {
                tolerance: 1e-9,
                gap_tolerance,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(s.status, Status::Optimal, "{s:?}");
        assert!(s.iterations < 100, "iterations={}", s.iterations);
        assert!(s.primal_residual <= 1e-9, "{}", s.primal_residual);
        assert!(s.dual_residual <= 1e-9, "{}", s.dual_residual);
        assert!(s.relative_gap <= gap_tolerance.unwrap_or(1e-9));
        assert!(
            (s.objective - 937.6319384978376).abs() < 2e-6,
            "{}",
            s.objective
        );
    }
}

#[test]
fn objective_scaling_recovers_original_duals() {
    for factor in [1e-6, 1., 1e6] {
        let lp = model(
            &[&[1., 1.], &[1., 0.]],
            &[3., 1.],
            &[-factor, -2. * factor],
            &[0., 0.],
            &[1e12, 1e12],
            &[Sense::LessEqual, Sense::GreaterEqual],
        );
        for solver in [SolverType::HiPO, SolverType::NormalEquations] {
            let s = solve(
                &lp,
                &Options {
                    solver,
                    tolerance: 1e-9,
                    ..Options::default()
                },
            )
            .unwrap();
            assert_eq!(s.status, Status::Optimal, "{s:?}");
            assert!(s.primal_residual <= 1e-9);
            assert!(s.dual_residual <= 1e-9);
            assert!(s.relative_gap <= 1e-9);
            assert!((s.objective + 5. * factor).abs() < 1e-7 * factor.max(1.));
            for (&actual, expected) in s.dual.iter().zip([-2. * factor, factor]) {
                assert!((actual - expected).abs() < 1e-7 * factor.max(1.), "{s:?}");
            }
        }
    }
}

/// Master captured before the C202 rank-1 triangular solve overflow.
/// The simplex reference objective is recorded in fixtures/README.md.
#[test]
fn c202_rank1_master_converges() {
    let (lp, gap) = captured_master(include_str!("fixtures/c202_rank1_master.txt"));
    for gap_tolerance in [Some(gap), None] {
        let s = solve(
            &lp,
            &Options {
                tolerance: 1e-9,
                gap_tolerance,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(s.status, Status::Optimal, "{s:?}");
        assert!(s.iterations < 100, "iterations={}", s.iterations);
        assert!(s.primal_residual <= 1e-9, "{}", s.primal_residual);
        assert!(s.dual_residual <= 1e-9, "{}", s.dual_residual);
        assert!(s.relative_gap <= gap_tolerance.unwrap_or(1e-9));
        assert!((s.objective - 618.2263546557791).abs() < 2e-6, "{}", s.objective);
    }
}

#[test]
fn persistent_solver_reuses_analysis_and_invalidates_changed_coordinates() {
    for kind in [SolverType::HiPO, SolverType::NormalEquations, SolverType::Auto] {
        let mut solver = Solver::default();
        let mut options = Options { tolerance: 1e-9, solver: kind, ..Options::default() };
        let mut lp = model(&[&[2., 1.]], &[2.], &[3., 2.], &[0., 0.], &[10., 10.], &[Sense::Equal]);
        let first = solver.solve(&lp, &options).unwrap();
        assert_eq!(first.status, Status::Optimal);
        let repeated = solver.solve(&lp, &options).unwrap();
        assert_eq!(repeated.status, Status::Optimal);
        assert_eq!(repeated.iterations, 0);
        assert_eq!(solver.statistics().symbolic_analyses, 1);
        assert_eq!(solver.statistics().warm_starts, 1);
        lp.c[0] = 5.;
        lp.b[0] = 3.;
        let edited = solver.solve(&lp, &options).unwrap();
        assert_eq!(edited.status, Status::Optimal);
        assert!((edited.objective - 6.).abs() < 1e-7);
        assert_eq!(solver.statistics().symbolic_analyses, 1);
        assert_eq!(solver.statistics().warm_starts, 1);
        lp.upper[1] = 0.; // fixes a variable and changes recovery indices
        let fixed = solver.solve(&lp, &options).unwrap();
        assert_eq!(fixed.status, Status::Optimal);
        assert!((fixed.objective - 7.5).abs() < 1e-7);
        assert_eq!(solver.statistics().symbolic_analyses, 2);
        options.scaling_iterations = 0;
        assert_eq!(solver.solve(&lp, &options).unwrap().status, Status::Optimal);
        assert_eq!(solver.statistics().symbolic_analyses, 3);
        assert_eq!(solver.statistics().warm_starts, 1);
        lp.a = SparseMatrix::from_triplets(1, 2, &[(0, 0, 3.), (0, 1, 1.)]).unwrap();
        let changed = solver.solve(&lp, &options).unwrap();
        assert_eq!(changed.status, Status::Optimal);
        assert!((changed.objective - 5.).abs() < 1e-7);
        assert_eq!(solver.statistics().symbolic_analyses, 4);
    }
}

#[test]
fn persistent_solver_invalidates_appended_columns_and_inequality_slacks() {
    for kind in [SolverType::HiPO, SolverType::NormalEquations] {
        let options = Options { tolerance: 1e-9, solver: kind, ..Options::default() };
        let mut solver = Solver::default();
        let first = model(&[&[2., 1.], &[1., 0.]], &[2., 0.8], &[3., 2.],
            &[0., 0.], &[10., 10.], &[Sense::Equal, Sense::LessEqual]);
        assert_eq!(solver.solve(&first, &options).unwrap().status, Status::Optimal);
        // New column also changes equilibration; old slack indices move.
        let grown = model(&[&[2., 1., 100.], &[1., 0., 2.]], &[2., 0.8], &[3., 2., 1.],
            &[0., 0., 0.], &[10., 10., 10.], &[Sense::Equal, Sense::LessEqual]);
        let warm = solver.solve(&grown, &options).unwrap();
        let cold = solve(&grown, &options).unwrap();
        assert_eq!(warm.status, Status::Optimal);
        assert!(warm.primal_residual <= 1e-9);
        assert!(warm.dual_residual <= 1e-9);
        assert!((warm.objective - cold.objective).abs() < 1e-7);
        assert_eq!(solver.statistics().warm_starts, 0);
        assert_eq!(solver.statistics().symbolic_analyses + solver.statistics().symbolic_extensions, 2);
        let mut fork = solver.clone();
        let mut edited = grown.clone();
        edited.c[2] = 1000.;
        let changed = fork.solve(&edited, &options).unwrap();
        assert_eq!(changed.status, Status::Optimal);
        assert!((changed.objective - warm.objective).abs() > 1.);
        assert_eq!(solver.solve(&grown, &options).unwrap().iterations, 0);
    }
}

#[test]
fn persistent_solver_preserves_budget_and_recovers_after_failed_calls() {
    let mut solver = Solver::default();
    let lp = model(&[&[1.]], &[1.], &[1.], &[0.], &[10.], &[Sense::Equal]);
    let options = Options { tolerance: 1e-9, ..Options::default() };
    assert_eq!(solver.solve(&lp, &options).unwrap().status, Status::Optimal);
    let mut grown = model(&[&[1., 1.]], &[1.], &[1., 0.5], &[0., 0.], &[10., 10.], &[Sense::Equal]);
    let limited = solver.solve(&grown, &Options { max_iterations: 1, ..options.clone() }).unwrap();
    assert_eq!(limited.status, Status::IterationLimit);
    assert!(limited.iterations <= 1);
    assert_eq!(solver.solve(&grown, &options).unwrap().status, Status::Optimal);
    let timed = solver.solve(&grown, &Options { time_limit: Some(std::time::Duration::ZERO), ..options.clone() }).unwrap();
    assert_eq!(timed.status, Status::TimeLimit);
    grown.b[0] = f64::NAN;
    assert!(solver.solve(&grown, &options).is_err());
    grown.b[0] = 1.;
    assert_eq!(solver.solve(&grown, &options).unwrap().status, Status::Optimal);
}

#[test]
fn persistent_rhs_and_cost_updates_preserve_bound_transformations() {
    for (lower, upper) in [(2., 5.), (f64::NEG_INFINITY, 5.), (f64::NEG_INFINITY, f64::INFINITY)] {
        let mut solver = Solver::default();
        let mut lp = model(&[&[2.]], &[6.], &[3.], &[lower], &[upper], &[Sense::Equal]);
        let options = Options { tolerance: 1e-9, ..Options::default() };
        assert_eq!(solver.solve(&lp, &options).unwrap().status, Status::Optimal);
        lp.b[0] = 8.; lp.c[0] = -2.;
        let solution = solver.solve(&lp, &options).unwrap();
        assert_eq!(solution.status, Status::Optimal);
        assert!((solution.objective + 8.).abs() < 1e-7);
        assert!((solution.dual[0] + 1.).abs() < 1e-7);
        assert_eq!(solver.statistics().symbolic_analyses, 1);
    }
}

#[test]
fn persistent_solver_extends_covered_column_cliques_without_new_analysis() {
    let mut solver = Solver::default();
    let options = Options { tolerance: 1e-9, ..Options::default() };
    let first = model(&[&[1., 1.], &[1., -1.]], &[2., 0.], &[3., 2.],
        &[0., 0.], &[10., 10.], &[Sense::Equal, Sense::Equal]);
    assert_eq!(solver.solve(&first, &options).unwrap().status, Status::Optimal);
    let grown = model(&[&[1., 1., 2.], &[1., -1., 0.]], &[2., 0.], &[3., 2., 0.5],
        &[0., 0., 0.], &[10., 10., 10.], &[Sense::Equal, Sense::Equal]);
    let solution = solver.solve(&grown, &options).unwrap();
    assert_eq!(solution.status, Status::Optimal);
    assert!((solution.objective - 0.5).abs() < 1e-7);
    assert_eq!(solver.statistics().symbolic_analyses, 1);
    assert_eq!(solver.statistics().symbolic_extensions, 1);
    assert_eq!(solver.statistics().warm_starts, 0);
}

#[test]
fn persistent_column_order_preserves_free_fixed_and_slack_recovery() {
    let mut solver = Solver::default();
    let options = Options { tolerance: 1e-9, ..Options::default() };
    let initial = model(&[&[1., 1., 1.], &[0., 0., 1.]], &[4., 10.], &[1., 2., 3.],
        &[f64::NEG_INFINITY, 2., 0.], &[f64::INFINITY, 2., 10.],
        &[Sense::Equal, Sense::LessEqual]);
    assert_eq!(solver.solve(&initial, &options).unwrap().status, Status::Optimal);
    let grown = model(&[&[1., 1., 1., 100.], &[0., 0., 1., 0.]], &[4., 10.], &[1., 2., 3., 40.],
        &[f64::NEG_INFINITY, 2., 0., 0.], &[f64::INFINITY, 2., 10., 0.01],
        &[Sense::Equal, Sense::LessEqual]);
    let solution = solver.solve(&grown, &options).unwrap();
    assert_eq!(solution.status, Status::Optimal);
    assert!(solution.primal_residual <= 1e-9);
    assert!((solution.objective - 5.4).abs() < 1e-7);
    assert!((solution.primal[0] - 1.).abs() < 1e-7);
    assert_eq!(solution.primal[1], 2.);
    assert_eq!(solver.statistics().symbolic_extensions, 1);
    assert_eq!(solver.solve(&grown, &options).unwrap().iterations, 0);
}
