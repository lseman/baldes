use super::*;

#[test]
fn reused_solve_workspace_matches_full_kkt_residuals_after_errors_and_resizing() {
    let mut workspace = SolveWorkspace::default();
    for (m, n) in [(5, 20), (1, 3), (0, 0), (7, 24)] {
        let mut entries = Vec::new();
        for j in 0..n {
            for i in 0..m {
                if (i + 3 * j) % 4 != 0 {
                    entries.push((i, j, ((i + j) % 7) as f64 / 9. - 0.25));
                }
            }
        }
        let a = SparseMatrix::from_triplets(m, n, &entries).unwrap();
        let mut solver = SparseSolver::new(a.clone(), SolverType::HiPO).unwrap();
        for scale in [0.1, 10.] {
            let theta = vec![scale; n];
            let rp = vec![0.1; n];
            let rd = vec![0.2; m];
            solver.factorize(&theta, &rp, &rd).unwrap();
            let matrix = augmented(&a, &theta, &rp, &rd).unwrap();
            let expected: Vec<_> = (0..n + m).map(|i| i as f64 / 13. - 0.5).collect();
            let rhs = matrix.mul(&expected);
            assert!(solver
                .solve_into(&vec![0.; n + m + 1], &mut workspace)
                .is_err());
            if n + m > 0 {
                assert!(solver
                    .solve_into(&vec![f64::NAN; n + m], &mut workspace)
                    .is_err());
            }
            workspace.permuted.fill(f64::NAN);
            workspace.solution.fill(f64::NAN);
            workspace.residual.fill(f64::NAN);
            workspace.denominator.fill(f64::NAN);
            workspace.correction.fill(f64::NAN);
            solver.solve_into(&rhs, &mut workspace).unwrap();
            for (&actual, &want) in workspace.solution.iter().zip(&expected) {
                assert!((actual - want).abs() < 1e-9);
            }
            let full_residual: Vec<_> = rhs
                .iter()
                .zip(matrix.mul(&workspace.solution))
                .map(|(b, ax)| b - ax)
                .collect();
            let mut full_denominator: Vec<_> = rhs.iter().map(|b| b.abs()).collect();
            for (j, x) in workspace.solution.iter().enumerate() {
                for (i, v) in matrix.column(j) {
                    full_denominator[i] += v.abs() * x.abs();
                }
            }
            assert_eq!(workspace.residual, full_residual);
            assert_eq!(workspace.denominator, full_denominator);
        }
    }
}

#[test]
fn appended_primal_symbolics_preserve_kkt_solutions_and_reject_new_fill() {
    let mut entries = Vec::new();
    for j in 0..8 {
        entries.push((j % 3, j, 1.));
        entries.push((j % 3 + 1, j, 2.));
    }
    let a = SparseMatrix::from_triplets(4, 8, &entries).unwrap();
    let original = HiPOLDLT::new(a).unwrap();
    entries.extend([(0, 8, 0.5), (1, 8, 3.)]);
    // New numerical values can include changed equilibration on old entries.
    for entry in &mut entries {
        entry.2 *= 0.25;
    }
    let a = SparseMatrix::from_triplets(4, 9, &entries).unwrap();
    let extended = original.with_appended_columns(a.clone()).unwrap().unwrap();
    for uplooking in [true, false] {
        let mut solver = extended.clone();
        if !uplooking {
            solver.uplooking = None;
        }
        let expected: Vec<_> = (0..13).map(|i| i as f64 / 7. - 0.5).collect();
        for diagonal in [0.1, 3.] {
            let theta = vec![diagonal; 9];
            let rp = vec![0.1; 9];
            let rd = vec![0.2; 4];
            let matrix = augmented(&a, &theta, &rp, &rd).unwrap();
            let rhs = matrix.mul(&expected);
            solver.factorize(&theta, &rp, &rd).unwrap();
            let solution = solver.solve_with_refinement(&rhs).unwrap().0;
            for (&x, &want) in solution.iter().zip(&expected) {
                assert!((x - want).abs() < 1e-9, "{x} != {want}");
            }
        }
    }
    entries.push((2, 9, 1.));
    let a = SparseMatrix::from_triplets(4, 10, &entries).unwrap();
    let extended = extended.with_appended_columns(a).unwrap().unwrap();
    entries.extend([(0, 10, 1.), (3, 10, 1.)]);
    let new_fill = SparseMatrix::from_triplets(4, 11, &entries).unwrap();
    assert!(extended.with_appended_columns(new_fill).unwrap().is_none());
    // A structural edit to an existing column also invalidates extension.
    entries.push((3, 0, 1.));
    let edited = SparseMatrix::from_triplets(4, 11, &entries).unwrap();
    assert!(extended.with_appended_columns(edited).unwrap().is_none());
}

#[test]
fn cached_normal_equations_refactor_and_recover_after_failure() {
    let a = SparseMatrix::from_triplets(
        3,
        4,
        &[
            (0, 0, 1.),
            (2, 0, -2.),
            (1, 1, 3.),
            (0, 2, 0.5),
            (1, 2, -1.),
            (2, 3, 2.),
        ],
    )
    .unwrap();
    let mut solver = NormalEqLDLT::new(a.clone()).unwrap();
    let expected = [0.5, -1., 2., 0.25, 1., -0.5, 0.75];
    for theta in [[1., 2., 3., 4.], [4., 0.5, 2., 1.]] {
        let rp = [0.1; 4];
        let rd = [0.2; 3];
        solver.factorize(&theta, &rp, &rd).unwrap();
        let rhs = augmented(&a, &theta, &rp, &rd).unwrap().mul(&expected);
        let (x, y) = solver.solve(&rhs[..4], &rhs[4..]).unwrap();
        for (actual, expected) in x.iter().chain(&y).zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
        assert!(solver.factorize(&theta, &[0.; 4], &rd).is_err());
        assert_eq!(
            solver.solve(&rhs[..4], &rhs[4..]).unwrap_err(),
            Error::NotFactorized
        );
    }
}

#[test]
fn sparse_ldl_clones_share_only_immutable_structure() {
    let n = 64;
    let entries: Vec<_> = (0..n).map(|j| (0, j, 1.)).collect();
    let mut original = HiPOLDLT::new(SparseMatrix::from_triplets(1, n, &entries).unwrap()).unwrap();
    assert!(original.uplooking.is_some());
    original
        .factorize(&vec![1.; n], &vec![0.1; n], &[0.2])
        .unwrap();
    let mut sibling = original.clone();
    sibling
        .factorize(&vec![2.; n], &vec![0.1; n], &[0.2])
        .unwrap();
    let mut rhs = vec![-0.1; n];
    rhs.push(n as f64 + 0.2);
    for x in original.solve(&rhs).unwrap() {
        assert!((x - 1.).abs() < 1e-10);
    }
    let mut rhs = vec![-0.8; n];
    rhs.push(n as f64 * 0.5 + 0.05);
    let x = sibling.solve(&rhs).unwrap();
    for &x in &x[..n] {
        assert!((x - 0.5).abs() < 1e-10);
    }
    assert!((x[n] - 0.25).abs() < 1e-10);
}

#[test]
fn sparse_ldl_unsafe_pivots_retry_pivoted_factorization_and_invalidate_on_error() {
    let n = 64;
    let a = SparseMatrix::from_triplets(1, n, &[]).unwrap();
    let mut solver = HiPOLDLT::new(a).unwrap();
    assert!(solver.uplooking.is_some());
    solver
        .factorize(&vec![1.; n], &vec![0.1; n], &[0.2])
        .unwrap();
    assert_eq!(solver.perturbed_pivots(), 0);
    // The fast path rejects these tiny pivots. The existing multifrontal
    // path must regularize them instead of publishing an invalid factor.
    solver
        .factorize(&vec![1e-20; n], &vec![1e-20; n], &[1e-20])
        .unwrap();
    assert_eq!(solver.perturbed_pivots(), n + 1);
    assert_eq!(solver.solve(&vec![0.; n + 1]).unwrap(), vec![0.; n + 1]);
    // The perturbed factor is finite but does not accurately solve the
    // requested system. Refinement must report failure to its caller.
    assert!(matches!(
        solver.solve_with_refinement(&vec![1.; n + 1]),
        Err(Error::NumericalFailure(_))
    ));
    assert!(solver.factorize(&vec![1.; n], &vec![0.; n], &[1.]).is_err());
    assert_eq!(
        solver.solve(&vec![0.; n + 1]).unwrap_err(),
        Error::NotFactorized
    );
}
