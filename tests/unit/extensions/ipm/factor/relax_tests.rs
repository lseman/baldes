use super::*;

#[test]
fn repeated_relaxation_does_not_replenish_the_artificial_fill_budget() {
    let n = 192;
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, 20.));
        for j in i + 1..(i + 25).min(n) {
            entries.push((i, j, 0.1));
            entries.push((j, i, 0.1));
        }
    }
    let a = SparseMatrix::from_triplets(n, n, &entries).unwrap();
    let mut symbolic = Symbolic::analyze(&a, (0..n).collect(), n);
    let before = symbolic.plan();
    symbolic.relax(n);
    symbolic.prepare_assembly(&a);
    let plan = symbolic.plan();
    assert!(plan.fronts < before.fronts);
    let assembly = Arc::clone(&symbolic.assembly);
    for _ in 0..3 {
        symbolic.relax(n);
        assert_eq!(symbolic.plan(), plan);
        assert!(Arc::ptr_eq(&symbolic.assembly, &assembly));
    }
}

#[test]
fn bounded_relaxation_preserves_boundary_storage_and_kkt_solution() {
    let n = 256;
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, 5.));
        for j in i + 1..(i + 5).min(n) {
            entries.push((i, j, 0.1));
            entries.push((j, i, 0.1));
        }
    }
    let a = SparseMatrix::from_triplets(n, n, &entries).unwrap();
    let count = |s: &Symbolic| {
        s.fronts
            .iter()
            .map(|f| {
                let width = f.end - f.lo;
                width * (width - 1) / 2 + width * f.updates.len()
            })
            .sum::<usize>()
    };
    for boundary in [16, n] {
        let original = Symbolic::analyze(&a, (0..n).collect(), boundary);
        let mut relaxed = original.clone();
        relaxed.relax(boundary);
        assert!(relaxed.fronts.len() < original.fronts.len());
        assert_eq!(relaxed.order, original.order);
        assert!(count(&relaxed) <= count(&original) + count(&original) / 8);
        if boundary != n {
            for (before, after) in original.fronts.iter().zip(&relaxed.fronts).take(boundary) {
                assert_eq!(
                    (before.lo, before.end, &before.updates),
                    (after.lo, after.end, &after.updates)
                );
            }
        }
        assert!(relaxed.fronts.iter().all(|f| f.end - f.lo <= 64));
        assert!(relaxed.assembly_is_missing());
        relaxed.prepare_assembly(&a);
        assert!(!Arc::ptr_eq(&original.assembly, &relaxed.assembly));
        let expected: Vec<_> = (0..n).map(|i| (i % 17) as f64 / 17. - 0.5).collect();
        let rhs = a.mul(&expected);
        for plan in [&original, &relaxed] {
            let factor = plan
                .factorize_cached(
                    &a,
                    false,
                    &Execution::default(),
                    &mut FactorWorkspace::default(),
                    None,
                )
                .unwrap();
            let actual = factor.solve(&rhs).unwrap();
            assert!(actual
                .iter()
                .zip(&expected)
                .all(|(x, y)| (x - y).abs() < 1e-12));
            assert!(a
                .mul(&actual)
                .iter()
                .zip(&rhs)
                .all(|(x, y)| (x - y).abs() < 1e-12));
        }
    }
}
