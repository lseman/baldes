use super::*;
use crate::extensions::ipm::{Parallel, ParallelType};

#[test]
fn dependency_scheduler_preserves_unbalanced_tree_results_and_recovers_after_failure() {
    let n = 197;
    let mut entries = Vec::new();
    let mut offset = 0;
    for size in [19, 47, 131] {
        for i in offset..offset + size {
            entries.push((i, i, 5.));
            for j in i + 1..(i + 4).min(offset + size) {
                entries.push((i, j, 0.1));
                entries.push((j, i, 0.1));
            }
        }
        offset += size;
    }
    let a = SparseMatrix::from_triplets(n, n, &entries).unwrap();
    let symbolic = Symbolic::analyze(&a, (0..n).collect(), n);
    assert!(symbolic.fronts.len() >= 64);
    assert!(symbolic.levels.len() > 1);
    let expected: Vec<_> = (0..n).map(|i| (i % 11) as f64 / 11. - 0.5).collect();
    let rhs = a.mul(&expected);
    for kind in [ParallelType::Tree, ParallelType::Both] {
        let execution = Execution::new(Parallel::On, kind, 3).unwrap();
        let mut work = FactorWorkspace::default();
        let factor = symbolic
            .factorize_dependencies(
                &a,
                false,
                &execution,
                &mut work,
                kind == ParallelType::Both,
                3,
            )
            .unwrap();
        let mut zero = a.clone();
        zero.val.fill(0.);
        drop(factor);
        assert!(symbolic
            .factorize_dependencies(
                &zero,
                false,
                &execution,
                &mut work,
                kind == ParallelType::Both,
                3
            )
            .is_err());
        for _ in 0..3 {
            let factor = symbolic
                .factorize_dependencies(
                    &a,
                    false,
                    &execution,
                    &mut work,
                    kind == ParallelType::Both,
                    3,
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
            assert_eq!(work.dense.len(), 3);
            assert!(work.spare.bytes <= SCHUR_CACHE_BYTES);
        }
    }
}

#[test]
fn dependency_scheduler_allows_nested_node_work_without_holding_workspace_locks() {
    let components = 5;
    let block_rows = 200;
    let rows = components * block_rows;
    let n = rows + components;
    let mut entries: Vec<_> = (0..components).map(|i| (i, i, -1.)).collect();
    for i in 0..rows {
        let j = i / block_rows;
        entries.push((i + components, i + components, 1.));
        entries.push((j, i + components, 0.2));
        entries.push((i + components, j, 0.2));
    }
    let a = SparseMatrix::from_triplets(n, n, &entries).unwrap();
    let symbolic = Symbolic::analyze(&a, (0..n).collect(), components);
    let expected: Vec<_> = (0..n).map(|i| (i % 13) as f64 / 13. - 0.5).collect();
    let rhs = a.mul(&expected);
    for threads in [2, 3] {
        assert!(
            components > threads,
            "ready fronts must exceed the buffer budget"
        );
        let execution = Execution::new(Parallel::On, ParallelType::Both, threads).unwrap();
        let mut work = FactorWorkspace::default();
        let factor = symbolic
            .factorize_dependencies(&a, false, &execution, &mut work, true, threads)
            .unwrap();
        let actual = factor.solve(&rhs).unwrap();
        assert_eq!(work.dense.len(), threads);
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
