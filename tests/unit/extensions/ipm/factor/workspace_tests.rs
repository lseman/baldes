use super::*;

#[cfg(feature = "ipm-blocked")]
#[test]
fn blocked_overflow_reassembles_and_recovers_with_scalar_subtractions() {
    let n = 129;
    let np = 64;
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, if i == 0 { -1. } else { 1. }));
    }
    for i in 1..=np {
        for j in i + 1..=np {
            entries.push((i, j, 1e-300));
            entries.push((j, i, 1e-300));
        }
    }
    for i in np + 1..n {
        let value = if i == n - 1 { 1e154 } else { 0.1 };
        entries.push((1, i, value));
        entries.push((i, 1, value));
    }
    entries.push((2, n - 1, 1e154));
    entries.push((n - 1, 2, 1e154));
    // A negative child pivot contributes +1.5e308 to the parent update.
    let child_value = (1.5e308_f64).sqrt();
    for (j, value) in [(1, 1e-150), (n - 1, child_value)] {
        entries.push((0, j, value));
        entries.push((j, 0, value));
    }
    let a = SparseMatrix::from_triplets(n, n, &entries).unwrap();
    let symbolic = Symbolic::analyze(&a, (0..n).collect(), np + 1);
    assert_eq!((symbolic.fronts[1].lo, symbolic.fronts[1].end), (1, np + 1));
    assert_eq!(symbolic.fronts[1].updates.len(), np);
    let mut scratch = DenseWorkspace::default();
    let mut child = FrontWork::new(0, &symbolic.fronts[0]);
    symbolic
        .compute_front_values(&a, false, false, |_| None, &mut scratch, &mut child)
        .unwrap();
    scratch.matrix = vec![f64::NAN; (2 * np) * (2 * np)];
    let mut task = FrontWork::new(1, &symbolic.fronts[1]);
    symbolic
        .compute_front_values(
            &a,
            false,
            false,
            |_| Some(child.update.as_slice()),
            &mut scratch,
            &mut task,
        )
        .unwrap();
    assert!(
        !scratch.multipliers.used_blocked,
        "the scalar retry must run"
    );
    assert!(task.update.iter().all(|x| x.is_finite()));
    // Two products of 1e308 overflow when summed; sequential subtraction
    // from the child contribution leaves a finite negative Schur diagonal.
    let last = *task.update.last().unwrap();
    assert!(last < 0.);
    assert!((last / -5e307 - 1.).abs() < 1e-14);
}

#[test]
fn consumed_schur_storage_is_reused_with_a_bounded_cache() {
    let mut pool = SchurPool::default();
    let entries = SCHUR_CACHE_BYTES / std::mem::size_of::<f64>() / 2;
    let first = vec![3.; entries];
    let second = vec![4.; entries];
    let pointers = [first.as_ptr(), second.as_ptr()];
    pool.put(first);
    pool.put(second);
    assert_eq!(pool.bytes, SCHUR_CACHE_BYTES);
    pool.put(vec![5.; entries]);
    assert_eq!(pool.bytes, SCHUR_CACHE_BYTES);
    let reused = pool.take(entries / 2);
    assert!(reused.is_empty());
    assert!(pointers.contains(&reused.as_ptr()));
    assert_eq!(pool.bytes, SCHUR_CACHE_BYTES / 2);
    pool.put(reused);
    assert_eq!(pool.bytes, SCHUR_CACHE_BYTES);
    // Empty roots and requests larger than the cache need no cached buffer.
    assert_eq!(pool.take(0).capacity(), 0);
    assert_eq!(pool.take(2 * entries + 1).capacity(), 0);
    assert_eq!(pool.bytes, SCHUR_CACHE_BYTES);
}
