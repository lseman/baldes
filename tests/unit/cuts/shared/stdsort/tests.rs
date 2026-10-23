#[test]
fn stable_sorts() {
    let mut v: Vec<(i32, usize)> = (0..200).map(|i| ((i * 7919) % 13, i as usize)).collect();
    super::stable_sort(&mut v, |a, b| a.0 < b.0);
    assert!(v
        .windows(2)
        .all(|w| w[0].0 < w[1].0 || (w[0].0 == w[1].0 && w[0].1 < w[1].1)));
}
#[test]
fn sorts() {
    let mut v: Vec<i32> = (0..200).map(|i| (i * 7919) % 101).collect();
    super::sort(&mut v, |a, b| a < b);
    assert!(v.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn borrowed_merge_buffer_preserves_tolerant_comparator_candidate_order() {
    // Golden permutations from the preceding snapshot-based merge algorithm.
    // This comparator is deliberately tolerant, like cut candidate ranking;
    // replacing the sorting algorithm can change its observable output order.
    for (n, expected) in [
        (0, 0xcbf29ce484222325_u64),
        (1, 0xa8c7f832281a39c5),
        (7, 0xbccaf249d31cf2e2),
        (8, 0x9bde4558632debe5),
        (14, 0x3284bef9fb084644),
        (15, 0x948577de695843ea),
        (16, 0xe15a94a210c94645),
        (64, 0xcc6dfbb7597a3aa5),
        (128, 0x2867bc9068859e05),
        (1024, 0x3b5ea571eb2629f1),
    ] {
        let mut values: Vec<_> = (0..n)
            .map(|i| (((i * 7919) % 31) as f64 * 0.75e-6, i))
            .collect();
        super::stable_sort(&mut values, |a, b| {
            if a.0 > b.0 + 1e-6 {
                true
            } else if b.0 - 1e-6 > a.0 {
                false
            } else {
                a.1 < b.1
            }
        });
        let actual = values
            .iter()
            .fold(0xcbf29ce484222325_u64, |hash, &(_, id)| {
                (id as u64)
                    .to_le_bytes()
                    .into_iter()
                    .fold(hash, |hash, byte| {
                        (hash ^ byte as u64).wrapping_mul(0x100000001b3)
                    })
            });
        assert_eq!(actual, expected, "candidate count {n}");
    }
}
