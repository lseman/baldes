use super::*;

#[test]
fn rng_matches_cpp_reference() {
    // Values from BALDES's C++ math/RNG.h and libstdc++ 16 std::shuffle.
    let mut rng = Xoroshiro128Plus::new(0);
    assert_eq!(
        [rng.next(), rng.next(), rng.next()],
        [
            2836985337983649843,
            17383111449422238058,
            15852657771511707500
        ]
    );
    assert_eq!(rng.next_u64(100), 86);
    let mut shuffled: Vec<Vec<usize>> = [1, 2, 10].iter().map(|&n| (0..n).collect()).collect();
    for v in &mut shuffled {
        rng.shuffle(v);
    }
    assert_eq!(
        shuffled,
        [vec![0], vec![1, 0], vec![7, 5, 9, 2, 3, 4, 1, 0, 8, 6]]
    );
}
#[test]
fn full_width_seeds_match_archived_cpp() {
    for (seed, expected) in [
        (
            1_u64,
            [
                10197084890663893492,
                16183632380530289291,
                17854549295427883785,
            ],
        ),
        (
            42_u64,
            [
                12966046507894987950,
                3238562683063015607,
                4709601321616383589,
            ],
        ),
        (
            9223372036854775808_u64,
            [
                7868295714955407760,
                1665575697381515169,
                10600772145784059665,
            ],
        ),
        (
            18446744073709551615_u64,
            [
                16200281118964062572,
                18254272431674763336,
                1028928092969063525,
            ],
        ),
    ] {
        let mut rng = Xoroshiro128Plus::from_seed(seed);
        assert_eq!([rng.next(), rng.next(), rng.next()], expected);
    }
    for seed in [i32::MIN, -1, 0, 1, i32::MAX] {
        let mut signed = Xoroshiro128Plus::new(seed);
        let mut unsigned = Xoroshiro128Plus::from_seed(seed as i64 as u64);
        for _ in 0..100 {
            assert_eq!(signed.next(), unsigned.next());
        }
    }
}

#[test]
fn distributions_handle_extreme_bounds() {
    let mut rng = Xoroshiro128Plus::new(42);
    for _ in 0..1000 {
        for bound in [1, 2, 3, (1 << 63) + 1, u64::MAX] {
            assert!(rng.uniform_below(bound) < bound);
            assert!(rng.next_u64(bound) < bound);
        }
        for (low, high) in [
            (i32::MIN, i32::MAX),
            (i32::MIN, i32::MIN),
            (i32::MAX, i32::MAX),
            (-5, 5),
        ] {
            assert!((low..=high).contains(&rng.uniform_i32(low, high)));
        }
        assert!((0.0..1.0).contains(&rng.next_f64()));
    }
}

#[test]
#[should_panic(expected = "random bound must be positive")]
fn uniform_zero_bound_is_rejected() {
    Xoroshiro128Plus::new(0).uniform_below(0);
}

#[test]
#[should_panic(expected = "random bound must be positive")]
fn modulo_zero_bound_is_rejected() {
    Xoroshiro128Plus::new(0).next_u64(0);
}

#[test]
#[should_panic(expected = "random interval must be ordered")]
fn reversed_interval_is_rejected() {
    Xoroshiro128Plus::new(0).uniform_i32(1, -1);
}

#[test]
fn shuffle_handles_empty_singleton_odd_and_even_lengths() {
    for n in [0, 1, 2, 3, 10, 31] {
        let mut rng = Xoroshiro128Plus::new(42);
        let mut values: Vec<_> = (0..n).collect();
        rng.shuffle(&mut values);
        values.sort_unstable();
        assert_eq!(values, (0..n).collect::<Vec<_>>());
    }
}
