use super::*;

fn compare<const P: bool, const F: bool, const R: usize, const N: bool>() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let values = [
        0.,
        -0.,
        EPSILON,
        -EPSILON,
        1.,
        1. + EPSILON,
        1. - EPSILON,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::from_bits(1),
        -f64::from_bits(1),
    ];
    let mut columns = HotColumns::<2>::default();
    for i in 0..137 {
        columns.push(HotLabel {
            key: values[i % values.len()],
            cost: 0.,
            additional_cost: 0.,
            resources: [
                values[(i * 7) % values.len()],
                values[(i * 5 + 1) % values.len()],
            ],
            // Spread masks over the whole configured width, including its top bit.
            ng: (i as NgMask).wrapping_mul(7919).rotate_left(i as u32),
            hops: 0,
        });
    }
    for len in [0, 1, 7, 8, 15, 16, 17, 31, 32, 65, 137] {
        let mut c = columns.clone();
        c.truncate(len);
        for from in 0..=len {
            for &a in &values {
                for &b in &values {
                    for ng in [0, 1, 0x5555, NgMask::MAX >> 1, NgMask::MAX] {
                        let expected = (from..len)
                            .find(|&i| {
                                let mut pass = true;
                                for r in 0..R {
                                    let candidate = if r == 0 { a } else { b };
                                    let (x, y) = if P {
                                        (c.resources[r][i], candidate)
                                    } else {
                                        (candidate, c.resources[r][i])
                                    };
                                    pass &= if F {
                                        x <= y + EPSILON
                                    } else {
                                        y - EPSILON <= x
                                    };
                                }
                                if N {
                                    let (x, y) = if P { (c.ng[i], ng) } else { (ng, c.ng[i]) };
                                    pass &= x & !y == 0;
                                }
                                pass || (P && c.key[i] > a)
                            })
                            .unwrap_or(len);
                        // SAFETY: CPU support checked at entry; kernel validates lengths before loading.
                        let actual =
                            unsafe { c.next_candidate_avx2::<P, F, R, N>(from, a, a, b, ng) };
                        assert_eq!(actual, expected);
                        #[cfg(feature = "simd-avx512")]
                        if std::is_x86_feature_detected!("avx512f") {
                            // SAFETY: CPU support checked; kernel validates lengths.
                            let wide =
                                unsafe { c.next_candidate_avx512::<P, F, R, N>(from, a, a, b, ng) };
                            assert_eq!(wide, expected);
                        }
                        assert_eq!(c.next_candidate::<P, F, R, N>(from, a, a, b, ng), expected);
                    }
                }
            }
        }
    }
}

#[test]
fn malformed_public_columns_panic_before_vector_loads() {
    let mut c = HotColumns::<2>::default();
    c.key.resize(16, 0.);
    assert!(std::panic::catch_unwind(|| {
        c.next_candidate::<true, true, 2, true>(0, 0., 0., 0., 0)
    })
    .is_err());
}

#[test]
#[ignore = "manual timing experiment"]
fn measure_dispatch() {
    use std::{hint::black_box, time::Instant};
    for n in [16, 64, 256, 1024] {
        for early in [false, true] {
            let mut c = HotColumns::<2>::default();
            for i in 0..n {
                c.push(HotLabel {
                    key: 0.,
                    cost: 0.,
                    additional_cost: 0.,
                    resources: if early && i == 0 { [0.; 2] } else { [1.; 2] },
                    hops: 0,
                    ng: 0,
                });
            }
            for backend in ["scalar", "avx2", "avx512", "dispatch"] {
                if backend == "avx2" && !std::is_x86_feature_detected!("avx2") {
                    continue;
                }
                if backend == "avx512"
                    && (!cfg!(feature = "simd-avx512") || !std::is_x86_feature_detected!("avx512f"))
                {
                    continue;
                }
                let start = Instant::now();
                for _ in 0..200_000 {
                    let c = black_box(&c);
                    // SAFETY: each backend's CPU features were checked above.
                    let result = match backend {
                        "avx2" => unsafe {
                            c.next_candidate_avx2::<true, true, 2, true>(
                                0,
                                f64::INFINITY,
                                0.,
                                0.,
                                0,
                            )
                        },
                        #[cfg(feature = "simd-avx512")]
                        "avx512" => unsafe {
                            c.next_candidate_avx512::<true, true, 2, true>(
                                0,
                                f64::INFINITY,
                                0.,
                                0.,
                                0,
                            )
                        },
                        "dispatch" => {
                            c.next_candidate::<true, true, 2, true>(0, f64::INFINITY, 0., 0., 0)
                        }
                        _ => c.next_candidate_scalar::<true, true, 2, true>(
                            0,
                            f64::INFINITY,
                            0.,
                            0.,
                            0,
                        ),
                    };
                    black_box(result);
                }
                println!(
                    "n={n} early={early} backend={backend} ns={}",
                    start.elapsed().as_nanos() / 200_000
                );
            }
        }
    }
}

#[test]
fn simd_matches_ordered_scalar_comparisons() {
    compare::<true, true, 1, true>();
    compare::<true, true, 1, false>();
    compare::<true, true, 2, true>();
    compare::<true, true, 2, false>();
    compare::<true, false, 1, true>();
    compare::<true, false, 1, false>();
    compare::<true, false, 2, true>();
    compare::<true, false, 2, false>();
    compare::<false, true, 1, true>();
    compare::<false, true, 1, false>();
    compare::<false, true, 2, true>();
    compare::<false, true, 2, false>();
    compare::<false, false, 1, true>();
    compare::<false, false, 1, false>();
    compare::<false, false, 2, true>();
    compare::<false, false, 2, false>();
}
