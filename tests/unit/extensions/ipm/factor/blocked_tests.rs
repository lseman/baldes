use super::*;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-11 * (1. + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn blocked_full_partial_fronts_and_mid_panel_fallback_match_scalar_reference() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for n in [128, 193, 512] {
        for np in [35, 67, n] {
            for fail_at in [17, 47, n] {
                let mut values = vec![0.; n * n];
                for i in 0..n {
                    for j in 0..=i {
                        let value = if i == j {
                            4.
                        } else {
                            ((i + j) % 7) as f64 * 0.002
                        };
                        values[i * n + j] = value;
                        values[j * n + i] = value;
                    }
                }
                if fail_at + 1 < np {
                    // Also force an interchange beyond the current panel.
                    let partner = if fail_at == 17 && np > 100 {
                        100
                    } else {
                        fail_at + 1
                    };
                    values[fail_at * n + fail_at] = 0.;
                    values[partner * n + fail_at] = 2.;
                    values[fail_at * n + partner] = 2.;
                    values[partner * n + partner] = 0.;
                }
                let mut expected = values.clone();
                let mut expected_stats = FactorStats {
                    min_abs_pivot: f64::INFINITY,
                    ..Default::default()
                };
                let (expected_blocks, expected_perm) =
                    bk(&mut expected, n, np, true, &mut expected_stats).unwrap();
                for node in [false, true] {
                    let mut actual = values.clone();
                    let mut blocks = Vec::new();
                    let mut perm = Vec::new();
                    let mut scratch = DenseScratch::default();
                    let mut stats = FactorStats {
                        min_abs_pivot: f64::INFINITY,
                        ..Default::default()
                    };
                    for i in 0..n {
                        for j in i + 1..n {
                            actual[i * n + j] = f64::NAN;
                        }
                    }
                    pool.install(|| {
                        bk_into(
                            &mut actual,
                            n,
                            np,
                            true,
                            &mut stats,
                            node,
                            &mut blocks,
                            &mut perm,
                            &mut scratch,
                            true,
                        )
                    })
                    .unwrap();
                    assert_eq!(perm, expected_perm);
                    assert_eq!(stats.num_2x2, expected_stats.num_2x2);
                    assert_eq!(stats.perturbed_pivots, expected_stats.perturbed_pivots);
                    for (actual, expected) in blocks.iter().zip(&expected_blocks) {
                        match (actual, expected) {
                            (Block::One(a), Block::One(b)) => close(*a, *b),
                            (Block::Two(a, b, c), Block::Two(x, y, z)) => {
                                close(*a, *x);
                                close(*b, *y);
                                close(*c, *z);
                            }
                            (Block::Skip, Block::Skip) => (),
                            _ => panic!("changed pivot block"),
                        }
                    }
                    for i in 0..n {
                        for j in 0..=i {
                            close(actual[i * n + j], expected[i * n + j]);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn blocked_kernel_keeps_numerical_failures_and_reuses_scratch() {
    let n = 128;
    let mut scratch = DenseScratch::default();
    let mut blocks = Vec::new();
    let mut perm = Vec::new();
    for failed in [false, true, false] {
        let mut values = vec![0.; n * n];
        if !failed {
            for i in 0..n {
                values[i * n + i] = 2.;
            }
        }
        let mut stats = FactorStats {
            min_abs_pivot: f64::INFINITY,
            ..Default::default()
        };
        let result = bk_into(
            &mut values,
            n,
            n,
            false,
            &mut stats,
            false,
            &mut blocks,
            &mut perm,
            &mut scratch,
            true,
        );
        assert_eq!(result.is_err(), failed);
        if !failed {
            assert_eq!(blocks.len(), n);
        }
    }
}
