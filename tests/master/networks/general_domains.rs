use super::*;
use baldes::master::preprocessing::Bounds;
use multi_pricing::{EnumerationOptions, MasterVariable};

fn variable(cost: f64, bounds: Bounds, entries: Vec<(usize, f64)>) -> MasterVariable {
    MasterVariable {
        id: 100,
        cost,
        bounds,
        integer: false,
        entries,
    }
}

#[test]
fn one_sided_and_free_master_domains_do_not_create_artificial_caps() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for (cost, bounds, expected) in [
            (
                0.,
                Bounds {
                    lower: f64::NEG_INFINITY,
                    upper: f64::INFINITY,
                },
                1.5,
            ),
            (
                1.,
                Bounds {
                    lower: 0.,
                    upper: f64::INFINITY,
                },
                1.5,
            ),
            (
                -1.,
                Bounds {
                    lower: f64::NEG_INFINITY,
                    upper: 0.,
                },
                1.5,
            ),
            (
                1.,
                Bounds {
                    lower: -2.,
                    upper: f64::INFINITY,
                },
                -0.5,
            ),
            (
                -1.,
                Bounds {
                    lower: f64::NEG_INFINITY,
                    upper: 2.,
                },
                -0.5,
            ),
        ] {
            let mut configured = options();
            configured.cg.bound_mode = mode;
            let output = multi_pricing::solve_with_master_variables(
                triangle(),
                vec![variable(cost, bounds, vec![])],
                configured,
            )
            .unwrap();
            optimal(&output, expected);
            assert_eq!(
                output.snapshot.preprocessing_report().master_bounds[0].1,
                bounds
            );
        }
    }
}

#[test]
fn input_unbounded_variables_can_be_bounded_by_shared_rows_or_later_fixing() {
    for preprocess in [false, true] {
        for enumerate in [false, true] {
            let mut configured = options();
            configured.cg.bound_mode = BoundMode::Certified;
            configured.preprocessing.enabled = preprocess;
            if enumerate {
                configured.enumeration = Some(EnumerationOptions::default());
            }
            let root = multi_pricing::solve_with_master_variables(
                triangle(),
                vec![variable(
                    0.25,
                    Bounds {
                        lower: 0.,
                        upper: f64::INFINITY,
                    },
                    vec![(0, 1.)],
                )],
                configured.clone(),
            )
            .unwrap();
            optimal(&root, 1.25);
            assert_eq!(root.master_values, vec![(100, 1.)]);
            let child = root
                .snapshot
                .with_master_variable_bounds(
                    100,
                    Bounds {
                        lower: 0.,
                        upper: 0.,
                    },
                )
                .unwrap();
            optimal(&child.solve(configured.clone()).unwrap(), 1.5);
            let unchanged = child
                .with_master_variable_bounds(
                    100,
                    Bounds {
                        lower: f64::NEG_INFINITY,
                        upper: f64::INFINITY,
                    },
                )
                .unwrap();
            optimal(&unchanged.solve(configured).unwrap(), 1.5);
        }
    }
}

#[test]
fn truly_unbounded_master_directions_do_not_stop_at_the_default_column_upper() {
    for (cost, bounds) in [
        (
            -1.,
            Bounds {
                lower: 0.,
                upper: f64::INFINITY,
            },
        ),
        (
            1.,
            Bounds {
                lower: f64::NEG_INFINITY,
                upper: 0.,
            },
        ),
    ] {
        let mut configured = options();
        configured.cg.bound_mode = BoundMode::Certified;
        let output = multi_pricing::solve_with_master_variables(
            triangle(),
            vec![variable(cost, bounds, vec![])],
            configured,
        )
        .unwrap();
        assert_eq!(
            output.master.status,
            Status::Unbounded,
            "{:?}",
            output.master
        );
        assert_eq!(output.master.bound, f64::NEG_INFINITY);
        #[cfg(feature = "highs")]
        {
            let complete = output
                .snapshot
                .enumerate(&EnumerationOptions::default())
                .unwrap();
            let mip = complete
                .solve_enumerated_mip(&Default::default())
                .unwrap()
                .unwrap();
            assert_ne!(
                mip.mip.status,
                baldes::master::solvers::mip::MipStatus::Optimal
            );
            assert_eq!(mip.mip.bound, f64::NEG_INFINITY);
        }
    }
}

#[test]
fn infinite_wrong_signs_and_nan_are_rejected_without_forbidding_valid_free_domains() {
    for bounds in [
        Bounds {
            lower: f64::INFINITY,
            upper: f64::INFINITY,
        },
        Bounds {
            lower: f64::NEG_INFINITY,
            upper: f64::NEG_INFINITY,
        },
        Bounds {
            lower: f64::NAN,
            upper: 1.,
        },
    ] {
        assert!(multi_pricing::solve_with_master_variables(
            triangle(),
            vec![variable(0., bounds, vec![])],
            options()
        )
        .is_err());
    }
    let root = multi_pricing::solve_with_master_variables(
        triangle(),
        vec![variable(
            0.,
            Bounds {
                lower: f64::NEG_INFINITY,
                upper: f64::INFINITY,
            },
            vec![],
        )],
        options(),
    )
    .unwrap();
    let fixed = root
        .snapshot
        .with_master_variable_bounds(
            100,
            Bounds {
                lower: 1.,
                upper: 1.,
            },
        )
        .unwrap()
        .solve(options())
        .unwrap();
    optimal(&fixed, 1.5);
    assert_eq!(fixed.master_values, vec![(100, 1.)]);
}

#[test]
fn feasible_values_above_the_backend_default_cap_remain_available() {
    let large = 2_000_000_000_000.;
    let mut configured = options();
    configured.cg.bound_mode = BoundMode::Certified;
    let root = multi_pricing::solve_with_master_variables(
        triangle(),
        vec![variable(
            0.,
            Bounds {
                lower: large,
                upper: f64::INFINITY,
            },
            vec![],
        )],
        configured.clone(),
    )
    .unwrap();
    optimal(&root, 1.5);
    assert_eq!(root.master_values, vec![(100, large)]);
    let child = root
        .snapshot
        .with_master_variable_bounds(
            100,
            Bounds {
                lower: large + 1.,
                upper: large + 1.,
            },
        )
        .unwrap()
        .solve(configured)
        .unwrap();
    optimal(&child, 1.5);
    assert_eq!(child.master_values, vec![(100, large + 1.)]);
    #[cfg(feature = "highs")]
    {
        let complete = root
            .snapshot
            .enumerate(&EnumerationOptions::default())
            .unwrap();
        let mip = complete
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            mip.mip.status,
            baldes::master::solvers::mip::MipStatus::Optimal
        );
        assert_eq!(mip.mip.solution.unwrap().objective, 2.);
        assert_eq!(mip.master_values, vec![(100, large)]);
    }
}

#[test]
fn proof_only_projection_handles_non_dyadic_duals_without_a_finite_sentinel() {
    for enumerate in [false, true] {
        let mut configured = options();
        configured.preprocessing.enabled = false;
        configured.cg.bound_mode = BoundMode::Certified;
        if enumerate {
            configured.enumeration = Some(EnumerationOptions::default());
        }
        let root = multi_pricing::solve_with_master_variables(
            triangle(),
            vec![variable(
                0.3,
                Bounds {
                    lower: 0.,
                    upper: f64::INFINITY,
                },
                vec![(0, 3.)],
            )],
            configured,
        )
        .unwrap();
        optimal(&root, 1.1);
        assert!((root.master_values[0].1 - 1. / 3.).abs() < 1e-10);
        assert_eq!(
            root.snapshot.preprocessing_report().skipped.as_deref(),
            Some("Disabled")
        );
    }
}
