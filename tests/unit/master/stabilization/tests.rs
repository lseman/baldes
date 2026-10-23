use super::*;

#[test]
fn unavailable_solution_moves_to_raw_center_without_replacing_direction() {
    let mut smoothing = DualPriceSmoothing::new(Parameters {
        beta_factor: 0.5,
        ..Parameters::default()
    });
    smoothing.center = vec![0., 0.];
    smoothing.raw = vec![2., 3.];
    smoothing.pricing = vec![1., 1.];
    smoothing.subgradient = vec![0.6, 0.8];
    smoothing.after_pricing(&Feedback {
        negative_columns: 0,
        subgradient_available: false,
        subgradient: vec![1., 1.],
        lagrangian: 100.,
        phase: 0,
    });
    smoothing.after_master(&[3., 4.], &[RowSense::Equal; 2], 1., 0, false);
    assert_eq!(smoothing.center, [2., 3.]);
    assert_eq!(smoothing.subgradient, [0.6, 0.8]);
    assert!(!smoothing.directional_available);
}

#[test]
fn unavailable_pricing_solution_retains_the_previous_angle() {
    for previous in [0.25, f64::NAN] {
        let mut smoothing = DualPriceSmoothing::new(Parameters::default());
        smoothing.angle = previous;
        smoothing.center = vec![0., 0.];
        smoothing.pricing = vec![1., -1.];
        smoothing.after_pricing(&Feedback {
            negative_columns: 0,
            subgradient_available: false,
            subgradient: vec![1., 1.],
            lagrangian: f64::NEG_INFINITY,
            phase: 0,
        });
        assert!(smoothing.angle == previous || smoothing.angle.is_nan() && previous.is_nan());
        assert_eq!(smoothing.alpha, 0.4);
    }
}

#[test]
fn directional_point_matches_compiled_r101_pricing_point() {
    let mut smoothing = DualPriceSmoothing::new(Parameters {
        beta_factor: 0.5,
        ..Parameters::default()
    });
    smoothing.center = vec![0., 2.76, 2.76, 2.76, 2.76, 2.76];
    smoothing.raw = vec![0., 5.52, 5.52, 5.52, 5.52, 5.52];
    smoothing.subgradient = vec![
        0.,
        -0.8944271909999159,
        0.22360679774997896,
        0.22360679774997896,
        0.22360679774997896,
        0.22360679774997896,
    ];
    smoothing.senses = vec![
        RowSense::LessEqual,
        RowSense::Equal,
        RowSense::Equal,
        RowSense::Equal,
        RowSense::Equal,
        RowSense::Equal,
    ];
    smoothing.publish(0.4, true);
    assert_eq!(
        smoothing.pricing,
        vec![
            0.,
            1.5890311703550772,
            4.5164532444673835,
            4.5164532444673835,
            4.5164532444673835,
            4.5164532444673835
        ]
    );
}

#[test]
// Preserve the recorded binary64 inputs from the reference pricing workflow.
#[allow(clippy::approx_constant)]
fn automatic_direction_is_invalidated_before_pending_center_installation() {
    // Fractional three-customer workflow, first pricing after a capacity
    // row: the reference retains its old vector but clears its validity.
    let mut smoothing = DualPriceSmoothing::new(Parameters {
        beta_factor: 1.,
        ..Parameters::default()
    });
    smoothing.pricing = vec![
        0.,
        1.7071067811865475,
        1.7071067811865475,
        1.7071067811865475,
    ];
    smoothing.raw = smoothing.pricing.clone();
    smoothing.after_pricing(&Feedback {
        subgradient_available: true,
        negative_columns: 0,
        subgradient: vec![0., -2., -2., 1.],
        lagrangian: 5.121320343559642,
        phase: 0,
    });
    smoothing.restart_schedule(2);
    smoothing.after_master(
        &[
            0.,
            1.414213562373095,
            1.414213562373095,
            1.414213562373095,
            -0.5857864376269051,
        ],
        &[
            RowSense::LessEqual,
            RowSense::Equal,
            RowSense::Equal,
            RowSense::Equal,
            RowSense::LessEqual,
        ],
        1.,
        2,
        false,
    );
    assert_eq!(
        smoothing.pricing,
        vec![
            0.,
            1.5606601717798212,
            1.5606601717798212,
            1.5606601717798212,
            -0.29289321881345254
        ]
    );
}

#[test]
fn mispricing_retry_retains_the_first_angle() {
    let mut smoothing = DualPriceSmoothing::default();
    smoothing.center = vec![0.];
    smoothing.pricing = vec![1.];
    smoothing.active = true;
    let mut feedback = Feedback {
        subgradient_available: true,
        negative_columns: 0,
        subgradient: vec![1.],
        lagrangian: f64::NEG_INFINITY,
        phase: 0,
    };
    assert!(smoothing.after_pricing(&feedback));
    assert_eq!(smoothing.angle, 1.);
    assert_eq!(smoothing.misprices, 1);
    feedback.subgradient = vec![-1.];
    assert!(!smoothing.after_pricing(&feedback));
    assert_eq!(smoothing.angle, 1.);
}

#[test]
fn effective_alpha_preserves_compiled_operation_order() {
    let mut smoothing = DualPriceSmoothing::default();
    smoothing.alpha = 0.09000000000000002;
    smoothing.effective_alpha = smoothing.alpha;
    smoothing.center = vec![0.];
    smoothing.pricing = vec![1.];
    smoothing.active = true;
    assert!(smoothing.after_pricing(&Feedback {
        subgradient_available: true,
        negative_columns: 0,
        subgradient: vec![1.],
        lagrangian: f64::NEG_INFINITY,
        phase: 0,
    }));
    // R101/5 pricing call 11, directly observed in the compiled reference.
    assert_eq!(smoothing.effective_alpha, -0.8199999999999998);
}

#[test]
fn angle_threshold_is_independent_of_bound_tolerance() {
    for (angle, expected_alpha) in [(2e-12, 0.4), (-2e-12, 0.55), (5e-13, 0.5), (-5e-13, 0.5)] {
        let mut smoothing = DualPriceSmoothing::default();
        smoothing.center = vec![0., 0.];
        smoothing.pricing = vec![1., 0.];
        assert!(!smoothing.after_pricing(&Feedback {
            subgradient_available: true,
            negative_columns: 1,
            subgradient: vec![angle, 1.],
            lagrangian: f64::NEG_INFINITY,
            phase: 1,
        }));
        assert_eq!(smoothing.alpha, expected_alpha, "angle {angle}");
    }
}

#[test]
fn row_remapping_compacts_pending_center_and_handles_short_maps() {
    let mut smoothing = DualPriceSmoothing::default();
    smoothing.center = vec![1., 2., 3., 4.];
    smoothing.raw = vec![5., 6., 7., 8.];
    smoothing.pricing = vec![9., 10., 11., 12.];
    smoothing.subgradient = vec![13., 14., 15., 16.];
    smoothing.angle_center = vec![17., 18., 19., 20.];
    smoothing.senses = vec![
        RowSense::Equal,
        RowSense::LessEqual,
        RowSense::GreaterEqual,
        RowSense::Equal,
    ];
    smoothing.next_center = Some((vec![21., 22., 23., 24.], vec![25., 26., 27., 28.]));
    smoothing.excluded_rows = vec![0, 1, 2, 3];

    smoothing.remap_rows(&[Some(0), None, Some(1)]);

    assert_eq!(smoothing.center, [1., 3.]);
    assert_eq!(smoothing.raw, [5., 7.]);
    assert_eq!(smoothing.pricing, [9., 11.]);
    assert_eq!(smoothing.subgradient, [13., 15.]);
    assert_eq!(smoothing.angle_center, [17., 19.]);
    assert_eq!(smoothing.senses, [RowSense::Equal, RowSense::GreaterEqual]);
    assert_eq!(
        smoothing.next_center,
        Some((vec![21., 23.], vec![25., 27.]))
    );
    assert_eq!(smoothing.excluded_rows, [0, 1]);

    smoothing.remap_rows(&[]);
    assert!(smoothing.center.is_empty());
    assert!(smoothing.raw.is_empty());
    assert!(smoothing.pricing.is_empty());
    assert_eq!(smoothing.next_center, Some((vec![], vec![])));
}
