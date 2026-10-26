use super::*;
#[test]
fn automatic_center_and_curvature_survive_all_pricing_phase_changes() {
    let parameters = ProximalParameters::automatic_five_piece();
    let mut state = ProximalState::default();
    state
        .begin_iteration(&[120.], &[0], 2, &parameters)
        .unwrap();
    state.observe(&[-0.5], 1., true);
    state.after_iteration(&parameters).unwrap();
    state.artificial_activity(true, &parameters).unwrap();
    assert_eq!(state.curvature.unwrap().curvature, 1200.);
    state
        .begin_iteration(&[-0.5], &[0], 1, &parameters)
        .unwrap();
    assert_eq!(state.center, vec![-0.5]);
    assert_eq!(state.curvature.unwrap().curvature, 1200.);
    // Changing the phase during an inner retry does not replace its mean.
    state.begin_iteration(&[99.], &[0], 0, &parameters).unwrap();
    state.after_iteration(&parameters).unwrap();
    assert_eq!(state.curvature.unwrap().outer_radius, 0.);
    assert_eq!(state.curvature.unwrap().curvature, 1200.);
    assert_eq!(state.center, vec![-0.5]);
    state.restart_schedule();
    state
        .begin_iteration(&[-0.5], &[0], 2, &parameters)
        .unwrap();
    assert_eq!(state.curvature.unwrap().curvature, 1200.);
    assert_eq!(state.center, vec![-0.5]);
}

#[test]
fn automatic_iteration_bootstraps_once_from_original_lp_before_center_move() {
    let parameters = ProximalParameters::automatic_five_piece();
    let mut state = ProximalState::default();
    state
        .begin_iteration(&[120., 999.], &[0], 0, &parameters)
        .unwrap();
    assert_eq!(state.center, vec![0., 0.]);
    assert!(state.curvature.unwrap().bands().unwrap().is_none());
    state.observe(&[120., 999.], 10., true);
    state
        .begin_iteration(&[0., 999.], &[0], 0, &parameters)
        .unwrap();
    assert!(state.after_iteration(&parameters).unwrap());
    let curvature = state.curvature.unwrap();
    assert_eq!(curvature.outer_radius, 120.);
    assert_eq!(curvature.inner_radius, Some(120.));
    assert_eq!(curvature.curvature, 120.);
    assert!(!state.after_iteration(&parameters).unwrap());
    assert!(state.artificial_activity(true, &parameters).unwrap());
    assert_eq!(state.curvature.unwrap().curvature, 1200.);
    // Restore the pre-activity state for the independent inheritance case.
    state.curvature = Some(curvature);
    let snapshot = state.snapshot().unwrap();
    let mut child = ProximalState::default();
    child.inherit_for_solve(&snapshot, 2).unwrap();
    child
        .begin_iteration(&[130., 999.], &[0], 2, &parameters)
        .unwrap();
    assert_eq!(child.center, vec![120., 999.]);
    assert_eq!(child.curvature, Some(curvature));
    assert!(child.after_iteration(&parameters).unwrap());
    assert_eq!(child.curvature.unwrap().outer_radius, 10.);
}

#[test]
fn automatic_feedback_uses_selected_rows_and_the_old_center() {
    use super::super::curvature::{AutomaticCurvatureParameters, CurvatureState};
    let parameters = ProximalParameters::automatic(AutomaticCurvatureParameters {
        initial: Some(CurvatureState {
            outer_radius: 10.,
            inner_radius: None,
            curvature: 2.,
        }),
        five_piece: false,
        kappa: 1.,
        advance_rate: 0.5,
        ..Default::default()
    });
    assert!(parameters.validate().is_ok());
    let mut state = ProximalState::default();
    state.center = vec![2., 1000.];
    state.phase = Some(0);
    state.curvature = parameters.curvature.as_ref().unwrap().initial;
    state.configuration = parameters.curvature.clone();
    assert!(!state.after_iteration(&parameters).unwrap());
    assert_eq!(state.curvature.unwrap().outer_radius, 10.);
    state
        .begin_iteration(&[6., -1000.], &[0], 0, &parameters)
        .unwrap();
    assert!(state.after_iteration(&parameters).unwrap());
    assert_eq!(state.curvature.unwrap().outer_radius, 7.);
    let snapshot = state.snapshot().unwrap();
    let mut child = ProximalState::default();
    child.inherit(&snapshot).unwrap();
    assert_eq!(child.center, state.center);
    assert_eq!(child.curvature, state.curvature);
    assert_eq!(child.bound, f64::NEG_INFINITY);
    child.reset();
    assert!(child.curvature.is_none());
    assert!(child.snapshot().is_none());
}

#[cfg(feature = "clp")]
#[test]
fn automatic_configuration_round_trips_without_explicit_bands() {
    let defaults: ProximalParameters = serde_json::from_str(r#"{"curvature":{}}"#).unwrap();
    assert!(defaults.validate().is_ok());
    assert_eq!(defaults.curvature.as_ref().unwrap().kappa, 1.);
    assert_eq!(defaults.curvature.as_ref().unwrap().advance_rate, 1.);
    assert!(defaults.curvature.as_ref().unwrap().initial.is_none());
    let parameters: ProximalParameters = serde_json::from_str(
        r#"{"curvature":{"initial":{"outer_radius":10.0,"inner_radius":null,"curvature":2.0},"kappa":1.0,"advance_rate":0.2}}"#,
    ).unwrap();
    assert!(parameters.validate().is_ok());
    let restored: ProximalParameters =
        serde_json::from_str(&serde_json::to_string(&parameters).unwrap()).unwrap();
    assert_eq!(
        restored.curvature.unwrap().initial.unwrap().outer_radius,
        10.
    );
    let mut conflicting = parameters;
    conflicting.bands = ProximalParameters::box_step(1.).bands;
    assert!(conflicting.validate().is_err());
}

#[test]
fn bands_reject_invalid_penalties() {
    assert!(ProximalParameters::box_step(1.).validate().is_ok());
    assert!(ProximalParameters::three_piece(0., 2.).validate().is_ok());
    assert!(ProximalParameters::three_piece(1., -1.).validate().is_err());
    assert!(ProximalParameters::box_step(f64::NAN).validate().is_err());
    let mut conflict = ProximalParameters::automatic_three_piece();
    conflict.multi_point = true;
    assert!(conflict.validate().is_err());
    assert!(ProximalParameters::five_piece(
        ProximalBand {
            half_width: 1.,
            penalty: f64::INFINITY
        },
        ProximalBand {
            half_width: 2.,
            penalty: 1.
        }
    )
    .validate()
    .is_err());
}

#[cfg(feature = "clp")]
#[test]
fn json_infinite_penalty_round_trips_as_null_and_unknown_fields_fail() {
    let parameters = ProximalParameters::box_step(2.);
    let encoded = serde_json::to_string(&parameters).unwrap();
    assert_eq!(encoded, r#"{"bands":[{"half_width":2.0,"penalty":null}]}"#);
    let decoded: ProximalParameters = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.bands[0].penalty, f64::INFINITY);
    assert!(serde_json::from_str::<ProximalParameters>(
        r#"{"bands":[{"half_width":1,"penalty":2,"extra":0}]}"#
    )
    .is_err());
    assert!(serde_json::from_str::<ProximalParameters>(r#"{"bands":[],"extra":0}"#).is_err());
}
#[test]
fn center_moves_only_on_completed_improving_pricing() {
    let mut state = ProximalState::default();
    state.reset();
    state.observe(&[1.], 2., true);
    state.observe(&[3.], 4., false);
    state.observe(&[5.], 1., true);
    assert_eq!(state.center, vec![1.]);
    state.observe(&[6.], 5., true);
    assert_eq!(state.center, vec![6.]);
}

#[cfg(feature = "clp")]
#[test]
fn hard_and_soft_bands_solve_only_a_private_master() {
    use crate::master::{clp::ClpMaster, RowSense};
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(10., &[(0, 1.)]).unwrap();
    assert!(master.solve());
    for (parameters, expected) in [
        (ProximalParameters::box_step(1.), 3.),
        (ProximalParameters::three_piece(1., 0.5), 10.),
        (
            ProximalParameters::five_piece(
                ProximalBand {
                    half_width: 1.,
                    penalty: 0.5,
                },
                ProximalBand {
                    half_width: 5.,
                    penalty: 0.75,
                },
            ),
            7.,
        ),
    ] {
        let mut state = ProximalState {
            center: vec![2.],
            bound: f64::NEG_INFINITY,
            phase: Some(0),
            samples: vec![],
            max_points: 0,
            next_point_id: 0,
            curvature: None,
            configuration: None,
            last_mean: None,
            pending_iteration: false,
            artificial_values: None,
            explicit_bands: vec![],
            explicit_baseline: None,
        };
        let point = state
            .pricing_point(&master, &[10.], 0, &parameters, &[], Duration::from_secs(5))
            .unwrap();
        assert!((point[0] - expected).abs() < 1e-7, "{point:?}");
        assert_eq!(master.costs(), &[10.]);
        assert_eq!(master.rhs(), &[1.]);
        assert!((master.objective() - 10.).abs() < 1e-7);
        assert_eq!(master.duals(), &[10.]);
        let excluded = state
            .pricing_point(
                &master,
                &[10.],
                0,
                &parameters,
                &[0],
                Duration::from_secs(5),
            )
            .unwrap();
        assert_eq!(excluded, vec![10.]);
    }
}

#[cfg(feature = "clp")]
#[test]
fn incompatible_hard_band_and_zero_budget_fall_back_to_original_duals() {
    use crate::master::{clp::ClpMaster, RowSense};
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(2., &[(0, 1.)]).unwrap();
    assert!(master.solve());
    let mut state = ProximalState {
        center: vec![10.],
        bound: f64::NEG_INFINITY,
        phase: Some(0),
        samples: vec![],
        max_points: 0,
        next_point_id: 0,
        curvature: None,
        configuration: None,
        last_mean: None,
        pending_iteration: false,
        artificial_values: None,
        explicit_bands: vec![],
        explicit_baseline: None,
    };
    for remaining in [Duration::from_secs(5), Duration::ZERO] {
        assert_eq!(
            state
                .pricing_point(
                    &master,
                    &[2.],
                    0,
                    &ProximalParameters::box_step(1.),
                    &[],
                    remaining
                )
                .unwrap(),
            vec![2.]
        );
    }
    assert_eq!(master.costs(), &[2.]);
}

#[cfg(feature = "clp")]
#[test]
fn multipoint_column_generation_keeps_beta_out_of_original_objectives_and_warm_models() {
    use crate::master::{cg, clp::ClpMaster, RowSense};
    use crate::pricing::phases::Phase;
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(10., &[(0, 1.)]).unwrap();
    let make = |cost: f64| cg::Column {
        cost,
        reduced_cost: 0.,
        entries: vec![(0, 1.)],
        acceptable: true,
        tag: cost as i32,
        identity: format!("route-{cost}"),
    };
    let mut parameters = ProximalParameters::three_piece(1., 0.75);
    parameters.multi_point = true;
    let out = cg::optimize(
        &mut master,
        &[Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        }],
        |_, duals| {
            let mut column = make(2.);
            column.reduced_cost = 2. - duals[0];
            Ok(cg::Answer {
                minimum: column.reduced_cost,
                columns: vec![column],
                exact: true,
                bound_minimum: None,
            })
        },
        &mut cg::NoController,
        cg::Options {
            proximal: Some(parameters),
            known: vec![make(10.)],
            artificial_cost: 100.,
            insert_non_negative: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(out.status, cg::Status::Optimal);
    assert!((out.lp - 2.).abs() < 1e-7);
    assert!((out.bound - 2.).abs() < 1e-7);
    assert_eq!(master.rhs(), &[1.]);
    assert!(master.costs().iter().all(|&c| [2., 10., 100.].contains(&c)));
    let warm = out.warm_start;
    assert_eq!(warm.columns.len(), master.costs().len());
    assert!(warm
        .columns
        .iter()
        .flatten()
        .all(|c| [2., 10.].contains(&c.cost)));
}

#[test]
fn explicit_center_intervals_and_baseline_survive_phase_and_schedule_changes() {
    let parameters = ProximalParameters::three_piece(5., 1.);
    let mut state = ProximalState::default();
    state.begin_iteration(&[10.], &[0], 2, &parameters).unwrap();
    assert_eq!(state.center, vec![0.]);
    state.explicit_bands = parameters.bands.clone();
    state.explicit_baseline = Some(5.);
    state.observe(&[3.], 8., true);
    state.begin_iteration(&[10.], &[0], 1, &parameters).unwrap();
    assert_eq!(state.center, vec![3.]);
    state.observe(&[4.], 2., true);
    assert_eq!(state.center, vec![4.]);
    state.restart_schedule();
    state.begin_iteration(&[99.], &[0], 2, &parameters).unwrap();
    assert_eq!(state.center, vec![4.]);
    assert_eq!(state.explicit_bands[0].half_width, 5.);
    assert_eq!(state.explicit_baseline, Some(5.));
    state.artificial_values = Some(vec![0.]);
    assert!(state.after_iteration(&parameters).unwrap());
    assert_eq!(state.explicit_bands[0].half_width, 2.5);
    assert_eq!(state.explicit_baseline, Some(5.));
}

#[test]
fn native_explicit_defaults_bootstrap_from_latched_mean_and_reduce_artificial_caps() {
    for five in [false, true] {
        let parameters = if five {
            ProximalParameters::native_five_piece()
        } else {
            ProximalParameters::native_three_piece()
        };
        let mut state = ProximalState::default();
        state
            .begin_iteration(&[120.], &[0], 2, &parameters)
            .unwrap();
        state.explicit_bands = parameters.native_explicit.as_ref().unwrap().initial_bands();
        state.explicit_baseline = Some(1e12);
        assert!(state
            .explicit_bands
            .iter()
            .all(|band| band.half_width == 1e12));
        // Serious-step center changes cannot replace the pre-pricing mean.
        state.observe(&[119.], 1., true);
        state.begin_iteration(&[1.], &[0], 2, &parameters).unwrap();
        assert!(state.after_iteration(&parameters).unwrap());
        let expected_outer = if five { 12. } else { 120. };
        assert_eq!(
            state.explicit_bands.last().unwrap().half_width,
            expected_outer
        );
        assert_eq!(state.explicit_baseline, Some(expected_outer));
        if five {
            assert_eq!(state.explicit_bands[0].half_width, 120.);
        }
        assert!(state.artificial_activity(true, &parameters).unwrap());
        for band in &state.explicit_bands {
            assert_eq!(band.penalty, (0.9_f32 as f64) / 10.);
        }
    }
    let parameters = ProximalParameters::native_explicit(NativeExplicitParameters {
        kappa: 0.,
        ..Default::default()
    });
    let bands = parameters.native_explicit.as_ref().unwrap().initial_bands();
    assert_eq!(bands[0].half_width, 10.);
}

#[cfg(feature = "clp")]
#[test]
fn native_rhs_scaled_cap_changes_private_query_without_changing_original_master() {
    use crate::master::{clp::ClpMaster, RowSense};
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 2., &[]).unwrap();
    master.add_column(10., &[(0, 1.)]).unwrap();
    assert!(master.solve());
    let unscaled = ProximalParameters::three_piece(1., 1.5);
    let mut scaled = unscaled.clone();
    scaled.native_row_scaling = true;
    let mut points = vec![];
    for parameters in [&unscaled, &scaled] {
        let mut state = ProximalState::default();
        state.begin_iteration(&[10.], &[0], 0, parameters).unwrap();
        points.push(
            state
                .pricing_point(&master, &[10.], 0, parameters, &[], Duration::from_secs(1))
                .unwrap()[0],
        );
    }
    assert!((points[0] - 10.).abs() < 1e-7);
    assert!((points[1] - 1.).abs() < 1e-7);
    assert_eq!(master.costs(), &[10.]);
    assert_eq!(master.rhs(), &[2.]);
    assert!((master.objective() - 20.).abs() < 1e-7);
}

#[test]
fn multipoint_checkpoint_retains_samples_activity_age_and_creation_order() {
    let mut state = ProximalState::default();
    let parameters = ProximalParameters::native_multipoint();
    state.begin_iteration(&[7.], &[0], 0, &parameters).unwrap();
    state.max_points = 3;
    state.register_point(&[2.], false);
    state.register_point(&[4.], true);
    state.samples[0].dual_activity = 3.;
    state.samples[0].zero_age = 1;
    let snapshot = state.snapshot().unwrap();
    let mut resumed = ProximalState::default();
    resumed.inherit_for_solve(&snapshot, 2).unwrap();
    assert_eq!(resumed.samples.len(), 2);
    assert_eq!(resumed.samples[0].point, vec![2.]);
    assert_eq!(resumed.samples[0].dual_activity, 3.);
    assert_eq!(resumed.samples[0].zero_age, 1);
    assert_eq!(resumed.samples[1].creation_id, 1);
    assert!(resumed.samples[1].protected_center);
    assert_eq!(resumed.next_point_id, 2);
}

#[cfg(feature = "clp")]
#[test]
fn native_multipoint_keeps_unscaled_sample_coefficients_on_rhs_scaled_artificials() {
    use crate::master::{clp::ClpMaster, RowSense};
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 2., &[]).unwrap();
    master.add_column(10., &[(0, 1.)]).unwrap();
    assert!(master.solve());
    for (native_scaling, expected) in [(false, 3.), (true, 1.5)] {
        let mut parameters = ProximalParameters::native_multipoint();
        parameters.native_row_scaling = native_scaling;
        parameters.bands.last_mut().unwrap().penalty = 2.;
        let mut state = ProximalState::default();
        state.begin_iteration(&[10.], &[0], 0, &parameters).unwrap();
        state.observe(&[3.], 1., true);
        let point = state
            .pricing_point(&master, &[10.], 0, &parameters, &[], Duration::from_secs(1))
            .unwrap();
        assert!((point[0] - expected).abs() < 1e-7, "{point:?}");
        assert!((master.objective() - 20.).abs() < 1e-7);
    }
}

#[test]
fn native_box_step_normalizes_angles_and_uses_function_one_bootstrap() {
    let parameters = ProximalParameters::native_box_step();
    let native = parameters.native_explicit.as_ref().unwrap();
    assert!(native.box_step);
    let mut state = ProximalState::default();
    state
        .begin_iteration(&[120.], &[0], 0, &parameters)
        .unwrap();
    state.explicit_bands = native.initial_bands();
    assert_eq!(state.explicit_bands.len(), 1);
    assert_eq!(state.explicit_bands[0].half_width, 1e12);
    assert_eq!(state.explicit_bands[0].penalty, 1.01_f32 as f64);
    state.after_iteration(&parameters).unwrap();
    assert_eq!(state.explicit_bands[0].half_width, 12.);
    assert_eq!(state.explicit_baseline, Some(12.));
}

#[test]
fn native_nonpositive_kappa_respects_public_interval_names_and_normalization() {
    for kappa in [0., -1.] {
        let parameters = ProximalParameters::native_explicit(NativeExplicitParameters {
            five_piece: true,
            kappa,
            outer_half_width: 100.,
            inner_half_width: 10.,
            outer_penalty: 0.75,
            inner_penalty: 0.5,
            ..Default::default()
        });
        let native = parameters.native_explicit.as_ref().unwrap();
        let mut state = ProximalState::default();
        state
            .begin_iteration(&[120.], &[0], 0, &parameters)
            .unwrap();
        state.explicit_bands = native.initial_bands();
        state.explicit_baseline = Some(10.);
        assert_eq!(state.explicit_bands[0].half_width, 100.);
        assert_eq!(state.explicit_bands[0].penalty, 0.75);
        assert_eq!(state.explicit_bands[1].half_width, 10.);
        assert_eq!(state.explicit_bands[1].penalty, 0.5);
        // Configured finite widths do not bootstrap, even with a positive mean.
        assert!(!state.after_iteration(&parameters).unwrap());
        assert_eq!(state.explicit_bands[1].half_width, 10.);
    }
    // Native postTreatment clamps public InnerHalfInterval to OuterHalfInterval.
    let native = NativeExplicitParameters {
        kappa: 0.,
        outer_half_width: 100.,
        inner_half_width: 1e12,
        ..Default::default()
    };
    assert_eq!(native.initial_bands()[0].half_width, 100.);
    assert!(
        ProximalParameters::native_explicit(NativeExplicitParameters {
            box_step: true,
            five_piece: true,
            ..Default::default()
        })
        .validate()
        .is_err()
    );
}

#[test]
fn native_multipoint_defaults_retain_both_artificial_pairs_and_only_relax_last_cap() {
    let parameters = ProximalParameters::native_multipoint();
    assert_eq!(parameters.bands.len(), 2);
    assert!(parameters
        .bands
        .iter()
        .all(|band| band.half_width == 0. && band.penalty == 0.9_f32 as f64));
    let mut state = ProximalState::default();
    state.explicit_bands = parameters.bands.clone();
    assert!(state.artificial_activity(true, &parameters).unwrap());
    assert_eq!(state.explicit_bands[0].penalty, 0.9_f32 as f64);
    assert_eq!(state.explicit_bands[1].penalty, (0.9_f32 as f64) / 10.);
}
