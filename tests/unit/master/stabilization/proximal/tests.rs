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
