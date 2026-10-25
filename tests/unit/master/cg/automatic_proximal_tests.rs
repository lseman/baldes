use super::*;
use crate::master::clp::ClpMaster;

fn checkpoint() -> (ClpMaster, ProximalState, ProximalParameters) {
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(10., &[(0, 1.)]).unwrap();
    let parameters = ProximalParameters::automatic_three_piece();
    let mut state = ProximalState::default();
    state.begin_iteration(&[10.], &[0], 2, &parameters).unwrap();
    state.observe(&[3.], 1., true);
    state.after_iteration(&parameters).unwrap();
    (master, state, parameters)
}

#[test]
fn automatic_checkpoint_validates_original_geometry_without_private_variables() {
    let (mut master, state, parameters) = checkpoint();
    let checkpoint =
        ProximalWarmStart::capture(&state, &master, &NoController, &parameters, &[], &[])
            .unwrap()
            .unwrap();
    assert_eq!(checkpoint.snapshot.curvature.unwrap().curvature, 10.);
    master.add_column(2., &[(0, 1.)]).unwrap();
    let restored = checkpoint
        .restore(&master, &NoController, &parameters)
        .unwrap()
        .unwrap();
    assert_eq!(restored.center, vec![3.]);
    assert_eq!(restored.bound, f64::NEG_INFINITY);
    // Appending ordinary CG routes preserves the checkpoint; rewriting an
    // existing coefficient or original cost does not.
    master.set_coefficient(0, 0, 2.).unwrap();
    assert!(checkpoint
        .restore(&master, &NoController, &parameters)
        .unwrap()
        .is_none());
    master.set_coefficient(0, 0, 1.).unwrap();
    master.set_cost(0, 11.).unwrap();
    assert!(checkpoint
        .restore(&master, &NoController, &parameters)
        .unwrap()
        .is_none());
}

#[test]
fn child_and_resume_keep_global_curvature_without_repeated_scaling() {
    let (mut master, state, parameters) = checkpoint();
    let checkpoint =
        ProximalWarmStart::capture(&state, &master, &NoController, &parameters, &[], &[]).unwrap();
    let mut warm = WarmStart::empty(BoundMode::Numerical);
    warm.proximal = checkpoint;
    warm.mark_proximal_child();
    master.add_row(RowSense::LessEqual, 1., &[(0, 1.)]).unwrap();
    let restored = warm
        .proximal
        .as_ref()
        .unwrap()
        .restore(&master, &NoController, &parameters)
        .unwrap()
        .unwrap();
    assert_eq!(restored.center, vec![3., 0.]);
    assert_eq!(restored.curvature.unwrap().curvature, 10.);
    let mut child = ProximalState::default();
    child.inherit_for_solve(&restored, 2).unwrap();
    let resumed = ProximalWarmStart::capture(&child, &master, &NoController, &parameters, &[], &[])
        .unwrap()
        .unwrap();
    assert!(!resumed.child);
    assert_eq!(
        resumed
            .restore(&master, &NoController, &parameters)
            .unwrap()
            .unwrap()
            .curvature,
        restored.curvature
    );
}

#[test]
fn automatic_controller_measures_rows_updates_caps_and_keeps_warm_models_original() {
    struct Rows;
    impl Controller<ClpMaster> for Rows {
        fn curvature_rows(&self, _: usize) -> Vec<CurvatureRow> {
            vec![
                CurvatureRow::default(),
                CurvatureRow {
                    class: CurvatureRowClass::Dynamic,
                    active: true,
                },
            ]
        }
    }
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_row(RowSense::LessEqual, 100., &[]).unwrap();
    master.add_column(10., &[(0, 1.)]).unwrap();
    let column = |cost: f64| Column {
        cost,
        reduced_cost: 0.,
        entries: vec![(0, 1.)],
        acceptable: true,
        tag: cost as i32,
        identity: format!("{cost}"),
    };
    let mut options = Options {
        proximal: Some(ProximalParameters::automatic_three_piece()),
        known: vec![column(10.)],
        artificial_cost: 0.,
        // A valid loose route-count cap keeps the first Lagrangian estimate
        // below the LP optimum, exercising the native no-column activity retry
        // rather than its legitimate early-bound termination shortcut.
        multiplicity: 3.,
        insert_non_negative: false,
        max_iterations: 100,
        ..Default::default()
    };
    let run = |master: &mut ClpMaster, options| {
        super::super::optimize(
            master,
            &[Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            }],
            |_, duals| {
                let mut route = column(2.);
                route.reduced_cost = 2. - duals[0];
                Ok(Answer {
                    minimum: route.reduced_cost,
                    columns: vec![route],
                    exact: true,
                    bound_minimum: None,
                })
            },
            &mut Rows,
            options,
        )
        .unwrap()
    };
    let output = run(&mut master, options.clone());
    assert_eq!(output.status, Status::Optimal);
    assert_eq!(output.lp, 2.);
    assert_eq!(output.bound, 2.);
    assert!(output.penalty_updates > 0);
    let automatic = output.warm_start.proximal.as_ref().unwrap();
    assert!(automatic.snapshot.curvature.unwrap().curvature >= 10.);
    assert_eq!(automatic.snapshot.center.len(), 2);
    assert!(master.costs().iter().all(|c| *c == 2. || *c == 10.));
    options.known.clear();
    options.warm_start = Some(output.warm_start);
    let resumed = run(&mut master, options);
    assert_eq!(resumed.lp, 2.);
    assert!(resumed.warm_start.proximal.is_some());
}

#[test]
fn native_stabilization_depth_limit_is_strict_and_signed() {
    let mut options = Options::default();
    options.stabilization_max_tree_depth = 2;
    for (depth, active) in [(0, true), (1, true), (2, false), (3, false)] {
        options.node_depth = depth;
        assert_eq!(options.stabilization_active_at_depth(), active);
    }
    options.node_depth = 0;
    for limit in [0, -1, i64::MIN] {
        options.stabilization_max_tree_depth = limit;
        assert!(!options.stabilization_active_at_depth());
    }
    options.stabilization_max_tree_depth = i64::MAX;
    options.node_depth = usize::MAX;
    assert!(!options.stabilization_active_at_depth());
    options.node_depth = 0;
    options.stabilize = false;
    assert!(!options.stabilization_active_at_depth());
}

#[test]
fn explicit_children_use_native_baseline_and_same_node_resume_does_not_rescale() {
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(10., &[(0, 1.)]).unwrap();
    assert!(master.solve());
    let parameters = ProximalParameters::five_piece(
        crate::master::stabilization::ProximalBand {
            half_width: 1.,
            penalty: 0.5,
        },
        crate::master::stabilization::ProximalBand {
            half_width: 5.,
            penalty: 0.5,
        },
    );
    let mut state = ProximalState::default();
    state.begin_iteration(&[10.], &[0], 2, &parameters).unwrap();
    state
        .pricing_point(&master, &[10.], 2, &parameters, &[], Duration::from_secs(1))
        .unwrap();
    state.observe(&[3.], 1., true);
    let mut checkpoint =
        ProximalWarmStart::capture(&state, &master, &NoController, &parameters, &[], &[])
            .unwrap()
            .unwrap();
    let resumed = checkpoint
        .restore(&master, &NoController, &parameters)
        .unwrap()
        .unwrap();
    assert_eq!(resumed.explicit_bands[0].half_width, 1.);
    assert_eq!(resumed.explicit_bands[1].half_width, 5.);
    checkpoint.child = true;
    let child = checkpoint
        .restore(&master, &NoController, &parameters)
        .unwrap()
        .unwrap();
    assert_eq!(child.center, vec![3.]);
    assert_eq!(child.explicit_bands[0].half_width, 500.);
    assert_eq!(child.explicit_bands[1].half_width, 50.);
    assert_eq!(child.explicit_baseline, Some(5.));
    let mut child_state = ProximalState::default();
    child_state.inherit_for_solve(&child, 2).unwrap();
    let resume_checkpoint =
        ProximalWarmStart::capture(&child_state, &master, &NoController, &parameters, &[], &[])
            .unwrap()
            .unwrap();
    let resumed_child = resume_checkpoint
        .restore(&master, &NoController, &parameters)
        .unwrap()
        .unwrap();
    assert_eq!(resumed_child.explicit_bands[0].half_width, 500.);
    assert_eq!(resumed_child.explicit_bands[1].half_width, 50.);
}
