use super::*;

#[test]
fn canonical_sparse_entries_match_tree_accumulation() {
    fn reference(entries: &[(usize, f64)]) -> (Vec<i32>, Vec<f64>) {
        let mut sums = std::collections::BTreeMap::new();
        for &(id, value) in entries {
            *sums.entry(id).or_insert(0.) += value;
        }
        sums.into_iter()
            .filter(|&(_, value)| value != 0.)
            .map(|(id, value)| (id as i32, value))
            .unzip()
    }
    for code in 0..6561usize {
        let mut digits = code;
        let mut entries = vec![];
        for _ in 0..4 {
            entries.push((digits % 3, [-1e16, 1., 1e16][digits / 3 % 3]));
            digits /= 9;
        }
        assert_eq!(entries_of(&entries, 3).unwrap(), reference(&entries));
    }
    for mask in 0..16 {
        let entries: Vec<_> = (0..4)
            .filter(|id| mask & (1 << id) != 0)
            .map(|id| (id, [-0., 1e-30, -2., 1e100][id]))
            .collect();
        assert_eq!(entries_of(&entries, 4).unwrap(), reference(&entries));
    }
    for entries in [
        vec![(4, 1.)],
        vec![(usize::MAX, 1.)],
        vec![(0, f64::NAN)],
        vec![(0, f64::INFINITY)],
        vec![(1, 1.), (0, f64::NEG_INFINITY)],
    ] {
        assert!(entries_of(&entries, 4).is_err());
    }
}

#[test]
fn clone_preserves_sparse_layout_through_column_insertion() {
    fn layout(master: &ClpMaster) -> (Vec<i32>, Vec<i32>) {
        // SAFETY: live Clp models expose n+1 starts and n column lengths;
        // both buffers are copied before the model can be mutated.
        unsafe {
            let model = master.model.as_ptr();
            (
                std::slice::from_raw_parts(Clp_getVectorStarts(model), master.costs.len() + 1)
                    .to_vec(),
                std::slice::from_raw_parts(Clp_getVectorLengths(model), master.costs.len())
                    .to_vec(),
            )
        }
    }
    let mut master = ClpMaster::new().unwrap();
    for _ in 0..3 {
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
    }
    for cost in [1., 2., 3.] {
        master
            .add_column(cost, &[(0, 1.), (1, 1.), (2, 1.)])
            .unwrap();
    }
    master.remove(&[], &[1]).unwrap();
    let before = layout(&master);
    assert!(before.0.last().unwrap() > &before.1.iter().sum());
    let mut copied = master.try_clone().unwrap();
    assert_eq!(layout(&copied), before);
    for model in [&mut master, &mut copied] {
        model.add_column(4., &[(0, 1.)]).unwrap();
    }
    assert_eq!(layout(&copied), layout(&master));
    assert!(master.solve() && copied.solve());
    assert_eq!(master.objective, 1.);
    assert_eq!(copied.values(), master.values());
    copied.set_cost(0, 5.).unwrap();
    assert!(copied.solve());
    assert_eq!(copied.objective, 3.);
    assert_eq!(master.costs[0], 1.);
    assert_eq!(master.objective, 1.);
}

fn base() -> ClpMaster {
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(2., &[(0, 1.)]).unwrap();
    assert!(master.solve());
    master
}

fn limits(master: &ClpMaster) -> [f64; 3] {
    let mut values = [0.; 3];
    // SAFETY: The native getter writes exactly three values for this model.
    unsafe {
        baldes_clp_solve_limits(master.model.as_ptr(), values.as_mut_ptr());
    }
    values
}

#[test]
fn limited_solve_preserves_absolute_deadlines_and_can_resume() {
    let mut master = base();
    // SAFETY: This exclusively owned model accepts a relative CPU limit.
    unsafe {
        Clp_setMaximumSeconds(master.model.as_ptr(), 60.);
    }
    let before = limits(&master);
    for (iterations, time) in [
        (0, Duration::from_secs(60)),
        (1000, Duration::ZERO),
        (1000, Duration::from_nanos(1)),
    ] {
        assert_eq!(
            master.solve_limited(iterations, time).unwrap(),
            ClpSolveOutcome::Stopped
        );
        assert!(!master.solved);
        assert_eq!(master.status, 3);
        assert_eq!(limits(&master), before);
        assert!(master.solve());
        assert_eq!(master.objective, 2.);
    }
    let status = master.status;
    assert!(master
        .solve_limited(usize::MAX, Duration::from_secs(60))
        .is_err());
    assert_eq!(master.status, status);
    assert!(master.solved);
    assert_eq!(limits(&master), before);
}

#[test]
fn iteration_stopped_model_is_not_an_infeasibility_certificate() {
    let mut master = ClpMaster::new().unwrap();
    for _ in 0..128 {
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
    }
    for row in 0..128 {
        master.add_column(1., &[(row, 1.)]).unwrap();
    }
    let before = limits(&master);
    assert_eq!(
        master.solve_limited(1, Duration::from_secs(60)).unwrap(),
        ClpSolveOutcome::Stopped
    );
    assert_eq!(master.status, 3);
    assert!(!master.solved);
    assert_eq!(limits(&master), before);
    assert!(master.solve());
    assert_eq!(master.objective, 128.);
    assert!(master.values.iter().all(|&value| value == 1.));
}

#[test]
fn cg_time_stop_preserves_bound_and_does_not_price() {
    use crate::master::column_generation::{self as cg, NoController};
    let mut master = ClpMaster::new().unwrap();
    for _ in 0..128 {
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
    }
    for row in 0..128 {
        master.add_column(1., &[(row, 1.)]).unwrap();
    }
    // SAFETY: expire the CPU budget of this exclusively owned model.
    unsafe {
        Clp_setMaximumSeconds(master.model.as_ptr(), 0.);
    }
    let before = limits(&master);
    let mut options = cg::Options::default();
    options.stabilize = false;
    let output = cg::optimize(
        &mut master,
        &[],
        |_, _| -> Result<cg::Answer> {
            panic!("stopped LP must not price");
        },
        &mut NoController,
        options,
    )
    .unwrap();
    assert_eq!(output.status, cg::Status::TimeLimit);
    assert_eq!(output.bound, f64::NEG_INFINITY);
    assert_eq!(output.pricing_calls, 0);
    assert!(!master.solved);
    assert_eq!(limits(&master), before);
    // SAFETY: remove the explicit test budget before resuming the model.
    unsafe {
        Clp_setMaximumSeconds(master.model.as_ptr(), -1.);
    }
    assert!(master.solve());
    assert_eq!(master.objective, 128.);
}

#[test]
fn limited_solve_distinguishes_backend_infeasibility_statuses() {
    let mut infeasible = ClpMaster::new().unwrap();
    infeasible.add_row(RowSense::Equal, 1., &[]).unwrap();
    assert_eq!(
        infeasible
            .solve_limited(1000, Duration::from_secs(60))
            .unwrap(),
        ClpSolveOutcome::PrimalInfeasible
    );
    let mut dual_infeasible = ClpMaster::new().unwrap();
    let column = dual_infeasible.add_column(-1., &[]).unwrap();
    // Columns default to the finite upper bound 1e12. Explicitly use
    // Clp's infinity convention for this backend-status fixture.
    dual_infeasible.set_upper(column, f64::MAX).unwrap();
    assert_eq!(
        dual_infeasible
            .solve_limited(1000, Duration::from_secs(60))
            .unwrap(),
        ClpSolveOutcome::DualInfeasible
    );
    let mut optimal = base();
    assert_eq!(
        optimal
            .solve_limited(1000, Duration::from_secs(60))
            .unwrap(),
        ClpSolveOutcome::Optimal
    );
    assert!(optimal.solved);
}

#[test]
fn continuation_restores_target_bounds_and_keeps_the_source_owned() {
    let mut previous = base();
    previous
        .add_row(RowSense::GreaterEqual, 0., &[(0, 1.)])
        .unwrap();
    previous.add_column(10., &[(1, 1.)]).unwrap();
    assert!(previous.solve());
    let previous_values = previous.values.clone();
    let mut target = base();
    target.set_rhs(&[2.]).unwrap();
    target.set_upper(0, 3.).unwrap();
    assert!(target.continue_from(&previous).unwrap());
    assert_eq!(target.rhs, [2.]);
    assert_eq!(target.upper, [3.]);
    assert_eq!(target.values.len(), 1);
    assert_eq!(target.duals.len(), 1);
    assert!(!target.solved);
    assert!(target.solve());
    assert_eq!(target.objective, 4.);
    assert_eq!(previous.rhs, [1., 0.]);
    assert_eq!(previous.costs, [2., 10.]);
    assert_eq!(previous.values, previous_values);
    assert!(previous.solved);
}

#[test]
fn continuation_rejects_a_different_model_without_mutation() {
    let previous = base();
    for change in 0..4 {
        let mut target = base();
        match change {
            0 => target.set_cost(0, 3.).unwrap(),
            1 => target.set_coefficient(0, 0, 2.).unwrap(),
            2 => {
                target.add_column(3., &[(0, 1.)]).unwrap();
            }
            _ => {
                target.add_row(RowSense::LessEqual, 5., &[(0, 1.)]).unwrap();
            }
        }
        let before = (
            target.rhs.clone(),
            target.costs.clone(),
            target.column_entries(0).unwrap(),
            target.solved,
            target.values.clone(),
        );
        assert!(!target.continue_from(&previous).unwrap());
        assert_eq!(
            before,
            (
                target.rhs.clone(),
                target.costs.clone(),
                target.column_entries(0).unwrap(),
                target.solved,
                target.values.clone()
            )
        );
    }
    assert_eq!(previous.objective, 2.);
    assert!(previous.solved);
}
