use super::*;
use crate::master::column_generation::{self as cg, NoController};

fn time_limit(master: &HighsMaster) -> f64 {
    let mut value = 0.;
    // SAFETY: live model and valid output scalar / NUL-terminated option.
    unsafe {
        HighsMaster::check(
            Highs_getDoubleOptionValue(
                master.highs.as_ptr(),
                b"time_limit\0".as_ptr().cast(),
                &mut value,
            ),
            "get time_limit",
        )
        .unwrap();
    }
    value
}
fn model() -> HighsMaster {
    let mut master = HighsMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(2., &[(0, 1.)]).unwrap();
    master
}
#[test]
fn scoped_time_budget_restores_options_and_resumes() {
    let mut master = model();
    master.set_double("time_limit", 3600.).unwrap();
    for duration in [
        Duration::from_secs(60),
        Duration::ZERO,
        Duration::from_nanos(1),
    ] {
        let outcome = master.solve_with_time_limit(duration).unwrap();
        assert_eq!(
            outcome,
            if duration == Duration::from_secs(60) {
                LpSolveOutcome::Optimal
            } else {
                LpSolveOutcome::TimeLimit
            }
        );
        assert_eq!(time_limit(&master), 3600.);
        if outcome != LpSolveOutcome::Optimal {
            assert!(!master.solved);
        }
        assert!(master.solve());
        assert_eq!(master.objective, 2.);
    }
}
#[test]
fn cg_backend_limits_do_not_report_infeasibility_or_call_pricing() {
    for iteration_stop in [false, true] {
        let mut master = HighsMaster::new().unwrap();
        for _ in 0..128 {
            master.add_row(RowSense::Equal, 1., &[]).unwrap();
        }
        for row in 0..128 {
            master.add_column(1., &[(row, 1.)]).unwrap();
        }
        if iteration_stop {
            // SAFETY: live handle and NUL-terminated option strings.
            unsafe {
                HighsMaster::check(
                    Highs_setStringOptionValue(
                        master.highs.as_ptr(),
                        b"presolve\0".as_ptr().cast(),
                        b"off\0".as_ptr().cast(),
                    ),
                    "presolve",
                )
                .unwrap();
                HighsMaster::check(
                    Highs_setStringOptionValue(
                        master.highs.as_ptr(),
                        b"solver\0".as_ptr().cast(),
                        b"simplex\0".as_ptr().cast(),
                    ),
                    "solver",
                )
                .unwrap();
                HighsMaster::check(
                    Highs_setIntOptionValue(
                        master.highs.as_ptr(),
                        b"simplex_iteration_limit\0".as_ptr().cast(),
                        0,
                    ),
                    "simplex_iteration_limit",
                )
                .unwrap();
            }
        } else {
            master.set_double("time_limit", 0.).unwrap();
        }
        let mut options = cg::Options::default();
        options.stabilize = false;
        let output = cg::optimize(
            &mut master,
            &[],
            |_, _| -> Result<cg::Answer> {
                panic!("stopped LP must not invoke pricing");
            },
            &mut NoController,
            options,
        )
        .unwrap();
        assert_eq!(
            output.status,
            if iteration_stop {
                cg::Status::IterationLimit
            } else {
                cg::Status::TimeLimit
            }
        );
        assert_eq!(output.pricing_calls, 0);
        assert_eq!(output.bound, f64::NEG_INFINITY);
        assert!(!master.solved);
    }
}
#[test]
fn empty_master_checks_its_required_row_activity() {
    for (rhs, expected) in [
        (0., LpSolveOutcome::Optimal),
        (1., LpSolveOutcome::PrimalInfeasible),
    ] {
        let mut master = HighsMaster::new().unwrap();
        master.add_row(RowSense::Equal, rhs, &[]).unwrap();
        assert_eq!(
            master
                .solve_with_time_limit(Duration::from_secs(60))
                .unwrap(),
            expected
        );
    }
}

#[test]
fn cg_stop_after_column_insertion_keeps_the_last_certificate() {
    struct StopNext;
    impl cg::Controller<HighsMaster> for StopNext {
        fn master_solved(&mut self, master: &HighsMaster, _: &[i32], _: &[usize]) -> Result<()> {
            master.set_double("time_limit", 0.)
        }
    }
    let mut master = model();
    let mut options = cg::Options::default();
    options.stabilize = false;
    let output = cg::optimize(
        &mut master,
        &[],
        |_, duals| {
            Ok(cg::Answer {
                columns: vec![cg::Column {
                    cost: 1.,
                    reduced_cost: 1. - duals[0],
                    entries: vec![(0, 1.)],
                    acceptable: true,
                    tag: 0,
                    identity: "improving".into(),
                }],
                minimum: 1. - duals[0],
                exact: true,
                bound_minimum: None,
            })
        },
        &mut StopNext,
        options,
    )
    .unwrap();
    assert_eq!(output.status, cg::Status::TimeLimit);
    assert_eq!(output.pricing_calls, 1);
    assert_eq!(output.bound, 1.);
    assert_eq!(output.lp, 2.);
    assert_eq!(output.history.len(), 1);
    assert!(!master.solved);
}
