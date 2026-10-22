//! Dual fake-bound failures must not turn bounded LPs into unbounded models.
use super::*;

fn augmented() -> ClpMaster {
    let mut master = ClpMaster::new().unwrap();
    master.set_column_default_upper(f64::MAX).unwrap();
    let amount = 2_f64.powi(41);
    for rhs in [amount, 0., amount] {
        master.add_row(RowSense::GreaterEqual, rhs, &[]).unwrap();
    }
    master.add_column(1., &[(0, 1.), (1, 1.), (2, 1.)]).unwrap();
    let negative = master
        .add_column(-1., &[(0, -1.), (1, -1.), (2, -1.)])
        .unwrap();
    master.set_upper(negative, 0.).unwrap();
    for row in 0..3 {
        master.add_column(100., &[(row, 1.)]).unwrap();
    }
    master
}
#[test]
fn large_bounded_augmented_lp_is_cross_checked_before_unbounded_publication() {
    for scaling in [ClpScaling::Off, ClpScaling::Auto] {
        for keep_work_areas in [false, true] {
            let mut master = augmented();
            master.set_clp_options(ClpOptions {
                scaling,
                keep_work_areas,
            });
            assert_eq!(
                master.solve_limited(100, Duration::from_secs(10)).unwrap(),
                ClpSolveOutcome::Optimal
            );
            assert_eq!(master.objective, 2_f64.powi(41));
            assert_eq!(master.values[0], 2_f64.powi(41));
            assert_eq!(master.upper[0], f64::MAX);
        }
    }
}
#[test]
fn primal_recovery_shares_the_iteration_budget_and_restores_native_limits() {
    fn limits(master: &ClpMaster) -> [f64; 3] {
        let mut values = [0.; 3];
        // SAFETY: live owned model; helper writes exactly three doubles.
        unsafe {
            baldes_clp_solve_limits(master.model.unwrap().as_ptr(), values.as_mut_ptr());
        }
        values
    }
    for iterations in 1..8 {
        let mut master = augmented();
        let before = limits(&master);
        let outcome = master
            .solve_limited(iterations, Duration::from_secs(10))
            .unwrap();
        assert_eq!(limits(&master), before);
        assert!(master.simplex_iterations <= iterations);
        assert!(matches!(
            outcome,
            ClpSolveOutcome::Optimal | ClpSolveOutcome::Stopped
        ));
        if outcome == ClpSolveOutcome::Optimal {
            assert_eq!(master.objective, 2_f64.powi(41));
        }
    }
    let mut unbounded = ClpMaster::new().unwrap();
    unbounded.set_column_default_upper(f64::MAX).unwrap();
    unbounded.add_column(-1., &[]).unwrap();
    assert_eq!(
        unbounded
            .solve_limited(100, Duration::from_secs(10))
            .unwrap(),
        ClpSolveOutcome::DualInfeasible
    );
}
