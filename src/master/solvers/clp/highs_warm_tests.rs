use super::*;
fn compare_cold(master: &ClpMaster) {
    let mut cold = ClpMaster::new().unwrap();
    cold.set_solver(MasterSolver::Highs);
    for (&sense, &rhs) in master.senses.iter().zip(&master.rhs) {
        cold.add_row(sense, rhs, &[]).unwrap();
    }
    for (j, &cost) in master.costs.iter().enumerate() {
        cold.add_column(cost, &master.column_entries(j).unwrap())
            .unwrap();
        cold.set_upper(j, master.upper[j]).unwrap();
    }
    cold.objective_offset = master.objective_offset;
    assert!(cold.solve());
    assert!((cold.objective - master.objective).abs() < 1e-8);
}
#[test]
fn persistent_highs_reuses_basis_across_edits_and_independent_forks() {
    let mut master = ClpMaster::new().unwrap();
    master.set_solver(MasterSolver::Highs);
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(3., &[(0, 1.)]).unwrap();
    master.add_column(1., &[(0, 1.)]).unwrap();
    assert!(master.solve());
    let identity = master.highs_backend.as_ref().unwrap().native_identity();
    assert!(master.solve());
    assert_eq!(
        master.highs_backend.as_ref().unwrap().native_identity(),
        identity
    );
    assert_eq!(
        master
            .highs_backend
            .as_ref()
            .unwrap()
            .simplex_iterations()
            .unwrap(),
        0
    );
    let mut fork = master.try_clone().unwrap();
    assert_ne!(
        fork.highs_backend.as_ref().unwrap().native_identity(),
        identity
    );
    // Force a native solve of the identical LP, rather than serving its
    // copied solution cache, to verify the installed basis itself.
    fork.set_cost(0, 3.).unwrap();
    assert!(fork.solve());
    assert_eq!(
        fork.highs_backend
            .as_ref()
            .unwrap()
            .simplex_iterations()
            .unwrap(),
        0
    );
    master.add_column(0.5, &[(0, 1.)]).unwrap();
    assert!(master.solve());
    assert_eq!(
        master.highs_backend.as_ref().unwrap().native_identity(),
        identity
    );
    assert!((master.objective - 0.5).abs() < 1e-8);
    assert!((fork.objective - 1.).abs() < 1e-8);
    master
        .add_row(RowSense::GreaterEqual, 0.25, &[(0, 1.)])
        .unwrap();
    assert!(master.solve());
    compare_cold(&master);
    master.set_rhs(&[1., 0.5]).unwrap();
    master.set_upper(2, 0.25).unwrap();
    master.set_cost(0, 5.).unwrap();
    master.set_coefficient(1, 1, 1.).unwrap();
    let mut edited_fork = master.try_clone().unwrap();
    assert!(master.solve());
    assert!(edited_fork.solve());
    assert!((edited_fork.objective - master.objective).abs() < 1e-8);
    compare_cold(&master);
    master.remove_cuts(&[1], &[0]).unwrap();
    assert!(master.solve());
    compare_cold(&master);
    assert_eq!(
        master.highs_backend.as_ref().unwrap().native_identity(),
        identity
    );
    assert_eq!(
        master.solve_with_time_limit(Duration::ZERO).unwrap(),
        LpSolveOutcome::TimeLimit
    );
    assert!(!master.solved);
    let previous_pivots = master.simplex_iterations;
    assert!(master.solve());
    assert_eq!(
        master.highs_backend.as_ref().unwrap().native_identity(),
        identity
    );
    assert_eq!(master.simplex_iterations, previous_pivots);
}
