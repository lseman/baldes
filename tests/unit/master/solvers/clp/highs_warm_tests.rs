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
    assert_eq!(
        master.highs_backend.as_ref().unwrap().configured_threads(),
        Some(std::thread::available_parallelism().map_or(1, usize::from))
    );
    assert!(master.set_backend_threads(1).is_err());
    assert_eq!(master.backend_threads(), 0);
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

#[test]
fn highs_owns_the_only_native_matrix_and_switches_preserve_edits() {
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(3., &[(0, 1.)]).unwrap();
    master.set_solver(MasterSolver::Highs);
    master.add_column(1., &[(0, 1.)]).unwrap();
    assert!(master.model.is_none());
    assert!(master.solve());
    assert!((master.objective - 1.).abs() < 1e-8);
    let fork = master.try_clone().unwrap();
    assert!(fork.model.is_none());
    master.set_coefficient(0, 1, 2.).unwrap();
    master.set_rhs(&[2.]).unwrap();
    master.set_cost(1, 2.).unwrap();
    master.set_upper(1, 0.5).unwrap();
    master.objective_offset = 7.;
    assert!(master.solve());
    assert!((master.objective - 11.).abs() < 1e-8);
    assert_eq!(fork.column_entries(1).unwrap(), vec![(0, 1.)]);
    assert!(master.model.is_none());
    master.set_solver(MasterSolver::Clp);
    // Reads and forks remain valid before the deferred conversion.
    assert_eq!(master.column_entries(1).unwrap(), vec![(0, 2.)]);
    let mut pending = master.try_clone().unwrap();
    assert!(pending.solve());
    assert!(pending.model.is_some());
    assert!(pending.highs_backend.is_none());
    assert!((pending.objective - 11.).abs() < 1e-8);
    master.set_cost(0, 4.).unwrap();
    assert!(master.model.is_some());
    assert!(master.highs_backend.is_none());
    assert!(master.solve());
    assert!((master.objective - 12.).abs() < 1e-8);
    master.set_solver(MasterSolver::Highs);
    master.remove(&[], &[0]).unwrap();
    assert!(master.model.is_none());
    master.set_upper(0, 1.).unwrap();
    assert!(master.solve());
    assert!((master.objective - 9.).abs() < 1e-8);
    master.set_solver(MasterSolver::Ipm);
    master.set_tolerances(1e-7, 1e-7).unwrap();
    assert!(master.model.is_some());
    assert!(master.highs_backend.is_none());
    assert!(master.solve());
    assert!((master.objective - 9.).abs() < 1e-5);
}

#[test]
fn rejected_highs_edit_keeps_the_only_matrix_available() {
    let mut master = ClpMaster::new().unwrap();
    master.set_solver(MasterSolver::Highs);
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(2., &[(0, 1.)]).unwrap();
    assert!(master.solve());
    assert!(master.add_columns(&[(1., &[(4, 1.)])]).is_err());
    assert!(master
        .add_rows(&[(RowSense::Equal, 0., &[(4, 1.)])])
        .is_err());
    assert!(master.set_coefficient(0, 4, 1.).is_err());
    assert!(master.set_upper(0, -1.).is_err());
    assert!(master.model.is_none());
    assert!(master.highs_backend.is_some());
    assert_eq!(master.column_entries(0).unwrap(), vec![(0, 1.)]);
    assert!(master.solve());
    assert!((master.objective - 2.).abs() < 1e-8);
}

#[test]
fn highs_column_insertion_preserves_the_compacted_wrapper_solution() {
    let mut master = ClpMaster::new().unwrap();
    master.set_solver(MasterSolver::Highs);
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    master.add_column(4., &[(0, 1.)]).unwrap();
    master.add_column(3., &[(0, 1.)]).unwrap();
    master.add_column(1., &[(0, 1.)]).unwrap();
    assert!(master.solve());
    master.remove_cuts(&[], &[0, 1]).unwrap();
    assert_eq!(master.values, vec![1.]);
    master.add_column(2., &[(0, 1.)]).unwrap();
    assert!(master.solved);
    assert_eq!(master.values, vec![1., 0.]);
    master.add_column(0.5, &[(0, 1.)]).unwrap();
    assert!(!master.solved);
    assert_eq!(master.values, vec![1., 0.]);
    assert!(master.solve());
    assert!((master.objective - 0.5).abs() < 1e-8);
}

#[test]
fn highs_sparse_read_cache_tracks_edits_and_independent_forks() {
    let mut master = ClpMaster::new().unwrap();
    master.set_solver(MasterSolver::Highs);
    master
        .add_rows(&[(RowSense::Equal, 1., &[]), (RowSense::LessEqual, 3., &[])])
        .unwrap();
    master.add_column(1., &[(1, 2.), (0, 1.)]).unwrap();
    assert_eq!(master.column_entries(0).unwrap(), vec![(0, 1.), (1, 2.)]);
    let mut child = master.try_clone().unwrap();
    child.set_coefficient(1, 0, 4.).unwrap();
    assert_eq!(child.column_entries(0).unwrap(), vec![(0, 1.), (1, 4.)]);
    assert_eq!(master.column_entries(0).unwrap(), vec![(0, 1.), (1, 2.)]);
    master.add_row(RowSense::LessEqual, 5., &[(0, 3.)]).unwrap();
    assert_eq!(
        master.column_entries(0).unwrap(),
        vec![(0, 1.), (1, 2.), (2, 3.)]
    );
    master.add_column(2., &[(2, 1.)]).unwrap();
    assert_eq!(master.column_entries(1).unwrap(), vec![(2, 1.)]);
    master.set_coefficient(1, 0, 0.).unwrap();
    assert_eq!(master.column_entries(0).unwrap(), vec![(0, 1.), (2, 3.)]);
    master.remove(&[0], &[1]).unwrap();
    assert_eq!(master.column_entries(0).unwrap(), vec![(1, 3.)]);
    master.set_cost(0, 7.).unwrap();
    master.set_upper(0, 2.).unwrap();
    master.set_rhs(&[9., 10.]).unwrap();
    let copy = master.try_clone().unwrap();
    assert_eq!(copy.costs, vec![7.]);
    assert_eq!(copy.upper, vec![2.]);
    assert_eq!(copy.rhs, vec![9., 10.]);
    assert_eq!(copy.column_entries(0).unwrap(), vec![(1, 3.)]);
    assert!(master.model.is_none());
    assert!(copy.model.is_none());
}
