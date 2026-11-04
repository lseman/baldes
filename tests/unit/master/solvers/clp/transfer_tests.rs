use super::*;

#[test]
fn transport_preserves_the_exact_lp_point_and_counters_without_resolving() {
    for solver in [MasterSolver::Clp, MasterSolver::Ipm]
        .into_iter()
        .chain(cfg!(feature = "highs").then_some(MasterSolver::Highs))
    {
        let mut source = ClpMaster::new().unwrap();
        source.set_solver(solver);
        for _ in 0..3 {
            source.add_row(RowSense::Equal, 1., &[]).unwrap();
        }
        for (cost, entries) in [
            (2., [(0, 1.), (1, 1.)]),
            (3., [(1, 1.), (2, 1.)]),
            (4., [(0, 1.), (2, 1.)]),
        ] {
            source.add_column(cost, &entries).unwrap();
        }
        assert!(source.solve());
        let mut restored = source.export_for_worker().unwrap().restore().unwrap();
        assert!(restored.solved);
        assert_eq!(restored.status, source.status);
        assert_eq!(restored.objective, source.objective);
        assert_eq!(restored.values, source.values);
        assert_eq!(restored.duals, source.duals);
        assert_eq!(restored.reduced_costs, source.reduced_costs);
        assert_eq!(restored.simplex_iterations, source.simplex_iterations);
        assert_eq!(restored.ipm_iterations, source.ipm_iterations);
        // The reconstructed backend remains independently editable and usable.
        restored.set_cost(0, 4.).unwrap();
        assert!(!restored.solved);
        assert!(restored.solve());
        assert!((restored.objective - 5.5).abs() < 1e-7);
        assert!((source.objective - 4.5).abs() < 1e-7);
        assert_eq!(source.costs[0], 2.);
    }
}
