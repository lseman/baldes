#![cfg(feature = "gurobi")]
use baldes::master::{solvers::GurobiMaster, LpMaster, LpSolveOutcome, RowSense};
use std::time::Duration;

fn fixture() -> GurobiMaster {
    let mut m = GurobiMaster::new().unwrap();
    m.add_rows(&[
        (RowSense::GreaterEqual, 1., &[]),
        (RowSense::Equal, 1., &[]),
    ])
    .unwrap();
    m.add_columns(&[(3., &[(0, 1.)]), (2., &[(1, 1.)])])
        .unwrap();
    m
}

#[test]
fn worker_budgets_are_configurable_and_survive_model_copies() {
    let automatic = GurobiMaster::new().unwrap();
    assert_eq!(
        automatic.configured_threads(),
        std::thread::available_parallelism().map_or(1, usize::from)
    );
    let mut single = GurobiMaster::with_threads(1).unwrap();
    single.add_row(RowSense::Equal, 2., &[]).unwrap();
    single.add_column(3., &[(0, 1.)]).unwrap();
    assert!(single.solve());
    let mut fork = single.try_clone().unwrap();
    assert_eq!(fork.configured_threads(), 1);
    fork.set_rhs(&[3.]).unwrap();
    assert!(fork.solve());
    assert_eq!(fork.objective(), 9.);
    assert_eq!(single.objective(), 6.);
    assert!(GurobiMaster::with_threads(usize::MAX).is_err());
}

#[test]
fn edits_duals_and_independent_clone() {
    let mut m = fixture();
    assert!(m.solve());
    assert!((m.objective() - 5.).abs() < 1e-7);
    assert_eq!(m.values(), &[1., 1.]);
    assert_eq!(m.duals(), &[3., 2.]);
    assert_eq!(m.reduced_costs(), &[0., 0.]);
    let mut child = m.try_clone().unwrap();
    child.set_rhs(&[2., 1.]).unwrap();
    assert!(child.solve());
    assert!((child.objective() - 8.).abs() < 1e-7);
    assert_eq!(m.rhs(), &[1., 1.]);
    assert_eq!(m.objective(), 5.);
    m.add_column(1., &[(0, 1.), (0, 1.)]).unwrap();
    assert_eq!(m.column_entries(2).unwrap(), vec![(0, 2.)]);
    assert!(m.solve());
    assert!((m.objective() - 2.5).abs() < 1e-7);
    m.set_coefficient(0, 2, 1.).unwrap();
    m.set_cost(1, 4.).unwrap();
    m.set_upper(2, 0.5).unwrap();
    assert!(m.solve());
    assert!((m.objective() - 6.).abs() < 1e-7);
    let (rows, columns) = m.remove(&[1, 1], &[1]).unwrap();
    assert_eq!(rows, vec![Some(0), None]);
    assert_eq!(columns, vec![Some(0), None, Some(1)]);
    assert_eq!(m.column_entries(1).unwrap(), vec![(0, 1.)]);
    assert!(m.solve());
    assert!((m.objective() - 2.).abs() < 1e-7);
}

#[test]
fn infeasible_and_zero_budget_do_not_authorize_solutions() {
    let mut m = fixture();
    assert_eq!(
        m.solve_with_time_limit(Duration::ZERO).unwrap(),
        LpSolveOutcome::TimeLimit
    );
    assert!(!m.solved());
    m.set_upper(1, 0.).unwrap();
    assert_eq!(
        m.solve_with_time_limit(Duration::from_secs(10)).unwrap(),
        LpSolveOutcome::PrimalInfeasible
    );
    assert!(!m.solved());
    let mut empty = GurobiMaster::new().unwrap();
    empty.add_row(RowSense::Equal, 1., &[]).unwrap();
    assert!(!empty.solve());
    assert_eq!(empty.status(), 1);
}

#[test]
fn removed_basic_cut_keeps_compacted_solution() {
    let mut m = fixture();
    let row = m.add_row(RowSense::LessEqual, 10., &[(0, 1.)]).unwrap();
    assert!(m.solve());
    assert!(m.row_is_basic(row).unwrap());
    m.remove_cuts(&[row], &[]).unwrap();
    assert!(m.solved());
    assert_eq!(m.duals().len(), 2);
    m.add_column(9., &[(0, 1.)]).unwrap();
    assert!(m.solved());
    assert_eq!(m.values(), &[1., 1., 0.]);
    assert!(m.solve());
}

#[cfg(feature = "clp")]
#[test]
fn wrapper_switches_backends_and_forks_without_losing_matrix() {
    use baldes::master::clp::{ClpMaster, MasterSolver};
    let mut m = ClpMaster::new().unwrap();
    m.add_row(RowSense::Equal, 2., &[]).unwrap();
    m.add_column(3., &[(0, 1.)]).unwrap();
    for solver in [
        MasterSolver::Gurobi,
        MasterSolver::Clp,
        MasterSolver::Gurobi,
    ] {
        m.set_solver(solver);
        assert!(m.solve());
        assert_eq!(m.objective(), 6.);
        let mut child = m.try_clone().unwrap();
        child.set_rhs(&[3.]).unwrap();
        assert!(child.solve());
        assert_eq!(child.objective(), 9.);
        assert_eq!(m.objective(), 6.);
    }
    #[cfg(feature = "highs")]
    for solver in [
        MasterSolver::Highs,
        MasterSolver::Gurobi,
        MasterSolver::Highs,
    ] {
        m.set_solver(solver);
        assert!(m.solve());
        assert_eq!(m.objective(), 6.);
    }
    m.set_solver(MasterSolver::Gurobi);
    assert_eq!(
        m.solve_limited(0, Duration::from_secs(1)).unwrap(),
        baldes::master::clp::ClpSolveOutcome::Stopped
    );
    assert!(!m.solved);
}

#[cfg(feature = "clp")]
#[test]
fn routing_root_matches_clp() {
    use baldes::inputs::{Instance, Rounding};
    use baldes::master::root::{self, MasterSolver, RootOptions, RootProblem};
    let instance = Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let mut objectives = vec![];
    for solver in [MasterSolver::Clp, MasterSolver::Gurobi] {
        let graph = instance.graph(3, Rounding::None).unwrap();
        let problem = RootProblem::set_partitioning(graph, instance.vehicles as f64).unwrap();
        let options = RootOptions {
            master_solver: solver,
            ..Default::default()
        };
        let output = root::solve(problem, options).unwrap();
        assert_eq!(output.master.status, baldes::master::cg::Status::Optimal);
        objectives.push(output.master.lp);
    }
    assert!((objectives[0] - objectives[1]).abs() < 1e-6);
}
