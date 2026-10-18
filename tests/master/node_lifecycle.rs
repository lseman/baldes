use baldes::pricing::Solver;

fn ordering_solver() -> Solver {
    let mut solver = Solver::new(4, 1).unwrap();
    for v in 0..4 {
        solver.graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    for (u, v, cost) in [
        (0, 1, 1.),
        (1, 2, 1.),
        (2, 3, 1.),
        (0, 2, 1.),
        (2, 1, 2.),
        (1, 3, 1.),
    ] {
        let id = solver.graph.add_arc(u, v, cost).unwrap();
        solver.graph.arc_mut(id).unwrap().consumption[0] = if v == 3 { 0. } else { 1. };
    }
    solver
}

#[test]
fn fork_keeps_arc_orderings_and_sibling_enumeration_is_independent() {
    for bidirectional in [0., 1.] {
        let mut parent = ordering_solver();
        parent
            .set_option("BALDESuseBidirectionalSearch", bidirectional)
            .unwrap();
        parent.enumeration_preserve_arc_paths = true;
        let routes = parent.enumerate(10.).unwrap();
        assert_eq!(routes.len(), 4);
        assert!(routes.iter().any(|r| r.arcs == [3, 4, 5]));
        let mut left = parent.fork_for_node();
        let mut right = parent.fork_for_node();
        // The more expensive ordering of {1,2} becomes cheaper after a branch dual.
        left.graph.arcs[1].reduced_cost = 100.;
        let left_routes = left.inspect_enumeration(10.).unwrap().unwrap();
        assert_eq!(left_routes.len(), 3);
        assert!(left_routes
            .iter()
            .any(|r| r.arcs == [3, 4, 5] && r.cost == 4.));
        assert_eq!(right.inspect_enumeration(10.).unwrap().unwrap().len(), 4);
        assert_eq!(parent.enumeration.as_ref().unwrap().paths.len, 4);
        assert_eq!(parent.graph.arcs[1].reduced_cost, 1.);
        // Empty child inspection remains an enumerated universe, not a signal
        // to fall back to pricing all routes again.
        assert!(left.inspect_enumeration(-100.).unwrap().unwrap().is_empty());
        assert!(left.enumeration.is_some());
        assert_eq!(parent.enumeration.as_ref().unwrap().paths.len, 4);
    }
}

fn bucket_arcs(s: &Solver) -> (usize, usize) {
    let grid = &s.engine.as_ref().unwrap().grid;
    (
        grid.forward
            .buckets
            .iter()
            .flatten()
            .map(|b| b.arcs.len())
            .sum(),
        grid.backward
            .buckets
            .iter()
            .flatten()
            .map(|b| b.arcs.len())
            .sum(),
    )
}

#[test]
fn fork_carries_fixing_options_and_rebuilds_dual_dependent_labels() {
    let mut parent = ordering_solver();
    parent.price_phase(0, 20, false).unwrap();
    let before = bucket_arcs(&parent);
    parent.fix(2.5).unwrap();
    let fixed = bucket_arcs(&parent);
    assert!(fixed.0 < before.0 || fixed.1 < before.1);
    let mut child = parent.fork_for_node();
    assert_eq!(bucket_arcs(&child), fixed);
    let engine = child.engine.as_ref().unwrap();
    assert!(!engine.half_ready && !engine.full_ready);
    assert!(engine.forward.labels.is_empty() && engine.backward.labels.is_empty());
    child
        .set_option("BALDEShardTimeThresholdInPricing", 1.)
        .unwrap();
    assert_eq!(
        bucket_arcs(&child),
        fixed,
        "runtime budget must preserve fixed arcs"
    );
    child.graph.arcs[0].reduced_cost = 0.5;
    let result = child.price_phase(0, 20, false).unwrap();
    assert!(result.exact);
    assert_eq!(result.minimum_reduced_cost, 1.5);
    assert_eq!(bucket_arcs(&parent), fixed);
    assert_eq!(parent.graph.arcs[0].reduced_cost, 1.);
}

#[cfg(feature = "clp")]
#[test]
fn enumeration_rewrites_existing_master_coefficients() {
    use baldes::master::{
        clp::ClpMaster,
        column_generation::{self as cg, Answer, Column, Controller, Options},
        RowSense,
    };
    struct Transition {
        done: bool,
        column: Column,
    }
    impl Controller<ClpMaster> for Transition {
        fn enumerated_universe(
            &mut self,
            _: &[f64],
            _: f64,
        ) -> baldes::Result<Option<Vec<Column>>> {
            if self.done {
                return Ok(None);
            }
            self.done = true;
            Ok(Some(vec![self.column.clone()]))
        }
    }
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::GreaterEqual, 2., &[]).unwrap();
    master.add_column(10., &[(0, 1.)]).unwrap();
    let column = Column {
        cost: 10.,
        reduced_cost: 0.,
        entries: vec![(0, 1.)],
        acceptable: true,
        tag: 0,
        identity: "same-route".into(),
    };
    let mut updated = column.clone();
    updated.entries = vec![(0, 2.)];
    let mut controller = Transition {
        done: false,
        column: updated,
    };
    let options = Options {
        known: vec![column],
        stabilize: false,
        ..Default::default()
    };
    let result = cg::optimize(
        &mut master,
        &[],
        |_, _| {
            Ok(Answer {
                columns: vec![],
                minimum: 0.,
                exact: true,
                bound_minimum: None,
            })
        },
        &mut controller,
        options,
    )
    .unwrap();
    assert!(controller.done);
    assert_eq!(master.column_entries(0).unwrap(), vec![(0, 2.)]);
    assert_eq!(result.lp, 10.);
}

#[cfg(feature = "clp")]
#[test]
fn enumerated_snapshot_survives_siblings_grandchildren_and_failed_setup() {
    use baldes::master::{
        root::{self, RootOptions, RootProblem},
        RowSense,
    };
    use baldes::vrptw::{Instance, Rounding};
    let instance = Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let problem =
        RootProblem::set_partitioning(instance.graph(1, Rounding::None).unwrap(), 3.).unwrap();
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.incumbent = 100.;
    options.cg.stabilize = false;
    options.cg.multiplicity = 3.;
    let parent = root::solve(problem.clone(), options.clone()).unwrap();
    let snapshot = parent.snapshot.as_ref().unwrap();
    let count = snapshot.enumerated_routes().expect("root must enumerate");
    assert!(count > 0);
    let arc = problem
        .graph
        .arcs
        .iter()
        .position(|a| a.tail == 1 && a.head == 2)
        .unwrap();
    assert!(root::inherit(
        snapshot,
        &[(usize::MAX, RowSense::LessEqual, 0.)],
        options.clone()
    )
    .is_err());
    // Fail after the model has already been forked and branch rows installed.
    let mut invalid_setup = options.clone();
    invalid_setup
        .pricing_options
        .push(("BALDESmaxNumOfColsPerExactIteration".into(), 0.));
    assert!(root::inherit(snapshot, &[(arc, RowSense::LessEqual, 0.)], invalid_setup).is_err());
    let mut expired = options.clone();
    expired.cg.time_limit = std::time::Duration::ZERO;
    let interrupted = root::inherit(snapshot, &[], expired).unwrap();
    assert_eq!(
        interrupted.master.status,
        baldes::master::column_generation::Status::TimeLimit
    );
    assert_eq!(snapshot.enumerated_routes(), Some(count));
    for (sense, rhs) in [(RowSense::LessEqual, 0.), (RowSense::GreaterEqual, 1.)] {
        let child = root::inherit(snapshot, &[(arc, sense, rhs)], options.clone()).unwrap();
        let child_snapshot = child.snapshot.as_ref().unwrap();
        assert_eq!(child_snapshot.branch_depth(), 1);
        assert!(child_snapshot.enumerated_routes().is_some());
        assert_eq!(snapshot.branch_depth(), 0);
        assert_eq!(snapshot.enumerated_routes(), Some(count));
        let cold = problem.clone();
        // Construct the equivalent independent one-row child through a root
        // snapshot with enumeration disabled, then compare certified bounds.
        let mut cold_options = options.clone();
        cold_options.enumerate = false;
        let fresh = root::solve(cold.clone(), cold_options.clone()).unwrap();
        let fresh_child = root::inherit(
            fresh.snapshot.as_ref().unwrap(),
            &[(arc, sense, rhs)],
            cold_options,
        )
        .unwrap();
        assert!(
            (child.master.bound - fresh_child.master.bound).abs() < 1e-6,
            "inherited={child:?}, fresh={fresh_child:?}"
        );
        let grandchild = root::inherit(child_snapshot, &[], options.clone()).unwrap();
        assert_eq!(grandchild.snapshot.as_ref().unwrap().branch_depth(), 1);
        assert!(grandchild
            .snapshot
            .as_ref()
            .unwrap()
            .enumerated_routes()
            .is_some());
        assert_eq!(snapshot.enumerated_routes(), Some(count));
    }
}

#[cfg(feature = "clp")]
#[test]
fn tree_honors_enumeration_control_during_node_processing() {
    use baldes::master::{
        root::{RootOptions, RootProblem},
        tree::{self, TreeOptions, TreeStatus},
    };
    use baldes::vrptw::{Instance, Rounding};
    let instance = Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    for enumerate in [false, true] {
        let problem =
            RootProblem::set_partitioning(instance.graph(1, Rounding::None).unwrap(), 3.).unwrap();
        let mut root = RootOptions::default();
        root.enumerate = enumerate;
        root.cg.stabilize = false;
        let result = tree::solve(
            problem,
            TreeOptions {
                root,
                strong_branching: None,
                diving: None,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
        assert!((result.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6);
        if enumerate {
            assert!(
                result.fixing_calls > 0 && result.enumerated_routes > 0,
                "{result:?}"
            );
        } else {
            assert_eq!(result.fixing_calls, 0);
            assert_eq!(result.enumerated_routes, 0);
        }
    }
}

#[test]
fn fixing_without_prior_pricing_keeps_its_grid_in_a_child() {
    let mut parent = ordering_solver();
    parent.complete_labels(true).unwrap();
    parent.fix(2.5).unwrap();
    let fixed = bucket_arcs(&parent);
    let mut child = parent.fork_for_node();
    child.price_phase(0, 20, false).unwrap();
    assert_eq!(bucket_arcs(&child), fixed);
    assert_eq!(bucket_arcs(&parent), fixed);
}
