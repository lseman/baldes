use baldes::pricing::Solver;

#[cfg(feature = "clp")]
mod network {
    use baldes::master::{
        formulation::Row,
        model::{NodeSnapshot, RowCoefficients},
        RowSense,
    };
    /// One arc-flow branch row on network 0, as the routing branches were.
    pub fn arc_row(
        node: &NodeSnapshot<2>,
        arcs: usize,
        arc: usize,
        sense: RowSense,
        rhs: f64,
    ) -> baldes::Result<NodeSnapshot<2>> {
        let mut coefficients = vec![0.; arcs];
        if let Some(c) = coefficients.get_mut(arc) {
            *c = 1.;
        } else {
            // An out-of-range arc yields a mismatched coefficient vector,
            // which the snapshot must reject.
            coefficients.push(1.);
        }
        node.with_row(
            Row { sense, rhs },
            &[RowCoefficients {
                network: 0,
                arcs: coefficients,
                per_route: 0.,
            }],
        )
    }
}

fn ordering_solver() -> Solver<2> {
    let mut solver = Solver::<2>::new(4, 1).unwrap();
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

fn bucket_arcs(s: &Solver<2>) -> (usize, usize) {
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
        cg::{self, Answer, Column, Controller, Options},
        clp::ClpMaster,
        RowSense,
    };
    struct Transition {
        done: bool,
        saved_done: bool,
        column: Column,
    }
    impl Controller<ClpMaster> for Transition {
        fn begin_enumeration_transaction(&mut self, _: f64) -> baldes::Result<bool> {
            self.saved_done = self.done;
            Ok(!self.done)
        }
        fn finish_enumeration_transaction(&mut self, commit: bool) {
            if !commit {
                self.done = self.saved_done;
            }
        }
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
        saved_done: false,
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
    assert_eq!(result.warm_start.generated_pool[0].entries, vec![(0, 2.)]);
    assert_eq!(result.lp, 10.);
}

#[cfg(feature = "clp")]
#[test]
fn enumerated_snapshot_survives_siblings_grandchildren_and_failed_setup() {
    use baldes::inputs::{Instance, Rounding};
    use baldes::master::{
        model::{self, EnumerationOptions, Options, Problem},
        RowSense,
    };
    use network::arc_row;
    let instance = Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let problem =
        Problem::set_partitioning(instance.graph(1, Rounding::None).unwrap(), 3.).unwrap();
    let arcs = problem.networks[0].graph.arcs.len();
    let mut options = Options::default();
    options.cg.stabilize = false;
    let parent = model::solve(problem.clone(), options.clone()).unwrap();
    let snapshot = parent
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    let report = snapshot.enumeration_report().clone();
    assert!(report.complete && report.routes > 0, "{report:?}");
    let arc = problem.networks[0]
        .graph
        .arcs
        .iter()
        .position(|a| a.tail == 1 && a.head == 2)
        .unwrap();
    assert!(arc_row(&snapshot, arcs, usize::MAX, RowSense::LessEqual, 0.).is_err());
    // Fail after the model has already been forked and branch rows installed.
    let mut invalid_setup = options.clone();
    invalid_setup
        .pricing_options
        .push(("BALDESmaxNumOfColsPerExactIteration".into(), 0.));
    assert!(arc_row(&snapshot, arcs, arc, RowSense::LessEqual, 0.)
        .unwrap()
        .solve(invalid_setup)
        .is_err());
    let mut expired = options.clone();
    expired.cg.time_limit = std::time::Duration::ZERO;
    let interrupted = snapshot.solve(expired).unwrap();
    assert_eq!(
        interrupted.master.status,
        baldes::master::cg::Status::TimeLimit
    );
    assert_eq!(snapshot.enumeration_report().routes, report.routes);
    for (sense, rhs) in [(RowSense::LessEqual, 0.), (RowSense::GreaterEqual, 1.)] {
        let child = arc_row(&snapshot, arcs, arc, sense, rhs)
            .unwrap()
            .solve(options.clone())
            .unwrap();
        assert!(child.snapshot.enumeration_report().complete);
        assert_eq!(snapshot.enumeration_report().routes, report.routes);
        // The equivalent child of the unenumerated root has the same bound.
        let fresh_child = arc_row(&parent.snapshot, arcs, arc, sense, rhs)
            .unwrap()
            .solve(options.clone())
            .unwrap();
        assert!(
            (child.master.bound - fresh_child.master.bound).abs() < 1e-6,
            "inherited={child:?}, fresh={fresh_child:?}"
        );
        let grandchild = child.snapshot.solve(options.clone()).unwrap();
        assert!(grandchild.snapshot.enumeration_report().complete);
        assert_eq!(snapshot.enumeration_report().routes, report.routes);
    }
}

#[cfg(feature = "clp")]
#[test]
fn tree_controls_enumeration_and_standalone_fixing_independently() {
    use baldes::inputs::{Instance, Rounding};
    use baldes::master::{
        model::Problem,
        tree::{self, ModelTreeOptions, TreeStatus},
    };
    let instance = Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    for (enumerate, fixing) in [(false, true), (false, false), (true, true)] {
        let problem =
            Problem::set_partitioning(instance.graph(1, Rounding::None).unwrap(), 3.).unwrap();
        let mut options = ModelTreeOptions::default();
        options.node.gap_enumeration = enumerate;
        options.node.cg.stabilize = false;
        options.node.pricing_options.push((
            "BALDESapplyReducedCostFixing".into(),
            if fixing { 1. } else { 0. },
        ));
        let result = tree::solve(problem, options).unwrap();
        assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
        assert!((result.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6);
        // Disabling gap enumeration never installs a route universe; the
        // reduced-cost fixing switch is independent of it.
        if !enumerate {
            assert!(
                result.enumeration_reports.iter().all(|r| r.attempts == 0),
                "{result:?}"
            );
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

#[cfg(feature = "clp")]
#[test]
fn warm_starts_preserve_uninserted_columns_and_same_batch_duplicates() {
    use baldes::master::{
        cg::{self, Answer, Column, Controller, NoController, Options},
        clp::ClpMaster,
        RowSense,
    };
    struct EnumerateOnce {
        universe: Option<Vec<Column>>,
        saved: Option<Vec<Column>>,
    }
    impl Controller<ClpMaster> for EnumerateOnce {
        fn begin_enumeration_transaction(&mut self, _: f64) -> baldes::Result<bool> {
            self.saved = self.universe.clone();
            Ok(self.universe.is_some())
        }
        fn finish_enumeration_transaction(&mut self, commit: bool) {
            if commit {
                self.saved = None;
            } else {
                self.universe = self.saved.take();
            }
        }
        fn enumerated_universe(
            &mut self,
            _: &[f64],
            _: f64,
        ) -> baldes::Result<Option<Vec<Column>>> {
            Ok(self.universe.take())
        }
    }
    let known = Column {
        cost: 5.,
        reduced_cost: 0.,
        entries: vec![(0, 1.)],
        acceptable: true,
        tag: 0,
        identity: "loaded".into(),
    };
    let cached = Column {
        cost: 6.,
        reduced_cost: 1.,
        tag: 1,
        identity: "cached".into(),
        ..known.clone()
    };
    let mut parent = ClpMaster::new().unwrap();
    parent.add_row(RowSense::GreaterEqual, 1., &[]).unwrap();
    parent.add_column(known.cost, &known.entries).unwrap();
    let initial = cg::optimize(
        &mut parent,
        &[],
        |_, _| {
            Ok(Answer {
                columns: vec![cached.clone(), cached.clone()],
                minimum: 0.,
                exact: true,
                bound_minimum: None,
            })
        },
        &mut NoController,
        Options {
            known: vec![known.clone()],
            stabilize: false,
            insert_non_negative: false,
            ..Default::default()
        },
    )
    .unwrap();
    let saved = initial.warm_start.clone();
    assert_eq!(saved.columns.len(), 1);
    assert_eq!(saved.generated_pool.len(), 3);
    let parent_values = parent.values.clone();
    let mut left = parent.try_clone().unwrap();
    let mut right = parent.try_clone().unwrap();
    let resume = |master: &mut ClpMaster, warm, universe| {
        cg::optimize(
            master,
            &[],
            |_, _| {
                Ok(Answer {
                    columns: vec![],
                    minimum: 0.,
                    exact: true,
                    bound_minimum: None,
                })
            },
            &mut EnumerateOnce {
                universe: Some(universe),
                saved: None,
            },
            Options {
                warm_start: Some(warm),
                stabilize: false,
                insert_non_negative: false,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let left_result = resume(
        &mut left,
        saved.clone(),
        vec![known.clone(), cached.clone()],
    );
    let right_result = resume(&mut right, saved.clone(), vec![known.clone()]);
    assert_eq!(left.costs, vec![5., 6., 6.]);
    assert_eq!(right.costs, vec![5.]);
    assert_eq!(parent.costs, vec![5.]);
    assert_eq!(parent.values, parent_values);
    assert_eq!(saved.generated_pool.len(), 3);
    assert_eq!(right_result.warm_start.generated_pool.len(), 3);
    let grandchild = resume(&mut left, left_result.warm_start, vec![known, cached]);
    assert_eq!(left.costs, vec![5., 6., 6.]);
    assert_eq!(grandchild.warm_start.generated_pool.len(), 3);
}

#[cfg(feature = "clp")]
#[test]
fn snapshots_keep_cached_columns_and_extend_branch_rows_without_aliasing() {
    use baldes::inputs::{Instance, Rounding};
    use baldes::master::{
        cg::Status,
        model::{self, Options, Problem},
        RowSense,
    };
    use network::arc_row;
    let mut instance = Instance::read("instances/Solomon/R201.txt").unwrap();
    instance.truncate(10).unwrap();
    let graph = instance.graph(12, Rounding::None).unwrap();
    let arcs = graph.arcs.len();
    let problem = Problem::set_partitioning(graph.clone(), 10.).unwrap();
    let mut options = Options::default();
    options.gap_enumeration = false;
    options.cg.insert_non_negative = false;
    let parent = model::solve(problem, options.clone()).unwrap();
    let pool = &parent.master.warm_start.generated_pool;
    let loaded = parent.master.warm_start.columns.iter().flatten().count();
    assert!(
        pool.len() > loaded,
        "the snapshot needs genuinely uninserted columns"
    );
    let original = format!("{pool:?}");
    let arc = graph
        .arcs
        .iter()
        .position(|a| a.tail != graph.source && a.head != graph.sink)
        .unwrap();
    let id = graph.arcs[arc].id;
    let row = parent.duals.len();
    options.cg.time_limit = std::time::Duration::ZERO;
    let left_node = arc_row(&parent.snapshot, arcs, arc, RowSense::LessEqual, 0.).unwrap();
    let left = left_node.solve(options.clone()).unwrap();
    assert_eq!(left.master.status, Status::TimeLimit);
    let left_pool = &left.master.warm_start.generated_pool;
    assert_eq!(left_pool.len(), pool.len());
    for (before, after) in pool.iter().zip(left_pool) {
        assert_eq!(before.identity, after.identity);
        let route = &parent.routes[before.tag as usize].route;
        let count = route.arcs.iter().filter(|&&a| a == id).count() as f64;
        let mut expected = before.entries.clone();
        if count != 0. {
            expected.push((row, count));
        }
        assert_eq!(after.entries, expected);
    }
    let right = parent.snapshot.solve(options.clone()).unwrap();
    assert_eq!(
        format!("{:?}", right.master.warm_start.generated_pool),
        original
    );
    let grandchild = left.snapshot.solve(options).unwrap();
    assert_eq!(
        format!("{:?}", grandchild.master.warm_start.generated_pool),
        format!("{left_pool:?}")
    );
    assert_eq!(
        format!("{:?}", parent.master.warm_start.generated_pool),
        original
    );
}

#[cfg(feature = "clp")]
#[test]
fn child_enumeration_never_reactivates_a_forbidden_cached_arc() {
    use baldes::inputs::{Instance, Rounding};
    use baldes::master::{
        model::{self, Options, Problem},
        RowSense,
    };
    use network::arc_row;
    let instance = Instance::read("tests/master/fixtures/fractional.txt").unwrap();
    let graph = instance.graph(5, Rounding::None).unwrap();
    let arcs = graph.arcs.len();
    let arc = graph
        .arcs
        .iter()
        .position(|a| a.tail == 1 && a.head == 2)
        .unwrap();
    let id = graph.arcs[arc].id;
    let problem = Problem::set_partitioning(graph, 3.).unwrap();
    let mut options = Options::default();
    options.gap_enumeration = false;
    let parent = model::solve(problem, options.clone()).unwrap();
    let original = format!("{:?}", parent.master.warm_start.generated_pool);
    assert!(parent
        .master
        .warm_start
        .generated_pool
        .iter()
        .any(|c| parent.routes[c.tag as usize].route.arcs.contains(&id)));
    // An incumbent authorizes child fixing/enumeration over cached columns.
    options.gap_enumeration = true;
    options.fixing_upper_bound = Some(10.);
    let child = arc_row(&parent.snapshot, arcs, arc, RowSense::LessEqual, 0.)
        .unwrap()
        .solve(options.clone())
        .unwrap();
    // Cached columns using the arc may stay loaded, but the branch row keeps
    // them at zero: the forbidden arc is never reactivated.
    for (route, &value) in child.routes.iter().zip(&child.values) {
        if value > 1e-9 {
            assert!(!route.route.arcs.contains(&id), "{route:?}");
        }
    }
    let sibling = parent.snapshot.solve(options).unwrap();
    assert!(sibling
        .master
        .warm_start
        .columns
        .iter()
        .flatten()
        .filter_map(|c| usize::try_from(c.tag)
            .ok()
            .and_then(|t| sibling.routes.get(t)))
        .any(|r| r.route.arcs.contains(&id)));
    assert_eq!(
        format!("{:?}", parent.master.warm_start.generated_pool),
        original
    );
}

#[cfg(feature = "clp")]
#[test]
fn cut_cleanup_remaps_loaded_and_uninserted_pool_rows_before_resume() {
    use baldes::master::{
        cg::{self, Answer, Column, Controller, Options, WarmStart},
        clp::ClpMaster,
        RowSense,
    };
    struct Cleanup {
        separated: bool,
        protected: usize,
    }
    impl Controller<ClpMaster> for Cleanup {
        fn separates(&self) -> bool {
            true
        }
        fn protected_rows(&self) -> Vec<usize> {
            vec![self.protected]
        }
        fn remap_rows(&mut self, map: &[Option<usize>]) -> baldes::Result<()> {
            self.protected = map[self.protected].unwrap();
            Ok(())
        }
        fn separate(&mut self, master: &mut ClpMaster, _: &[i32]) -> baldes::Result<usize> {
            if self.separated {
                return Ok(0);
            }
            self.separated = true;
            master.add_row(RowSense::LessEqual, 200., &[])?;
            Ok(1)
        }
    }
    let known = Column {
        cost: 5.,
        reduced_cost: 0.,
        entries: vec![(0, 1.), (1, 2.), (2, 3.)],
        acceptable: true,
        tag: 0,
        identity: "loaded".into(),
    };
    let cached = Column {
        cost: 6.,
        reduced_cost: 1.,
        tag: 1,
        identity: "cached".into(),
        ..known.clone()
    };
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::GreaterEqual, 1., &[]).unwrap();
    master.add_row(RowSense::LessEqual, 100., &[]).unwrap(); // removable cut
    master.add_row(RowSense::LessEqual, 100., &[]).unwrap(); // protected branch
    master.add_column(known.cost, &known.entries).unwrap();
    let warm = WarmStart {
        bound: f64::NEG_INFINITY,
        bound_mode: cg::BoundMode::Numerical,
        numerical_bound: f64::NEG_INFINITY,
        certified_bound: f64::NEG_INFINITY,
        stabilization: None,
        proximal: None,
        columns: vec![Some(known.clone())],
        generated_pool: vec![known, cached],
        artificials: vec![],
        cut_artificials: vec![],
        initial_rows: 1,
    };
    let mut controller = Cleanup {
        separated: false,
        protected: 2,
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
        Options {
            warm_start: Some(warm),
            stabilize: false,
            remove_basic_cuts: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        master.rhs.len(),
        2,
        "rhs={:?} protected={} pool={:?}",
        master.rhs,
        controller.protected,
        result.warm_start.generated_pool
    );
    assert_eq!(controller.protected, 1);
    assert_eq!(master.column_entries(0).unwrap(), vec![(0, 1.), (1, 3.)]);
    let saved = &result.warm_start;
    assert_eq!(
        saved.columns[0].as_ref().unwrap().entries,
        vec![(0, 1.), (1, 3.)]
    );
    assert_eq!(saved.generated_pool.len(), 2);
    for column in &saved.generated_pool {
        assert_eq!(column.entries, vec![(0, 1.), (1, 3.)]);
    }
    let mut invalid = saved.clone();
    invalid.generated_pool[1].entries.push((2, 1.));
    let costs = master.costs.clone();
    let error = cg::optimize(
        &mut master,
        &[],
        |_, _| panic!("invalid row metadata must be rejected before pricing"),
        &mut cg::NoController,
        Options {
            warm_start: Some(invalid),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(error.message.contains("warm master metadata"));
    assert_eq!(master.costs, costs);
}

#[cfg(feature = "clp")]
#[test]
fn child_regeneration_rechecks_ryan_foster_for_cached_columns() {
    use baldes::inputs::{Instance, Rounding};
    use baldes::master::model::{self, Options, Problem, RyanFosterMember};
    use baldes::pricing::restrictions::RyanFosterConstraint;
    // Two odd clusters keep one fractional triangle after branching on the
    // other, so both children actually reach fixing/enumeration.
    let instance = Instance::parse("clusters\nVEHICLE\nNUMBER CAPACITY\n6 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 1 .1 1 0 100 0\n3 1 -.1 1 0 100 0\n4 -1 0 1 0 100 0\n5 -1 .1 1 0 100 0\n6 -1 -.1 1 0 100 0\n").unwrap();
    let problem =
        Problem::set_partitioning(instance.graph(8, Rounding::None).unwrap(), 6.).unwrap();
    let mut options = Options::default();
    options.gap_enumeration = false;
    let parent = model::solve(problem, options.clone()).unwrap();
    let pool = format!("{:?}", parent.master.warm_start.generated_pool);
    options.gap_enumeration = true;
    options.fixing_upper_bound = Some(100.);
    for together in [false, true] {
        let rule = RyanFosterConstraint {
            first: 1,
            second: 2,
            together,
        };
        let nodes =
            |route: &baldes::Route| route.nodes.iter().map(|&v| v as usize).collect::<Vec<_>>();
        assert!(parent
            .master
            .warm_start
            .generated_pool
            .iter()
            .any(|column| !rule.satisfied_by(nodes(&parent.routes[column.tag as usize].route))));
        let child = parent
            .snapshot
            .with_ryan_foster(
                &[RyanFosterMember {
                    network: 0,
                    first: 1,
                    second: 2,
                }],
                together,
            )
            .unwrap()
            .solve(options.clone())
            .unwrap();
        // Cached violating columns may remain in the master, but the branch
        // row keeps their solution values at zero.
        for (route, &value) in child.routes.iter().zip(&child.values) {
            if value > 1e-8 {
                assert!(rule.satisfied_by(nodes(&route.route)));
            }
        }
        assert_eq!(
            format!("{:?}", parent.master.warm_start.generated_pool),
            pool
        );
    }
}

#[test]
fn ng_strengthening_cannot_stamp_a_stale_resource_grid_as_current() {
    use baldes::graph::relaxation::{Controls, FractionalPath};
    let mut solver = ordering_solver();
    solver.enumeration_preserve_arc_paths = true;
    solver.graph.set_bounds(2, 0, 0., 0.5).unwrap();
    solver.price_phase(0, 20, false).unwrap();
    let parent = solver;
    let mut solver = parent.fork_for_node();
    solver.graph.set_bounds(2, 0, 0., 2.).unwrap();
    let counts = solver
        .strengthen_ng(
            &[FractionalPath {
                arcs: vec![0, 1, 4, 5],
                value: 0.5,
            }],
            &Controls::default(),
        )
        .unwrap();
    assert_eq!(counts.columns_forbidden, 1);
    let mut fresh = Solver::<2>::new(4, 1).unwrap();
    fresh.graph = solver.graph.clone();
    fresh.enumeration_preserve_arc_paths = true;
    let paths = |routes: Vec<baldes::Route>| {
        let mut paths: Vec<_> = routes.into_iter().map(|route| route.arcs).collect();
        paths.sort();
        paths
    };
    let expected = paths(fresh.enumerate(10.).unwrap());
    // Capacity two allows either customer alone or both in either order.
    assert_eq!(
        expected,
        vec![vec![0, 1, 2], vec![0, 5], vec![3, 2], vec![3, 4, 5]]
    );
    assert_eq!(paths(solver.enumerate(10.).unwrap()), expected);
    let mut sibling = parent.fork_for_node();
    assert_eq!(paths(sibling.enumerate(10.).unwrap()), vec![vec![0, 5]]);
}
#[cfg(feature = "clp")]
#[path = "node_lifecycle/enumeration_transaction.rs"]
mod enumeration_transaction;
