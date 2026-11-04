use super::*;

#[derive(Clone)]
struct Snapshot {
    group: i32,
    count: usize,
}

#[derive(Default)]
struct Oracle {
    live: BTreeMap<i32, usize>,
    trace: Vec<(char, i32, usize, Option<usize>)>,
    reject: bool,
    fail_record: bool,
}
impl ImportOracle<2> for Oracle {
    fn setup_node(
        &mut self,
        node: &ImportOracleNode<2>,
        state: Option<&ImportOracleState>,
        _: Duration,
    ) -> Result<ImportOracleSetup> {
        let group = node.subproblem().id;
        let previous = state.map(|state| {
            let snapshot = state.downcast_ref::<Snapshot>().unwrap();
            assert_eq!(snapshot.group, group);
            snapshot.count
        });
        self.trace.push(('s', group, node.depth(), previous));
        self.live
            .insert(group, previous.unwrap_or(0) + node.depth() + 1);
        assert_eq!(
            node.original_bounds(0)?.lower,
            if node.depth() == 0 { 0. } else { 1. }
        );
        assert_eq!(
            node.column_bounds(0)?.upper,
            if group == 7 { 1. } else { 2. }
        );
        Ok(if self.reject {
            ImportOracleSetup::Infeasible
        } else {
            ImportOracleSetup::Ready
        })
    }
    fn price(&mut self, point: &ImportPricing<2>, _: Duration) -> Result<ImportOracleReply> {
        let group = point.subproblem().id;
        *self.live.get_mut(&group).unwrap() += 1;
        self.trace.push(('p', group, 0, None));
        Ok(ImportOracleReply {
            batch: ColumnBatch {
                columns: vec![],
                complete: false,
            },
            minimum: Some(point.numerical_minimum(0.)?),
        })
    }
    fn record_node(
        &mut self,
        node: &ImportOracleNode<2>,
        _: Duration,
    ) -> Result<Option<ImportOracleState>> {
        if self.fail_record {
            return Err(invalid("oracle record failure"));
        }
        let group = node.subproblem().id;
        self.trace.push(('r', group, node.depth(), None));
        Ok(Some(ImportOracleState::new(Snapshot {
            group,
            count: self.live[&group],
        })))
    }
}

fn options(mode: cg::BoundMode) -> ModelTreeOptions {
    let mut options = ModelTreeOptions::default();
    options.node.cg.bound_mode = mode;
    options.node.cg.stabilize = false;
    options.node_limit = 1;
    options
}

fn session() -> MultiImport<2> {
    let bounds = |lower, upper| crate::master::preprocessing::Bounds { lower, upper };
    let root = crate::master::model::imported::solve(
        Problem {
            networks: vec![],
            rows: vec![Row {
                sense: RowSense::GreaterEqual,
                rhs: 0.5,
            }],
        },
        vec![MasterVariable {
            id: 0,
            cost: 1.,
            bounds: bounds(0., 2.),
            integer: true,
            entries: vec![(0, 1.)],
        }],
        [7, 42]
            .into_iter()
            .map(|id| Subproblem {
                id,
                variables: vec![0],
                multiplicity: bounds(0., 3.),
                columns: vec![Column {
                    id: 0,
                    cost: 0.,
                    entries: vec![(0, 1.)],
                    integer: false,
                    bounds: bounds(0., 3.),
                }],
            })
            .collect(),
        Options::default(),
    )
    .unwrap();
    root.snapshot()
        .unwrap()
        .begin_imports(&[7, 42])
        .unwrap()
        .with_column_bounds(7, 0, bounds(0., 1.))
        .unwrap()
        .with_column_bounds(42, 0, bounds(0., 2.))
        .unwrap()
}

#[test]
fn tolerance_closed_import_bounds_survive_finalization_and_immutable_resume() {
    let session = session()
        .collect(&ImportOptions::default(), |_, _| {
            Ok(ColumnBatch {
                columns: vec![],
                complete: true,
            })
        })
        .unwrap()
        .session;
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let mut initial = session.begin_tree();
        initial.retain_solution(
            1.,
            vec![],
            Values {
                original: vec![(0, 1.)],
                columns: [7, 42]
                    .into_iter()
                    .map(|subproblem| crate::master::model::imported::ColumnValue {
                        subproblem,
                        column: 0,
                        value: 1.,
                    })
                    .collect(),
            },
        );
        let proof = 1. - 5e-7;
        initial.pending[0].bound = proof;
        initial.bound_mode = Some(mode);
        let mut oracle = Oracle::default();
        let first = initial
            .run_with_oracle(&options(mode), &ImportOptions::default(), &mut oracle)
            .unwrap();
        assert_eq!(first.status, TreeStatus::Optimal);
        assert_eq!(first.open_nodes, 0);
        assert_eq!(first.bound, proof);
        assert_eq!(first.incumbent, 1.);
        assert!(oracle.trace.is_empty());
        let resumed = first
            .tree
            .run_with_oracle(&options(mode), &ImportOptions::default(), &mut oracle)
            .unwrap();
        assert_eq!(resumed.bound, proof);
        assert_eq!(resumed.open_nodes, 0);
        assert_eq!(initial.pending.len(), 1);
        assert_eq!(initial.closed_bound, f64::INFINITY);
    }
}

#[test]
fn imported_oracle_records_before_branching_and_restores_independent_siblings_and_resumes() {
    let initial = session().begin_tree();
    let mut oracle = Oracle::default();
    let first = initial
        .run_with_oracle(
            &options(cg::BoundMode::Numerical),
            &ImportOptions::default(),
            &mut oracle,
        )
        .unwrap();
    assert_eq!(first.open_nodes, 2);
    assert_eq!(first.nodes, 1);
    assert_eq!(
        oracle
            .trace
            .iter()
            .filter(|(kind, _, _, _)| *kind == 'p')
            .count(),
        2
    );
    for child in &first.tree.pending {
        assert_eq!(child.depth, 1);
        assert_eq!(
            child.oracle_states[&7]
                .downcast_ref::<Snapshot>()
                .unwrap()
                .count,
            2
        );
        assert_eq!(
            child.oracle_states[&42]
                .downcast_ref::<Snapshot>()
                .unwrap()
                .count,
            2
        );
    }
    assert!(Rc::ptr_eq(
        &first.tree.pending[0].oracle_states,
        &first.tree.pending[1].oracle_states
    ));
    assert!(initial.pending[0].oracle_states.is_empty());
    let right = first
        .tree
        .pending
        .iter()
        .find(|child| {
            let context = ImportOracleNode::new(
                child.session.definition.clone(),
                child.session.context.clone(),
                7,
                child.depth,
            )
            .unwrap();
            context.original_bounds(0).unwrap().lower == 1.
        })
        .unwrap()
        .clone();
    let mut fork = first.tree.clone();
    fork.pending = vec![right];
    oracle.live.insert(7, 999);
    oracle.trace.clear();
    let no_batches = ImportOptions {
        max_batches: 0,
        ..Default::default()
    };
    let second = fork
        .run_with_oracle(&options(cg::BoundMode::Numerical), &no_batches, &mut oracle)
        .unwrap();
    assert_eq!(second.open_nodes, 1);
    assert_eq!(oracle.trace[0], ('s', 7, 1, Some(2)));
    assert_eq!(
        second.tree.pending[0].oracle_states[&7]
            .downcast_ref::<Snapshot>()
            .unwrap()
            .count,
        4
    );
    assert_eq!(
        fork.pending[0].oracle_states[&7]
            .downcast_ref::<Snapshot>()
            .unwrap()
            .count,
        2
    );
    assert_eq!(
        first.tree.pending[0].oracle_states[&7]
            .downcast_ref::<Snapshot>()
            .unwrap()
            .count,
        2
    );
    let third = second
        .tree
        .run_with_oracle(&options(cg::BoundMode::Numerical), &no_batches, &mut oracle)
        .unwrap();
    assert_eq!(
        third.tree.pending[0].oracle_states[&7]
            .downcast_ref::<Snapshot>()
            .unwrap()
            .count,
        6
    );
    assert_eq!(
        second.tree.pending[0].oracle_states[&7]
            .downcast_ref::<Snapshot>()
            .unwrap()
            .count,
        4
    );
}

#[test]
fn imported_oracle_setup_rejection_stops_later_groups_without_a_certified_infeasibility_claim() {
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let initial = session().begin_tree();
        let mut oracle = Oracle {
            reject: true,
            ..Default::default()
        };
        let result = initial
            .run_with_oracle(&options(mode), &ImportOptions::default(), &mut oracle)
            .unwrap();
        assert_eq!(oracle.trace, vec![('s', 7, 0, None)]);
        assert_eq!(result.batches, 0);
        assert_eq!(result.nodes, 1);
        if mode == cg::BoundMode::Certified {
            assert_eq!(result.status, TreeStatus::Incomplete);
            assert_eq!(result.open_nodes, 1);
            assert_eq!(result.bound, f64::NEG_INFINITY);
            assert!(result.tree.pending[0].oracle_states.is_empty());
        } else {
            assert_eq!(result.status, TreeStatus::Infeasible);
            assert_eq!(result.open_nodes, 0);
        }
        assert_eq!(initial.pending.len(), 1);
        assert!(initial.pending[0].oracle_states.is_empty());
    }
}

#[test]
fn imported_oracle_record_errors_preserve_frontier_and_previous_immutable_snapshots() {
    let mut oracle = Oracle::default();
    let result = session()
        .begin_tree()
        .run_with_oracle(
            &options(cg::BoundMode::Numerical),
            &ImportOptions::default(),
            &mut oracle,
        )
        .unwrap();
    oracle.fail_record = true;
    // Force an incomplete evaluation with no pricing batches, so state must
    // be captured for resume. Reject neither group during setup.
    let mut right = result.tree.clone();
    right.pending.retain(|child| {
        ImportOracleNode::new(
            child.session.definition.clone(),
            child.session.context.clone(),
            7,
            child.depth,
        )
        .unwrap()
        .original_bounds(0)
        .unwrap()
        .lower
            == 1.
    });
    let no_batches = ImportOptions {
        max_batches: 0,
        ..Default::default()
    };
    assert!(right
        .run_with_oracle(&options(cg::BoundMode::Numerical), &no_batches, &mut oracle)
        .is_err());
    assert_eq!(right.pending.len(), 1);
    assert_eq!(
        right.pending[0].oracle_states[&7]
            .downcast_ref::<Snapshot>()
            .unwrap()
            .count,
        2
    );
}

#[test]
fn imported_oracle_requests_and_finite_rebuild_preserve_child_depth_before_cg() {
    let child = session()
        .with_variable_bounds(
            0,
            crate::master::preprocessing::Bounds {
                lower: 1.,
                upper: 2.,
            },
        )
        .unwrap();
    let mut configured = Options::default();
    configured.cg.node_depth = 3;
    configured.cg.stabilization_max_tree_depth = 1;
    let view = child.view(7).unwrap();
    let feasibility = view.feasibility_pricing(configured.clone()).unwrap();
    assert_eq!(feasibility.node_depth(), 3);
    let objective = view.pricing(configured.clone()).unwrap();
    assert_eq!(objective.node_depth(), 3);
    assert_eq!(objective.for_subproblem(42).unwrap().node_depth(), 3);
    let (restricted, _) = view.objective_output(configured.clone()).unwrap();
    assert_eq!(restricted.snapshot.node_depth(), 3);
    let complete = child
        .collect(&ImportOptions::default(), |_, _| {
            Ok(ColumnBatch {
                columns: vec![],
                complete: true,
            })
        })
        .unwrap()
        .session;
    let finite = complete.solve_at_depth(configured, 3).unwrap();
    assert_eq!(finite.output.snapshot.node_depth(), 3);
}
