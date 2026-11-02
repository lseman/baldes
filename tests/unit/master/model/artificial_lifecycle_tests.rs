use super::*;

fn point() -> (NodeSnapshot<2>, Options) {
    let problem = crate::master::model::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let mut options = Options::default();
    options.cg.artificial_cost = 1.2;
    options.cg.stabilize = false;
    options.gap_enumeration = false;
    options.reduced_cost_fixing = false;
    options.enumeration = None;
    let output = crate::master::model::solve(problem, options.clone()).unwrap();
    assert!(!output.snapshot.warm.artificials.is_empty());
    (output.snapshot, options)
}

fn raise(node: &mut NodeSnapshot<2>) {
    for &column in &node.warm.artificials {
        node.master.set_cost(column, 4.8).unwrap();
    }
}

#[test]
fn evaluation_resets_ordinary_penalties_from_initial_cost_without_compounding() {
    let (mut node, _) = point();
    raise(&mut node);
    assert_eq!(node.state.ordinary_artificial_cost, 1.2);
    let inherited = node.fork().unwrap();
    for &column in &inherited.warm.artificials {
        assert_eq!(inherited.master.costs[column], 4.8);
    }
    node.reset_evaluation_artificials().unwrap();
    node.reset_evaluation_artificials().unwrap();
    for &column in &node.warm.artificials {
        assert_eq!(node.master.costs[column], 1.2);
    }
}

#[test]
fn new_branch_rows_use_initial_effective_cost_not_current_penalty_or_floor() {
    let (mut parent, _) = point();
    raise(&mut parent);
    let oracle = &parent.state.oracles[0];
    let coefficients = RowCoefficients {
        network: oracle.network.id,
        arcs: vec![1.; oracle.network.graph.arcs.len()],
        per_route: 0.,
    };
    let count = parent.warm.artificials.len();
    let mut child = parent
        .with_row(
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.,
            },
            &[coefficients],
        )
        .unwrap();
    assert!(child.warm.artificials.len() > count);
    for &column in &child.warm.artificials[count..] {
        assert_eq!(child.master.costs[column], 1.2);
    }
    child.reset_evaluation_artificials().unwrap();
    for &column in &child.warm.artificials {
        assert_eq!(child.master.costs[column], 1.2);
    }
    for &column in &parent.warm.artificials {
        assert_eq!(parent.master.costs[column], 4.8);
    }
}

#[test]
fn separated_row_artificials_keep_their_distinct_cost_policy() {
    let (mut node, _) = point();
    raise(&mut node);
    let entries = [(0, 1.)];
    let column = node
        .master
        .add_columns(&[(7.2, entries.as_slice())])
        .unwrap()
        .next()
        .unwrap();
    node.warm.columns.push(None);
    node.warm.artificials.push(column);
    node.warm.cut_artificials.push((0, column));
    node.reset_evaluation_artificials().unwrap();
    assert_eq!(node.master.costs[column], 7.2);
    // Missing creation metadata deliberately leaves this separated column unchanged.
}

#[test]
fn reset_preserves_only_the_selected_original_proof_as_a_tree_floor() {
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let (mut node, _) = point();
        raise(&mut node);
        node.warm.bound_mode = mode;
        node.warm.bound = 0.25;
        node.warm.numerical_bound = 0.5;
        node.warm.certified_bound = 0.25;
        node.reset_evaluation_artificials().unwrap();
        assert_eq!(node.state.tree_bound_floor, 0.25);
        assert_eq!(node.warm.bound_mode, mode);
        assert_eq!(node.warm.bound, f64::NEG_INFINITY);
        assert_eq!(node.warm.numerical_bound, f64::NEG_INFINITY);
        assert_eq!(node.warm.certified_bound, f64::NEG_INFINITY);
    }
}

#[test]
fn same_node_and_raw_column_generation_resumes_keep_current_penalties() {
    let (mut node, options) = point();
    raise(&mut node);
    let ids = node.warm.artificials.clone();
    let resumed = node.solve(options).unwrap();
    for &column in &resumed.snapshot.warm.artificials {
        assert_eq!(resumed.snapshot.master.costs[column], 4.8);
    }
    let mut raw = node.master.try_clone().unwrap();
    let mut controller = cg::NoController;
    let options = cg::Options {
        artificial_cost: 1.2,
        stabilize: false,
        warm_start: Some(node.warm.clone().into_owned()),
        ..Default::default()
    };
    let phases = [crate::pricing::phases::Phase {
        check_ng: true,
        jump_arcs: true,
        labels_per_bucket: i64::MAX,
    }];
    cg::optimize(
        &mut raw,
        &phases,
        |_, _| {
            Ok(cg::Answer {
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
    for column in ids {
        assert_eq!(raw.costs[column], 4.8);
    }
}

fn controller_for_cut_test(state: Rc<RefCell<State<2>>>) -> ColumnGenerationController<2> {
    ColumnGenerationController {
        state,
        objective_cutoff: f64::INFINITY,
        request_initial_enumeration: false,
        search_inactive_columns: false,
        artificial_penalty: None,
        enumeration_options: None,
        enumeration_checkpoint: None,
        removed_columns: vec![],
        trust_preprocessing_infeasibility: false,
        allow_objective_fixing: false,
        checkpoints: CheckpointStore::default(),
    }
}

#[test]
fn separated_artificials_reset_from_individual_creation_bases_without_compounding() {
    use crate::master::cg::Controller;
    let (mut node, _) = point();
    let mut ids = vec![];
    for base in [2., 5.] {
        let entries = [(0, 1.)];
        let column = node
            .master
            .add_columns(&[(base, entries.as_slice())])
            .unwrap()
            .next()
            .unwrap();
        node.warm.columns.push(None);
        node.warm.artificials.push(column);
        node.warm.cut_artificials.push((0, column));
        ids.push(column);
    }
    let state = Rc::new(RefCell::new(node.state));
    let mut controller = controller_for_cut_test(state.clone());
    for (&column, base) in ids.iter().zip([2., 5.]) {
        controller.cut_artificial_added(0, column, base).unwrap();
    }
    drop(controller);
    node.state = Rc::try_unwrap(state).ok().unwrap().into_inner();
    let parent = node.fork().unwrap();
    for &column in &ids {
        node.master.set_cost(column, 99.).unwrap();
    }
    node.reset_evaluation_artificials().unwrap();
    node.reset_evaluation_artificials().unwrap();
    for ((&column, expected), base) in ids.iter().zip([2.4, 6.]).zip([2., 5.]) {
        assert_eq!(node.master.costs[column], expected);
        assert_eq!(parent.master.costs[column], base);
    }
}

#[test]
fn cut_creation_metadata_follows_real_column_compaction_and_reused_slots() {
    use crate::master::cg::Controller;
    let (mut node, _) = point();
    let entries = [(0, 1.)];
    let ids: Vec<_> = node
        .master
        .add_columns(&[(2., entries.as_slice()), (7., entries.as_slice())])
        .unwrap()
        .collect();
    let state = Rc::new(RefCell::new(node.state));
    let mut controller = controller_for_cut_test(state.clone());
    controller.cut_artificial_added(0, ids[0], 2.).unwrap();
    controller.cut_artificial_added(1, ids[1], 7.).unwrap();
    let (_, map) = node.master.remove_cuts(&[], &[ids[0]]).unwrap();
    controller.remap_columns(&map).unwrap();
    let survivor = map[ids[1]].unwrap();
    assert_eq!(
        state.borrow().cut_artificial_bases,
        HashMap::from([(survivor, 7.)])
    );
    let reused = node
        .master
        .add_columns(&[(11., entries.as_slice())])
        .unwrap()
        .next()
        .unwrap();
    assert_eq!(reused, ids[1]);
    assert!(!state.borrow().cut_artificial_bases.contains_key(&reused));
    controller.cut_artificial_added(2, reused, 11.).unwrap();
    let cloned = state.borrow().fork_state();
    let (_, map) = node.master.remove_cuts(&[], &[survivor, reused]).unwrap();
    controller.remap_columns(&map).unwrap();
    assert!(state.borrow().cut_artificial_bases.is_empty());
    assert_eq!(
        cloned.cut_artificial_bases,
        HashMap::from([(survivor, 7.), (reused, 11.)])
    );
    for bad in [f64::NAN, f64::INFINITY, f64::MAX, -1.] {
        assert!(controller.cut_artificial_added(0, reused, bad).is_err());
        assert!(state.borrow().cut_artificial_bases.is_empty());
    }
    let mut generic = cg::NoController;
    <cg::NoController as Controller<ClpMaster>>::cut_artificial_added(&mut generic, 0, 3, f64::MAX)
        .unwrap();
}

#[test]
fn generated_rank_one_cut_artificials_retain_real_creation_costs() {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/unit/master/model/generated_cut_seven.json"
    ));
    let problem = format::Model::from_json(text)
        .unwrap()
        .problem::<2>()
        .unwrap();
    let mut options = Options::default();
    options.cg.artificial_cost = 1200.;
    options.cg.stabilize = false;
    options.separation.rank_one.enabled = true;
    options.enumeration = None;
    options.gap_enumeration = false;
    options.reduced_cost_fixing = false;
    let mut node = solve(problem, options).unwrap().snapshot;
    assert!(!node.warm.cut_artificials.is_empty());
    for &(_, column) in &node.warm.cut_artificials {
        assert!(node.warm.artificials.contains(&column));
        assert_eq!(node.state.cut_artificial_bases[&column], 1000.);
        node.master.set_cost(column, 4800.).unwrap();
    }
    node.reset_evaluation_artificials().unwrap();
    for &(_, column) in &node.warm.cut_artificials {
        assert_eq!(node.master.costs[column], 1200.);
    }
}

#[test]
fn model_defaults_match_observed_unconfigured_artificial_setup_cost() {
    let observation: serde_json::Value =
        serde_json::from_str(include_str!("artificial_default_fixture.json")).unwrap();
    let expected = observation["effective"].as_f64().unwrap();
    assert_eq!(
        (1. + observation["uplift"].as_f64().unwrap()) * observation["base"].as_f64().unwrap(),
        expected
    );
    assert_eq!(Options::default().cg.artificial_cost, expected);
    let mut frontend = format::SolveOptions::default();
    assert_eq!(frontend.artificial_cost, expected);
    assert_eq!(
        frontend.node_options().unwrap().cg.artificial_cost,
        expected
    );
    assert_eq!(cg::Options::default().artificial_cost, 0.);
    frontend.artificial_cost = 12.5;
    assert_eq!(frontend.node_options().unwrap().cg.artificial_cost, 12.5);
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let output = solve(problem, Options::default()).unwrap();
    assert_eq!(output.snapshot.state.ordinary_artificial_cost, expected);
    assert!(!output.snapshot.warm.artificials.is_empty());
    for &column in &output.snapshot.warm.artificials {
        assert_eq!(output.snapshot.master.costs[column], expected);
    }
}
