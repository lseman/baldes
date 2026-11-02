use super::*;
use crate::master::tree::{StrongBranchingOptions, StrongPhase};
use std::time::Duration;

fn root(coefficient: f64) -> Output<2> {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (a, b, c) in [(0, 1, 1.), (1, 3, 0.), (0, 2, 5.), (2, 3, 0.)] {
        graph.add_arc(a, b, c).unwrap();
    }
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, coefficient)], vec![], vec![(0, 1.)], vec![]],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.preprocessing.enabled = false;
    options.gap_enumeration = false;
    solve(problem, options).unwrap()
}

fn options() -> ModelTreeOptions {
    let mut options = ModelTreeOptions::default();
    options.branching_registry.branching = NetworkBranchingPolicy::Phased;
    options.branching_registry.strong_branching = Some(StrongBranchingOptions {
        phases: vec![StrongPhase {
            candidates: 4,
            subtree_ratio: 0.,
            restricted_master_only: false,
            exact: Some(true),
            ..Default::default()
        }],
        ..Default::default()
    });
    options.branching_registry.component_generators = vec![ComponentClassGenerator {
        name: "classes".into(),
        network: 7,
        priority: 10.,
        coordinates: vec![ComponentGeneratorCoordinate {
            coordinate: ComponentCoordinate {
                id: 9,
                arc_values: vec![Some(1), None, None, None],
            },
            kind: ComponentVariableKind::Binary,
            priority: 1.,
            upper: 1.,
            current_upper: 1.,
        }],
        ordering_coordinates: vec![],
        priority_rule: ComponentPriorityRule::AddedDepth,
        minimum_fractionality: 0.02,
        integrality_tolerance: 1e-8,
    }];
    options
}

#[test]
fn evaluated_components_keep_trial_diagnostics_without_generic_history() {
    let initial = root(2.);
    let options = options();
    let proposals = registered_proposals(&initial, &options, None).unwrap();
    assert!(!proposals.is_empty());
    assert!(proposals.iter().all(|p| !p.history_eligible));
    let mut history = History::default();
    let selected = choose_proposals(
        &initial,
        &options,
        &mut history,
        10.,
        Instant::now() + Duration::from_secs(10),
        proposals,
    )
    .unwrap()
    .unwrap();
    assert!(history.trials > 0);
    assert!(!history.evaluations.is_empty());
    assert!(history.templates.is_empty());
    assert!(history.costs.is_empty());
    assert_eq!(history.ranked, 0);
    assert!(selected
        .children
        .iter()
        .all(|c| c.state.branch_history_origin.is_none()));
    assert!(selected
        .children
        .iter()
        .all(|c| !c.state.component_branches.is_empty()));

    // A valid strengthening changes the same model's LP class activity to
    // .99. Fresh separation omits it; generic history must not resurrect it.
    let child = initial
        .snapshot
        .with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.01,
            },
            &[RowCoefficients {
                network: 7,
                arcs: vec![1., 0., 0., 0.],
                per_route: 0.,
            }],
        )
        .unwrap();
    let mut node_options = Options::default();
    node_options.cg.stabilize = false;
    node_options.preprocessing.enabled = false;
    node_options.gap_enumeration = false;
    node_options.reduced_cost_fixing = false;
    let mut current = child.solve(node_options).unwrap();
    current.snapshot.state.node_depth = 1;
    assert!((current.values.iter().sum::<f64>() - 0.99).abs() < 1e-9);
    let fresh = registered_proposals(&current, &options, None).unwrap();
    assert!(fresh.is_empty());
    assert!(choose_proposals(
        &current,
        &options,
        &mut history,
        10.,
        Instant::now() + Duration::from_secs(10),
        fresh,
    )
    .unwrap()
    .is_none());
    assert_eq!(history.ranked, 0);
}

#[test]
fn skipped_component_screen_does_not_publish_committed_history_origin() {
    let initial = root(2.);
    let mut options = options();
    let phase = &mut options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .phases[0];
    phase.restricted_master_only = true;
    phase.exact = Some(false);
    let proposals = registered_proposals(&initial, &options, None).unwrap();
    assert_eq!(proposals.len(), 1);
    let mut history = History::default();
    let selection = choose_proposals(
        &initial,
        &options,
        &mut history,
        10.,
        Instant::now() + Duration::from_secs(10),
        proposals,
    )
    .unwrap()
    .unwrap();
    assert_eq!(history.trials, 0);
    assert!(history.templates.is_empty());
    assert!(selection
        .children
        .iter()
        .all(|c| c.state.branch_history_origin.is_none()));
    let mut committed = selection.children[1].solve(options.node.clone()).unwrap();
    assert_eq!(committed.master.status, cg::Status::Optimal);
    history
        .record_committed(&mut committed, 10., false, options.tolerance)
        .unwrap();
    assert!(history.costs.is_empty());
    assert!(history.evaluations.is_empty());
    assert!(!committed.snapshot.state.component_branches.is_empty());
}
