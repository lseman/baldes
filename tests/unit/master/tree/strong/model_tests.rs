use super::*;
use crate::master::{
    preprocessing::Bounds,
    tree::{StrongBranchingOptions, StrongPhase},
    RowSense,
};
use std::time::Duration;

impl History {
    fn record(&mut self, evaluation: MappedBranchEvaluation) {
        let key = evaluation.name.clone();
        self.record_with_key(&key, evaluation);
    }
}

fn root() -> Output<2> {
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem()
    .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    solve_with_master_variables(problem, vec![], options).unwrap()
}

fn mapping(name: &str, network: i32) -> MappedBranchVariable {
    MappedBranchVariable {
        name: name.into(),
        priority: 1.,
        networks: vec![MappedNetworkVariable {
            network,
            activities: vec![],
            arcs: vec![],
            per_route: 1.,
        }],
        masters: vec![],
    }
}

#[test]
fn native_mapped_direction_orders_trials_and_preserves_paired_child_bounds() {
    let output = root();
    for direction in [None, Some(NetworkBranchDirection::LowerFirst)] {
        let mut options = ModelTreeOptions::default();
        options.branching = NetworkBranchingPolicy::Phased;
        options.node.cg.stabilize = false;
        options.mapped_variables = vec![mapping("count42", 42)];
        if let Some(direction) = direction {
            options
                .branch_directions
                .insert("count42".into(), direction);
        }
        let mut history = History::default();
        let mut selected = choose(
            &output,
            &options,
            &mut history,
            2.,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap()
        .unwrap();
        let expected = direction
            .unwrap_or(NetworkBranchDirection::UpperFirst)
            .sides();
        assert_eq!(
            history
                .evaluations
                .iter()
                .map(|e| e.side)
                .collect::<Vec<_>>(),
            expected
        );
        // Canonical storage remains lower, upper even when evaluated upper first.
        for (side, bound) in [(0, 2.), (1, 5. / 3.)] {
            assert!((selected.children[side].warm.numerical_bound - bound).abs() < 1e-8);
            assert!(
                (selected.evaluated[side]
                    .as_ref()
                    .unwrap()
                    .master
                    .numerical_bound
                    - bound)
                    .abs()
                    < 1e-8
            );
        }
        selected.admit(NetworkBranchDirection::UpperFirst);
        let first_bound = if expected[0] == 1 { 5. / 3. } else { 2. };
        assert!((selected.children[0].warm.numerical_bound - first_bound).abs() < 1e-8);
        assert!(
            (selected.evaluated[0]
                .as_ref()
                .unwrap()
                .master
                .numerical_bound
                - first_bound)
                .abs()
                < 1e-8
        );
    }
}

#[test]
fn native_direction_without_trials_orders_the_original_disjunction() {
    let output = root();
    let mut options = ModelTreeOptions::default();
    options.branching = NetworkBranchingPolicy::Phased;
    options.mapped_strong_branching = None;
    options.mapped_variables = vec![mapping("count42", 42)];
    options
        .branch_directions
        .insert("count42".into(), NetworkBranchDirection::LowerFirst);
    let mut history = History::default();
    let mut selected = choose(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    selected.admit(NetworkBranchDirection::UpperFirst);
    assert_eq!(history.trials, 0);
    assert_eq!(
        selected.children[0].state.shared_rows.last().unwrap().sense,
        RowSense::LessEqual
    );
    assert_eq!(
        selected.children[1].state.shared_rows.last().unwrap().sense,
        RowSense::GreaterEqual
    );
}

#[test]
fn mapped_native_history_retains_signed_means_and_caps_predictions_by_gap() {
    let mut history = History::default();
    for (side, estimate) in [(0, 8.), (0, 12.), (1, 14.)] {
        history.record(MappedBranchEvaluation {
            name: "x".into(),
            depth: 0,
            side,
            stage: 2,
            kind: MappedEvaluationKind::RestrictedMaster,
            parent: 10.,
            estimate,
            distance: 0.5,
        });
    }
    // Side zero's signed gains cancel. Clipping negative observations before
    // averaging would instead produce a positive score.
    assert_eq!(history.prediction("x", 0.5, 2.), Some((2, 0.)));
    history.record(MappedBranchEvaluation {
        name: "x".into(),
        depth: 0,
        side: 0,
        stage: 3,
        kind: MappedEvaluationKind::HeuristicPricing,
        parent: 10.,
        estimate: 12.,
        distance: 0.5,
    });
    assert_eq!(history.prediction("x", 0.5, 2.), Some((3, 0.)));
}

#[test]
fn mapped_native_phase_skips_preserve_parent_and_reuse_exact_children() {
    let output = root();
    let before_rows = output.snapshot.state.shared_rows.len();
    let before_columns = output.snapshot.master.costs.len();
    let mut options = ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    options.mapped_variables = vec![mapping("network42", 42)];
    let mut history = History::default();
    let children = choose(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    // Both intermediate phases are skipped for one candidate. Exact phase
    // remains mandatory and is the first evaluation on each side.
    assert_eq!(history.trials, 2);
    assert!(!history.evaluations.is_empty());
    assert!(history
        .evaluations
        .iter()
        .all(|e| e.stage == 1 && e.kind == MappedEvaluationKind::ExactPricing));
    for child in children.children {
        assert!(child.warm.bound >= output.snapshot.warm.bound);
        assert_eq!(child.state.shared_rows.len(), before_rows + 1);
        assert!(child.master.costs.len() >= before_columns);
    }
    assert_eq!(output.snapshot.state.shared_rows.len(), before_rows);
    assert_eq!(output.snapshot.master.costs.len(), before_columns);
}

#[test]
fn strong_trials_retain_original_incumbents_and_only_completed_exact_bounds() {
    for restricted_master_only in [false, true] {
        let output = root();
        let mut options = ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.mapped_variables = vec![mapping("network42", 42)];
        if restricted_master_only {
            options.mapped_strong_branching.as_mut().unwrap().phases = vec![StrongPhase {
                candidates: 1,
                subtree_ratio: 0.,
                pricing_phase_floor: 0,
                max_iterations: 100,
                tree_depth_score: false,
                restricted_master_only: true,
            }];
        }
        let mut history = History::default();
        let children = choose(
            &output,
            &options,
            &mut history,
            f64::INFINITY,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            children.evaluated.iter().filter(|o| o.is_some()).count(),
            if restricted_master_only { 0 } else { 2 }
        );
        let (objective, routes, masters) = history.incumbent.as_ref().expect("trial incumbent");
        assert_eq!(*objective, 2.);
        assert_eq!(
            super::tree_adapter::validate_routes(&output.snapshot.state, routes, masters).unwrap(),
            Some(*objective)
        );
        assert!(output
            .snapshot
            .integer_witness(options.tolerance)
            .unwrap()
            .is_none());
        for child in children.children {
            assert!(child.warm.bound >= output.snapshot.warm.bound);
            assert!(child.warm.numerical_bound >= output.snapshot.warm.numerical_bound);
        }
    }
}

#[test]
fn mapped_native_subtree_budget_limits_actual_restricted_trials() {
    let mut output = root();
    output.snapshot.state.mapped_subtree_size = 2.;
    output.snapshot.state.mapped_subtree_priority = 1.;
    let mut options = ModelTreeOptions::default();
    let a = mapping("count42", 42);
    // The two distinct original mappings are fractional at the triangle root.
    let b =
        MappedBranchVariable::from_shared_row("assignment", 1., &output.snapshot, 0, &[42], &[])
            .unwrap();
    assert!(b.value(&output).unwrap() > 0. && b.value(&output).unwrap() < 1.);
    options.mapped_variables = vec![a, b];
    options.mapped_strong_branching = Some(StrongBranchingOptions {
        phases: vec![StrongPhase {
            candidates: 100,
            subtree_ratio: 0.3,
            pricing_phase_floor: 0,
            max_iterations: 1,
            tree_depth_score: false,
            restricted_master_only: true,
        }],
        ..Default::default()
    });
    let mut history = History::default();
    let children = choose(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    // Native permits the first candidate at evaluated=0, then exhausts a
    // small subtree quota; two sides are actual evaluations.
    assert_eq!(history.trials, 2);
    assert!(history
        .evaluations
        .iter()
        .all(|e| e.kind == MappedEvaluationKind::RestrictedMaster));
    assert!(children
        .children
        .iter()
        .all(|c| c.warm.bound == output.snapshot.warm.bound));
    assert!(exceeds(1., 0.));
    assert!(!exceeds(1., 1. + 1e-8));
    // Applying a registered bound remains independent of candidate history.
    assert!(options.mapped_variables[0]
        .apply(
            &output.snapshot,
            Bounds {
                lower: 0.,
                upper: 1.
            }
        )
        .is_ok());
}

#[test]
fn heuristic_and_incomplete_trials_never_prepare_child_proof_bounds() {
    for (pricing_phase_floor, max_iterations) in [(1, 100), (0, 1)] {
        let output = root();
        let mut options = ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.mapped_variables = vec![mapping("network42", 42)];
        options.mapped_strong_branching.as_mut().unwrap().phases = vec![StrongPhase {
            candidates: 1,
            subtree_ratio: 0.,
            pricing_phase_floor,
            max_iterations,
            tree_depth_score: false,
            restricted_master_only: false,
        }];
        let mut history = History::default();
        let selected = choose(
            &output,
            &options,
            &mut history,
            f64::INFINITY,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap()
        .unwrap();
        assert!(selected.evaluated.iter().all(Option::is_none));
        assert!(selected
            .children
            .iter()
            .all(|child| child.warm.bound == output.snapshot.warm.bound
                && child.warm.numerical_bound == output.snapshot.warm.numerical_bound));
    }
}

#[test]
fn mapped_scoring_uses_numerical_parent_without_changing_selected_proof_channel() {
    for (numerical, expected) in [(1.5, 1.5), (f64::NEG_INFINITY, -1e12)] {
        let mut output = root();
        assert_eq!(output.master.numerical_bound, 1.5);
        output.master.numerical_bound = numerical;
        output.snapshot.warm.numerical_bound = numerical;
        // A valid weaker directed certificate must not change native ranking
        // gains or signed history distances from the ordinary LP optimum.
        output.master.bound = 0.5;
        output.master.certified_bound = 0.5;
        output.master.bound_mode = cg::BoundMode::Certified;
        output.snapshot.warm.bound_mode = cg::BoundMode::Certified;
        output.snapshot.warm.bound = 0.5;
        output.snapshot.warm.certified_bound = 0.5;
        let mut options = ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.mapped_variables = vec![mapping("count42", 42)];
        let mut history = History::default();
        choose(
            &output,
            &options,
            &mut history,
            2.,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap()
        .unwrap();
        assert!(!history.evaluations.is_empty());
        assert!(history.evaluations.iter().all(|e| e.parent == expected));
        assert_eq!(output.master.bound, 0.5);
        assert_eq!(output.snapshot.warm.bound, 0.5);
        assert_eq!(output.snapshot.warm.certified_bound, 0.5);
    }
}

#[test]
fn trial_incumbent_is_published_after_the_scoring_pass() {
    let output = root();
    let mut options = ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    let mut total = mapping("total-routes", 7);
    total.networks.extend(mapping("other", 42).networks);
    options.mapped_variables = vec![total];
    let mut history = History::default();
    let selected = choose(
        &output,
        &options,
        &mut history,
        10.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    assert_eq!(history.incumbent.as_ref().unwrap().0, 2.);
    // Native retains the parent's entry incumbent during this pass. For
    // capped gains near 8.5 and 0.5, the leaf equation is approximately
    // x^-1 + x^(-1/17) = 1. Its root lies strictly between 2 and 50;
    // the resulting size 2*(x-1) is between 2 and 100. Prematurely
    // publishing the trial incumbent gives size 2 or about 262142,
    // depending on which normalization is incorrectly refreshed.
    for child in &selected.children {
        assert!(
            child.state.mapped_subtree_size > 2. && child.state.mapped_subtree_size < 100.,
            "size={}",
            child.state.mapped_subtree_size
        );
    }
    assert!(proven_infeasible(selected.evaluated[0].as_ref().unwrap()));
    assert!(selected.children[0].state.phase_one_infeasible);
    assert_eq!(output.snapshot.warm.bound, 1.5);
    assert_eq!(output.master.numerical_bound, 1.5);
}

#[test]
fn scoring_cutoff_margin_changes_subtree_prediction_without_changing_proof_bounds() {
    let output = root();
    let mut sizes = vec![];
    for margin in [0., 0.4] {
        let mut options = ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.scoring_gap_tolerance = margin;
        options.mapped_variables = vec![mapping("count42", 42)];
        let mut history = History::default();
        let selected = choose(
            &output,
            &options,
            &mut history,
            2.,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap()
        .unwrap();
        sizes.push(selected.children[0].state.mapped_subtree_size);
        assert_eq!(selected.children[0].warm.bound, 2.);
        assert!((selected.children[1].warm.bound - 5. / 3.).abs() < 1e-8);
        assert_eq!(output.snapshot.warm.bound, 1.5);
    }
    assert!(sizes[1] < sizes[0], "{sizes:?}");
}

#[test]
fn exact_infeasible_trials_retain_original_phase_one_proof_but_heuristics_do_not() {
    let output = root();
    for floor in [0, 1] {
        let mut options = ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.node.cg.max_penalty_updates = 0;
        let mut total = mapping("total", 7);
        total.networks.extend(mapping("other", 42).networks);
        options.mapped_variables = vec![total];
        options.mapped_strong_branching.as_mut().unwrap().phases = vec![StrongPhase {
            candidates: 1,
            subtree_ratio: 1.,
            pricing_phase_floor: floor,
            max_iterations: 100,
            tree_depth_score: true,
            restricted_master_only: false,
        }];
        let mut history = History::default();
        let selected = choose(
            &output,
            &options,
            &mut history,
            f64::INFINITY,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap()
        .unwrap();
        if floor == 0 {
            let lower = selected.evaluated[0]
                .as_ref()
                .expect("cached exact infeasibility");
            assert!(proven_infeasible(lower));
            assert!(lower.snapshot.state.phase_one_infeasible);
            let resumed = selected.children[0].solve(options.node.clone()).unwrap();
            assert!(proven_infeasible(&resumed));
        } else {
            assert!(selected.evaluated.iter().all(Option::is_none));
            assert_eq!(selected.children[0].warm.bound, output.snapshot.warm.bound);
        }
        assert_eq!(output.snapshot.warm.bound, 1.5);
        assert!(output.snapshot.integer_witness(1e-6).unwrap().is_none());
    }
}

fn compact_pricing_root() -> Output<2> {
    let mut output = root();
    let removed: Vec<_> = output
        .snapshot
        .warm
        .columns
        .iter()
        .enumerate()
        .filter_map(|(i, column)| {
            column
                .as_ref()
                .filter(|c| c.tag >= 0 && output.snapshot.master.values()[i].abs() < 1e-12)
                .map(|_| i)
        })
        .collect();
    assert!(!removed.is_empty());
    let (_, map) = output.snapshot.master.remove_cuts(&[], &removed).unwrap();
    let warm = &mut output.snapshot.warm;
    warm.columns = warm
        .columns
        .drain(..)
        .zip(&map)
        .filter_map(|(column, index)| index.map(|_| column))
        .collect();
    warm.artificials = warm.artificials.iter().filter_map(|&c| map[c]).collect();
    warm.cut_artificials = warm
        .cut_artificials
        .iter()
        .filter_map(|&(r, c)| Some((r, map[c]?)))
        .collect();
    // Removing loaded columns alone leaves the original generated pool able
    // to restore them before pricing. These tests exercise actual generation.
    warm.generated_pool.clear();
    // Two phases make the heuristic-to-exact restoration observable.
    for oracle in &mut output.snapshot.state.oracles {
        oracle.solver.phases.truncate(2);
    }
    output
}

fn continuation_options(exact_last: bool) -> ModelTreeOptions {
    let mut options = ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    options.node.cg.artificial_cost = 100.;
    options.mapped_variables = vec![mapping("screen-a", 42), mapping("screen-b", 42)];
    options.mapped_strong_branching.as_mut().unwrap().phases = vec![
        StrongPhase {
            candidates: 2,
            subtree_ratio: 1.,
            pricing_phase_floor: 1,
            max_iterations: 100,
            tree_depth_score: false,
            restricted_master_only: false,
        },
        StrongPhase {
            candidates: 1,
            subtree_ratio: 1.,
            pricing_phase_floor: 0,
            max_iterations: 100,
            tree_depth_score: exact_last,
            restricted_master_only: !exact_last,
        },
    ];
    options
}

#[test]
fn phase_survivor_retains_priced_columns_through_restricted_lp_and_commit_without_proof() {
    let output = compact_pricing_root();
    let parent_columns = output.snapshot.master.costs.len();
    let parent_rows = output.snapshot.state.shared_rows.len();
    let options = continuation_options(false);
    let mut history = History::default();
    let selected = choose(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    assert_eq!(history.trials, 6);
    assert_eq!(history.processed_trials, 0);
    let restricted: Vec<_> = history
        .evaluations
        .iter()
        .filter(|e| e.stage == 2 && e.kind == MappedEvaluationKind::RestrictedMaster)
        .collect();
    assert_eq!(restricted.len(), 2);
    for e in restricted {
        let expected = if e.side == 0 { 2. } else { 5. / 3. };
        assert!(
            (e.estimate - expected).abs() < 1e-8,
            "side {} LP {}",
            e.side,
            e.estimate
        );
    }
    assert!(selected.evaluated.iter().all(Option::is_none));
    for child in selected.children {
        assert!(child.master.costs.len() > parent_columns);
        assert_eq!(child.warm.bound, output.snapshot.warm.bound);
        assert!(child
            .state
            .oracles
            .iter()
            .all(|o| o.solver.phases.len() == 2));
    }
    assert_eq!(output.snapshot.master.costs.len(), parent_columns);
    assert_eq!(output.snapshot.state.shared_rows.len(), parent_rows);
}

#[test]
fn heuristic_survivor_restores_full_schedule_before_exact_phase_and_reuses_only_exact_proofs() {
    let output = compact_pricing_root();
    let parent_columns = output.snapshot.master.costs.len();
    let options = continuation_options(true);
    let mut history = History::default();
    let selected = choose(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    assert_eq!(history.trials, 6);
    assert_eq!(history.processed_trials, 2);
    assert!(history
        .evaluations
        .iter()
        .any(|e| e.stage == 1 && e.kind == MappedEvaluationKind::HeuristicPricing));
    assert_eq!(
        history
            .evaluations
            .iter()
            .filter(|e| e.stage == 2 && e.kind == MappedEvaluationKind::ExactPricing)
            .count(),
        2
    );
    for (side, evaluated) in selected.evaluated.iter().enumerate() {
        let evaluated = evaluated.as_ref().expect("completed exact survivor");
        assert_eq!(evaluated.master.status, cg::Status::Optimal);
        assert!(evaluated
            .snapshot
            .state
            .oracles
            .iter()
            .all(|o| o.solver.phases.len() == 2 && o.solver.phases[0].exact()));
        let expected = if side == 0 { 2. } else { 5. / 3. };
        assert!((evaluated.master.bound - expected).abs() < 1e-8);
    }
    assert_eq!(output.snapshot.master.costs.len(), parent_columns);
    assert!(output
        .snapshot
        .state
        .oracles
        .iter()
        .all(|o| o.solver.phases.len() == 2));
}

#[test]
fn native_restricted_screen_control_counts_exact_work_and_continues_lp_survivors() {
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let mut options = ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    options.branching = NetworkBranchingPolicy::ResourceFirst;
    options.mapped_variables = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/multi_network_native_phased_mappings.json"
    )))
    .unwrap();
    options.mapped_strong_branching.as_mut().unwrap().phases[1].candidates = 1;
    let result = crate::master::tree::solve(problem, options).unwrap();
    assert_eq!(result.status, crate::master::tree::TreeStatus::Optimal);
    assert_eq!(result.incumbent, 2.);
    assert_eq!(result.bound, 2.);
    assert_eq!(result.nodes, 1);
    assert_eq!(result.strong_branching_trials, 6);
    assert_eq!(result.processed_nodes, 3);
    assert_eq!(result.iterations, 10);
    // Nonnegative box admission skips physical pricing without changing
    // phase work. Native reports twenty accounting calls and eight physical
    // entries here; its remaining admission guards are a separate boundary.
    assert_eq!(
        result
            .oracle_calls
            .iter()
            .map(|(_, count)| count)
            .sum::<usize>(),
        2
    );
    assert_eq!(
        result
            .mapped_branch_evaluations
            .iter()
            .filter(|e| e.stage == 1 && e.kind == MappedEvaluationKind::RestrictedMaster)
            .count(),
        4
    );
}

#[test]
fn selected_heuristic_workspace_survives_final_phase_and_resumes_exact_pricing() {
    for bound_mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let mut output = compact_pricing_root();
        output.snapshot.warm.bound_mode = bound_mode;
        output.snapshot.warm.certified_bound = 0.5;
        output.snapshot.warm.bound = match bound_mode {
            cg::BoundMode::Numerical => output.snapshot.warm.numerical_bound,
            cg::BoundMode::Certified => 0.5,
        };
        let parent_columns = output.snapshot.master.costs.len();
        let mut options = continuation_options(false);
        options.node.cg.bound_mode = bound_mode;
        options
            .mapped_strong_branching
            .as_mut()
            .unwrap()
            .phases
            .truncate(1);
        let mut history = History::default();
        let selected = choose(
            &output,
            &options,
            &mut history,
            f64::INFINITY,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap()
        .unwrap();
        assert!(selected.evaluated.iter().all(Option::is_none));
        assert_eq!(history.processed_trials, 0);
        for (side, child) in selected.children.into_iter().enumerate() {
            assert!(child.master.costs.len() > parent_columns);
            assert_eq!(child.warm.bound, output.snapshot.warm.bound);
            assert_eq!(
                child.warm.numerical_bound,
                output.snapshot.warm.numerical_bound
            );
            assert_eq!(child.warm.certified_bound, 0.5);
            assert!(!child.state.phase_one_infeasible);
            assert!(child
                .state
                .oracles
                .iter()
                .all(|o| o.solver.phases.len() == 2 && o.solver.phases[0].exact()));
            // Fresh treatment reaches the original relaxation, independently
            // of the heuristic estimate or retained previous scan coverage.
            let resumed = child.solve(options.node.clone()).unwrap();
            assert_eq!(resumed.master.status, cg::Status::Optimal);
            let expected = if side == 0 { 2. } else { 5. / 3. };
            assert!((resumed.master.bound - expected).abs() < 1e-8);
        }
        assert_eq!(output.snapshot.master.costs.len(), parent_columns);
    }
}

#[test]
fn interrupted_exact_survivor_keeps_columns_without_publishing_a_completed_output() {
    let output = compact_pricing_root();
    let parent_columns = output.snapshot.master.costs.len();
    let mut options = continuation_options(true);
    options.mapped_strong_branching.as_mut().unwrap().phases[1].max_iterations = 1;
    let mut history = History::default();
    let selected = choose(
        &output,
        &options,
        &mut history,
        f64::INFINITY,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    assert_eq!(history.trials, 6);
    assert!(selected.evaluated.iter().all(Option::is_none));
    for (side, child) in selected.children.into_iter().enumerate() {
        assert!(child.master.costs.len() > parent_columns);
        assert_eq!(child.warm.bound, output.snapshot.warm.bound);
        assert_eq!(
            child.warm.numerical_bound,
            output.snapshot.warm.numerical_bound
        );
        assert!(child
            .state
            .oracles
            .iter()
            .all(|o| o.solver.phases.len() == 2 && o.solver.phases[0].exact()));
        let resumed = child.solve(options.node.clone()).unwrap();
        assert_eq!(resumed.master.status, cg::Status::Optimal);
        let expected = if side == 0 { 2. } else { 5. / 3. };
        assert!((resumed.master.bound - expected).abs() < 1e-8);
    }
    assert_eq!(output.snapshot.master.costs.len(), parent_columns);
}

#[test]
fn incomplete_exact_trial_keeps_independent_published_bounds_without_a_completed_output() {
    let mut output = compact_pricing_root();
    // Root phase-I recovery fixed its artificial columns at zero. Reopen the
    // finite-penalty extension so this test reaches ordinary exact pricing
    // rather than spending its single iteration restoring LP feasibility.
    for &column in &output.snapshot.warm.artificials {
        output.snapshot.master.set_upper(column, f64::MAX).unwrap();
    }
    // A valid weak numerical bound makes the first exact pricing pass's
    // improvement observable even while the restricted LP needs new columns.
    output.snapshot.warm.bound_mode = cg::BoundMode::Certified;
    output.snapshot.warm.numerical_bound = -1e12;
    output.snapshot.warm.certified_bound = f64::NEG_INFINITY;
    output.snapshot.warm.bound = f64::NEG_INFINITY;
    for oracle in &mut output.snapshot.state.oracles {
        oracle.solver.phases.truncate(1);
        assert!(oracle.solver.phases[0].exact());
    }
    let mut options = ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    options.node.cg.artificial_cost = 100.;
    options.node.cg.bound_mode = cg::BoundMode::Certified;
    options.mapped_variables = vec![mapping("count42", 42)];
    options.mapped_strong_branching.as_mut().unwrap().phases = vec![StrongPhase {
        candidates: 1,
        subtree_ratio: 0.,
        pricing_phase_floor: 0,
        max_iterations: 1,
        tree_depth_score: false,
        restricted_master_only: false,
    }];
    let parent_columns = output.snapshot.master.costs.len();
    let mut history = History::default();
    let selected = choose(
        &output,
        &options,
        &mut history,
        f64::INFINITY,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    assert!(
        selected.evaluated.iter().all(Option::is_none),
        "pricing adds missing columns; one iteration cannot finish these LPs"
    );
    for (side, child) in selected.children.iter().enumerate() {
        let expected = if side == 0 { 2. } else { 5. / 3. };
        assert!(child.master.costs.len() > parent_columns);
        assert!(child.warm.numerical_bound.is_finite());
        assert!(child.warm.numerical_bound > -1e12);
        // These original child optima are independently fixed by the triangle
        // cover equations, not by the trial's restricted LP objective.
        assert!(child.warm.numerical_bound <= expected + 1e-8);
        assert_eq!(child.warm.certified_bound, f64::NEG_INFINITY);
        assert_eq!(child.warm.bound, f64::NEG_INFINITY);
        let solved = child.solve(options.node.clone()).unwrap();
        assert_eq!(solved.master.status, cg::Status::Optimal);
        assert!((solved.master.bound - expected).abs() < 1e-8);
    }
    assert_eq!(output.snapshot.master.costs.len(), parent_columns);
    assert_eq!(output.snapshot.warm.numerical_bound, -1e12);
    assert_eq!(output.snapshot.warm.certified_bound, f64::NEG_INFINITY);
}

#[test]
fn admitted_zero_budget_trials_keep_unknown_estimates_and_inherited_proofs() {
    let output = root();
    let mut options = ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    options.mapped_variables = vec![mapping("count42", 42)];
    let strong = options.mapped_strong_branching.as_mut().unwrap();
    strong.trial_time_limit = Duration::ZERO;
    strong.phases = vec![StrongPhase {
        candidates: 1,
        subtree_ratio: 0.,
        pricing_phase_floor: 0,
        max_iterations: 100,
        tree_depth_score: true,
        restricted_master_only: false,
    }];
    let mut history = History::default();
    let selected = choose(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    assert_eq!(history.trials, 2);
    assert_eq!(history.processed_trials, 2);
    assert_eq!(history.iterations, 0);
    assert_eq!(history.pricing_calls, 0);
    assert!(history.oracle_calls.values().all(|&calls| calls == 0));
    assert!(history.evaluations.is_empty());
    assert!(history.incumbent.is_none());
    assert!(selected.evaluated.iter().all(Option::is_none));
    for child in selected.children {
        assert_eq!(child.warm.bound, output.snapshot.warm.bound);
        assert_eq!(
            child.warm.numerical_bound,
            output.snapshot.warm.numerical_bound
        );
        assert_eq!(
            child.warm.certified_bound,
            output.snapshot.warm.certified_bound
        );
        // Unknown LP estimates are native's high sentinel. With both estimated
        // leaves above cutoff, the predicted binary subtree has two leaves
        // and two descendant nodes. This prediction never closes either child.
        assert!((child.state.mapped_subtree_size - 2.).abs() < 1e-5);
    }
}

#[test]
fn admitted_count_candidate_finishes_sibling_attempt_after_local_deadline() {
    let output = root();
    let mut options = ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    // This fixture is also exercised by the retained read-only GDB observer,
    // which pauses after the first result until this local budget expires.
    options.mapped_variables = vec![mapping("count42", 42)];
    let strong = options.mapped_strong_branching.as_mut().unwrap();
    strong.time_limit = Duration::from_secs(3);
    strong.phases = vec![StrongPhase {
        candidates: 1,
        subtree_ratio: 0.,
        pricing_phase_floor: 0,
        max_iterations: 100,
        tree_depth_score: true,
        restricted_master_only: false,
    }];
    let proposals = vec![Proposal {
        name: "count42".into(),
        history_key: "count42".into(),
        priority: 1.,
        value: 0.5,
        restriction: Restriction::Multiplicity { network: 42 },
        history_eligible: false,
        fresh: true,
    }];
    let mut history = History::default();
    let selected = choose_proposals(
        &output,
        &options,
        &mut history,
        f64::INFINITY,
        Instant::now() + Duration::from_secs(30),
        proposals,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        history.trials, 2,
        "admission includes the sibling after expiry"
    );
    assert_eq!(history.processed_trials, 2);
    assert!(
        selected.evaluated[1].is_some(),
        "upper count side runs first"
    );
    assert!((selected.children[1].warm.bound - 5. / 3.).abs() < 1e-8);
    if selected.evaluated[0].is_some() {
        assert!((selected.children[0].warm.bound - 2.).abs() < 1e-8);
    } else {
        // A timed-out sibling is still attempted; its untouched inherited
        // bound remains independent of the completed first side's proof.
        assert_eq!(selected.children[0].warm.bound, output.snapshot.warm.bound);
    }
}
