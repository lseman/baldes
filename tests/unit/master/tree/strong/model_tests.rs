use super::*;
use crate::master::LpMaster;

/// The production selection over the registered generators.
fn choose_registered<const D: usize>(
    output: &Output<D>,
    options: &ModelTreeOptions,
    history: &mut History,
    incumbent: f64,
    deadline: Instant,
) -> Result<Option<Selection<D>>> {
    let proposals = registered_proposals(output, options, None)?;
    choose_proposals(output, options, history, incumbent, deadline, proposals)
}
use crate::master::{
    preprocessing::Bounds,
    tree::{StrongBranchingOptions, StrongPhase},
    RowSense,
};
use std::time::Duration;

#[test]
fn multiway_selection_permutation_keeps_evaluated_outputs_with_children() {
    let mut children = vec![];
    let mut evaluated = vec![];
    for side in 0..3 {
        let mut output = root();
        output.snapshot.state.mapped_lineage = side;
        children.push(output.snapshot.fork().unwrap());
        evaluated.push(Some(output));
    }
    let mut selection = Selection::fresh(children);
    assert_eq!(selection.evaluated.len(), 3);
    selection.evaluated = evaluated;
    selection.order = Some(vec![2, 0, 1]);
    selection.admit(NetworkBranchDirection::UpperFirst);
    for (position, expected) in [2, 0, 1].into_iter().enumerate() {
        assert_eq!(selection.children[position].state.mapped_lineage, expected);
        assert_eq!(
            selection.evaluated[position]
                .as_ref()
                .unwrap()
                .snapshot
                .state
                .mapped_lineage,
            expected
        );
    }
}

#[test]
fn history_zero_scores_use_name_ties_for_both_zero_signs() {
    for negative_name_first in [false, true] {
        let mut output = root();
        output.snapshot.state.node_depth = 1;
        let mut options = strong_tree_options();
        options.branching_registry.use_branching_history = true;
        options.branching_registry.mapped_variables =
            vec![mapping("z-zero", 42), mapping("a-negative", 42)];
        if negative_name_first {
            options.branching_registry.mapped_variables.reverse();
        }
        options
            .branching_registry
            .strong_branching
            .as_mut()
            .unwrap()
            .phases = vec![StrongPhase {
            candidates: 2,
            subtree_ratio: 0.,
            restricted_master_only: true,
            tree_depth_score: false,
            max_cut_rounds: 0,
            reduced_cost_fixing: false,
            ..StrongPhase::default()
        }];
        let mut history = History::default();
        for (name, parent) in [("a-negative", 2.), ("z-zero", 1.5)] {
            history.record_with_key(
                &format!("mapped:{name}"),
                MappedBranchEvaluation {
                    name: name.into(),
                    depth: 0,
                    side: 0,
                    stage: 1,
                    kind: MappedEvaluationKind::ExactPricing,
                    parent,
                    estimate: 1.5,
                    distance: 0.5,
                },
                2.,
            );
        }
        assert_eq!(
            history
                .prediction("mapped:a-negative", 1.5, 0.5)
                .unwrap()
                .1
                .to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(
            history
                .prediction("mapped:z-zero", 1.5, 0.5)
                .unwrap()
                .1
                .to_bits(),
            0_f64.to_bits()
        );
        let selected = choose_registered(
            &output,
            &options,
            &mut history,
            2.,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap()
        .unwrap();
        assert_eq!(history.ranked, 1);
        assert_eq!(selected.children.len(), 2);
        for child in &selected.children {
            // Family allocation may leave only the history candidate, so the
            // nonexact screen can be skipped. Check the committed identity
            // rather than requiring a trial solely to observe its name.
            assert_eq!(
                child.state.branch_history_origin.as_ref().unwrap().name,
                "a-negative"
            );
        }
    }
}

impl History {
    fn record(&mut self, evaluation: MappedBranchEvaluation) {
        let key = evaluation.name.clone();
        self.record_with_key(&key, evaluation, f64::INFINITY);
    }
}

#[test]
fn signed_and_small_history_distances_match_original_finite_recording() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("history_distance_fixtures.json")).unwrap();
    let mut checked = 0;
    for case in fixtures["cases"].as_array().unwrap() {
        let Some(lower_mean) = case["means"][0].as_f64() else {
            continue;
        };
        let Some(upper_mean) = case["means"][1].as_f64() else {
            continue;
        };
        let fraction = case["fraction"].as_f64().unwrap();
        let mut history = History::default();
        for (side, estimate, distance) in [(0, 12., fraction), (1, 14., 1. - fraction)] {
            history.record_with_key(
                "aggregate",
                MappedBranchEvaluation {
                    name: "aggregate".into(),
                    depth: 1,
                    side,
                    stage: 1,
                    kind: MappedEvaluationKind::RestrictedMaster,
                    parent: 10.,
                    estimate,
                    distance,
                },
                50.,
            );
        }
        let stats = &history.costs["aggregate"][&1];
        for (side, expected) in [lower_mean, upper_mean].into_iter().enumerate() {
            assert_eq!(stats[side].count, 1, "fraction {fraction}, side {side}");
            assert_eq!(
                stats[side].mean.to_bits(),
                expected.to_bits(),
                "fraction {fraction}, side {side}"
            );
        }
        assert_eq!(history.evaluations.len(), 2);
        checked += 1;
    }
    // Zero-denominator IEEE observations remain a separate audit boundary.
    assert_eq!(checked, 11);
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
fn mapped_near_ties_keep_registration_order_without_moving_other_families() {
    let variables = vec![mapping("z-first", 42), mapping("a-second", 42)];
    let mapped = |variable: &MappedBranchVariable, value| Proposal {
        name: variable.name.clone(),
        history_key: variable.name.clone(),
        priority: 1.,
        value,
        restriction: Restriction::Mapped(variable.clone()),
        history_eligible: true,
        fresh: true,
        origin: None,
        component_admission: None,
    };
    let count = Proposal {
        name: "multiplicity".into(),
        history_key: "multiplicity".into(),
        priority: 1.,
        value: 0.5,
        restriction: Restriction::Multiplicity { network: 42 },
        history_eligible: false,
        fresh: true,
        origin: None,
        component_admission: None,
    };
    // This is the old exact/name sort: a is marginally closer to one half.
    let mut proposals = vec![
        mapped(&variables[1], 0.5),
        count,
        mapped(&variables[0], 0.4999995),
    ];
    let mut order = vec![0, 1, 2];
    rank_mapped_slots(
        &proposals,
        &ModelTreeOptions {
            branching_registry: {
                let mut __branch_registry = crate::master::model::BranchingRegistry::default();
                __branch_registry.mapped_variables = variables.clone();
                __branch_registry
            },
            ..Default::default()
        },
        &mut order,
    );
    assert_eq!(order, [2, 1, 0]);
    // An actual difference outside the tolerance band wins over registration.
    proposals[2].value = 0.49;
    let mut order = vec![0, 1, 2];
    rank_mapped_slots(
        &proposals,
        &ModelTreeOptions {
            branching_registry: {
                let mut __branch_registry = crate::master::model::BranchingRegistry::default();
                __branch_registry.mapped_variables = variables.clone();
                __branch_registry
            },
            ..Default::default()
        },
        &mut order,
    );
    assert_eq!(order, [0, 1, 2]);
}

#[test]
fn final_nonexact_screen_skips_the_single_admitted_candidate() {
    let output = root();
    let mut options = strong_tree_options();
    options.branching_registry.mapped_variables =
        vec![mapping("z-first", 42), mapping("a-second", 42)];
    options.branching_registry.strong_branching = Some(StrongBranchingOptions {
        phases: vec![StrongPhase {
            candidates: 1,
            subtree_ratio: 0.,
            pricing_phase_floor: 0,
            max_iterations: 1,
            tree_depth_score: false,
            restricted_master_only: true,
            ..StrongPhase::default()
        }],
        ..Default::default()
    });
    let mut history = History::default();
    choose_registered(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    assert_eq!(history.trials, 0);
    assert!(history.evaluations.is_empty());
}

#[test]
fn native_mapped_direction_orders_trials_and_preserves_paired_child_bounds() {
    let output = root();
    for direction in [None, Some(NetworkBranchDirection::LowerFirst)] {
        let mut options = strong_tree_options();
        options.branching_registry.branching = NetworkBranchingPolicy::Phased;
        options.node.cg.stabilize = false;
        options.branching_registry.mapped_variables = vec![mapping("count42", 42)];
        if let Some(direction) = direction {
            options
                .branching_registry
                .branch_directions
                .insert("count42".into(), direction);
        }
        let mut history = History::default();
        let mut selected = choose_registered(
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
    let mut options = strong_tree_options();
    options.branching_registry.branching = NetworkBranchingPolicy::Phased;
    options.branching_registry.strong_branching = None;
    options.branching_registry.mapped_variables = vec![mapping("count42", 42)];
    options
        .branching_registry
        .branch_directions
        .insert("count42".into(), NetworkBranchDirection::LowerFirst);
    let mut history = History::default();
    let mut selected = choose_registered(
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
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    options.branching_registry.mapped_variables = vec![mapping("network42", 42)];
    let mut history = History::default();
    let children = choose_registered(
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
        let mut options = strong_tree_options();
        options.node.cg.stabilize = false;
        options.branching_registry.mapped_variables =
            vec![mapping("network42", 42), mapping("other42", 42)];
        if restricted_master_only {
            options
                .branching_registry
                .strong_branching
                .as_mut()
                .unwrap()
                .phases = vec![StrongPhase {
                candidates: 2,
                subtree_ratio: 0.,
                pricing_phase_floor: 0,
                max_iterations: 100,
                tree_depth_score: false,
                restricted_master_only: true,
                ..StrongPhase::default()
            }];
        }
        let mut history = History::default();
        let children = choose_registered(
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
            super::tree_adapter::validate_routes(
                &output.snapshot.state,
                routes,
                &mut masters.clone()
            )
            .unwrap(),
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
fn subtree_budget_reduces_admission_to_one_and_skips_the_final_nonexact_screen() {
    let mut output = root();
    output.snapshot.state.mapped_subtree_size = 2.;
    output.snapshot.state.mapped_subtree_priority = 1.;
    let mut options = strong_tree_options();
    let a = mapping("count42", 42);
    // The two distinct original mappings are fractional at the triangle root.
    let b =
        MappedBranchVariable::from_shared_row("assignment", 1., &output.snapshot, 0, &[42], &[])
            .unwrap();
    assert!(b.value(&output).unwrap() > 0. && b.value(&output).unwrap() < 1.);
    options.branching_registry.mapped_variables = vec![a, b];
    options.branching_registry.strong_branching = Some(StrongBranchingOptions {
        phases: vec![StrongPhase {
            candidates: 100,
            subtree_ratio: 0.3,
            pricing_phase_floor: 0,
            max_iterations: 1,
            tree_depth_score: false,
            restricted_master_only: true,
            ..StrongPhase::default()
        }],
        ..Default::default()
    });
    let mut history = History::default();
    let children = choose_registered(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    // The subtree budget admits ceil(0.3 * 2) = one candidate.
    // A final nonexact screen cannot choose between alternatives, so it skips.
    assert_eq!(history.trials, 0);
    assert!(history.evaluations.is_empty());
    assert!(children.children.iter().all(|child| child
        .state
        .branch_history_origin
        .as_ref()
        .unwrap()
        .stage
        == 1));
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
    assert!(options.branching_registry.mapped_variables[0]
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
        let mut options = strong_tree_options();
        options.node.cg.stabilize = false;
        options.branching_registry.mapped_variables = vec![mapping("network42", 42)];
        options
            .branching_registry
            .strong_branching
            .as_mut()
            .unwrap()
            .phases = vec![StrongPhase {
            candidates: 1,
            subtree_ratio: 0.,
            pricing_phase_floor,
            max_iterations,
            tree_depth_score: false,
            restricted_master_only: false,
            ..StrongPhase::default()
        }];
        let mut history = History::default();
        let selected = choose_registered(
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
        let mut options = strong_tree_options();
        options.node.cg.stabilize = false;
        options.branching_registry.mapped_variables = vec![mapping("count42", 42)];
        let mut history = History::default();
        choose_registered(
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
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    let mut total = mapping("total-routes", 7);
    total.networks.extend(mapping("other", 42).networks);
    options.branching_registry.mapped_variables = vec![total];
    let mut history = History::default();
    let selected = choose_registered(
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
    // The entry incumbent lets exact pricing conquer this child before a
    // separate phase-I proof or the later trial incumbent is available.
    let conquered = selected.evaluated[0].as_ref().unwrap();
    assert!(conquered.master.pruned_by_bound);
    assert_eq!(conquered.master.bound, 10.);
    assert!(!proven_infeasible(conquered));
    assert!(!selected.children[0].state.phase_one_infeasible);
    assert_eq!(output.snapshot.warm.bound, 1.5);
    assert_eq!(output.master.numerical_bound, 1.5);
}

#[test]
fn scoring_cutoff_margin_changes_subtree_prediction_without_changing_proof_bounds() {
    let output = root();
    let mut sizes = vec![];
    for margin in [0., 0.4] {
        let mut options = strong_tree_options();
        options.node.cg.stabilize = false;
        options.scoring_gap_tolerance = margin;
        options.branching_registry.mapped_variables = vec![mapping("count42", 42)];
        let mut history = History::default();
        let selected = choose_registered(
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
        let mut options = strong_tree_options();
        options.node.cg.stabilize = false;
        options.node.cg.max_penalty_updates = 0;
        let mut total = mapping("total", 7);
        total.networks.extend(mapping("other", 42).networks);
        options.branching_registry.mapped_variables = vec![total];
        options
            .branching_registry
            .strong_branching
            .as_mut()
            .unwrap()
            .phases = vec![StrongPhase {
            candidates: 1,
            subtree_ratio: 1.,
            pricing_phase_floor: floor,
            max_iterations: 100,
            tree_depth_score: true,
            restricted_master_only: false,
            ..StrongPhase::default()
        }];
        let mut history = History::default();
        let selected = choose_registered(
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
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    options.node.cg.artificial_cost = 100.;
    options.branching_registry.mapped_variables =
        vec![mapping("screen-a", 42), mapping("screen-b", 42)];
    options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .phases = vec![
        StrongPhase {
            candidates: 2,
            subtree_ratio: 1.,
            pricing_phase_floor: 1,
            max_iterations: 100,
            tree_depth_score: false,
            restricted_master_only: false,
            ..StrongPhase::default()
        },
        StrongPhase {
            candidates: 1,
            subtree_ratio: 1.,
            pricing_phase_floor: 0,
            max_iterations: 100,
            tree_depth_score: exact_last,
            restricted_master_only: !exact_last,
            ..StrongPhase::default()
        },
    ];
    options
}

#[test]
fn heuristic_survivor_skips_redundant_final_restricted_screen_and_keeps_workspace() {
    let output = compact_pricing_root();
    let parent_columns = output.snapshot.master.costs.len();
    let parent_rows = output.snapshot.state.shared_rows.len();
    let options = continuation_options(false);
    let mut history = History::default();
    let selected = choose_registered(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    // Both heuristic groups run; equal score pairs collapse to one survivor.
    // The final restricted screen then skips its implicit next quota of one.
    assert_eq!(history.trials, 4);
    assert_eq!(history.processed_trials, 0);
    assert!(history
        .evaluations
        .iter()
        .all(|evaluation| evaluation.stage == 1
            && evaluation.kind == MappedEvaluationKind::HeuristicPricing));
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
    let selected = choose_registered(
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
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    options.branching_registry.branching = NetworkBranchingPolicy::ResourceFirst;
    options.branching_registry.mapped_variables = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/multi_network_native_phased_mappings.json"
    )))
    .unwrap();
    options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .phases[1]
        .candidates = 1;
    let result = crate::master::tree::solve(problem, options).unwrap();
    assert_eq!(result.status, crate::master::tree::TreeStatus::Optimal);
    assert_eq!(result.incumbent, 2.);
    assert_eq!(result.bound, 2.);
    assert_eq!(result.nodes, 1);
    assert_eq!(result.strong_branching_trials, 6);
    assert_eq!(result.processed_nodes, 3);
    assert_eq!(result.iterations, 10);
    // This whole-incoming assignment mapping is separate from the pinned
    // half-incident native control. Retiring a redundant child row shifts
    // dual prices onto proved-zero arcs; conservative all-arc admission adds
    // three physical calls without changing phase work or proof bounds.
    assert_eq!(
        result
            .oracle_calls
            .iter()
            .map(|(_, count)| count)
            .sum::<usize>(),
        5
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
            .branching_registry
            .strong_branching
            .as_mut()
            .unwrap()
            .phases
            .truncate(1);
        let mut history = History::default();
        let selected = choose_registered(
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
    options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .phases[1]
        .max_iterations = 1;
    let mut history = History::default();
    let selected = choose_registered(
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
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    options.node.cg.artificial_cost = 100.;
    options.node.cg.bound_mode = cg::BoundMode::Certified;
    options.branching_registry.mapped_variables = vec![mapping("count42", 42)];
    options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .phases = vec![StrongPhase {
        candidates: 1,
        subtree_ratio: 0.,
        pricing_phase_floor: 0,
        max_iterations: 1,
        tree_depth_score: false,
        restricted_master_only: false,
        ..StrongPhase::default()
    }];
    let parent_columns = output.snapshot.master.costs.len();
    let mut history = History::default();
    let selected = choose_registered(
        &output,
        &options,
        &mut history,
        f64::INFINITY,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    assert_eq!(history.evaluations.len(), 2);
    assert!(history.evaluations.iter().all(|entry| entry.kind
        == MappedEvaluationKind::ExactPricing
        && entry.estimate.is_finite()));
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
fn admitted_zero_budget_trials_record_sentinels_and_keep_inherited_proofs() {
    let output = root();
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    options.branching_registry.mapped_variables = vec![mapping("count42", 42)];
    let strong = options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap();
    strong.trial_time_limit = Duration::ZERO;
    strong.phases = vec![StrongPhase {
        candidates: 1,
        subtree_ratio: 0.,
        pricing_phase_floor: 0,
        max_iterations: 100,
        tree_depth_score: true,
        restricted_master_only: false,
        ..StrongPhase::default()
    }];
    let mut history = History::default();
    let selected = choose_registered(
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
    assert_eq!(history.evaluations.len(), 2);
    assert!(history
        .evaluations
        .iter()
        .all(|entry| entry.estimate == 1e12 && entry.kind == MappedEvaluationKind::ExactPricing));
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
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    // This fixture is also exercised by the retained read-only GDB observer,
    // which pauses after the first result until this local budget expires.
    options.branching_registry.mapped_variables = vec![mapping("count42", 42)];
    let strong = options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap();
    strong.time_limit = Duration::from_secs(3);
    strong.phases = vec![StrongPhase {
        candidates: 1,
        subtree_ratio: 0.,
        pricing_phase_floor: 0,
        max_iterations: 100,
        tree_depth_score: true,
        restricted_master_only: false,
        ..StrongPhase::default()
    }];
    let proposals = vec![Proposal {
        name: "count42".into(),
        history_key: "count42".into(),
        priority: 1.,
        value: 0.5,
        restriction: Restriction::Multiplicity { network: 42 },
        history_eligible: false,
        fresh: true,
        origin: None,
        component_admission: None,
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

/// Strong trials are opt-in (native `ApplyStrongBranchingEvaluation = false`);
/// these tests exercise them with the default phase schedule.
fn strong_tree_options() -> ModelTreeOptions {
    ModelTreeOptions {
        branching_registry: {
            let mut __branch_registry = crate::master::model::BranchingRegistry::default();
            __branch_registry.strong_branching =
                Some(crate::master::tree::StrongBranchingOptions::default());
            __branch_registry
        },
        ..ModelTreeOptions::default()
    }
}

#[test]
fn near_family_priorities_compare_fractionality_before_level_bits() {
    for reverse in [false, true] {
        let mut a = mapping("more-fractional", 42);
        a.priority = 1.;
        let mut b = mapping("higher-level", 42);
        b.priority = 1. + 5e-7;
        let proposed = |variable: &MappedBranchVariable, value| Proposal {
            name: variable.name.clone(),
            history_key: variable.name.clone(),
            priority: variable.priority,
            value,
            restriction: Restriction::Mapped(variable.clone()),
            history_eligible: true,
            fresh: true,
            origin: None,
            component_admission: None,
        };
        let mut candidates = vec![proposed(&a, 0.5), proposed(&b, 0.1)];
        if reverse {
            candidates.reverse();
        }
        let options = ModelTreeOptions::default();
        order_proposals(&mut candidates, &options);
        let mut order: Vec<_> = (0..candidates.len()).collect();
        rank_mapped_slots(
            &candidates,
            &ModelTreeOptions {
                branching_registry: {
                    let mut __branch_registry = crate::master::model::BranchingRegistry::default();
                    __branch_registry.mapped_variables = vec![a.clone(), b.clone()];
                    __branch_registry
                },
                ..Default::default()
            },
            &mut order,
        );
        assert_eq!(candidates[order[0]].name, "more-fractional");
        // A separation outside the recovered band remains the primary key.
        candidates
            .iter_mut()
            .find(|p| p.name == "higher-level")
            .unwrap()
            .priority = 1. + 2e-6;
        order_proposals(&mut candidates, &options);
        assert_eq!(candidates[0].name, "higher-level");
    }
}

#[test]
fn nearby_family_priority_chains_have_bounded_repeatable_admission() {
    let make = || {
        (0..128)
            .map(|index| {
                let mut variable = mapping(&format!("mapping-{index}"), 42);
                variable.priority = 1. + (index % 7) as f64 * 7.5e-7;
                Proposal {
                    name: variable.name.clone(),
                    history_key: variable.name.clone(),
                    priority: variable.priority,
                    value: ((index * 37) % 89 + 1) as f64 / 90.,
                    restriction: Restriction::Mapped(variable),
                    history_eligible: true,
                    fresh: true,
                    origin: None,
                    component_admission: None,
                }
            })
            .collect::<Vec<_>>()
    };
    let options = ModelTreeOptions::default();
    let mut a = make();
    let mut b = make();
    order_proposals(&mut a, &options);
    order_proposals(&mut b, &options);
    assert_eq!(a.len(), 128);
    assert_eq!(
        a.iter().map(|p| &p.name).collect::<Vec<_>>(),
        b.iter().map(|p| &p.name).collect::<Vec<_>>()
    );
    let unique: HashSet<_> = a.iter().map(|p| &p.name).collect();
    assert_eq!(unique.len(), 128);
}

#[test]
fn suppressed_packing_assignment_is_refreshed_for_history_ranking() {
    let problem: Problem<2> = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/multi_network_native_tree.json"
    )))
    .unwrap()
    .problem()
    .unwrap();
    let mut packing = PackingBranching::new(
        1000.,
        vec![
            NetworkPackingAssignment::new(
                7,
                vec![(1, 0), (1, 1), (2, 1), (2, 2), (3, 0), (4, 1), (5, 2)],
            ),
            NetworkPackingAssignment::new(42, vec![(1, 0), (1, 2), (2, 0), (3, 1), (4, 2)]),
        ],
    );
    packing
        .prepare(problem.networks.iter().map(|n| (n.id, &n.graph)), &[])
        .unwrap();
    let mut node_options = Options::default();
    node_options.cg.stabilize = false;
    node_options.cg.artificial_cost = 100.;
    let mut output = solve(problem, node_options).unwrap();
    output.snapshot.state.node_depth = 1;
    for (route, value) in output.routes.iter().zip(&mut output.values) {
        if route.network == 42 {
            *value *= 0.6;
        }
    }
    let fresh = packing.proposals(&output, 1e-6);
    assert!(!fresh.iter().any(|p| p.name == "ESA_42_2"));
    let mut options = strong_tree_options();
    options.branching_registry.branching = NetworkBranchingPolicy::Phased;
    options.branching_registry.packing = Some(packing);
    options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .time_limit = Duration::ZERO;
    let mut history = History::default();
    history.templates.insert(
        "mapped:ESA_42_2".into(),
        HistoryTemplate::PackingAssignment("ESA_42_2".into()),
    );
    for side in 0..2 {
        history.record_with_key(
            "mapped:ESA_42_2",
            MappedBranchEvaluation {
                name: "ESA_42_2".into(),
                depth: 0,
                side,
                stage: 1,
                kind: MappedEvaluationKind::RestrictedMaster,
                parent: 0.,
                estimate: 10.,
                distance: 0.5,
            },
            f64::INFINITY,
        );
    }
    let selection = choose_proposals(
        &output,
        &options,
        &mut history,
        100.,
        Instant::now() + Duration::from_secs(5),
        fresh,
    )
    .unwrap();
    assert!(selection.is_some());
    assert_eq!(
        history.ranked, 1,
        "the suppressed network assignment enters history selection"
    );
}

#[test]
fn pricing_trials_use_previous_lp_order_and_keep_ties_and_canonical_sides() {
    // Executed against original BestNodeLpValue, continuous minimization tag 2.
    assert_eq!(trial_sides([0, 1], [12., 14.], true), [1, 0]);
    assert_eq!(trial_sides([0, 1], [14., 12.], true), [0, 1]);
    assert_eq!(trial_sides([0, 1], [12., 12.0000001], true), [0, 1]);
    assert_eq!(trial_sides([1, 0], [12., 12.0000001], true), [1, 0]);
    assert_eq!(trial_sides([0, 1], [-4., -2.], true), [1, 0]);
    // LP-only treatment does not reorder the generator's child group.
    assert_eq!(trial_sides([0, 1], [12., 14.], false), [0, 1]);
    assert_eq!(trial_sides([1, 0], [14., 12.], false), [1, 0]);
}

#[test]
fn multiway_trial_order_keeps_ties_and_screen_generator_order() {
    let estimates = [12., 14., 14.0000001, 9.];
    let generator_order = vec![2, 0, 1, 3];
    assert_eq!(
        trial_sides(generator_order.clone(), estimates, false),
        generator_order
    );
    assert_eq!(trial_sides(generator_order, estimates, true), [2, 1, 0, 3]);
    assert_eq!(
        trial_sides([0, 1, 2], [12., 12.0000001, 12.], true),
        [0, 1, 2]
    );
}

#[test]
fn heuristic_trial_disables_cut_rounds_fixing_and_enumeration_locally() {
    let mut node = Options::default();
    node.integer_primal_termination = true;
    node.cg.min_cut_rounds = 2;
    node.cg.max_cut_rounds = 50;
    node.separation.rank_one.min_rounds_before_stop_by_sp = 2;
    let original = node.clone();
    let phase = StrongBranchingOptions::default().phases.remove(1);
    configure_trial_options(&mut node, &phase, 20.);
    assert!(!node.integer_primal_termination);
    assert!(!node.reduced_cost_fixing);
    assert!(!node.gap_enumeration);
    assert_eq!(node.fixing_upper_bound, None);
    assert_eq!(node.cg.min_cut_rounds, 0);
    assert_eq!(node.cg.max_cut_rounds, 0);
    assert_eq!(node.separation.rank_one.min_rounds_before_stop_by_sp, 0);
    assert_eq!(node.cg.max_iterations, 10000);
    assert!(original.reduced_cost_fixing);
    assert!(original.gap_enumeration);
    assert_eq!(original.cg.min_cut_rounds, 2);
    assert_eq!(original.cg.max_cut_rounds, 50);
}

#[test]
fn exact_trial_retains_global_minimum_cut_policy_and_enables_fixing() {
    let mut node = Options::default();
    assert!(!node.integer_primal_termination);
    node.cg.min_cut_rounds = 2;
    node.separation.rank_one.min_rounds_before_stop_by_sp = 2;
    let phase = StrongBranchingOptions::default().phases.remove(2);
    configure_trial_options(&mut node, &phase, 20.);
    assert!(node.integer_primal_termination);
    assert!(node.reduced_cost_fixing);
    assert!(node.gap_enumeration);
    assert_eq!(node.fixing_upper_bound, Some(20.));
    assert_eq!(node.cg.min_cut_rounds, 2);
    assert_eq!(node.cg.max_cut_rounds, 10000);
    assert_eq!(node.separation.rank_one.min_rounds_before_stop_by_sp, 2);
}

#[test]
fn phase_exactness_can_be_disabled_independently_of_pricing_floor() {
    let mut phase = StrongPhase::default();
    assert!(phase.is_exact());
    phase.exact = Some(false);
    assert!(!phase.is_exact());
    let mut node = Options::default();
    node.integer_primal_termination = true;
    configure_trial_options(&mut node, &phase, 20.);
    assert!(!node.integer_primal_termination);
    let mut options = StrongBranchingOptions::default();
    options.phases = vec![phase];
    assert!(validate(&options).is_ok());
    options.phases[0].exact = Some(true);
    options.phases[0].pricing_phase_floor = 1;
    assert!(validate(&options).is_err());
}

#[test]
fn partial_exact_group_is_abandoned_only_after_a_usable_best_score() {
    let best = strong::tree_depth_score(10., 20., [14., 16.], 1.);
    assert!(partial_group_cannot_improve(
        10.,
        20.,
        [11., 12.],
        1,
        1.,
        best
    ));
    assert!(!partial_group_cannot_improve(
        10.,
        20.,
        [18., 19.],
        1,
        1.,
        best
    ));
    assert!(!partial_group_cannot_improve(
        10.,
        20.,
        [11., 12.],
        1,
        1.,
        -1e12
    ));
    // Comparison includes numerical slack, preserving an existing best group
    // when the optimistic partial score is equivalent.
    assert!(partial_group_cannot_improve(
        10.,
        20.,
        [14., 16.],
        1,
        1.,
        best
    ));
}

#[test]
fn simplified_phase_defaults_keep_zero_cg_lp_screen_and_native_work_controls() {
    let options = StrongBranchingOptions::default();
    assert!(validate(&options).is_ok());
    assert_eq!(options.phases[0].max_iterations, 0);
    assert!(options.phases[0].restricted_master_only);
    assert!(!options.phases[0].is_exact());
    assert_eq!(options.phases[1].max_cut_rounds, 0);
    assert!(!options.phases[1].reduced_cost_fixing);
    assert!(!options.phases[1].is_exact());
    assert!(options.phases[2].is_exact());
    assert_eq!(options.phases[2].max_cut_rounds, 10000);
    assert!(options.phases[2].reduced_cost_fixing);
    assert!(!options.stop_after_improved_subtree);
    assert_eq!(options.partial_score_conservativeness, 1.);
    assert_eq!(options.time_limit, Duration::MAX);
    assert_eq!(options.trial_time_limit, Duration::MAX);
}

#[test]
fn exact_phase_interrupt_stops_after_an_improved_finite_subtree_estimate() {
    let mut output = root();
    output.snapshot.state.mapped_subtree_size = 100.;
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    options.branching_registry.mapped_variables = vec![mapping("first", 42), mapping("second", 42)];
    let config = options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap();
    config.phases = vec![StrongPhase {
        candidates: 2,
        subtree_ratio: 0.,
        // Keep both siblings' ranking information independent of incumbent
        // gap enumeration; the interruption policy only changes later work.
        reduced_cost_fixing: false,
        ..StrongPhase::default()
    }];
    config.time_limit = Duration::from_secs(30);
    config.trial_time_limit = Duration::from_secs(10);
    let mut full_history = History::default();
    let full = choose_registered(
        &output,
        &options,
        &mut full_history,
        3.,
        Instant::now() + Duration::from_secs(30),
    )
    .unwrap()
    .unwrap();
    options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .stop_after_improved_subtree = true;
    let mut interrupted_history = History::default();
    let interrupted = choose_registered(
        &output,
        &options,
        &mut interrupted_history,
        3.,
        Instant::now() + Duration::from_secs(30),
    )
    .unwrap()
    .unwrap();
    assert_eq!(interrupted_history.trials, 2);
    assert!(full_history.trials > interrupted_history.trials);
    assert!(interrupted.children[0].state.mapped_subtree_size < 100.);
    for (ordinary, shortened) in full.children.iter().zip(&interrupted.children) {
        assert_eq!(
            ordinary.warm.numerical_bound,
            shortened.warm.numerical_bound
        );
        assert_eq!(
            ordinary.warm.certified_bound,
            shortened.warm.certified_bound
        );
        assert_eq!(
            ordinary.state.shared_rows.len(),
            shortened.state.shared_rows.len()
        );
    }
    assert_eq!(output.snapshot.state.mapped_subtree_size, 100.);
}

#[test]
fn discarded_partial_trial_does_not_admit_an_unvisited_successor() {
    // A completed; B was abandoned by its partial score; the ratio/deadline
    // stops before C. Removing B first would incorrectly keep untouched C.
    let mut candidates = vec![
        ("completed", true),
        ("abandoned", false),
        ("unvisited", true),
    ];
    retain_trial_prefix(&mut candidates, 2, |candidate| candidate.1);
    assert_eq!(candidates, [("completed", true)]);
    // A deadline reached before any trial leaves ordinary branching choices.
    let mut pending = vec![("unsupported", false), ("deferred", true)];
    retain_trial_prefix(&mut pending, 0, |candidate| candidate.1);
    assert_eq!(pending, [("deferred", true)]);
}

#[test]
fn final_admission_keeps_the_last_pricing_phase_child_order_and_proofs() {
    let output = root();
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    options.branching_registry.mapped_variables = vec![mapping("first", 42), mapping("second", 42)];
    options
        .branching_registry
        .branch_directions
        .insert("first".into(), NetworkBranchDirection::UpperFirst);
    let config = options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap();
    config.phases = vec![
        StrongPhase {
            candidates: 2,
            subtree_ratio: 0.,
            max_iterations: 0,
            tree_depth_score: false,
            restricted_master_only: true,
            max_cut_rounds: 0,
            reduced_cost_fixing: false,
            ..StrongPhase::default()
        },
        StrongPhase {
            subtree_ratio: 0.,
            reduced_cost_fixing: false,
            ..StrongPhase::default()
        },
    ];
    config.time_limit = Duration::from_secs(30);
    config.trial_time_limit = Duration::from_secs(10);
    let mut history = History::default();
    let mut selection = choose_registered(
        &output,
        &options,
        &mut history,
        3.,
        Instant::now() + Duration::from_secs(30),
    )
    .unwrap()
    .unwrap();
    let mut preceding = [0.; 2];
    for evaluation in history.evaluations.iter().filter(|evaluation| {
        evaluation.name == "first" && evaluation.kind == MappedEvaluationKind::RestrictedMaster
    }) {
        preceding[evaluation.side] = evaluation.estimate;
    }
    assert!(exceeds(preceding[0], preceding[1]));
    assert_eq!(
        selection.direction,
        Some(NetworkBranchDirection::LowerFirst)
    );
    let lower_numerical = selection.children[0].warm.numerical_bound;
    let lower_certified = selection.children[0].warm.certified_bound;
    // The tree's default/registered upper-first direction must not undo the
    // group's recovered LP sort at final insertion.
    selection.admit(NetworkBranchDirection::UpperFirst);
    assert_eq!(selection.children[0].warm.numerical_bound, lower_numerical);
    assert_eq!(selection.children[0].warm.certified_bound, lower_certified);
    assert!(
        selection.children[0].state.shared_rows.len() > output.snapshot.state.shared_rows.len()
    );
    if let Some(evaluated) = &selection.evaluated[0] {
        assert_eq!(evaluated.snapshot.warm.numerical_bound, lower_numerical);
        assert_eq!(evaluated.snapshot.warm.certified_bound, lower_certified);
    }
}

#[test]
fn completed_objective_fixed_trials_restore_published_bounds_after_selection_fork() {
    let output = root();
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    options.branching_registry.mapped_variables = vec![mapping("count42", 42)];
    let mut history = History::default();
    let selected = choose_registered(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    assert!(selected
        .evaluated
        .iter()
        .flatten()
        .any(|trial| trial.snapshot.state.objective_fixed));
    for (child, trial) in selected.children.iter().zip(&selected.evaluated) {
        let trial = trial.as_ref().unwrap();
        // Selection forks restore objective-dependent pricing restrictions,
        // while the independently published original-domain bounds survive.
        assert!(!child.state.objective_fixed);
        assert!(child.warm.numerical_bound >= trial.master.numerical_bound);
        assert!(child.warm.certified_bound >= trial.master.certified_bound);
        assert!(child.warm.bound >= output.snapshot.warm.bound);
        assert_eq!(child.warm.bound_mode, output.snapshot.warm.bound_mode);
        assert_eq!(child.warm.bound, child.warm.numerical_bound);
    }
}

#[test]
fn exact_trial_cannot_reenable_globally_disabled_reduced_cost_fixing() {
    let mut node = Options::default();
    node.reduced_cost_fixing = false;
    node.enumeration = Some(EnumerationOptions::default());
    let phase = StrongPhase::default();
    configure_trial_options(&mut node, &phase, 20.);
    assert!(!node.reduced_cost_fixing);
    assert!(!node.gap_enumeration);
    assert!(node.fixing_upper_bound.is_none());
    // An explicitly requested complete-pool enumeration is independent of
    // incumbent-dependent fixing. The CLI disabling both supplies None.
    assert!(node.enumeration.is_some());
}

fn capped_history_observation(
    name: &str,
    side: usize,
    estimate: f64,
    distance: f64,
) -> MappedBranchEvaluation {
    MappedBranchEvaluation {
        name: name.into(),
        depth: 0,
        side,
        stage: 1,
        kind: MappedEvaluationKind::RestrictedMaster,
        parent: 10.,
        estimate,
        distance,
    }
}

#[test]
fn child_local_incumbent_caps_pseudocost_history_and_changes_later_ranking() {
    let mut history = History::default();
    history.record_with_key("a", capped_history_observation("a", 0, 30., 0.1), 20.);
    history.record_with_key("a", capped_history_observation("a", 1, 14., 0.9), 20.);
    history.record_with_key("b", capped_history_observation("b", 0, 12., 0.5), 20.);
    history.record_with_key("b", capped_history_observation("b", 1, 13., 0.5), 20.);
    // Original native recording gives A means [100, 4.444...] rather than
    // [200, 4.444...]. With the same gap and a changed current fraction, that
    // reverses A/B history order: 4.4 < 6 rather than 8.8 > 6.
    let a = history.prediction("a", 0.01, 10.).unwrap();
    let b = history.prediction("b", 0.5, 10.).unwrap();
    assert_eq!(a.0, 1);
    assert!((a.1 - 4.4).abs() < 1e-12);
    assert_eq!(b, (1, 6.));
    assert!(b.1 > a.1);
    // The trace remains a raw-LP diagnostic, independent of capped pseudocosts.
    assert_eq!(history.evaluations[0].estimate, 30.);
}

#[test]
fn history_cap_respects_numerical_slack_and_preserves_signed_observations() {
    for (estimate, expected) in [(20. + 5e-7, 20.000000999999997), (20. + 5e-6, 20.)] {
        let mut history = History::default();
        history.record_with_key(
            "near",
            capped_history_observation("near", 0, estimate, 0.5),
            20.,
        );
        assert!((history.costs["near"][&1][0].mean - expected).abs() < 1e-12);
    }
    let mut history = History::default();
    for estimate in [8., 12.] {
        history.record_with_key(
            "signed",
            capped_history_observation("signed", 0, estimate, 0.5),
            20.,
        );
    }
    history.record_with_key(
        "signed",
        capped_history_observation("signed", 1, 14., 0.5),
        20.,
    );
    assert_eq!(history.prediction("signed", 0.5, 10.), Some((1, 0.)));
}

#[test]
fn invalid_history_inputs_do_not_become_finite_through_incumbent_capping() {
    let mut history = History::default();
    for estimate in [f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
        history.record_with_key(
            "invalid",
            capped_history_observation("invalid", 0, estimate, 0.5),
            20.,
        );
    }
    for incumbent in [f64::NEG_INFINITY, f64::NAN] {
        history.record_with_key(
            "invalid",
            capped_history_observation("invalid", 0, 30., 0.5),
            incumbent,
        );
    }
    let mut invalid_parent = capped_history_observation("invalid", 0, 30., 0.5);
    invalid_parent.parent = f64::NEG_INFINITY;
    history.record_with_key("invalid", invalid_parent, 20.);
    assert!(history.costs.is_empty());
    assert!(history.evaluations.is_empty());
    // An unknown incumbent leaves finite signed costs usable.
    history.record_with_key(
        "unknown",
        capped_history_observation("unknown", 0, 30., 0.5),
        f64::INFINITY,
    );
    assert_eq!(history.costs["unknown"][&1][0].mean, 40.);
}

#[test]
fn partial_score_conservativeness_scales_only_the_unprocessed_child() {
    let best = strong::tree_depth_score(10., 20., [12.5, 12.5], 1.);
    // Actual native partial score is -0.232479..., which can beat -0.399999...
    // Normalizing both gains by the factor instead produced -0.694242... and
    // discarded this candidate before its improving final score was known.
    let partial = strong::partial_tree_depth_score(10., 20., [12., 14.], 1, 2.);
    assert!((partial - -0.23247909557169236).abs() < 1e-12);
    assert!(!partial_group_cannot_improve(
        10.,
        20.,
        [12., 14.],
        1,
        2.,
        best
    ));
    assert!(
        (strong::partial_tree_depth_score(10., 20., [12., 14.], 2, 2.)
            - strong::tree_depth_score(10., 20., [12., 14.], 1.))
        .abs()
            < 1e-12
    );
    // The caller passes the actual trial order, independently of canonical
    // lower/upper storage: only the second entry is pending in each fixture.
    assert_ne!(
        partial,
        strong::partial_tree_depth_score(10., 20., [14., 12.], 1, 2.)
    );
}

#[test]
fn partial_score_detects_zero_gains_before_scaling_and_clips_after_scaling() {
    // A small positive pending gain crosses below 1e-6 after scaling. Native
    // does not reclassify it as zero or replace it by gap * 0.0001.
    let small = strong::partial_tree_depth_score(10., 20., [12., 10.000002], 1, 0.01);
    assert!((small - -11.302792258242055).abs() < 1e-12);
    // Scaling precedes gap clipping: a pending raw gain of 11 becomes 5.5,
    // rather than first clipping it to 10 and then scaling it to 5.
    let clipped = strong::partial_tree_depth_score(10., 20., [14., 21.], 1, 0.5);
    assert_eq!(clipped, strong::tree_depth_score(10., 20., [14., 15.5], 1.));
    assert_eq!(
        strong::partial_tree_depth_score(10., 20., [10., 10.0000001], 0, 1e6),
        -1e12
    );
    for processed in 0..=2 {
        assert_eq!(
            strong::partial_tree_depth_score(10., 20., [12., 14.], processed, 1.),
            strong::tree_depth_score(10., 20., [12., 14.], 1.)
        );
    }
}

#[test]
fn committed_nonexact_children_record_configured_followup_stage_once() {
    for phase_count in [1, 2] {
        let output = root();
        let mut options = strong_tree_options();
        options.node.cg.stabilize = false;
        options.branching_registry.mapped_variables =
            vec![mapping("screen-a", 42), mapping("screen-b", 42)];
        let phase = StrongPhase {
            candidates: 2,
            subtree_ratio: 0.,
            restricted_master_only: true,
            max_iterations: 0,
            tree_depth_score: false,
            ..StrongPhase::default()
        };
        options
            .branching_registry
            .strong_branching
            .as_mut()
            .unwrap()
            .phases = vec![phase; phase_count];
        let mut history = History::default();
        let selected = choose_registered(
            &output,
            &options,
            &mut history,
            2.,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap()
        .unwrap();
        assert_eq!(history.trials, 4);
        assert!(selected.evaluated.iter().all(Option::is_none));
        for (side, child) in selected.children.into_iter().enumerate() {
            let origin = child.state.branch_history_origin.as_ref().unwrap();
            assert_eq!(origin.side, side);
            assert_eq!(origin.stage, phase_count + 1);
            assert_eq!(origin.parent, 1.5);
            assert_eq!(origin.distance, 0.5);
            let mut solved = child.solve(options.node.clone()).unwrap();
            let before = history.evaluations.len();
            history
                .record_committed(&mut solved, 2., false, options.tolerance)
                .unwrap();
            assert_eq!(history.evaluations.len(), before + 1);
            let recorded = history.evaluations.last().unwrap();
            assert_eq!((recorded.side, recorded.stage), (side, phase_count + 1));
            assert_eq!(recorded.kind, MappedEvaluationKind::ExactPricing);
            assert_eq!(recorded.depth, 0);
            history
                .record_committed(&mut solved, 2., false, options.tolerance)
                .unwrap();
            assert_eq!(history.evaluations.len(), before + 1);
        }
    }
}

#[test]
fn completed_exact_children_do_not_repeat_committed_history() {
    let output = root();
    let mut options = strong_tree_options();
    options.branching_registry.mapped_variables = vec![mapping("screen-a", 42)];
    let mut history = History::default();
    let selected = choose_registered(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    let before = history.evaluations.len();
    for mut solved in selected.evaluated.into_iter().flatten() {
        assert!(solved.snapshot.state.branch_history_origin.is_none());
        history
            .record_committed(&mut solved, 2., true, options.tolerance)
            .unwrap();
    }
    assert_eq!(history.evaluations.len(), before);
}

#[test]
fn heuristic_workspace_survives_a_restricted_lp_resolve_before_exact_commit() {
    let output = compact_pricing_root();
    let parent_columns = output.snapshot.master.costs.len();
    let mut options = continuation_options(false);
    options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .phases
        .truncate(1);
    let mut history = History::default();
    let selected = choose_registered(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    assert_eq!(history.trials, 4);
    assert!(history
        .evaluations
        .iter()
        .all(|evaluation| evaluation.kind == MappedEvaluationKind::HeuristicPricing));
    assert!(selected.evaluated.iter().all(Option::is_none));
    for (side, mut child) in selected.children.into_iter().enumerate() {
        let priced_columns = child.master.costs.len();
        assert!(priced_columns > parent_columns);
        let expected = if side == 0 { 2. } else { 5. / 3. };
        // Exercise the same restricted-master operation used by the phase
        // loop directly: equal-score scheduler collapse would skip this
        // redundant final screen for the triangle fixture.
        assert_eq!(
            child
                .master
                .solve_limited(i32::MAX as usize, Duration::from_secs(10))
                .unwrap(),
            crate::master::clp::ClpSolveOutcome::Optimal
        );
        assert!((child.master.objective() - expected).abs() < 1e-8);
        assert_eq!(child.master.costs.len(), priced_columns);
        assert_eq!(child.warm.bound, output.snapshot.warm.bound);
        assert_eq!(
            child.warm.numerical_bound,
            output.snapshot.warm.numerical_bound
        );
        assert_eq!(
            child.warm.certified_bound,
            output.snapshot.warm.certified_bound
        );
        assert!(child
            .state
            .oracles
            .iter()
            .all(|oracle| oracle.solver.phases.len() == 2));
        let resumed = child.solve(options.node.clone()).unwrap();
        assert_eq!(resumed.master.status, cg::Status::Optimal);
        assert!((resumed.master.bound - expected).abs() < 1e-8);
        assert!(resumed.snapshot.master.costs.len() >= priced_columns);
    }
    assert_eq!(output.snapshot.master.costs.len(), parent_columns);
}

// A controlled LP point on retained real routes. This fixture tests candidate
// admission and row construction, not optimality or feasibility of this point.
fn aggregate_projection_point(
    integer_quantity: f64,
    continuous_quantity: f64,
) -> (Output<2>, ModelTreeOptions) {
    let mut output = root();
    output.values.fill(0.);
    output.snapshot.master.values.fill(0.);
    let mut members = vec![];
    let mut mapped = mapping("AGG_9", 42);
    mapped.networks.clear();
    for (network, quantity, integer) in [
        (42, integer_quantity, true),
        (7, continuous_quantity, false),
    ] {
        let position = output
            .routes
            .iter()
            .position(|route| route.network == network)
            .unwrap();
        let route = output.routes[position].clone();
        let arc = route.route.arcs[0];
        output.values[position] = quantity;
        for (column, &tag) in output.master.column_tags.iter().enumerate() {
            if tag >= 0 {
                let stored = &output.snapshot.state.routes[tag as usize];
                if stored.network == network && stored.route.arcs == route.route.arcs {
                    output.snapshot.master.values[column] = quantity;
                }
            }
        }
        members.push(
            serde_json::from_value::<super::OriginalPricingVariable>(serde_json::json!({
                "network": network, "id": 9, "cost": 0., "arcs": [[arc, 1.]], "integer": integer
            }))
            .unwrap(),
        );
        mapped.networks.push(MappedNetworkVariable {
            network,
            activities: vec![],
            arcs: vec![(arc, 1.)],
            per_route: 0.,
        });
    }
    let mut options = strong_tree_options();
    options.branching_registry.branching = NetworkBranchingPolicy::Phased;
    options.branching_registry.use_branching_history = true;
    options.branching_registry.mapped_variables = vec![mapped];
    options
        .branching_registry
        .aggregate_projection
        .insert("AGG_9".into(), super::AggregateProjection::new(members));
    options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .phases = vec![StrongPhase {
        candidates: 2,
        subtree_ratio: 0.,
        restricted_master_only: true,
        max_iterations: 0,
        tree_depth_score: false,
        ..StrongPhase::default()
    }];
    (output, options)
}

#[test]
fn aggregate_projected_fraction_admits_an_integral_full_child_expression() {
    let (output, options) = aggregate_projection_point(0.5, 0.5);
    let proposals = mapped_proposals(&output, &options).unwrap();
    assert_eq!(proposals.len(), 1);
    assert!(proposals[0].fresh);
    assert_eq!(proposals[0].value, 1.);
    assert_eq!(
        options.branching_registry.aggregate_projection["AGG_9"].value(&output, true),
        Some(0.5)
    );
    let children = proposals[0].children(&output.snapshot).unwrap();
    for (side, child) in children.iter().enumerate() {
        let row = child.state.shared_rows.last().unwrap();
        assert_eq!(row.rhs, 1.);
        assert_eq!(
            row.sense,
            if side == 0 {
                RowSense::LessEqual
            } else {
                RowSense::GreaterEqual
            }
        );
        // Both integer and continuous original networks remain in the full row.
        for network in [42, 7] {
            let before = output
                .snapshot
                .state
                .oracles
                .iter()
                .find(|oracle| oracle.network.id == network)
                .unwrap();
            let after = child
                .state
                .oracles
                .iter()
                .find(|oracle| oracle.network.id == network)
                .unwrap();
            assert!(after
                .network
                .arc_entries
                .iter()
                .zip(before.network.arc_entries.iter())
                .any(|(a, b)| a.len() > b.len()));
        }
    }
    let selected = choose_proposals(
        &output,
        &options,
        &mut History::default(),
        2.,
        Instant::now() + Duration::from_secs(10),
        proposals,
    )
    .unwrap()
    .unwrap();
    assert_eq!(selected.direction, Some(NetworkBranchDirection::UpperFirst));
}

#[test]
fn aggregate_history_refresh_survives_fresh_projection_rejection_but_not_integral_full_value() {
    let (mut output, options) = aggregate_projection_point(1., 0.5);
    assert_eq!(
        options.branching_registry.aggregate_projection["AGG_9"].value(&output, true),
        Some(0.)
    );
    assert!(
        mapped_proposals(&output, &options).unwrap().is_empty(),
        "root has no retained history candidates"
    );
    output.snapshot.state.node_depth = 1;
    let proposals = mapped_proposals(&output, &options).unwrap();
    assert_eq!(proposals.len(), 1);
    assert!(!proposals[0].fresh);
    assert_eq!(proposals[0].value, 1.5);
    let mut absent_history = History::default();
    assert!(choose_proposals(
        &output,
        &options,
        &mut absent_history,
        2.,
        Instant::now() + Duration::from_secs(10),
        mapped_proposals(&output, &options).unwrap()
    )
    .unwrap()
    .is_none());
    let mut history = History::default();
    for side in 0..2 {
        history.record_with_key(
            "mapped:AGG_9",
            MappedBranchEvaluation {
                name: "AGG_9".into(),
                depth: 0,
                side,
                stage: 2,
                kind: MappedEvaluationKind::ExactPricing,
                parent: 1.,
                estimate: 1.5,
                distance: 0.5,
            },
            2.,
        );
    }
    assert_eq!(
        history.prediction("mapped:AGG_9", 1.5, 0.5),
        Some((2, 0.25))
    );
    let selected = choose_proposals(
        &output,
        &options,
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
        proposals,
    )
    .unwrap();
    assert!(selected.is_some());
    assert_eq!(history.ranked, 1);
    assert_eq!(history.trials, 0);
    let (mut integral, integral_options) = aggregate_projection_point(1., 0.);
    integral.snapshot.state.node_depth = 1;
    assert!(mapped_proposals(&integral, &integral_options)
        .unwrap()
        .is_empty());
}

#[test]
fn fresh_aggregate_selection_uses_placeholder_activity_and_preserves_full_values() {
    let first = mapping("z-first", 42);
    let second = mapping("a-second", 42);
    let ordinary = mapping("ordinary", 42);
    let proposal = |variable: &MappedBranchVariable, value, aggregate| Proposal {
        name: variable.name.clone(),
        history_key: variable.name.clone(),
        priority: 1.,
        value,
        restriction: if aggregate {
            Restriction::Aggregate {
                variable: variable.clone(),
                fixed_activity: 0.,
            }
        } else {
            Restriction::Mapped(variable.clone())
        },
        history_eligible: true,
        fresh: true,
        origin: None,
        component_admission: None,
    };
    let mut options = ModelTreeOptions {
        branching_registry: {
            let mut __branch_registry = crate::master::model::BranchingRegistry::default();
            __branch_registry.mapped_variables =
                vec![first.clone(), second.clone(), ordinary.clone()];
            __branch_registry
        },
        ..Default::default()
    };
    // Retain projection metadata and the actual aggregate payload produced by
    // materialization; this test isolates ordering from candidate projection.
    for name in ["z-first", "a-second"] {
        options
            .branching_registry
            .aggregate_projection
            .insert(name.into(), super::AggregateProjection::new(vec![]));
    }
    let mut candidates = vec![
        proposal(&first, 0.1, true),
        proposal(&second, 0.5, true),
        proposal(&ordinary, 0.4, false),
    ];
    order_proposals(&mut candidates, &options);
    let mut order = vec![0, 1, 2];
    rank_mapped_slots(&candidates, &options, &mut order);
    assert_eq!(
        order
            .iter()
            .map(|&i| candidates[i].name.as_str())
            .collect::<Vec<_>>(),
        ["ordinary", "z-first", "a-second"]
    );
    assert_eq!(
        candidates
            .iter()
            .find(|p| p.name == "z-first")
            .unwrap()
            .value,
        0.1
    );
    assert_eq!(
        candidates
            .iter()
            .find(|p| p.name == "a-second")
            .unwrap()
            .value,
        0.5
    );
    // Refreshed history candidates retain their real full activity view.
    for candidate in &mut candidates {
        candidate.fresh = false;
    }
    order_proposals(&mut candidates, &options);
    let mut order = vec![0, 1, 2];
    rank_mapped_slots(&candidates, &options, &mut order);
    assert_eq!(
        order
            .iter()
            .map(|&i| candidates[i].name.as_str())
            .collect::<Vec<_>>(),
        ["a-second", "ordinary", "z-first"]
    );
}

#[test]
fn explicit_aggregate_selection_metadata_compares_the_placeholder_view() {
    let first = mapping("z-first", 42);
    let second = mapping("a-second", 42);
    let mut options = ModelTreeOptions {
        branching_registry: {
            let mut __branch_registry = crate::master::model::BranchingRegistry::default();
            __branch_registry.mapped_variables = vec![first.clone(), second.clone()];
            __branch_registry
        },
        ..Default::default()
    };
    for (name, factor) in [("z-first", 10.), ("a-second", 0.)] {
        options
            .branching_registry
            .aggregate_projection
            .insert(name.into(), super::AggregateProjection::new(vec![]));
        options.branching_registry.branch_selection.insert(
            name.into(),
            super::BranchSelection {
                family: "aggregate".into(),
                default_factor: factor,
                ..Default::default()
            },
        );
    }
    let mut candidates = vec![
        Proposal {
            name: first.name.clone(),
            history_key: first.name.clone(),
            priority: 1.,
            value: 0.5,
            restriction: Restriction::Aggregate {
                variable: first,
                fixed_activity: 0.,
            },
            history_eligible: true,
            fresh: true,
            origin: None,
            component_admission: None,
        },
        Proposal {
            name: second.name.clone(),
            history_key: second.name.clone(),
            priority: 1.,
            value: 0.1,
            restriction: Restriction::Aggregate {
                variable: second,
                fixed_activity: 0.,
            },
            history_eligible: true,
            fresh: true,
            origin: None,
            component_admission: None,
        },
    ];
    order_proposals(&mut candidates, &options);
    assert_eq!(
        candidates[0].name, "a-second",
        "equal placeholders honor the explicit default factor"
    );
    assert_eq!(candidates[0].value, 0.1);
}

#[test]
fn skipped_packing_screen_retains_generator_before_committed_observation() {
    let problem: Problem<2> = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/multi_network_native_tree.json"
    )))
    .unwrap()
    .problem()
    .unwrap();
    let mut packing = PackingBranching::new(
        1000.,
        vec![
            NetworkPackingAssignment::new(
                7,
                vec![(1, 0), (1, 1), (2, 1), (2, 2), (3, 0), (4, 1), (5, 2)],
            ),
            NetworkPackingAssignment::new(42, vec![(1, 0), (1, 2), (2, 0), (3, 1), (4, 2)]),
        ],
    );
    packing
        .prepare(problem.networks.iter().map(|n| (n.id, &n.graph)), &[])
        .unwrap();
    let mut node_options = Options::default();
    node_options.cg.stabilize = false;
    node_options.cg.artificial_cost = 100.;
    let output = solve(problem, node_options).unwrap();
    let proposals = packing.proposals(&output, 1e-6);
    assert!(!proposals.is_empty());
    let mut options = strong_tree_options();
    options.branching_registry.branching = NetworkBranchingPolicy::Phased;
    options.branching_registry.packing = Some(packing);
    options.node.cg.stabilize = false;
    options.node.cg.artificial_cost = 100.;
    options.node.reduced_cost_fixing = false;
    options.node.gap_enumeration = false;
    options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .phases = vec![StrongPhase {
        candidates: 1,
        subtree_ratio: 0.,
        restricted_master_only: true,
        exact: Some(false),
        ..Default::default()
    }];
    let mut history = History::default();
    let selected = choose_proposals(
        &output,
        &options,
        &mut history,
        100.,
        Instant::now() + Duration::from_secs(10),
        proposals,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        history.trials, 0,
        "singleton nonexact phase must be skipped"
    );
    assert!(history.costs.is_empty());
    assert_eq!(history.templates.len(), 1);
    let origin = selected.children[0]
        .state
        .branch_history_origin
        .as_ref()
        .unwrap();
    let key = origin.key.clone();
    assert!(
        matches!(history.templates.get(&key), Some(HistoryTemplate::PackingAssignment(name)) if *name == origin.name)
    );
    let mut observations = 0;
    for child in selected.children {
        let mut solved = child.solve(options.node.clone()).unwrap();
        history
            .record_committed(&mut solved, 100., false, options.tolerance)
            .unwrap();
        observations += usize::from(history.costs.contains_key(&key));
    }
    assert!(
        observations > 0,
        "ordinary treatment must populate the retained generator"
    );
    assert!(!history.evaluations.is_empty());
}

#[test]
fn packing_history_matches_original_finite_sentinel_and_capped_infeasible_entries() {
    #[derive(serde::Deserialize)]
    struct Stage {
        means_bits: Vec<u64>,
        counts: Vec<usize>,
    }
    #[derive(serde::Deserialize)]
    struct Entry {
        key: String,
        stage: usize,
        side: usize,
        parent_bits: u64,
        fraction_bits: u64,
        estimate_bits: u64,
        incumbent_bits: u64,
        after: Vec<Stage>,
    }
    let entries: Vec<Entry> =
        serde_json::from_str(include_str!("packing_history_fixtures.json")).unwrap();
    let mut history = History::default();
    for entry in entries {
        let fraction = f64::from_bits(entry.fraction_bits);
        history.record_with_key(
            &entry.key,
            MappedBranchEvaluation {
                name: entry.key.clone(),
                depth: 0,
                side: entry.side,
                stage: entry.stage,
                kind: if entry.stage == 1 {
                    MappedEvaluationKind::HeuristicPricing
                } else {
                    MappedEvaluationKind::ExactPricing
                },
                parent: f64::from_bits(entry.parent_bits),
                estimate: f64::from_bits(entry.estimate_bits),
                distance: if entry.side == 0 {
                    fraction
                } else {
                    1. - fraction
                },
            },
            f64::from_bits(entry.incumbent_bits),
        );
        for (index, native) in entry.after.iter().enumerate() {
            let stats = &history.costs[&entry.key][&(index + 1)];
            // Native U generators store upper then lower; Rust uses canonical sides.
            for child in 0..2 {
                let expected_count = native.counts.get(child).copied().unwrap_or(0);
                assert_eq!(stats[1 - child].count, expected_count);
                if let Some(&bits) = native.means_bits.get(child) {
                    assert_eq!(stats[1 - child].mean.to_bits(), bits);
                }
            }
        }
    }
    assert_eq!(history.evaluations.len(), 8);
    assert!(history
        .evaluations
        .iter()
        .any(|entry| entry.estimate == 1e12));
}

#[test]
fn heuristic_packing_trials_record_partial_penalty_and_unresolved_sentinel_without_proof_promotion()
{
    for (max_iterations, expected_sentinel) in [(2, false), (10000, true)] {
        let problem: Problem<2> = format::Model::from_json(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/packing_quarter_history.json"
        )))
        .unwrap()
        .problem()
        .unwrap();
        let mut packing = PackingBranching::new(
            100.,
            vec![NetworkPackingAssignment::new(
                42,
                vec![(0, 0), (1, 1), (3, 2), (4, 3), (5, 4), (6, 5)],
            )],
        );
        packing
            .prepare(problem.networks.iter().map(|n| (n.id, &n.graph)), &[])
            .unwrap();
        let mut node = Options::default();
        node.cg.stabilize = false;
        node.preprocessing.enabled = false;
        node.gap_enumeration = false;
        node.reduced_cost_fixing = false;
        let output = solve(problem, node.clone()).unwrap();
        assert!((output.master.lp - 1.4375).abs() < 1e-9);
        let proposals = packing.proposals(&output, 1e-6);
        assert_eq!(proposals.len(), 3);
        let mut options = strong_tree_options();
        options.node = node;
        options.branching_registry.branching = NetworkBranchingPolicy::Phased;
        options.branching_registry.packing = Some(packing);
        options
            .branching_registry
            .strong_branching
            .as_mut()
            .unwrap()
            .phases = vec![
            StrongPhase {
                candidates: 100,
                subtree_ratio: 0.,
                pricing_phase_floor: 1,
                exact: Some(false),
                max_iterations,
                tree_depth_score: false,
                max_cut_rounds: 0,
                reduced_cost_fixing: false,
                ..Default::default()
            },
            StrongPhase {
                candidates: 1,
                subtree_ratio: 0.,
                exact: Some(true),
                max_cut_rounds: 0,
                reduced_cost_fixing: false,
                ..Default::default()
            },
        ];
        let parent_proofs = (
            output.snapshot.warm.numerical_bound,
            output.snapshot.warm.certified_bound,
        );
        let mut history = History::default();
        let selected = choose_proposals(
            &output,
            &options,
            &mut history,
            1e12,
            Instant::now() + Duration::from_secs(10),
            proposals,
        )
        .unwrap()
        .unwrap();
        let sentinel = history
            .evaluations
            .iter()
            .find(|entry| {
                entry.kind == MappedEvaluationKind::HeuristicPricing
                    && if expected_sentinel {
                        entry.estimate == 1e12
                    } else {
                        entry.estimate > 100. && entry.estimate < 1e12
                    }
            })
            .expect("native partial-penalty or unresolved-sentinel history observation");
        let stats = history
            .costs
            .get(&format!("mapped:{}", sentinel.name))
            .unwrap();
        assert!(stats[&sentinel.stage][sentinel.side].count > 0);
        assert!(stats[&sentinel.stage][sentinel.side].mean.is_finite());
        assert_eq!(
            (
                output.snapshot.warm.numerical_bound,
                output.snapshot.warm.certified_bound
            ),
            parent_proofs
        );
        for child in selected.children {
            let result = child.solve(options.node.clone()).unwrap();
            if !proven_infeasible(&result) {
                assert!(result.master.bound.is_finite() && result.master.bound < 1e12);
            }
        }
    }
}

#[test]
fn committed_iteration_limit_records_penalized_lp_once_without_proof_promotion() {
    let problem: Problem<2> = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/packing_quarter_history.json"
    )))
    .unwrap()
    .problem()
    .unwrap();
    let mut packing = PackingBranching::new(
        100.,
        vec![NetworkPackingAssignment::new(
            42,
            vec![(0, 0), (1, 1), (3, 2), (4, 3), (5, 4), (6, 5)],
        )],
    );
    packing
        .prepare(problem.networks.iter().map(|n| (n.id, &n.graph)), &[])
        .unwrap();
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    options.node.preprocessing.enabled = false;
    options.node.gap_enumeration = false;
    options.node.reduced_cost_fixing = false;
    let output = solve(problem, options.node.clone()).unwrap();
    let proposal = packing.proposals(&output, 1e-6).remove(0);
    options.branching_registry.packing = Some(packing);
    options.branching_registry.branching = NetworkBranchingPolicy::Phased;
    options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .phases = vec![StrongPhase {
        candidates: 1,
        subtree_ratio: 0.,
        restricted_master_only: true,
        exact: Some(false),
        ..Default::default()
    }];
    let mut history = History::default();
    let mut selected = choose_proposals(
        &output,
        &options,
        &mut history,
        1e12,
        Instant::now() + Duration::from_secs(10),
        vec![proposal],
    )
    .unwrap()
    .unwrap();
    assert_eq!(history.trials, 0);
    let child = selected.children.pop().unwrap();
    let origin = child.state.branch_history_origin.as_ref().unwrap();
    let (key, side, stage, parent, distance) = (
        origin.key.clone(),
        origin.side,
        origin.stage,
        origin.parent,
        origin.distance,
    );
    options.node.cg.max_iterations = 1;
    let mut solved = child.solve(options.node.clone()).unwrap();
    assert_eq!(solved.master.status, cg::Status::IterationLimit);
    assert!(solved
        .master
        .artificials
        .iter()
        .any(|&column| solved.snapshot.master.values[column].abs() > 1e-6));
    assert!(!proven_infeasible(&solved));
    let proofs = (
        solved.master.bound,
        solved.master.numerical_bound,
        solved.master.certified_bound,
    );
    let estimate = solved.master.lp.min(1e12);
    history
        .record_committed(&mut solved, 1e12, false, options.tolerance)
        .unwrap();
    assert_eq!(history.evaluations.len(), 1);
    assert_eq!(history.evaluations[0].estimate, estimate);
    assert_eq!(history.costs[&key][&stage][side].count, 1);
    assert_eq!(
        history.costs[&key][&stage][side].mean,
        (estimate - parent) / distance
    );
    assert_eq!(
        (
            solved.master.bound,
            solved.master.numerical_bound,
            solved.master.certified_bound
        ),
        proofs
    );
    assert_eq!(solved.master.status, cg::Status::IterationLimit);
    history
        .record_committed(&mut solved, 1e12, false, options.tolerance)
        .unwrap();
    assert_eq!(history.evaluations.len(), 1);
}

#[test]
fn committed_pre_lp_timeout_records_unavailable_lp_sentinel_without_proof_promotion() {
    let problem: Problem<2> = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/packing_quarter_history.json"
    )))
    .unwrap()
    .problem()
    .unwrap();
    let mut packing = PackingBranching::new(
        100.,
        vec![NetworkPackingAssignment::new(
            42,
            vec![(0, 0), (1, 1), (3, 2), (4, 3), (5, 4), (6, 5)],
        )],
    );
    packing
        .prepare(problem.networks.iter().map(|n| (n.id, &n.graph)), &[])
        .unwrap();
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    options.node.preprocessing.enabled = false;
    options.node.gap_enumeration = false;
    options.node.reduced_cost_fixing = false;
    let output = solve(problem, options.node.clone()).unwrap();
    let proposal = packing.proposals(&output, 1e-6).remove(0);
    options.branching_registry.packing = Some(packing);
    options.branching_registry.branching = NetworkBranchingPolicy::Phased;
    options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .phases = vec![StrongPhase {
        candidates: 1,
        subtree_ratio: 0.,
        restricted_master_only: true,
        exact: Some(false),
        ..Default::default()
    }];
    let mut history = History::default();
    let mut selected = choose_proposals(
        &output,
        &options,
        &mut history,
        1e12,
        Instant::now() + Duration::from_secs(10),
        vec![proposal],
    )
    .unwrap()
    .unwrap();
    assert_eq!(history.trials, 0);
    let child = selected.children.pop().unwrap();
    let origin = child.state.branch_history_origin.as_ref().unwrap();
    let (key, side, stage, parent, distance) = (
        origin.key.clone(),
        origin.side,
        origin.stage,
        origin.parent,
        origin.distance,
    );
    options.node.cg.time_limit = Duration::ZERO;
    let mut solved = child.solve(options.node.clone()).unwrap();
    assert_eq!(solved.master.status, cg::Status::TimeLimit);
    assert_eq!(solved.master.iterations, 0);
    assert_eq!(solved.master.lp, 0.);
    assert_eq!(solved.master.pricing_calls, 0);
    assert!(!proven_infeasible(&solved));
    let proofs = (
        solved.master.bound,
        solved.master.numerical_bound,
        solved.master.certified_bound,
    );
    let estimate = 1e12;
    history
        .record_committed(&mut solved, 1e12, false, options.tolerance)
        .unwrap();
    assert_eq!(history.evaluations.len(), 1);
    assert_eq!(history.evaluations[0].estimate, estimate);
    assert_eq!(history.costs[&key][&stage][side].count, 1);
    assert_eq!((parent, distance), (1.4375, 0.75));
    assert_eq!(
        history.costs[&key][&stage][side].mean.to_bits(),
        1333333333331.4167f64.to_bits()
    );
    assert_eq!(solved.master.lp, 0.);
    assert_eq!(
        history.costs[&key][&stage][side].mean,
        (estimate - parent) / distance
    );
    assert_eq!(
        (
            solved.master.bound,
            solved.master.numerical_bound,
            solved.master.certified_bound
        ),
        proofs
    );
    assert_eq!(solved.master.status, cg::Status::TimeLimit);
    history
        .record_committed(&mut solved, 1e12, false, options.tolerance)
        .unwrap();
    assert_eq!(history.evaluations.len(), 1);
}

#[test]
fn committed_timeout_after_partial_lp_records_sentinel_instead_of_penalty_objective() {
    // Native deadline trace: one LP iteration reports 801.9167, but the
    // interrupted node records its unavailable-LP sentinel in ordinary history.
    let mut output = root();
    output.master.status = cg::Status::TimeLimit;
    output.master.iterations = 1;
    output.master.lp = 801.9166666666666;
    output.snapshot.state.branch_history_origin = Some(BranchHistoryOrigin {
        key: "timeout".into(),
        name: "timeout".into(),
        depth: 0,
        side: 1,
        stage: 1,
        parent: 1.4375,
        distance: 0.75,
    });
    let proofs = (
        output.master.bound,
        output.master.numerical_bound,
        output.master.certified_bound,
    );
    let mut history = History::default();
    history
        .record_committed(&mut output, 1e12, false, 1e-6)
        .unwrap();
    assert_eq!(history.evaluations.len(), 1);
    assert_eq!(history.evaluations[0].estimate, 1e12);
    assert_eq!(history.costs["timeout"][&1][1].count, 1);
    assert_eq!(
        history.costs["timeout"][&1][1].mean.to_bits(),
        1333333333331.4167f64.to_bits()
    );
    assert_eq!(output.master.status, cg::Status::TimeLimit);
    assert_eq!(output.master.lp, 801.9166666666666);
    assert_eq!(
        (
            output.master.bound,
            output.master.numerical_bound,
            output.master.certified_bound
        ),
        proofs
    );
    history
        .record_committed(&mut output, 1e12, false, 1e-6)
        .unwrap();
    assert_eq!(history.evaluations.len(), 1);
}

#[test]
fn priced_strong_phase_clamps_floor_to_last_available_stage() {
    for count in [1, 2, 3] {
        let mut output = root();
        for oracle in &mut output.snapshot.state.oracles {
            oracle.solver.phases.truncate(count);
            assert_eq!(oracle.solver.phases.len(), count);
        }
        let parent_schedules: Vec<_> = output
            .snapshot
            .state
            .oracles
            .iter()
            .map(|oracle| oracle.solver.phases.clone())
            .collect();
        let parent_bound = output.snapshot.warm.bound;
        let mut reference = None;
        for requested in [count - 1, count, 99] {
            let mut options = strong_tree_options();
            options.node.cg.stabilize = false;
            options.branching_registry.mapped_variables =
                vec![mapping("screen-a", 42), mapping("screen-b", 42)];
            options
                .branching_registry
                .strong_branching
                .as_mut()
                .unwrap()
                .phases = vec![StrongPhase {
                candidates: 2,
                subtree_ratio: 0.,
                pricing_phase_floor: requested,
                max_iterations: 2,
                exact: Some(false),
                tree_depth_score: false,
                reduced_cost_fixing: false,
                max_cut_rounds: 0,
                ..Default::default()
            }];
            let mut history = History::default();
            let selected = choose_registered(
                &output,
                &options,
                &mut history,
                2.,
                Instant::now() + Duration::from_secs(10),
            )
            .unwrap()
            .unwrap();
            assert!(history.trials > 0);
            assert!(
                !history.evaluations.is_empty(),
                "nonexact floor-zero trials still record history"
            );
            assert!(selected.evaluated.iter().all(Option::is_none));
            let observed = (
                selected
                    .children
                    .iter()
                    .map(|child| {
                        child
                            .state
                            .oracles
                            .iter()
                            .map(|oracle| oracle.solver.phases.clone())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
                history.trials,
                history.pricing_calls,
                history.iterations,
                history
                    .evaluations
                    .iter()
                    .map(|e| (e.name.clone(), e.side, e.stage, e.estimate.to_bits()))
                    .collect::<Vec<_>>(),
                selected
                    .children
                    .iter()
                    .map(|child| {
                        (
                            child.warm.numerical_bound.to_bits(),
                            child.warm.certified_bound.to_bits(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            if let Some(expected) = &reference {
                assert_eq!(&observed, expected);
            } else {
                reference = Some(observed);
            }
        }
        assert_eq!(output.snapshot.warm.bound, parent_bound);
        assert_eq!(
            output
                .snapshot
                .state
                .oracles
                .iter()
                .map(|o| o.solver.phases.clone())
                .collect::<Vec<_>>(),
            parent_schedules
        );
    }
}

#[test]
fn exact_iteration_limited_packing_trial_records_penalty_without_reusable_output() {
    let problem: Problem<2> = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/packing_quarter_history.json"
    )))
    .unwrap()
    .problem()
    .unwrap();
    let mut packing = PackingBranching::new(
        100.,
        vec![NetworkPackingAssignment::new(
            42,
            vec![(0, 0), (1, 1), (3, 2), (4, 3), (5, 4), (6, 5)],
        )],
    );
    packing
        .prepare(problem.networks.iter().map(|n| (n.id, &n.graph)), &[])
        .unwrap();
    let mut options = strong_tree_options();
    options.node.cg.stabilize = false;
    options.node.preprocessing.enabled = false;
    options.node.gap_enumeration = false;
    options.node.reduced_cost_fixing = false;
    let output = solve(problem, options.node.clone()).unwrap();
    let proposal = packing.proposals(&output, 1e-6).remove(0);
    let name = proposal.name.clone();
    options.branching_registry.packing = Some(packing);
    options.branching_registry.branching = NetworkBranchingPolicy::Phased;
    options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap()
        .phases = vec![StrongPhase {
        candidates: 1,
        subtree_ratio: 0.,
        max_iterations: 1,
        exact: Some(true),
        reduced_cost_fixing: false,
        max_cut_rounds: 0,
        ..Default::default()
    }];
    let parent_proofs = (
        output.snapshot.warm.numerical_bound,
        output.snapshot.warm.certified_bound,
    );
    let mut history = History::default();
    let selected = choose_proposals(
        &output,
        &options,
        &mut history,
        1e12,
        Instant::now() + Duration::from_secs(10),
        vec![proposal],
    )
    .unwrap()
    .unwrap();
    let partial = history
        .evaluations
        .iter()
        .find(|entry| {
            entry.name == name
                && entry.kind == MappedEvaluationKind::ExactPricing
                && entry.estimate > 100.
                && entry.estimate < 1e12
        })
        .expect("native exact treatment records its finite partial penalty LP");
    assert_eq!(partial.side, 1);
    assert!(
        selected.evaluated[partial.side].is_none(),
        "history does not authorize completed-output reuse"
    );
    assert!(history.costs[&format!("mapped:{name}")][&partial.stage][partial.side].count > 0);
    let child = &selected.children[partial.side];
    assert!(child
        .warm
        .artificials
        .iter()
        .any(|&c| child.master.values[c].abs() > 1e-6));
    assert!(child.warm.numerical_bound < partial.estimate);
    assert_eq!(
        (
            output.snapshot.warm.numerical_bound,
            output.snapshot.warm.certified_bound
        ),
        parent_proofs
    );
    let resumed = child.solve(options.node.clone()).unwrap();
    assert!(proven_infeasible(&resumed));
}

#[test]
fn strong_timeouts_record_sentinel_without_iterations_or_completed_output() {
    for exact in [false, true] {
        let output = root();
        let parent_proofs = (
            output.snapshot.warm.numerical_bound,
            output.snapshot.warm.certified_bound,
            output.snapshot.warm.bound,
        );
        let mut options = strong_tree_options();
        options.node.cg.stabilize = false;
        options.branching_registry.branching = NetworkBranchingPolicy::Phased;
        options.branching_registry.mapped_variables =
            vec![mapping("timeout-a", 42), mapping("timeout-b", 42)];
        let config = options
            .branching_registry
            .strong_branching
            .as_mut()
            .unwrap();
        config.trial_time_limit = Duration::ZERO;
        config.phases = vec![StrongPhase {
            candidates: 2,
            subtree_ratio: 0.,
            max_iterations: 10000,
            exact: Some(exact),
            reduced_cost_fixing: false,
            max_cut_rounds: 0,
            ..Default::default()
        }];
        let mut history = History::default();
        let selected = choose_registered(
            &output,
            &options,
            &mut history,
            1e12,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap()
        .unwrap();
        assert!(history.trials >= 2 && history.trials % 2 == 0);
        assert_eq!(history.iterations, 0);
        assert_eq!(history.evaluations.len(), history.trials);
        let names: std::collections::BTreeSet<_> = history
            .evaluations
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        for name in names {
            assert!(["timeout-a", "timeout-b"].contains(&name));
            for side in 0..2 {
                let entry = history
                    .evaluations
                    .iter()
                    .find(|entry| entry.name == name && entry.side == side)
                    .unwrap();
                assert_eq!(entry.stage, 1);
                assert_eq!(entry.estimate, 1e12);
                assert_eq!(
                    entry.kind,
                    if exact {
                        MappedEvaluationKind::ExactPricing
                    } else {
                        MappedEvaluationKind::HeuristicPricing
                    }
                );
                assert_eq!(history.costs[&format!("mapped:{name}")][&1][side].count, 1);
            }
        }
        assert!(selected.evaluated.iter().all(Option::is_none));
        assert!(history.incumbent.is_none());
        assert_eq!(
            (
                output.snapshot.warm.numerical_bound,
                output.snapshot.warm.certified_bound,
                output.snapshot.warm.bound,
            ),
            parent_proofs
        );
    }
}

#[test]
fn restricted_zero_budget_trials_record_sentinels_without_proof_or_output_reuse() {
    let output = root();
    let mut options = strong_tree_options();
    options.branching_registry.branching = NetworkBranchingPolicy::Phased;
    options.branching_registry.mapped_variables =
        vec![mapping("screen-a", 42), mapping("screen-b", 42)];
    let config = options
        .branching_registry
        .strong_branching
        .as_mut()
        .unwrap();
    config.trial_time_limit = Duration::ZERO;
    config.phases = vec![StrongPhase {
        candidates: 2,
        subtree_ratio: 0.,
        max_iterations: 0,
        restricted_master_only: true,
        exact: Some(false),
        ..Default::default()
    }];
    let mut history = History::default();
    let selected = choose_registered(
        &output,
        &options,
        &mut history,
        1e12,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap()
    .unwrap();
    assert!(history.trials >= 2 && history.trials % 2 == 0);
    assert_eq!(history.processed_trials, 0);
    assert_eq!(history.iterations, 0);
    assert_eq!(history.pricing_calls, 0);
    assert_eq!(history.evaluations.len(), history.trials);
    assert!(history
        .evaluations
        .iter()
        .all(|entry| entry.kind == MappedEvaluationKind::RestrictedMaster
            && entry.estimate == 1e12
            && entry.stage == 1));
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
    }
}
