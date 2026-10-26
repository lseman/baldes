use super::*;
use crate::master::{
    preprocessing::Bounds,
    tree::{StrongBranchingOptions, StrongPhase},
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
            arcs: vec![],
            per_route: 1.,
        }],
        masters: vec![],
    }
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
fn mapped_native_phase_skips_do_not_consume_stage_or_publish_trial_bounds() {
    let output = root();
    let before_rows = output.snapshot.state.shared_rows.len();
    let before_columns = output.snapshot.master.costs.len();
    let mut options = AssignmentTreeOptions::default();
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
    for child in children {
        assert_eq!(child.warm.bound, output.snapshot.warm.bound);
        assert_eq!(child.state.shared_rows.len(), before_rows + 1);
        assert!(child.master.costs.len() >= before_columns);
    }
    assert_eq!(output.snapshot.state.shared_rows.len(), before_rows);
    assert_eq!(output.snapshot.master.costs.len(), before_columns);
}

#[test]
fn mapped_native_subtree_budget_limits_actual_restricted_trials() {
    let mut output = root();
    output.snapshot.state.mapped_subtree_size = 2.;
    output.snapshot.state.mapped_subtree_priority = 1.;
    let mut options = AssignmentTreeOptions::default();
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
