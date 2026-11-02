use super::*;
use crate::master::preprocessing::Bounds;

fn child() -> NodeSnapshot<2> {
    let problem = format::Model::from_json(include_str!("objective_cutoff_model.json"))
        .unwrap()
        .problem::<2>()
        .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 1000.;
    solve(problem, options)
        .unwrap()
        .snapshot
        .with_multiplicity(
            42,
            Bounds {
                lower: 0.,
                upper: 1.,
            },
        )
        .unwrap()
}

#[test]
fn external_objective_cutoff_keeps_artificial_solution_out_of_primal_publication() {
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.bound_mode = mode;
        options.cg.artificial_cost = 1000.;
        options.integer_primal_termination = true;
        options.objective_cutoff = Some(2.75);
        options.reduced_cost_fixing = false;
        let result = child().solve(options.clone()).unwrap();
        assert_eq!(result.master.status, cg::Status::Optimal);
        assert!(result.master.pruned_by_bound);
        assert_eq!(result.master.bound, 2.75);
        assert_eq!(result.master.pricing_calls, 3);
        assert_eq!(result.master.penalty_updates, 0);
        assert!(result.master.feasibility.is_empty());
        assert!(!result.snapshot.state.phase_one_infeasible);
        assert_eq!(result.snapshot.state.tree_incumbent, f64::INFINITY);
        assert!(result.snapshot.integer_witness(1e-6).unwrap().is_none());
        assert!(result
            .master
            .artificials
            .iter()
            .any(|&column| result.snapshot.master.values()[column] > 1e-6));
        // A fresh request on the retained snapshot must not inherit the old
        // external cutoff when the caller no longer supplies it.
        options.objective_cutoff = None;
        let resumed = result.snapshot.solve(options).unwrap();
        assert!(!resumed.master.pruned_by_bound);
        assert_eq!(resumed.master.status, cg::Status::Infeasible);
    }
}

#[test]
fn objective_cutoff_requires_finite_value() {
    for cutoff in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut options = Options::default();
        options.objective_cutoff = Some(cutoff);
        assert!(solve::validate_options(&options).is_err());
    }
}

#[test]
fn objective_cutoff_does_not_change_an_ordinary_lp_request() {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 1000.;
    options.objective_cutoff = Some(2.75);
    assert!(!options.integer_primal_termination);
    let result = child().solve(options).unwrap();
    assert!(!result.master.pruned_by_bound);
    assert_eq!(result.master.status, cg::Status::Infeasible);
    assert!(result.master.pricing_calls > 3);
}

#[test]
fn objective_cutoff_trial_is_retained_without_duplicate_processed_node_accounting() {
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let problem = format::Model::from_json(include_str!("objective_cutoff_model.json"))
            .unwrap()
            .problem::<2>()
            .unwrap();
        let mut options = crate::master::tree::ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.node.cg.bound_mode = mode;
        options.node.cg.artificial_cost = 1000.;
        options.branching_registry.branching = crate::master::tree::NetworkBranchingPolicy::Phased;
        options.branching_registry.strong_branching = Some(Default::default());
        let output = crate::master::tree::solve(problem, options).unwrap();
        assert_eq!(output.status, crate::master::tree::TreeStatus::Optimal);
        assert_eq!(output.incumbent, 2.75);
        assert_eq!(output.bound, 2.75);
        assert_eq!(output.strong_branching_trials, 2);
        assert_eq!(output.pricing_calls, 10);
        assert_eq!(output.processed_nodes, 3);
        assert_eq!(output.mapped_branch_evaluations.len(), 1);
        assert_eq!(output.mapped_branch_evaluations[0].side, 1);
    }
}
