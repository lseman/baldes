use super::*;
use crate::inputs::{Instance, Rounding};

fn rc101(customers: usize) -> Problem {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut instance = Instance::read(format!("{root}/instances/Solomon/RC101.txt")).unwrap();
    instance.truncate(customers).unwrap();
    let graph = instance.graph(8, Rounding::None).unwrap();
    Problem::set_partitioning(graph, customers as f64).unwrap()
}

#[test]
fn dives_follow_the_native_depth_limit_and_frequency() {
    let root_only = DivingOptions::default();
    assert!(root_only.due(0, 1));
    assert!(!root_only.due(1, 2));
    let periodic = DivingOptions {
        depth_limit: 3,
        frequency: 4,
        ..Default::default()
    };
    assert!(periodic.due(0, 1));
    assert!(!periodic.due(2, 3));
    assert!(periodic.due(2, 5));
    assert!(!periodic.due(4, 5));
    assert!(DivingOptions {
        strong_candidates: 0,
        ..Default::default()
    }
    .validate()
    .is_err());
}

fn root_tree(diving: Option<DivingOptions>) -> ModelTreeOutput {
    let problem = rc101(25);
    let mut configured = ModelTreeOptions::default();
    configured.mapped_variables = problem.arc_flow_variables(1.);
    configured.mapped_strong_branching = None;
    configured.node_limit = 1;
    configured.diving = diving;
    crate::master::tree::solve(problem, configured).unwrap()
}

#[test]
fn root_dive_supplies_a_valid_incumbent_without_changing_the_bound() {
    let plain = root_tree(None);
    assert!(
        plain.incumbent.is_infinite(),
        "the RC101 root is fractional"
    );
    for diving in [
        DivingOptions::default(),
        DivingOptions {
            strong_candidates: 3,
            lds_depth: 2,
            max_discrepancy: 1,
            ..Default::default()
        },
    ] {
        let dived = root_tree(Some(diving));
        assert!(dived.diving_nodes > 0);
        assert!(dived.incumbent.is_finite(), "{dived:?}");
        assert_eq!(dived.bound, plain.bound);
        assert_eq!(dived.nodes, plain.nodes);
        let replayed: f64 = dived
            .routes
            .iter()
            .map(|r| r.quantity * r.route.route.original_cost)
            .sum();
        assert!((replayed - dived.incumbent).abs() < 1e-6);
    }
}

#[test]
fn diving_tree_reaches_the_default_optimum() {
    let solve = |diving: Option<DivingOptions>| {
        let problem = rc101(25);
        let mut configured = ModelTreeOptions::default();
        configured.mapped_variables = problem.arc_flow_variables(1.);
        configured.node_limit = 2000;
        configured.diving = diving;
        crate::master::tree::solve(problem, configured).unwrap()
    };
    let plain = solve(None);
    let dived = solve(Some(DivingOptions {
        depth_limit: usize::MAX,
        frequency: 5,
        strong_candidates: 2,
        lds_depth: 3,
        max_discrepancy: 1,
        stop_after_first_solution: true,
        ..Default::default()
    }));
    assert_eq!(plain.status, crate::master::tree::TreeStatus::Optimal);
    assert_eq!(dived.status, crate::master::tree::TreeStatus::Optimal);
    assert!(dived.diving_nodes > 0);
    assert!(
        (dived.incumbent - plain.incumbent).abs() < 1e-6,
        "{} {}",
        dived.incumbent,
        plain.incumbent
    );
    // Root dive incumbents never undercut the proved optimum.
    let root = root_tree(Some(DivingOptions::default()));
    assert!(root.incumbent >= plain.incumbent - 1e-6);
}
