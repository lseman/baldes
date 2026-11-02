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
fn parallel_batches_prove_the_sequential_optimum() {
    let solve = |threads: usize| {
        let problem = rc101(25);
        let mut configured = ModelTreeOptions::default();
        configured.branching_registry.mapped_variables = problem.arc_flow_variables(1.);
        configured.branching_registry.strong_branching = None;
        configured.node_limit = 2000;
        configured.threads = threads;
        crate::master::tree::solve(problem, configured).unwrap()
    };
    let sequential = solve(1);
    let parallel = solve(4);
    assert_eq!(sequential.status, crate::master::tree::TreeStatus::Optimal);
    assert_eq!(parallel.status, crate::master::tree::TreeStatus::Optimal);
    assert!(parallel.nodes > 1);
    assert_eq!(sequential.parallel_batches, 0);
    assert!(parallel.parallel_batches > 0);
    assert!(parallel.parallel_batches * 2 <= parallel.nodes);
    let report = crate::master::model::format::SolveResult::tree(&parallel, 0.);
    assert_eq!(report.parallel_batches, parallel.parallel_batches);
    assert!(
        (sequential.incumbent - parallel.incumbent).abs() < 1e-6,
        "{} {}",
        sequential.incumbent,
        parallel.incumbent
    );
    assert!((parallel.bound - parallel.incumbent).abs() < 1e-6);
    let mut invalid = ModelTreeOptions::default();
    invalid.threads = 0;
    assert!(crate::master::tree::solve(rc101(3), invalid).is_err());
}

#[test]
fn transferred_snapshots_resolve_to_the_same_lp() {
    let output = crate::master::model::solve(rc101(10), Options::default()).unwrap();
    let workers = workers(2).unwrap().unwrap();
    let results = evaluate(
        &workers,
        &[&output.snapshot, &output.snapshot],
        &Options::default(),
    )
    .unwrap();
    for result in results {
        let resolved = result.unwrap();
        assert!((resolved.master.lp - output.master.lp).abs() < 1e-6);
        // The restored output is usable by the coordinator.
        resolved.snapshot.fork().unwrap();
    }
}
