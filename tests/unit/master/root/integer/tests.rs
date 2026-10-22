use super::*;
use crate::master::solvers::mip::MipStatus;

#[test]
fn incumbent_witness_omitted_by_the_cutoff_survives_enumerated_mip() {
    let mut graph = Graph::<5>::new(3, 5).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.enumerate = false;
    let output = solve(RootProblem::set_partitioning(graph, 1.).unwrap(), options).unwrap();
    assert_eq!(output.master.lp, 2.);
    let mut node = Rc::try_unwrap(output.snapshot.unwrap()).ok().unwrap();
    assert_eq!(node.incumbent, 2.);
    // The only original integer route is the validated incumbent. Strict
    // improvement enumeration can omit it while covering every improvement.
    node.pricing.enumerate(100.).unwrap();
    node.pricing
        .enumeration
        .as_mut()
        .unwrap()
        .paths
        .retain(|_| false);
    assert!(node.pricing.enumeration.as_ref().unwrap().completed);
    let costs = node.master.costs.clone();
    let values = node.master.values.clone();
    let mip = node
        .solve_enumerated_mip(&MipOptions::default())
        .unwrap()
        .unwrap();
    assert_eq!(mip.mip.status, MipStatus::Optimal);
    assert_eq!(mip.mip.solution.unwrap().objective, 2.);
    assert_eq!(mip.coverage_cutoff, 2.);
    assert_eq!(mip.route_values.len(), 1);
    assert_eq!(mip.route_values[0].0.arcs, [0, 1]);
    assert_eq!(mip.route_values[0].1, 1.);
    assert_eq!(node.master.costs, costs);
    assert_eq!(node.master.values, values);
    assert_eq!(node.pricing.enumeration.as_ref().unwrap().paths.len, 0);
}
