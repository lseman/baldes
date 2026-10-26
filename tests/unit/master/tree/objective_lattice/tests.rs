use super::*;
fn problem(cost: f64) -> RoutingContext<2> {
    let mut graph = Graph::new(3, 1).unwrap();
    graph.add_arc(0, 1, cost).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    RoutingContext::set_partitioning(graph, 1.).unwrap()
}
use crate::Graph;
#[test]
fn decimal_lattice_closes_small_gaps_without_exceeding_arc_costs() {
    let problem = problem(220.4);
    let lattice = ObjectiveLattice::prepare(&problem).unwrap();
    for bound in [220.39999796, 220.4, 220.30000001, 220.3, 220.29999999] {
        let tightened = lattice.strengthen(bound);
        assert!(tightened >= bound && tightened <= 220.4);
    }
    assert!(lattice.strengthen(220.39999796) > 220.4 - 1e-9);
    assert!(lattice.strengthen(220.3) <= 220.3);
}
#[test]
fn arbitrary_costs_and_empty_routes_decline_lattice_strengthening() {
    assert!(ObjectiveLattice::prepare(&problem(2f64.sqrt())).is_none());
    let mut problem = problem(0.1);
    problem.model.networks[0].graph.add_arc(0, 2, 0.).unwrap();
    assert!(ObjectiveLattice::prepare(&problem).is_none());
}

#[test]
fn nearby_binary_costs_keep_their_error_in_the_strengthened_bound() {
    for cost in [0.1 - 1e-10, 0.1, 0.1 + 1e-10, 220.4 - 1e-10, 220.4 + 1e-10] {
        let lattice = ObjectiveLattice::prepare(&problem(cost)).unwrap();
        for bound in [cost, cost - 1e-6, cost - 0.1] {
            assert!(lattice.strengthen(bound) <= cost);
        }
    }
}
