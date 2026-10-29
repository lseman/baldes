use super::*;
use crate::graph::coupled::{
    CoupledResource, FrontOperationPolicy, PairedCostNormalization, PairedResourceMode,
    PreparedSlopeFunction, SlopeRelationArc,
};

fn problem(customers: usize, capacity: f64) -> (RoutingContext<2>, ClpMaster, Vec<f64>) {
    let sink = customers + 1;
    let mut graph = Graph::<2>::new(sink + 1, 0).unwrap();
    let function = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, capacity)]).unwrap()
    };
    let mut arcs = vec![];
    for tail in 0..sink {
        for head in 1..=sink {
            if tail == head || (tail == 0 && head == sink) {
                continue;
            }
            graph.add_arc(tail, head, 0.).unwrap();
            arcs.push(SlopeRelationArc {
                arrival: function(if head == sink { 0. } else { 1. }, 1.),
                increment: function(0., 0.),
                lower: function(0., 0.),
                upper: function(8., 0.),
                waiting: 0,
            });
        }
    }
    graph
        .set_coupled_resources(&[CoupledResource::SlopePaired {
            initial: function(0., 0.),
            initial_base: crate::graph::Window::new(0., capacity).unwrap(),
            arcs,
            mode: PairedResourceMode::CostUnbounded,
            normalization: PairedCostNormalization::AbsoluteFinalValue,
            policy: FrontOperationPolicy::FrontAlgebra,
        }])
        .unwrap();
    graph.set_front_program_domain(true).unwrap();
    graph.suppress_coupled_costs();
    let problem = RoutingContext::set_partitioning(graph, 1.).unwrap();
    let mut master = ClpMaster::new().unwrap();
    for row in &problem.model.rows {
        master.add_row(row.sense, row.rhs, &[]).unwrap();
    }
    let mut duals = vec![0.5; customers + 1];
    duals[0] = -1.;
    (problem, master, duals)
}

#[test]
fn coupled_zero_objective_proof_distinguishes_infeasible_and_feasible_fleet_limits() {
    let (problem, mut master, mut duals) = problem(3, 2.5);
    let deadline = || Instant::now() + Duration::from_secs(2);
    assert_eq!(
        zero_objective_lower_bound(&problem, &master, &duals, deadline()),
        Some(0.5)
    );
    // Dynamic cut rows are not assumed valid or assigned fabricated path
    // coefficients. Even an extreme dual on such a row cannot enter the proof.
    master.add_row(RowSense::Equal, -100., &[]).unwrap();
    duals.push(-1e6);
    assert_eq!(
        zero_objective_lower_bound(&problem, &master, &duals, deadline()),
        Some(0.5)
    );
    let (feasible, master, duals) = self::problem(3, 3.5);
    assert!(zero_objective_lower_bound(&feasible, &master, &duals, deadline()).unwrap() <= 0.);
}

#[test]
fn coupled_phase_one_proof_keeps_costs_deadlines_and_incomplete_domains_guarded() {
    let (mut problem, master, duals) = problem(3, 2.5);
    assert!(zero_objective_lower_bound(&problem, &master, &duals, Instant::now()).is_none());
    let mut original_objective = problem.clone();
    let definitions = original_objective.model.networks[0]
        .graph
        .coupled_kernel()
        .unwrap()
        .definitions()
        .to_vec();
    original_objective.model.networks[0]
        .graph
        .set_coupled_resources(&definitions)
        .unwrap();
    assert!(original_objective.model.networks[0]
        .graph
        .coupled_kernel()
        .unwrap()
        .has_costs());
    assert!(zero_objective_lower_bound(
        &original_objective,
        &master,
        &duals,
        Instant::now() + Duration::from_secs(2)
    )
    .is_none());
    problem.model.networks[0].graph.arcs[0].cost = 1.;
    assert!(zero_objective_lower_bound(
        &problem,
        &master,
        &duals,
        Instant::now() + Duration::from_secs(2)
    )
    .is_none());
    let (large, master, duals) = self::problem(9, 2.5);
    assert!(zero_objective_lower_bound(
        &large,
        &master,
        &duals,
        Instant::now() + Duration::from_secs(2)
    )
    .is_none());
}
