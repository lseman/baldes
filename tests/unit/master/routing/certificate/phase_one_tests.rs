use super::*;
use crate::graph::coupled::descriptor::{
    FrontierBudget, FrontierOutcome, MachineRootInput, RawAffine,
};
use crate::graph::coupled::{
    CoupledResource, MachineArcFunctionInput, PairedResourceMode, PreparedSlopeFunction,
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
            arcs.push(function(if head == sink { 0. } else { 1. }, 0.));
        }
    }
    graph
        .set_coupled_resources(&[CoupledResource::PairedFront {
            mode: PairedResourceMode::CostUnbounded,
        }])
        .unwrap();
    let bounds = crate::graph::Window::new(0., capacity).unwrap();
    let affine = |value| MachineRootInput::Affine(RawAffine::new(value, 0.).unwrap());
    let inputs: Vec<_> = arcs
        .iter()
        .enumerate()
        .map(|(physical_arc, base)| MachineArcFunctionInput {
            physical_arc,
            tail_bounds: bounds,
            waiting_bounds: bounds,
            waiting_rate: 0,
            base,
            dependent: affine(0.),
            dependent_lower: affine(0.),
            dependent_upper: affine(8.),
        })
        .collect();
    let result = graph.prepare_paired_front_program(
        0,
        &inputs,
        affine(0.),
        bounds,
        bounds,
        FrontierBudget::default(),
    );
    assert!(
        matches!(result, FrontierOutcome::Complete(())),
        "{result:?}"
    );
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
