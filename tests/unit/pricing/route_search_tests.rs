use super::*;
use crate::extensions::stochastic_restocking::{Demand, Problem, RestockingResource};
use crate::graph::coupled::{CoupledResource, PluginResource};

#[test]
fn route_search_replays_true_recourse_cuts_and_forbidden_arcs_without_publishing_a_minimum() {
    let p = Problem::new(
        4,
        vec![Demand::new(vec![(0, 0.5), (4, 0.5)]).unwrap(); 2],
        vec![vec![0., 3., 1.], vec![3., 0., 2.], vec![1., 2., 0.]],
    )
    .unwrap();
    let mut solver = Solver::<2>::new(4, 0).unwrap();
    let mut endpoints = vec![];
    for (i, j) in [(0, 1), (1, 3), (0, 2), (2, 3), (1, 2), (2, 1)] {
        let model_j = if j == 3 { 0 } else { j };
        let id = solver.graph.add_arc(i, j, p.costs()[i][model_j]).unwrap();
        solver.graph.arc_mut(id).unwrap().reduced_cost -= if j == 2 { 4. } else { 0. };
        endpoints.push((i, model_j));
    }
    solver
        .set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(
            RestockingResource::new(p, endpoints).unwrap(),
        ))])
        .unwrap();
    solver.heuristic_seeds = vec![vec![1]];
    let cut = solver.cuts.add_strong_k_path_cut(4, &[2], 1, -3.).unwrap();
    assert!(solver
        .search_route_columns(20, Duration::from_secs(1))
        .is_none());
    solver.cuts.cuts[cut].dual = -0.5;
    let result = solver
        .search_route_columns(20, Duration::from_secs(1))
        .unwrap();
    assert!(!result.exact);
    assert_eq!(result.minimum_reduced_cost, 0.);
    assert_eq!(result.columns[0].nodes, vec![0, 2, 3]);
    assert_eq!(result.columns[0].cost, -1.5);
    assert!(solver.staged_minimum.is_none());
    solver.set_forbidden_arcs(vec![2, 4]).unwrap();
    assert!(solver
        .search_route_columns(20, Duration::from_secs(1))
        .is_none());
}

#[test]
fn completed_scalar_sweep_retains_a_bound_without_claiming_bellman_completion() {
    let p = Problem::new(
        4,
        vec![Demand::new(vec![(0, 0.5), (4, 0.5)]).unwrap(); 2],
        vec![vec![0., 3., 1.], vec![3., 0., 2.], vec![1., 2., 0.]],
    )
    .unwrap();
    let mut solver = Solver::<2>::new(4, 0).unwrap();
    let mut endpoints = vec![];
    for (i, j) in [(0, 1), (1, 3), (0, 2), (2, 3), (1, 2), (2, 1)] {
        let model_j = if j == 3 { 0 } else { j };
        solver.graph.add_arc(i, j, p.costs()[i][model_j]).unwrap();
        endpoints.push((i, model_j));
    }
    solver
        .set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(
            RestockingResource::new(p, endpoints).unwrap(),
        ))])
        .unwrap();
    assert!(solver.scalar_preprice(20, Duration::from_secs(1)).is_none());
    let bound = solver.staged_minimum.unwrap();
    assert!(bound > 1.999999 && bound <= 2.);
    solver.staged_minimum = None;
    assert!(solver.scalar_preprice(20, Duration::ZERO).is_none());
    assert!(solver.staged_minimum.is_none());
}
