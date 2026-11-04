use super::*;
use crate::master::model::{self, Options, Problem};
use std::time::Duration;

fn fixture() -> model::Output<2> {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., 3.).unwrap();
    }
    for (tail, head, q) in [(0, 1, 1.), (1, 2, 1.), (2, 1, 1.), (1, 3, 0.), (2, 3, 0.)] {
        let id = graph.add_arc(tail, head, 0.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = q;
    }
    let mut options = Options::default();
    options.cg.time_limit = Duration::ZERO;
    model::solve(Problem::set_partitioning(graph, 2.).unwrap(), options).unwrap()
}

#[test]
fn bounded_walks_keep_repeated_vertices_and_enforce_arc_windows() {
    let output = fixture();
    let oracle = &output.snapshot.state.oracles[0];
    let mut graph = oracle.network.graph.as_ref().clone();
    let weights = [0., 0., -5., 0., 0.];
    let run = |graph: &Graph<2>| {
        minimum(
            oracle,
            graph,
            &weights,
            0,
            &[],
            &mut 100_000,
            Instant::now() + Duration::from_secs(1),
        )
    };
    // The cheapest permitted walk repeats vertex 1. An elementary-path proof
    // would incorrectly publish zero as a lower bound on this walk domain.
    assert_eq!(run(&graph), Some(Some(-5.)));
    graph.edit().arcs[2].bounds[0] = Some(crate::Window::new(0., 2.).unwrap());
    assert_eq!(run(&graph), Some(Some(0.)));
}

#[test]
fn bounded_walks_decline_partial_coverage_and_expired_deadlines() {
    let output = fixture();
    let oracle = &output.snapshot.state.oracles[0];
    let graph = &oracle.network.graph;
    let weights = [0., 0., -5., 0., 0.];
    assert_eq!(
        minimum(
            oracle,
            graph,
            &weights,
            0,
            &[],
            &mut 5,
            Instant::now() + Duration::from_secs(1)
        ),
        None
    );
    assert_eq!(
        minimum(
            oracle,
            graph,
            &weights,
            0,
            &[],
            &mut 100_000,
            Instant::now()
        ),
        None
    );
}

#[test]
fn bounded_walks_charge_registered_penalties_and_cover_both_memory_encodings() {
    let output = fixture();
    let mut node = output
        .snapshot
        .with_rank_one_row(
            crate::master::formulation::Row {
                sense: RowSense::LessEqual,
                rhs: 1.,
            },
            &[model::RankOneCoefficients {
                network: 0,
                weights: vec![0, 1, 0, 0],
                denominator: 2,
            }],
        )
        .unwrap();
    let offset = node.state.shared_offset;
    let rows = offset + node.state.shared_rows.len();
    let run = |oracle: &Oracle<2>| {
        let mut duals = vec![0.; rows];
        duals[offset + oracle.nonlinear_rows[0].1] = -4.;
        minimum(
            oracle,
            &oracle.network.graph,
            &[0., 0., -5., 0., 0.],
            offset,
            &duals,
            &mut 100_000,
            Instant::now() + Duration::from_secs(1),
        )
    };
    // Repeated walk 0,1,2,1,3 pays the floor row once: -5 + 4 = -1.
    assert_eq!(run(&node.state.oracles[0]), Some(Some(-1.)));
    // Limited memory forgets the first visit to 1. The certificate must cover
    // that zero coefficient even if enumerated columns use full memory.
    let oracle = &mut node.state.oracles[0];
    let index = oracle.nonlinear_rows[0].0;
    oracle.solver.cuts.cuts[index].memory[2] = false;
    assert_eq!(run(oracle), Some(Some(-5.)));
}
