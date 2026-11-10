use super::*;
use crate::graph::coupled::{CoupledResource, PluginResource};
use crate::labeling::engine::{EngineControls, LabelingEngine};
use crate::pricing::phases::Phase;
use crate::{Controls, Graph};

fn fixture() -> (Problem, Graph<2>, RestockingResource) {
    let p = Problem::new(
        8,
        vec![Demand::new(vec![(0, 0.5), (4, 0.5)]).unwrap(); 3],
        vec![
            vec![0., 1., 5., 3.],
            vec![2., 0., 4., 6.],
            vec![7., 1., 0., 2.],
            vec![3., 4., 8., 0.],
        ],
    )
    .unwrap();
    let mut graph = Graph::<2>::new(5, 0).unwrap();
    let mut endpoints = vec![];
    for i in 0..4 {
        for j in 0..4 {
            if i == j {
                continue;
            }
            let from = if j == 0 { 0 } else { j };
            let to = if i == 0 { 4 } else { i };
            let arc = graph.add_arc(from, to, p.costs()[i][j]).unwrap();
            graph.arc_mut(arc).unwrap().reduced_cost -= if i == 0 { 0. } else { 10. };
            endpoints.push((i, j));
        }
    }
    let resource = RestockingResource::new(p.clone(), endpoints)
        .unwrap()
        .with_reversed_pricing();
    graph
        .set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(
            resource.clone(),
        ))])
        .unwrap();
    (p, graph, resource)
}

#[test]
fn reversed_forward_pricing_replay_joins_and_certificates_preserve_original_asymmetric_routes() {
    let (p, graph, resource) = fixture();
    let weighted: Vec<_> = resource
        .arcs
        .iter()
        .copied()
        .zip(&graph.arcs)
        .map(|((i, j), arc)| (i, j, arc.reduced_cost))
        .collect();
    assert_eq!(
        completion::customer_potentials(&resource, &weighted),
        vec![0., 10., 10., 10.]
    );
    let mut original_resource = resource.clone();
    original_resource.reversed = false;
    let original_weights: Vec<_> = resource
        .arcs
        .iter()
        .map(|&(i, j)| (i, j, p.costs()[i][j] - if j == 0 { 0. } else { 10. }))
        .collect();
    assert_eq!(
        completion::customer_potentials(&original_resource, &original_weights),
        vec![0., 10., 10., 10.]
    );
    let mut optimum = f64::INFINITY;
    for original in [
        vec![1],
        vec![2],
        vec![3],
        vec![1, 2],
        vec![2, 1],
        vec![1, 3],
        vec![3, 1],
        vec![2, 3],
        vec![3, 2],
        vec![1, 2, 3],
        vec![1, 3, 2],
        vec![2, 1, 3],
        vec![2, 3, 1],
        vec![3, 1, 2],
        vec![3, 2, 1],
    ] {
        let mut nodes = vec![0];
        nodes.extend(original.iter().rev());
        nodes.push(4);
        let arcs: Vec<_> = nodes
            .windows(2)
            .map(|w| {
                graph
                    .arcs
                    .iter()
                    .position(|a| a.tail == w[0] && a.head == w[1])
                    .unwrap()
            })
            .collect();
        let ids: Vec<_> = arcs.iter().map(|&a| graph.arcs[a].id).collect();
        let route = graph.elementary_route(&ids).unwrap().unwrap();
        let expected = p
            .evaluate_route(
                &original,
                super::super::stochastic::Policy::OptimalRestocking,
            )
            .unwrap();
        assert!((route.original_cost - expected.total_cost).abs() < 1e-10);
        let (valid, range) = resource
            .certify_objective(&arcs, &mut 1_000_000, None)
            .unwrap();
        assert!(
            valid
                && range[0] <= expected.expected_recourse_cost
                && expected.expected_recourse_cost <= range[1]
        );
        optimum = optimum.min(expected.total_cost - 10. * original.len() as f64);
        for split in 0..=arcs.len() {
            let mut state = resource.root(true);
            for &arc in &arcs[..split] {
                state = resource.extend(&state, arc, true).unwrap().unwrap();
            }
            assert!(resource
                .join(&state, &mut arcs[split..].iter().copied())
                .unwrap());
        }
    }
    let mut engine = LabelingEngine::prepare(
        &graph,
        EngineControls {
            bidirectional: false,
            staged_pricing: true,
            ..Default::default()
        },
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        },
    )
    .unwrap();
    let priced = engine.solve(&graph, &Controls::default()).unwrap();
    assert!(priced.exact);
    assert!((priced.minimum_reduced_cost - optimum).abs() < 1e-9);
    for route in priced.columns {
        let original: Vec<_> = route.nodes[1..route.nodes.len() - 1]
            .iter()
            .rev()
            .map(|&v| v as usize)
            .collect();
        let total = p
            .evaluate_route(
                &original,
                super::super::stochastic::Policy::OptimalRestocking,
            )
            .unwrap()
            .total_cost;
        assert!((route.cost - total + 10. * original.len() as f64).abs() < 1e-9);
    }
    let mut solver = crate::pricing::Solver::<2>::new(5, 0).unwrap();
    solver.graph = graph;
    assert!(solver.graph.coupled_kernel().unwrap().bellman_pricing());
    assert!(!solver.graph.coupled_kernel().unwrap().backward_pricing());
}
