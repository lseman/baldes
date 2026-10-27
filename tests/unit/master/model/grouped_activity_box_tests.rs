use super::*;
use crate::graph::coupled::CoupledResource;

fn signed_graph(cyclic: bool) -> Graph<2> {
    let mut graph = Graph::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    if cyclic {
        graph.add_arc(0, 0, 0.).unwrap();
    }
    let n = graph.arcs.len();
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![-1.; n],
            weights: vec![1.; n],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    graph
        .set_coupled_objective_coefficients(&[(0, -2.)])
        .unwrap();
    graph
}

#[test]
fn signed_intrinsic_envelopes_do_not_inherit_model_cost_or_zero_orthant() {
    let graph = signed_graph(false);
    let envelope = activity_range(&graph, 0).unwrap().unwrap();
    assert_eq!(envelope.lower, -1.);
    assert_eq!(envelope.upper, -1.);
    assert_eq!(graph.coupled_path_cost(&[0]).unwrap(), Some(2.));
}

#[test]
fn an_unbounded_walk_domain_does_not_become_an_empty_activity_envelope() {
    let graph = signed_graph(true);
    assert!(activity_range(&graph, 0).is_none());
}

#[test]
fn integer_program_proof_covers_millions_of_paths_without_enumerating_them() {
    let layers = 24usize;
    let mut graph = Graph::<2>::new(2 * layers + 2, 0).unwrap();
    for head in [1, 2] {
        graph.add_arc(0, head, 0.).unwrap();
    }
    for layer in 0..layers - 1 {
        for tail in [1 + 2 * layer, 2 + 2 * layer] {
            for head in [3 + 2 * layer, 4 + 2 * layer] {
                graph.add_arc(tail, head, 0.).unwrap();
            }
        }
    }
    for tail in [2 * layers - 1, 2 * layers] {
        graph.add_arc(tail, 2 * layers + 1, 0.).unwrap();
    }
    let count = graph.arcs.len();
    graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial: 3.,
            increments: vec![1.; count],
            weights: vec![2.; count],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    assert!(activity_proof(&graph, 0, true).is_none());
    let proof = integral_activity(&graph, 0).unwrap();
    let envelope = activity_range(&graph, 0).unwrap().unwrap();
    assert_eq!((envelope.lower, envelope.upper), (proof.lower, proof.upper));
    let steps = (layers + 1) as f64;
    let actual = 6. * steps + steps * (steps + 1.);
    assert_eq!(proof.lower, 0.);
    assert!(proof.upper >= actual);
    let boxed = GroupedVariableBound {
        variable: OriginalPricingVariable {
            network: 42,
            id: 37,
            cost: 1.,
            entries: vec![],
            arcs: vec![],
            accumulated_resource: Some(0),
            integer: true,
            priority: 0.,
        },
        lower: proof.lower,
        upper: Some(proof.upper),
    };
    let forbidden = vec![false; graph.arcs.len()];
    prepare(
        &[boxed],
        0,
        std::iter::once((42, &graph, forbidden.as_slice())),
    )
    .unwrap();
    let mapping = MappedBranchVariable {
        name: "large-domain-activity".into(),
        priority: 0.,
        masters: vec![],
        networks: vec![MappedNetworkVariable {
            network: 42,
            arcs: vec![],
            activities: vec![(0, 1.)],
            per_route: 0.,
        }],
    };
    mapping
        .validate(std::iter::once((42, &graph)), &[])
        .unwrap();
}

#[test]
fn symbolic_integer_proof_rejects_overflow_but_retains_exact_half_weight_fallback() {
    for (increment, weight, expected) in [(9_007_199_254_740_992., 2., None), (2., 0.5, Some(1.))] {
        let mut graph = Graph::<2>::new(2, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::CustomResource {
                increments: vec![increment],
                weights: vec![weight],
                forward_bound: None,
                backward_bound: None,
            }])
            .unwrap();
        assert!(integer_program_range(&graph, 0).is_none());
        match expected {
            Some(value) => {
                let range = integral_activity(&graph, 0).unwrap();
                assert_eq!((range.lower, range.upper), (value, value));
            }
            None => assert!(integral_activity(&graph, 0).is_none()),
        }
    }
}

#[test]
fn signed_integer_program_bounds_cover_bounded_repeated_walks() {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (2, 1), (1, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    graph
        .set_route_domain(crate::graph::RouteDomain::BoundedWalks { max_arcs: 8 })
        .unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![-1.; 4],
            weights: vec![1.; 4],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    let proof = integer_program_range(&graph, 0).unwrap();
    for repeats in 0..4 {
        let mut path = vec![0];
        for _ in 0..repeats {
            path.extend([1, 2]);
        }
        path.push(3);
        let activity = graph
            .coupled_kernel()
            .unwrap()
            .accumulated_activity(0, &path)
            .unwrap();
        assert_eq!(activity.fract(), 0.);
        assert!(proof.lower <= activity && activity <= proof.upper);
    }
}
