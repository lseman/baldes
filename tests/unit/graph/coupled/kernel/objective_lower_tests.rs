use super::*;

#[test]
fn encoded_literal_slope_costs_have_a_polynomial_directed_bound() {
    use crate::graph::coupled::{EncodedFunction, EvaluationPolicy, PreparedSlopeFunction};
    let function = EncodedFunction::SlopeSegments {
        function: PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (0., 10.)]).unwrap(),
        policy: EvaluationPolicy::StrictDomain,
    };
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_coupled_resources(&[
            CoupledResource::EncodedPiecewiseLinearResource {
                initial: 0.,
                bounds: Window::new(0., 10.).unwrap(),
                functions: vec![function.clone(); 2],
            },
            CoupledResource::EncodedPiecewiseLinearCost {
                parent: 0,
                functions: vec![function; 2],
            },
        ])
        .unwrap();
    assert_eq!(
        graph.coupled_kernel().unwrap().objective_walk_lower_bound(
            &graph,
            2,
            &mut 10_000,
            Instant::now() + std::time::Duration::from_secs(10)
        ),
        Some(4.)
    );
}

#[test]
fn directed_walk_relaxation_encloses_every_signed_cumulative_path() {
    for initial in [-0.1, 0., 1.25] {
        let mut graph = Graph::<2>::new(5, 0).unwrap();
        for (tail, head) in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)] {
            graph.add_arc(tail, head, 0.).unwrap();
        }
        graph
            .set_coupled_resources(&[CoupledResource::CumulativeCost {
                initial,
                increments: vec![-1.1, 2.3, 0.7, -4.5, 0.3],
                weights: vec![1.7, -0.8, -2.1, 3.1, -0.4],
                sink_upper: None,
                source_upper: None,
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        let mut work = 100_000;
        let bound = kernel
            .objective_walk_lower_bound(&graph, 4, &mut work, deadline)
            .unwrap();
        for path in [&[0, 2, 4][..], &[1, 3, 4][..]] {
            let exact = kernel.replay(path).unwrap().unwrap();
            assert!(bound <= exact, "{bound} > {exact}");
        }
        assert!(work < 100_000);
        assert!(kernel
            .objective_walk_lower_bound(&graph, 4, &mut 0, deadline)
            .is_none());
        assert!(kernel
            .objective_walk_lower_bound(&graph, 4, &mut 100_000, Instant::now())
            .is_none());
    }
}

#[test]
fn numeric_and_interval_objectives_keep_their_declared_accumulation_order() {
    let constant = |v| PiecewiseLinearFunction::new(vec![[-10., v], [10., v]]).unwrap();
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[
            CoupledResource::PiecewiseLinearResource {
                initial: 0.,
                bounds: Window::new(-10., 10.).unwrap(),
                functions: vec![constant(1.); 3],
            },
            CoupledResource::PiecewiseLinearCost {
                parent: 0,
                functions: vec![constant(-0.1); 3],
            },
            CoupledResource::IntervalCost {
                initial: Window::new(0., 0.).unwrap(),
                bounds: Window::new(-10., 10.).unwrap(),
                increments: vec![0.; 3],
                functions: vec![constant(0.2); 3],
            },
        ])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let bound = kernel
        .objective_walk_lower_bound(
            &graph,
            3,
            &mut 100_000,
            Instant::now() + std::time::Duration::from_secs(10),
        )
        .unwrap();
    assert!(bound <= kernel.replay(&[0, 1, 2]).unwrap().unwrap());
    assert!(bound > 0.29);
}
