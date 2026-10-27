use super::*;

fn layered_cumulative_graph(layers: usize) -> Graph<2> {
    let sink = 2 * layers + 1;
    let mut graph = Graph::<2>::new(sink + 1, 0).unwrap();
    let mut previous = vec![0];
    for layer in 0..layers {
        let current = vec![2 * layer + 1, 2 * layer + 2];
        for &tail in &previous {
            for &head in &current {
                graph.add_arc(tail, head, 0.).unwrap();
            }
        }
        previous = current;
    }
    for tail in previous {
        graph.add_arc(tail, sink, 0.).unwrap();
    }
    let arcs = graph.arcs.len();
    graph
        .set_coupled_resources(&[crate::graph::coupled::CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![1.; arcs],
            weights: vec![1.; arcs],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    graph
}

#[test]
fn exponentially_many_routes_keep_a_polynomial_directed_resource_bound() {
    const LAYERS: usize = 20;
    let graph = layered_cumulative_graph(LAYERS);
    let mut work = 1_000_000;
    // The full 2^20 universe exceeds the caller's finite budget, so the
    // independent route-wide certificate must remain available as fallback.
    let value = minimum(
        &graph,
        &mut work,
        Instant::now() + std::time::Duration::from_secs(10),
    );
    assert_eq!(value, Some(Some(231.)));
    assert!(work > 0 && work < 1_000_000);
}

#[test]
fn coupled_objective_certifies_a_route_universe_larger_than_four_thousand() {
    const LAYERS: usize = 13;
    let graph = layered_cumulative_graph(LAYERS);
    let mut work = 1_000_000;
    // 2^13 routes fit the caller budget and exceed the former 4,096-path cap.
    // Every route has LAYERS+1 arcs and the independent sum 1+...+14 = 105.
    let value = minimum(
        &graph,
        &mut work,
        Instant::now() + std::time::Duration::from_secs(10),
    );
    assert_eq!(value, Some(Some(105.)));
    assert!(work > 0 && work < 1_000_000);
}
