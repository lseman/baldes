use super::*;

fn kernel_graph(tolerance: f64) -> Graph<2> {
    let mut graph = Graph::new(3, 1).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::SimultaneousPickupDelivery {
            capacity: 1.,
            deliveries: vec![0., 0.],
            pickups: vec![0.6, 0.4000005],
            tolerance,
        }])
        .unwrap();
    graph
}

#[test]
fn capacity_admits_exactly_the_configured_tolerance() {
    let graph = kernel_graph(1e-6);
    let kernel = graph.coupled_kernel().unwrap();
    assert!(kernel.replay_state(&[0, 1]).unwrap().is_some());
    let graph = kernel_graph(0.);
    let kernel = graph.coupled_kernel().unwrap();
    assert!(kernel.replay_state(&[0, 1]).unwrap().is_none());
}

#[test]
fn dominance_uses_the_resource_tolerance() {
    let a = [[0.5000005, 0.5000005, 0.5000005]];
    let b = [[0.5, 0.5, 0.5]];
    for (tolerance, expected) in [(1e-6, true), (0., false)] {
        let graph = kernel_graph(tolerance);
        let kernel = graph.coupled_kernel().unwrap();
        let mut x = kernel.root(true);
        let mut y = kernel.root(true);
        x.loads = StateValues::from_slice(&a);
        y.loads = StateValues::from_slice(&b);
        assert_eq!(loads_dominate(&x, &y, &kernel.load_tolerances), expected);
    }
}
