use super::*;
#[test]
fn numeric_suffix_join_retains_literal_order_without_rebuilding_labels() {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (u, v) in [(0, 1), (1, 2), (2, 3)] {
        graph.add_arc(u, v, 0.).unwrap();
    }
    let kernel = CoupledKernel::prepare(
        &graph,
        &[CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![1.; 3],
            weights: vec![1e16, -1e16, 1.],
            future_weight: None,
        }],
    )
    .unwrap();
    let expected = kernel.replay(&[0, 1, 2]).unwrap();
    for split in 0..=3 {
        let f = kernel
            .replay_state(&(0..split).collect::<Vec<_>>())
            .unwrap()
            .unwrap()
            .0;
        let mut b = kernel.root(false);
        for arc in (split..3).rev() {
            b = kernel.extend(&b, arc, false).unwrap().unwrap().0;
        }
        assert_eq!(kernel.join_resource_objective(&f, &b).unwrap(), expected);
    }
}
