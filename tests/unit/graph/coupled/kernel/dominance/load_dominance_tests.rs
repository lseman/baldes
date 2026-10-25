use super::*;

fn kernel_graph() -> Graph<2> {
    let mut graph = Graph::new(4, 1).unwrap();
    for (u, v) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
        graph.add_arc(u, v, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::SimultaneousPickupDelivery {
            capacity: 20.,
            deliveries: vec![5., 0., 0., 0.],
            pickups: vec![0., 6., 0., 0.],
            tolerance: 0.,
        }])
        .unwrap();
    graph
}

#[test]
fn prefixes_compare_only_pickups_and_peak() {
    let graph = kernel_graph();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let deliver = kernel.extend(&root, 0, true).unwrap().unwrap().0;
    let pickup = kernel.extend(&root, 1, true).unwrap().unwrap().0;
    // (D,N,M) = (5,-5,0) vs (0,6,6): fewer pickups and a lower peak.
    assert!(CoupledKernel::dominates(Some(&deliver), Some(&pickup)));
    assert!(!CoupledKernel::dominates(Some(&pickup), Some(&deliver)));
}

#[test]
fn reversed_prefixes_also_compare_their_suffix_role() {
    let graph = kernel_graph();
    let kernel = graph.coupled_kernel().unwrap();
    let mut workspace = CoupledWorkspace::default();
    let root = kernel.reversal_root(true);
    let mut extend = |arc| {
        kernel
            .extend_reversal_with_workspace(&root, arc, true, &mut workspace)
            .unwrap()
            .unwrap()
            .0
    };
    let deliver = extend(0);
    let pickup = extend(1);
    // As a reversed suffix the delivering prefix carries more deliveries.
    assert!(!CoupledKernel::dominates(Some(&deliver), Some(&pickup)));
}
