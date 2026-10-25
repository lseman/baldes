use super::*;
use crate::graph::coupled::PiecewiseLinearFunction;

#[test]
fn scalar_bases_survive_resource_cancellation_and_stay_independent() {
    let large = 9_007_199_254_740_994.;
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, -large).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let constant = |value| PiecewiseLinearFunction::new(vec![[0., value], [1., value]]).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::IntervalCost {
            initial: Window::new(0., 1.).unwrap(),
            bounds: Window::new(0., 1.).unwrap(),
            increments: vec![0.; 2],
            functions: vec![constant(1.), constant(large)],
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let (mut first, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
    first.account_extension(&root, -large, -large + 2.).unwrap();
    let (mut last, _) = kernel.extend(&first, 1, true).unwrap().unwrap();
    let costs = last.account_extension(&first, 0., 0.).unwrap();
    assert_eq!(costs, (2., 4.));
    assert_eq!(last.accounting.unwrap().reduced, -large);
    assert_eq!(last.accounting.unwrap().original, -large + 2.);
    // Interleaving deltas would incorrectly produce four instead of the canonical two.
    assert_eq!((-large + 1.) + ((1. + large) - 1.), 4.);
}

#[test]
fn missing_prefix_accounting_declines_instead_of_inventing_a_scalar_base() {
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::Selection {
            initial: 1,
            allowed: vec![1],
            vertices: vec![],
            relaxed: false,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let (mut next, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
    let untracked = next.clone();
    assert!(next.account_extension(&untracked, 1., 2.).is_err());
}
