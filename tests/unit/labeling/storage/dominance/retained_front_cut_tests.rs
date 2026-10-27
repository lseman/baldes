use super::*;
use crate::graph::coupled::{CoupledResource, PiecewiseLinearFunction};
use crate::Window;

#[test]
fn front_cost_margin_cannot_pay_cut_state_tradeoffs_and_words_are_read_once() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head, cost) in [(0, 1, -2.), (0, 1, 0.), (1, 2, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::IntervalCost {
            initial: Window::new(0., 1.).unwrap(),
            bounds: Window::new(0., 1.).unwrap(),
            increments: vec![0.; 3],
            functions: vec![PiecewiseLinearFunction::new(vec![[0., 0.], [1., 0.]]).unwrap(); 3],
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let (mut left, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
    let (mut right, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
    let (left_cost, _) = left.account_extension(&root, -2., -2.).unwrap();
    let (right_cost, _) = right.account_extension(&root, 0., 0.).unwrap();
    let base = HotLabel {
        key: 0.,
        cost: right_cost,
        additional_cost: 0.,
        resources: [0.; 2],
        hops: 1,
        ng: 0,
    };
    let mut a = HotLabel {
        cost: left_cost,
        additional_cost: 2.,
        ..base
    };
    let words = [0u32; 1];
    assert!(!LabelStorage::dominates_coupled_slices(
        &graph,
        &a,
        &base,
        true,
        1,
        InsertionTest::Full,
        false,
        &[],
        Some(&left),
        Some(&right),
        || (&words[..], &words[..]),
    ));
    a.additional_cost = 0.;
    let different = [1u32; 1];
    assert!(!LabelStorage::dominates_coupled_slices(
        &graph,
        &a,
        &base,
        true,
        1,
        InsertionTest::Full,
        false,
        &[],
        Some(&left),
        Some(&right),
        || (&different[..], &words[..]),
    ));
    let empty: [u32; 0] = [];
    assert!(LabelStorage::dominates_coupled_slices(
        &graph,
        &a,
        &base,
        true,
        1,
        InsertionTest::Full,
        false,
        &[],
        Some(&left),
        Some(&right),
        || (&empty[..], &words[..]),
    ));
    let reads = std::cell::Cell::new(0);
    assert!(LabelStorage::dominates_coupled_slices(
        &graph,
        &a,
        &base,
        true,
        1,
        InsertionTest::Full,
        false,
        &[],
        Some(&left),
        Some(&right),
        || {
            reads.set(reads.get() + 1);
            (&words[..], &words[..])
        },
    ));
    assert_eq!(reads.get(), 1);
}
