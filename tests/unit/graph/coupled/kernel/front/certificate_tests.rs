use super::*;
#[test]
fn interval_front_witnesses_bound_opposing_costs_and_decline_fractional_translations() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let resource = |increment| CoupledResource::IntervalCost {
        initial: Window {
            lower: 0.,
            upper: 4.,
        },
        bounds: Window {
            lower: -4.,
            upper: 8.,
        },
        increments: vec![increment, -increment],
        functions: vec![
            PiecewiseLinearFunction::new(vec![[-4., -4.], [8., 8.]]).unwrap(),
            PiecewiseLinearFunction::new(vec![[-4., 4.], [8., -8.]]).unwrap(),
        ],
    };
    let mut kernel = CoupledKernel::prepare(&graph, &[resource(0.)]).unwrap();
    let (valid, [lo, hi]) = kernel.certify(&[0, 1]).unwrap();
    assert!(valid && lo <= 0. && (0. ..1e-10).contains(&hi));
    kernel.suppress_costs();
    let (valid, [lo, hi]) = kernel.certify(&[0, 1]).unwrap();
    assert!(valid && lo <= 0. && hi >= 0.);
    assert_eq!(
        CoupledKernel::prepare(&graph, &[resource(0.25)])
            .unwrap()
            .certify(&[0, 1]),
        None
    );
}
