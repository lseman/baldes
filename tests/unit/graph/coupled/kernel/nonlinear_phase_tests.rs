use super::*;
fn model(cost_upper: f64) -> CoupledKernel {
    let mut graph: Graph<2> = Graph::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    CoupledKernel::prepare(
        &graph,
        &[
            CoupledResource::PiecewiseLinearResource {
                initial: 0.,
                bounds: Window {
                    lower: 0.,
                    upper: 2.,
                },
                functions: vec![
                    PiecewiseLinearFunction::new(vec![[0., 1.], [1., 2.], [2., 2.]])
                        .unwrap();
                    2
                ],
            },
            CoupledResource::PiecewiseLinearCost {
                parent: 0,
                functions: vec![
                    PiecewiseLinearFunction::new(vec![
                        [0., 0.],
                        [cost_upper, 2. * cost_upper]
                    ])
                    .unwrap();
                    2
                ],
            },
        ],
    )
    .unwrap()
}
#[test]
fn phase_i_suppresses_nonlinear_costs_and_retains_all_numeric_constraints() {
    let mut kernel = model(2.);
    assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(6.));
    let (valid, [lower, upper]) = kernel.certify(&[0, 1]).unwrap();
    assert!(valid && lower <= 6. && upper >= 6.);
    kernel.suppress_costs();
    assert!(!kernel.has_costs());
    let (state, cost) = kernel.replay_state(&[0, 1]).unwrap().unwrap();
    assert_eq!(state.numeric_values().unwrap(), [2.]);
    assert_eq!(cost, 0.);
    let (f, _) = kernel.extend(&kernel.root(true), 0, true).unwrap().unwrap();
    let (b, bc) = kernel
        .extend(&kernel.root(false), 1, false)
        .unwrap()
        .unwrap();
    assert_eq!(bc, 0.);
    assert_eq!(kernel.join(&f, &b).unwrap(), Some(0.));
    assert_eq!(kernel.certify(&[0, 1]), Some((true, [0., 0.])));
    let mut restricted = model(1.5);
    assert!(restricted.replay(&[0, 1]).unwrap().is_none());
    restricted.suppress_costs();
    assert!(restricted.replay(&[0, 1]).unwrap().is_none());
    assert_eq!(restricted.certify(&[0, 1]), Some((false, [0., 0.])));
}
