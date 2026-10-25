use super::*;
#[test]
fn merged_bucket_extrema_match_literal_state_inclusion_and_keep_unknowns() {
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial: 2.,
            increments: vec![1.],
            weights: vec![-3.],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    for forward in [true, false] {
        let a = kernel.root(forward);
        let b = kernel.extend(&a, 0, forward).unwrap().unwrap().0;
        let (mut x, mut y, mut literal) = (
            CoupledStateBounds::default(),
            CoupledStateBounds::default(),
            CoupledStateBounds::default(),
        );
        x.include(Some(&a));
        y.include(Some(&b));
        literal.include(Some(&a));
        literal.include(Some(&b));
        x.prepare(kernel);
        x.include_hull(&y);
        assert!(x.initial_dot.is_none());
        assert_eq!(x.numeric, literal.numeric);
        assert_eq!(x.coefficients, literal.coefficients);
        assert_eq!(x.state_costs, literal.state_costs);
        assert_eq!(x.direction, literal.direction);
        assert_eq!(x.count, literal.count);
        let query = kernel.root(!forward);
        assert_eq!(
            kernel.completion_correction_lower(!forward, Some(&query), &x),
            kernel.completion_correction_lower(!forward, Some(&query), &literal)
        );
        let mut unknown = CoupledStateBounds::default();
        unknown.include(None);
        x.include_hull(&unknown);
        assert!(!x.usable());
    }
}
