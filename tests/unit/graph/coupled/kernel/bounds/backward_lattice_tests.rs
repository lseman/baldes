use super::*;

fn kernel(initial: f64) -> CoupledKernel {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 1), (0, 1), (1, 2), (1, 2), (1, 2)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial,
            increments: vec![-1., 1., 2., 0., 0., 0.],
            weights: vec![0., 0., 0., -2., 0., 3.],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    graph.coupled_kernel().unwrap().clone()
}

#[test]
fn exact_backward_compensation_bounds_literal_shared_prefix_corrections() {
    for initial in [-2., 0., 3.] {
        let kernel = kernel(initial);
        let root = kernel.root(false);
        for left in 3..6 {
            for right in 3..6 {
                let (a, _) = kernel.extend(&root, left, false).unwrap().unwrap();
                let (b, _) = kernel.extend(&root, right, false).unwrap().unwrap();
                let a = a.suffix.as_ref().unwrap();
                let b = b.suffix.as_ref().unwrap();
                let penalty = kernel
                    .exact_backward_penalty(a, b, &kernel.optimization.numeric)
                    .unwrap();
                for entry in -2..=4 {
                    let x = initial + f64::from(entry);
                    let ca = a.coefficients[0] * x - a.coefficients[0] * initial;
                    let cb = b.coefficients[0] * x - b.coefficients[0] * initial;
                    assert!(ca - cb <= penalty);
                }
            }
        }
    }
}

#[test]
fn backward_lattice_requires_integral_coefficients_finite_domains_and_exact_dots() {
    let mut kernel = kernel(0.);
    let root = kernel.root(false);
    let (a, _) = kernel.extend(&root, 3, false).unwrap().unwrap();
    let (b, _) = kernel.extend(&root, 4, false).unwrap().unwrap();
    let mut a = a.suffix.unwrap();
    let b = b.suffix.unwrap();
    assert!(kernel
        .exact_backward_penalty(&a, &b, &kernel.optimization.numeric)
        .is_some());
    a.coefficients[0] = 0.5;
    assert_eq!(
        kernel.exact_backward_penalty(&a, &b, &kernel.optimization.numeric),
        None
    );
    a.coefficients[0] = (1u64 << 53) as f64;
    assert_eq!(
        kernel.exact_backward_penalty(&a, &b, &kernel.optimization.numeric),
        None
    );
    a.coefficients[0] = -2.;
    kernel.optimization.numeric[0] = Range::unknown();
    assert_eq!(
        kernel.exact_backward_penalty(&a, &b, &kernel.optimization.numeric),
        None
    );
    kernel.optimization.numeric[0] = Range { lo: -2., hi: 4. };
    kernel.optimization.exact_costs = false;
    assert_eq!(
        kernel.exact_backward_penalty(&a, &b, &kernel.optimization.numeric),
        None
    );
}
