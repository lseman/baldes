use super::*;

#[test]
fn dependency_closure_preserves_cost_parent_equality_across_inline_and_wide_states() {
    for count in [2, 8, 9, 16] {
        let mut definitions = vec![CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![0.],
            weights: vec![0.],
            sink_upper: None,
            source_upper: None,
        }];
        for parent in 0..count - 1 {
            definitions.push(CoupledResource::Dependent {
                parent,
                initial: 0.,
                bounds: Window {
                    lower: 0.,
                    upper: f64::INFINITY,
                },
                offsets: vec![0.],
                factors: vec![1.],
            });
        }
        let mut slots: Vec<_> = (0..count).map(Some).collect();
        let domains = vec![Range { lo: 0., hi: 1000. }; count];
        let mut a = vec![0.; count];
        let mut b = vec![0.; count];
        a[0] = 1.;
        b[0] = 2.;
        assert!(
            DependencyOrder::prepare(&definitions, &slots, &domains).preserves(&a, &b, &domains)
        );
        definitions.push(CoupledResource::PiecewiseLinearCost {
            parent: count - 1,
            functions: vec![PiecewiseLinearFunction::new(vec![[0., 0.], [1000., 1.]]).unwrap()],
        });
        slots.push(None);
        let order = DependencyOrder::prepare(&definitions, &slots, &domains);
        // A future update propagates the root difference through every
        // dependent, so the eventual cost argument cannot remain equal.
        assert!(!order.preserves(&a, &b, &domains));
        assert!(order.preserves(&a, &a, &domains));
    }
}

fn prepared(factors: Vec<f64>, bounds: Window) -> (DependencyOrder, Vec<Range>) {
    let definitions = [
        CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![1.; factors.len()],
            weights: vec![0.; factors.len()],
            sink_upper: None,
            source_upper: None,
        },
        CoupledResource::Dependent {
            parent: 0,
            initial: 0.,
            bounds,
            offsets: vec![0.; factors.len()],
            factors,
        },
    ];
    let domains = vec![
        Range { lo: 0., hi: 10. },
        Range {
            lo: bounds.lower,
            hi: bounds.upper,
        },
    ];
    (
        DependencyOrder::prepare(&definitions, &[Some(0), Some(1)], &domains),
        domains,
    )
}

#[test]
fn increasing_dependencies_admit_smaller_states_and_reject_reverse_order() {
    let (order, domains) = prepared(
        vec![1.],
        Window {
            lower: 0.,
            upper: 100.,
        },
    );
    assert!(order.preserves(&[1., 2.], &[2., 5.], &domains));
    assert!(order.preserves(&[1., 5.], &[2., 5.], &domains));
    assert!(!order.preserves(&[2., 5.], &[1., 2.], &domains));
}

#[test]
fn negative_factors_reverse_parent_order_and_reject_conflicting_differences() {
    let (order, domains) = prepared(
        vec![-1.],
        Window {
            lower: 0.,
            upper: 100.,
        },
    );
    assert!(order.preserves(&[1., 8.], &[2., 5.], &domains));
    assert!(!order.preserves(&[1., 2.], &[2., 5.], &domains));
    let (mixed, domains) = prepared(
        vec![-1., 1.],
        Window {
            lower: 0.,
            upper: 100.,
        },
    );
    assert!(!mixed.preserves(&[1., 5.], &[2., 5.], &domains));
}

#[test]
fn constant_dependencies_accept_distinct_values_but_overflowing_domains_do_not() {
    let (order, domains) = prepared(
        vec![0.],
        Window {
            lower: 0.,
            upper: 100.,
        },
    );
    assert!(order.preserves(&[1., 2.], &[2., 5.], &domains));
    assert!(order.preserves(&[2., 5.], &[1., 2.], &domains));
    let (order, domains) = prepared(
        vec![f64::MAX],
        Window {
            lower: 0.,
            upper: 100.,
        },
    );
    assert!(!order.preserves(&[1., 2.], &[2., 5.], &domains));
}

#[test]
fn comparison_directions_propagate_through_dependency_chains() {
    let definitions = [
        CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![0.],
            weights: vec![0.],
            sink_upper: None,
            source_upper: None,
        },
        CoupledResource::Dependent {
            parent: 0,
            initial: 0.,
            bounds: Window {
                lower: 0.,
                upper: 100.,
            },
            offsets: vec![0.],
            factors: vec![1.],
        },
        CoupledResource::Dependent {
            parent: 1,
            initial: 0.,
            bounds: Window {
                lower: -100.,
                upper: 0.,
            },
            offsets: vec![0.],
            factors: vec![-1.],
        },
    ];
    let domains = [
        Range { lo: 0., hi: 10. },
        Range { lo: 0., hi: 100. },
        Range { lo: -100., hi: 0. },
    ];
    let order = DependencyOrder::prepare(&definitions, &[Some(0), Some(1), Some(2)], &domains);
    assert!(order.preserves(&[1., 2., -2.], &[2., 5., -5.], &domains));
    assert!(!order.preserves(&[1., 2., -10.], &[2., 5., -5.], &domains));
    assert!(!order.preserves(&[11., 2., -2.], &[2., 5., -5.], &domains));
}

#[test]
fn accepted_orders_preserve_feasibility_over_exhaustive_small_continuations() {
    // Independent recurrence replay, rather than the sign propagation used
    // by the implementation. Every sequence of up to four two-way arcs.
    for factor in [-1., 1.] {
        let bounds = Window {
            lower: 0.,
            upper: 40.,
        };
        let (order, domains) = prepared(vec![factor, factor], bounds);
        for pa in 0..=3 {
            for pb in 0..=3 {
                for ya in [0., 5., 20., 40.] {
                    for yb in [0., 5., 20., 40.] {
                        let a = [pa as f64, ya];
                        let b = [pb as f64, yb];
                        if !order.preserves(&a, &b, &domains) {
                            continue;
                        }
                        for length in 0..=4 {
                            for path in 0..(1usize << length) {
                                let (mut a, mut b) = (a, b);
                                let (mut fa, mut fb) = (true, true);
                                for step in 0..length {
                                    let increment = ((path >> step) & 1) as f64;
                                    a[0] += increment;
                                    b[0] += increment;
                                    a[1] += factor * a[0];
                                    b[1] += factor * b[0];
                                    fa &= a[1] >= bounds.lower && a[1] <= bounds.upper;
                                    fb &= b[1] >= bounds.lower && b[1] <= bounds.upper;
                                }
                                assert!(!fb || fa, "factor={factor} path={path} length={length}");
                            }
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn flat_piecewise_values_do_not_erase_argument_domain_constraints() {
    let definitions = [
        CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![1.],
            weights: vec![0.],
            sink_upper: None,
            source_upper: None,
        },
        CoupledResource::PiecewiseLinearDependent {
            parent: 0,
            initial: 0.,
            bounds: Window {
                lower: 0.,
                upper: 10.,
            },
            functions: vec![PiecewiseLinearFunction::new(vec![[2., 0.], [3., 0.]]).unwrap()],
        },
    ];
    let domains = [Range { lo: 0., hi: 10. }, Range { lo: 0., hi: 10. }];
    let order = DependencyOrder::prepare(&definitions, &[Some(0), Some(1)], &domains);
    // One next arc takes a's argument to 4 (invalid) and b's to 3 (valid),
    // although both dependent outputs remain zero wherever defined.
    assert!(!order.preserves(&[3., 0.], &[2., 0.], &domains));
}
