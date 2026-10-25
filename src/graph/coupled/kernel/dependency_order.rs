//! Sufficient forward dependency orders for every continuation in the domain.
use super::bounds::Range;
use super::*;

const LESS: u8 = 1;
const GREATER: u8 = 2;

#[derive(Clone, Debug, PartialEq)]
struct Dependency {
    slot: usize,
    parent: usize,
    factors: u8,
    lower_safe: bool,
    upper_safe: bool,
    finite: bool,
    parent_domain_safe: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub(super) struct DependencyOrder {
    dependencies: Vec<Dependency>,
    numeric_orders: Vec<[bool; 2]>,
    required_equal: Vec<bool>,
}
impl DependencyOrder {
    pub(super) fn prepare(
        definitions: &[CoupledResource],
        slots: &[Option<usize>],
        domains: &[Range],
    ) -> Self {
        let mut required_equal = vec![false; domains.len()];
        for (r, definition) in definitions.iter().enumerate() {
            match definition {
                CoupledResource::PiecewiseLinearResource { .. }
                | CoupledResource::EncodedPiecewiseLinearResource { .. } => {
                    required_equal[slots[r].unwrap()] = true
                }
                CoupledResource::PiecewiseLinearCost { parent, .. }
                | CoupledResource::EncodedPiecewiseLinearCost { parent, .. } => {
                    required_equal[slots[*parent].unwrap()] = true
                }
                CoupledResource::EncodedPiecewiseLinearDependent { parent, .. } => {
                    required_equal[slots[r].unwrap()] = true;
                    required_equal[slots[*parent].unwrap()] = true;
                }
                _ => {}
            }
        }
        let mut numeric_orders = vec![[true; 2]; domains.len()];
        for (r, definition) in definitions.iter().enumerate() {
            if let CoupledResource::CumulativeCost { increments, .. } = definition {
                let slot = slots[r].unwrap();
                let bounded = domains[slot].finite()
                    && increments
                        .iter()
                        .all(|&delta| domains[slot].add(Range::point(delta)).finite());
                // For unbounded domains, only the comparison direction that
                // cannot overflow away from a finite comparator is safe.
                numeric_orders[slot] = [
                    bounded || increments.iter().all(|&delta| delta >= 0.),
                    bounded || increments.iter().all(|&delta| delta <= 0.),
                ];
            }
        }
        let dependencies = definitions
            .iter()
            .enumerate()
            .filter_map(|(r, definition)| {
                if let CoupledResource::PiecewiseLinearDependent {
                    parent,
                    bounds,
                    functions,
                    ..
                } = definition
                {
                    let slot = slots[r].unwrap();
                    let parent = slots[*parent].unwrap();
                    let mut direction = 0;
                    let (mut lower_safe, mut upper_safe, mut finite) = (true, true, true);
                    let mut parent_domain_safe = true;
                    for function in functions {
                        for pair in function.points.windows(2) {
                            direction |= if pair[1][1] > pair[0][1] {
                                GREATER
                            } else if pair[1][1] < pair[0][1] {
                                LESS
                            } else {
                                0
                            };
                        }
                        let domain = function.prepared_domain();
                        parent_domain_safe &= domains[parent].lo >= domain.lower
                            && domains[parent].hi <= domain.upper;
                        let delta = function
                            .range(domains[parent])
                            .unwrap_or_else(Range::unknown);
                        finite &= domains[parent].lo >= domain.lower
                            && domains[parent].hi <= domain.upper
                            && delta.finite()
                            && domains[slot].add(delta).finite();
                        lower_safe &= delta.lo >= 0.;
                        upper_safe &= delta.hi <= 0.;
                    }
                    return Some(Dependency {
                        slot,
                        parent,
                        factors: direction,
                        lower_safe: bounds.lower == f64::NEG_INFINITY || lower_safe,
                        upper_safe: bounds.upper == f64::INFINITY || upper_safe,
                        finite,
                        parent_domain_safe,
                    });
                }
                let CoupledResource::Dependent {
                    parent,
                    bounds,
                    offsets,
                    factors,
                    ..
                } = definition
                else {
                    return None;
                };
                let slot = slots[r].unwrap();
                let parent = slots[*parent].unwrap();
                let mut direction = 0;
                let mut lower_safe = true;
                let mut upper_safe = true;
                let mut finite = true;
                for (&offset, &factor) in offsets.iter().zip(factors) {
                    direction |= if factor > 0. {
                        GREATER
                    } else if factor < 0. {
                        LESS
                    } else {
                        0
                    };
                    // Enclose each literal operation separately. Apart from proving
                    // increment signs, require all intermediates to stay finite so
                    // deletion cannot conceal an overflow unique to the dominator.
                    let product = Range::point(factor).multiply(domains[parent]);
                    let delta = Range::point(offset).add(product);
                    let image = domains[slot].add(delta);
                    finite &= product.finite() && delta.finite() && image.finite();
                    lower_safe &= delta.lo >= 0.;
                    upper_safe &= delta.hi <= 0.;
                }
                Some(Dependency {
                    slot,
                    parent,
                    factors: direction,
                    lower_safe: bounds.lower == f64::NEG_INFINITY || lower_safe,
                    upper_safe: bounds.upper == f64::INFINITY || upper_safe,
                    finite,
                    parent_domain_safe: true,
                })
            })
            .collect();
        Self {
            dependencies,
            numeric_orders,
            required_equal,
        }
    }

    pub(super) fn preserves(&self, a: &[f64], b: &[f64], domains: &[Range]) -> bool {
        if a.len() != b.len() || a.len() != domains.len() {
            return false;
        }
        if a == b
            || (self.dependencies.is_empty() && !self.required_equal.iter().any(|&equal| equal))
        {
            return true;
        }
        // Bounds used in preparation must actually contain both entry states.
        // Check this even in phase I; feasibility is independent of cost scale.
        let mut directions = StateValues::<u8, 8>::filled(a.len(), 0);
        for (slot, ((&a, &b), domain)) in a.iter().zip(b).zip(domains).enumerate() {
            if !a.is_finite()
                || !b.is_finite()
                || a < domain.lo
                || a > domain.hi
                || b < domain.lo
                || b > domain.hi
            {
                return false;
            }
            if (a < b && !self.numeric_orders[slot][0]) || (a > b && !self.numeric_orders[slot][1])
            {
                return false;
            }
            directions[slot] = if a < b {
                LESS
            } else if a > b {
                GREATER
            } else {
                0
            };
        }
        // Earlier-parent validation gives a DAG in declaration order. The
        // union includes the child's retained difference and all signs its
        // already-closed parent can induce. It is invariant under any number
        // and ordering of physical arcs, including rounded monotone updates.
        for dependency in &self.dependencies {
            let parent = directions[dependency.parent];
            // A constant function can hide every value difference while its
            // argument domain still distinguishes the two routes.
            if parent != 0 && !dependency.parent_domain_safe {
                return false;
            }
            let mut direction = directions[dependency.slot];
            if dependency.factors & GREATER != 0 {
                direction |= parent;
            }
            if dependency.factors & LESS != 0 {
                direction |= ((parent & LESS) << 1) | ((parent & GREATER) >> 1);
            }
            if direction != 0
                && (!dependency.finite
                    || (direction & LESS != 0 && !dependency.lower_safe)
                    || (direction & GREATER != 0 && !dependency.upper_safe))
            {
                return false;
            }
            directions[dependency.slot] = direction;
        }
        self.required_equal
            .iter()
            .zip(directions.iter().copied())
            .all(|(&equal, direction)| !equal || direction == 0)
    }
}

#[cfg(test)]
mod tests {
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
            assert!(DependencyOrder::prepare(&definitions, &slots, &domains)
                .preserves(&a, &b, &domains));
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
                                    assert!(
                                        !fb || fa,
                                        "factor={factor} path={path} length={length}"
                                    );
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
}
