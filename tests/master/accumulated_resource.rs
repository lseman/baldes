use baldes::{
    graph::{ResourceKind, Window},
    pricing::{restrictions::AccumulatedResourceBranch, Solver},
};

#[cfg(feature = "clp")]
#[path = "accumulated_resource/grouped.rs"]
mod grouped;

fn solver(kind: ResourceKind) -> Solver<5> {
    let mut s = Solver::<5>::new(3, 5).unwrap();
    s.graph.elementary = true;
    s.graph.resources[4] = kind;
    for v in 0..3 {
        for r in 0..5 {
            s.graph.set_bounds(v, r, 0., 3.).unwrap();
        }
    }
    s
}

#[test]
fn stored_threshold_membership_matches_adjacent_floats() {
    let boundary = 1. - AccumulatedResourceBranch::THRESHOLD_OFFSET;
    // Includes the controlled reference fixture and both sides of each boundary.
    for q in [
        boundary - 1e-6,
        f64::from_bits(boundary.to_bits() - 1),
        boundary,
        f64::from_bits(boundary.to_bits() + 1),
        1. - 0.5e-6,
        f64::from_bits(1f64.to_bits() - 1),
        1.,
        f64::from_bits(1f64.to_bits() + 1),
    ] {
        let down = AccumulatedResourceBranch {
            arc: 0,
            resource: 4,
            threshold: 1.,
            upper: false,
        };
        let up = AccumulatedResourceBranch {
            upper: true,
            ..down
        };
        assert_eq!(up.boundary(), boundary);
        assert_eq!(down.satisfied_by(q), q <= boundary);
        assert_eq!(up.satisfied_by(q), q >= boundary);
        assert!(down.satisfied_by(q) || up.satisfied_by(q));
        assert_eq!(down.satisfied_by(q) && up.satisfied_by(q), q == boundary);
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!down.satisfied_by(invalid));
            assert!(!up.satisfied_by(invalid));
        }
        #[cfg(feature = "clp")]
        {
            use baldes::master::{branching::BranchConstraint, routing::RoutingContext};
            let mut s = solver(ResourceKind::Disposable);
            let target = s.graph.add_arc(0, 1, 1.).unwrap();
            let parallel = s.graph.add_arc(0, 1, 2.).unwrap();
            let back = s.graph.add_arc(1, 0, 0.).unwrap();
            let end = s.graph.add_arc(1, 2, 0.).unwrap();
            let p = RoutingContext::set_partitioning(s.graph, 1.).unwrap();
            let profile = vec![0., 0., 0., 0., q];
            let mut route = baldes::Route {
                nodes: vec![0, 1, 0, 1, 2],
                arcs: vec![target, back, target, end],
                cost: 2.,
                original_cost: 2.,
                resources: profile.clone(),
                resource_profile: vec![
                    vec![0.; 5],
                    profile.clone(),
                    profile.clone(),
                    profile.clone(),
                    profile,
                ],
            };
            for rule in [down, up] {
                let branch = BranchConstraint::AccumulatedResource(AccumulatedResourceBranch {
                    arc: target,
                    ..rule
                });
                assert_eq!(
                    branch.coefficient(&p, &route),
                    if rule.satisfied_by(q) { 0. } else { 1. }
                );
                route.arcs = vec![parallel, back, parallel, end];
                assert_eq!(branch.coefficient(&p, &route), 0.);
                route.arcs = vec![target, back, target, end];
            }
            route.resource_profile.clear();
            let branch = BranchConstraint::AccumulatedResource(AccumulatedResourceBranch {
                arc: target,
                ..down
            });
            assert_eq!(branch.coefficient(&p, &route), 1.);
        }
    }
}

#[test]
fn windows_pricing_enumeration_and_forks_use_the_same_boundary() {
    let boundary = 1. - AccumulatedResourceBranch::THRESHOLD_OFFSET;
    for q in [
        f64::from_bits(boundary.to_bits() - 1),
        boundary,
        f64::from_bits(boundary.to_bits() + 1),
        1.,
    ] {
        for kind in [ResourceKind::Disposable, ResourceKind::Equality] {
            for mode in [0., 1., 2.] {
                for upper in [false, true] {
                    let mut parent = solver(kind);
                    parent
                        .set_option("BALDESuseBidirectionalSearch", mode)
                        .unwrap();
                    if kind == ResourceKind::Equality {
                        parent.graph.set_bounds(2, 4, q, q).unwrap();
                    }
                    let arc = parent.graph.add_arc(0, 1, -5.).unwrap();
                    parent.graph.arc_mut(arc).unwrap().consumption[4] = q;
                    parent.graph.arc_mut(arc).unwrap().bounds[4] = Some(Window::new(q, q).unwrap());
                    let parallel = parent.graph.add_arc(0, 1, -2.).unwrap();
                    parent.graph.arc_mut(parallel).unwrap().consumption[4] = q;
                    parent.graph.add_arc(1, 2, 0.).unwrap();
                    let baseline = parent.graph.clone();
                    assert!(parent
                        .enumerate(100.)
                        .unwrap()
                        .iter()
                        .any(|r| r.arcs.contains(&arc)));
                    let mut child = parent.fork_for_node();
                    let rule = AccumulatedResourceBranch {
                        arc,
                        resource: 4,
                        threshold: 1.,
                        upper,
                    };
                    child.add_accumulated_resource_branch(rule).unwrap();
                    assert!(child.enumeration.is_none());
                    let priced = child.price_phase(0, 100, false).unwrap();
                    let enumerated = child.enumerate(100.).unwrap();
                    assert!(priced.exact);
                    for routes in [&priced.columns, &enumerated] {
                        assert_eq!(
                            routes.iter().any(|r| r.arcs.contains(&arc)),
                            rule.satisfied_by(q),
                            "q={q:?} {kind:?} mode={mode} upper={upper}"
                        );
                        if !rule.satisfied_by(q) {
                            assert!(routes.iter().any(|r| r.arcs.contains(&parallel)));
                        }
                        for route in routes.iter().filter(|r| r.arcs.contains(&arc)) {
                            assert!(rule.satisfied_by(route.resource_profile[1][4]));
                        }
                    }
                    assert_eq!(
                        priced.minimum_reduced_cost,
                        if rule.satisfied_by(q) { -5. } else { -2. }
                    );
                    assert_eq!(parent.graph, baseline);
                    child.clear_accumulated_resource_branches().unwrap();
                    assert_eq!(child.graph, baseline);
                    assert_eq!(
                        child
                            .price_phase(0, 100, false)
                            .unwrap()
                            .minimum_reduced_cost,
                        -5.
                    );
                }
            }
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn threshold_children_preserve_rows_bounds_and_restore_the_parent() {
    use baldes::master::{
        branching::BranchConstraint,
        routing::{self, RootOptions, RoutingContext},
    };
    let boundary = 1. - AccumulatedResourceBranch::THRESHOLD_OFFSET;
    let mut s = solver(ResourceKind::Disposable);
    for (cost, q) in [(1., boundary - 0.5e-6), (2., boundary + 0.5e-6)] {
        let arc = s.graph.add_arc(0, 1, cost).unwrap();
        s.graph.arc_mut(arc).unwrap().consumption[4] = q;
    }
    let arc = s.graph.add_arc(1, 2, 0.).unwrap();
    let problem = RoutingContext::set_partitioning(s.graph, 1.).unwrap();
    for enumerate in [false, true] {
        let mut options = RootOptions::default();
        options.capture_state = true;
        options.enumerate = enumerate;
        options.incumbent = 10.;
        options.cg.stabilize = false;
        let parent = routing::solve(problem.clone(), options.clone()).unwrap();
        let snapshot = parent.snapshot.as_ref().unwrap();
        for upper in [false, true] {
            let rule = AccumulatedResourceBranch {
                arc,
                resource: 4,
                threshold: 1.,
                upper,
            };
            let branch = BranchConstraint::AccumulatedResource(rule);
            let child =
                routing::inherit_branches(snapshot, &[branch.clone()], options.clone()).unwrap();
            let mut cold = problem.clone();
            cold.branch(branch.clone()).unwrap();
            let fresh = routing::solve(cold, options.clone()).unwrap();
            assert_eq!(child.master.status, fresh.master.status);
            // Pricing enforcement can wait at the new lower window on the
            // cheap route. Master-only enforcement keeps the original earliest
            // schedule, so the upper child uses the expensive route instead.
            let expected = if upper
                && !baldes::master::branching::enforcement::ACCUMULATED_RESOURCE_IN_PRICING
            {
                2.
            } else {
                1.
            };
            assert!(
                (fresh.master.lp - expected).abs() < 1e-8,
                "upper={upper}: {fresh:?}"
            );
            if child.master.pruned_by_bound {
                // The inherited incumbent can close the node before the new
                // schedule is reoptimized. Its RMP objective is not the root LP.
                assert!(child.master.bound <= fresh.master.lp + 1e-8);
                assert!(child.master.bound >= parent.master.bound - 1e-8);
                assert!((child.master.bound - child.incumbent).abs() < 1e-8);
                assert!(child.incumbent <= fresh.master.lp + 1e-8);
            } else {
                assert!((child.master.lp - fresh.master.lp).abs() < 1e-8);
            }
            for (route, value) in child.route_values() {
                if value > 1e-6 {
                    assert_eq!(branch.coefficient(&problem, route), 0.);
                }
            }
            assert_eq!(snapshot.branch_depth(), 0);
        }
        let restored = routing::inherit_branches(snapshot, &[], options.clone()).unwrap();
        assert_eq!(restored.master.status, parent.master.status);
        assert!((restored.master.lp - parent.master.lp).abs() < 1e-8);
    }
}

#[cfg(feature = "clp")]
#[test]
fn accumulated_branch_enforcement_keeps_master_rows_in_both_build_modes() {
    use baldes::master::{
        branching::{enforcement::ACCUMULATED_RESOURCE_IN_PRICING, BranchConstraint},
        routing::RoutingContext,
    };
    let mut s = solver(ResourceKind::Disposable);
    let arc = s.graph.add_arc(0, 1, 1.).unwrap();
    s.graph.add_arc(1, 2, 0.).unwrap();
    let mut problem = RoutingContext::set_partitioning(s.graph, 1.).unwrap();
    let old_rows = problem.model.rows.len();
    let branch = AccumulatedResourceBranch {
        arc,
        resource: 4,
        threshold: 1.,
        upper: false,
    };
    problem
        .branch(BranchConstraint::AccumulatedResource(branch))
        .unwrap();
    assert_eq!(problem.model.rows.len(), old_rows + 1);
    let window = problem.model.networks[0]
        .graph
        .arcs
        .iter()
        .find(|a| a.id == arc)
        .unwrap()
        .bounds[4];
    if ACCUMULATED_RESOURCE_IN_PRICING {
        assert_eq!(window.unwrap().upper, branch.boundary());
    } else {
        assert_eq!(window, None);
    }
}

#[test]
#[cfg(all(feature = "clp", feature = "accumulated-resource-master"))]
fn impossible_resource_window_remains_a_master_only_resource_branch() {
    use baldes::master::{branching::BranchConstraint, routing::RoutingContext};
    let mut s = solver(ResourceKind::Disposable);
    let arc = s.graph.add_arc(0, 1, 1.).unwrap();
    s.graph.add_arc(1, 2, 0.).unwrap();
    let mut problem = RoutingContext::set_partitioning(s.graph, 1.).unwrap();
    // Head upper bound is 3: the upper child excludes every use of this arc.
    problem
        .branch(BranchConstraint::AccumulatedResource(
            AccumulatedResourceBranch {
                arc,
                resource: 4,
                threshold: 4.,
                upper: true,
            },
        ))
        .unwrap();
    assert!(problem.model.networks[0]
        .graph
        .arcs
        .iter()
        .find(|a| a.id == arc)
        .unwrap()
        .bounds[4]
        .is_none());
}
