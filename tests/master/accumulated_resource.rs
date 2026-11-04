use baldes::{
    graph::{ResourceKind, Window},
    pricing::{restrictions::AccumulatedResourceBranch, Solver},
};

#[cfg(feature = "clp")]
#[path = "accumulated_resource/grouped.rs"]
mod grouped;

fn solver(kind: ResourceKind) -> Solver<5> {
    let mut s = Solver::<5>::new(3, 5).unwrap();
    s.graph.edit().elementary = true;
    s.graph.edit().resources[4] = kind;
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
        // Route coefficients for this boundary are covered by the grouped
        // branch unit test `single_arc_group_coefficients_match_...`.
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
#[cfg(feature = "clp")]
fn resource_branch(
    node: &baldes::master::model::NodeSnapshot<5>,
    arcs: Vec<i32>,
    threshold: f64,
    upper: bool,
) -> baldes::master::model::NodeSnapshot<5> {
    // The vertex-1 cover is shared row 0 of the one-network set partitioning.
    node.with_resource_group_branch(
        0,
        4,
        threshold,
        upper,
        &[baldes::master::model::ResourceGroupMember { network: 0, arcs }],
    )
    .unwrap()
}

#[cfg(feature = "clp")]
fn cold(
    problem: baldes::master::model::Problem<5>,
    options: &baldes::master::model::Options,
) -> baldes::master::model::NodeSnapshot<5> {
    let mut cold = options.clone();
    cold.cg.time_limit = std::time::Duration::ZERO;
    baldes::master::model::solve(problem, cold)
        .unwrap()
        .snapshot
}

#[cfg(feature = "clp")]
#[test]
fn threshold_children_preserve_rows_bounds_and_restore_the_parent() {
    use baldes::master::{
        branching::ACCUMULATED_RESOURCE_IN_PRICING,
        model::{self, EnumerationOptions, Options, Problem},
    };
    let boundary = 1. - AccumulatedResourceBranch::THRESHOLD_OFFSET;
    let mut s = solver(ResourceKind::Disposable);
    let mut arrivals = vec![];
    for (cost, q) in [(1., boundary - 0.5e-6), (2., boundary + 0.5e-6)] {
        let arc = s.graph.add_arc(0, 1, cost).unwrap();
        s.graph.arc_mut(arc).unwrap().consumption[4] = q;
        arrivals.push(arc);
    }
    s.graph.add_arc(1, 2, 0.).unwrap();
    let problem = Problem::set_partitioning(s.graph, 1.).unwrap();
    for enumerate in [false, true] {
        let mut options = Options::default();
        options.cg.stabilize = false;
        let parent = model::solve(problem.clone(), options.clone()).unwrap();
        let snapshot = if enumerate {
            let node = parent
                .snapshot
                .enumerate(&EnumerationOptions::default())
                .unwrap();
            assert!(node.enumeration_report().complete);
            node
        } else {
            parent.snapshot.solve(options.clone()).unwrap().snapshot
        };
        for upper in [false, true] {
            // The resource on arrival at vertex 1, over both arcs covering it.
            let child = resource_branch(&snapshot, arrivals.clone(), 1., upper)
                .solve(options.clone())
                .unwrap();
            let fresh = resource_branch(
                &cold(problem.clone(), &options),
                arrivals.clone(),
                1.,
                upper,
            )
            .solve(options.clone())
            .unwrap();
            assert_eq!(child.master.status, fresh.master.status);
            // The network model always enforces resource branches in pricing,
            // so the upper child can wait at the new lower window on the
            // cheap route; the `accumulated-resource-master` feature only
            // affects the removed routing root.
            let _ = ACCUMULATED_RESOURCE_IN_PRICING;
            let expected = 1.;
            assert!(
                (fresh.master.lp - expected).abs() < 1e-8,
                "upper={upper}: {fresh:?}"
            );
            assert!(
                (child.master.lp - fresh.master.lp).abs() < 1e-8,
                "enumerate={enumerate} upper={upper}: {child:?}"
            );
            assert_eq!(snapshot.node_depth(), 0);
        }
        let restored = snapshot.solve(options.clone()).unwrap();
        assert_eq!(restored.master.status, parent.master.status);
        assert!((restored.master.lp - parent.master.lp).abs() < 1e-8);
    }
}

#[cfg(feature = "clp")]
#[test]
fn accumulated_branch_registers_one_master_row_in_both_build_modes() {
    use baldes::master::model::{self, Options, Problem};
    let mut s = solver(ResourceKind::Disposable);
    let arc = s.graph.add_arc(0, 1, 1.).unwrap();
    s.graph.add_arc(1, 2, 0.).unwrap();
    let problem = Problem::set_partitioning(s.graph, 1.).unwrap();
    let rows = problem.rows.len();
    let mut options = Options::default();
    options.cg.stabilize = false;
    let parent = model::solve(problem, options.clone()).unwrap();
    let child = resource_branch(&parent.snapshot, vec![arc], 1., false);
    let branches: Vec<_> = child.resource_group_branches().collect();
    assert_eq!(branches.len(), 1);
    let (network, row, group) = branches[0];
    assert_eq!((network, row), (0, rows));
    assert_eq!(group.arcs(), [arc]);
    assert_eq!(parent.snapshot.resource_group_branches().count(), 0);
    // The cheap route's head resource (0 + 0) satisfies the lower side.
    let solved = child.solve(options).unwrap();
    assert!((solved.master.lp - parent.master.lp).abs() < 1e-8);
}

#[test]
#[cfg(all(feature = "clp", feature = "accumulated-resource-master"))]
fn impossible_resource_window_remains_a_master_only_resource_branch() {
    use baldes::master::model::{self, Options, Problem};
    let mut s = solver(ResourceKind::Disposable);
    let arc = s.graph.add_arc(0, 1, 1.).unwrap();
    s.graph.add_arc(1, 2, 0.).unwrap();
    let problem = Problem::set_partitioning(s.graph, 1.).unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    let parent = model::solve(problem, options.clone()).unwrap();
    // Head upper bound is 3: the upper child excludes every use of this arc,
    // and with it the only route covering vertex 1.
    let child = resource_branch(&parent.snapshot, vec![arc], 4., true)
        .solve(options)
        .unwrap();
    assert_eq!(
        child.master.status,
        baldes::master::cg::Status::Infeasible,
        "{child:?}"
    );
}
