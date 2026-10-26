use baldes::{
    graph::Window,
    master::{
        branching::{AccumulatedResourceGroupBranch, BranchConstraint},
        routing::{self, BoundMode, RootOptions, RoutingContext},
    },
    Graph,
};

#[test]
fn grouped_children_price_parallel_alternatives_and_preserve_parent_snapshots() {
    let mut graph = Graph::<5>::new(3, 5).unwrap();
    for v in 0..3 {
        for r in 0..5 {
            graph.set_bounds(v, r, 0., 3.).unwrap();
        }
    }
    let mut arcs = vec![];
    for (cost, quantity) in [(1., 0.5), (2., 2.5)] {
        let arc = graph.add_arc(0, 1, cost).unwrap();
        let data = graph.arc_mut(arc).unwrap();
        data.consumption[4] = quantity;
        data.bounds[4] = Some(Window {
            lower: 0.,
            upper: quantity,
        });
        arcs.push(arc);
    }
    graph.add_arc(1, 2, 0.).unwrap();
    let problem = RoutingContext::set_partitioning(graph, 1.).unwrap();
    let group = AccumulatedResourceGroupBranch::new(1, arcs, 4, 1. + 1e-6, false).unwrap();
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for enumerate in [false, true] {
            for direction in [0., 1., 2.] {
                let mut options = RootOptions::default();
                options.capture_state = true;
                options.bound_mode = mode;
                options.enumerate = enumerate;
                options.incumbent = 10.;
                options.cg.stabilize = false;
                options
                    .pricing_options
                    .push(("BALDESuseBidirectionalSearch".to_owned(), direction));
                let parent = routing::solve(problem.clone(), options.clone()).unwrap();
                assert!((parent.master.lp - 1.).abs() < 1e-8);
                let snapshot = parent.snapshot.as_ref().unwrap();
                for upper in [false, true] {
                    let branch =
                        BranchConstraint::AccumulatedResourceGroup(group.with_upper(upper));
                    let child =
                        routing::inherit_branches(snapshot, &[branch.clone()], options.clone())
                            .unwrap();
                    let mut cold = problem.clone();
                    cold.branch(branch.clone()).unwrap();
                    let fresh = routing::solve(cold, options.clone()).unwrap();
                    let expected = if upper { 2. } else { 1. };
                    assert!(
                        (fresh.master.lp - expected).abs() < 1e-8,
                        "mode={mode:?} direction={direction} upper={upper}: {fresh:?}"
                    );
                    assert_eq!(child.master.status, fresh.master.status);
                    if child.master.pruned_by_bound {
                        // A parent's validated global incumbent remains a valid
                        // tree cutoff even when that witness is outside this
                        // child. The driver caps its published bound at it.
                        assert!(child.master.bound <= expected + 1e-8);
                        assert!(child.master.bound >= parent.master.bound - 1e-8);
                        assert!((child.master.bound - child.incumbent).abs() < 1e-8);
                        assert!(child.incumbent <= expected + 1e-8);
                    } else {
                        assert!((child.master.lp - expected).abs() < 1e-8);
                    }
                    for (route, value) in child.route_values() {
                        if value > 1e-6 {
                            assert_eq!(branch.coefficient(&problem, route), 0.);
                        }
                    }
                    assert_eq!(snapshot.branch_depth(), 0);
                }
                let restored = routing::inherit_branches(snapshot, &[], options.clone()).unwrap();
                assert!((restored.master.lp - 1.).abs() < 1e-8);
            }
        }
    }
}
