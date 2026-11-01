use baldes::{
    graph::Window,
    master::{
        cg::BoundMode,
        model::{self, Options, Problem, ResourceGroupMember},
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
    let problem = Problem::set_partitioning(graph, 1.).unwrap();
    let branch = |node: &model::NodeSnapshot<5>, upper| {
        node.with_resource_group_branch(
            0,
            4,
            1. + 1e-6,
            upper,
            &[ResourceGroupMember {
                network: 0,
                arcs: arcs.clone(),
            }],
        )
        .unwrap()
    };
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for direction in [0., 1., 2.] {
            let mut options = Options::default();
            options.cg.bound_mode = mode;
            options.cg.stabilize = false;
            options
                .pricing_options
                .push(("BALDESuseBidirectionalSearch".to_owned(), direction));
            let parent = model::solve(problem.clone(), options.clone()).unwrap();
            assert!((parent.master.lp - 1.).abs() < 1e-8);
            let mut cold_options = options.clone();
            cold_options.cg.time_limit = std::time::Duration::ZERO;
            let cold = model::solve(problem.clone(), cold_options)
                .unwrap()
                .snapshot;
            for upper in [false, true] {
                let child = branch(&parent.snapshot, upper)
                    .solve(options.clone())
                    .unwrap();
                let fresh = branch(&cold, upper).solve(options.clone()).unwrap();
                let expected = if upper { 2. } else { 1. };
                assert!(
                    (fresh.master.lp - expected).abs() < 1e-8,
                    "mode={mode:?} direction={direction} upper={upper}: {fresh:?}"
                );
                assert_eq!(child.master.status, fresh.master.status);
                assert!((child.master.lp - expected).abs() < 1e-8, "{child:?}");
                // Positive routes use the alternative permitted by the side.
                for (route, &value) in child.routes.iter().zip(&child.values) {
                    if value > 1e-6 {
                        assert!(route.route.arcs.contains(&arcs[usize::from(upper)]));
                    }
                }
                assert_eq!(parent.snapshot.node_depth(), 0);
            }
            let restored = parent.snapshot.solve(options.clone()).unwrap();
            assert!((restored.master.lp - 1.).abs() < 1e-8);
        }
    }
}
