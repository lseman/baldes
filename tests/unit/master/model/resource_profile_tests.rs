use super::*;
use crate::master::LpMaster;

fn root() -> NodeSnapshot<2> {
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem()
    .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    solve(problem, options).unwrap().snapshot
}

#[test]
fn unchanged_profile_refresh_preserves_shared_registries() {
    let parent = root();
    let mut child = parent.fork().unwrap();
    assert!(child.master.solved());
    let touched = child.state.oracles.iter().map(|o| o.network.id).collect();
    assert!(!refresh(&mut child, &parent.state, &touched).unwrap());
    assert!(
        child.master.solved(),
        "unchanged coefficients must not invalidate the solved LP"
    );
    assert!(Arc::ptr_eq(&child.state.routes, &parent.state.routes));
    assert!(Arc::ptr_eq(&child.state.tags, &parent.state.tags));
}

#[test]
fn synchronized_profile_refresh_detaches_once_and_preserves_parent_tags() {
    let parent = root();
    let mut child = parent.fork().unwrap();
    child
        .state
        .oracles
        .iter_mut()
        .find(|o| o.network.id == 7)
        .unwrap()
        .synchronized_windows
        .push((3, 0, Window::new(5., 10.).unwrap()));
    let touched = HashSet::from([7]);
    assert!(!refresh(&mut child, &parent.state, &touched).unwrap());
    assert!(!Arc::ptr_eq(&child.state.routes, &parent.state.routes));
    assert!(!Arc::ptr_eq(&child.state.tags, &parent.state.tags));
    let tag = child
        .state
        .routes
        .iter()
        .position(|r| r.network == 7 && r.route.arcs == [4, 5])
        .unwrap();
    assert_eq!(child.state.routes[tag].route.resources[0], 5.);
    assert_eq!(parent.state.routes[tag].route.resources[0], 0.);
    for (tag, route) in parent.state.routes.iter().enumerate() {
        assert_eq!(
            parent.state.tags[&(route.network, route.route.identity())],
            tag as i32
        );
    }
    for (tag, route) in child.state.routes.iter().enumerate() {
        assert_eq!(
            child.state.tags[&(route.network, route.route.identity())],
            tag as i32
        );
        if route.network != 7 {
            assert!(Arc::ptr_eq(route, &parent.state.routes[tag]));
        }
    }
    let sibling = child.fork().unwrap();
    assert!(!refresh(&mut child, &sibling.state, &touched).unwrap());
    assert!(Arc::ptr_eq(&child.state.routes, &sibling.state.routes));
    assert!(Arc::ptr_eq(&child.state.tags, &sibling.state.tags));
}

#[test]
fn sparse_refresh_merge_matches_full_row_replacement_with_legacy_metadata() {
    let cases = [
        (vec![], vec![]),
        (vec![], vec![(0, 2.), (5, 3.)]),
        (vec![(0, 2.), (5, 3.)], vec![]),
        (
            vec![(0, 2.), (2, 3.), (7, 4.)],
            vec![(0, 2.), (3, 8.), (7, 9.)],
        ),
        (
            vec![(5, 1.), (0, 2.), (5, 7.), (0, 3.)],
            vec![(0, 4.), (2, 9.), (5, 7.)],
        ),
        (vec![(0, -0.), (2, 4.)], vec![(0, 0.), (2, 4.)]),
    ];
    for (before, after) in cases {
        let ordered = ordered_coefficients(&before);
        let stable = before.windows(2).all(|pair| pair[0].0 < pair[1].0);
        assert_eq!(matches!(ordered, Cow::Borrowed(_)), stable);
        let previous: BTreeMap<_, _> = before.iter().copied().collect();
        assert_eq!(
            ordered.as_ref(),
            previous.iter().map(|(&r, &v)| (r, v)).collect::<Vec<_>>()
        );
        let expected: BTreeMap<_, _> = before
            .iter()
            .map(|&(row, _)| (row, 0.))
            .chain(after.iter().copied())
            .collect();
        let mut actual = previous.clone();
        let pairs: Vec<_> = merged_coefficients(&ordered, &after).collect();
        assert_eq!(pairs.len(), expected.len());
        for (row, old, new) in pairs {
            assert_eq!(old, previous.get(&row).copied().unwrap_or(0.));
            assert_eq!(new, expected[&row]);
            if !stable || old != new {
                actual.insert(row, new);
            }
        }
        actual.retain(|_, value| *value != 0.);
        let expected: BTreeMap<_, _> = expected
            .into_iter()
            .filter(|&(_, value)| value != 0.)
            .collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn resource_enforcement_switch_preserves_master_profiles_and_safe_completion() {
    use crate::master::branching::ACCUMULATED_RESOURCE_IN_PRICING;
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 3.).unwrap();
    }
    let cheap = graph.add_arc(0, 1, 1.).unwrap();
    graph.arc_mut(cheap).unwrap().consumption[0] = 0.5;
    let alternative = graph.add_arc(0, 1, 2.).unwrap();
    graph.arc_mut(alternative).unwrap().consumption[0] = 2.5;
    graph.add_arc(1, 2, 0.).unwrap();
    let mut configured = Options::default();
    configured.cg.stabilize = false;
    configured.preprocessing.enabled = false;
    configured.reduced_cost_fixing = false;
    configured.gap_enumeration = false;
    let parent = solve(
        Problem::set_partitioning(graph, 1.).unwrap(),
        configured.clone(),
    )
    .unwrap();
    let child = parent
        .snapshot
        .with_resource_group_branch(
            0,
            0,
            1.,
            true,
            &[ResourceGroupMember {
                network: 0,
                arcs: vec![cheap, alternative],
            }],
        )
        .unwrap();
    let oracle = &child.state.oracles[0];
    assert_eq!(
        oracle.solver.accumulated_resource_branches().len(),
        if ACCUMULATED_RESOURCE_IN_PRICING {
            2
        } else {
            0
        }
    );
    let cheap_arc = oracle
        .solver
        .graph
        .arcs
        .iter()
        .find(|a| a.id == cheap)
        .unwrap();
    assert_eq!(
        cheap_arc.bounds[0].map(|w| w.lower),
        ACCUMULATED_RESOURCE_IN_PRICING.then_some(1. - 1e-6)
    );
    let retained = child
        .state
        .routes
        .iter()
        .find(|r| r.route.arcs.contains(&cheap))
        .unwrap();
    assert_eq!(
        retained.route.resource_profile[1][0],
        if ACCUMULATED_RESOURCE_IN_PRICING {
            1. - 1e-6
        } else {
            0.5
        }
    );
    assert_eq!(
        super::super::resource_groups::entries(oracle, &retained.route)[0].1,
        if ACCUMULATED_RESOURCE_IN_PRICING {
            0.
        } else {
            1.
        }
    );
    if !ACCUMULATED_RESOURCE_IN_PRICING {
        assert!(!super::super::lifecycle::integer_route_feasible(oracle, &retained.route).unwrap());
    }
    let solved = child.solve(configured.clone()).unwrap();
    let optimum = if ACCUMULATED_RESOURCE_IN_PRICING {
        1.
    } else {
        2.
    };
    assert!(solved.master.bound <= optimum + 1e-6, "{:?}", solved.master);
    if ACCUMULATED_RESOURCE_IN_PRICING {
        assert_eq!(solved.master.status, cg::Status::Optimal);
        assert!((solved.master.lp - optimum).abs() < 1e-6);
    } else {
        assert_eq!(
            solved.master.status,
            cg::Status::Stalled,
            "{:?}",
            solved.master
        );
    }
    // Full original-path enumeration proves feasibility independently of the
    // relaxed pricing result, and must not invent waiting in master-only mode.
    let complete = solved
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    assert!(complete.enumeration_report().complete);
    let exact = complete.solve(configured).unwrap();
    assert_eq!(
        exact.master.status,
        cg::Status::Optimal,
        "{:?}",
        exact.master
    );
    assert!((exact.master.lp - optimum).abs() < 1e-6);
    assert!((exact.master.bound - optimum).abs() < 1e-6);
    let expected_arc = if ACCUMULATED_RESOURCE_IN_PRICING {
        cheap
    } else {
        alternative
    };
    assert!(exact
        .routes
        .iter()
        .zip(&exact.values)
        .any(|(r, &q)| q > 0.9 && r.route.arcs.contains(&expected_arc)));
    assert!(parent.snapshot.state.oracles[0]
        .solver
        .accumulated_resource_branches()
        .is_empty());
    assert_eq!(
        parent
            .snapshot
            .state
            .routes
            .iter()
            .find(|r| r.route.arcs.contains(&cheap))
            .unwrap()
            .route
            .resource_profile[1][0],
        0.5
    );
}
