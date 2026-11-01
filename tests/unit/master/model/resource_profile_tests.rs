use super::*;

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
    let touched = child.state.oracles.iter().map(|o| o.network.id).collect();
    assert!(!refresh(&mut child, &parent.state, &touched).unwrap());
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
