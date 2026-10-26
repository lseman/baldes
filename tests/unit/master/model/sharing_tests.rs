use super::*;

fn solved() -> Output<2> {
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
    options.preprocessing.enabled = false;
    solve(problem, options).unwrap()
}

#[test]
fn forks_share_immutable_inputs_and_routes_but_isolate_row_edits_and_outputs() {
    let mut root = solved();
    assert!(!root.snapshot.state.routes.is_empty());
    let oracle = &root.snapshot.state.oracles[0];
    let network = oracle.network.id;
    let entries = oracle.network.arc_entries.as_ref().clone();
    let original_route = root.snapshot.state.routes[0].as_ref().clone();
    let mut child = root
        .snapshot
        .with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 10.,
            },
            &[RowCoefficients {
                network,
                arcs: vec![1.; oracle.network.graph.arcs.len()],
                per_route: 1.,
            }],
        )
        .unwrap();
    let child_oracle = &mut child.state.oracles[0];
    assert!(Arc::ptr_eq(
        &oracle.network.graph,
        &child_oracle.network.graph
    ));
    assert!(Arc::ptr_eq(&oracle.index, &child_oracle.index));
    assert!(!Arc::ptr_eq(
        &oracle.network.arc_entries,
        &child_oracle.network.arc_entries
    ));
    assert_eq!(oracle.network.arc_entries.as_ref(), &entries);
    assert_eq!(
        child_oracle.network.arc_entries[0].len(),
        entries[0].len() + 1
    );
    assert!(Arc::ptr_eq(
        &root.snapshot.state.routes[0],
        &child.state.routes[0]
    ));
    child_oracle.solver.graph.vertices[0].bounds[0].upper = 42.;
    assert_ne!(oracle.solver.graph.vertices[0].bounds[0].upper, 42.);

    // Public outputs remain owned values even though private node payloads share.
    root.routes[0].route.nodes.clear();
    assert_eq!(root.snapshot.state.routes[0].route, original_route.route);
    let sibling = root.snapshot.fork().unwrap();
    assert_eq!(
        sibling.state.oracles[0].network.arc_entries.as_ref(),
        &entries
    );
    assert_eq!(sibling.state.routes[0].route, original_route.route);
}

#[test]
fn repeated_network_pricing_reuses_cost_buffers_and_keeps_the_same_answer() {
    let root = solved();
    let mut node = root.snapshot.fork().unwrap();
    let phase = node.state.oracles[0]
        .solver
        .phases
        .iter()
        .position(|p| p.exact())
        .unwrap();
    let first = node.state.price(phase, &root.duals).unwrap();
    let buffers: Vec<_> = node
        .state
        .oracles
        .iter()
        .map(|oracle| (oracle.costs.as_ptr(), oracle.solver.variable_costs.as_ptr()))
        .collect();
    let second = node.state.price(phase, &root.duals).unwrap();
    assert_eq!(first.minimum, second.minimum);
    assert_eq!(first.exact, second.exact);
    assert_eq!(first.columns.len(), second.columns.len());
    for (oracle, &(costs, cached)) in node.state.oracles.iter().zip(&buffers) {
        assert_eq!(oracle.costs.as_ptr(), costs);
        assert_eq!(oracle.solver.variable_costs.as_ptr(), cached);
    }
}
