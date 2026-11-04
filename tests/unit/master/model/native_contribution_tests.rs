use super::*;

#[test]
fn preserving_raw_first_route_does_not_preserve_gradient_ties() {
    let mut graph = Graph::<2>::new(6, 0).unwrap();
    for (tail, head, cost) in [
        (0, 1, 1e16),
        (1, 2, -1.),
        (2, 5, -1e16),
        (0, 3, -2.),
        (3, 5, 0.),
        (0, 4, -2.),
        (4, 5, 0.),
    ] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let problem = Problem {
        rows: vec![
            Row {
                sense: RowSense::Equal,
                rhs: 0.,
            };
            3
        ],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![
                vec![],
                vec![(0, 1.)],
                vec![],
                vec![(1, 1.)],
                vec![],
                vec![(2, 1.)],
                vec![],
            ],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.preprocessing.enabled = false;
    let (_, mut state, _) =
        super::super::solve::prepare_with_seed_routes(problem, vec![], options, &[], &[], &[])
            .unwrap();
    let mut duals = vec![0.; state.rhs.len()];
    duals[state.shared_offset] = 1.;
    let raw_first = (1e16 + (-1. - duals[state.shared_offset])) - 1e16;
    assert_eq!(raw_first, -2.);
    let route = |nodes, arcs| Route {
        coupled_values: vec![],
        nodes,
        arcs,
        cost: raw_first,
        original_cost: 0.,
        resources: vec![],
        resource_profile: vec![],
    };
    let first = route(vec![0, 1, 2, 5], vec![0, 1, 2]);
    let second = route(vec![0, 3, 5], vec![3, 4]);
    let third = route(vec![0, 4, 5], vec![5, 6]);
    let mut adapter = NetworkPricingAdapter {
        state: &mut state,
        tolerance: 1e-6,
    };
    let collect = |adapter: &mut NetworkPricingAdapter<'_, 2>, routes| {
        crate::master::pricing::Networks::collect(adapter, 0, routes, 0., &duals).unwrap()
    };
    let original = collect(
        &mut adapter,
        vec![first.clone(), second.clone(), third.clone()],
    );
    let reordered = collect(&mut adapter, vec![first, third, second]);
    assert_eq!(original.columns[0].reduced_cost, -1.);
    assert_eq!(original.columns[1].reduced_cost, -2.);
    assert_eq!(original.columns[2].reduced_cost, -2.);
    assert_eq!(original.best_route.unwrap().arcs, [3, 4]);
    assert_eq!(reordered.best_route.unwrap().arcs, [5, 6]);
    assert_ne!(original.best_entries, reordered.best_entries);
}

#[test]
fn finite_network_batches_replay_native_contributions_and_zero_cost_quantities() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/native_network_contributions.json")).unwrap();
    assert_eq!(
        fixture["sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 40);
    for pair in cases.chunks(2) {
        let networks = pair
            .iter()
            .enumerate()
            .map(|(index, case)| {
                let mut graph = Graph::<2>::new(3, 0).unwrap();
                graph.add_arc(0, 1, case["cost"].as_f64().unwrap()).unwrap();
                graph.add_arc(1, 2, 0.).unwrap();
                Network {
                    id: [7, 42][index],
                    graph,
                    multiplicity: Window {
                        lower: case["lower"].as_f64().unwrap(),
                        upper: case["upper"].as_f64().unwrap(),
                    },
                    arc_entries: vec![vec![]; 2],
                    route_entries: vec![],
                    activity_entries: vec![],
                }
            })
            .collect();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.artificial_cost = 100.;
        options.preprocessing.enabled = false;
        let root = solve(
            Problem {
                rows: vec![],
                networks,
            },
            options,
        )
        .unwrap();
        let mut node = root.snapshot.fork().unwrap();
        let phase = node.state.oracles[0]
            .solver
            .phases
            .iter()
            .position(|phase| phase.exact())
            .unwrap();
        let answer = node
            .state
            .price(phase, &vec![0.; node.state.rhs.len()])
            .unwrap();
        assert!(answer.exact);
        let expected: f64 = pair
            .iter()
            .map(|case| case["native"]["contribution"].as_f64().unwrap())
            .sum();
        assert_eq!(node.state.correction, Some(expected));
        assert_eq!(root.master.lp, expected);
        // A network admitted by its variable box has no best path, so the
        // reference subgradient (upper multiplicity times that path) is
        // unknown and the batch publishes none.
        let admitted = pair.iter().any(|case| {
            case["cost"].as_f64().unwrap() >= 0. && case["lower"].as_f64().unwrap() == 0.
        });
        if admitted {
            assert!(node.state.gradient.is_none());
            continue;
        }
        let gradient = node.state.gradient.as_ref().unwrap();
        // The contribution uses the sign-dependent quantity, but the
        // stabilization subgradient weights the best path by the upper
        // multiplicity (largest convexity right-hand side) regardless of sign.
        for (oracle, case) in node.state.oracles.iter().zip(pair) {
            let upper = case["upper"].as_f64().unwrap();
            let weight = if upper.is_finite() {
                upper
            } else {
                case["native"]["quantity"].as_f64().unwrap()
            };
            for &(row, coefficient) in &oracle.route_entries {
                assert_eq!(gradient[row], node.state.rhs[row] - weight * coefficient);
            }
        }
    }
}
