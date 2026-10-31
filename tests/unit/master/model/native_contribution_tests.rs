use super::*;

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
