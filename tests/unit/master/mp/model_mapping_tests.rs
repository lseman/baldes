use super::*;

fn triangle() -> Problem<2> {
    format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem()
    .unwrap()
}

fn options() -> Options {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options
}

fn variable() -> MappedBranchVariable {
    MappedBranchVariable {
        name: "cross-network-flow".into(),
        priority: 2.,
        networks: vec![
            MappedNetworkVariable {
                network: 7,
                arcs: vec![(0, 1.), (2, 1.)],
                per_route: 0.,
            },
            MappedNetworkVariable {
                network: 42,
                arcs: vec![],
                per_route: 1.,
            },
        ],
        masters: vec![(99, -1.)],
    }
}

#[test]
fn mapped_cross_network_variable_includes_fixed_routes_and_signed_static_terms() {
    let static_variable = MasterVariable {
        id: 99,
        cost: 0.,
        bounds: Bounds {
            lower: 1.,
            upper: 1.,
        },
        integer: true,
        entries: vec![],
    };
    let output = solve_with_master_variables(triangle(), vec![static_variable], options()).unwrap();
    let mapping = variable();
    assert!((mapping.value(&output).unwrap() - 0.5).abs() < 1e-8);
    let original = output.snapshot.state.shared_rows.len();
    let children = children(&output, &[mapping.clone()], 1e-6)
        .unwrap()
        .unwrap();
    for (child, upper) in children.iter().zip([true, false]) {
        let row = &child.state.shared_rows[original];
        assert_eq!(
            row.sense,
            if upper {
                RowSense::LessEqual
            } else {
                RowSense::GreaterEqual
            }
        );
        assert_eq!(row.rhs, if upper { 0. } else { 1. });
        assert!(child.state.master_variables[0]
            .entries
            .contains(&(original, -1.)));
        let solved = child.solve(options()).unwrap();
        let value = mapping.value(&solved).unwrap();
        assert!(if upper {
            value <= 1e-7
        } else {
            value >= 1. - 1e-7
        });
        assert!(solved.master.lp >= 2. - 1e-7);
    }
    assert_eq!(output.snapshot.solve(options()).unwrap().master.lp, 1.5);
    let restricted = output.snapshot.fix_route(0, 1.).unwrap();
    {
        let fixed_output = restricted.solve(options()).unwrap();
        let mapped = mapping
            .apply(
                &fixed_output.snapshot,
                Bounds {
                    lower: -3.,
                    upper: 3.,
                },
            )
            .unwrap();
        let last = mapped.state.shared_rows.len() - 1;
        let fixed_sum: f64 = fixed_output
            .snapshot
            .state
            .fixed_routes
            .iter()
            .map(|f| {
                let c = mapping
                    .networks
                    .iter()
                    .find(|c| c.network == f.route.network)
                    .unwrap();
                f.quantity
                    * f.route.route.arcs.iter().fold(c.per_route, |s, id| {
                        s + c.arcs.iter().find(|(a, _)| a == id).map_or(0., |(_, a)| *a)
                    })
            })
            .sum();
        assert_eq!(
            mapped.master.rhs[mapped.state.shared_offset + last],
            3. - fixed_sum
        );
    }
}

#[test]
fn malformed_mappings_are_rejected_even_with_zero_tree_budget() {
    for mutation in 0..5 {
        let mut mapping = variable();
        mapping.masters.clear();
        match mutation {
            0 => mapping.networks[0].network = 987,
            1 => mapping.networks[0].arcs[0].0 = 987,
            2 => mapping.networks[0].arcs[0].1 = 0.5,
            3 => mapping.networks.push(mapping.networks[0].clone()),
            _ => mapping.priority = f64::NAN,
        }
        let tree = AssignmentTreeOptions {
            mapped_variables: vec![mapping],
            node_limit: 0,
            ..Default::default()
        };
        assert!(solve_assignment_tree(triangle(), tree).is_err());
    }
    let mut mapping = variable();
    mapping.networks.clear();
    let continuous = MasterVariable {
        id: 99,
        cost: 0.,
        bounds: Bounds {
            lower: 0.,
            upper: 1.,
        },
        integer: false,
        entries: vec![],
    };
    assert!(solve_assignment_tree_with_master_variables(
        triangle(),
        vec![continuous],
        AssignmentTreeOptions {
            mapped_variables: vec![mapping],
            node_limit: 0,
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn registered_variable_tree_preserves_original_integer_optimum_and_json_mapping() {
    let mut mapping = variable();
    mapping.masters.clear();
    let json = serde_json::json!({"tree": true, "mapped_variables": [mapping]});
    let configured = format::SolveOptions::from_json(&json.to_string()).unwrap();
    let mut tree = configured.tree_options().unwrap();
    tree.node = options();
    let result = solve_assignment_tree(triangle(), tree).unwrap();
    assert_eq!(result.status, crate::master::tree::TreeStatus::Optimal);
    assert_eq!(result.incumbent, 2.);
}

#[test]
fn shared_row_mapping_keeps_network_identity_and_declines_rounded_integer_sums() {
    let output = solve_with_master_variables(triangle(), vec![], options()).unwrap();
    let mapping = MappedBranchVariable::from_shared_row(
        "packing-edge-assignment",
        3.,
        &output.snapshot,
        0,
        &[42],
        &[],
    )
    .unwrap();
    assert_eq!(mapping.networks.len(), 1);
    assert_eq!(mapping.networks[0].network, 42);
    assert!((mapping.value(&output).unwrap() - 0.5).abs() < 1e-8);
    assert_eq!(exact_coefficient([0.5, 0.5].into_iter()).unwrap(), 1.);
    assert!(exact_coefficient([0.1, 0.9].into_iter()).is_err());
    assert!(
        MappedBranchVariable::from_shared_row("bad", 1., &output.snapshot, 0, &[42, 42], &[])
            .is_err()
    );
}
