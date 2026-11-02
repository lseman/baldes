use super::*;

#[test]
fn aggregate_admission_matches_original_separator_fixtures() {
    // Actual GenAggrSubProbVar separator at ELF 0x1a83a0, reference SHA256
    // 103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6.
    // See internal/branching-goal-20261101/registration/aggregate-admission.cpp.
    let fixtures: Vec<(f64, bool)> =
        serde_json::from_str(include_str!("aggregate_admission_fixtures.json")).unwrap();
    for (value, expected) in fixtures {
        assert_eq!(aggregate_fractional(value, 1e-8), expected, "{value}");
    }
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        9_007_199_254_740_992.,
    ] {
        assert!(!aggregate_fractional(value, 1e-8));
    }
}

#[test]
fn aggregate_admission_validates_names_and_progressing_fractional_thresholds() {
    let v = variable();
    for threshold in [1e-8, 0.02, 0.5] {
        validate_aggregate_admission([&v], &BTreeMap::from([(v.name.clone(), threshold)])).unwrap();
    }
    for threshold in [0., -1., 0.500001, f64::NAN, f64::INFINITY] {
        assert!(
            validate_aggregate_admission([&v], &BTreeMap::from([(v.name.clone(), threshold)]))
                .is_err()
        );
        assert!(!aggregate_fractional(0.5, threshold));
    }
    assert!(
        validate_aggregate_admission([&v], &BTreeMap::from([("missing".into(), 1e-8)])).is_err()
    );
    assert!(aggregate_fractional(0.5, 0.5));
    assert!(!aggregate_fractional(0.49, 0.5));
}

#[test]
fn aggregate_metadata_selects_raw_fraction_admission_without_changing_ordinary_mappings() {
    let mut output = solve(triangle(), options()).unwrap();
    let tag = output
        .routes
        .iter()
        .position(|route| route.network == 42)
        .unwrap();
    for (value, ordinary, aggregate) in [
        (5e-7_f64, false, true),
        (0.5, true, true),
        (-0.5, true, false),
    ] {
        output.values.fill(0.);
        output.values[tag] = value.abs();
        let mapping = MappedBranchVariable {
            name: "aggregate".into(),
            priority: 1.,
            masters: vec![],
            networks: vec![MappedNetworkVariable {
                network: 42,
                arcs: vec![],
                activities: vec![],
                per_route: value.signum(),
            }],
        };
        let candidates = |admission: &BTreeMap<String, f64>| {
            super::super::model_strong::mapped_proposals(
                &output,
                &ModelTreeOptions {
                    tolerance: 0.1,

                    branching_registry: {
                        let mut __branch_registry =
                            crate::master::model::BranchingRegistry::default();
                        __branch_registry.mapped_variables = vec![mapping.clone()];
                        __branch_registry.aggregate_admission = admission.clone();
                        __branch_registry
                    },
                    ..Default::default()
                },
            )
            .unwrap()
        };
        assert_eq!(!candidates(&BTreeMap::new()).is_empty(), ordinary);
        let admission = BTreeMap::from([(mapping.name.clone(), 1e-8)]);
        assert_eq!(!candidates(&admission).is_empty(), aggregate);
    }
}

#[test]
fn mapped_display_name_does_not_select_the_path_count_family_direction() {
    let output = solve(triangle(), options()).unwrap();
    let mapping = MappedBranchVariable {
        name: "paths:display-only".into(),
        priority: 1.,
        masters: vec![],
        networks: vec![MappedNetworkVariable {
            network: 42,
            arcs: vec![],
            activities: vec![],
            per_route: 1.,
        }],
    };
    let mut controls = ModelTreeOptions::default();
    controls.branching_registry.branching = NetworkBranchingPolicy::ResourceFirst;
    controls.branching_registry.mapped_variables = vec![mapping];
    let proposals = super::super::model_strong::mapped_proposals(&output, &controls).unwrap();
    let selected = super::super::model_strong::choose_proposals(
        &output,
        &controls,
        &mut super::super::model_strong::History::default(),
        2.,
        Instant::now() + std::time::Duration::from_secs(10),
        proposals,
    )
    .unwrap()
    .unwrap();
    assert_eq!(selected.direction, None);
}

#[test]
fn family_selection_rules_match_binary_comparator_fixtures() {
    // Actual comparator at ELF 0x1a5260 in SHA256 103caf4a9c58d6141e935848294bff166
    // 77f400323466dc23cd4b9e5c15889d6. The audit's rules-probe.cpp uses its original
    // fraction/directed-fraction routines, not rewritten numeric getters.
    let fixtures: Vec<(usize, f64, f64, u8, u8)> =
        serde_json::from_str(include_str!("branch_selection_fixtures.json")).unwrap();
    let rules = [
        BranchSelectionRule::FirstFound,
        BranchSelectionRule::HighestPriority,
        BranchSelectionRule::MostFractional,
        BranchSelectionRule::LeastFractional,
        BranchSelectionRule::FractionWeightedPriority,
        BranchSelectionRule::ClosestToUpper,
        BranchSelectionRule::ClosestToLower,
    ];
    for (rule, x, y, ab, ba) in fixtures {
        let a = BranchSelection {
            rule: rules[rule - 1],
            variable_priority: 100.,
            default_factor: 100.,
            ..BranchSelection::default()
        };
        let b = BranchSelection {
            rule: a.rule,
            ..BranchSelection::default()
        };
        assert_eq!(
            selection_before(&a, x, 0, &b, y, 1),
            ab != 0,
            "{rule}: {x}, {y}"
        );
        assert_eq!(
            selection_before(&b, y, 1, &a, x, 0),
            ba != 0,
            "{rule}: {y}, {x}"
        );
    }
}

#[test]
fn mapped_family_selection_separates_family_and_variable_priorities() {
    let mut a = variable();
    a.name = "a".into();
    a.priority = 1.;
    let mut b = a.clone();
    b.name = "b".into();
    let metadata = BTreeMap::from([
        (
            a.name.clone(),
            BranchSelection {
                family: "original".into(),
                variable_priority: 100.,
                default_factor: 100.,
                ..BranchSelection::default()
            },
        ),
        (
            b.name.clone(),
            BranchSelection {
                family: "original".into(),
                ..BranchSelection::default()
            },
        ),
    ]);
    validate_selection(&[a.clone(), b.clone()], &metadata).unwrap();
    let ranked = rank_candidates_with_selection([(&a, 0.1), (&b, 0.5)].into_iter(), &metadata);
    assert_eq!(ranked[0].0.name, "b");
    let ranked = rank_candidates_with_selection([(&a, 0.5), (&b, 0.5)].into_iter(), &metadata);
    assert_eq!(ranked[0].0.name, "b");
    b.priority = 2.;
    assert!(validate_selection(&[a.clone(), b.clone()], &metadata).is_err());
    b.priority = 1.;
    let mut unknown = metadata.clone();
    unknown.insert("missing".into(), BranchSelection::default());
    assert!(validate_selection(&[a.clone(), b.clone()], &unknown).is_err());
    let mut invalid = metadata;
    invalid.get_mut("b").unwrap().default_factor = f64::NAN;
    assert!(validate_selection(&[a, b], &invalid).is_err());
}

#[test]
fn differing_family_rules_skip_primary_keys() {
    let a = BranchSelection {
        rule: BranchSelectionRule::HighestPriority,
        variable_priority: 100.,
        default_factor: 2.,
        ..BranchSelection::default()
    };
    let b = BranchSelection::default();
    assert!(!selection_before(&a, 0.5, 0, &b, 0.1, 1));
    assert!(selection_before(&b, 0.1, 1, &a, 0.5, 0));
}

#[test]
fn mixed_selection_metadata_keeps_unregistered_mapping_comparison() {
    let mut a = variable();
    a.name = "a".into();
    let mut b = a.clone();
    b.name = "b".into();
    let metadata = BTreeMap::from([(
        a.name.clone(),
        BranchSelection {
            rule: BranchSelectionRule::HighestPriority,
            variable_priority: 100.,
            default_factor: -10.,
            ..BranchSelection::default()
        },
    )]);
    let ranked = rank_candidates_with_selection([(&a, 0.1), (&b, 0.5)].into_iter(), &metadata);
    assert_eq!(ranked[0].0.name, "b");
}

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

#[test]
fn fixed_activity_uses_only_fixed_routes_in_full_expression_coordinates() {
    let mut output = solve(triangle(), options()).unwrap();
    let route = output
        .routes
        .iter()
        .find(|r| r.network == 42)
        .unwrap()
        .clone();
    output.snapshot.state.fixed_routes = vec![FixedRoute {
        route,
        quantity: 2.,
    }];
    output.values.fill(123.);
    let mapping = MappedBranchVariable {
        name: "fixed-offset".into(),
        priority: 1.,
        networks: vec![MappedNetworkVariable {
            network: 42,
            arcs: vec![],
            activities: vec![],
            per_route: 3.,
        }],
        masters: vec![],
    };
    assert_eq!(mapping.fixed_activity(&output), Some(6.));
    output.snapshot.state.fixed_routes.clear();
    assert_eq!(mapping.fixed_activity(&output), Some(0.));
    let mut static_mapping = mapping;
    static_mapping.masters.push((99, 1.));
    assert_eq!(static_mapping.fixed_activity(&output), None);
}

#[test]
fn fixed_activity_declines_nonrepresentable_products_and_totals() {
    let mut output = solve(triangle(), options()).unwrap();
    let route = output
        .routes
        .iter()
        .find(|r| r.network == 42)
        .unwrap()
        .clone();
    let mut mapping = MappedBranchVariable {
        name: "fixed-offset".into(),
        priority: 1.,
        networks: vec![MappedNetworkVariable {
            network: 42,
            arcs: vec![],
            activities: vec![],
            per_route: 3.,
        }],
        masters: vec![],
    };
    output.snapshot.state.fixed_routes = vec![FixedRoute {
        route: route.clone(),
        quantity: 4_503_599_627_370_497.,
    }];
    assert_eq!(mapping.fixed_activity(&output), None);
    mapping.networks[0].per_route = 1.;
    output.snapshot.state.fixed_routes = vec![
        FixedRoute {
            route: route.clone(),
            quantity: 9_007_199_254_740_992.,
        },
        FixedRoute {
            route,
            quantity: 1.,
        },
    ];
    assert_eq!(mapping.fixed_activity(&output), None);
}

#[test]
fn residual_aggregate_row_avoids_sequential_fixed_cancellation() {
    let mut output = solve(triangle(), options()).unwrap();
    let positive = output
        .routes
        .iter()
        .find(|r| r.network == 42)
        .unwrap()
        .clone();
    let negative = output
        .routes
        .iter()
        .find(|r| r.network == 7)
        .unwrap()
        .clone();
    let large = 9_007_199_254_740_992.;
    output.snapshot.state.fixed_routes = vec![
        FixedRoute {
            route: positive.clone(),
            quantity: large,
        },
        FixedRoute {
            route: positive,
            quantity: 1.,
        },
        FixedRoute {
            route: negative,
            quantity: large,
        },
    ];
    let mapping = MappedBranchVariable {
        name: "fixed-cancellation".into(),
        priority: 1.,
        networks: vec![
            MappedNetworkVariable {
                network: 42,
                arcs: vec![],
                activities: vec![],
                per_route: 2.,
            },
            MappedNetworkVariable {
                network: 7,
                arcs: vec![],
                activities: vec![],
                per_route: -2.,
            },
        ],
        masters: vec![],
    };
    let fixed = mapping.fixed_activity(&output).unwrap();
    assert_eq!(fixed, 2.);
    assert_ne!(((3. - 2. * large) - 2.) + 2. * large, 1.);
    let physical = output.snapshot.master.rhs.len();
    let child = mapping
        .apply_residual(
            &output.snapshot,
            Bounds {
                lower: f64::NEG_INFINITY,
                upper: 1.,
            },
            fixed,
        )
        .unwrap();
    assert_eq!(child.master.rhs[physical], 1.);
    assert_eq!(child.state.rhs[physical], 1.);
    assert_eq!(child.state.shared_rows.last().unwrap().rhs, 3.);
}

#[test]
fn residual_aggregate_activity_is_installed_before_column_coefficients() {
    use crate::graph::coupled::CoupledResource;
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.elementary = true;
    graph.add_arc(0, 1, -1.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![2.],
            weights: vec![1.],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    graph.suppress_coupled_costs();
    let mut output = solve(
        Problem {
            rows: vec![],
            networks: vec![Network {
                id: 42,
                graph,
                multiplicity: Window::new(0., 1.).unwrap(),
                arc_entries: vec![vec![]],
                route_entries: vec![],
                activity_entries: vec![],
            }],
        },
        options(),
    )
    .unwrap();
    output.snapshot.state.fixed_routes.push(FixedRoute {
        route: output.routes[0].clone(),
        quantity: 1.,
    });
    let mapping = MappedBranchVariable {
        name: "fixed-activity".into(),
        priority: 1.,
        masters: vec![],
        networks: vec![MappedNetworkVariable {
            network: 42,
            arcs: vec![],
            activities: vec![(0, 1.)],
            per_route: 0.,
        }],
    };
    let fixed = mapping.fixed_activity(&output).unwrap();
    assert_eq!(fixed, 2.);
    let physical = output.snapshot.master.rhs.len();
    let child = mapping
        .apply_residual(
            &output.snapshot,
            Bounds {
                lower: f64::NEG_INFINITY,
                upper: 1.,
            },
            fixed,
        )
        .unwrap();
    assert_eq!(child.master.rhs[physical], 1.);
    assert_eq!(child.state.shared_rows.last().unwrap().rhs, 1. + fixed);
    let shared = child.state.shared_rows.len() - 1;
    assert!(child.state.oracles[0]
        .activity_rows
        .iter()
        .any(|e| e.row == shared));
    assert!(child
        .warm
        .columns
        .iter()
        .flatten()
        .filter(|c| c.tag >= 0)
        .all(|c| c.entries.iter().any(|&(r, a)| r == physical && a == fixed)));
}

#[test]
fn residual_aggregate_declines_unrepresentable_full_coordinate_endpoint() {
    let output = solve(triangle(), options()).unwrap();
    let mapping = MappedBranchVariable {
        name: "fixed-endpoint".into(),
        priority: 1.,
        masters: vec![],
        networks: vec![MappedNetworkVariable {
            network: 42,
            arcs: vec![],
            activities: vec![],
            per_route: 1.,
        }],
    };
    assert!(mapping
        .apply_residual(
            &output.snapshot,
            Bounds {
                lower: f64::NEG_INFINITY,
                upper: 1.
            },
            9_007_199_254_740_992.
        )
        .is_err());
}

fn variable() -> MappedBranchVariable {
    MappedBranchVariable {
        name: "cross-network-flow".into(),
        priority: 2.,
        networks: vec![
            MappedNetworkVariable {
                network: 7,
                activities: vec![],
                arcs: vec![(0, 1.), (2, 1.)],
                per_route: 0.,
            },
            MappedNetworkVariable {
                network: 42,
                activities: vec![],
                arcs: vec![],
                per_route: 1.,
            },
        ],
        masters: vec![(99, -1.)],
    }
}

#[test]
fn mapped_activity_preserves_fractional_and_product_residuals_after_cancellation() {
    let large = 9_007_199_254_740_992.;
    let product_input = 4_503_599_627_370_497.;
    let product_rounded = 13_510_798_882_111_492.;
    let variables: Vec<_> = [large, large, product_input, product_rounded]
        .into_iter()
        .enumerate()
        .map(|(id, value)| MasterVariable {
            id: id as i32,
            cost: 0.,
            bounds: Bounds {
                lower: value,
                upper: value,
            },
            integer: true,
            entries: vec![],
        })
        .collect();
    let output = solve_with_master_variables(triangle(), variables, options()).unwrap();
    let mut mapping = MappedBranchVariable {
        name: "signed-activity".into(),
        priority: 1.,
        networks: vec![MappedNetworkVariable {
            network: 42,
            activities: vec![],
            arcs: vec![],
            per_route: 1.,
        }],
        masters: vec![(0, 1.), (1, -1.)],
    };
    // The network contributes 0.5. Rounded addition to the large static terms
    // used to erase it, hiding this fractional original variable from branching.
    assert_eq!(mapping.value(&output), Some(0.5));
    mapping.masters.reverse();
    assert_eq!(mapping.value(&output), Some(0.5));
    assert_eq!(
        super::super::model_strong::mapped_proposals(
            &output,
            &ModelTreeOptions {
                branching_registry: {
                    let mut __branch_registry = crate::master::model::BranchingRegistry::default();
                    __branch_registry.mapped_variables = vec![mapping.clone()];
                    __branch_registry
                },
                ..Default::default()
            },
        )
        .unwrap()
        .len(),
        1
    );

    mapping.networks.clear();
    mapping.masters = vec![(2, 3.), (3, -1.)];
    // Exact 3 * (2^52 + 1) is one below its rounded binary64 product.
    assert_eq!(mapping.value(&output), Some(-1.));
    mapping.masters.reverse();
    assert_eq!(mapping.value(&output), Some(-1.));
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
        let tree = ModelTreeOptions {
            node_limit: 0,

            branching_registry: {
                let mut __branch_registry = crate::master::model::BranchingRegistry::default();
                __branch_registry.mapped_variables = vec![mapping];
                __branch_registry
            },
            ..Default::default()
        };
        assert!(crate::master::tree::solve(triangle(), tree).is_err());
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
    assert!(crate::master::tree::solve(
        (triangle(), vec![continuous]),
        ModelTreeOptions {
            node_limit: 0,

            branching_registry: {
                let mut __branch_registry = crate::master::model::BranchingRegistry::default();
                __branch_registry.mapped_variables = vec![mapping];
                __branch_registry
            },
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn registered_variable_tree_preserves_original_integer_optimum_and_json_mapping() {
    let mut mapping = variable();
    mapping.masters.clear();
    let json = serde_json::json!({"tree": true, "branching_registrations": [{"kind":"mapped", "config":mapping}]});
    let configured = format::SolveOptions::from_json(&json.to_string()).unwrap();
    let mut tree = configured.tree_options().unwrap();
    tree.node = options();
    let result = crate::master::tree::solve(triangle(), tree).unwrap();
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

#[test]
fn native_half_incident_arcs_preserve_integer_route_disjunctions() {
    let output = solve(triangle(), options()).unwrap();
    let incoming = MappedBranchVariable {
        name: "ESA_7_1".into(),
        priority: 1000.,
        networks: vec![MappedNetworkVariable {
            network: 7,
            activities: vec![],
            arcs: vec![(0, 1.), (6, 1.)],
            per_route: 0.,
        }],
        masters: vec![],
    };
    let mut incident = incoming.clone();
    incident.networks[0].arcs = vec![(0, 0.5), (1, 0.5), (6, 0.5), (7, 0.5)];
    assert_eq!(
        incoming.value(&output).unwrap(),
        incident.value(&output).unwrap()
    );
    for bounds in [
        Bounds {
            lower: 1.,
            upper: f64::INFINITY,
        },
        Bounds {
            lower: 0.,
            upper: 0.,
        },
    ] {
        let a = incoming
            .apply(&output.snapshot, bounds)
            .unwrap()
            .solve(options())
            .unwrap();
        let b = incident
            .apply(&output.snapshot, bounds)
            .unwrap()
            .solve(options())
            .unwrap();
        assert_eq!(a.master.status, cg::Status::Optimal);
        assert_eq!(b.master.status, cg::Status::Optimal);
        assert!((a.master.bound - b.master.bound).abs() < 1e-8);
        assert_eq!(b.master.bound, 2.);
    }
}

#[test]
fn half_arc_parity_rejects_fractional_paths_and_odd_cycles() {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 3, 0.).unwrap();
    graph.add_arc(0, 2, 0.).unwrap();
    graph.add_arc(2, 3, 0.).unwrap();
    assert!(integral_path_parity(&graph, &[(0, 0.5), (1, 0.5)]));
    assert!(integral_path_parity(&graph, &[(0, -0.5), (1, 0.5)]));
    assert!(!integral_path_parity(&graph, &[(0, 0.5)]));
    let cycle = graph.add_arc(1, 1, 0.).unwrap();
    assert!(!integral_path_parity(
        &graph,
        &[(0, 0.5), (1, 0.5), (cycle, 0.5)]
    ));
}

#[test]
fn half_arc_validation_rejects_invalid_topology_without_panicking() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph.source = graph.vertices.len();
    assert!(!integral_path_parity(&graph, &[(0, 0.5), (1, 0.5)]));
}

#[test]
fn near_tied_fractionality_keeps_registration_order_like_the_reference() {
    let named = |name: &str, priority: f64| MappedBranchVariable {
        name: name.into(),
        priority,
        networks: vec![],
        masters: vec![],
    };
    let (z, a, b) = (
        named("z-first", 0.),
        named("a-second", 0.),
        named("m-third", 0.),
    );
    // z and a lie within the reference band (1e-10 relative, 1e-6 absolute) of
    // each other: registration order wins, not the name or the tiny margin.
    let ranked = rank_candidates([(&z, 0.4999995), (&a, 0.5)].into_iter());
    assert_eq!(ranked[0].0.name, "z-first");
    // Outside the band the more fractional value still wins.
    let ranked = rank_candidates([(&z, 0.49), (&a, 0.5), (&b, 0.3)].into_iter());
    let names: Vec<_> = ranked.iter().map(|(v, _)| v.name.as_str()).collect();
    assert_eq!(names, ["a-second", "z-first", "m-third"]);
    // Priority is compared before fractionality.
    let high = named("high", 1.);
    let ranked = rank_candidates([(&a, 0.5), (&high, 0.9)].into_iter());
    assert_eq!(ranked[0].0.name, "high");
}
