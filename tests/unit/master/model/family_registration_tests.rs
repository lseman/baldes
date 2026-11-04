use super::*;

fn options() -> ModelTreeOptions {
    let mut options = ModelTreeOptions::default();
    for name in ["a", "b", "solo"] {
        options
            .branching_registry
            .mapped_variables
            .push(MappedBranchVariable {
                name: name.into(),
                priority: 1.,
                networks: vec![],
                masters: vec![],
            });
    }
    for name in ["a", "b"] {
        options.branching_registry.branch_selection.insert(
            name.into(),
            BranchSelection {
                family: "arcs".into(),
                ..Default::default()
            },
        );
    }
    options.branching_registry.packing = Some(PackingBranching::new(1., vec![]));
    options.branching_registry.ryan_foster = Some(RyanFosterBranching::new(1.));
    options
}

#[test]
fn declared_registration_order_canonicalizes_aliases_and_rejects_unknowns() {
    let mut options = options();
    options.branching_registry.family_registration_order = vec![
        BranchFamilyRegistration::MappedFamily("arcs".into()),
        BranchFamilyRegistration::PackingResource,
        BranchFamilyRegistration::PackingAssignment,
        BranchFamilyRegistration::PackingRyanFoster,
    ];
    let order = registration_order(&options).unwrap();
    assert_eq!(order[&BranchFamilyRegistration::PackingResource], 1);
    options
        .branching_registry
        .family_registration_order
        .push(BranchFamilyRegistration::MappedRegistration(1));
    assert!(registration_order(&options).is_err());
    for registration in [
        BranchFamilyRegistration::MappedFamily("missing".into()),
        BranchFamilyRegistration::MappedRegistration(9),
        BranchFamilyRegistration::ComponentGenerator(0),
    ] {
        options.branching_registry.family_registration_order = vec![registration];
        assert!(registration_order(&options).is_err());
    }
    options.branching_registry.packing = None;
    options.branching_registry.family_registration_order =
        vec![BranchFamilyRegistration::PackingAssignment];
    assert!(registration_order(&options).is_err());
    options.branching_registry.ryan_foster = None;
    options.branching_registry.family_registration_order =
        vec![BranchFamilyRegistration::PackingRyanFoster];
    assert!(registration_order(&options).is_err());
}

#[test]
fn family_registration_order_round_trips_json_and_requires_tree_mode() {
    let mut options = format::SolveOptions::from_json(r#"{"tree":true,"branching_registrations":[{"kind":"family-order","config":{"mapped-family":"arcs"}},{"kind":"family-order","config":"packing-resource"},{"kind":"family-order","config":{"component-generator":2}}]}"#).unwrap();
    let round_trip =
        format::SolveOptions::from_json(&serde_json::to_string(&options).unwrap()).unwrap();
    assert_eq!(
        serde_json::to_value(&round_trip).unwrap(),
        serde_json::to_value(&options).unwrap()
    );
    options.tree = false;
    assert!(options.node_options().is_err());
}

#[test]
fn construction_order_groups_mapped_members_and_keeps_cross_family_order() {
    let mut options = ModelTreeOptions::default();
    let registry = &mut options.branching_registry;
    registry
        .register_batch([
            BranchingRegistration::Mapped(MappedBranchVariable {
                name: "a".into(),
                priority: 1.,
                networks: vec![],
                masters: vec![],
            }),
            BranchingRegistration::Cluster(NetworkClusterBranching::new(0)),
            BranchingRegistration::Mapped(MappedBranchVariable {
                name: "b".into(),
                priority: 1.,
                networks: vec![],
                masters: vec![],
            }),
            BranchingRegistration::Selection {
                name: "a".into(),
                selection: BranchSelection {
                    family: "shared".into(),
                    ..Default::default()
                },
            },
            BranchingRegistration::Selection {
                name: "b".into(),
                selection: BranchSelection {
                    family: "shared".into(),
                    ..Default::default()
                },
            },
        ])
        .unwrap();
    let order = registration_order(&options).unwrap();
    assert_eq!(order.len(), 2);
    assert_eq!(
        order[&BranchFamilyRegistration::MappedFamily("shared".into())],
        0
    );
    assert_eq!(order[&BranchFamilyRegistration::Cluster(0)], 1);
}

#[test]
fn explicit_order_overrides_construction_and_failed_batch_is_atomic() {
    let mut options = ModelTreeOptions::default();
    options
        .branching_registry
        .register(BranchingRegistration::Cluster(
            NetworkClusterBranching::new(0),
        ))
        .unwrap();
    let before = options.branching_registry.construction_order().to_vec();
    let failed = options.branching_registry.register_batch([
        BranchingRegistration::Policy(NetworkBranchingPolicy::Phased),
        BranchingRegistration::Packing(PackingBranching::new(1., vec![])),
        BranchingRegistration::Packing(PackingBranching::new(1., vec![])),
    ]);
    assert!(failed.is_err());
    assert_eq!(options.branching_registry.construction_order(), before);
    assert_eq!(
        options.branching_registry.policy(),
        NetworkBranchingPolicy::MultiplicityFirst
    );
    assert!(options.branching_registry.packing().is_none());
    options
        .branching_registry
        .register(BranchingRegistration::FamilyOrder(
            BranchFamilyRegistration::PackingResource,
        ))
        .unwrap();
    let order = registration_order(&options).unwrap();
    assert_eq!(order.len(), 1);
    assert_eq!(order[&BranchFamilyRegistration::PackingResource], 0);
    assert!(!order.contains_key(&BranchFamilyRegistration::Cluster(0)));
}

#[test]
fn selection_metadata_resolves_explicit_path_counts_and_packing_assignments() {
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let mut packing =
        PackingBranching::new(2., vec![NetworkPackingAssignment::new(42, vec![(1, 0)])]);
    packing
        .prepare(problem.networks.iter().map(|n| (n.id, &n.graph)), &[])
        .unwrap();
    let mut options = ModelTreeOptions::default();
    options
        .branching_registry
        .register_batch([
            BranchingRegistration::PathsPerNetwork(PathsPerNetworkBranching {
                network: 42,
                priority: 2.,
            }),
            BranchingRegistration::Packing(packing),
            BranchingRegistration::Selection {
                name: "paths:42".into(),
                selection: BranchSelection {
                    family: "weighted".into(),
                    rule: BranchSelectionRule::FractionWeightedPriority,
                    variable_priority: 3.,
                    ..Default::default()
                },
            },
            BranchingRegistration::Selection {
                name: "ESA_42_0".into(),
                selection: BranchSelection {
                    family: "weighted".into(),
                    rule: BranchSelectionRule::FractionWeightedPriority,
                    variable_priority: 7.,
                    ..Default::default()
                },
            },
        ])
        .unwrap();
    validate_selection(&options).unwrap();
    let order = selection_order(&options);
    assert!(order["paths:42"] < order["ESA_42_0"]);
}

#[test]
fn selection_ties_follow_cross_family_construction_order() {
    let mut options = ModelTreeOptions::default();
    options
        .branching_registry
        .register_batch([
            BranchingRegistration::PathsPerNetwork(PathsPerNetworkBranching {
                network: 42,
                priority: 1.,
            }),
            BranchingRegistration::Mapped(MappedBranchVariable {
                name: "mapped".into(),
                priority: 1.,
                networks: vec![],
                masters: vec![],
            }),
            BranchingRegistration::Selection {
                name: "paths:42".into(),
                selection: BranchSelection::default(),
            },
            BranchingRegistration::Selection {
                name: "mapped".into(),
                selection: BranchSelection::default(),
            },
        ])
        .unwrap();
    let order = selection_order(&options);
    assert!(order["paths:42"] < order["mapped"]);
}

#[test]
fn compilation_invalidates_and_atomically_publishes_derived_registration_state() {
    let mut configured = ModelTreeOptions::default();
    configured.compile_branching_registry().unwrap();
    assert!(configured.branching_registry.prepared.is_some());
    configured
        .branching_registry
        .register(BranchingRegistration::History(false))
        .unwrap();
    assert!(configured.branching_registry.prepared.is_none());
    configured.compile_branching_registry().unwrap();
    assert!(configured.branching_registry.prepared.is_some());
    configured
        .branching_registry
        .register(BranchingRegistration::FamilyOrder(
            BranchFamilyRegistration::MappedRegistration(99),
        ))
        .unwrap();
    assert!(configured.compile_branching_registry().is_err());
    assert!(configured.branching_registry.prepared.is_none());
    assert!(registration_order(&configured).is_err());
}
