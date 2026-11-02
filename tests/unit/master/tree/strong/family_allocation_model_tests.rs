use super::*;
use crate::master::model::{
    compile_pricing_variables, AggregateVariableBranching, OriginalPricingVariable,
};
use crate::master::tree::{StrongBranchingOptions, StrongPhase};
use std::time::Duration;

fn scalar_aggregates() -> (Output<2>, ModelTreeOptions) {
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let arc = problem
        .networks
        .iter()
        .find(|n| n.id == 42)
        .unwrap()
        .graph
        .arcs[0]
        .id;
    let definitions: Vec<OriginalPricingVariable> = (9..12)
        .map(|id| {
            serde_json::from_value(serde_json::json!({
                "network":42,"id":id,"cost":0.,"integer":true,"arcs":[[arc,1.]]
            }))
            .unwrap()
        })
        .collect();
    let (problem, registry) = compile_pricing_variables(problem, &definitions).unwrap();
    let mut node = Options::default();
    node.cg.stabilize = false;
    node.cg.artificial_cost = 100.;
    let mut output = solve_with_master_variables(problem, vec![], node).unwrap();
    // Controlled physical master point: one column projects to all three
    // independent scalar aggregate registrations. This is an admission test.
    output.snapshot.master.values.fill(0.);
    let column = output
        .master
        .column_tags
        .iter()
        .position(|&tag| {
            tag >= 0
                && output.snapshot.state.routes[tag as usize].network == 42
                && output.snapshot.state.routes[tag as usize]
                    .route
                    .arcs
                    .contains(&arc)
        })
        .unwrap();
    output.snapshot.master.values[column] = 0.5;
    let mut options = ModelTreeOptions::default();
    options.branching_registry.branching = NetworkBranchingPolicy::Phased;
    options.branching_registry.strong_branching = Some(StrongBranchingOptions {
        phases: vec![StrongPhase {
            candidates: 1,
            subtree_ratio: 0.,
            restricted_master_only: true,
            tree_depth_score: false,
            ..StrongPhase::default()
        }],
        ..StrongBranchingOptions::default()
    });
    let aggregates: Vec<_> = (9..12)
        .map(|id| AggregateVariableBranching {
            id,
            networks: vec![42],
            priority: 10.,
        })
        .collect();
    registry
        .compile_branching(&mut options.branching_registry, &[], &aggregates)
        .unwrap();
    (output, options)
}

fn unknown(priority: f64) -> Proposal {
    Proposal {
        name: "adapter-count".into(),
        history_key: "adapter-count".into(),
        priority,
        value: 0.5,
        restriction: Restriction::Multiplicity { network: 42 },
        history_eligible: false,
        fresh: true,
        origin: None,
        component_admission: None,
    }
}

#[test]
fn quota_one_three_compiled_scalar_aggregates_selects_last_registered_family() {
    for lower_unknown in [false, true] {
        let (output, options) = scalar_aggregates();
        let mut proposals = registered_proposals(&output, &options, None).unwrap();
        assert_eq!(proposals.len(), 3);
        assert!(proposals.iter().all(|p| p.fresh));
        if lower_unknown {
            proposals.push(unknown(0.));
        }
        let selected = choose_proposals(
            &output,
            &options,
            &mut History::default(),
            2.,
            Instant::now() + Duration::from_secs(10),
            proposals,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            selected.children[0]
                .state
                .branch_history_origin
                .as_ref()
                .unwrap()
                .name,
            "AGG_11"
        );
    }
}

#[test]
fn visited_unknown_origin_keeps_adapter_flat_admission() {
    let (output, options) = scalar_aggregates();
    let mut proposals = registered_proposals(&output, &options, None).unwrap();
    proposals.push(unknown(10.));
    // A visited unknown family deliberately selects the established adapter
    // ordering rather than inventing a native registration for that producer.
    order_proposals(&mut proposals, &options);
    let expected = proposals[0].name.clone();
    let selected = choose_proposals(
        &output,
        &options,
        &mut History::default(),
        2.,
        Instant::now() + Duration::from_secs(10),
        proposals,
    )
    .unwrap()
    .unwrap();
    let origin = selected.children[0].state.branch_history_origin.as_ref();
    assert_eq!(
        origin.map(|o| o.name.as_str()),
        (expected != "adapter-count").then_some(expected.as_str())
    );
}

#[test]
fn explicit_family_identity_retains_first_registration_even_if_its_first_member_is_absent() {
    let (_, mut options) = scalar_aggregates();
    let mut selection = crate::master::model::BranchSelection::default();
    selection.family = "originals".into();
    options
        .branching_registry
        .branch_selection
        .insert("AGG_9".into(), selection.clone());
    options
        .branching_registry
        .branch_selection
        .insert("AGG_11".into(), selection);
    let make = |name: &str| Proposal {
        name: name.into(),
        history_key: name.into(),
        priority: 10.,
        value: 0.5,
        restriction: Restriction::Mapped(
            options
                .branching_registry
                .mapped_variables
                .iter()
                .find(|v| v.name == name)
                .unwrap()
                .clone(),
        ),
        history_eligible: true,
        fresh: true,
        origin: None,
        component_admission: None,
    };
    let proposals = vec![make("AGG_10"), make("AGG_11")];
    assert_eq!(
        mapped_family_batches(&proposals, &options, 10.),
        Some(vec![vec![1], vec![0]])
    );
}

#[test]
fn explicit_cross_kind_registration_order_changes_real_admission() {
    use crate::master::model::BranchFamilyRegistration::*;
    for (order, expected) in [
        (vec![PackingAssignment, PackingResource], "zz-resource"),
        (vec![PackingResource, PackingAssignment], "aa-packing"),
        (vec![], "aa-packing"),
        (vec![PackingResource], "aa-packing"),
    ] {
        let (output, mut options) = scalar_aggregates();
        options.branching_registry.family_registration_order = order;
        options.branching_registry.packing = Some(PackingBranching::new(10., vec![]));
        let variable = options.branching_registry.mapped_variables[0].clone();
        let make = |name: &str, origin| Proposal {
            name: name.into(),
            history_key: name.into(),
            priority: 10.,
            value: 0.5,
            restriction: Restriction::Mapped(variable.clone()),
            history_eligible: true,
            fresh: true,
            origin: Some(origin),
            component_admission: None,
        };
        // Deliberately use the same child-expression shape: origin is supplied
        // by the actual producer, never inferred from a Restriction variant.
        let proposals = vec![
            make("zz-resource", ProducerFamily::PackingResource),
            make("aa-packing", ProducerFamily::PackingAssignment),
        ];
        let selected = choose_proposals(
            &output,
            &options,
            &mut History::default(),
            2.,
            Instant::now() + Duration::from_secs(10),
            proposals,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            selected.children[0]
                .state
                .branch_history_origin
                .as_ref()
                .unwrap()
                .name,
            expected
        );
    }
}

#[test]
fn automatically_recorded_order_keeps_fallback_for_implicit_resource_family() {
    let (_, mut options) = scalar_aggregates();
    options
        .branching_registry
        .register(BranchingRegistration::Packing(PackingBranching::new(
            10.,
            vec![],
        )))
        .unwrap();
    let variable = options.branching_registry.mapped_variables[0].clone();
    let proposal = |name: &str, origin| Proposal {
        name: name.into(),
        history_key: name.into(),
        priority: 10.,
        value: 0.5,
        restriction: Restriction::Mapped(variable.clone()),
        history_eligible: true,
        fresh: true,
        origin: Some(origin),
        component_admission: None,
    };
    let proposals = vec![
        proposal("resource", ProducerFamily::PackingResource),
        proposal("packing", ProducerFamily::PackingAssignment),
    ];
    let declaration = super::family_registration::registration_order(&options).unwrap();
    assert!(declaration.contains_key(&BranchFamilyRegistration::PackingAssignment));
    assert!(!declaration.contains_key(&BranchFamilyRegistration::PackingResource));
    assert!(registered_family_batches(&proposals, &options, &declaration, 10.).is_none());
}

#[test]
fn component_registration_identity_survives_legacy_global_publication_cap() {
    use crate::master::model::{
        BranchFamilyRegistration, ComponentClassGenerator, ComponentCoordinate,
        ComponentGeneratorCoordinate, ComponentPriorityRule, ComponentVariableKind,
    };
    for (reverse, unknown_priority, expected) in [
        (false, None, "g2"),
        (true, None, "g0"),
        (false, Some(0.), "g2"),
        (false, Some(10.), "g0"),
    ] {
        let (output, mut options) = scalar_aggregates();
        let policy = options.branching_registry.branching;
        let strong = options.branching_registry.strong_branching.clone();
        options.branching_registry = BranchingRegistry::default();
        options.branching_registry.branching = policy;
        options.branching_registry.strong_branching = strong;
        let mut values = vec![
            None;
            output
                .snapshot
                .state
                .oracles
                .iter()
                .find(|o| o.network.id == 42)
                .unwrap()
                .network
                .graph
                .arcs
                .len()
        ];
        values[0] = Some(1);
        options.branching_registry.component_generators = (0..3)
            .map(|i| ComponentClassGenerator {
                name: format!("g{i}"),
                network: 42,
                priority: 10.,
                coordinates: vec![ComponentGeneratorCoordinate {
                    coordinate: ComponentCoordinate {
                        id: 11,
                        arc_values: values.clone(),
                    },
                    kind: ComponentVariableKind::Binary,
                    priority: 1.,
                    current_upper: 1.,
                    upper: 1.,
                }],
                ordering_coordinates: vec![],
                priority_rule: ComponentPriorityRule::AddedDepth,
                minimum_fractionality: 0.02,
                integrality_tolerance: 1e-8,
            })
            .collect();
        if reverse {
            options.branching_registry.family_registration_order = (0..3)
                .rev()
                .map(BranchFamilyRegistration::ComponentGenerator)
                .collect();
        }
        let mut proposals = registered_proposals(&output, &options, None).unwrap();
        assert_eq!(
            proposals.len(),
            3,
            "all families must survive the old shared cap"
        );
        assert_eq!(
            proposals
                .iter()
                .filter(|p| p
                    .component_admission
                    .as_ref()
                    .unwrap()
                    .legacy_order
                    .is_some())
                .count(),
            1
        );
        if let Some(priority) = unknown_priority {
            let mut proposal = unknown(priority);
            proposal.name = "zz-adapter".into();
            proposals.push(proposal);
        }
        let selected = choose_proposals(
            &output,
            &options,
            &mut History::default(),
            2.,
            Instant::now() + Duration::from_secs(10),
            proposals,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            selected.children[0]
                .state
                .component_branches
                .last()
                .unwrap()
                .name,
            expected
        );
    }
}
