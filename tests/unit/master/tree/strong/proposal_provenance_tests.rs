use super::*;
use crate::master::tree::{StrongBranchingOptions, StrongPhase};
use std::time::Duration;

fn rf_root() -> Output<2> {
    let mut graph = Graph::<2>::new(5, 0).unwrap();
    for (tail, head, cost) in [
        (0, 1, 0.),
        (1, 2, 0.),
        (2, 4, 0.),
        (1, 3, 0.),
        (3, 4, 0.),
        (0, 4, 1.),
    ] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let problem = Problem {
        rows: vec![
            Row {
                sense: RowSense::Equal,
                rhs: 1.
            };
            2
        ],
        networks: vec![Network {
            id: 0,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![
                vec![],
                vec![(0, 4.)],
                vec![],
                vec![(1, 2.)],
                vec![],
                vec![(0, 1.), (1, 1.)],
            ],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options.preprocessing.enabled = false;
    options.gap_enumeration = false;
    solve(problem, options).unwrap()
}

fn screen_options() -> ModelTreeOptions {
    let mut options = ModelTreeOptions::default();
    options.branching_registry.branching = NetworkBranchingPolicy::Phased;
    options.branching_registry.strong_branching = Some(StrongBranchingOptions {
        phases: vec![StrongPhase {
            candidates: 1,
            subtree_ratio: 0.,
            restricted_master_only: true,
            ..Default::default()
        }],
        ..Default::default()
    });
    options
}

fn mapped(name: &str) -> MappedBranchVariable {
    MappedBranchVariable {
        name: name.into(),
        priority: 100.,
        networks: vec![MappedNetworkVariable {
            network: 0,
            per_route: 4.,
            arcs: vec![],
            activities: vec![],
        }],
        masters: vec![],
    }
}

#[test]
fn real_ryan_foster_family_ignores_colliding_mapped_selection_metadata() {
    let output = rf_root();
    assert_eq!(output.master.lp, 0.);
    for metadata in [false, true] {
        let mut options = screen_options();
        let mut rf = RyanFosterBranching::new(100.);
        rf.prepare(
            output
                .snapshot
                .state
                .oracles
                .iter()
                .map(|o| (o.network.id, o.network.graph.as_ref())),
            None,
        )
        .unwrap();
        options.branching_registry.ryan_foster = Some(rf);
        options.branching_registry.mapped_variables =
            ["PS_0_1", "PS_0_2"].into_iter().map(mapped).collect();
        super::super::model_mapping::validate(
            &options.branching_registry.mapped_variables,
            output
                .snapshot
                .state
                .oracles
                .iter()
                .map(|o| (o.network.id, o.network.graph.as_ref())),
            &[],
        )
        .unwrap();
        for variable in &options.branching_registry.mapped_variables {
            assert_eq!(variable.value(&output), Some(3.));
        }
        if metadata {
            for (name, priority) in [("PS_0_1", 10.), ("PS_0_2", 1.)] {
                options.branching_registry.branch_selection.insert(
                    name.into(),
                    BranchSelection {
                        family: "unrelated-mapped".into(),
                        rule: BranchSelectionRule::HighestPriority,
                        variable_priority: priority,
                        ..Default::default()
                    },
                );
            }
            super::super::model_mapping::validate_selection(
                &options.branching_registry.mapped_variables,
                &options.branching_registry.branch_selection,
            )
            .unwrap();
        }
        let proposals = registered_proposals(&output, &options, None).unwrap();
        assert_eq!(proposals.len(), 2);
        assert!(proposals
            .iter()
            .all(|p| p.origin == Some(ProducerFamily::PackingRyanFoster)));
        assert_eq!(
            proposals.iter().find(|p| p.name == "PS_0_1").unwrap().value,
            0.25
        );
        assert_eq!(
            proposals.iter().find(|p| p.name == "PS_0_2").unwrap().value,
            0.5
        );
        let selected = choose_proposals(
            &output,
            &options,
            &mut History::default(),
            1.,
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
            "PS_0_2"
        );
    }
    // The direct source-sink column is a feasible integer solution of cost1.
    let direct = output.snapshot.state.oracles[0].network.graph.arcs[5].id;
    assert!(output.snapshot.state.oracles[0].network.arc_entries[5]
        .iter()
        .all(|&(_, a)| a == 1.));
    assert_eq!(direct, 5);
}

fn aggregate_component_collision() -> (Output<2>, ModelTreeOptions, Vec<Proposal>) {
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let graph = &problem.networks.iter().find(|n| n.id == 42).unwrap().graph;
    let arc = graph.arcs[0].id;
    let mut arc_values = vec![None; graph.arcs.len()];
    arc_values[0] = Some(1);
    let definitions: Vec<OriginalPricingVariable> = vec![serde_json::from_value(
        serde_json::json!({"network":42,"id":9,"cost":0.,"integer":true,"arcs":[[arc,1.]]}),
    )
    .unwrap()];
    let (problem, registry) = compile_pricing_variables(problem, &definitions).unwrap();
    let mut node = Options::default();
    node.cg.stabilize = false;
    node.cg.artificial_cost = 100.;
    let output = solve(problem, node).unwrap();
    let mut options = screen_options();
    registry
        .compile_branching(
            &mut options.branching_registry,
            &[],
            &[AggregateVariableBranching {
                id: 9,
                networks: vec![42],
                priority: 100.,
            }],
        )
        .unwrap();
    options.branching_registry.component_generators = vec![ComponentClassGenerator {
        name: "AGG_9".into(),
        network: 42,
        priority: 100.,
        coordinates: vec![ComponentGeneratorCoordinate {
            coordinate: ComponentCoordinate { id: 11, arc_values },
            kind: ComponentVariableKind::Binary,
            priority: 1.,
            current_upper: 1.,
            upper: 1.,
        }],
        ordering_coordinates: vec![],
        priority_rule: ComponentPriorityRule::AddedDepth,
        minimum_fractionality: 0.02,
        integrality_tolerance: 1e-8,
    }];
    let proposals = registered_proposals(&output, &options, None).unwrap();
    (output, options, proposals)
}

#[test]
fn real_component_display_collision_cannot_inherit_aggregate_placeholder_or_direction() {
    let (output, mut options, proposals) = aggregate_component_collision();
    assert_eq!(proposals.len(), 2);
    let aggregate = proposals
        .iter()
        .find(|p| matches!(p.restriction, Restriction::Aggregate { .. }))
        .unwrap();
    let component = proposals
        .iter()
        .find(|p| p.origin == Some(ProducerFamily::ComponentGenerator(0)))
        .unwrap();
    assert_eq!(aggregate.name, component.name);
    assert_ne!(aggregate.history_key, component.history_key);
    assert_eq!(selection_value(aggregate, &options), 1.);
    assert_eq!(selection_value(component, &options), component.value);
    assert_eq!(component.value, 0.5);
    assert_eq!(
        family_direction(aggregate, &options),
        Some(NetworkBranchDirection::UpperFirst)
    );
    assert_eq!(family_direction(component, &options), None);
    options
        .branching_registry
        .branch_directions
        .insert("AGG_9".into(), NetworkBranchDirection::LowerFirst);
    let component = proposals
        .into_iter()
        .find(|p| p.origin == Some(ProducerFamily::ComponentGenerator(0)))
        .unwrap();
    let selected = choose_proposals(
        &output,
        &options,
        &mut History::default(),
        2.,
        Instant::now() + Duration::from_secs(10),
        vec![component],
    )
    .unwrap()
    .unwrap();
    assert_eq!(selected.direction, Some(NetworkBranchDirection::LowerFirst));
}

#[test]
fn component_name_matching_a_prepared_packing_assignment_keeps_ordinary_name_ties() {
    let (output, mut options, _) = aggregate_component_collision();
    options.branching_registry.component_generators[0].name = "ESA_42_0".into();
    options.branching_registry.packing = Some(PackingBranching::new(
        100.,
        vec![NetworkPackingAssignment::new(42, vec![(1, 0)])],
    ));
    options
        .branching_registry
        .packing
        .as_mut()
        .unwrap()
        .prepare(
            output
                .snapshot
                .state
                .oracles
                .iter()
                .map(|o| (o.network.id, o.network.graph.as_ref())),
            &[],
        )
        .unwrap();
    let proposals = registered_proposals(&output, &options, None).unwrap();
    let component = proposals
        .into_iter()
        .find(|p| p.origin == Some(ProducerFamily::ComponentGenerator(0)))
        .unwrap();
    let ordinary = Proposal {
        name: "AAA-first".into(),
        history_key: "count".into(),
        priority: 100.,
        value: component.value,
        restriction: Restriction::Multiplicity { network: 42 },
        history_eligible: false,
        fresh: true,
        origin: None,
        component_admission: None,
    };
    let mut proposals = vec![component, ordinary];
    order_proposals(&mut proposals, &options);
    assert_eq!(proposals[0].name, "AAA-first");
    assert_ne!(
        options
            .branching_registry
            .packing
            .as_ref()
            .unwrap()
            .tie_key("ESA_42_0"),
        (1, 0, 0)
    );
}
