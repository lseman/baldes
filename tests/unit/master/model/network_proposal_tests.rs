use super::*;
use crate::master::tree::{StrongBranchingOptions, TreeStatus};
use std::time::Duration;

fn problem() -> Problem<2> {
    format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem()
    .unwrap()
}
fn options() -> ModelTreeOptions {
    let mut options = strong_tree_options();
    options.branching_registry.branching = NetworkBranchingPolicy::Phased;
    options.node.cg.stabilize = false;
    options.node.cg.artificial_cost = 100.;
    options
}

#[test]
fn full_registry_is_lightweight_and_route_count_generators_exclude_history() {
    let output = solve_with_master_variables(problem(), vec![], options().node).unwrap();
    let proposals = collect(&output, &options(), &[]).unwrap();
    assert!(proposals
        .iter()
        .any(|p| matches!(p.restriction, Restriction::Assignment { .. })));
    assert!(proposals
        .iter()
        .any(|p| matches!(p.restriction, Restriction::Activity { .. })));
    let count = proposals
        .iter()
        .find(|p| matches!(p.restriction, Restriction::Multiplicity { .. }))
        .unwrap();
    assert_eq!(count.name, "multiplicity:42");
    assert!(!count.history_eligible);
    assert!(proposals
        .iter()
        .filter(|p| !matches!(p.restriction, Restriction::Multiplicity { .. }))
        .all(|p| p.history_eligible));
    // Proposals contain only restriction metadata, so collecting a registry
    // cannot add any master columns/rows or consume pricing calls.
    assert_eq!(output.master.lp, 1.5);
    assert_eq!(output.snapshot.state.shared_rows.len(), 3);
}

#[test]
fn separate_variable_priority_uses_family_fractionality_with_and_without_trials() {
    let output = solve_with_master_variables(problem(), vec![], options().node).unwrap();
    let variables: Vec<_> = ["less-fractional", "more-fractional"]
        .into_iter()
        .map(|name| MappedBranchVariable {
            name: name.into(),
            priority: 20.,
            networks: vec![super::super::MappedNetworkVariable {
                network: 42,
                arcs: vec![],
                activities: vec![],
                per_route: 1.,
            }],
            masters: vec![],
        })
        .collect();
    for trials in [false, true] {
        let mut controls = options();
        controls.branching_registry.mapped_variables = variables.clone();
        for (variable, priority) in variables.iter().zip([100., 1.]) {
            controls.branching_registry.branch_selection.insert(
                variable.name.clone(),
                super::super::BranchSelection {
                    family: "original-variables".into(),
                    variable_priority: priority,
                    default_factor: priority,
                    ..Default::default()
                },
            );
        }
        if trials {
            let mut strong = StrongBranchingOptions::default();
            strong.phases[0].candidates = 1;
            strong.phases.truncate(1);
            controls.branching_registry.strong_branching = Some(strong);
        } else {
            controls.branching_registry.strong_branching = None;
        }
        let proposals = variables
            .iter()
            .zip([1.1, 2.5])
            .map(|(variable, value)| Proposal {
                name: variable.name.clone(),
                history_key: variable.name.clone(),
                priority: variable.priority,
                value,
                restriction: Restriction::Mapped(variable.clone()),
                history_eligible: true,
                fresh: true,
                origin: None,
                component_admission: None,
            })
            .collect();
        let mut history = model_strong::History::default();
        let selected = model_strong::choose_proposals(
            &output,
            &controls,
            &mut history,
            f64::INFINITY,
            Instant::now() + Duration::from_secs(10),
            proposals,
        )
        .unwrap()
        .unwrap();
        if trials {
            assert!(history
                .evaluations
                .iter()
                .all(|e| e.name == "more-fractional"));
        }
        assert_eq!(*selected.children[0].master.rhs.last().unwrap(), 2.);
    }
}

#[test]
fn explicit_path_registration_preserves_priority_and_replaces_inferred_count() {
    let output = solve_with_master_variables(problem(), vec![], options().node).unwrap();
    for policy in [
        NetworkBranchingPolicy::MultiplicityFirst,
        NetworkBranchingPolicy::ResourceFirst,
        NetworkBranchingPolicy::Phased,
    ] {
        let mut controls = options();
        controls.branching_registry.branching = policy;
        controls.branching_registry.paths_per_network.push(
            super::super::PathsPerNetworkBranching {
                network: 42,
                priority: -2.5,
            },
        );
        let proposals = collect(&output, &controls, &[]).unwrap();
        let explicit = proposals.iter().find(|p| p.name == "paths:42").unwrap();
        assert_eq!(explicit.priority, -2.5);
        assert!(explicit.history_eligible);
        assert!(!proposals.iter().any(|p| p.name == "multiplicity:42"));
        let solved = crate::master::tree::solve(problem(), controls).unwrap();
        assert_eq!(solved.status, TreeStatus::Optimal);
        assert_eq!(solved.incumbent, 2.);
    }
}

#[test]
fn native_priority_batches_choose_count_and_stage_three_without_history_leak() {
    let output = solve_with_master_variables(problem(), vec![], options().node).unwrap();
    let mut history = model_strong::History::default();
    let selected = model_strong::choose_proposals(
        &output,
        &options(),
        &mut history,
        2.,
        Instant::now() + Duration::from_secs(10),
        collect(&output, &options(), &[]).unwrap(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(history.trials, 2);
    // Count priority11 is the only nonempty admitted family; its two
    // intermediate phases skip. Its exact trials are traced but never
    // become pseudocost history, so nothing is ranked from them.
    assert!(!history.evaluations.is_empty());
    assert!(history
        .evaluations
        .iter()
        .all(|e| e.name.starts_with("multiplicity:")));
    assert_eq!(history.ranked, 0);
    let lower = selected.evaluated[0].as_ref().expect("exact lower child");
    let upper = selected.evaluated[1].as_ref().expect("exact upper child");
    assert!((lower.master.lp - 2.).abs() < 1e-8);
    assert!((upper.master.lp - 5. / 3.).abs() < 1e-8);
    assert!(selected
        .children
        .iter()
        .all(|c| c.warm.bound >= output.snapshot.warm.bound));
}

#[test]
fn explicit_family_priorities_and_completed_subtree_history_preserve_optimum() {
    for averaging in [false, true] {
        let mut controls = options();
        controls.branching_registry.branch_priorities.assignment = 12.;
        controls.branching_registry.strong_branching = Some(StrongBranchingOptions {
            use_history_subtree_size: averaging,
            ..Default::default()
        });
        let output = crate::master::tree::solve(problem(), controls).unwrap();
        assert_eq!(output.status, TreeStatus::Optimal);
        assert_eq!(output.incumbent, 2.);
        assert!(output.strong_branching_trials > 0);
        assert!(output
            .mapped_branch_evaluations
            .iter()
            .any(|e| e.name.starts_with("assignment:")));
    }
    let mut invalid = options();
    invalid.branching_registry.branch_priorities.arc = f64::NAN;
    invalid.node_limit = 0;
    assert!(crate::master::tree::solve(problem(), invalid).is_err());
    assert_eq!(
        NetworkBranchingPolicy::parse("phased").unwrap(),
        NetworkBranchingPolicy::Phased
    );
}

#[test]
fn retained_resource_generators_keep_their_threshold_and_refresh_current_mass() {
    let mut output = solve_with_master_variables(problem(), vec![], options().node).unwrap();
    for route in &mut output.routes {
        let level = if route.network == 7 { 1. } else { 3. };
        for point in route.route.resource_profile.iter_mut().skip(1) {
            point[0] = level;
        }
    }
    let template = ResourceGroupCandidate {
        packing_row: 0,
        resource: 0,
        threshold: 2.,
        value: 0.25,
        members: vec![
            ResourceGroupMember {
                network: 7,
                arcs: vec![0],
            },
            ResourceGroupMember {
                network: 42,
                arcs: vec![0],
            },
        ],
    };
    let proposal = refresh_resource(&output, &template, 10., 1e-6).unwrap();
    assert!(!proposal.fresh);
    assert_eq!(
        proposal.origin,
        Some(crate::master::model::model_strong::ProducerFamily::PackingResource)
    );
    assert_eq!(
        resource_proposal(template.clone(), 10.).origin,
        proposal.origin
    );
    assert!((proposal.value - 0.5).abs() < 1e-8);
    let Restriction::Resource(refreshed) = proposal.restriction else {
        panic!("resource generator")
    };
    assert_eq!(refreshed.threshold, template.threshold);
    let mut different = template.clone();
    different.threshold = 2.5;
    assert_ne!(resource_key(&template), resource_key(&different));
}

#[test]
fn user_display_names_cannot_collide_with_generated_history_identities() {
    let output = solve_with_master_variables(problem(), vec![], options().node).unwrap();
    let mut controls = options();
    controls
        .branching_registry
        .mapped_variables
        .push(MappedBranchVariable {
            name: "arc:7:0".into(),
            priority: 1.,
            networks: vec![MappedNetworkVariable {
                network: 42,
                activities: vec![],
                arcs: vec![],
                per_route: 1.,
            }],
            masters: vec![],
        });
    let proposals = collect(&output, &controls, &[]).unwrap();
    let same_name: Vec<_> = proposals.iter().filter(|p| p.name == "arc:7:0").collect();
    assert_eq!(same_name.len(), 2);
    assert_ne!(same_name[0].history_key, same_name[1].history_key);
}

#[test]
fn explicit_packing_registration_replaces_inferred_shared_row_assignments() {
    let output = solve_with_master_variables(problem(), vec![], options().node).unwrap();
    let mut options = options();
    let mut packing = PackingBranching::new(
        1000.,
        vec![
            NetworkPackingAssignment::new(7, vec![(1, 1), (2, 2), (3, 0), (4, 1), (5, 2)]),
            NetworkPackingAssignment::new(42, vec![(1, 2), (2, 0), (3, 1), (4, 2)]),
        ],
    );
    packing
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
    options.branching_registry.packing = Some(packing);
    let proposals = collect(&output, &options, &[]).unwrap();
    assert!(!proposals
        .iter()
        .any(|p| matches!(p.restriction, Restriction::Assignment { .. })));
    assert_eq!(
        proposals
            .iter()
            .filter(|p| p.name.starts_with("ESA_"))
            .count(),
        2
    );
    let assignments: Vec<_> = proposals
        .iter()
        .filter(|p| p.name.starts_with("ESA_"))
        .collect();
    for proposal in assignments {
        assert_eq!(
            proposal.origin,
            Some(crate::master::model::model_strong::ProducerFamily::PackingAssignment)
        );
        let refreshed = options
            .branching_registry
            .packing
            .as_ref()
            .unwrap()
            .refresh(&output, &proposal.name, options.tolerance)
            .unwrap();
        assert_eq!(refreshed.origin, proposal.origin);
        assert!(!refreshed.fresh);
    }
    assert_eq!(output.master.lp, 1.5);
    options.branching_registry.strong_branching = None;
    let ordinary = collect(&output, &options, &[]).unwrap();
    assert!(!ordinary.iter().any(|p| p.name.starts_with("ESA_")));
    assert!(ordinary
        .iter()
        .any(|p| matches!(p.restriction, Restriction::Multiplicity { .. })));
}

#[test]
fn multiplicity_first_runs_exact_strong_trials_on_the_route_count_split() {
    let mut instance = problem();
    instance.networks.truncate(1);
    // Three pair routes cover a different pair of rows; their LP uses each
    // with quantity one half, so the total route count is 1.5.
    instance.networks[0].arc_entries[4] = vec![(0, 1.), (2, 1.)];
    for strong in [false, true] {
        let mut configured = strong_tree_options();
        configured.node.cg.stabilize = false;
        configured.node.cg.artificial_cost = 100.;
        configured.node_limit = 20;
        configured.branching_registry.strong_branching =
            strong.then(StrongBranchingOptions::default);
        let output = crate::master::tree::solve(instance.clone(), configured).unwrap();
        assert_eq!(output.status, TreeStatus::Optimal, "{output:?}");
        assert!((output.incumbent - 2.).abs() < 1e-8);
        if strong {
            assert!(output.strong_branching_trials >= 2, "{output:?}");
            // Route-count trials are traced, but they are not pseudocost
            // history generators, so nothing is ranked from them.
            assert!(
                output.mapped_branch_evaluations.iter().any(|e| {
                    e.name == "multiplicity:7"
                        && e.kind == model_strong::MappedEvaluationKind::ExactPricing
                }),
                "{output:?}"
            );
            assert_eq!(output.mapped_history_ranked, 0);
        } else {
            assert_eq!(output.strong_branching_trials, 0);
        }
    }
}

/// Strong trials are opt-in (native `ApplyStrongBranchingEvaluation = false`);
/// these tests exercise them with the default phase schedule.
fn strong_tree_options() -> ModelTreeOptions {
    ModelTreeOptions {
        branching_registry: {
            let mut __branch_registry = crate::master::model::BranchingRegistry::default();
            __branch_registry.strong_branching =
                Some(crate::master::tree::StrongBranchingOptions::default());
            __branch_registry
        },
        ..ModelTreeOptions::default()
    }
}

#[test]
fn component_registrations_bound_both_priority_presets() {
    use crate::master::model::{
        ComponentClassBranching, ComponentClassGenerator, ComponentCoordinate,
        ComponentGeneratorCoordinate, ComponentPriorityRule, ComponentSequence,
        ComponentVariableKind,
    };
    let output = solve_with_master_variables(problem(), vec![], options().node).unwrap();
    let oracle = output
        .snapshot
        .state
        .oracles
        .iter()
        .find(|oracle| oracle.network.id == 42)
        .unwrap();
    let coordinate = ComponentCoordinate {
        id: 0,
        arc_values: oracle
            .network
            .graph
            .arcs
            .iter()
            .map(|arc| (arc.tail == oracle.network.graph.source).then_some(1))
            .collect(),
    };
    for (class_priority, generator_priority) in [(10., -10.), (-10., 10.)] {
        for policy in [
            NetworkBranchingPolicy::MultiplicityFirst,
            NetworkBranchingPolicy::ResourceFirst,
        ] {
            let mut controls = options();
            controls.branching_registry.branching = policy;
            controls.branching_registry.component_classes = vec![ComponentClassBranching {
                name: "explicit-class".into(),
                priority: class_priority,
                minimum_fractionality: 0.02,
                network: 42,
                coordinates: vec![coordinate.clone()],
                sequence: ComponentSequence::default(),
            }];
            controls.branching_registry.component_generators = vec![ComponentClassGenerator {
                name: "generated-class".into(),
                network: 42,
                priority: generator_priority,
                coordinates: vec![ComponentGeneratorCoordinate {
                    coordinate: coordinate.clone(),
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
            let proposals = collect(&output, &controls, &[]).unwrap();
            assert!(proposals.iter().any(|p| {
                matches!(p.restriction, Restriction::Component(_)) && p.priority == class_priority
            }));
            assert!(proposals.iter().any(|p| {
                matches!(p.restriction, Restriction::GeneratedComponent { .. })
                    && p.priority == generator_priority
            }));
            for proposal in &proposals {
                if let Some(family) = inferred(proposal) {
                    if policy == NetworkBranchingPolicy::MultiplicityFirst
                        && family == Inferred::Multiplicity
                    {
                        assert!(proposal.priority > 10.);
                    } else {
                        assert!(proposal.priority < -10.);
                    }
                }
            }
        }
    }
}

#[test]
fn route_count_candidates_use_class_violation_admission_and_keep_other_splits() {
    let mut output = solve_with_master_variables(problem(), vec![], options().node).unwrap();
    let selected = output
        .routes
        .iter()
        .position(|route| route.network == 42)
        .unwrap();
    for (count, accepted) in [
        (1.01, false),
        (1.0200009, false),
        (1.020002, true),
        (1.98, false),
        (1.979998, true),
    ] {
        output.values.fill(0.);
        output.values[selected] = count;
        let proposals = collect(&output, &options(), &[]).unwrap();
        assert_eq!(
            proposals
                .iter()
                .any(|p| matches!(p.restriction, Restriction::Multiplicity { network: 42 })),
            accepted,
            "route count {count}"
        );
        // Declining a near-integer count must not suppress the physical-arc
        // candidates that can still split the same fractional point.
        assert!(proposals
            .iter()
            .any(|p| { matches!(p.restriction, Restriction::Activity { network: 42, .. }) }));
    }
    output.values.fill(0.);
    output.values[selected] = 1.025;
    let mut controls = options();
    controls.tolerance = 0.03;
    assert!(!collect(&output, &controls, &[])
        .unwrap()
        .iter()
        .any(|p| matches!(p.restriction, Restriction::Multiplicity { network: 42 })));
}
