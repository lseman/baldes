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
    let mut options = ModelTreeOptions::default();
    options.branching = NetworkBranchingPolicy::Phased;
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
    // intermediate phases skip, and its exact trials do not enter history.
    assert!(history.evaluations.is_empty());
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
        controls.branch_priorities.assignment = 12.;
        controls.mapped_strong_branching = Some(StrongBranchingOptions {
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
    invalid.branch_priorities.arc = f64::NAN;
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
    controls.mapped_variables.push(MappedBranchVariable {
        name: "arc:7:0".into(),
        priority: 1.,
        networks: vec![MappedNetworkVariable {
            network: 42,
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
    options.packing = Some(packing);
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
    assert_eq!(output.master.lp, 1.5);
    options.mapped_strong_branching = None;
    let ordinary = collect(&output, &options, &[]).unwrap();
    assert!(!ordinary.iter().any(|p| p.name.starts_with("ESA_")));
    assert!(ordinary
        .iter()
        .any(|p| matches!(p.restriction, Restriction::Multiplicity { .. })));
}
