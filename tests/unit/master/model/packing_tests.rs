use super::*;

fn problem() -> Problem<2> {
    format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/multi_network_native_tree.json"
    )))
    .unwrap()
    .problem()
    .unwrap()
}

fn registry() -> PackingBranching {
    PackingBranching::new(
        1000.,
        vec![
            NetworkPackingAssignment::new(
                7,
                vec![(1, 0), (1, 1), (2, 1), (2, 2), (3, 0), (4, 1), (5, 2)],
            ),
            NetworkPackingAssignment::new(42, vec![(1, 0), (1, 2), (2, 0), (3, 1), (4, 2)]),
        ],
    )
}

fn prepare(registry: &mut PackingBranching, problem: &Problem<2>) {
    registry
        .prepare(problem.networks.iter().map(|n| (n.id, &n.graph)), &[])
        .unwrap();
}

fn root() -> Output<2> {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    solve(problem(), options).unwrap()
}

#[test]
fn last_packing_setter_produces_native_half_incident_expression() {
    let mut packing = registry();
    prepare(&mut packing, &problem());
    let mapping = &packing
        .prepared
        .iter()
        .find(|(_, _, v)| v.name == "ESA_7_1")
        .unwrap()
        .2;
    assert_eq!(
        mapping.networks[0].arcs,
        vec![(0, 0.5), (1, 0.5), (6, 0.5), (7, 0.5)]
    );
    assert_eq!(mapping.value(&root()), Some(0.5));
}

#[test]
fn adjacent_vertices_in_same_group_contribute_full_arc_coefficient() {
    let mut problem = problem();
    let mut arc = problem.networks[0].graph.arcs[0].clone();
    arc.id = 99;
    arc.tail = 1;
    arc.head = 2;
    problem.networks[0].graph.arcs.push(arc);
    let mut packing = PackingBranching::new(
        1.,
        vec![NetworkPackingAssignment::new(7, vec![(1, 4), (2, 4)])],
    );
    prepare(&mut packing, &problem);
    assert!(packing.prepared[0].2.networks[0].arcs.contains(&(99, 1.)));
}

#[test]
fn packing_candidates_deduplicate_groups_by_configuration_identity() {
    let mut packing = registry();
    prepare(&mut packing, &problem());
    let output = root();
    let proposals = packing.proposals(&output, 1e-6);
    assert_eq!(proposals.len(), 2);
    assert!(proposals.iter().any(|p| p.name == "ESA_7_1"));
    let group_two = proposals.iter().find(|p| p.name.ends_with("_2")).unwrap();
    let expected = packing
        .prepared
        .iter()
        .filter(|(_, g, _)| *g == 2)
        .min_by_key(|(id, _, _)| *id)
        .unwrap();
    assert_eq!(group_two.name, expected.2.name);
    let cloned = packing.clone();
    assert_eq!(
        packing.tie_key(&group_two.name),
        cloned.tie_key(&group_two.name)
    );
    assert_eq!(
        proposals.iter().map(|p| &p.name).collect::<Vec<_>>(),
        cloned
            .proposals(&output, 1e-6)
            .iter()
            .map(|p| &p.name)
            .collect::<Vec<_>>()
    );
}

#[test]
fn group_deduplication_is_not_a_two_candidate_cap() {
    let mut packing = registry();
    packing.networks[1].vertices.push((1, 3));
    prepare(&mut packing, &problem());
    assert_eq!(packing.proposals(&root(), 1e-6).len(), 3);
}

#[test]
fn packing_admits_only_activity_strictly_inside_unit_interval() {
    let mut packing = registry();
    prepare(&mut packing, &problem());
    let mut output = root();
    for value in &mut output.values {
        *value *= 3.;
    }
    assert!(packing.proposals(&output, 1e-6).is_empty());
    for value in &mut output.values {
        *value = 0.;
    }
    assert!(packing.proposals(&output, 1e-6).is_empty());
}

#[test]
fn registration_is_validated_even_with_zero_tree_budget() {
    for (network, vertex, group) in [
        (99, 1, 0),
        (7, 0, 0),
        (7, 6, 0),
        (7, 99, 0),
        (7, 1, usize::MAX),
    ] {
        let mut options = ModelTreeOptions::default();
        options.node_limit = 0;
        options.packing = Some(PackingBranching::new(
            1.,
            vec![NetworkPackingAssignment::new(
                network,
                vec![(vertex, group)],
            )],
        ));
        assert!(crate::master::tree::solve(problem(), options).is_err());
    }
}

#[test]
fn duplicate_networks_and_explicit_name_collisions_are_rejected() {
    let problem = problem();
    let mut packing = registry();
    packing.networks.push(packing.networks[0].clone());
    assert!(packing
        .prepare(problem.networks.iter().map(|n| (n.id, &n.graph)), &[])
        .is_err());
    let mut packing = registry();
    let explicit = MappedBranchVariable {
        name: "ESA_7_1".into(),
        priority: 1.,
        networks: vec![],
        masters: vec![],
    };
    assert!(packing
        .prepare(
            problem.networks.iter().map(|n| (n.id, &n.graph)),
            &[explicit]
        )
        .is_err());
}

#[test]
fn serialized_registration_has_no_process_identity_or_prepared_state() {
    let mut packing = registry();
    prepare(&mut packing, &problem());
    let json = serde_json::to_string(&packing).unwrap();
    assert!(!json.contains("identity"));
    assert!(!json.contains("prepared"));
    let mut restored: PackingBranching = serde_json::from_str(&json).unwrap();
    prepare(&mut restored, &problem());
    assert_eq!(restored.prepared.len(), packing.prepared.len());
    assert!(serde_json::from_str::<PackingBranching>(
        r#"{"priority":1,"networks":[],"prepared":[]}"#
    )
    .is_err());
}

#[test]
fn unreachable_sink_does_not_invalidate_integer_packing_expression() {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let mut packing =
        PackingBranching::new(1., vec![NetworkPackingAssignment::new(7, vec![(1, 0)])]);
    packing.prepare(std::iter::once((7, &graph)), &[]).unwrap();
    assert_eq!(packing.prepared[0].2.networks[0].arcs, vec![(0, 0.5)]);
}

#[cfg(feature = "multi-time-windows")]
#[test]
fn preparation_includes_every_time_window_arc_copy() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph.set_bounds(1, 0, 0., 10.).unwrap();
    graph
        .set_time_windows(
            1,
            0,
            &[Window::new(0., 2.).unwrap(), Window::new(8., 10.).unwrap()],
        )
        .unwrap();
    let vertices = graph.vertices.len();
    let split = graph.split_time_window_arcs(0).unwrap();
    assert_eq!(graph.vertices.len(), vertices);
    let mut packing =
        PackingBranching::new(1., vec![NetworkPackingAssignment::new(7, vec![(1, 0)])]);
    packing.prepare(std::iter::once((7, &graph)), &[]).unwrap();
    let coefficients = &packing.prepared[0].2.networks[0].arcs;
    assert_eq!(coefficients.len(), 3);
    for arc in &split.arcs {
        assert!(coefficients.contains(&(arc.arc, 0.5)));
    }
    assert!(coefficients.contains(&(1, 0.5)));
}

#[test]
fn half_distance_precedes_configuration_identity_and_interval_is_strict() {
    let mut packing = registry();
    prepare(&mut packing, &problem());
    let mut output = root();
    let original = output.values.clone();
    for (route, value) in output.routes.iter().zip(&mut output.values) {
        if route.network == 42 {
            *value *= 0.6;
        }
    }
    let proposals = packing.proposals(&output, 1e-6);
    assert!(proposals.iter().any(|p| p.name == "ESA_7_2"));
    assert!(!proposals.iter().any(|p| p.name == "ESA_42_2"));
    for scale in [2e-6, 2. * (1. - 1e-6)] {
        output.values = original.iter().map(|value| value * scale).collect();
        assert!(packing.proposals(&output, 1e-6).is_empty());
    }
}
