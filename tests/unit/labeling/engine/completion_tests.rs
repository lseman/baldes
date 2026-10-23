use super::*;
use crate::graph::coupled::CoupledResource;

fn model() -> Graph<2> {
    let mut graph = Graph::new(3, 0).unwrap();
    graph.add_arc(0, 1, -10.).unwrap();
    graph.add_arc(1, 2, -10.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![1., 1.],
            weights: vec![1., -2.],
        }])
        .unwrap();
    graph
}
fn engine(g: &Graph<2>) -> LabelingEngine<2> {
    LabelingEngine::prepare(
        g,
        EngineControls {
            bidirectional: true,
            bidirectional_mode: 1,
            direction_flags: 3,
            use_symmetric_execution: false,
            ..Default::default()
        },
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        },
    )
    .unwrap()
}
#[test]
fn completion_rebuilds_after_changes_and_does_not_cross_preparations_or_forks() {
    let graph = model();
    let mut engine = engine(&graph);
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    assert!(engine
        .coupled_completion
        .iter()
        .all(|c| c.identity.is_some()));
    let v = engine.grid.internal[1];
    let id = engine.forward.buckets[v][0].active[0];
    let mut label = engine.forward.labels.view(id).to_owned();
    // A sparse model deliberately has no eager pair table. Install a
    // conservative test hull and a poisoned gate to exercise membership.
    let mut hull = crate::graph::coupled::CoupledStateBounds::default();
    hull.include(label.coupled_state.as_deref());
    engine.coupled_completion[1].queries[v] = vec![hull];
    engine.coupled_completion[1].pairs[v] = vec![Some(vec![
        crate::graph::coupled::CoupledPairSummary::default(),
    ])];
    // A valid standalone resource prefix outside this bucket's retained hull
    // must not reuse even a cached rejection for the bucket pair.
    label.coupled_state = Some(
        graph
            .coupled_kernel()
            .unwrap()
            .replay_state(&[0; 12])
            .unwrap()
            .unwrap()
            .0,
    );
    assert!(!engine.coupled_completion[1].queries[v][0].contains(label.coupled_state.as_deref()));
    engine.coupled_completion[1].pairs[v][0].as_mut().unwrap()[0].possible = false;
    let expected = engine.ctx().resource_bucket_key(
        &graph,
        true,
        label.coupled_state.as_deref(),
        &label.presence_state,
        label.key(),
        v,
        0,
    );
    assert_eq!(
        engine.ctx().bucket_completion(
            &graph,
            true,
            (&label).into(),
            label.key(),
            BucketId {
                vertex: v,
                bucket: 0
            },
            engine.backward.buckets[v][0].hot.key.first().copied(),
            f64::INFINITY
        ),
        Some(expected)
    );
    let child = engine.fork_for_node();
    assert!(child
        .coupled_completion
        .iter()
        .all(|c| c.identity.is_none()));
    let mut reinstalled = graph.clone();
    reinstalled
        .set_coupled_resources(graph.coupled_kernel().unwrap().definitions())
        .unwrap();
    assert_eq!(
        engine.ctx().resource_completion_key(
            &reinstalled,
            true,
            label.coupled_state.as_deref(),
            &label.presence_state,
            0.
        ),
        f64::NEG_INFINITY
    );
    engine.load_presence(&graph, vec![]).unwrap();
    assert!(engine
        .coupled_completion
        .iter()
        .all(|c| c.identity.is_none()));
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    assert!(engine.load_cuts(&graph, vec![], vec![]));
    assert!(engine
        .coupled_completion
        .iter()
        .all(|c| c.identity.is_none()));
}
#[test]
fn pair_preparation_budget_keeps_uncached_rows_as_unknown() {
    let graph = model();
    let mut engine = engine(&graph);
    for forward in [true, false] {
        let storage = engine.storage_mut(forward);
        *storage = LabelStorage::new([300]);
        for bucket in 0..300 {
            let label = StoredLabel {
                id: 0,
                local_id: 0,
                bucket: BucketId { vertex: 0, bucket },
                vertex: 0,
                cost: 0.,
                original_cost: 0.,
                resources: [0.; 2].into(),
                ng: 0,
                branch_state: None,
                presence_state: None,
                coupled_state: Some(graph.coupled_kernel().unwrap().root(forward)),
                parent: None,
                arc: None,
                hops: 0,
                expanded: false,
                cut_words: [0; LIMITED_WORDS],
                additional_cost: 0.,
                half_potential: 0.,
            };
            let hot = HotLabel::of(&label);
            let id = storage.retain(label);
            for _ in 0..8 {
                storage.buckets[0][bucket].active.push(id);
                storage.buckets[0][bucket].hot.push(hot);
            }
        }
    }
    engine.refresh_coupled_completion(&graph);
    let entries: usize = engine
        .coupled_completion
        .iter()
        .flat_map(|c| c.pairs.iter().flatten())
        .flatten()
        .map(Vec::len)
        .sum();
    assert!(entries > 0 && entries <= 65_536);
    assert!(engine
        .coupled_completion
        .iter()
        .flat_map(|c| c.pairs.iter().flatten())
        .any(Option::is_none));
}

#[test]
fn replacement_resources_rebuild_full_labels_even_at_unchanged_arc_costs() {
    let mut graph = model();
    let mut engine = engine(&graph);
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    let old = engine.coupled_identity.clone().unwrap();
    let definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    graph.set_coupled_resources(&definitions).unwrap();
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    assert!(!std::sync::Arc::ptr_eq(
        &old,
        engine.coupled_identity.as_ref().unwrap()
    ));
    assert!(std::sync::Arc::ptr_eq(
        engine.coupled_identity.as_ref().unwrap(),
        graph.coupled_kernel().unwrap().cache_identity()
    ));
    assert!(engine.full_ready);
}
#[test]
fn sparse_completion_skips_pair_tables_but_keeps_point_bounds() {
    let graph = model();
    let mut engine = engine(&graph);
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    assert!(engine.coupled_completion.iter().all(|c| c
        .pairs
        .iter()
        .flatten()
        .all(Option::is_none)));
    assert!(engine
        .coupled_completion
        .iter()
        .all(|c| c.queries.iter().all(Vec::is_empty)));
}
