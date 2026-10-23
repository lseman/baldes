use baldes::graph::buckets::BucketId;
use baldes::labeling::storage::{InsertionTest, LabelStorage, StoredLabel};
use baldes::{Graph, Statistics};

#[test]
fn single_label_phase_replaces_cheaper_prefix_and_requests_expansion() {
    let graph = Graph::<2>::new(3, 1).unwrap();
    let mut storage = LabelStorage::new([1, 1, 1]);
    let mut stats = Statistics::default();
    let label = |cost, resource| StoredLabel {
        id: 0,
        local_id: 0,
        bucket: BucketId {
            vertex: 1,
            bucket: 0,
        },
        vertex: 1,
        cost,
        original_cost: cost,
        resources: [resource, 0.].into(),
        ng: 0,
        branch_state: None,
        presence_state: None,
        coupled_state: None,
        parent: None,
        arc: None,
        hops: 1,
        expanded: false,
        cut_words: [0; 20],
        additional_cost: 0.,
        half_potential: 0.,
    };
    assert!(storage.insert(
        &graph,
        label(5., 1.),
        true,
        InsertionTest::ResourcesAndCost,
        1,
        false,
        &[],
        &mut stats
    ));
    let old = storage.buckets[1][0].active[0];
    storage.labels[old].expanded = true;
    // A worse resource value must not block the native cost-only heuristic.
    assert!(storage.insert(
        &graph,
        label(4., 2.),
        true,
        InsertionTest::ResourcesAndCost,
        1,
        false,
        &[],
        &mut stats
    ));
    let current = storage.buckets[1][0].active[0];
    assert_eq!(storage.labels[current].cost, 4.);
    assert!(!storage.labels[current].expanded);
    assert_eq!(storage.buckets[1][0].archived, vec![old]);
    assert!(!storage.insert(
        &graph,
        label(4. - 0.5e-6, 0.),
        true,
        InsertionTest::ResourcesAndCost,
        1,
        false,
        &[],
        &mut stats
    ));
    assert_eq!(storage.buckets[1][0].active, vec![current]);
}
