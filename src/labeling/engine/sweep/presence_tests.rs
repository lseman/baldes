use super::*;
use crate::cuts::soft_conflicts::{Mode, PresenceRow};

#[test]
fn other_bucket_filter_preserves_distinct_presence_and_filters_equal_presence() {
    for forward in [false, true] {
        for equal in [false, true] {
            for branching in [false, true] {
                let mut graph = Graph::<2>::new(3, 1).unwrap();
                for v in &mut graph.vertices {
                    v.bounds[0] = crate::Window::new(0., 10.).unwrap();
                }
                graph.add_arc(0, 1, 0.).unwrap();
                graph.add_arc(1, 2, 0.).unwrap();
                for arc in &mut graph.arcs {
                    arc.consumption[0] = 1.;
                }
                let mut engine = LabelingEngine::prepare(
                    &graph,
                    EngineControls {
                        bidirectional: true,
                        steps: [1., 0.].into(),
                        target_buckets: 10,
                        use_symmetric_execution: false,
                        ..Default::default()
                    },
                    Phase {
                        check_ng: true,
                        jump_arcs: true,
                        labels_per_bucket: i64::MAX,
                    },
                )
                .unwrap();
                let vertex = engine.grid.internal[1];
                let direction = engine.grid.direction(forward);
                assert!(direction.depth[vertex] > 0);
                let (candidate_bucket, predecessor) = direction.buckets[vertex]
                    .iter()
                    .enumerate()
                    .find_map(|(i, b)| b.predecessors.first().map(|p| (i, p.bucket.bucket)))
                    .expect("resource grid must have a predecessor bucket");
                let row = PresenceRow::new(3, vec![(0, 2)], Mode::AnyPair).unwrap();
                let mut a = row.empty_state();
                row.visit(&mut a, 0).unwrap();
                let mut b = row.empty_state();
                row.visit(&mut b, if equal { 0 } else { 1 }).unwrap();
                assert_eq!(row.coefficient(&a).unwrap(), row.coefficient(&b).unwrap());
                let make = |bucket, state, cost| StoredLabel {
                    id: 0,
                    local_id: 0,
                    bucket: BucketId { vertex, bucket },
                    vertex: 1,
                    cost,
                    original_cost: 0.,
                    resources: direction.corners[vertex][bucket].into(),
                    ng: 0,
                    branch_state: None,
                    presence_state: Some(vec![state].into_boxed_slice()),
                    parent: None,
                    arc: None,
                    hops: 1,
                    expanded: false,
                    cut_words: [0; LIMITED_WORDS],
                    additional_cost: 0.,
                    half_potential: 0.,
                };
                let lower = make(predecessor, a, -2.);
                let candidate = make(candidate_bucket, b, -1.);
                assert_eq!(
                    LabelStorage::dominates(
                        &graph,
                        &lower,
                        &candidate,
                        forward,
                        InsertionTest::Full,
                        false,
                        &[]
                    ),
                    equal
                );
                let storage = if forward {
                    &mut engine.forward
                } else {
                    &mut engine.backward
                };
                assert!(storage.insert(
                    &graph,
                    lower,
                    forward,
                    InsertionTest::Full,
                    100,
                    false,
                    &[],
                    &mut Statistics::default()
                ));
                assert!(storage.insert(
                    &graph,
                    candidate,
                    forward,
                    InsertionTest::Full,
                    100,
                    false,
                    &[],
                    &mut Statistics::default()
                ));
                storage.buckets[vertex][predecessor].lower_cost = -2.;
                let mut terminals = vec![];
                let counts = (AtomicUsize::new(0), AtomicUsize::new(0));
                let mut sweep = engine.sweep(forward, &mut terminals, &counts);
                let bucket = BucketId {
                    vertex,
                    bucket: candidate_bucket,
                };
                if branching {
                    sweep.filter::<true>(&graph, forward, bucket, &Controls::default());
                } else {
                    sweep.filter::<false>(&graph, forward, bucket, &Controls::default());
                }
                assert_eq!(
                    sweep.storage.buckets[vertex][candidate_bucket]
                        .active
                        .is_empty(),
                    equal,
                    "forward={forward} equal={equal} branching={branching}"
                );
            }
        }
    }
}
