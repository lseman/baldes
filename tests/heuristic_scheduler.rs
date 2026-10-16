use baldes::graph::buckets::BucketId;
use baldes::labeling::storage::{InsertionTest, LabelStorage, StoredLabel};
use baldes::{Graph, Statistics};

#[test]
fn single_label_phase_replaces_cheaper_prefix_and_requests_expansion() {
    let graph = Graph::new(3, 1).unwrap();
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
        resources: [resource, 0.],
        ng: 0,
        parent: None,
        arc: None,
        hops: 1,
        expanded: false,
        cut_words: [0; 20],
        additional_cost: 0.,
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

#[cfg(feature = "clp")]
#[test]
fn c201_25_root_matches_native_heuristic_and_master_trajectory() {
    use baldes::master::{
        column_generation::Status,
        root::{self, RootOptions, RootProblem},
    };
    use baldes::vrptw::{Instance, Rounding};
    let mut instance =
        Instance::parse(include_str!("../instances/C201.txt")).unwrap();
    instance.truncate(25).unwrap();
    let mut graph = instance.graph(8, Rounding::None).unwrap();
    for v in 0..27 {
        let neighbours = if v == 0 || v == 26 {
            vec![v]
        } else {
            let mut ids: Vec<_> = (1..=25).collect();
            ids.sort_by(|&a, &b| {
                instance
                    .distance(v, a, Rounding::None)
                    .total_cmp(&instance.distance(v, b, Rounding::None))
                    .then(a.cmp(&b))
            });
            ids.truncate(8);
            ids
        };
        graph.set_ng(v, &neighbours).unwrap();
    }
    let mut options = RootOptions::default();
    options.enumerate = false;
    options.cg.multiplicity = 25.;
    let result = root::solve(RootProblem::set_partitioning(graph, 25.).unwrap(), options).unwrap();
    assert_eq!(result.master.status, Status::Optimal);
    assert!((result.master.lp - 215.54256527689378).abs() < 1e-9);
    assert_eq!(result.master.history[0].new_columns, 26);
    assert_eq!(result.master.iterations, 51);
    assert_eq!(result.master.pricing_calls, 65);
}
