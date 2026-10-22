use super::*;

#[test]
fn export_order_survives_partition_tuning_replacement_retention_and_forks() {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for (tail, head, cost) in [(0, 1, 1.), (1, 3, 1.), (0, 2, 2.), (2, 3, 2.), (1, 2, 1.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    for count in [1, 8] {
        let mut storage = PathStorage::new(count).unwrap();
        let make = |arcs: Vec<usize>, hash, visited, cost| {
            let mut bits = [0; 16];
            bits[0] = visited;
            Path {
                arcs,
                element_hash: hash,
                visited: bits,
                original_cost: cost,
                reduced_cost: cost,
            }
        };
        storage.insert(make(vec![0, 1], 1, 2, 2.));
        storage.insert(make(vec![2, 3], 0, 1, 4.));
        storage.insert(make(vec![0, 4, 3], 1, 3, 4.));
        let paths = |s: &PathStorage| {
            s.routes(&graph)
                .unwrap()
                .into_iter()
                .map(|r| r.arcs)
                .collect::<Vec<_>>()
        };
        assert_eq!(paths(&storage), [vec![0, 1], vec![0, 4, 3], vec![2, 3]]);
        let mut child = storage.clone();
        child.retain(|p| p.arcs != [0, 1]);
        assert_eq!(paths(&child), [vec![0, 4, 3], vec![2, 3]]);
        assert!(child.insert(make(vec![0, 4, 3], 1, 3, 3.5)));
        assert_eq!(paths(&child), [vec![0, 4, 3], vec![2, 3]]);
        assert_eq!(paths(&storage).len(), 3);
        child.inspect(&graph, 3., &[], &[]).unwrap();
        assert!(paths(&child).is_empty());
        assert!(child.path_order.is_empty());
    }
}
