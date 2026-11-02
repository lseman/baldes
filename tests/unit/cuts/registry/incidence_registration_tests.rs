use super::*;

fn graph() -> GraphData {
    GraphData {
        id: 7,
        source: 0,
        sink: 2,
        elem_set_count: 1,
        packing_set_count: 1,
        vertices: vec![
            VertexData {
                id: 0,
                ..Default::default()
            },
            VertexData {
                id: 1,
                packing_sets: vec![0],
                ..Default::default()
            },
            VertexData {
                id: 2,
                ..Default::default()
            },
        ],
        arcs: vec![
            ArcData {
                id: 10,
                tail: 0,
                head: 1,
                packing_sets: vec![0],
                ..Default::default()
            },
            ArcData {
                id: 12,
                tail: 1,
                head: 1,
                ..Default::default()
            },
            ArcData {
                id: 11,
                tail: 1,
                head: 2,
                ..Default::default()
            },
        ],
        ..Default::default()
    }
}

fn cut() -> SeparatedCut {
    SeparatedCut {
        kind: 0,
        num_rows: 1,
        denominator: 2,
        numerators: vec![1],
        set_ids: vec![0],
        memory_type: 1,
        memory: [(7, [10, 11].into_iter().collect())].into_iter().collect(),
        ..Default::default()
    }
}

#[test]
fn registration_deduplicates_arc_and_head_set_and_preserves_missing_memory() {
    let g = graph();
    let mut registry = CutRegistry::default();
    let index = registry.register_separated_cut(&[&g], &cut()).unwrap();
    assert_eq!(registry.cuts[index].graph_arc_weights[&7].get(&10), None);
    let route = [(0, 1, 10), (1, 1, 12), (1, 2, 11)];
    assert_eq!(
        coefficient(&registry.cuts[index], 7, &route, false).unwrap(),
        0
    );
    let mut without_memory = cut();
    without_memory.memory.clear();
    let index = registry
        .register_separated_cut(&[&g], &without_memory)
        .unwrap();
    assert!(!registry.cuts[index].graph_arc_memories.contains_key(&7));
    assert_eq!(
        coefficient(&registry.cuts[index], 7, &route, false).unwrap(),
        0
    );
    assert_eq!(
        coefficient(&registry.cuts[index], 7, &route, true).unwrap(),
        1
    );
}

#[test]
fn registration_rejects_invalid_descriptors_without_adding_a_cut() {
    let g = graph();
    let mut registry = CutRegistry::default();
    let mut invalid = cut();
    invalid.numerators[0] = 2;
    assert!(registry.register_separated_cut(&[&g], &invalid).is_err());
    invalid = cut();
    invalid.memory.get_mut(&7).unwrap().insert(99);
    assert!(registry.register_separated_cut(&[&g], &invalid).is_err());
    invalid = cut();
    invalid.memory.insert(8, BTreeSet::new());
    assert!(registry.register_separated_cut(&[&g], &invalid).is_err());
    let mut bad_graph = graph();
    bad_graph.arcs[0].packing_sets = vec![0, 0];
    assert!(registry
        .register_separated_cut(&[&bad_graph], &cut())
        .is_err());
    assert!(registry.cuts.is_empty());
}
