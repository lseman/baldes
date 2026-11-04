use super::*;

#[allow(dead_code)]
#[path = "ng_preparation_reference.rs"]
mod reference;

fn assert_reference(graph: &Graph<2>) {
    let actual = PreparedNg::new(graph).unwrap();
    let expected = reference::PreparedNg::new(graph).unwrap();
    assert_eq!(actual.vertex_bits, expected.vertex_bits);
    assert_eq!(actual.local_elements, expected.local_elements);
    for (a, b) in actual
        .forward
        .iter()
        .chain(&actual.backward)
        .zip(expected.forward.iter().chain(&expected.backward))
    {
        assert_eq!(
            (a.forbidden, a.base, &a.transfer),
            (b.forbidden, b.base, &b.transfer)
        );
        for before in [0, 1, 2, 3, 7, 15, NgMask::MAX] {
            for vertex in [0, 1, 3, NgMask::MAX] {
                assert_eq!(a.extend(before, vertex), b.extend(before, vertex));
            }
        }
    }
}

#[test]
fn preparation_preserves_duplicate_and_reordered_arc_and_vertex_elements() {
    let mut graph = Graph::<2>::new(6, 0).unwrap();
    graph.edit().grouped = true;
    for vertex in &mut graph.edit().vertices {
        vertex.memory = vec![5, 3, 1, 0, 4, 2, 1];
    }
    for tail in 0..5 {
        for head in 1..6 {
            if tail == head {
                continue;
            }
            let arc = graph.add_arc(tail, head, 0.).unwrap();
            graph.arc_mut(arc).unwrap().elements = vec![4, 2, 4];
            graph.arc_mut(arc).unwrap().memory = vec![3, 0, 3];
        }
    }
    assert_reference(&graph);
    for vertex in &mut graph.edit().vertices {
        vertex.elements.extend(vertex.elements.clone());
        vertex.elements.reverse();
        vertex.memory.reverse();
    }
    for arc in &mut graph.edit().arcs {
        arc.elements.reverse();
        arc.elements.extend(arc.elements.clone());
    }
    assert_reference(&graph);
    for vertex in &mut graph.edit().vertices {
        let mut duplicates = vec![0; 10_000];
        duplicates.extend(vertex.elements.clone());
        vertex.elements = duplicates;
    }
    assert_reference(&graph);
    graph.edit().elementary = true;
    assert_reference(&graph);
    graph.edit().elementary = false;
    let mut ungrouped = graph.clone();
    ungrouped.edit().grouped = false;
    for (id, vertex) in ungrouped.edit().vertices.iter_mut().enumerate() {
        vertex.elements = vec![id];
    }
    for arc in &mut ungrouped.edit().arcs {
        arc.elements.clear();
    }
    ungrouped
        .set_route_domain(crate::graph::RouteDomain::BoundedWalks { max_arcs: 8 })
        .unwrap();
    assert_reference(&ungrouped);
    graph.edit().grouped = true;
    graph
        .set_route_domain(crate::graph::RouteDomain::BoundedWalks { max_arcs: 8 })
        .unwrap();
    assert!(graph.hard_walk_groups());
    assert_reference(&graph);
}

#[test]
fn preparation_preserves_highest_available_local_memory_bit() {
    let mut graph = Graph::<2>::new(NG_MEMORY_CAPACITY, 0).unwrap();
    graph.edit().grouped = true;
    for vertex in &mut graph.edit().vertices {
        vertex.memory = (0..NG_MEMORY_CAPACITY).rev().collect();
    }
    let arc = graph.add_arc(0, NG_MEMORY_CAPACITY - 1, 0.).unwrap();
    graph.arc_mut(arc).unwrap().elements = vec![NG_MEMORY_CAPACITY - 1, 0, NG_MEMORY_CAPACITY - 1];
    assert_reference(&graph);
}

#[test]
fn compact_element_sets_match_reference_across_word_and_domain_boundaries() {
    for universe in [0, 1, 15, 16, 63, 64, 65, 127, 128, 129, 257] {
        for seed in 0..12_usize {
            let mut graph = Graph::<2>::new(6, 0).unwrap();
            graph.edit().grouped = true;
            graph.edit().element_count = universe;
            let mut pool = vec![];
            if universe > 0 {
                pool.extend([0, universe / 2, universe - 1]);
                if universe > 64 {
                    pool.extend([63, 64]);
                }
                pool.sort_unstable();
                pool.dedup();
            }
            for (index, vertex) in graph.edit().vertices.iter_mut().enumerate() {
                vertex.elements = pool
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (i + index + seed) % 3 == 0)
                    .map(|(_, &id)| id)
                    .collect();
                vertex.memory = pool
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (i + index * 3 + seed) % 2 == 0)
                    .map(|(_, &id)| id)
                    .collect();
                vertex.elements.extend(vertex.elements.clone());
                vertex.memory.reverse();
            }
            for (tail, head) in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 1)] {
                let id = graph.add_arc(tail, head, 0.).unwrap();
                let arc = graph.arc_mut(id).unwrap();
                arc.elements = pool
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (i + tail + seed) % 3 == 0)
                    .map(|(_, &id)| id)
                    .collect();
                arc.memory = pool
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (i + head + seed) % 2 == 0)
                    .map(|(_, &id)| id)
                    .collect();
                arc.memory.extend(arc.memory.clone());
                arc.elements.reverse();
            }
            assert_reference(&graph);
        }
    }
}

#[test]
fn compact_and_sparse_preparation_preserve_capacity_and_invalid_id_errors() {
    for universe in [128, 129] {
        let mut graph = Graph::<2>::new(4, 0).unwrap();
        graph.edit().grouped = true;
        graph.edit().element_count = universe;
        for vertex in &mut graph.edit().vertices {
            vertex.memory = (0..=NG_MEMORY_CAPACITY).collect();
        }
        graph.add_arc(0, 3, 0.).unwrap();
        assert_eq!(
            PreparedNg::new(&graph).err().unwrap().message,
            reference::PreparedNg::new(&graph).err().unwrap().message
        );
        for vertex in &mut graph.edit().vertices {
            vertex.memory = vec![universe];
        }
        assert_eq!(
            PreparedNg::new(&graph).err().unwrap().message,
            reference::PreparedNg::new(&graph).err().unwrap().message
        );
    }
}
