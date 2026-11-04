use super::*;

fn graph() -> Graph<1> {
    let mut g = Graph::<1>::new(4, 1).unwrap();
    for (u, v) in [(0, 1), (0, 2), (1, 2), (2, 1), (1, 3), (2, 3)] {
        g.add_arc(u, v, 1.).unwrap();
    }
    g.add_elementarity_set(&[1]).unwrap();
    g.add_elementarity_set(&[2]).unwrap();
    for v in &mut g.edit().vertices {
        v.memory.clear();
    }
    g
}

const DISTANCES: [f64; 4] = [0., 1., 1., 0.];

#[test]
fn distance_memories_fill_every_vertex_and_arc_when_all_are_empty() {
    let mut g = graph();
    assert!(initialize_distances(&mut g, &DISTANCES, 2).unwrap());
    assert!(g.vertices[0].memory.is_empty());
    assert_eq!(g.vertices[1].memory, [0, 1]);
    assert_eq!(g.vertices[2].memory, [0, 1]);
    assert!(g.vertices[3].memory.is_empty());
}

#[test]
fn distance_memories_write_nothing_when_any_memory_exists() {
    // Native scans all vertex then all arc memories before writing anything.
    let mut g = graph();
    g.edit().vertices[2].memory = vec![1];
    assert!(!initialize_distances(&mut g, &DISTANCES, 2).unwrap());
    assert!(g.vertices[1].memory.is_empty());
    assert_eq!(g.vertices[2].memory, [1]);

    let mut g = graph();
    g.edit().arcs[5].memory = vec![1];
    assert!(!initialize_distances(&mut g, &DISTANCES, 2).unwrap());
    assert!(g.vertices.iter().all(|v| v.memory.is_empty()));
}
