use baldes::{
    cuts::{CutClass, MemoryKind, RankOneCut},
    pricing::Solver,
};
use std::{collections::BTreeSet, str::FromStr};

fn parse<T: FromStr>(s: &str) -> T {
    s.parse().ok().unwrap()
}

#[test]
fn covering_cut_keeps_below_cutoff_paths_through_its_support() {
    let mut lines =
        include_str!("../fixtures/enumeration/covering-active-staged-capture.txt").lines();
    let mut header = lines.next().unwrap().split_whitespace();
    let n: usize = parse(header.next().unwrap());
    let arc_count: usize = parse(header.next().unwrap());
    let cutoff: f64 = parse(header.next().unwrap());
    let mut solver = Solver::<2>::new(n, 1).unwrap();
    solver.graph.vertices[0].elements.clear();
    solver.graph.vertices[n - 1].elements.clear();
    for vertex in 0..n {
        solver.graph.set_bounds(vertex, 0, 0., 2.).unwrap();
    }
    let mut after_cut = Vec::new();
    for _ in 0..arc_count {
        let mut row = lines.next().unwrap().split_whitespace();
        let tail = parse(row.next().unwrap());
        let head = parse(row.next().unwrap());
        let cost = parse(row.next().unwrap());
        after_cut.push(parse::<f64>(row.next().unwrap()));
        let consumption = parse(row.next().unwrap());
        let id = solver.graph.add_arc(tail, head, cost).unwrap();
        solver.graph.arc_mut(id).unwrap().consumption[0] = consumption;
    }
    let before_cut: Vec<f64> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(parse)
        .collect();
    assert_eq!(before_cut.len(), arc_count);
    let native_paths: BTreeSet<Vec<i32>> = lines
        .map(|line| line.split_whitespace().map(parse).collect())
        .collect();
    assert_eq!(native_paths.len(), 12);
    solver.bucket_steps[0] = 1.;
    for (arc, cost) in solver.graph.arcs.iter_mut().zip(before_cut) {
        arc.reduced_cost = cost;
    }
    solver.fix(1.499999996403).unwrap();
    for (arc, cost) in solver.graph.arcs.iter_mut().zip(after_cut) {
        arc.reduced_cost = cost;
    }
    let mut weights = vec![0; n];
    let mut memory = vec![false; n];
    for vertex in 1..=3 {
        weights[vertex] = 1;
        memory[vertex] = true;
    }
    solver.cuts.cuts.push(RankOneCut {
        weights,
        graph_support: Default::default(),
        graph_pricing_support: Default::default(),
        general_kpath_support: None,
        memory,
        graph_dependent_memory: false,
        graph_memories: Default::default(),
        memory_kind: MemoryKind::Vertex,
        arc_memory: Default::default(),
        graph_arc_memories: Default::default(),
        arc_weights: Default::default(),
        graph_arc_weights: Default::default(),
        visit_once: false,
        type2: false,
        denominator: 2,
        dual: 1.,
        class: CutClass::Covering,
        rhs: 2,
        kind: 1,
        id: 0,
        ordered_support: Some(vec![(1, 1), (2, 1), (3, 1)]),
    });
    solver.fix(cutoff).unwrap();
    let routes = solver.enumerate(cutoff).unwrap();
    let actual: BTreeSet<Vec<i32>> = routes.iter().map(|route| route.arcs.clone()).collect();
    assert!(native_paths.is_subset(&actual));
    let extras: BTreeSet<Vec<i32>> = actual.difference(&native_paths).cloned().collect();
    assert_eq!(
        extras,
        BTreeSet::from([
            vec![0, 1],
            vec![0, 2, 5],
            vec![0, 3, 9],
            vec![4, 5],
            vec![4, 7, 9],
            vec![8, 9],
        ])
    );
    let singleton = routes.iter().find(|route| route.arcs == [0, 1]).unwrap();
    assert_eq!(singleton.cost, 0.);
}
