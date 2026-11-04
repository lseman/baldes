use super::*;
use std::time::Duration;

fn diamond() -> (Graph<2>, Vec<f64>) {
    // 0 -> {1, 2} -> 3, plus the pair route 0 -> 1 -> 2 -> 3.
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head, cost) in [
        (0, 1, -1.),
        (1, 3, 0.),
        (0, 2, -1.),
        (2, 3, 0.),
        (1, 2, -1.),
    ] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let weights = graph.arcs.iter().map(|a| a.cost).collect();
    (graph, weights)
}

fn run(graph: &Graph<2>, weights: &[f64], rules: &[RyanFosterConstraint]) -> Option<Option<f64>> {
    minimum(
        graph,
        weights,
        rules,
        |_| false,
        &mut 1_000_000,
        Instant::now() + Duration::from_secs(1),
    )
}

#[test]
fn ryan_foster_walks_exclude_split_or_joined_pairs() {
    let (graph, weights) = diamond();
    let rule = |together| RyanFosterConstraint {
        first: 1,
        second: 2,
        together,
    };
    // Together: only the pair route (cost -2) remains.
    assert_eq!(run(&graph, &weights, &[rule(true)]), Some(Some(-2.)));
    // Apart: single-customer routes (cost -1) remain.
    assert_eq!(run(&graph, &weights, &[rule(false)]), Some(Some(-1.)));
    // Without the pair arc, the together child has no permitted route.
    let mut split = graph.clone();
    split.edit().arcs.pop();
    let w: Vec<_> = split.arcs.iter().map(|a| a.cost).collect();
    assert_eq!(run(&split, &w, &[rule(true)]), Some(None));
    // No rows: the caller keeps its own unrestricted bound.
    assert_eq!(run(&graph, &weights, &[]), None);
}
