use super::*;

fn walk_costs(
    g: &Graph<2>,
    events: &[Vec<usize>],
    mode: Mode,
    dual: f64,
    cutoff: f64,
) -> std::collections::BTreeMap<Vec<i32>, f64> {
    fn walk(g: &Graph<2>, at: usize, path: &mut Vec<usize>, all: &mut Vec<Vec<usize>>) {
        if at == g.sink {
            all.push(path.clone());
            return;
        }
        if path.len() == 6 {
            return;
        }
        for (i, a) in g.arcs.iter().enumerate().filter(|(_, a)| a.tail == at) {
            path.push(i);
            walk(g, a.head, path, all);
            path.pop();
        }
    }
    let mut all = vec![];
    walk(g, g.source, &mut vec![], &mut all);
    all.into_iter()
        .filter_map(|p| {
            // Enumeration is elementary on the original vertex/arc element
            // sets even when pricing uses an NG relaxation. Variable events
            // may still repeat on distinct elementary vertices or arcs.
            let mut elements: HashSet<_> = g.vertices[g.source].elements.iter().copied().collect();
            for &arc in &p {
                for &e in g.arcs[arc]
                    .elements
                    .iter()
                    .chain(&g.vertices[g.arcs[arc].head].elements)
                {
                    if !elements.insert(e) {
                        return None;
                    }
                }
            }
            let value = cost(g, events, mode, dual, &p);
            (value <= cutoff - baldes::labeling::EPSILON)
                .then(|| (p.iter().map(|&a| g.arcs[a].id).collect(), value))
        })
        .collect()
}
fn route_map(routes: &[baldes::Route]) -> std::collections::BTreeMap<Vec<i32>, f64> {
    routes.iter().map(|r| (r.arcs.clone(), r.cost)).collect()
}
fn compare_paths(actual: &[baldes::Route], expected: &std::collections::BTreeMap<Vec<i32>, f64>) {
    let actual = route_map(actual);
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>()
    );
    for (path, cost) in actual {
        assert!(
            (cost - expected[&path]).abs() < 1e-10,
            "{path:?}: {cost} != {}",
            expected[&path]
        );
    }
}

#[test]
fn presence_enumeration_matches_every_elementary_path_and_cutoff() {
    for symmetric in [false, true] {
        let (g, events) = fixtures(symmetric);
        for mode in [Mode::PairCount, Mode::AnyPair] {
            for dual in [-0.625, 0., 0.7] {
                for bidirectional in [false, true] {
                    for cutoff in [0., 1., 2.5, 10.] {
                        let mut solver = owned_solver(g.clone(), bidirectional);
                        solver
                            .set_presence_rows(input(&g, &events, mode, dual))
                            .unwrap();
                        let routes = solver.enumerate(cutoff).unwrap();
                        let expected = walk_costs(&g, &events, mode, dual, cutoff);
                        assert_eq!(route_map(&routes).keys().collect::<Vec<_>>(),expected.keys().collect::<Vec<_>>(),
                            "symmetric={symmetric} mode={mode:?} dual={dual} bidirectional={bidirectional} cutoff={cutoff}");
                        compare_paths(&routes, &expected);
                        assert!(solver.enumeration.as_ref().unwrap().completed);
                        let repriced = solver.price_phase(0, 150, false).unwrap();
                        compare_paths(&repriced.columns, &expected);
                        if let Some(minimum) = expected.values().copied().reduce(f64::min) {
                            assert!((repriced.minimum_reduced_cost - minimum).abs() < 1e-10);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn retained_presence_paths_reprice_and_fork_without_losing_coefficients() {
    let (mut g, events) = fixtures(false);
    let original_graph = g.clone();
    let mut parent = owned_solver(g.clone(), true);
    parent
        .set_presence_rows(input(&g, &events, Mode::PairCount, 0.7))
        .unwrap();
    compare_paths(
        &parent.enumerate(10.).unwrap(),
        &walk_costs(&g, &events, Mode::PairCount, 0.7, 10.),
    );
    let mut child = parent.fork_for_node();
    compare_paths(
        &child.price_phase(0, 150, false).unwrap().columns,
        &walk_costs(&g, &events, Mode::PairCount, 0.7, 10.),
    );
    for arc in &mut child.graph.arcs {
        arc.reduced_cost -= 0.125;
    }
    for arc in &mut g.arcs {
        arc.reduced_cost -= 0.125;
    }
    compare_paths(
        &child.inspect_enumeration(10.).unwrap().unwrap(),
        &walk_costs(&g, &events, Mode::PairCount, 0.7, 10.),
    );
    child
        .set_presence_rows(input(&g, &events, Mode::AnyPair, -0.625))
        .unwrap();
    assert!(child.enumeration.is_none());
    compare_paths(
        &child.enumerate(10.).unwrap(),
        &walk_costs(&g, &events, Mode::AnyPair, -0.625, 10.),
    );
    assert_eq!(parent.presence_rows()[0].dual, 0.7);
    assert!(parent.enumeration.as_ref().unwrap().completed);
    compare_paths(
        &parent.price_phase(0, 150, false).unwrap().columns,
        &walk_costs(&original_graph, &events, Mode::PairCount, 0.7, 10.),
    );
}

#[test]
fn presence_enumeration_budget_failure_never_publishes_partial_paths() {
    let (g, events) = fixtures(false);
    let mut solver = owned_solver(g.clone(), true);
    solver
        .set_presence_rows(input(&g, &events, Mode::PairCount, 0.7))
        .unwrap();
    solver
        .set_option("BALDESmaxNumOfEnumeratedSolutions", 1.)
        .unwrap();
    assert!(solver.enumerate(10.).is_err());
    assert!(solver.enumeration.is_none());
    solver
        .set_option("BALDESmaxNumOfEnumeratedSolutions", 1000.)
        .unwrap();
    compare_paths(
        &solver.enumerate(10.).unwrap(),
        &walk_costs(&g, &events, Mode::PairCount, 0.7, 10.),
    );
    assert!(solver.enumeration.as_ref().unwrap().completed);
}

#[test]
fn presence_enumeration_and_repricing_compose_with_registered_cut_tables() {
    let mut g = Graph::<2>::new(4, 1).unwrap();
    for v in 0..4 {
        g.set_bounds(v, 0, 0., 4.).unwrap();
    }
    for v in 0..3 {
        let id = g.add_arc(v, v + 1, 0.).unwrap();
        g.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    let events = vec![vec![0], vec![1], vec![]];
    for bidirectional in [false, true] {
        for mode in [Mode::PairCount, Mode::AnyPair] {
            for dual in [-0.75, 0.75] {
                let mut solver = owned_solver(g.clone(), bidirectional);
                solver
                    .set_presence_rows(input(&g, &events, mode, 0.3))
                    .unwrap();
                solver
                    .cuts
                    .add_strong_k_path_cut(4, &[1, 2], 1, dual)
                    .unwrap();
                let expected = cost(&g, &events, mode, 0.3, &[0, 1, 2]) - dual;
                let routes = solver.enumerate(10.).unwrap();
                assert_eq!(routes.len(), 1);
                assert!((routes[0].cost - expected).abs() < 1e-12);
                let repriced = solver.price_phase(0, 150, false).unwrap();
                assert_eq!(repriced.columns.len(), 1);
                assert!((repriced.minimum_reduced_cost - expected).abs() < 1e-12);
            }
        }
    }
}
