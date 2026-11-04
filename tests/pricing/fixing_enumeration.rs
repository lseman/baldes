mod covering_fixing_parity;
mod cut_active_enumeration;
mod window_parity;

use baldes::enumeration::{Controls, Enumeration};
use baldes::pricing::Solver;

#[test]
fn raw_functor_pricing_restores_discarded_paths_and_keeps_branch_restrictions() {
    let mut solver = baldes::pricing::Solver::<2>::new(3, 1).unwrap();
    for v in 0..3 {
        solver.graph.set_bounds(v, 0, 0., 10.).unwrap();
    }
    solver.graph.edit().elementary = true;
    let a = solver.graph.add_arc(0, 1, 0.).unwrap();
    solver.graph.arc_mut(a).unwrap().variables = vec![(0, 1e8)];
    let b = solver.graph.add_arc(0, 1, 0.05).unwrap();
    solver.graph.add_arc(1, 2, 0.).unwrap();
    solver.maximum_variable_id = Some(0);
    solver.cache_variable_costs(&[1e-9]).unwrap();
    solver
        .options
        .set("BALDESapplyReducedCostFixing", 1.)
        .unwrap();
    solver.price_phase(0, 10, false).unwrap();
    assert_eq!(solver.run_fixing_enumeration(0, 0., 0.01).unwrap().state, 2);
    assert_eq!(solver.enumeration.as_ref().unwrap().paths.len, 1);
    let cached: Vec<_> = solver.graph.arcs.iter().map(|a| a.reduced_cost).collect();
    let raw = solver.price_functor_phase(0, 1).unwrap();
    assert!(raw.exact);
    assert_eq!(raw.minimum_reduced_cost, 0.05);
    assert_eq!(raw.columns.len(), 1);
    assert!(raw.columns[0].arcs.contains(&b));
    assert_eq!(
        solver
            .graph
            .arcs
            .iter()
            .map(|a| a.reduced_cost)
            .collect::<Vec<_>>(),
        cached
    );
    assert_eq!(solver.enumeration.as_ref().unwrap().paths.len, 1);
    assert!(raw.statistics.generated > 0);
    assert_eq!(
        solver.engine.as_ref().unwrap().statistics.generated,
        raw.statistics.generated
    );

    // Cached inspection itself has no labeling sweep; the raw validation must
    // still respect a zero saved-point budget and request rollback on failure.
    solver.rollback_point_saved = true;
    solver
        .options
        .set("BALDEShardTimeThresholdInPricing", 0.)
        .unwrap();
    assert!(solver.price_functor_phase(0, 1).is_err());
    assert!(solver.requires_rollback);
    assert_eq!(solver.enumeration.as_ref().unwrap().paths.len, 1);
    assert_eq!(
        solver
            .graph
            .arcs
            .iter()
            .map(|a| a.reduced_cost)
            .collect::<Vec<_>>(),
        cached
    );

    solver.rollback_point_saved = false;
    solver
        .options
        .set("BALDEShardTimeThresholdInPricing", 20.)
        .unwrap();
    solver.set_forbidden_arcs(vec![1]).unwrap();
    let raw = solver.price_functor_phase(0, 1).unwrap();
    assert_eq!(raw.minimum_reduced_cost, 0.1);
    assert!(raw.columns[0].arcs.contains(&a));
    solver.set_forbidden_arcs(vec![0]).unwrap();
    let raw = solver.price_functor_phase(0, 1).unwrap();
    assert_eq!(raw.minimum_reduced_cost, 0.05);
    assert!(raw.columns[0].arcs.contains(&b));
}

#[test]
fn pricing_deadlines_respect_direct_options_and_explicit_caller_caps() {
    let mut solver = baldes::pricing::Solver::<2>::new(3, 1).unwrap();
    solver.graph.add_arc(0, 1, -1.).unwrap();
    solver.graph.add_arc(1, 2, 0.).unwrap();
    solver.controls.time_limit = std::time::Duration::ZERO;
    solver.rollback_point_saved = true;
    solver
        .options
        .set("BALDEShardTimeThresholdInPricing", 20.)
        .unwrap();
    assert!(solver.price_phase(0, 1, false).is_err());
    assert!(solver.requires_rollback);
    assert_eq!(solver.controls.time_limit, std::time::Duration::ZERO);
    assert_eq!(
        solver
            .options
            .get("BALDEShardTimeThresholdInPricing")
            .unwrap(),
        20.
    );
    // Typed option updates must also take effect without going through the
    // convenience setter that synchronizes the default controls budget.
    solver.rollback_point_saved = false;
    solver.controls.time_limit = std::time::Duration::from_secs(1000);
    solver
        .options
        .set("BALDEShardTimeThresholdInPricing", 0.)
        .unwrap();
    assert!(solver.price_phase(0, 1, false).is_err());
    assert!(!solver.requires_rollback);
    assert_eq!(
        solver.controls.time_limit,
        std::time::Duration::from_secs(1000)
    );
}

#[test]
fn zero_deadline_applies_to_small_heuristic_and_exact_sweeps() {
    for phase in 0..3 {
        for saved in [false, true] {
            let mut solver = Solver::<2>::new(3, 1).unwrap();
            for vertex in 0..3 {
                solver.graph.set_bounds(vertex, 0, 0., 2.).unwrap();
            }
            solver.graph.add_arc(0, 1, -1.).unwrap();
            solver.graph.add_arc(1, 2, 0.).unwrap();
            solver.controls.time_limit = std::time::Duration::ZERO;
            solver.rollback_point_saved = saved;
            let error = solver.price_phase(phase, 10, false).unwrap_err();
            assert!(error.message.contains("time limit"), "{error}");
            assert_eq!(solver.requires_rollback, saved);
            solver.controls.time_limit = std::time::Duration::from_secs(1);
            let result = solver.price_phase(phase, 10, false).unwrap();
            assert!(!result.columns.is_empty());
            assert!(!solver.requires_rollback);
        }
    }
}

fn frontier_solver() -> Solver<2> {
    let mut solver = Solver::<2>::new(4, 1).unwrap();
    for vertex in 0..4 {
        solver.graph.set_bounds(vertex, 0, 0., 4.).unwrap();
    }
    for (tail, head, cost) in [(0, 1, 0.), (1, 3, 1.), (0, 2, 0.), (2, 3, 2.), (0, 3, 3.)] {
        let arc = solver.graph.add_arc(tail, head, cost).unwrap();
        solver.graph.arc_mut(arc).unwrap().consumption[0] = 1.;
    }
    solver.price_phase(0, 10, false).unwrap();
    solver
}

#[test]
fn enumeration_frontier_records_discarded_terminal_scores_in_their_cost_epoch() {
    let solver = frontier_solver();
    let graph = &solver.graph;
    let grid = &solver.engine.as_ref().unwrap().grid;
    let controls = Controls {
        cutoff: 1.5,
        preserve_arc_paths: true,
        ..Default::default()
    };
    let universe = Enumeration::run(graph, grid, &controls, None).unwrap();
    assert_eq!(universe.paths.len, 1);
    assert_eq!(universe.pruning.terminal_rejections, 2);
    assert_eq!(universe.pruning.terminal_minimum, Some(2.));
    assert_eq!(universe.pruning.completion_rejections, 0);
    assert!(!universe.pruning.representative_pruning_enabled);
    let mut parent = solver;
    parent.enumeration = Some(universe);
    let mut child = parent.fork_for_node();
    for arc in &mut child.graph.edit().arcs {
        arc.reduced_cost += 10.;
    }
    child.inspect_enumeration(f64::INFINITY).unwrap();
    assert_eq!(
        child.enumeration.as_ref().unwrap().pruning,
        parent.enumeration.as_ref().unwrap().pruning
    );
    assert_eq!(
        parent
            .enumeration
            .as_ref()
            .unwrap()
            .paths
            .routes(&parent.graph)
            .unwrap()[0]
            .cost,
        1.
    );
}

#[test]
fn enumeration_frontier_keeps_completion_rejections_separate_from_numeric_scores() {
    let solver = frontier_solver();
    let universe = Enumeration::run_directional(
        &solver.graph,
        &solver.engine.as_ref().unwrap().grid,
        &Controls {
            cutoff: 100.,
            ..Default::default()
        },
        |_, _| false,
    )
    .unwrap();
    assert!(universe.pruning.completion_rejections > 0);
    assert!(universe.pruning.representative_pruning_enabled);
    assert_eq!(universe.pruning.terminal_minimum, None);
    assert_eq!(universe.pruning.terminal_rejections, 0);
}

#[test]
fn enumeration_frontier_reports_ordered_join_pruning_separately() {
    let mut solver = frontier_solver();
    for arc in &mut solver.graph.edit().arcs {
        arc.consumption[0] = if arc.tail == 0 && arc.head != 3 {
            3.
        } else {
            0.
        };
    }
    solver.restore_arcs();
    solver.price_phase(0, 10, false).unwrap();
    let universe = Enumeration::run_directional(
        &solver.graph,
        &solver.engine.as_ref().unwrap().grid,
        &Controls {
            cutoff: 1.5,
            preserve_arc_paths: true,
            ..Default::default()
        },
        |_, _| true,
    )
    .unwrap();
    assert_eq!(universe.paths.len, 1);
    assert!(universe.pruning.ordered_join_breaks > 0);
    assert_eq!(universe.pruning.ordered_join_minimum, Some(2.));
    assert_eq!(universe.pruning.completion_rejections, 0);
    assert!(!universe.pruning.representative_pruning_enabled);
}

#[test]
fn completion_callbacks_never_receive_non_elementary_vertex_extensions() {
    let mut solver = Solver::<2>::new(5, 2).unwrap();
    solver.graph.edit().vertices[0].elements.clear();
    solver.graph.edit().vertices[4].elements.clear();
    for vertex in 0..5 {
        solver.graph.set_bounds(vertex, 0, 0., 100.).unwrap();
        solver.graph.set_bounds(vertex, 1, 0., 2.).unwrap();
    }
    for customer in 1..4 {
        let mut arcs = vec![(0, customer), (customer, 4)];
        arcs.extend(
            (1..4)
                .filter(|&other| other != customer)
                .map(|other| (customer, other)),
        );
        for (tail, head) in arcs {
            let id = solver.graph.add_arc(tail, head, 1.).unwrap();
            solver.graph.arc_mut(id).unwrap().consumption =
                [1., if head == 4 { 0. } else { 1. }].into();
        }
    }
    solver.price_phase(0, 10, false).unwrap();
    let calls = std::cell::Cell::new(0);
    let universe = Enumeration::run_directional(
        &solver.graph,
        &solver.engine.as_ref().unwrap().grid,
        &Controls {
            cutoff: 100.,
            preserve_arc_paths: true,
            ..Default::default()
        },
        |_, label| {
            for &element in &solver.graph.vertices[label.vertex].elements {
                assert_eq!(label.visited[element / 64] & (1u64 << (element % 64)), 0);
            }
            calls.set(calls.get() + 1);
            true
        },
    )
    .unwrap();
    assert_eq!(calls.get(), 24);
    assert_eq!(universe.paths.len, 9);
}

#[test]
fn inspection_drops_enumeration_after_resource_bounds_change() {
    let mut solver = Solver::<2>::new(3, 1).unwrap();
    for vertex in 0..3 {
        solver.graph.set_bounds(vertex, 0, 0., 4.).unwrap();
    }
    for (tail, head) in [(0, 1), (1, 2)] {
        let id = solver.graph.add_arc(tail, head, 1.).unwrap();
        solver.graph.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    assert_eq!(solver.enumerate(10.).unwrap().len(), 1);
    solver.graph.set_bounds(1, 0, 0., 0.5).unwrap();
    // The only prefix consumes one unit, so no route can visit vertex one.
    assert!(solver.inspect_enumeration(10.).unwrap().is_none());
    assert!(solver.enumeration.is_none());
    assert!(solver.price_phase(0, 10, false).unwrap().columns.is_empty());
}

#[test]
fn enumeration_rebuilds_completion_after_resource_relaxation() {
    let mut solver = Solver::<2>::new(3, 1).unwrap();
    for vertex in 0..3 {
        solver.graph.set_bounds(vertex, 0, 0., 4.).unwrap();
    }
    for (tail, head) in [(0, 1), (1, 2)] {
        let id = solver.graph.add_arc(tail, head, 1.).unwrap();
        solver.graph.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    solver.graph.set_bounds(1, 0, 0., 0.5).unwrap();
    assert!(solver.price_phase(0, 10, false).unwrap().columns.is_empty());
    solver.complete_labels(true).unwrap();
    solver.graph.set_bounds(1, 0, 0., 4.).unwrap();
    // Both unit extensions now fit; the old grid must not hide the route.
    assert_eq!(solver.enumerate(10.).unwrap().len(), 1);
}

#[test]
fn incomplete_retained_universe_never_closes_pricing_or_inspection() {
    use baldes::pricing::phases::Phase;
    use baldes::Graph;
    for inspect_first in [false, true] {
        let mut solver = Solver::<5>::new(4, 5).unwrap();
        let mut graph = Graph::<5>::new(4, 5).unwrap();
        for vertex in 0..4 {
            for resource in 0..5 {
                graph.set_bounds(vertex, resource, 0., 10.).unwrap();
            }
        }
        graph.add_arc(0, 1, -5.).unwrap();
        graph.add_arc(1, 3, 0.).unwrap();
        graph.add_arc(0, 2, -1.).unwrap();
        graph.add_arc(2, 3, 0.).unwrap();
        solver.graph = graph;
        solver.phases = vec![Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        }];
        solver.price_phase(0, 10, false).unwrap();
        let routes = solver.enumerate(0.).unwrap();
        assert_eq!(routes.len(), 2);
        let universe = solver.enumeration.as_mut().unwrap();
        universe.paths.retain(|path| !path.arcs.contains(&0));
        universe.completed = false;
        assert_eq!(universe.paths.len, 1);
        if inspect_first {
            assert!(solver.inspect_enumeration(0.).unwrap().is_none());
        }
        let resumed = solver.price_phase(0, 10, false).unwrap();
        assert!(resumed.exact);
        assert_eq!(resumed.minimum_reduced_cost, -5.);
        assert!(resumed.columns.iter().any(|route| route.arcs == [0, 1]));
        assert!(solver.enumeration.is_none());
        let complete = solver.enumerate(0.).unwrap();
        assert_eq!(complete.len(), 2);
        assert!(solver.enumeration.as_ref().unwrap().completed);
    }
}

fn check_reference_enumeration(input: &str, resources: usize) {
    let mut lines = input.lines();
    let header: Vec<_> = lines.next().unwrap().split_whitespace().collect();
    let vertices: usize = header[0].parse().unwrap();
    let arcs: usize = header[1].parse().unwrap();
    let cutoff: f64 = header[2].parse().unwrap();
    let mut solver = Solver::<2>::new(vertices, resources).unwrap();
    solver.graph.edit().vertices[0].elements.clear();
    solver.graph.edit().vertices[vertices - 1].elements.clear();
    for vertex in 0..vertices {
        solver.graph.set_bounds(vertex, 0, 0., 2.).unwrap();
        if resources == 2 {
            solver.graph.set_bounds(vertex, 1, 0., 1.).unwrap();
        }
    }
    for _ in 0..arcs {
        let fields: Vec<_> = lines.next().unwrap().split_whitespace().collect();
        let id = solver
            .graph
            .add_arc(
                fields[0].parse().unwrap(),
                fields[1].parse().unwrap(),
                fields[2].parse().unwrap(),
            )
            .unwrap();
        let arc = solver.graph.arc_mut(id).unwrap();
        arc.reduced_cost = fields[3].parse().unwrap();
        for resource in 0..resources {
            arc.consumption[resource] = fields[4 + resource].parse().unwrap();
        }
    }
    solver.bucket_steps[0] = 1.;
    let mut actual: Vec<Vec<usize>> = solver
        .enumerate(cutoff)
        .unwrap()
        .iter()
        .map(|route| route.arcs.iter().map(|&arc| arc as usize).collect())
        .collect();
    let mut expected: Vec<Vec<usize>> = lines
        .map(|line| {
            line.split_whitespace()
                .map(|arc| arc.parse().unwrap())
                .collect()
        })
        .collect();
    assert_eq!(actual, expected, "captured path export order");
    actual.sort();
    expected.sort();
    assert_eq!(expected.len(), 20);
    assert_eq!(actual, expected);
    if resources == 2 {
        // This route is below the cost cutoff but violates the second resource.
        let excluded = [0, 2, 9];
        assert!(
            excluded
                .iter()
                .map(|&i| solver.graph.arcs[i].reduced_cost)
                .sum::<f64>()
                < cutoff
        );
        assert!(
            excluded
                .iter()
                .map(|&i| solver.graph.arcs[i].consumption[1])
                .sum::<f64>()
                > 1.
        );
        assert!(!actual.iter().any(|path| path == &excluded));
    }
}

#[test]
fn standard_enumeration_matches_captured_reference_paths() {
    // Captured from standard-resource enumeration on a seven-customer model.
    check_reference_enumeration(
        include_str!("../fixtures/enumeration/standard-reference-capture.txt"),
        1,
    );
}

#[test]
fn two_resource_enumeration_matches_captured_reference_paths() {
    // The second resource removes a route that passes the cost cutoff.
    check_reference_enumeration(
        include_str!("../fixtures/enumeration/standard-two-resource-reference-capture.txt"),
        2,
    );
}

#[test]
fn retained_standard_paths_match_captured_repricing_bits() {
    use std::collections::BTreeMap;
    use std::str::FromStr;

    fn take<T: FromStr>(input: &mut std::str::SplitWhitespace<'_>) -> T {
        input.next().unwrap().parse().ok().unwrap()
    }

    let mut input =
        include_str!("../fixtures/enumeration/standard-repricing-reference-capture.txt")
            .split_whitespace();
    let vertices: usize = take(&mut input);
    let arc_count: usize = take(&mut input);
    let cutoff: f64 = take(&mut input);
    let path_count: usize = take(&mut input);
    let cases: usize = take(&mut input);
    let mut solver = Solver::<2>::new(vertices, 1).unwrap();
    solver.graph.edit().vertices[0].elements.clear();
    solver.graph.edit().vertices[vertices - 1].elements.clear();
    for vertex in 0..vertices {
        solver.graph.set_bounds(vertex, 0, 0., 2.).unwrap();
    }
    for _ in 0..arc_count {
        let tail = take(&mut input);
        let head = take(&mut input);
        let cost = take(&mut input);
        let reduced_cost = take(&mut input);
        let consumption = take(&mut input);
        let id = solver.graph.add_arc(tail, head, cost).unwrap();
        let arc = solver.graph.arc_mut(id).unwrap();
        arc.reduced_cost = reduced_cost;
        arc.consumption[0] = consumption;
    }
    let paths: Vec<Vec<i32>> = (0..path_count)
        .map(|_| {
            let len: usize = take(&mut input);
            (0..len).map(|_| take(&mut input)).collect()
        })
        .collect();
    solver.bucket_steps[0] = 1.;
    assert_eq!(solver.enumerate(cutoff).unwrap().len(), path_count);
    for case in 0..cases {
        for arc in &mut solver.graph.edit().arcs {
            arc.reduced_cost = take(&mut input);
        }
        let expected: BTreeMap<_, _> = paths
            .iter()
            .cloned()
            .map(|path| (path, take::<u64>(&mut input)))
            .collect();
        let retained = &mut solver.enumeration.as_mut().unwrap().paths;
        retained
            .inspect(&solver.graph, f64::INFINITY, &[], &[])
            .unwrap();
        let actual: BTreeMap<_, _> = retained
            .routes(&solver.graph)
            .unwrap()
            .into_iter()
            .map(|route| (route.arcs, route.cost.to_bits()))
            .collect();
        assert_eq!(actual, expected, "repricing case {case}");
    }
    assert_eq!((path_count, cases), (20, 3));
}
#[test]
fn fixing_matches_native_prepared_grid_and_selected_labels() {
    use baldes::graph::Graph;
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::labeling::Controls;
    use baldes::pricing::phases::Phase;

    let mut lines = include_str!("../fixtures/cuts/seed8_fixing_prepared_input.txt").lines();
    let header: Vec<_> = lines.next().unwrap().split_whitespace().collect();
    let n: usize = header[0].parse().unwrap();
    let arc_count: usize = header[1].parse().unwrap();
    let cutoff: f64 = header[2].parse().unwrap();
    let mut graph = Graph::<2>::new(n, 1).unwrap();
    for v in [0, n - 1] {
        graph.edit().vertices[v].elements.clear();
        graph.edit().vertices[v].memory.clear();
    }
    for v in 0..n {
        let bounds: Vec<f64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        graph.set_bounds(v, 0, bounds[0], bounds[1]).unwrap();
    }
    for _ in 0..arc_count {
        let arc: Vec<_> = lines.next().unwrap().split_whitespace().collect();
        let id = graph
            .add_arc(
                arc[0].parse().unwrap(),
                arc[1].parse().unwrap(),
                arc[2].parse().unwrap(),
            )
            .unwrap();
        let arc_mut = graph.arc_mut(id).unwrap();
        arc_mut.reduced_cost = arc[3].parse().unwrap();
        arc_mut.consumption[0] = arc[4].parse().unwrap();
    }
    let options = EngineControls {
        steps: [0.25, 0.].into(),
        bidirectional: true,
        bidirectional_mode: 1,
        direction_flags: 3,
        use_symmetric_execution: false,
        ..Default::default()
    };
    let phase = Phase {
        check_ng: true,
        jump_arcs: true,
        labels_per_bucket: i64::MAX,
    };
    let mut engine = LabelingEngine::prepare(&graph, options, phase).unwrap();
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    let mut actual = Vec::new();
    let mut capture = |stage: &str, engine: &LabelingEngine<2>| {
        for (direction, grid) in [
            ("forward", &engine.grid.forward),
            ("backward", &engine.grid.backward),
        ] {
            for (vertex, buckets) in grid.buckets.iter().enumerate() {
                for (bucket, row) in buckets.iter().enumerate() {
                    for arc in &row.arcs {
                        let physical = &graph.arcs[arc.physical];
                        actual.push(format!(
                            "{stage} {direction} {} {bucket} {} {} {}",
                            engine.grid.order[vertex],
                            physical.tail,
                            physical.head,
                            arc.jump.map_or(-1, |v| v as isize)
                        ));
                    }
                }
            }
        }
    };
    capture("before", &engine);
    engine.fix_bucket_arcs(&graph, cutoff).unwrap();
    capture("after", &engine);
    for line in include_str!("../fixtures/cuts/seed8_fixing_selected_labels.txt").lines() {
        let mut f = line.split_whitespace();
        let direction = f.next().unwrap();
        let vertex: usize = f.next().unwrap().parse().unwrap();
        let bucket: usize = f.next().unwrap().parse().unwrap();
        let lower: f64 = f.next().unwrap().parse().unwrap();
        let count: usize = f.next().unwrap().parse().unwrap();
        let storage = if direction == "forward" {
            &mut engine.forward
        } else {
            &mut engine.backward
        };
        let vertex = engine.grid.internal[vertex];
        let old = storage.buckets[vertex][bucket].active.clone();
        let mut chosen = Vec::new();
        for _ in 0..count {
            let cost: f64 = f.next().unwrap().parse().unwrap();
            let resource: f64 = f.next().unwrap().parse().unwrap();
            let ng: u16 = f.next().unwrap().parse().unwrap();
            let id = *old
                .iter()
                .find(|&&id| {
                    let label = &storage.labels[id];
                    (label.cost - cost).abs() < 1e-10
                        && (label.resources[0] - resource).abs() < 1e-10
                        && label.ng == ng
                        && !chosen.contains(&id)
                })
                .expect("captured native label must exist in the independently built Rust store");
            chosen.push(id);
        }
        let hot: Vec<_> = chosen
            .iter()
            .map(|&id| baldes::labeling::storage::HotLabel::of(&storage.labels[id]))
            .collect();
        let row = &mut storage.buckets[vertex][bucket];
        row.active = chosen;
        row.hot = Default::default();
        for label in hot {
            row.hot.push(label);
        }
        row.lower_cost = lower;
    }
    // This second pass deliberately reuses selected native label state; it is
    // a comparison probe, not a complete independent pricing certificate.
    engine.full_ready = true;
    engine.fix_bucket_arcs(&graph, 0.25).unwrap();
    capture("second", &engine);
    actual.sort();
    let mut expected: Vec<_> = include_str!("../fixtures/cuts/seed8_fixing_native_arcs.txt")
        .lines()
        .map(str::to_owned)
        .collect();
    expected.sort();
    assert_eq!(actual, expected);
}

#[test]
fn fixing_completion_matches_native_backward_label_pruning() {
    use baldes::graph::Graph;
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;

    let mut lines = include_str!("../fixtures/cuts/seed8_fixing_prepared_input.txt").lines();
    let header: Vec<_> = lines.next().unwrap().split_whitespace().collect();
    let n: usize = header[0].parse().unwrap();
    let arc_count: usize = header[1].parse().unwrap();
    let mut graph = Graph::<2>::new(n, 1).unwrap();
    for vertex in [0, n - 1] {
        graph.edit().vertices[vertex].elements.clear();
        graph.edit().vertices[vertex].memory.clear();
    }
    for vertex in 0..n {
        let bounds: Vec<f64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        graph.set_bounds(vertex, 0, bounds[0], bounds[1]).unwrap();
    }
    for _ in 0..arc_count {
        let arc: Vec<_> = lines.next().unwrap().split_whitespace().collect();
        let id = graph
            .add_arc(
                arc[0].parse().unwrap(),
                arc[1].parse().unwrap(),
                arc[2].parse().unwrap(),
            )
            .unwrap();
        let arc_mut = graph.arc_mut(id).unwrap();
        arc_mut.reduced_cost = arc[3].parse().unwrap();
        arc_mut.consumption[0] = arc[4].parse().unwrap();
    }
    let engine = LabelingEngine::prepare(
        &graph,
        EngineControls {
            steps: [0.25, 0.].into(),
            bidirectional: true,
            bidirectional_mode: 1,
            direction_flags: 3,
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
    let mut solver = baldes::pricing::Solver::<2>::new(n, 1).unwrap();
    solver.graph = graph;
    solver.engine = Some(engine);
    solver
        .options
        .set("BALDESuseComplBoundsInRedCostFixing", 2.)
        .unwrap();
    solver.fix(header[2].parse().unwrap()).unwrap();
    let engine = solver.engine.as_ref().unwrap();
    assert!(engine.statistics.completion_pruned > 0);
    let quantize = |x: f64| (x * 1e9).round() as i64;
    let mut actual = Vec::new();
    for (direction, storage) in [("forward", &engine.forward), ("backward", &engine.backward)] {
        for (vertex, buckets) in storage.buckets.iter().enumerate() {
            for (bucket, row) in buckets.iter().enumerate() {
                for &id in &row.active {
                    let label = &storage.labels[id];
                    actual.push((
                        direction.to_owned(),
                        engine.grid.order[vertex],
                        bucket,
                        quantize(label.cost),
                        quantize(label.resources[0]),
                        label.ng,
                    ));
                }
            }
        }
    }
    let mut expected = Vec::new();
    for line in include_str!("../fixtures/cuts/seed8_fixing_selected_labels.txt").lines() {
        let mut fields = line.split_whitespace();
        let direction = fields.next().unwrap().to_owned();
        let vertex: usize = fields.next().unwrap().parse().unwrap();
        let bucket: usize = fields.next().unwrap().parse().unwrap();
        let _: f64 = fields.next().unwrap().parse().unwrap();
        let count: usize = fields.next().unwrap().parse().unwrap();
        for _ in 0..count {
            let cost: f64 = fields.next().unwrap().parse().unwrap();
            let resource: f64 = fields.next().unwrap().parse().unwrap();
            let ng = fields.next().unwrap().parse().unwrap();
            expected.push((
                direction.clone(),
                vertex,
                bucket,
                quantize(cost),
                quantize(resource),
                ng,
            ));
        }
    }
    actual.sort();
    expected.sort();
    assert_eq!(expected.len(), 74);
    assert_eq!(actual, expected);

    let mut arcs = Vec::new();
    let graph = &solver.graph;
    let mut capture = |stage: &str, engine: &LabelingEngine<2>| {
        for (direction, grid) in [
            ("forward", &engine.grid.forward),
            ("backward", &engine.grid.backward),
        ] {
            for (vertex, buckets) in grid.buckets.iter().enumerate() {
                for (bucket, row) in buckets.iter().enumerate() {
                    for arc in &row.arcs {
                        let physical = &graph.arcs[arc.physical];
                        arcs.push(format!(
                            "{stage} {direction} {} {bucket} {} {} {}",
                            engine.grid.order[vertex],
                            physical.tail,
                            physical.head,
                            arc.jump.map_or(-1, |v| v as isize)
                        ));
                    }
                }
            }
        }
    };
    let engine = solver.engine.as_mut().unwrap();
    capture("after", engine);
    // The native second pass is debugger-scheduled. Reusing these matched
    // completion-pruned labels likewise requires a diagnostic override.
    engine.full_ready = true;
    engine.fix_bucket_arcs(graph, 0.25).unwrap();
    capture("second", engine);
    arcs.sort();
    let mut expected_arcs: Vec<_> = include_str!("../fixtures/cuts/seed8_fixing_native_arcs.txt")
        .lines()
        .filter(|line| !line.starts_with("before "))
        .map(str::to_owned)
        .collect();
    expected_arcs.sort();
    assert_eq!(arcs, expected_arcs);
}

fn check_captured_enumeration(
    input: &str,
    captured_paths: &str,
    exact: bool,
    align_elements: bool,
    count: usize,
) {
    use baldes::graph::Graph;
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;
    use std::collections::BTreeSet;

    let mut lines = input.lines();
    let header: Vec<_> = lines.next().unwrap().split_whitespace().collect();
    let n: usize = header[0].parse().unwrap();
    let arc_count: usize = header[1].parse().unwrap();
    let cutoff: f64 = header[2].parse().unwrap();
    let mut graph = Graph::<2>::new(n, 1).unwrap();
    for vertex in [0, n - 1] {
        graph.edit().vertices[vertex].elements.clear();
        graph.edit().vertices[vertex].memory.clear();
    }
    if align_elements {
        for vertex in 1..n - 1 {
            graph.edit().vertices[vertex].elements = vec![vertex - 1];
            graph.edit().vertices[vertex].memory = vec![vertex - 1];
        }
    }
    for vertex in 0..n {
        let bounds: Vec<f64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        graph.set_bounds(vertex, 0, bounds[0], bounds[1]).unwrap();
    }
    for _ in 0..arc_count {
        let arc: Vec<_> = lines.next().unwrap().split_whitespace().collect();
        let id = graph
            .add_arc(
                arc[0].parse().unwrap(),
                arc[1].parse().unwrap(),
                arc[2].parse().unwrap(),
            )
            .unwrap();
        let arc_mut = graph.arc_mut(id).unwrap();
        arc_mut.reduced_cost = arc[3].parse().unwrap();
        arc_mut.consumption[0] = arc[4].parse().unwrap();
    }
    let engine = LabelingEngine::prepare(
        &graph,
        EngineControls {
            steps: [0.25, 0.].into(),
            bidirectional: true,
            bidirectional_mode: 1,
            direction_flags: 3,
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
    let mut solver = baldes::pricing::Solver::<2>::new(n, 1).unwrap();
    solver.graph = graph;
    solver.engine = Some(engine);
    solver
        .options
        .set("BALDESuseComplBoundsInRedCostFixing", 2.)
        .unwrap();
    solver.fix(cutoff).unwrap();
    let routes = solver.enumerate(cutoff).unwrap();
    let signature = |arcs: &[i32]| {
        let mut vertices = BTreeSet::new();
        let mut cost = 0.;
        let mut original_cost = 0.;
        for &index in arcs {
            let arc = &solver.graph.arcs[index as usize];
            if arc.head != solver.graph.sink {
                vertices.insert(arc.head);
            }
            cost += arc.reduced_cost;
            original_cost += arc.cost;
        }
        (
            vertices,
            (cost * 1e9).round() as i64,
            (original_cost * 1e9).round() as i64,
        )
    };
    let mut actual: Vec<_> = routes
        .iter()
        .map(|route| {
            let result = signature(&route.arcs);
            assert_eq!((route.cost * 1e9).round() as i64, result.1);
            assert_eq!((route.original_cost * 1e9).round() as i64, result.2);
            result
        })
        .collect();
    let mut expected: Vec<_> = captured_paths
        .lines()
        .map(|line| {
            let arcs: Vec<i32> = line
                .split_whitespace()
                .map(|id| id.parse().unwrap())
                .collect();
            signature(&arcs)
        })
        .collect();
    assert_eq!(expected.len(), count);
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected);
    if exact {
        let mut actual_paths: Vec<_> = routes.iter().map(|route| route.arcs.clone()).collect();
        let mut native_paths: Vec<Vec<i32>> = captured_paths
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|id| id.parse().unwrap())
                    .collect()
            })
            .collect();
        if !align_elements {
            actual_paths.sort();
            native_paths.sort();
        }
        assert_eq!(actual_paths, native_paths);
    }
}

#[test]
fn fixing_enumeration_matches_native_visited_sets_and_reduced_costs() {
    check_captured_enumeration(
        include_str!("../fixtures/cuts/seed8_fixing_prepared_input.txt"),
        include_str!("../fixtures/cuts/seed8_enumeration_native_paths.txt"),
        false,
        false,
        44,
    );
}

#[test]
fn fixing_enumeration_matches_captured_paths_with_default_element_ids() {
    check_captured_enumeration(
        include_str!("../fixtures/cuts/seed8_enumeration_external_input.txt"),
        include_str!("../fixtures/cuts/seed8_enumeration_external_native_paths.txt"),
        true,
        false,
        44,
    );
}

#[test]
fn fixing_enumeration_matches_native_ordered_paths_with_aligned_element_ids() {
    check_captured_enumeration(
        include_str!("../fixtures/cuts/seed8_enumeration_external_input.txt"),
        include_str!("../fixtures/cuts/seed8_enumeration_external_native_paths.txt"),
        true,
        true,
        44,
    );
}

#[test]
fn fixing_enumeration_matches_second_native_ordered_capture() {
    check_captured_enumeration(
        include_str!("../fixtures/cuts/seed2_enumeration_external_input.txt"),
        include_str!("../fixtures/cuts/seed2_enumeration_external_native_paths.txt"),
        true,
        true,
        55,
    );
}
