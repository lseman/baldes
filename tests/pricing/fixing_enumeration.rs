use baldes::enumeration::{Controls, Enumeration};
use baldes::inputs::{Instance, Rounding};
use baldes::pricing::Solver;


#[test]
fn raw_functor_pricing_restores_discarded_paths_and_keeps_branch_restrictions() {
    let mut solver = baldes::pricing::Solver::<2>::new(3, 1).unwrap();
    for v in 0..3 {
        solver.graph.set_bounds(v, 0, 0., 10.).unwrap();
    }
    solver.graph.elementary = true;
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
    for arc in &mut child.graph.arcs {
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
    for arc in &mut solver.graph.arcs {
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
    solver.graph.vertices[0].elements.clear();
    solver.graph.vertices[4].elements.clear();
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
