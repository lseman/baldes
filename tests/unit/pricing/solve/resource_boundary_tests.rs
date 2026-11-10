use super::*;

#[test]
fn full_kpath_repricing_agrees_before_and_after_retained_inspection() {
    let mut solver = Solver::<2>::new(4, 1).unwrap();
    solver.graph.edit().elementary = true;
    solver.enumeration_preserve_arc_paths = true;
    for (tail, head, cost) in [(0, 1, -4.), (1, 3, 0.), (0, 2, -2.), (2, 3, 0.)] {
        solver.graph.add_arc(tail, head, cost).unwrap();
    }
    let cut = solver.cuts.add_strong_k_path_cut(4, &[1], 1, -3.).unwrap();
    let labeled = solver.price_phase(0, 10, false).unwrap();
    assert_eq!(labeled.minimum_reduced_cost, -2.);
    assert_eq!(solver.applied_cuts.3.len(), 1);
    solver.enumerate(100.).unwrap();
    let retained = solver.price_phase(0, 10, false).unwrap();
    assert_eq!(retained.minimum_reduced_cost, labeled.minimum_reduced_cost);
    solver.cuts.cuts[cut].dual = -1.;
    let heuristic = solver.price_phase(1, 10, false).unwrap();
    assert!(!heuristic.exact && heuristic.columns.is_empty());
    let repriced = solver.price_phase(0, 10, false).unwrap();
    assert!(repriced.exact);
    assert_eq!(repriced.minimum_reduced_cost, -3.);
}

#[test]
fn heuristic_pricing_respects_requested_column_cap_across_join_groups() {
    for phase in [1, 2] {
        for maximum in [1, 2, 3] {
            let mut solver = Solver::<2>::new(6, 1).unwrap();
            solver.graph.edit().elementary = true;
            for vertex in 0..6 {
                solver.graph.set_bounds(vertex, 0, 0., 10.).unwrap();
            }
            // Each route crosses the bidirectional resource split at a
            // different vertex, producing several representative groups.
            for vertex in 1..5 {
                let outward = solver.graph.add_arc(0, vertex, -(vertex as f64)).unwrap();
                solver.graph.arc_mut(outward).unwrap().consumption[0] = 6.;
                let inward = solver.graph.add_arc(vertex, 5, 0.).unwrap();
                solver.graph.arc_mut(inward).unwrap().consumption[0] = 2.;
            }
            let answer = solver.price_phase(phase, maximum, false).unwrap();
            let costs: Vec<_> = answer.columns.iter().map(|route| route.cost).collect();
            let expected: Vec<_> = (0..maximum).map(|index| -4. + index as f64).collect();
            assert_eq!(costs, expected, "phase={phase}, maximum={maximum}");
            assert_eq!(answer.minimum_reduced_cost, -4.);
            assert!(!answer.exact);
        }
    }
}

fn retained_costs(costs: &[f64]) -> Solver<2> {
    let mut solver = Solver::<2>::new(costs.len() + 2, 1).unwrap();
    solver.graph.edit().elementary = true;
    solver.enumeration_preserve_arc_paths = true;
    let sink = solver.graph.sink;
    for (index, &cost) in costs.iter().enumerate() {
        solver.graph.add_arc(0, index + 1, cost).unwrap();
        solver.graph.add_arc(index + 1, sink, 0.).unwrap();
    }
    solver.price_phase(0, 10, false).unwrap();
    solver.enumerate(100.).unwrap();
    solver
}

#[test]
fn retained_pricing_filters_at_the_labeling_threshold_without_changing_the_minimum() {
    let mut solver = retained_costs(&[-2., -crate::labeling::EPSILON, 0., 2.]);
    let answer = solver.price_phase(0, 10, true).unwrap();
    assert!(answer.exact);
    assert_eq!(answer.minimum_reduced_cost, -2.);
    assert_eq!(answer.columns.len(), 1);
    assert_eq!(answer.columns[0].cost, -2.);

    // Filtering is output-only, including a nonzero caller threshold.
    solver.controls.zero_threshold = 1.;
    let answer = solver.price_phase(0, 10, true).unwrap();
    assert_eq!(answer.columns.len(), 3);
    assert_eq!(answer.minimum_reduced_cost, -2.);
    let answer = solver.price_phase(0, 10, false).unwrap();
    assert_eq!(answer.columns.len(), 4);
    assert_eq!(answer.minimum_reduced_cost, -2.);
}

#[test]
fn retained_minimum_survives_filtered_and_zero_column_outputs() {
    let mut solver = retained_costs(&[-crate::labeling::EPSILON, 0., 2.]);
    let answer = solver.price_phase(0, 10, true).unwrap();
    assert!(answer.exact && answer.columns.is_empty());
    assert_eq!(answer.minimum_reduced_cost, -crate::labeling::EPSILON);

    let answer = solver.price_phase(0, 0, false).unwrap();
    assert!(answer.exact && answer.columns.is_empty());
    assert_eq!(answer.minimum_reduced_cost, -crate::labeling::EPSILON);
    solver.controls.max_columns = 0;
    let answer = solver.price_phase(0, 10, false).unwrap();
    assert!(answer.columns.is_empty());
    assert_eq!(answer.minimum_reduced_cost, -crate::labeling::EPSILON);
}

#[test]
fn pricing_forks_share_topology_snapshots_and_isolate_working_graphs_and_paths() {
    let mut parent = retained_costs(&[-2., 1.]);
    let mut child = parent.fork_for_node();
    assert!(Arc::ptr_eq(
        parent.prepared.as_ref().unwrap(),
        child.prepared.as_ref().unwrap()
    ));
    assert!(Arc::ptr_eq(
        parent.enumeration_snapshot.as_ref().unwrap(),
        child.enumeration_snapshot.as_ref().unwrap()
    ));
    child.graph.arc_mut(0).unwrap().reduced_cost = 3.;
    child.inspect_enumeration(2.).unwrap();
    assert_eq!(child.enumeration.as_ref().unwrap().paths.len, 1);
    assert_eq!(parent.enumeration.as_ref().unwrap().paths.len, 2);
    assert_eq!(parent.graph.arcs[0].reduced_cost, -2.);
    assert_eq!(
        parent
            .price_phase(0, 10, false)
            .unwrap()
            .minimum_reduced_cost,
        -2.
    );
}

fn boundary(alternative: bool) -> Solver<2> {
    let mut s = Solver::<2>::new(if alternative { 4 } else { 3 }, 1).unwrap();
    let upper = f64::from_bits(1f64.to_bits() - 1);
    s.graph.edit().initial[0] = Some(0.);
    for v in 0..s.graph.vertices.len() {
        s.graph.set_bounds(v, 0, 0., upper).unwrap();
    }
    let sink = s.graph.sink;
    let mut arcs = vec![(0, 1, 0.7, -2.), (1, sink, 0.3, -2.)];
    if alternative {
        arcs.extend([(0, 2, 0.2, -1.), (2, sink, 0.2, -1.)]);
    }
    for (u, v, consumption, cost) in arcs {
        let id = s.graph.add_arc(u, v, cost).unwrap();
        s.graph.arc_mut(id).unwrap().consumption[0] = consumption;
    }
    s
}
#[test]
fn reverse_only_boundary_path_retries_and_graph_changes_restore_normal_search() {
    let mut s = boundary(false);
    let result = s.price(1, false).unwrap();
    assert!(result.exact && result.columns.is_empty());
    assert!(!s.engine.as_ref().unwrap().controls.bidirectional);
    for v in 0..s.graph.vertices.len() {
        s.graph.set_bounds(v, 0, 0., 1.).unwrap();
    }
    let result = s.price(1, false).unwrap();
    assert!(result.exact);
    assert_eq!(result.columns[0].arcs, [0, 1]);
    assert!(s.engine.as_ref().unwrap().controls.bidirectional);
}
#[test]
fn retry_recovers_displaced_valid_paths_and_preserves_physical_exclusions() {
    let mut s = boundary(true);
    let result = s.price(1, false).unwrap();
    assert!(result.exact);
    assert_eq!(result.columns.len(), 1);
    assert_eq!(result.columns[0].arcs, [2, 3]);
    assert_eq!(result.minimum_reduced_cost, -2.);
    assert!(!s.engine.as_ref().unwrap().controls.bidirectional);
    s.forbidden_arcs = vec![false, false, true, false];
    let result = s.price(1, false).unwrap();
    assert!(result.exact && result.columns.is_empty());
    s.controls.time_limit = std::time::Duration::ZERO;
    let error = s.price(1, false).unwrap_err();
    assert_eq!(error.kind, crate::core::error::ErrorKind::Runtime);
    assert!(error.message.contains("time limit"));
    let engine = s.engine.as_ref().unwrap();
    assert!(!engine.half_ready && !engine.full_ready);
}

#[test]
fn disabled_retained_phase_leaves_cached_duals_and_working_state_untouched() {
    let mut solver = Solver::<2>::new(4, 1).unwrap();
    solver.enumeration_preserve_arc_paths = true;
    for (tail, head, cost) in [(0, 1, -4.), (1, 3, 0.), (0, 2, -2.), (2, 3, 0.)] {
        solver.graph.add_arc(tail, head, cost).unwrap();
    }
    let cut = solver.cuts.add_strong_k_path_cut(4, &[1], 1, -3.).unwrap();
    solver.price_phase(0, 10, false).unwrap();
    solver.enumerate(100.).unwrap();
    solver.cuts.cuts[cut].dual = -1.;
    let before = format!("{solver:?}");
    assert!(solver.price_phase(usize::MAX, 10, false).is_err());
    assert_eq!(format!("{solver:?}"), before);
    for phase in [1, 2] {
        let result = solver.price_phase(phase, 10, false).unwrap();
        assert!(!result.exact && result.columns.is_empty());
        assert_eq!(result.minimum_reduced_cost, 1e12);
        assert_eq!(format!("{solver:?}"), before);
    }
    let refreshed = solver.price_phase(0, 10, false).unwrap();
    assert!(refreshed.exact);
    assert_eq!(refreshed.minimum_reduced_cost, -3.);
}

#[test]
fn retained_skip_defers_resource_invalidation_but_partial_stores_cannot_skip() {
    use crate::graph::coupled::CoupledResource;
    let mut solver = retained_costs(&[-2., -1.]);
    solver
        .graph
        .set_coupled_resources(&[CoupledResource::Selection {
            initial: 1,
            allowed: vec![0, 1, 1, 1],
            vertices: vec![],
            relaxed: false,
        }])
        .unwrap();
    let before = format!("{solver:?}");
    let skipped = solver.price_phase(1, 10, false).unwrap();
    assert!(!skipped.exact && skipped.columns.is_empty());
    assert_eq!(format!("{solver:?}"), before);
    let active = solver.price_phase(0, 10, false).unwrap();
    assert!(active.exact);
    assert_eq!(active.minimum_reduced_cost, -1.);
    assert!(solver.enumeration.is_none());
    solver.enumerate(100.).unwrap();
    solver.enumeration.as_mut().unwrap().completed = false;
    let rebuilt = solver.price_phase(1, 10, false).unwrap();
    assert!(!rebuilt.columns.is_empty());
    assert!(solver.enumeration.is_none());
}

#[test]
fn raw_functor_retained_skip_does_not_require_or_refresh_objective_cache() {
    let mut solver = retained_costs(&[-2., -1.]);
    solver.restore_arcs();
    solver.graph.edit().arcs[0].variables = vec![(0, 1.)];
    solver.price_phase(0, 10, false).unwrap();
    solver.enumerate(100.).unwrap();
    assert!(solver.variable_costs.is_empty() && solver.raw_pricing.is_none());
    let before = format!("{solver:?}");
    assert!(solver.price_functor_phase(usize::MAX, 10).is_err());
    assert_eq!(format!("{solver:?}"), before);
    for phase in [1, 2] {
        let skipped = solver.price_functor_phase(phase, 10).unwrap();
        assert!(!skipped.exact && skipped.columns.is_empty());
        assert_eq!(skipped.minimum_reduced_cost, 1e12);
        assert_eq!(format!("{solver:?}"), before);
    }
    solver.cache_variable_costs(&[-0.25]).unwrap();
    let before = format!("{solver:?}");
    assert!(!solver.price_functor_phase(1, 10).unwrap().exact);
    assert_eq!(format!("{solver:?}"), before);
    let active = solver.price_functor_phase(0, 10).unwrap();
    assert!(active.exact);
    assert_eq!(active.minimum_reduced_cost, -2.25);
}

#[test]
fn bellman_threshold_stages_include_route_offsets_and_do_not_claim_full_pricing() {
    use crate::extensions::stochastic_restocking::{Demand, Problem, RestockingResource};
    use crate::graph::coupled::{CoupledResource, PluginResource};
    let p = Problem::new(
        4,
        vec![Demand::new(vec![(0, 1.)]).unwrap()],
        vec![vec![0., 10.], vec![10., 0.]],
    )
    .unwrap();
    let mut solver = Solver::<2>::new(3, 0).unwrap();
    solver.graph.add_arc(0, 1, 10.).unwrap();
    solver.graph.add_arc(1, 2, 10.).unwrap();
    solver.graph.set_reduced_costs(&[-15., -15.]).unwrap();
    solver
        .set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(
            RestockingResource::new(p, vec![(0, 1), (1, 0)]).unwrap(),
        ))])
        .unwrap();
    solver.exact_target = Some(-20.);
    let controls = Controls {
        zero_threshold: -25.,
        ..Default::default()
    };
    let mut engine = solver
        .new_engine(solver.engine_controls(true), solver.phases[0])
        .unwrap();
    let result = solver
        .bellman_stages(
            &mut engine,
            &controls,
            std::time::Instant::now(),
            Duration::from_secs(2),
        )
        .unwrap();
    assert!(!result.exact, "the full threshold was never priced");
    assert_eq!(result.columns.len(), 1);
    assert_eq!(result.columns[0].cost, -30.);
    assert_eq!(solver.staged_minimum, Some(-30.));
    assert_eq!(solver.exact_target, Some(-1.25));
    let fork = solver.fork_pricing(false);
    assert!(
        fork.staged_minimum.is_none(),
        "a node/cost fork cannot inherit a point-dependent proof"
    );
    solver.cache_variable_costs(&[]).unwrap();
    assert!(
        solver.staged_minimum.is_none(),
        "a changed dual point invalidates stage minima"
    );
}

#[test]
fn heuristic_requested_bellman_fallback_uses_raw_costs_for_completed_minima() {
    use crate::extensions::stochastic_restocking::{Demand, Problem, RestockingResource};
    use crate::graph::coupled::{CoupledResource, PluginResource};
    let p = Problem::new(
        4,
        vec![Demand::new(vec![(0, 1.)]).unwrap()],
        vec![vec![0., 1.], vec![1., 0.]],
    )
    .unwrap();
    let mut solver = Solver::<2>::new(3, 0).unwrap();
    for (id, (from, to)) in [(0, 1), (1, 2)].into_iter().enumerate() {
        let arc = solver.graph.add_arc(from, to, 0.).unwrap();
        solver.graph.arc_mut(arc).unwrap().variables = vec![(id, 1.)];
    }
    solver
        .set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(
            RestockingResource::new(p, vec![(0, 1), (1, 0)]).unwrap(),
        ))])
        .unwrap();
    solver
        .set_option("BALDESuseRoundedHeuristicPricing", 1.)
        .unwrap();
    solver.cache_variable_costs(&[-4e-9, 0.]).unwrap();
    assert_eq!(solver.graph.arcs[0].reduced_cost, 0.);
    let priced = solver.price_functor_phase(2, 20).unwrap();
    assert!(
        priced.exact,
        "the Bellman fallback completes even in a heuristic requested phase"
    );
    assert!((priced.minimum_reduced_cost + 4e-9).abs() < 1e-20);
}

#[test]
fn scalar_ng_proof_refines_cycles_without_exporting_walks_or_interrupted_minima() {
    use crate::extensions::stochastic_restocking::{Demand, Problem, RestockingResource};
    use crate::graph::coupled::{CoupledResource, PluginResource};
    let problem = Problem::new(
        4,
        vec![Demand::new(vec![(0, 1.)]).unwrap(); 4],
        vec![vec![0.; 5]; 5],
    )
    .unwrap();
    let mut solver = Solver::<2>::new(6, 1).unwrap();
    let mut endpoints = Vec::new();
    for (from, to, reduced) in [
        (0, 1, 5.),
        (1, 2, -3.),
        (2, 3, -3.),
        (3, 1, -3.),
        (1, 5, 0.),
        (2, 5, 10.),
        (3, 5, 10.),
    ] {
        let id = solver.graph.add_arc(from, to, 0.).unwrap();
        solver.graph.arc_mut(id).unwrap().reduced_cost = reduced;
        solver.graph.arc_mut(id).unwrap().consumption[0] = 1.;
        endpoints.push((from, if to == 5 { 0 } else { to }));
    }
    for vertex in 0..6 {
        solver.graph.set_bounds(vertex, 0, 0., 5.).unwrap();
    }
    solver
        .set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(
            RestockingResource::new(problem, endpoints).unwrap(),
        ))])
        .unwrap();
    // The unrefined walk 0,1,2,3,1,5 costs -4; every elementary route costs
    // at least 5. A complete-first sweep must remove that cycle and retry.
    assert!(solver.scalar_preprice(20, Duration::from_secs(2)).is_none());
    let bound = solver.staged_minimum.unwrap();
    assert!((0. ..=5.).contains(&bound), "refined bound {bound}");
    assert!(solver.graph.vertices[2].memory == vec![2]);
    solver.staged_minimum = None;
    assert!(solver.scalar_preprice(20, Duration::ZERO).is_none());
    assert!(solver.staged_minimum.is_none());
}
