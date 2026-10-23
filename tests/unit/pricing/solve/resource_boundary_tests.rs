use super::*;

#[test]
fn full_kpath_repricing_agrees_before_and_after_retained_inspection() {
    let mut solver = Solver::<2>::new(4, 1).unwrap();
    solver.graph.elementary = true;
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
            solver.graph.elementary = true;
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
    solver.graph.elementary = true;
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
    s.graph.initial[0] = Some(0.);
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
