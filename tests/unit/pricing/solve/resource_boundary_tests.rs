use super::*;
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
