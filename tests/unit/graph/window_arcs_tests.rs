use super::*;
use crate::{price, Controls};

fn windows(pairs: &[(f64, f64)]) -> Vec<Window> {
    pairs
        .iter()
        .map(|&(a, b)| Window::new(a, b).unwrap())
        .collect()
}

/// 0 -> 1 -> 2 -> 3. Vertex 1 has windows [0, 2] and [8, 12]; vertex 2 has
/// [0, 12] and [15, 20], so arriving at 2 at time 11 can start at 11 or 15.
fn chain() -> Graph<2> {
    let mut g = Graph::new(4, 1).unwrap();
    for v in 0..4 {
        g.set_bounds(v, 0, 0., 30.).unwrap();
    }
    for (u, v, c, t) in [(0, 1, -3., 4.), (1, 2, -4., 3.), (2, 3, -2., 2.)] {
        let id = g.add_arc(u, v, c).unwrap();
        g.arc_mut(id).unwrap().consumption[0] = t;
    }
    g.set_time_windows(1, 0, &windows(&[(0., 2.), (8., 12.)]))
        .unwrap();
    g.set_time_windows(2, 0, &windows(&[(0., 12.), (15., 20.)]))
        .unwrap();
    g
}

#[test]
fn split_copies_each_window_once_and_maps_copies_to_input_arcs() {
    let mut g = chain();
    let split = g.split_time_window_arcs(0).unwrap();
    assert_eq!(g.arcs.len(), 5);
    assert_eq!(split.arcs.len(), 4);
    let selections = split.selections();
    assert_eq!(
        selections.keys().copied().collect::<Vec<_>>(),
        [(1, 0), (1, 1), (2, 0), (2, 1)]
    );
    // Originals keep their IDs on the first window; copies get fresh IDs.
    assert_eq!(selections[&(1, 0)], [0]);
    assert_eq!(selections[&(2, 0)], [1]);
    let copy = selections[&(2, 1)][0];
    assert_eq!(split.original(copy), 1);
    let arc = g.arcs.iter().find(|a| a.id == copy).unwrap();
    assert_eq!(arc.bounds[0], Some(Window::new(15., 20.).unwrap()));
    // The arc into the single-window sink is untouched.
    assert_eq!(split.original(2), 2);
    assert_eq!(g.arcs[2].bounds[0], None);
    // A second split finds one window per arc and adds nothing.
    let again = g.split_time_window_arcs(0).unwrap();
    assert_eq!(g.arcs.len(), 5);
    assert_eq!(again.arcs.len(), 4);
}

#[test]
fn split_skips_windows_outside_the_vertex_and_arc_bounds() {
    let mut g = chain();
    g.set_bounds(2, 0, 0., 14.).unwrap();
    let split = g.split_time_window_arcs(0).unwrap();
    assert!(!split.selections().contains_key(&(2, 1)));
    assert_eq!(g.arcs.len(), 4);
}

#[test]
fn split_pricing_matches_and_reaches_later_windows() {
    let original = chain();
    let mut g = chain();
    let split = g.split_time_window_arcs(0).unwrap();
    let base = price(&original, &Controls::default()).unwrap();
    let result = price(&g, &Controls::default()).unwrap();
    assert!(result.exact);
    assert_eq!(result.minimum_reduced_cost, base.minimum_reduced_cost);
    let mut route = result.columns[0].clone();
    split.restore(&mut route);
    assert_eq!(route.arcs, base.columns[0].arcs);
    // Earliest service at 2 is 11; a cheaper copy for [15, 20] makes waiting
    // optimal, which pricing on the unsplit graph cannot represent.
    let later = split.selections()[&(2, 1)][0];
    g.arcs.iter_mut().find(|a| a.id == later).unwrap().reduced_cost -= 1.;
    let result = price(&g, &Controls::default()).unwrap();
    assert_eq!(result.minimum_reduced_cost, base.minimum_reduced_cost - 1.);
    assert!(result.columns[0].arcs.contains(&later));
    assert_eq!(result.columns[0].resource_profile[2], vec![15.]);
}

#[test]
fn input_routes_replay_onto_the_copy_of_their_window() {
    let mut g = chain();
    let split = g.split_time_window_arcs(0).unwrap();
    let mapped = g.route_on_window_arcs(&split, &[0, 1, 2]).unwrap();
    // Service at 1 starts at 8 (second window) and at 2 at 11 (first window).
    assert_eq!(mapped[0], split.selections()[&(1, 1)][0]);
    assert_eq!(mapped[1], 1);
    assert_eq!(mapped[2], 2);
    assert!(g.route_on_window_arcs(&split, &[0, 99]).is_none());
}

#[test]
fn split_requires_a_disposable_resource() {
    let mut g = chain();
    assert!(g.split_time_window_arcs(1).is_err());
}
