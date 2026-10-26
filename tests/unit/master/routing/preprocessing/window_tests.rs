use super::*;
use crate::graph::Window;
use crate::master::branching::{BranchConstraint, WindowGroup};
use crate::master::RowSense;

/// Depot 0, customers 1 and 2, sink 3; customer 1 has two windows.
fn split_problem() -> (RoutingContext<2>, crate::graph::WindowArcs) {
    let mut g = crate::Graph::<2>::new(4, 1).unwrap();
    for v in 0..4 {
        g.set_bounds(v, 0, 0., 30.).unwrap();
    }
    for (u, v) in [(0, 1), (0, 2), (1, 2), (2, 1), (1, 3), (2, 3)] {
        let id = g.add_arc(u, v, 1.).unwrap();
        g.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    let windows = [Window::new(0., 2.).unwrap(), Window::new(8., 12.).unwrap()];
    g.set_time_windows(1, 0, &windows).unwrap();
    let split = g.split_time_window_arcs(0).unwrap();
    (RoutingContext::set_partitioning(g, 2.).unwrap(), split)
}

fn selection(
    problem: &RoutingContext<2>,
    ids: &[i32],
    sense: RowSense,
    rhs: f64,
) -> BranchConstraint {
    let mut arcs: Vec<_> = ids
        .iter()
        .map(|&id| (problem.arc_position(id).unwrap(), 1.))
        .collect();
    arcs.sort_by_key(|&(a, _)| a);
    BranchConstraint::WindowFlow {
        group: WindowGroup::Selection {
            vertex: 1,
            window: 1,
        },
        arcs,
        sense,
        rhs,
    }
}

#[test]
fn selection_rows_alone_do_not_imply_the_other_windows_are_unused() {
    // Cover = 1 and "window 1 selected at least once" together imply that
    // window 0 is unused, but arc-bound propagation does not derive it; the
    // up child therefore carries the "other windows <= 0" row explicitly.
    let (mut problem, split) = split_problem();
    let selections = split.selections();
    problem
        .branch(selection(
            &problem,
            &selections[&(1, 1)],
            RowSense::GreaterEqual,
            1.,
        ))
        .unwrap();
    let (_, bounds) = problem.preprocess(&[]).unwrap();
    assert!(selections[&(1, 0)]
        .iter()
        .any(|id| bounds[problem.arc_position(*id).unwrap()].upper > 0.));
}

#[test]
fn forcing_a_window_removes_the_copies_of_the_others() {
    let (mut problem, split) = split_problem();
    let selections = split.selections();
    problem
        .branch(selection(
            &problem,
            &selections[&(1, 0)],
            RowSense::LessEqual,
            0.,
        ))
        .unwrap();
    let (_, bounds) = problem.preprocess(&[]).unwrap();
    for id in &selections[&(1, 0)] {
        assert_eq!(bounds[problem.arc_position(*id).unwrap()].upper, 0., "{id}");
    }
    for id in &selections[&(1, 1)] {
        assert!(
            bounds[problem.arc_position(*id).unwrap()].upper > 0.,
            "{id}"
        );
    }
}

#[test]
fn forbidding_a_window_removes_its_copies() {
    let (mut problem, split) = split_problem();
    let chosen = &split.selections()[&(1, 1)];
    problem
        .branch(selection(&problem, chosen, RowSense::LessEqual, 0.))
        .unwrap();
    let (_, bounds) = problem.preprocess(&[]).unwrap();
    for id in chosen {
        assert_eq!(bounds[problem.arc_position(*id).unwrap()].upper, 0., "{id}");
    }
}
