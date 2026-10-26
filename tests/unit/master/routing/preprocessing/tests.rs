use super::*;
use crate::pricing::restrictions::RyanFosterConstraint;

fn complete_problem() -> RoutingContext<2> {
    let mut graph = Graph::<2>::new(6, 0).unwrap();
    graph.elementary = true;
    for tail in 0..5 {
        for head in 1..6 {
            if tail != head {
                graph.add_arc(tail, head, 1.).unwrap();
            }
        }
    }
    RoutingContext::set_partitioning(graph, 4.).unwrap()
}

fn paths(
    graph: &Graph<2>,
    vertex: usize,
    seen: u64,
    path: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if vertex == graph.sink {
        out.push(path.clone());
        return;
    }
    for (a, arc) in graph.arcs.iter().enumerate() {
        if arc.tail == vertex && seen & (1 << arc.head) == 0 {
            path.push(a);
            paths(graph, arc.head, seen | (1 << arc.head), path, out);
            path.pop();
        }
    }
}

#[test]
fn ryan_foster_arc_deductions_preserve_every_admissible_elementary_path() {
    let base = complete_problem();
    let mut universe = vec![];
    paths(
        &base.model.networks[0].graph,
        base.model.networks[0].graph.source,
        1,
        &mut vec![],
        &mut universe,
    );
    let pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)];
    let mut total_fixed = 0;
    for mut code in 0..3usize.pow(pairs.len() as u32) {
        let mut problem = base.clone();
        let mut rules = vec![];
        for (first, second) in pairs {
            let state = code % 3;
            code /= 3;
            if state > 0 {
                let rule = RyanFosterConstraint {
                    first,
                    second,
                    together: state == 1,
                };
                problem.branch(BranchConstraint::RyanFoster(rule)).unwrap();
                rules.push(rule);
            }
        }
        let mut bounds = vec![
            Bounds {
                lower: 0.,
                upper: f64::INFINITY
            };
            problem.model.networks[0].graph.arcs.len()
        ];
        total_fixed += problem.propagate_ryan_foster(&mut bounds);
        for path in &universe {
            let nodes: Vec<_> = path
                .iter()
                .map(|&a| problem.model.networks[0].graph.arcs[a].head)
                .collect();
            if rules
                .iter()
                .all(|rule| rule.satisfied_by(nodes.iter().copied()))
            {
                assert!(
                    path.iter().all(|&a| bounds[a].upper > 0.),
                    "rules={rules:?}, path={path:?}"
                );
            }
        }
    }
    assert!(total_fixed > 0);
}

#[test]
fn apart_rule_excludes_arcs_between_transitive_together_classes() {
    let mut problem = complete_problem();
    for (first, second, together) in [(1, 2, true), (2, 3, false)] {
        problem
            .branch(BranchConstraint::RyanFoster(RyanFosterConstraint {
                first,
                second,
                together,
            }))
            .unwrap();
    }
    let (report, bounds) = problem.preprocess(&[]).unwrap();
    assert!(!report.infeasible);
    for (a, arc) in problem.model.networks[0].graph.arcs.iter().enumerate() {
        if (arc.tail == 1 && arc.head == 3) || (arc.tail == 3 && arc.head == 1) {
            assert_eq!(bounds[a].upper, 0.);
        }
    }
}

#[test]
fn contradictory_optional_component_is_excluded_without_false_infeasibility() {
    let mut problem = complete_problem();
    for vertex in [1, 2] {
        problem.model.rows[vertex].rhs = 0.;
    }
    for together in [true, false] {
        problem
            .branch(BranchConstraint::RyanFoster(RyanFosterConstraint {
                first: 1,
                second: 2,
                together,
            }))
            .unwrap();
    }
    let (report, bounds) = problem.preprocess(&[]).unwrap();
    assert!(!report.infeasible && report.complete);
    for (arc, bound) in problem.model.networks[0].graph.arcs.iter().zip(bounds) {
        if [1, 2].contains(&arc.tail) || [1, 2].contains(&arc.head) {
            assert_eq!(bound.upper, 0.);
        }
    }
}

#[test]
fn inherited_cut_rows_follow_physical_remapping_and_retirement() {
    use crate::inputs::{Instance, Rounding};
    let instance = Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let problem =
        RoutingContext::set_partitioning(instance.graph(3, Rounding::None).unwrap(), 3.).unwrap();
    let mut options =
        RootOptions::default().with_capacity_cuts(CapacityCutConfig::new(vec![0., 1., 1., 1.], 2.));
    options.capture_state = true;
    options.enumerate = false;
    let parent = super::super::solve(problem, options).unwrap();
    let snapshot = parent.snapshot.unwrap();
    let graph = &snapshot.problem.model.networks[0].graph;
    let mut master = snapshot.master.try_clone().unwrap();
    let mut state = snapshot.cut.clone();
    let before = state.preprocessing_rows(graph, &master).unwrap();
    assert_eq!(before.len(), 1);
    let old = state.capacity_rows[0].unwrap();
    // A new physical index must use its current RHS and sense, while
    // retaining the cut's arc coefficients after remapping.
    let new = master.add_row(RowSense::GreaterEqual, 17., &[]).unwrap();
    state.capacity_rows[0] = Some(new);
    for entries in &mut state.capacity_arc_entries {
        for (row, _) in entries {
            if *row == old {
                *row = new;
            }
        }
    }
    let remapped = state.preprocessing_rows(graph, &master).unwrap();
    assert_eq!(remapped[0].sense, RowSense::GreaterEqual);
    assert_eq!(remapped[0].rhs, 17.);
    assert_eq!(remapped[0].entries, before[0].entries);
    state.capacity_rows[0] = None;
    assert!(state.preprocessing_rows(graph, &master).unwrap().is_empty());
    // The cloned changes above must not alter the parent's active pool.
    let unchanged = snapshot
        .cut
        .preprocessing_rows(graph, &snapshot.master)
        .unwrap();
    assert_eq!(unchanged[0].rhs, before[0].rhs);
    assert_eq!(unchanged[0].entries, before[0].entries);
}
