use super::*;
use crate::master::tree::candidates::resource_groups;

fn fixture() -> (RootProblem<5>, Vec<Route>) {
    let mut graph = crate::Graph::<5>::new(5, 5).unwrap();
    for vertex in 0..5 {
        for axis in 0..5 {
            graph.set_bounds(vertex, axis, 0., 3.).unwrap();
        }
    }
    let mut paths = vec![];
    for (vertex, quantity) in [(1, 0.5), (2, 1.5), (3, 2.5)] {
        graph.set_bounds(vertex, 4, 0., quantity).unwrap();
        let arc = graph.add_arc(0, vertex, 1.).unwrap();
        graph.arc_mut(arc).unwrap().consumption[4] = quantity;
        let end = graph.add_arc(vertex, 4, 0.).unwrap();
        paths.push(vec![arc, end]);
    }
    let mut problem = RootProblem::set_partitioning(graph, 1.).unwrap();
    for path in &paths {
        let position = problem.arc_position(path[0]).unwrap();
        problem.arc_entries[position] = vec![(1, 1.)];
    }
    problem.rows[2].sense = RowSense::LessEqual;
    problem.rows[3].sense = RowSense::LessEqual;
    let routes = paths
        .iter()
        .map(|p| problem.elementary_route(p).unwrap())
        .collect();
    (problem, routes)
}

#[test]
fn history_reuses_old_threshold_when_fresh_separation_selects_another() {
    let (problem, routes) = fixture();
    let initial = [(&routes[0], 0.75), (&routes[1], 0.25)];
    let old = resource_groups::candidates(&problem, &initial, &[]).remove(0);
    let mut history = BranchEvaluationHistory::default();
    for side in 0..2 {
        history.record_at_stage(
            &old,
            side,
            0,
            EvaluationKind::RestrictedMaster,
            Some(1),
            0.,
            1.,
            0.,
        );
    }
    let current = [(&routes[0], 0.1), (&routes[1], 0.2), (&routes[2], 0.7)];
    let mut fresh = resource_groups::candidates(&problem, &current, &[]);
    assert_eq!(fresh.len(), 1);
    assert_ne!(key(&fresh[0].down), key(&old.down));
    history.refresh_resource_generators(&problem, &current, &[], &mut fresh);
    assert_eq!(fresh.len(), 2);
    assert_eq!(fresh[1].down, old.down);
    assert_eq!(fresh[1].value, 0.1);
    assert_eq!(old.value, 0.75);
    history.refresh_resource_generators(&problem, &current, &[], &mut fresh);
    assert_eq!(fresh.len(), 2);
    history.prioritize(&mut fresh, 2, &BranchHistoryOptions::default(), 10.);
    assert_eq!(fresh[0].down, old.down);
    assert_eq!(history.ranked_candidates, 1);

    let mut child = vec![];
    history.refresh_resource_generators(&problem, &current, &[old.up.clone()], &mut child);
    assert!(child.is_empty());
    history.refresh_resource_generators(&problem, &[(&routes[0], 1.)], &[], &mut child);
    assert!(child.is_empty());
    let mut invalid_proof = problem.clone();
    invalid_proof.rows[1].rhs = 2.;
    history.refresh_resource_generators(&invalid_proof, &current, &[], &mut child);
    assert!(child.is_empty());
    let mut missing = routes[0].clone();
    missing.resource_profile.clear();
    history.refresh_resource_generators(&problem, &[(&missing, 0.5)], &[], &mut child);
    assert!(child.is_empty());
    history.refresh_resource_generators(&problem, &[(&routes[0], f64::NAN)], &[], &mut child);
    assert!(child.is_empty());
}
