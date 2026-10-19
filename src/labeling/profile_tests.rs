use super::*;

fn prepared() -> (Graph, LabelingEngine) {
    let mut graph = Graph::new(3, 1).unwrap();
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    for (tail, head) in [(0, 1), (1, 2)] {
        let id = graph.add_arc(tail, head, 1.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    let mut engine = LabelingEngine::prepare(
        &graph,
        EngineControls {
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
    engine.solve(&graph, &Controls::default()).unwrap();
    (graph, engine)
}

fn at(storage: &LabelStorage, vertex: usize) -> usize {
    (0..storage.labels.len())
        .find(|&id| storage.labels.view(id).vertex() == vertex)
        .unwrap()
}

#[test]
fn reconstruction_keeps_a_feasible_waiting_schedule_and_its_identity() {
    let (graph, mut engine) = prepared();
    let terminal = at(&engine.forward, graph.sink);
    let earliest = engine
        .reconstruct(&graph, (Some(terminal), None), 2.)
        .unwrap()
        .unwrap();
    let mut delayed = engine.forward.labels.view(terminal).to_owned();
    delayed.resources[0] = 5.;
    let delayed = engine.forward.retain(delayed);
    let route = engine
        .reconstruct(&graph, (Some(delayed), None), 2.)
        .unwrap()
        .unwrap();
    // Travel takes one unit per arc; arriving at time five is feasible waiting.
    assert_eq!(route.resource_profile, vec![vec![0.], vec![1.], vec![5.]]);
    assert_eq!(route.resources, vec![5.]);
    assert_eq!(route.arcs, earliest.arcs);
    assert_eq!(route.original_cost, earliest.original_cost);
    assert_ne!(route.identity(), earliest.identity());
    let mut signed_zero = route.clone();
    signed_zero.resource_profile[0][0] = -0.;
    assert_eq!(signed_zero.identity(), route.identity());
}

#[test]
fn joined_reconstruction_extends_the_actual_forward_schedule() {
    let (graph, mut engine) = prepared();
    let forward = at(&engine.forward, 1);
    let backward = at(&engine.backward, 1);
    let mut waiting = engine.forward.labels.view(forward).to_owned();
    waiting.resources[0] = 4.;
    let waiting = engine.forward.retain(waiting);
    let route = engine
        .reconstruct(&graph, (Some(waiting), Some(backward)), 2.)
        .unwrap()
        .unwrap();
    assert_eq!(route.arcs, vec![0, 1]);
    assert_eq!(route.resource_profile, vec![vec![0.], vec![4.], vec![5.]]);
    assert_eq!(route.resources, vec![5.]);
}
