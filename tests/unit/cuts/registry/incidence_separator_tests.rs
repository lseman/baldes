use super::*;

fn fixture() -> (GraphData, Vec<Solution>) {
    let mut graph = GraphData {
        id: 7,
        source: 0,
        sink: 4,
        elem_set_count: 3,
        packing_set_count: 3,
        vertices: (0..5)
            .map(|v| {
                let sets = if (1..=3).contains(&v) {
                    vec![v - 1]
                } else {
                    vec![]
                };
                VertexData {
                    id: v,
                    elem_sets: sets.clone(),
                    packing_sets: sets,
                    ..Default::default()
                }
            })
            .collect(),
        ..Default::default()
    };
    for tail in 0..4 {
        for head in 1..5 {
            if tail == head {
                continue;
            }
            let id = graph.arcs.len() as i32;
            graph.arcs.push(ArcData {
                id,
                tail,
                head,
                ..Default::default()
            });
        }
    }
    let solutions = [(1, 2), (1, 3), (2, 3)]
        .into_iter()
        .map(|(a, b)| {
            let arc_ids = [(0, a), (a, b), (b, 4)]
                .into_iter()
                .map(|(tail, head)| {
                    graph
                        .arcs
                        .iter()
                        .find(|arc| arc.tail == tail && arc.head == head)
                        .unwrap()
                        .id
                })
                .collect();
            Solution {
                graph_id: 7,
                enumerated: false,
                arc_ids,
            }
        })
        .collect();
    (graph, solutions)
}

fn parameters() -> Parameters {
    Parameters {
        max_num_rows: 3,
        max_num_per_round: 20,
        max_per_vertex: 1,
        memory_type: 1,
        arc_memory_type: 2,
        ls_num_iterations: 0,
        ..Default::default()
    }
}

#[test]
fn separates_registers_and_counts_existing_cuts_between_rounds() {
    let (graph, solutions) = fixture();
    let mut separator = IncidenceCutSeparator::new(parameters(), &[&graph]).unwrap();
    let mut registry = CutRegistry::default();
    let input = Input {
        solutions: solutions.iter().collect(),
        values: vec![0.5; 3],
        cut_memory_type: 1,
        solutions_are_set_sequences: false,
        separation_phase: 3,
        active_cuts: vec![],
    };
    let generated = separator
        .separate_and_register(&mut registry, &input)
        .unwrap();
    assert_eq!(generated.len(), 1);
    assert_eq!(generated[0].0, 0);
    assert_eq!(registry.cuts.len(), 1);
    assert_eq!(separator.state.num_separate_calls, 1);
    let again = separator
        .separate_and_register(&mut registry, &input)
        .unwrap();
    assert!(again.is_empty());
    assert_eq!(separator.state.num_separate_calls, 2);
    assert_eq!(registry.cuts.len(), 1);
    registry.retire(0);
    let after_retire = separator
        .separate_and_register(&mut registry, &input)
        .unwrap();
    assert_eq!(after_retire.len(), 1);
    assert_eq!(after_retire[0].0, 1);
}

#[test]
fn rejects_invalid_inputs_without_advancing_state_or_registry() {
    let (graph, mut solutions) = fixture();
    let mut separator = IncidenceCutSeparator::new(parameters(), &[&graph]).unwrap();
    let mut registry = CutRegistry::default();
    solutions[0].arc_ids[0] = 999;
    let input = Input {
        solutions: solutions.iter().collect(),
        values: vec![0.5; 3],
        cut_memory_type: 1,
        solutions_are_set_sequences: false,
        separation_phase: 3,
        active_cuts: vec![],
    };
    assert!(separator
        .separate_and_register(&mut registry, &input)
        .is_err());
    assert!(registry.cuts.is_empty());
    assert_eq!(separator.state.num_separate_calls, 0);
    let mut invalid = parameters();
    invalid.max_num_rows = 6;
    assert!(IncidenceCutSeparator::new(invalid, &[&graph]).is_err());
}

#[test]
fn explicit_active_cut_memory_is_inherited_in_a_later_round() {
    let (graph, solutions) = fixture();
    let mut params = parameters();
    params.max_per_vertex = 0;
    let mut separator = IncidenceCutSeparator::new(params, &[&graph]).unwrap();
    let mut registry = CutRegistry::default();
    let mut input = Input {
        solutions: solutions.iter().collect(),
        values: vec![0.5; 3],
        cut_memory_type: 1,
        solutions_are_set_sequences: false,
        separation_phase: 3,
        active_cuts: vec![],
    };
    let first = separator
        .separate_and_register(&mut registry, &input)
        .unwrap();
    assert_eq!(first.len(), 1);
    let inherited_arc = graph
        .arcs
        .iter()
        .find(|arc| arc.tail == 0 && arc.head == 1)
        .unwrap()
        .id;
    let mut active = first[0].1.clone();
    assert!(!active.memory[&7].contains(&inherited_arc));
    active.memory.get_mut(&7).unwrap().insert(inherited_arc);
    input.active_cuts.push(active);
    let second = separator
        .separate_and_register(&mut registry, &input)
        .unwrap();
    assert_eq!(second.len(), 1);
    assert!(second[0].1.memory[&7].contains(&inherited_arc));
    assert!(registry.cuts[second[0].0].graph_arc_memories[&7].contains(&inherited_arc));
}

#[test]
fn failed_registration_rolls_back_the_staged_separation() {
    let (graph, solutions) = fixture();
    let mut separator = IncidenceCutSeparator::new(parameters(), &[&graph]).unwrap();
    // Corrupt only the private conversion snapshot after preparation so that
    // separation finds a cut but conversion rejects the physical vertex IDs.
    separator.graphs[0].vertices[1].id = 99;
    let mut registry = CutRegistry::default();
    let input = Input {
        solutions: solutions.iter().collect(),
        values: vec![0.5; 3],
        cut_memory_type: 1,
        solutions_are_set_sequences: false,
        separation_phase: 3,
        active_cuts: vec![],
    };
    assert!(separator
        .separate_and_register(&mut registry, &input)
        .is_err());
    assert!(registry.cuts.is_empty());
    assert_eq!(separator.state.num_separate_calls, 0);
}
