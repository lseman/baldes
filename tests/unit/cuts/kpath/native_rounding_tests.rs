use super::*;
use crate::cuts::rank_one::separation::{
    prepare_incidence, ArcData, GraphData, Solution, VertexData,
};

fn cycle_graph() -> (KPathGraph, Vec<Solution>) {
    let mut graph = GraphData {
        id: 0,
        source: 0,
        sink: 17,
        elem_set_count: 16,
        packing_set_count: 16,
        covering_set_count: 16,
        ..Default::default()
    };
    for id in 0..=17 {
        let sets = if (1..=16).contains(&id) {
            vec![id - 1]
        } else {
            vec![]
        };
        graph.vertices.push(VertexData {
            id,
            elem_sets: sets.clone(),
            packing_sets: sets.clone(),
            covering_sets: sets,
        });
    }
    let mut paths = vec![];
    for customer in 1..=16 {
        let next = customer % 16 + 1;
        let mut arc_ids = vec![];
        for (tail, head) in [(0, customer), (customer, next), (next, 17)] {
            let id = graph.arcs.len() as i32;
            graph.arcs.push(ArcData {
                id,
                tail,
                head,
                ..Default::default()
            });
            arc_ids.push(id);
        }
        paths.push(Solution {
            graph_id: 0,
            enumerated: true,
            arc_ids,
        });
    }
    let incidence = Some(prepare_incidence(&graph).unwrap());
    (
        KPathGraph {
            graph,
            incidence,
            ..Default::default()
        },
        paths,
    )
}

#[test]
fn native_decimal_cuts_preserve_feasible_matchings_and_separate_real_overload() {
    let (graph, paths) = cycle_graph();
    for packing in [false, true] {
        for demand in [0.1, 0.100_000_1] {
            let mut separator = GeneralKPathSeparator::new(packing);
            separator
                .prepare(std::slice::from_ref(&graph), &[demand; 16], 0.2, -1)
                .unwrap();
            assert_eq!(separator.preparation.capacity, 0.2);
            let cuts = separator.separate(&paths, &[0.5; 16]).unwrap();
            if demand == 0.1 {
                // Odd and even cycle edges are two exact feasible eight-route
                // matchings. The input is their equally weighted average.
                assert_eq!(demand + demand, separator.preparation.capacity);
                for cut in cuts {
                    for offset in [0, 1] {
                        let lhs: i32 = paths
                            .iter()
                            .skip(offset)
                            .step_by(2)
                            .map(|p| separator.coefficient(p, &cut).unwrap())
                            .sum();
                        assert!(lhs >= cut.rhs);
                    }
                }
            } else {
                assert!(cuts.iter().any(|cut| {
                    let lhs: f64 = paths
                        .iter()
                        .map(|p| 0.5 * f64::from(separator.coefficient(p, cut).unwrap()))
                        .sum();
                    cut.rhs == 9 && lhs < f64::from(cut.rhs)
                }));
            }
        }
    }
}

#[test]
fn native_separation_transfers_pulse_storage_and_restores_it_on_rejected_input() {
    let (graph, paths) = cycle_graph();
    let mut separator = GeneralKPathSeparator::new(true);
    separator.prepare(&[graph], &[0.1; 16], 0.2, -1).unwrap();
    let pulse = &mut separator.preparation.pulse;
    assert!(pulse.set_ready_time(vec![0.; 18]));
    assert!(pulse.set_due_date(vec![100.; 18]));
    assert!(pulse.set_distance_matrix(vec![vec![1.; 18]; 18]));
    let ready = pulse.ready_time.as_ptr();
    let due = pulse.due_date.as_ptr();
    let distance = pulse.distance.as_ptr();
    let costs = pulse.arc_cost.as_ptr();
    let before = format!("{pulse:?}");
    let invalid = Solution {
        graph_id: 99,
        enumerated: false,
        arc_ids: vec![0],
    };
    assert!(separator.separate(&[invalid], &[1.]).is_err());
    assert_eq!(format!("{:?}", separator.preparation.pulse), before);
    for _ in 0..2 {
        separator.separate(&paths, &[0.5; 16]).unwrap();
        let pulse = &separator.preparation.pulse;
        assert_eq!(pulse.ready_time.as_ptr(), ready);
        assert_eq!(pulse.due_date.as_ptr(), due);
        assert_eq!(pulse.distance.as_ptr(), distance);
        assert_eq!(pulse.arc_cost.as_ptr(), costs);
        assert_eq!(format!("{pulse:?}"), before);
    }
}
