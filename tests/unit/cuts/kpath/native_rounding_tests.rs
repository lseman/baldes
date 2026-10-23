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
            let mut separator = NativeKPathSeparator::new(packing);
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
