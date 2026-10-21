use super::*;
use crate::cuts::capacity::CapacityKind;
use std::collections::BTreeMap;

type ArcEntries = Vec<Vec<(usize, f64)>>;

fn fixture() -> (
    RootProblem<2>,
    Vec<CapacityCut>,
    Vec<Option<usize>>,
    ArcEntries,
) {
    let mut graph = Graph::<2>::new(5, 1).unwrap();
    for tail in 0..4 {
        for head in 1..5 {
            if tail != head {
                let id = 100 + graph.arcs.len() as i32 * 37;
                graph.add_arc_with_id(id, tail, head, 0.).unwrap();
            }
        }
    }
    graph.add_arc_with_id(9999, 1, 2, 0.).unwrap();
    let problem = RootProblem::set_partitioning(graph, 4.).unwrap();
    let cuts: Vec<_> = (0..4)
        .map(|id| CapacityCut {
            id,
            customers: vec![1, 2],
            other: vec![3],
            kind: if id == 2 {
                CapacityKind::Complement
            } else {
                CapacityKind::Capacity
            },
            rhs: 2,
            k: 1,
            demand: 2.,
            violation: 1.,
            edges: (0..4)
                .flat_map(|a| {
                    (a + 1..4).map(move |b| {
                        let weight = match id {
                            0 => 0.25,
                            1 => 100., // Retired row; no cached entries may survive.
                            2 => ((a * 3 + b) % 5) as f64 * 0.25 - 0.5,
                            _ => 0.,
                        };
                        ((a, b), weight)
                    })
                })
                .collect::<BTreeMap<_, _>>(),
        })
        .collect();
    let rows = vec![Some(3), None, Some(11), Some(5)];
    let cache = problem
        .graph
        .arcs
        .iter()
        .map(|arc| {
            let head = if arc.head == problem.graph.sink {
                0
            } else {
                arc.head
            };
            cuts.iter()
                .zip(&rows)
                .filter_map(|(cut, &row)| {
                    let row = row?;
                    let value = cut.edge_coefficient(arc.tail, head);
                    (value != 0.).then_some((row, value))
                })
                .collect()
        })
        .collect();
    (problem, cuts, rows, cache)
}

fn route(problem: &RootProblem<2>, customers: &[usize]) -> Route {
    let nodes: Vec<_> = std::iter::once(0)
        .chain(customers.iter().copied())
        .chain(std::iter::once(problem.graph.sink))
        .map(|v| v as i32)
        .collect();
    let arcs = nodes
        .windows(2)
        .map(|edge| {
            problem
                .graph
                .arcs
                .iter()
                .rev()
                .find(|a| a.tail as i32 == edge[0] && a.head as i32 == edge[1])
                .unwrap()
                .id
        })
        .collect();
    Route {
        nodes,
        arcs,
        cost: 0.,
        original_cost: 0.,
        resources: vec![0.],
        resource_profile: vec![],
    }
}

fn expected(
    cuts: &[CapacityCut],
    rows: &[Option<usize>],
    customers: &[usize],
) -> Vec<(usize, f64)> {
    cuts.iter()
        .zip(rows)
        .filter_map(|(cut, &row)| {
            let row = row?;
            let value = cut.coefficient(customers);
            (value != 0).then_some((row, value as f64))
        })
        .collect()
}

#[test]
fn cached_capacity_matches_every_small_walk_and_rounds_after_summing() {
    let (problem, cuts, rows, cache) = fixture();
    let mut walks = vec![vec![]];
    let mut frontier = vec![vec![]];
    for _ in 0..5 {
        let mut next = vec![];
        for walk in &frontier {
            for v in 1..4 {
                if walk.last() == Some(&v) {
                    continue;
                }
                let mut extended = walk.clone();
                extended.push(v);
                next.push(extended);
            }
        }
        walks.extend(next.iter().cloned());
        frontier = next;
    }
    for customers in walks {
        let route = route(&problem, &customers);
        assert_eq!(
            capacity_entries_from_arcs(&problem, &route, &rows, &cache),
            Some(expected(&cuts, &rows, &customers)),
            "{customers:?}"
        );
    }
    // Four quarter-weight edges sum to one; rounding each arc would give zero.
    let repeated = route(&problem, &[1, 2, 1]);
    assert_eq!(
        capacity_entries_from_arcs(&problem, &repeated, &rows, &cache).unwrap()[0],
        (3, 1.)
    );
    assert!(repeated.arcs.contains(&9999));
}

#[test]
fn cached_capacity_respects_retirement_and_row_remapping() {
    let (problem, cuts, mut rows, mut cache) = fixture();
    let mut remap = [None; 12];
    remap[11] = Some(2);
    remap[5] = Some(0);
    for row in &mut rows {
        *row = row.and_then(|r| remap[r]);
    }
    for entries in &mut cache {
        *entries = entries
            .iter()
            .filter_map(|&(r, a)| remap[r].map(|row| (row, a)))
            .collect();
    }
    for customers in [vec![], vec![1], vec![1, 2, 1, 3], vec![3, 2, 3]] {
        assert_eq!(
            capacity_entries_from_arcs(&problem, &route(&problem, &customers), &rows, &cache),
            Some(expected(&cuts, &rows, &customers))
        );
    }
    assert_eq!(
        capacity_entries_from_arcs(&problem, &route(&problem, &[1]), &[None], &[]),
        Some(vec![])
    );
}

#[test]
fn cached_capacity_rejects_unmatched_arcs_or_noncanonical_depots() {
    let (problem, _, rows, cache) = fixture();
    let valid = route(&problem, &[1, 2]);
    let mut invalid = valid.clone();
    invalid.arcs[0] = -999;
    assert!(capacity_entries_from_arcs(&problem, &invalid, &rows, &cache).is_none());
    invalid = valid.clone();
    invalid.arcs[0] = valid.arcs[1];
    assert!(capacity_entries_from_arcs(&problem, &invalid, &rows, &cache).is_none());
    invalid = valid.clone();
    invalid.arcs.pop();
    assert!(capacity_entries_from_arcs(&problem, &invalid, &rows, &cache).is_none());
    invalid = valid.clone();
    invalid.nodes[0] = 1;
    assert!(capacity_entries_from_arcs(&problem, &invalid, &rows, &cache).is_none());
    invalid = valid.clone();
    invalid.nodes[1] = problem.graph.sink as i32;
    assert!(capacity_entries_from_arcs(&problem, &invalid, &rows, &cache).is_none());
    assert!(capacity_entries_from_arcs(&problem, &valid, &rows, &[]).is_none());
    let mut other_source = problem.clone();
    other_source.graph.source = 1;
    assert!(capacity_entries_from_arcs(&other_source, &valid, &rows, &cache).is_none());
}
