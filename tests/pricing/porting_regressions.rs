//! Independent pricing and enumeration regression checks.
use baldes::{pricing::Solver, Graph, Route};
use std::collections::BTreeMap;

#[path = "mixed_reversal.rs"]
mod mixed_reversal;

fn solver(symmetric: bool, cut: bool) -> Solver<2> {
    let mut s = Solver::<2>::new(5, 1).unwrap();
    s.graph.vertices[0].elements.clear();
    s.graph.vertices[4].elements.clear();
    for v in 0..5 {
        s.graph.set_bounds(v, 0, 0., 4.).unwrap();
    }
    s.graph.elementary = true;
    for u in 0..4 {
        for v in 1..5 {
            if u == v || (u == 0 && v == 4) {
                continue;
            }
            let cost = if u == 0 || v == 4 {
                1.
            } else {
                (u + v) as f64 / 4.
            };
            let a = s.graph.add_arc(u, v, cost).unwrap();
            s.graph.arc_mut(a).unwrap().consumption[0] = 1.;
        }
    }
    if !symmetric {
        // Disable the symmetric resource domain without changing path feasibility.
        s.graph.vertices[1].bounds[0].lower = 0.25;
    }
    if cut {
        s.cuts.add_strong_k_path_cut(5, &[1, 2], 1, -1.25).unwrap();
    }
    s.enumeration_preserve_arc_paths = true;
    s
}

fn costs(routes: Vec<Route>) -> BTreeMap<Vec<i32>, f64> {
    routes.into_iter().map(|r| (r.arcs, r.cost)).collect()
}

fn assert_costs(actual: &BTreeMap<Vec<i32>, f64>, expected: &BTreeMap<Vec<i32>, f64>) {
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>()
    );
    for (path, value) in expected {
        assert!(
            (actual[path] - value).abs() < 1e-8,
            "{path:?}: {} != {value}",
            actual[path]
        );
    }
}

// Exhaustive elementary paths: no buckets, dominance, completion bounds or
// production coefficient routines. The optional cut charges a support once.
fn exhaustive(g: &Graph<2>, cut: bool, threshold: f64) -> BTreeMap<Vec<i32>, f64> {
    fn walk(
        g: &Graph<2>,
        v: usize,
        seen: u64,
        q: Vec<f64>,
        cost: f64,
        path: &mut Vec<i32>,
        cut: bool,
        threshold: f64,
        out: &mut BTreeMap<Vec<i32>, f64>,
    ) {
        if v == g.sink {
            let value = cost
                + if cut && seen & ((1 << 1) | (1 << 2)) != 0 {
                    1.25
                } else {
                    0.
                };
            if value < threshold {
                out.insert(path.clone(), value);
            }
            return;
        }
        for arc in &g.arcs {
            if arc.tail != v || seen & (1 << arc.head) != 0 {
                continue;
            }
            let mut next = q.clone();
            let mut feasible = true;
            for r in 0..g.resources.len() {
                let bounds = arc.bounds[r].unwrap_or(g.vertices[arc.head].bounds[r]);
                next[r] = (next[r] + arc.consumption[r]).max(bounds.lower);
                feasible &= next[r] <= bounds.upper;
            }
            if !feasible {
                continue;
            }
            path.push(arc.id);
            walk(
                g,
                arc.head,
                seen | (1 << arc.head),
                next,
                cost + arc.reduced_cost,
                path,
                cut,
                threshold,
                out,
            );
            path.pop();
        }
    }
    let mut out = BTreeMap::new();
    walk(
        g,
        g.source,
        1 << g.source,
        (0..g.resources.len())
            .map(|r| g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower))
            .collect(),
        0.,
        &mut vec![],
        cut,
        threshold,
        &mut out,
    );
    out
}

#[test]
fn halfway_completion_matches_fresh_completion_and_exhaustive_paths() {
    for cut in [false, true] {
        let mut retained = solver(false, cut);
        retained.price_phase(0, 100, false).unwrap();
        assert!(retained.engine.as_ref().unwrap().half_ready);
        assert!(!retained.engine.as_ref().unwrap().consumption_joins);
        let labels_before = retained.engine.as_ref().unwrap().forward.labels.len();
        retained.complete_labels(true).unwrap();
        assert!(retained.engine.as_ref().unwrap().full_ready);
        assert!(retained.engine.as_ref().unwrap().forward.labels.len() >= labels_before);
        let actual = costs(retained.enumerate(100.).unwrap());
        let mut fresh = solver(false, cut);
        fresh.complete_labels(true).unwrap();
        assert_costs(&actual, &costs(fresh.enumerate(100.).unwrap()));
        assert_costs(&actual, &exhaustive(&retained.graph, cut, 100.));
    }
}

#[test]
fn changed_duals_reject_direct_continuation_and_rebuild_completion() {
    let mut s = solver(false, true);
    s.price_phase(0, 100, false).unwrap();
    s.graph.arcs[0].reduced_cost -= 3.;
    assert!(s
        .engine
        .as_mut()
        .unwrap()
        .continue_after_halfway(&s.graph, &s.controls)
        .is_err());
    s.complete_labels(true).unwrap();
    assert_costs(
        &costs(s.enumerate(100.).unwrap()),
        &exhaustive(&s.graph, true, 100.),
    );
}

#[test]
fn symmetric_cut_pricing_and_fixing_preserve_the_exhaustive_universe() {
    for cut in [false, true] {
        for symmetric in [false, true] {
            for threshold in [2.5, 3.5, 5.] {
                let mut s = solver(symmetric, cut);
                let priced = s.price_phase(0, 100, false).unwrap();
                assert_eq!(s.engine.as_ref().unwrap().consumption_joins, symmetric);
                let expected = exhaustive(&s.graph, cut, threshold);
                let minimum = exhaustive(&s.graph, cut, 100.)
                    .values()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                assert!((priced.minimum_reduced_cost - minimum).abs() < 1e-8);
                let mut before = solver(symmetric, cut);
                assert_costs(&costs(before.enumerate(threshold).unwrap()), &expected);
                s.fix(threshold).unwrap();
                assert_eq!(s.engine.as_ref().unwrap().consumption_joins, symmetric);
                assert_costs(&costs(s.enumerate(threshold).unwrap()), &expected);
            }
        }
    }
}

#[test]
fn asymmetric_reduced_costs_on_a_symmetric_resource_graph_match_direct_path_costs() {
    let mut s = solver(true, true);
    for (i, arc) in s.graph.arcs.iter_mut().enumerate() {
        arc.reduced_cost = (i as f64 * 7. % 13.) - 6.;
    }
    let expected = exhaustive(&s.graph, true, 100.);
    let minimum = expected.values().copied().fold(f64::INFINITY, f64::min);
    let result = s.price_phase(0, 100, false).unwrap();
    assert!(
        (result.minimum_reduced_cost - minimum).abs() < 1e-8,
        "minimum={} expected={minimum}, symmetric={}",
        result.minimum_reduced_cost,
        s.engine.as_ref().unwrap().consumption_joins
    );
    for route in result.columns {
        assert!(
            (route.cost - expected[&route.arcs]).abs() < 1e-8,
            "{:?}: {} != {}",
            route.arcs,
            route.cost,
            expected[&route.arcs]
        );
    }
}

#[test]
fn dual_change_after_symmetric_preparation_falls_back_for_pricing_and_completion() {
    for complete in [false, true] {
        let mut s = solver(true, true);
        s.price_phase(0, 100, false).unwrap();
        assert!(s.engine.as_ref().unwrap().consumption_joins);
        s.graph.arcs[0].reduced_cost -= 7.;
        let expected = exhaustive(&s.graph, true, 100.);
        if complete {
            s.complete_labels(true).unwrap();
        } else {
            let priced = s.price_phase(0, 100, false).unwrap();
            let minimum = expected.values().copied().fold(f64::INFINITY, f64::min);
            assert!((priced.minimum_reduced_cost - minimum).abs() < 1e-8);
        }
        assert!(!s.engine.as_ref().unwrap().consumption_joins);
        assert_costs(&costs(s.enumerate(100.).unwrap()), &expected);
    }
}

#[test]
fn depot_elements_are_transformed_for_symmetric_labeling() {
    let mut s = solver(true, false);
    s.graph.vertices[0].elements = vec![0];
    s.graph.vertices[4].elements = vec![4];
    s.price_phase(0, 100, false).unwrap();
    assert!(s.engine.as_ref().unwrap().consumption_joins);
    assert_costs(
        &costs(s.enumerate(2.5).unwrap()),
        &exhaustive(&s.graph, false, 2.5),
    );
}

#[test]
fn nonzero_initial_resource_is_transformed_for_symmetric_labeling() {
    let mut s = solver(true, true);
    for v in 0..5 {
        s.graph.set_bounds(v, 0, 3., 7.).unwrap();
    }
    s.graph.initial[0] = Some(3.);
    let result = s.price_phase(0, 100, false).unwrap();
    assert!(s.engine.as_ref().unwrap().consumption_joins);
    let expected = exhaustive(&s.graph, true, 100.);
    let minimum = expected.values().copied().fold(f64::INFINITY, f64::min);
    assert!((result.minimum_reduced_cost - minimum).abs() < 1e-8);
    assert_costs(&costs(s.enumerate(100.).unwrap()), &expected);
}

#[test]
fn rank_one_memory_policy_separates_shared_and_graph_dependent_memories() {
    use baldes::cuts::separation::{
        prepare_incidence, ArcData, ColumnInfo, GraphData, SeparatedCut, Separation, VertexData,
    };
    use std::collections::BTreeSet;
    let graph = |id| GraphData {
        id,
        source: 0,
        sink: 4,
        elem_set_count: 2,
        packing_set_count: 2,
        vertices: (0..5)
            .map(|v| VertexData {
                id: v,
                elem_sets: if v == 1 {
                    vec![0]
                } else if v == 2 {
                    vec![1]
                } else {
                    vec![]
                },
                packing_sets: if v == 1 {
                    vec![0]
                } else if v == 2 {
                    vec![1]
                } else {
                    vec![]
                },
                ..Default::default()
            })
            .collect(),
        arcs: [(0, 1), (1, 3), (3, 2), (2, 4), (1, 2), (1, 3)]
            .into_iter()
            .enumerate()
            .map(|(a, (tail, head))| ArcData {
                id: a as i32,
                tail,
                head,
                elem_sets: if head == 1 {
                    vec![0]
                } else if head == 2 {
                    vec![1]
                } else {
                    vec![]
                },
                packing_sets: if head == 1 {
                    vec![0]
                } else if head == 2 {
                    vec![1]
                } else {
                    vec![]
                },
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let (g0, g1) = (graph(0), graph(1));
    for dependent in [false, true] {
        for memory_type in [0, 1] {
            for arc_mode in 0..=2 {
                let mut separation = Separation::default();
                separation.params.sp_dependent_memory = dependent;
                separation.params.memory_type = memory_type;
                separation.params.arc_memory_type = arc_mode;
                separation.cut_memory_type = memory_type;
                let i0 = prepare_incidence(&g0).unwrap();
                let column = ColumnInfo::from_arcs(0, &i0, 0.5, vec![0, 1, 2, 3], false, 2, 0);
                assert!(separation.prepare(
                    &[&g0, &g1],
                    vec![(0, i0), (1, prepare_incidence(&g1).unwrap())]
                ));
                let mut cut = SeparatedCut {
                    kind: 0,
                    num_rows: 2,
                    denominator: 2,
                    numerators: vec![1, 1],
                    set_ids: vec![0, 1],
                    memory_type,
                    ..Default::default()
                };
                assert!(separation.set_cut_memory(&mut cut, &[column]));
                let expected = if memory_type == 0 {
                    BTreeSet::from([1, 2, 3])
                } else {
                    match arc_mode {
                        0 => BTreeSet::from([1, 2, 5]),
                        1 => BTreeSet::from([1, 2, 4, 5]),
                        _ => BTreeSet::from([0, 1, 2, 4, 5]),
                    }
                };
                assert_eq!(cut.memory[&0], expected);
                if dependent {
                    assert!(cut.memory[&1].is_empty());
                } else {
                    assert_eq!(cut.memory[&1], expected);
                }
            }
        }
    }
}

#[test]
fn rank_one_memory_handles_sets_absent_from_one_graph() {
    use baldes::cuts::separation::{
        prepare_incidence, ArcData, ColumnInfo, GraphData, SeparatedCut, Separation, VertexData,
    };

    let graph = |id, set_count| GraphData {
        id,
        source: 0,
        sink: 3,
        elem_set_count: 2,
        packing_set_count: set_count,
        covering_set_count: set_count,
        vertices: (0..4)
            .map(|v| {
                let sets = if v == 1 || (v == 2 && set_count == 2) {
                    vec![v - 1]
                } else {
                    vec![]
                };
                VertexData {
                    id: v,
                    elem_sets: sets.clone(),
                    packing_sets: sets.clone(),
                    covering_sets: sets,
                    ..Default::default()
                }
            })
            .collect(),
        arcs: [(0, 1), (1, 2), (2, 3)]
            .into_iter()
            .enumerate()
            .map(|(id, (tail, head))| ArcData {
                id: id as i32,
                tail,
                head,
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let (g0, g1) = (graph(0, 2), graph(1, 1));
    let (i0, i1) = (
        prepare_incidence(&g0).unwrap(),
        prepare_incidence(&g1).unwrap(),
    );
    for memory_type in [0, 1] {
        for covering in [false, true] {
            for dependent in [false, true] {
                let mut separation = Separation::default();
                separation.params.sp_dependent_memory = dependent;
                separation.params.memory_type = memory_type;
                separation.params.arc_memory_type = 2;
                separation.cut_memory_type = memory_type;
                assert!(separation.prepare(&[&g0, &g1], vec![(0, i0.clone()), (1, i1.clone())]));
                let column = ColumnInfo::from_arcs(0, &i0, 0.5, vec![0, 1, 2], false, 2, 2);
                let mut cut = SeparatedCut {
                    kind: i32::from(covering),
                    num_rows: 2,
                    denominator: if covering { -2 } else { 2 },
                    numerators: vec![if covering { -1 } else { 1 }; 2],
                    set_ids: vec![0, 1],
                    memory_type,
                    ..Default::default()
                };
                assert!(separation.set_cut_memory(&mut cut, &[column]));
                assert!(!cut.memory[&0].is_empty());
                if dependent {
                    assert!(cut.memory[&1].is_empty());
                } else {
                    assert_eq!(cut.memory[&0], cut.memory[&1]);
                }
            }
        }
    }
}

#[test]
fn rank_one_preparation_rejects_invalid_neighbour_data() {
    use baldes::cuts::separation::{prepare_incidence, GraphData, Separation, VertexData};

    let graph = |id| GraphData {
        id,
        source: 0,
        sink: 1,
        elem_set_count: 2,
        packing_set_count: 2,
        vertices: vec![
            VertexData {
                id: 0,
                ..Default::default()
            },
            VertexData {
                id: 1,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let g0 = graph(0);
    let mut g1 = graph(1);
    g1.elem_set_distances = vec![vec![0.]];
    let check = |g1: &GraphData| {
        let mut separator = Separation::default();
        separator.params.max_num_rows = 4;
        separator.prepare(
            &[&g0, g1],
            vec![
                (0, prepare_incidence(&g0).unwrap()),
                (1, prepare_incidence(g1).unwrap()),
            ],
        )
    };
    assert!(!check(&g1));
    g1.elem_set_distances.clear();
    g1.packing_neighbourhood = vec![vec![1024]];
    assert!(!check(&g1));
    g1.packing_neighbourhood = vec![vec![-1]];
    assert!(!check(&g1));
    g1.packing_neighbourhood.clear();
    g1.elem_set_distances = vec![vec![0., f64::NAN], vec![0., 0.]];
    assert!(!check(&g1));
    g1.elem_set_distances = vec![vec![0., 1.], vec![1., 0.]];
    assert!(check(&g1));
}

#[test]
fn rank_one_neighbours_match_two_graph_reference_captures() {
    use baldes::cuts::separation::{bit, prepare_incidence, GraphData, Separation, VertexData};

    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/rank_one_neighbours_native.json"
    ))
    .unwrap();
    assert_eq!(
        capture["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    for case in capture["cases"].as_array().unwrap() {
        let graph = |id, index| GraphData {
            id,
            source: 0,
            sink: 1,
            elem_set_count: 5,
            packing_set_count: 5,
            vertices: vec![
                VertexData {
                    id: 0,
                    ..Default::default()
                },
                VertexData {
                    id: 1,
                    ..Default::default()
                },
            ],
            elem_set_distances: case["matrices"][index]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect()
                })
                .collect(),
            packing_neighbourhood: case["neighbourhoods"][index]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_i64().unwrap() as i32)
                        .collect()
                })
                .collect(),
            ..Default::default()
        };
        let (g0, g1) = (graph(7, 0), graph(8, 1));
        let mut separator = Separation::default();
        separator.params.max_num_rows = 4;
        separator.params.neighbourhood_size = case["k"].as_i64().unwrap() as i32;
        assert!(separator.prepare(
            &[&g0, &g1],
            vec![
                (7, prepare_incidence(&g0).unwrap()),
                (8, prepare_incidence(&g1).unwrap())
            ]
        ));
        let expected = case["neighbours"].as_array().unwrap();
        assert_eq!(separator.packing_neighbours.len(), expected.len());
        for (row, bits) in separator.packing_neighbours.iter().enumerate() {
            let actual: Vec<_> = (0..5).filter(|&column| bit(bits, column)).collect();
            let native: Vec<_> = expected[row]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(actual, native, "row {row}");
        }
    }
}

#[test]
fn rank_one_memory_matches_two_graph_reference_captures() {
    use baldes::cuts::separation::{
        prepare_incidence, ArcData, ColumnInfo, GraphData, SeparatedCut, Separation, VertexData,
    };
    use std::collections::BTreeSet;

    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/rank_one_two_graph_memory_native.json"
    ))
    .unwrap();
    assert_eq!(
        capture["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let integer = |value: &serde_json::Value| value.as_i64().unwrap() as i32;
    let graph = |id, set_count, covering, variant| GraphData {
        id,
        source: 0,
        sink: 6,
        elem_set_count: 5,
        packing_set_count: set_count,
        covering_set_count: if covering { 5 } else { 0 },
        vertices: (0..7)
            .map(|vertex| {
                let sets = if (1..=5).contains(&vertex) {
                    vec![vertex - 1]
                } else {
                    vec![]
                };
                let present = if id == 8 && vertex == 5 {
                    vec![]
                } else {
                    sets.clone()
                };
                VertexData {
                    id: vertex,
                    elem_sets: sets,
                    packing_sets: present.clone(),
                    covering_sets: if covering { present } else { vec![] },
                    ..Default::default()
                }
            })
            .collect(),
        arcs: capture[match (id, variant) {
            (8, 1) => "arcs_permuted",
            (8, 2) => "arcs_parallel",
            _ => "arcs",
        }]
        .as_array()
        .unwrap()
        .iter()
        .map(|arc| ArcData {
            id: integer(&arc["id"]),
            tail: integer(&arc["tail"]),
            head: integer(&arc["head"]),
            ..Default::default()
        })
        .collect(),
        ..Default::default()
    };
    let mut checked = 0;
    for variant in 0..3 {
        for covering in [false, true] {
            for other_set_count in [4, 5] {
                if (covering || variant != 0) && other_set_count == 4 {
                    continue;
                }
                let (g7, g8) = (
                    graph(7, 5, covering, variant),
                    graph(8, other_set_count, covering, variant),
                );
                let (i7, i8) = (
                    prepare_incidence(&g7).unwrap(),
                    prepare_incidence(&g8).unwrap(),
                );
                for (index, hit) in capture["hits"].as_array().unwrap().iter().enumerate() {
                    // The unequal-count reference run completes its first memory call,
                    // then faults while constructing a later five-row cut.
                    if other_set_count == 4 && index != 0 {
                        break;
                    }
                    let captured_variant = if hit["parallel"].as_bool().unwrap_or(false) {
                        2
                    } else if hit["permuted"].as_bool().unwrap_or(false) {
                        1
                    } else {
                        0
                    };
                    if captured_variant != variant {
                        continue;
                    }
                    let covering_domain = hit["covering_domain"]
                        .as_bool()
                        .unwrap_or_else(|| integer(&hit["cut"]["kind"]) == 1);
                    if covering_domain != covering {
                        continue;
                    }
                    checked += 1;
                    assert_eq!(integer(&hit["owner_arc_memory_mode"]), 2);
                    let dependent = integer(&hit["owner_graph_dependent"]) != 0;
                    let memory_type = integer(&hit["cut_memory_type"]);
                    assert!(matches!(memory_type, 0 | 1));
                    let mut separator = Separation::default();
                    separator.params.memory_type = memory_type;
                    separator.params.arc_memory_type = 2;
                    separator.params.sp_dependent_memory = dependent;
                    separator.cut_memory_type = memory_type;
                    assert!(separator.prepare(&[&g7, &g8], vec![(7, i7.clone()), (8, i8.clone())]));
                    let columns: Vec<_> = hit["columns"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|column| {
                            let graph = integer(&column["graph"]);
                            let incidence = if graph == 7 { &i7 } else { &i8 };
                            let replay = ColumnInfo::from_arcs(
                                graph,
                                incidence,
                                column["value"].as_f64().unwrap(),
                                column["arc_ids"]
                                    .as_array()
                                    .unwrap()
                                    .iter()
                                    .map(integer)
                                    .collect(),
                                integer(&column["enumerated"]) != 0,
                                5,
                                if covering { 5 } else { 0 },
                            );
                            let captured_counts = |name: &str| {
                                column[name]
                                    .as_array()
                                    .unwrap()
                                    .iter()
                                    .map(integer)
                                    .collect::<Vec<_>>()
                            };
                            assert_eq!(replay.packing_set_count, captured_counts("packing_counts"));
                            assert_eq!(
                                replay.covering_set_count,
                                captured_counts("covering_counts")
                            );
                            replay
                        })
                        .collect();
                    let item = &hit["cut"];
                    let list =
                        |name: &str| item[name].as_array().unwrap().iter().map(integer).collect();
                    let mut cut = SeparatedCut {
                        id: integer(&item["id"]),
                        rhs: item["rhs"].as_f64().unwrap(),
                        kind: integer(&item["kind"]),
                        num_rows: integer(&item["rows"]),
                        class: integer(&item["class"]),
                        denominator: integer(&item["denominator"]),
                        numerators: list("numerators"),
                        set_ids: list("set_ids"),
                        memory_type,
                        ..Default::default()
                    };
                    assert_eq!(
                        separator.set_cut_memory(&mut cut, &columns),
                        integer(&hit["success"]) != 0
                    );
                    for entry in hit["memory"].as_array().unwrap() {
                        let graph = integer(&entry["graph"]);
                        let expected: BTreeSet<_> = entry["values"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(integer)
                            .collect();
                        assert_eq!(cut.memory[&graph], expected, "cut {} graph {graph}", cut.id);
                    }
                }
            }
        }
    }
    assert_eq!(checked, 116);
}

#[test]
fn mixed_arc_and_vertex_set_memory_matches_two_graph_reference_captures() {
    use baldes::cuts::separation::{
        prepare_incidence, ArcData, ColumnInfo, GraphData, SeparatedCut, Separation, Solution,
        VertexData,
    };
    use baldes::cuts::{coefficient, CutRegistry};
    use std::collections::BTreeSet;
    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_memory_native.json"
    ))
    .unwrap();
    assert_eq!(
        capture["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let number = |value: &serde_json::Value| value.as_i64().unwrap() as i32;
    let make_graph = |id, arcs: &str| GraphData {
        id,
        source: 0,
        sink: 7,
        elem_set_count: 6,
        packing_set_count: 6,
        covering_set_count: 6,
        vertices: (0..8)
            .map(|vertex| {
                let sets = if vertex == 1 {
                    vec![1]
                } else if (3..=6).contains(&vertex) {
                    vec![vertex - 1]
                } else {
                    vec![]
                };
                let rows = if id == 8 && vertex == 6 {
                    vec![]
                } else {
                    sets.clone()
                };
                VertexData {
                    id: vertex,
                    elem_sets: sets,
                    packing_sets: rows.clone(),
                    covering_sets: rows,
                }
            })
            .collect(),
        arcs: capture[arcs]
            .as_array()
            .unwrap()
            .iter()
            .map(|arc| {
                let tail = number(&arc["tail"]);
                let head = number(&arc["head"]);
                let mut sets = vec![];
                if head == 1 && tail != 4 && tail != 5 {
                    sets.push(0);
                }
                ArcData {
                    id: number(&arc["id"]),
                    tail,
                    head,
                    elem_sets: sets.clone(),
                    packing_sets: sets.clone(),
                    covering_sets: sets,
                    ..Default::default()
                }
            })
            .collect(),
        ..Default::default()
    };
    let (g7, g8) = (make_graph(7, "arcs7"), make_graph(8, "arcs8"));
    let (i7, i8) = (
        prepare_incidence(&g7).unwrap(),
        prepare_incidence(&g8).unwrap(),
    );
    let mut checked = 0;
    for hit in capture["hits"].as_array().unwrap() {
        let memory_type = number(&hit["cut_memory_type"]);
        let mut separator = Separation::default();
        separator.params.memory_type = memory_type;
        separator.params.arc_memory_type = number(&hit["owner_arc_memory_mode"]);
        separator.params.sp_dependent_memory = number(&hit["owner_graph_dependent"]) != 0;
        separator.cut_memory_type = memory_type;
        assert!(separator.prepare(&[&g7, &g8], vec![(7, i7.clone()), (8, i8.clone())]));
        let columns: Vec<_> = hit["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|column| {
                let graph = number(&column["graph"]);
                let incidence = if graph == 7 { &i7 } else { &i8 };
                let result = ColumnInfo::from_arcs(
                    graph,
                    incidence,
                    column["value"].as_f64().unwrap(),
                    column["arc_ids"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(number)
                        .collect(),
                    number(&column["enumerated"]) != 0,
                    6,
                    6,
                );
                let list = |name: &str| {
                    column[name]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(number)
                        .collect::<Vec<_>>()
                };
                assert_eq!(result.packing_set_count, list("packing_counts"));
                assert_eq!(result.covering_set_count, list("covering_counts"));
                result
            })
            .collect();
        let item = &hit["cut"];
        let list = |name: &str| item[name].as_array().unwrap().iter().map(number).collect();
        let mut cut = SeparatedCut {
            id: number(&item["id"]),
            rhs: item["rhs"].as_f64().unwrap(),
            kind: number(&item["kind"]),
            num_rows: number(&item["rows"]),
            class: number(&item["class"]),
            denominator: number(&item["denominator"]),
            numerators: list("numerators"),
            set_ids: list("set_ids"),
            memory_type,
            ..Default::default()
        };
        assert_eq!(
            separator.set_cut_memory(&mut cut, &columns),
            number(&hit["success"]) != 0,
            "mode {} cut {}",
            hit["capture_mode"],
            cut.id
        );
        assert_eq!(cut.memory_type, number(&hit["after_cut_memory_type"]));
        for entry in hit["memory"].as_array().unwrap() {
            let graph = number(&entry["graph"]);
            let expected: BTreeSet<_> = entry["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect();
            assert_eq!(
                cut.memory[&graph], expected,
                "mode {} cut {} graph {graph}",
                hit["capture_mode"], cut.id
            );
        }
        let mut registry = CutRegistry::default();
        let index = registry.register_separated_cut(&[&g7, &g8], &cut).unwrap();
        for column in &columns {
            let graph = if column.graph_id == 7 { &g7 } else { &g8 };
            let route: Vec<_> = column
                .arc_ids
                .iter()
                .map(|&id| {
                    let arc = graph.arcs.iter().find(|arc| arc.id == id).unwrap();
                    (arc.tail as usize, arc.head as usize, id)
                })
                .collect();
            for enumerated in [false, true] {
                let solution = Solution {
                    graph_id: column.graph_id,
                    enumerated,
                    arc_ids: column.arc_ids.clone(),
                };
                assert_eq!(
                    f64::from(
                        coefficient(&registry.cuts[index], column.graph_id, &route, enumerated)
                            .unwrap()
                    ),
                    separator.coefficient(&solution, &cut),
                    "mode {} cut {} graph {}",
                    hit["capture_mode"],
                    cut.id,
                    column.graph_id
                );
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 158);
}

#[test]
fn mixed_incidence_separation_matches_two_graph_reference_capture() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_separation_native.json"
    ));
}

#[test]
fn complete_mixed_incidence_separation_matches_two_graph_reference_capture() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_complete_separation_native.json"
    ));
}

fn check_mixed_incidence_separation_capture(capture: &str) {
    use baldes::cuts::separation::{
        prepare_incidence, ArcData, GraphData, Input, Parameters, Separation, Solution, VertexData,
    };
    use std::collections::{BTreeMap, BTreeSet};
    let capture: serde_json::Value = serde_json::from_str(capture).unwrap();
    assert_eq!(
        capture["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    assert_eq!(capture["controlled_covering_seed"], 1);
    let number = |value: &serde_json::Value| value.as_i64().unwrap() as i32;
    let complete_incidence = capture["complete_incidence"].as_bool().unwrap_or(false);
    let make_graph = |id, arcs: &str| GraphData {
        id,
        source: 0,
        sink: 7,
        elem_set_count: 6,
        packing_set_count: 6,
        covering_set_count: 6,
        vertices: (0..8)
            .map(|vertex| {
                let sets = if vertex == 1 {
                    vec![1]
                } else if (3..=6).contains(&vertex) {
                    vec![vertex - 1]
                } else {
                    vec![]
                };
                let rows = if !complete_incidence && id == 8 && vertex == 6 {
                    vec![]
                } else {
                    sets.clone()
                };
                VertexData {
                    id: vertex,
                    elem_sets: sets,
                    packing_sets: rows.clone(),
                    covering_sets: rows,
                }
            })
            .collect(),
        arcs: capture[arcs]
            .as_array()
            .unwrap()
            .iter()
            .map(|arc| {
                let tail = number(&arc["tail"]);
                let head = number(&arc["head"]);
                let mut sets = vec![];
                if head == 1 && tail != 4 && tail != 5 {
                    sets.push(0);
                }
                ArcData {
                    id: number(&arc["id"]),
                    tail,
                    head,
                    elem_sets: sets.clone(),
                    packing_sets: sets.clone(),
                    covering_sets: sets,
                    ..Default::default()
                }
            })
            .collect(),
        ..Default::default()
    };
    let (g7, g8) = (make_graph(7, "arcs7"), make_graph(8, "arcs8"));
    if complete_incidence {
        for (graph, entries) in [(&g7, "master_entries7"), (&g8, "master_entries8")] {
            let incidence = prepare_incidence(graph).unwrap();
            let rows = capture[entries].as_array().unwrap();
            assert_eq!(rows.len(), graph.arcs.len());
            for (arc, entries) in graph.arcs.iter().zip(rows) {
                let mut expected: Vec<i32> = entries
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|entry| {
                        assert_eq!(entry[1].as_f64().unwrap(), 1.);
                        number(&entry[0])
                    })
                    .collect();
                expected.sort_unstable();
                assert_eq!(
                    incidence.arc_packing_sets[arc.id as usize], expected,
                    "graph {} arc {}",
                    graph.id, arc.id
                );
                assert_eq!(
                    incidence.arc_covering_sets[arc.id as usize], expected,
                    "graph {} arc {}",
                    graph.id, arc.id
                );
            }
        }
    }
    let call = &capture["call"];
    let params = Parameters {
        max_num_rows: 5,
        max_num_per_round: 20,
        memory_type: 1,
        arc_memory_type: 2,
        sp_dependent_memory: false,
        ls_num_iterations: 1000,
        neighbourhood_size: 16,
        max_num_triplets: 10000,
        violation_tolerance: 0.02,
        ..Default::default()
    };
    let mut integrated =
        baldes::cuts::IncidenceCutSeparator::new(params.clone(), &[&g7, &g8]).unwrap();
    let mut separator = Separation::new(params);
    assert!(separator.prepare(
        &[&g7, &g8],
        vec![
            (7, prepare_incidence(&g7).unwrap()),
            (8, prepare_incidence(&g8).unwrap())
        ]
    ));
    assert!(call["neighbours"]
        .as_array()
        .unwrap()
        .iter()
        .all(|x| x.as_array().unwrap().is_empty()));
    let columns: Vec<_> = call["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|column| Solution {
            graph_id: number(&column["graph"]),
            enumerated: number(&column["enumerated"]) != 0,
            arc_ids: column["arc_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect(),
        })
        .collect();
    let input = Input {
        solutions: columns.iter().collect(),
        values: call["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["value"].as_f64().unwrap())
            .collect(),
        cut_memory_type: number(&call["memory_type"]),
        solutions_are_set_sequences: number(&call["set_sequences"]) != 0,
        separation_phase: number(&call["phase"]),
        active_cuts: vec![],
    };
    let cuts = separator.separate(&input).unwrap();
    let native = call["cuts"].as_array().unwrap();
    assert_eq!(cuts.len(), native.len());
    let mut registry = baldes::cuts::CutRegistry::default();
    let registered = integrated
        .separate_and_register(&mut registry, &input)
        .unwrap();
    assert_eq!(registered.len(), cuts.len());
    for ((index, actual), expected) in registered.iter().zip(&cuts) {
        assert_eq!(actual, expected);
        assert_eq!(registry.cuts[*index].id, *index as i32);
    }
    for (cut, expected) in cuts.iter().zip(native) {
        let item = &expected["descriptor"];
        let list = |name: &str| {
            item[name]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            (
                cut.kind,
                cut.num_rows,
                cut.class,
                cut.denominator,
                &cut.numerators,
                &cut.set_ids
            ),
            (
                number(&item["kind"]),
                number(&item["rows"]),
                number(&item["class"]),
                number(&item["denominator"]),
                &list("numerators"),
                &list("set_ids")
            ),
            "cut {}",
            cut.id
        );
        assert_eq!(cut.id, number(&item["id"]));
        assert_eq!(cut.rhs, item["rhs"].as_f64().unwrap());
        assert_eq!(cut.memory_type, number(&expected["memory_type"]));
        let memory: BTreeMap<i32, BTreeSet<i32>> = expected["memory"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| {
                (
                    number(&entry["graph"]),
                    entry["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(number)
                        .collect(),
                )
            })
            .collect();
        assert_eq!(cut.memory, memory, "cut {}", cut.id);
        let index = cut.id as usize;
        for column in &columns {
            let graph = if column.graph_id == 7 { &g7 } else { &g8 };
            let route: Vec<_> = column
                .arc_ids
                .iter()
                .map(|&id| {
                    let arc = graph.arcs.iter().find(|arc| arc.id == id).unwrap();
                    (arc.tail as usize, arc.head as usize, id)
                })
                .collect();
            for enumerated in [false, true] {
                let solution = Solution {
                    graph_id: column.graph_id,
                    enumerated,
                    arc_ids: column.arc_ids.clone(),
                };
                assert_eq!(
                    f64::from(
                        baldes::cuts::coefficient(
                            &registry.cuts[index],
                            column.graph_id,
                            &route,
                            enumerated
                        )
                        .unwrap()
                    ),
                    separator.coefficient(&solution, cut)
                );
            }
        }
    }
}

#[test]
fn rank_one_coefficients_match_two_graph_reference_captures() {
    check_two_graph_coefficient_captures(
        include_str!("../fixtures/cuts/rank_one_two_graph_coefficient_native.json"),
        false,
        false,
        4000,
    );
}

#[test]
fn rank_one_arc_incidence_coefficients_match_two_graph_reference_captures() {
    check_two_graph_coefficient_captures(
        include_str!("../fixtures/cuts/rank_one_two_graph_arc_coefficient_native.json"),
        true,
        false,
        850,
    );
}

#[test]
fn rank_one_partial_arc_incidence_coefficients_match_two_graph_reference_captures() {
    check_two_graph_coefficient_captures(
        include_str!("../fixtures/cuts/rank_one_two_graph_arc_subset_coefficient_native.json"),
        true,
        true,
        1150,
    );
}

fn check_two_graph_coefficient_captures(
    capture: &str,
    arc_incidence: bool,
    partial_arcs: bool,
    expected_count: usize,
) {
    use baldes::cuts::separation::{
        prepare_incidence, ArcData, GraphData, SeparatedCut, Separation, Solution, VertexData,
    };
    use baldes::cuts::{coefficient, CutClass, MemoryKind, RankOneCut};
    use std::collections::{BTreeMap, BTreeSet};

    let capture: serde_json::Value = serde_json::from_str(capture).unwrap();
    assert_eq!(
        capture["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let integer = |value: &serde_json::Value| value.as_i64().unwrap() as i32;
    let graph = |id, arcs: &str| GraphData {
        id,
        source: 0,
        sink: 6,
        elem_set_count: 5,
        packing_set_count: 5,
        covering_set_count: 5,
        vertices: (0..7)
            .map(|vertex| {
                let sets = if (1..=5).contains(&vertex) && (!arc_incidence || vertex != 1) {
                    vec![vertex - 1]
                } else {
                    vec![]
                };
                let present = if id == 8 && vertex == 5 {
                    vec![]
                } else {
                    sets.clone()
                };
                VertexData {
                    id: vertex,
                    elem_sets: sets,
                    packing_sets: present.clone(),
                    covering_sets: present,
                }
            })
            .collect(),
        arcs: capture[arcs]
            .as_array()
            .unwrap()
            .iter()
            .map(|arc| ArcData {
                id: integer(&arc["id"]),
                tail: integer(&arc["tail"]),
                head: integer(&arc["head"]),
                elem_sets: if arc_incidence
                    && integer(&arc["head"]) == 1
                    && (!partial_arcs || ![4, 5].contains(&integer(&arc["tail"])))
                {
                    vec![0]
                } else {
                    vec![]
                },
                packing_sets: if arc_incidence
                    && integer(&arc["head"]) == 1
                    && (!partial_arcs || ![4, 5].contains(&integer(&arc["tail"])))
                {
                    vec![0]
                } else {
                    vec![]
                },
                covering_sets: if arc_incidence
                    && integer(&arc["head"]) == 1
                    && (!partial_arcs || ![4, 5].contains(&integer(&arc["tail"])))
                {
                    vec![0]
                } else {
                    vec![]
                },
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let (g7, g8) = (graph(7, "arcs7"), graph(8, "arcs8"));
    let mut separator = Separation::default();
    assert!(separator.prepare(
        &[&g7, &g8],
        vec![
            (7, prepare_incidence(&g7).unwrap()),
            (8, prepare_incidence(&g8).unwrap())
        ]
    ));
    let mut cuts = Vec::new();
    for item in capture["cuts"].as_array().unwrap() {
        let list = |name: &str| {
            item[name]
                .as_array()
                .unwrap()
                .iter()
                .map(integer)
                .collect::<Vec<_>>()
        };
        let memory: BTreeMap<_, BTreeSet<_>> = item["memory"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| {
                (
                    integer(&entry["graph"]),
                    entry["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(integer)
                        .collect(),
                )
            })
            .collect();
        let kind = integer(&item["kind"]);
        let numerators = list("numerators");
        let set_ids = list("set_ids");
        let memory_type = integer(&item["memory_type"]);
        let separated = SeparatedCut {
            kind,
            num_rows: integer(&item["rows"]),
            denominator: integer(&item["denominator"]),
            numerators: numerators.clone(),
            set_ids: set_ids.clone(),
            memory_type,
            memory: memory.clone(),
            ..Default::default()
        };
        let support = |graph| {
            let mut values: Vec<_> = set_ids
                .iter()
                .zip(&numerators)
                .filter_map(|(&set, &weight)| {
                    ((graph != 8 || set != 4) && (!partial_arcs || set != 0))
                        .then_some(((set + 1) as usize, if kind == 1 { -weight } else { weight }))
                })
                .collect();
            values.sort_unstable();
            values
        };
        let graph_memories = memory
            .iter()
            .map(|(&graph, values)| {
                let mut bits = vec![false; 7];
                if memory_type == 0 {
                    for &v in values {
                        bits[v as usize] = true;
                    }
                }
                (graph, bits)
            })
            .collect();
        let graph_arc_weights = if partial_arcs {
            set_ids
                .iter()
                .position(|&id| id == 0)
                .map_or_else(BTreeMap::new, |row| {
                    let weight = numerators[row].abs();
                    [&g7, &g8]
                        .into_iter()
                        .map(|graph| {
                            (
                                graph.id,
                                graph
                                    .arcs
                                    .iter()
                                    .filter(|arc| arc.packing_sets.contains(&0))
                                    .map(|arc| (arc.id, weight))
                                    .collect(),
                            )
                        })
                        .collect()
                })
        } else {
            BTreeMap::new()
        };
        let registered = RankOneCut {
            weights: vec![0; 7],
            graph_support: BTreeMap::from([(7, support(7)), (8, support(8))]),
            graph_pricing_support: BTreeMap::new(),
            general_kpath_support: None,
            memory: vec![],
            graph_dependent_memory: true,
            graph_memories,
            memory_kind: if memory_type == 1 {
                MemoryKind::Arc
            } else {
                MemoryKind::Vertex
            },
            arc_memory: BTreeSet::new(),
            graph_arc_memories: memory,
            arc_weights: BTreeMap::new(),
            graph_arc_weights,
            visit_once: false,
            type2: false,
            denominator: integer(&item["denominator"]).abs(),
            dual: 0.,
            class: if kind == 1 {
                CutClass::Covering
            } else {
                CutClass::Packing
            },
            rhs: 0,
            kind,
            id: 0,
            ordered_support: None,
        };
        cuts.push((separated, registered));
    }
    let mut checked = 0;
    let mut shared_arc_map_mismatches = 0;
    for record in capture["records"].as_array().unwrap() {
        let graph_id = integer(&record["graph"]);
        let graph = if graph_id == 7 { &g7 } else { &g8 };
        let arcs: Vec<i32> = record["arcs"]
            .as_array()
            .unwrap()
            .iter()
            .map(integer)
            .collect();
        let route: Vec<_> = arcs
            .iter()
            .map(|&id| {
                let arc = graph.arcs.iter().find(|arc| arc.id == id).unwrap();
                (arc.tail as usize, arc.head as usize, id)
            })
            .collect();
        let (separated, registered) = &cuts[record["cut"].as_u64().unwrap() as usize];
        let enumerated = integer(&record["enumerated"]) != 0;
        if partial_arcs && graph_id == 8 && !registered.graph_arc_weights.is_empty() {
            let mut shared = registered.clone();
            let weight = *registered.graph_arc_weights[&7].values().next().unwrap();
            shared.graph_arc_weights.clear();
            shared.arc_weights = g7
                .arcs
                .iter()
                .filter(|arc| arc.packing_sets.contains(&0))
                .map(|arc| (arc.id, weight))
                .collect();
            if f64::from(coefficient(&shared, graph_id, &route, enumerated).unwrap())
                != record["coefficient"].as_f64().unwrap()
            {
                shared_arc_map_mismatches += 1;
            }
        }
        let check = |enumerated, expected: f64| {
            assert_eq!(
                separator.coefficient(
                    &Solution {
                        graph_id,
                        enumerated,
                        arc_ids: arcs.clone()
                    },
                    separated
                ),
                expected
            );
            assert_eq!(
                f64::from(coefficient(registered, graph_id, &route, enumerated).unwrap()),
                expected
            );
        };
        check(enumerated, record["coefficient"].as_f64().unwrap());
        checked += 1;
        if let Some(expected) = record.get("other_coefficient") {
            check(!enumerated, expected.as_f64().unwrap());
            checked += 1;
        }
    }
    assert_eq!(checked, expected_count);
    if partial_arcs {
        assert!(shared_arc_map_mismatches > 0);
        eprintln!("shared graph-7 arc map mismatches on graph 8: {shared_arc_map_mismatches}");
    }
}

#[test]
fn mixed_arc_and_vertex_sets_match_reference_coefficient_captures() {
    check_mixed_incidence_coefficient_captures(
        include_str!("../fixtures/cuts/rank_one_two_graph_mixed_incidence_coefficient_native.json"),
        2056,
    );
}

#[test]
fn complete_mixed_arc_and_vertex_sets_match_reference_coefficient_captures() {
    check_mixed_incidence_coefficient_captures(
        include_str!("../fixtures/cuts/rank_one_two_graph_mixed_complete_coefficient_native.json"),
        2056,
    );
}

#[test]
fn mixed_arc_and_vertex_sets_match_controlled_weight_captures() {
    check_mixed_incidence_coefficient_captures(
        include_str!("../fixtures/cuts/rank_one_two_graph_mixed_weight_controlled_native.json"),
        180,
    );
}

fn check_mixed_incidence_coefficient_captures(capture: &str, expected_count: usize) {
    use baldes::cuts::separation::{
        prepare_incidence, ArcData, GraphData, SeparatedCut, Separation, Solution, VertexData,
    };
    use baldes::cuts::{coefficient, CutRegistry};
    use std::collections::{BTreeMap, BTreeSet};
    let capture: serde_json::Value = serde_json::from_str(capture).unwrap();
    assert_eq!(
        capture["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let number = |value: &serde_json::Value| value.as_i64().unwrap() as i32;
    let complete_incidence = capture["complete_incidence"].as_bool().unwrap_or(false);
    let make_graph = |id, arcs: &str| GraphData {
        id,
        source: 0,
        sink: 7,
        elem_set_count: 6,
        packing_set_count: 6,
        covering_set_count: 6,
        vertices: (0..8)
            .map(|vertex| {
                let sets = if vertex == 1 {
                    vec![1]
                } else if (3..=6).contains(&vertex) {
                    vec![vertex - 1]
                } else {
                    vec![]
                };
                let rows = if !complete_incidence && id == 8 && vertex == 6 {
                    vec![]
                } else {
                    sets.clone()
                };
                VertexData {
                    id: vertex,
                    elem_sets: sets,
                    packing_sets: rows.clone(),
                    covering_sets: rows,
                }
            })
            .collect(),
        arcs: capture[arcs]
            .as_array()
            .unwrap()
            .iter()
            .map(|arc| {
                let tail = number(&arc["tail"]);
                let head = number(&arc["head"]);
                let mut sets = vec![];
                if head == 1 && tail != 4 && tail != 5 {
                    sets.push(0);
                }
                ArcData {
                    id: number(&arc["id"]),
                    tail,
                    head,
                    elem_sets: sets.clone(),
                    packing_sets: sets.clone(),
                    covering_sets: sets,
                    ..Default::default()
                }
            })
            .collect(),
        ..Default::default()
    };
    let (g7, g8) = (make_graph(7, "arcs7"), make_graph(8, "arcs8"));
    let mut separator = Separation::default();
    assert!(separator.prepare(
        &[&g7, &g8],
        vec![
            (7, prepare_incidence(&g7).unwrap()),
            (8, prepare_incidence(&g8).unwrap())
        ]
    ));
    let mut cuts = vec![];
    for item in capture["cuts"].as_array().unwrap() {
        let list = |name: &str| {
            item[name]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect::<Vec<_>>()
        };
        let kind = number(&item["kind"]);
        let sets = list("set_ids");
        let numerators = list("numerators");
        let memory: BTreeMap<_, BTreeSet<_>> = item["memory"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| {
                (
                    number(&entry["graph"]),
                    entry["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(number)
                        .collect(),
                )
            })
            .collect();
        let separated = SeparatedCut {
            kind,
            num_rows: number(&item["rows"]),
            denominator: number(&item["denominator"]),
            numerators: numerators.clone(),
            set_ids: sets.clone(),
            memory_type: number(&item["memory_type"]),
            memory: memory.clone(),
            ..Default::default()
        };
        let mut registry = CutRegistry::default();
        let index = registry
            .register_separated_cut(&[&g7, &g8], &separated)
            .unwrap();
        let registered = registry.cuts.remove(index);
        cuts.push((separated, registered));
    }
    let mut checked = 0;
    let mut priced = BTreeSet::new();
    for record in capture["records"].as_array().unwrap() {
        let graph_id = number(&record["graph"]);
        let graph = if graph_id == 7 { &g7 } else { &g8 };
        let arcs: Vec<_> = record["arcs"]
            .as_array()
            .unwrap()
            .iter()
            .map(number)
            .collect();
        let route: Vec<_> = arcs
            .iter()
            .map(|&id| {
                let arc = graph.arcs.iter().find(|arc| arc.id == id).unwrap();
                (arc.tail as usize, arc.head as usize, id)
            })
            .collect();
        let (separated, registered) = &cuts[record["cut"].as_u64().unwrap() as usize];
        let incidence = separator.incidence(graph_id);
        let actual_sets = if separated.kind == 0 {
            &incidence.arc_packing_sets
        } else {
            &incidence.arc_covering_sets
        };
        for (&arc, expected_sets) in arcs.iter().zip(record["sets"].as_array().unwrap()) {
            let expected: Vec<_> = expected_sets
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect();
            assert_eq!(
                actual_sets[arc as usize], expected,
                "graph {graph_id}, arc {arc}"
            );
        }
        let flag = number(&record["enumerated"]) != 0;
        if expected_count == 180
            && !flag
            && record["coefficient"].as_f64().unwrap() > 0.
            && priced.insert((separated.kind, graph_id))
        {
            for direction in [0., 1., 2.] {
                let mut solver = Solver::<2>::new(8, 1).unwrap();
                solver.graph.graph_id = graph_id;
                solver.graph.vertices[0].elements.clear();
                solver.graph.vertices[7].elements.clear();
                for vertex in 0..8 {
                    solver.graph.set_bounds(vertex, 0, 0., 10.).unwrap();
                }
                for (index, &(tail, head, id)) in route.iter().enumerate() {
                    solver
                        .graph
                        .add_arc_with_id(id, tail, head, if index == 0 { -10. } else { 0. })
                        .unwrap();
                    solver.graph.arc_mut(id).unwrap().consumption[0] = 1.;
                }
                let mut cut = registered.clone();
                cut.dual = -1.;
                for weights in cut.graph_arc_weights.values_mut() {
                    weights.retain(|id, _| arcs.contains(id));
                }
                for memory in cut.graph_arc_memories.values_mut() {
                    memory.retain(|id| arcs.contains(id));
                }
                solver.cuts.cuts.push(cut);
                solver
                    .options
                    .set("BALDESuseBidirectionalSearch", direction)
                    .unwrap();
                let actual = solver.price(10, false).unwrap().minimum_reduced_cost;
                let expected = -10. + record["coefficient"].as_f64().unwrap();
                assert!(
                    (actual - expected).abs() < 1e-8,
                    "native controlled cut pricing: {record:?}; got {actual}, expected {expected}"
                );
                let enumerated = solver.enumerate(0.).unwrap();
                assert_eq!(enumerated.len(), 1);
                let full_expected = -10. + record["other_coefficient"].as_f64().unwrap();
                assert!((enumerated[0].cost - full_expected).abs() < 1e-8,
                "native controlled cut enumeration, direction {direction}: {record:?}; got {}, expected {full_expected}", enumerated[0].cost);
            }
        }
        let mut check = |enumerated, expected: f64| {
            assert_eq!(
                separator.coefficient(
                    &Solution {
                        graph_id,
                        enumerated,
                        arc_ids: arcs.clone()
                    },
                    separated
                ),
                expected,
                "record {record:?}, cut {separated:?}, enumerated {enumerated}"
            );
            assert_eq!(
                f64::from(coefficient(registered, graph_id, &route, enumerated).unwrap()),
                expected,
                "record {record:?}, cut {separated:?}, enumerated {enumerated}"
            );
            checked += 1;
        };
        check(flag, record["coefficient"].as_f64().unwrap());
        if let Some(expected) = record.get("other_coefficient") {
            check(!flag, expected.as_f64().unwrap());
        }
    }
    assert_eq!(checked, expected_count);
    if expected_count == 180 {
        assert_eq!(priced.len(), 4);
    }
}

#[test]
fn capacity_and_complement_cuts_hold_for_every_small_capacity_feasible_partition() {
    use baldes::cuts::capacity::CapacitySeparator;
    // This fractional point covers each customer exactly once but uses 1.5
    // routes where capacity requires two. Enumerate all feasible partitions.
    let fractional = [(vec![1, 2], 0.5), (vec![1, 3], 0.5), (vec![2, 3], 0.5)];
    let partitions = [
        vec![vec![1], vec![2], vec![3]],
        vec![vec![1, 2], vec![3]],
        vec![vec![1, 3], vec![2]],
        vec![vec![2, 3], vec![1]],
    ];
    for complement in [false, true] {
        let mut sep = CapacitySeparator::new(vec![0., 1., 1., 1.], 2.).unwrap();
        sep.parameters.complement_form = complement;
        sep.parameters.use_two_path = false;
        let cuts = sep.separate(&fractional).unwrap();
        assert!(!cuts.is_empty());
        for cut in cuts {
            let lhs: f64 = fractional
                .iter()
                .map(|(r, x)| cut.coefficient(r) as f64 * x)
                .sum();
            let violation = if cut.greater_equal() {
                cut.rhs as f64 - lhs
            } else {
                lhs - cut.rhs as f64
            };
            assert!((violation - cut.violation).abs() < 1e-8);
            for partition in &partitions {
                // Test both customer orders of every two-customer route.
                for reverse in [false, true] {
                    let lhs: i32 = partition
                        .iter()
                        .map(|r| {
                            let mut r = r.clone();
                            if reverse {
                                r.reverse();
                            }
                            cut.coefficient(&r)
                        })
                        .sum();
                    assert!(
                        if cut.greater_equal() {
                            lhs >= cut.rhs
                        } else {
                            lhs <= cut.rhs
                        },
                        "{cut:?}, {partition:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn one_row_covering_set_switch_changes_the_selected_support_universe() {
    use baldes::cuts::separation::{
        prepare_incidence, ArcData, GraphData, Input, Separation, Solution, VertexData,
    };
    // The revisited vertex belongs only to the covering universe. The default
    // packing scan must not discover this cut through a packing-set alias.
    let graph = GraphData {
        id: 0,
        source: 0,
        sink: 2,
        elem_set_count: 1,
        packing_set_count: 1,
        covering_set_count: 1,
        vertices: vec![
            VertexData {
                id: 0,
                ..Default::default()
            },
            VertexData {
                id: 1,
                elem_sets: vec![0],
                covering_sets: vec![0],
                ..Default::default()
            },
            VertexData {
                id: 2,
                ..Default::default()
            },
        ],
        arcs: [(0, 1), (1, 1), (1, 2)]
            .into_iter()
            .enumerate()
            .map(|(i, (tail, head))| ArcData {
                id: i as i32,
                tail,
                head,
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let route = Solution {
        graph_id: 0,
        enumerated: false,
        arc_ids: vec![0, 1, 2],
    };
    for enabled in [false, true] {
        let mut sep = Separation::default();
        sep.params.max_num_rows = 1;
        sep.params.max_num_per_round = 10;
        sep.params.type_to_separate = 1;
        sep.params.use_cov_sets_for_one_row_packing = enabled;
        assert!(sep.prepare(&[&graph], vec![(0, prepare_incidence(&graph).unwrap())]));
        let cuts = sep
            .separate(&Input {
                solutions: vec![&route],
                values: vec![0.5],
                cut_memory_type: 0,
                solutions_are_set_sequences: false,
                separation_phase: 1,
                active_cuts: vec![],
            })
            .unwrap();
        assert_eq!(
            cuts.iter().any(|c| c.kind == 2 && c.num_rows == 1),
            enabled,
            "{cuts:?}"
        );
    }
}

#[test]
fn transformed_symmetry_preserves_routes_after_dual_changes_and_fixing() {
    for cut in [false, true] {
        for threshold in [-1., 2.5, 5.] {
            let mut s = solver(true, cut);
            s.graph.vertices[0].elements = vec![0];
            s.graph.vertices[4].elements = vec![4];
            s.graph.initial[0] = Some(3.);
            for v in 0..5 {
                s.graph.set_bounds(v, 0, 3., 7.).unwrap();
            }
            s.price_phase(0, 100, false).unwrap();
            // Head-assigned customer duals are conservative under reversal,
            // despite unequal individual reverse-arc reduced costs.
            let duals = [0., 2., 3., 1.5, 0.];
            for a in &mut s.graph.arcs {
                a.reduced_cost = a.cost - duals[a.head];
            }
            let expected = exhaustive(&s.graph, cut, threshold);
            let minimum = exhaustive(&s.graph, cut, 100.)
                .values()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let priced = s.price_phase(0, 100, false).unwrap();
            assert!(s.engine.as_ref().unwrap().consumption_joins);
            assert!((priced.minimum_reduced_cost - minimum).abs() < 1e-8);
            for route in priced.columns {
                let all = exhaustive(&s.graph, cut, 100.);
                assert!((route.cost - all[&route.arcs]).abs() < 1e-8);
            }
            assert_costs(&costs(s.enumerate(threshold).unwrap()), &expected);
            s.fix(threshold).unwrap();
            assert!(s.engine.as_ref().unwrap().consumption_joins);
            assert_costs(&costs(s.enumerate(threshold).unwrap()), &expected);
        }
    }
}

#[test]
fn mirrored_fixing_preserves_two_resource_frontiers() {
    for seed in 0..16 {
        let mut s = Solver::<2>::new(5, 2).unwrap();
        s.graph.elementary = true;
        s.enumeration_preserve_arc_paths = true;
        for v in 0..5 {
            s.graph.set_bounds(v, 0, 2., 8.).unwrap();
            s.graph.set_bounds(v, 1, 1., 10.).unwrap();
        }
        s.graph.initial = [Some(2.), Some(1.)].into();
        for u in 0..4 {
            for v in 1..5 {
                if u == v || (u == 0 && v == 4) {
                    continue;
                }
                let (a, b) = (if u == 0 { 0 } else { u }, if v == 4 { 0 } else { v });
                let key = a.min(b) * 7 + a.max(b) * 3 + seed;
                let id = s.graph.add_arc(u, v, (key % 9) as f64 / 4.).unwrap();
                let arc = s.graph.arc_mut(id).unwrap();
                arc.consumption = [1. + (key % 2) as f64, 1. + (key % 3) as f64].into();
                arc.reduced_cost -= [0., 1., 2., 0.5, 0.][v];
            }
        }
        let threshold = 1. + (seed % 4) as f64;
        let expected = exhaustive(&s.graph, false, threshold);
        s.fix(threshold).unwrap();
        assert!(s.engine.as_ref().unwrap().consumption_joins);
        assert_costs(&costs(s.enumerate(threshold).unwrap()), &expected);
    }
}

#[test]
fn capacity_candidate_ids_and_counters_persist_and_reset() {
    use baldes::cuts::capacity::{CapacitySeparator, CapacityStatistics};
    let mut sep = CapacitySeparator::new(vec![0., 1., 1., 1.], 2.).unwrap();
    let columns = vec![(vec![1, 2], 0.5), (vec![1, 3], 0.5), (vec![2, 3], 0.5)];
    assert_eq!(sep.statistics(), CapacityStatistics::default());
    for round in 0..3 {
        let cuts = sep.separate(&columns).unwrap();
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0].id, round);
        assert_eq!(sep.statistics().calls, round + 1);
        assert_eq!(sep.statistics().cuts_created, round + 1);
        assert_eq!(sep.statistics().columns, 3);
    }
    sep.reset();
    assert_eq!(sep.statistics(), CapacityStatistics::default());
    assert_eq!(sep.separate(&columns).unwrap()[0].id, 0);
}

fn coupled_solver(symmetric: bool, cut: bool) -> Solver<2> {
    use baldes::graph::coupled::CoupledResource;
    let mut s = solver(symmetric, cut);
    for arc in &mut s.graph.arcs {
        arc.reduced_cost = -2.;
    }
    let allowed = s
        .graph
        .arcs
        .iter()
        .map(|a| {
            let mut domain = 3;
            if a.tail == 1 || a.head == 1 {
                domain &= 1;
            }
            if a.tail == 2 || a.head == 2 {
                domain &= 2;
            }
            domain
        })
        .collect();
    let n = s.graph.arcs.len();
    s.set_coupled_resources(&[
        CoupledResource::Selection {
            initial: 3,
            allowed,
            vertices: vec![],
            relaxed: false,
        },
        CoupledResource::SimultaneousPickupDelivery {
            capacity: 6.,
            deliveries: vec![1.; n],
            pickups: vec![2.; n],
            tolerance: 0.,
        },
    ])
    .unwrap();
    s
}
fn coupled_exhaustive(g: &Graph<2>, cut: bool, cutoff: f64) -> BTreeMap<Vec<i32>, f64> {
    // Independent literal service recurrence, not the coupled kernel/replay.
    exhaustive(g, cut, cutoff)
        .into_iter()
        .filter(|(path, _)| {
            let (mut domain, mut deliveries, mut net, mut peak) = (3, 0., 0f64, 0f64);
            for &id in path {
                let a = g.arcs.iter().find(|a| a.id == id).unwrap();
                if a.tail == 1 || a.head == 1 {
                    domain &= 1;
                }
                if a.tail == 2 || a.head == 2 {
                    domain &= 2;
                }
                deliveries += 1.;
                net += 1.;
                peak = peak.max(net);
            }
            domain != 0 && deliveries + peak <= 6.
        })
        .collect()
}
#[test]
fn coupled_prefix_reversal_preserves_pricing_fixing_and_cut_bearing_enumeration() {
    for symmetric in [false, true] {
        for cut in [false, true] {
            for cutoff in [-5., -3., 100.] {
                let mut s = coupled_solver(symmetric, cut);
                let expected = coupled_exhaustive(&s.graph, cut, cutoff);
                let minimum = coupled_exhaustive(&s.graph, cut, 100.)
                    .values()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                let priced = s.price_phase(0, 100, false).unwrap();
                assert_eq!(s.engine.as_ref().unwrap().consumption_joins, symmetric);
                assert!((priced.minimum_reduced_cost - minimum).abs() < 1e-9);
                for route in &priced.columns {
                    let original: f64 = route
                        .arcs
                        .iter()
                        .map(|id| s.graph.arcs.iter().find(|a| a.id == *id).unwrap().cost)
                        .sum();
                    assert_eq!(route.original_cost, original);
                }
                s.fix(cutoff).unwrap();
                assert_costs(&costs(s.enumerate(cutoff).unwrap()), &expected);
            }
        }
    }
}
#[test]
fn coupled_prefix_reversal_falls_back_for_duals_and_isolates_forbidden_children() {
    let mut s = coupled_solver(true, false);
    s.price_phase(0, 100, false).unwrap();
    assert!(s.engine.as_ref().unwrap().consumption_joins);
    let mut child = s.fork_for_node();
    child.set_forbidden_arcs(vec![0]).unwrap();
    let priced = child.price_phase(0, 100, false).unwrap();
    assert!(!child.engine.as_ref().unwrap().consumption_joins);
    assert!(priced
        .columns
        .iter()
        .all(|r| !r.arcs.contains(&s.graph.arcs[0].id)));
    assert!(s.engine.as_ref().unwrap().consumption_joins);
    for (i, a) in s.graph.arcs.iter_mut().enumerate() {
        a.reduced_cost = (i * 7 % 13) as f64 - 6.;
    }
    let expected = coupled_exhaustive(&s.graph, false, 100.);
    let result = s.price_phase(0, 100, false).unwrap();
    assert!(!s.engine.as_ref().unwrap().consumption_joins);
    assert!(
        (result.minimum_reduced_cost - expected.values().copied().fold(f64::INFINITY, f64::min))
            .abs()
            < 1e-9
    );
    assert_costs(&costs(s.enumerate(100.).unwrap()), &expected);
}

#[test]
fn g13_coupled_retained_universe_survives_duals_fixing_and_sibling_branches() {
    for symmetric in [false, true] {
        for cut in [false, true] {
            for seed in 0..16 {
                let mut parent = coupled_solver(symmetric, false);
                let initial = coupled_exhaustive(&parent.graph, false, 100.);
                assert_costs(&costs(parent.enumerate(100.).unwrap()), &initial);
                if cut {
                    parent
                        .cuts
                        .add_strong_k_path_cut(5, &[1, 2], 1, -1.25)
                        .unwrap();
                }
                let original = coupled_exhaustive(&parent.graph, cut, 100.);
                assert_costs(
                    &costs(parent.inspect_enumeration(100.).unwrap().unwrap()),
                    &original,
                );
                let mut sibling = parent.fork_for_node();
                let mut child = parent.fork_for_node();
                let forbidden = seed % child.graph.arcs.len();
                let forbidden_id = child.graph.arcs[forbidden].id;
                child.set_forbidden_arcs(vec![forbidden]).unwrap();
                for (i, arc) in child.graph.arcs.iter_mut().enumerate() {
                    arc.reduced_cost = ((i * 7 + seed * 3) % 17) as f64 - 8.;
                }
                let expected: BTreeMap<_, _> = coupled_exhaustive(&child.graph, cut, 100.)
                    .into_iter()
                    .filter(|(path, _)| !path.contains(&forbidden_id))
                    .collect();
                // All paths were retained below 100 before the change; refreshing
                // coefficients/costs must neither resurrect the forbidden arc nor
                // discard a previously expensive physical ordering.
                let inspected = child.inspect_enumeration(100.).unwrap();
                if let Some(routes) = inspected {
                    assert_costs(&costs(routes), &expected);
                }
                let priced = child.price_phase(0, 100, false).unwrap();
                assert!(priced.exact);
                assert!(
                    (priced.minimum_reduced_cost
                        - expected.values().copied().fold(f64::INFINITY, f64::min))
                    .abs()
                        < 1e-8,
                    "seed={seed}, symmetric={symmetric}, cut={cut}"
                );
                child.fix(0.).unwrap();
                let negative = expected.into_iter().filter(|(_, c)| *c < 0.).collect();
                assert_costs(&costs(child.enumerate(0.).unwrap()), &negative);
                assert_costs(
                    &costs(sibling.inspect_enumeration(100.).unwrap().unwrap()),
                    &original,
                );
                assert_costs(
                    &costs(parent.inspect_enumeration(100.).unwrap().unwrap()),
                    &original,
                );
            }
        }
    }
}

#[test]
fn g13_bounded_cycles_jumps_and_terminal_equality_match_literal_walks() {
    use baldes::graph::{coupled::CoupledResource, ResourceKind};
    use baldes::pricing::phases::Phase;
    for mode in [0., 1., 2.] {
        for jumps in [false, true] {
            for equality in [false, true] {
                let mut s = Solver::<2>::new(3, 1).unwrap();
                for v in 0..3 {
                    s.graph.vertices[v].elements.clear();
                    s.graph.set_bounds(v, 0, 0., 4.).unwrap();
                }
                if equality {
                    s.graph.resources[0] = ResourceKind::Equality;
                }
                s.graph.set_bounds(2, 0, 4., 4.).unwrap();
                for (tail, head, cost) in [(0, 1, -1.), (1, 1, -2.), (1, 2, 0.)] {
                    let id = s.graph.add_arc(tail, head, cost).unwrap();
                    s.graph.arc_mut(id).unwrap().consumption[0] = 1.;
                }
                s.set_coupled_resources(&[
                    CoupledResource::Selection {
                        initial: 3,
                        allowed: vec![3; 3],
                        vertices: vec![],
                        relaxed: false,
                    },
                    CoupledResource::SimultaneousPickupDelivery {
                        capacity: 4.,
                        deliveries: vec![1.; 3],
                        pickups: vec![1.; 3],
                        tolerance: 0.,
                    },
                ])
                .unwrap();
                s.set_option("BALDESuseBidirectionalSearch", mode).unwrap();
                s.phases = vec![Phase {
                    check_ng: true,
                    jump_arcs: jumps,
                    labels_per_bucket: i64::MAX,
                }];
                s.enumeration_preserve_arc_paths = true;
                // Two terminal arcs plus k loops consume k+2 units. Equality
                // cannot wait to four; an ordinary window may wait at the sink.
                let expected: BTreeMap<_, _> = (0..=2)
                    .filter(|k| !equality || *k == 2)
                    .map(|k| {
                        let mut path = vec![0];
                        path.extend(std::iter::repeat_n(1, k));
                        path.push(2);
                        (path, -1. - 2. * k as f64)
                    })
                    .collect();
                let priced = s.price_phase(0, 100, false).unwrap();
                assert_eq!(
                    priced.exact, jumps,
                    "mode={mode} equality={equality}: {priced:?}"
                );
                assert_eq!(
                    priced.minimum_reduced_cost, -5.,
                    "mode={mode} jumps={jumps} equality={equality}"
                );
                assert_costs(&costs(s.enumerate(100.).unwrap()), &expected);
                s.fix(-2.).unwrap();
                let negative = expected.into_iter().filter(|(_, c)| *c < -2.).collect();
                assert_costs(&costs(s.enumerate(-2.).unwrap()), &negative);
            }
        }
    }
}

#[test]
fn symmetric_consumption_join_counts_nonzero_source_state_once() {
    for (initial, explicit) in [(2., false), (2., true), (3., false), (3., true)] {
        let mut symmetric = solver(true, false);
        for vertex in 0..5 {
            symmetric
                .graph
                .set_bounds(vertex, 0, initial, initial + 3.)
                .unwrap();
        }
        symmetric.graph.initial[0] = explicit.then_some(initial);
        let expected = exhaustive(&symmetric.graph, false, 100.);
        assert!(expected.keys().any(|path| path.len() == 3));
        assert!(!expected.keys().any(|path| path.len() > 3));
        let mut directed = solver(true, false);
        directed.graph = symmetric.graph.clone();
        directed
            .set_option("BALDESuseBidirectionalSearch", 0.)
            .unwrap();
        assert_costs(&costs(directed.enumerate(100.).unwrap()), &expected);
        assert_costs(&costs(symmetric.enumerate(100.).unwrap()), &expected);
        assert!(symmetric.engine.as_ref().unwrap().consumption_joins);
    }
}

#[test]
fn arc_memory_ng_strengthening_mirrors_on_symmetric_execution() {
    use baldes::graph::relaxation::{Controls, FractionalPath};
    for symmetric in [false, true] {
        let mut s = solver(symmetric, false);
        s.graph.elementary = false;
        s.price_phase(0, 100, false).unwrap();
        assert_eq!(s.engine.as_ref().unwrap().consumption_joins, symmetric);
        let id = |s: &Solver<2>, tail: usize, head: usize| {
            s.graph
                .arcs
                .iter()
                .find(|a| a.tail == tail && a.head == head)
                .unwrap()
                .id
        };
        let memory = |s: &Solver<2>, tail: usize, head: usize| {
            let a = id(s, tail, head);
            s.graph
                .arcs
                .iter()
                .find(|b| b.id == a)
                .unwrap()
                .memory
                .clone()
        };
        // 0 -> 1 -> 2 -> 3 -> 1 -> 4 revisits element 1 through 1-2-3-1.
        let path = [(0, 1), (1, 2), (2, 3), (3, 1), (1, 4)];
        let arcs = path.iter().map(|&(u, v)| id(&s, u, v)).collect();
        let controls = Controls {
            mode: 2,
            ..Controls::default()
        };
        let counts = s
            .strengthen_ng(&[FractionalPath { arcs, value: 0.5 }], &controls)
            .unwrap();
        assert_eq!(counts.columns_forbidden, 1);
        for (u, v) in [(1, 2), (2, 3), (3, 1)] {
            assert!(memory(&s, u, v).contains(&1));
            assert_eq!(memory(&s, v, u).contains(&1), symmetric, "{v}->{u}");
        }
        s.price_phase(0, 100, false).unwrap();
        assert_eq!(s.engine.as_ref().unwrap().consumption_joins, symmetric);
    }
}

/// Native symmetric joins test `f + b <= sink.ub`, omitting the source
/// initial value both halves start from (it drops this route when the source
/// lower bound is positive, root LP 4 instead of 2 in the reference probe).
/// BALDES joins at `f + b - init`, which keeps the feasible route.
#[test]
fn symmetric_join_subtracts_the_source_initial_value() {
    for init in [0., 2.] {
        let mut s = Solver::<2>::new(4, 1).unwrap();
        s.graph.vertices[0].elements.clear();
        s.graph.vertices[3].elements.clear();
        s.graph.elementary = true;
        for v in 0..4 {
            s.graph.set_bounds(v, 0, init, init + 8.5).unwrap();
        }
        for (u, v, cost, used) in [
            (0, 1, 1., 3.),
            (1, 3, 1., 3.),
            (0, 2, 1., 3.),
            (2, 3, 1., 3.),
            (1, 2, -5., 2.),
            (2, 1, -5., 2.),
        ] {
            let a = s.graph.add_arc(u, v, cost).unwrap();
            s.graph.arc_mut(a).unwrap().consumption[0] = used;
        }
        let result = s.price_phase(0, 100, false).unwrap();
        assert!(s.engine.as_ref().unwrap().consumption_joins, "init={init}");
        // Both two-customer routes arrive at init + 8 <= init + 8.5 but cross
        // the half-way point at their second customer, so only a join finds them.
        assert!(
            (result.minimum_reduced_cost + 3.).abs() < 1e-9,
            "init={init}"
        );
        assert!(
            result
                .columns
                .iter()
                .any(|r| r.arcs.len() == 3 && (r.cost + 3.).abs() < 1e-9),
            "init={init}: {:?}",
            result.columns
        );
    }
}
