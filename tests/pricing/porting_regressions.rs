//! Independent pricing and enumeration regression checks.
use baldes::{pricing::Solver, Graph, Route};
use std::collections::BTreeMap;

#[path = "mixed_reversal.rs"]
mod mixed_reversal;

fn solver(symmetric: bool, cut: bool) -> Solver<2> {
    let mut s = Solver::<2>::new(5, 1).unwrap();
    s.graph.edit().vertices[0].elements.clear();
    s.graph.edit().vertices[4].elements.clear();
    for v in 0..5 {
        s.graph.set_bounds(v, 0, 0., 4.).unwrap();
    }
    s.graph.edit().elementary = true;
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
        s.graph.edit().vertices[1].bounds[0].lower = 0.25;
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
    s.graph.edit().arcs[0].reduced_cost -= 3.;
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
    for (i, arc) in s.graph.edit().arcs.iter_mut().enumerate() {
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
        s.graph.edit().arcs[0].reduced_cost -= 7.;
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
    s.graph.edit().vertices[0].elements = vec![0];
    s.graph.edit().vertices[4].elements = vec![4];
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
    s.graph.edit().initial[0] = Some(3.);
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

#[test]
fn mixed_incidence_separation_matches_two_native_rounds() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_multiround_native.json"
    ));
}

#[test]
fn mixed_incidence_graph_dependent_memory_matches_three_native_rounds() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_multiround_dependent_native.json"
    ));
}

#[test]
fn mixed_incidence_vertex_memory_matches_three_native_rounds() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_multiround_vertex_native.json"
    ));
}

#[test]
fn mixed_incidence_automatic_memory_matches_native_two_pass_and_child_calls() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_eight_automatic_tree_native.json"
    ));
}

#[test]
fn mixed_incidence_automatic_checkpoint_matches_native_with_retained_separator() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_eight_automatic_checkpoint_native.json"
    ));
}

#[test]
fn mixed_incidence_eight_row_arc_memory_matches_native_rounds() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_eight_arc_native.json"
    ));
}

#[test]
fn mixed_incidence_eight_row_arc_memory_matches_native_child_inputs() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_eight_arc_tree_native.json"
    ));
}

#[test]
fn mixed_incidence_matches_native_cleanup_active_inputs() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_eight_cleanup_native.json"
    ));
}

#[test]
fn mixed_incidence_matches_native_cuts_generated_after_cleanup() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_ten_cleanup_native.json"
    ));
}

#[test]
fn mixed_incidence_matches_native_weighted_cleanup_rounds() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_weighted_cleanup_native.json"
    ));
}

#[test]
fn mixed_incidence_siblings_match_with_shared_separator_history() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_ryan_foster_siblings_native.json"
    ));
}

#[test]
fn mixed_incidence_strong_branching_matches_native_separation_inputs() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_strong_trials_native.json"
    ));
}

#[test]
fn native_ryan_foster_checks_arc_defined_set_membership() {
    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/ryan_foster_arc_set_satisfies_native.json"
    ))
    .unwrap();
    let graph: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_weighted_cleanup_native.json"
    ))
    .unwrap();
    assert_eq!(
        capture["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    assert_eq!(capture["sets"], serde_json::json!([0, 2]));
    let cases = capture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 256);
    for case in cases {
        let graph_id = case["graph"].as_i64().unwrap();
        let arcs = graph[if graph_id == 7 { "arcs7" } else { "arcs8" }]
            .as_array()
            .unwrap();
        let mut first = false;
        let mut second = false;
        for id in case["arcs"].as_array().unwrap() {
            let arc = &arcs[id.as_u64().unwrap() as usize];
            let tail = arc["tail"].as_i64().unwrap();
            let head = arc["head"].as_i64().unwrap();
            first |= head == 1 && tail != 4 && tail != 5;
            second |= head == 3;
        }
        assert_eq!(
            case["native_count"].as_u64().unwrap(),
            u64::from(first) + u64::from(second)
        );
        let expected = if case["together"].as_bool().unwrap() {
            first == second
        } else {
            !(first && second)
        };
        assert_eq!(case["accepted"].as_bool().unwrap(), expected, "{case}");
    }
    // Both routes visit vertex 1, but only the first enters it on an arc in set 0.
    for (arcs, accepted) in [(vec![0, 1], false), (vec![27, 29, 1], true)] {
        assert!(cases.iter().any(|case| {
            case["graph"] == 7
                && case["together"] == true
                && case["arcs"] == serde_json::json!(arcs)
                && case["accepted"] == accepted
        }));
    }
}

#[test]
fn arc_set_branch_labels_match_native_route_decisions() {
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;
    use baldes::pricing::restrictions::{ArcSetConstraint, RyanFosterResources};
    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/ryan_foster_arc_set_satisfies_native.json"
    ))
    .unwrap();
    let graphs: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_weighted_cleanup_native.json"
    ))
    .unwrap();
    for case in capture["cases"].as_array().unwrap() {
        let source = graphs[if case["graph"] == 7 { "arcs7" } else { "arcs8" }]
            .as_array()
            .unwrap();
        let mut graph = baldes::graph::Graph::<2>::new(10, 0).unwrap();
        let mut first = vec![];
        let mut second = vec![];
        for value in case["arcs"].as_array().unwrap() {
            let id = value.as_i64().unwrap() as i32;
            let arc = &source[id as usize];
            let tail = arc["tail"].as_u64().unwrap() as usize;
            let head = arc["head"].as_u64().unwrap() as usize;
            graph.add_arc_with_id(id, tail, head, -1.).unwrap();
            if head == 1 && tail != 4 && tail != 5 {
                first.push(id);
            }
            if head == 3 {
                second.push(id);
            }
        }
        let branch = ArcSetConstraint {
            first,
            second,
            together: case["together"].as_bool().unwrap(),
        };
        for bidirectional in [false, true] {
            let mut engine = LabelingEngine::prepare(
                &graph,
                EngineControls {
                    bidirectional,
                    ..Default::default()
                },
                Phase {
                    check_ng: true,
                    jump_arcs: true,
                    labels_per_bucket: i64::MAX,
                },
            )
            .unwrap();
            engine.branching =
                RyanFosterResources::prepare_arc_sets(&graph, &[], &[branch.clone()]).unwrap();
            let result = engine.solve(&graph, &baldes::Controls::default()).unwrap();
            assert_eq!(
                !result.columns.is_empty(),
                case["accepted"].as_bool().unwrap(),
                "bidirectional={bidirectional} {case}"
            );
        }
        let mut solver = baldes::pricing::Solver::<2>::new(10, 0).unwrap();
        solver.graph = graph;
        assert!(
            !solver.enumerate(1.).unwrap().is_empty(),
            "unrestricted {case}"
        );
        solver.set_arc_set_branches(vec![branch]).unwrap();
        let priced = solver.price_phase(0, 150, false).unwrap();
        assert_eq!(
            !priced.columns.is_empty(),
            case["accepted"].as_bool().unwrap()
        );
        let enumerated = solver.enumerate(1.).unwrap();
        assert_eq!(
            !enumerated.is_empty(),
            case["accepted"].as_bool().unwrap(),
            "{case}"
        );
    }
}

#[test]
fn overlapping_arc_set_events_are_atomic() {
    use baldes::pricing::restrictions::{ArcSetConstraint, RyanFosterResources};
    let mut graph = baldes::graph::Graph::<2>::new(3, 0).unwrap();
    graph.add_arc_with_id(17, 0, 1, 0.).unwrap();
    let constraint = |together| ArcSetConstraint {
        first: vec![17, 17],
        second: vec![17],
        together,
    };
    let apart = RyanFosterResources::prepare_arc_sets(&graph, &[], &[constraint(false)]).unwrap();
    let mut state = None;
    assert!(!apart.traverse_arc(&mut state, 0));
    assert!(state.is_none());
    let together = RyanFosterResources::prepare_arc_sets(&graph, &[], &[constraint(true)]).unwrap();
    assert!(together.traverse_arc(&mut state, 0));
    assert!(together.accepts(&state));
    let saved = state.clone();
    assert!(!together.traverse_arc(&mut state, 0));
    assert_eq!(state, saved);
}

#[test]
fn mixed_incidence_preserves_covering_vertex_cost_missing_from_native() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_signed_pricing_native.json"
    ));
}

#[test]
fn exact_root_pricing_preserves_raw_costs_when_native_rounding_ties_routes() {
    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/rounded_exact_root_pricing_native.json"
    ))
    .unwrap();
    let topology: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_eight_vertex_native.json"
    ))
    .unwrap();
    let raw: Vec<f64> = capture["raw_arc_costs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let mut solver = Solver::<2>::new(10, 1).unwrap();
    solver.graph.edit().graph_id = 7;
    for vertex in 0..10 {
        solver.graph.set_bounds(vertex, 0, 0., 2.).unwrap();
    }
    solver
        .graph
        .configure_ng(
            8,
            (0..10)
                .map(|v| {
                    if (1..=8).contains(&v) {
                        vec![v - 1]
                    } else {
                        vec![]
                    }
                })
                .collect(),
            vec![(0..8).collect(); 10],
        )
        .unwrap();
    for (index, arc) in topology["arcs7"].as_array().unwrap().iter().enumerate() {
        let head = arc["head"].as_u64().unwrap() as usize;
        let id = solver
            .graph
            .add_arc(arc["tail"].as_u64().unwrap() as usize, head, 0.)
            .unwrap();
        let arc = solver.graph.arc_mut(id).unwrap();
        arc.consumption[0] = if head == 9 { 0. } else { 1. };
        arc.variables = vec![(index, 1.)];
    }
    solver.maximum_variable_id = Some(raw.len() - 1);
    solver.cache_variable_costs(&raw).unwrap();
    let recorded = capture["recorded_costs"].as_array().unwrap();
    assert_eq!(recorded.len(), 64);
    for record in recorded {
        let arcs = record["arcs"].as_array().unwrap();
        let cost: f64 = arcs
            .iter()
            .map(|id| solver.graph.arcs[id.as_u64().unwrap() as usize].reduced_cost)
            .sum();
        assert_eq!(cost.to_bits(), record["cost"].as_f64().unwrap().to_bits());
    }
    let first = vec![18, 25, 55];
    let second = vec![18, 23, 37];
    let position = |arcs: &[i32]| {
        capture["returned_arcs"]
            .as_array()
            .unwrap()
            .iter()
            .position(|path| *path == serde_json::json!(arcs))
            .unwrap()
    };
    assert!(position(&first) < position(&second));
    let cached = solver.price_phase(0, 1000, false).unwrap();
    let cached_position = |arcs: &[i32]| {
        cached
            .columns
            .iter()
            .position(|route| route.arcs == arcs)
            .unwrap()
    };
    assert!(cached_position(&first) < cached_position(&second));
    let native: Vec<Vec<i32>> = capture["returned_arcs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| serde_json::from_value(v.clone()).unwrap())
        .collect();
    let cached_arcs: Vec<_> = cached.columns.iter().map(|r| r.arcs.clone()).collect();
    assert_eq!(cached_arcs, native);
    let exact = solver.price_functor_phase(0, 1000).unwrap();
    assert!(exact.exact);
    assert_eq!(exact.columns[0].arcs, vec![54, 62, 64]);
    assert_eq!(native[0], vec![27, 35, 64]);
    let cost = |arcs: &[i32]| exact.columns.iter().find(|r| r.arcs == arcs).unwrap().cost;
    assert!(cost(&exact.columns[0].arcs) < cost(&native[0]));
    assert!(cost(&second) < cost(&first));
    let mut reordered: Vec<_> = exact.columns.iter().collect();
    reordered.sort_by(|a, b| {
        let cached_cost = |route: &Route| -> f64 {
            route
                .arcs
                .iter()
                .map(|&id| solver.graph.arcs[id as usize].reduced_cost)
                .sum()
        };
        cached_cost(a).total_cmp(&cached_cost(b))
    });
    assert_eq!(reordered.len(), native.len());
    // A stable post-sort cannot recreate candidate IDs assigned by the cached
    // search. Ties retain their raw-search order instead.
    assert_eq!(
        reordered
            .iter()
            .zip(&native)
            .filter(|(actual, expected)| actual.arcs != **expected)
            .count(),
        32
    );
    let raw_by_path: BTreeMap<_, _> = exact
        .columns
        .iter()
        .map(|route| (route.arcs.clone(), route))
        .collect();
    assert_eq!(raw_by_path.len(), exact.columns.len());
    let ranked: Vec<_> = cached
        .columns
        .iter()
        .filter_map(|route| raw_by_path.get(&route.arcs).copied())
        .collect();
    assert_eq!(ranked.len(), exact.columns.len());
    assert_eq!(
        ranked.iter().map(|route| &route.arcs).collect::<Vec<_>>(),
        native.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        ranked
            .iter()
            .map(|route| route.cost)
            .min_by(f64::total_cmp)
            .unwrap()
            .to_bits(),
        exact.minimum_reduced_cost.to_bits()
    );
    for maximum in [1, 2, 8, 16, 32, 64] {
        let cached = solver.price_phase(0, maximum, false).unwrap();
        let exact = solver.price_functor_phase(0, maximum).unwrap();
        let raw_by_path: BTreeMap<_, _> = exact
            .columns
            .iter()
            .map(|route| (route.arcs.clone(), route))
            .collect();
        let ranked: Vec<_> = cached
            .columns
            .iter()
            .filter_map(|route| raw_by_path.get(&route.arcs).copied())
            .collect();
        assert_eq!(ranked.len(), exact.columns.len(), "maximum={maximum}");
        assert_eq!(
            ranked.iter().map(|route| &route.arcs).collect::<Vec<_>>(),
            native.iter().take(maximum).collect::<Vec<_>>(),
            "maximum={maximum}"
        );
        assert!(
            ranked
                .iter()
                .any(|route| { route.cost.to_bits() == exact.minimum_reduced_cost.to_bits() }),
            "maximum={maximum}"
        );
    }
}

#[test]
fn mixed_incidence_eight_row_vertex_memory_matches_native_rounds() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_eight_vertex_native.json"
    ));
}

#[test]
fn mixed_incidence_automatic_memory_matches_native_root_rounds() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_automatic_native.json"
    ));
}

#[test]
fn mixed_incidence_full_memory_matches_two_native_rounds() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_multiround_full_native.json"
    ));
}

#[test]
fn mixed_incidence_inherits_native_active_cut_memory() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_mixed_active_memory_native.json"
    ));
}

#[test]
fn mixed_incidence_excludes_inactive_cut_memory_like_native() {
    for capture in [
        include_str!("../fixtures/cuts/rank_one_two_graph_mixed_inactive_memory_native.json"),
        include_str!("../fixtures/cuts/rank_one_two_graph_mixed_selective_memory_native.json"),
    ] {
        check_mixed_incidence_separation_capture(capture);
    }
}

#[test]
fn rank_one_separation_matches_native_arc_head_overlap() {
    for capture in [
        include_str!("../fixtures/cuts/rank_one_two_graph_overlap_packing_native.json"),
        include_str!("../fixtures/cuts/rank_one_two_graph_overlap_covering_native.json"),
    ] {
        check_mixed_incidence_separation_capture(capture);
    }
}

#[test]
fn empty_packing_set_sequence_columns_match_native() {
    use baldes::cuts::separation::ColumnInfo;

    let cases: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/rank_one_set_sequence_zero_native.json"
    ))
    .unwrap();
    for case in cases.as_array().unwrap() {
        let sequence: Vec<_> = case["sequence"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect();
        let column = ColumnInfo::from_set_sequence(0, 0.5, &sequence, 0);
        assert!(column.packing_set_count.is_empty());
        assert_eq!(column.elementary, case["elementary"].as_i64().unwrap() != 0);
        assert_eq!(column.enumerated, case["enumerated"].as_i64().unwrap() != 0);
    }
}

#[test]
fn rank_one_separation_matches_native_multiple_sets_on_one_arc() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_multiple_sets_native.json"
    ));
}

#[test]
fn rank_one_separation_matches_native_multiple_sets_on_one_vertex() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_multiple_vertex_sets_native.json"
    ));
}

#[test]
fn rank_one_separation_matches_native_duplicate_memberships() {
    for capture in [
        include_str!("../fixtures/cuts/rank_one_two_graph_duplicate_vertex_sets_native.json"),
        include_str!("../fixtures/cuts/rank_one_two_graph_duplicate_arc_sets_native.json"),
    ] {
        check_mixed_incidence_separation_capture(capture);
    }
}

#[test]
fn rank_one_separation_matches_native_unsorted_vertex_registration() {
    check_mixed_incidence_separation_capture(include_str!(
        "../fixtures/cuts/rank_one_two_graph_unsorted_vertex_sets_native.json"
    ));
}

#[test]
fn rank_one_multiple_set_coefficients_and_pricing_match_native_control() {
    for capture in [
        include_str!("../fixtures/cuts/rank_one_two_graph_four_sets_native.json"),
        include_str!("../fixtures/cuts/rank_one_two_graph_four_sets_feasible_native.json"),
    ] {
        check_mixed_incidence_separation_capture(capture);
    }
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
    let num_sets = capture["num_sets"].as_i64().unwrap_or(6) as i32;
    let make_graph = |id, arcs: &str| GraphData {
        id,
        source: 0,
        sink: num_sets + 1,
        elem_set_count: num_sets,
        packing_set_count: num_sets,
        covering_set_count: num_sets,
        vertices: (0..num_sets + 2)
            .map(|vertex| {
                let mut sets = if vertex == 1 {
                    if capture["multi_vertex_set"].is_number() {
                        vec![1, number(&capture["multi_vertex_set"])]
                    } else {
                        vec![1]
                    }
                } else if vertex == 2 && capture["multi_vertex_set"].is_number() {
                    vec![0]
                } else if (3..=num_sets).contains(&vertex) {
                    vec![vertex - 1]
                } else {
                    vec![]
                };
                if vertex == 1 && capture["duplicate_vertex_set"] == true {
                    sets.insert(1, 1);
                }
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
                if head == 1 && tail != 4 && tail != 5 && !capture["multi_vertex_set"].is_number() {
                    sets.push(0);
                }
                let mut elem = sets.clone();
                let mut packing = sets.clone();
                let mut covering = sets;
                if tail == 4 && head == 1 {
                    match capture["overlap_kind"].as_str() {
                        Some("packing") => {
                            elem.push(1);
                            packing.push(1);
                        }
                        Some("covering") => {
                            elem.push(1);
                            covering.push(1);
                        }
                        _ => {}
                    }
                }
                if tail == 2 && head == 1 {
                    if capture["duplicate_arc_set"] == true {
                        elem.push(0);
                        packing.push(0);
                        covering.push(0);
                    }
                    if let Some(set) = capture["multi_arc_set"].as_i64() {
                        elem.push(set as i32);
                        packing.push(set as i32);
                        covering.push(set as i32);
                    }
                }
                if tail == 3 && head == 1 {
                    if let Some(extra) = capture["multi_arc_sets"].as_array() {
                        for set in extra {
                            let set = number(set);
                            elem.push(set);
                            packing.push(set);
                            covering.push(set);
                        }
                    }
                }
                ArcData {
                    id: number(&arc["id"]),
                    tail,
                    head,
                    elem_sets: elem,
                    packing_sets: packing,
                    covering_sets: covering,
                    ..Default::default()
                }
            })
            .collect(),
        ..Default::default()
    };
    let (g7, g8) = (make_graph(7, "arcs7"), make_graph(8, "arcs8"));
    if complete_incidence && capture["skip_master_entry_validation"] != true {
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
    let params = Parameters {
        max_num_rows: 5,
        max_num_per_round: capture["max_num_per_round"].as_i64().unwrap_or(20) as i32,
        memory_type: capture["parameter_memory_type"].as_i64().unwrap_or(1) as i32,
        arc_memory_type: 2,
        sp_dependent_memory: capture["sp_dependent_memory"].as_bool().unwrap_or(false),
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
    if let Some(controls) = capture["quad_controls"].as_array() {
        use baldes::cuts::separation::SeparatedCut;
        for control in controls {
            let kind = number(&control["kind"]);
            let memory: BTreeMap<_, _> = [&g7, &g8]
                .into_iter()
                .map(|graph| (graph.id, graph.arcs.iter().map(|arc| arc.id).collect()))
                .collect();
            let cut = SeparatedCut {
                kind,
                num_rows: 5,
                denominator: number(&control["denominator"]),
                numerators: control["numerators"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(number)
                    .collect(),
                set_ids: control["ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(number)
                    .collect(),
                memory_type: 1,
                memory,
                ..Default::default()
            };
            let arc_ids: Vec<_> = control["arcs"]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect();
            let route: Vec<_> = arc_ids
                .iter()
                .map(|&id| {
                    let arc = g7.arcs.iter().find(|arc| arc.id == id).unwrap();
                    (arc.tail as usize, arc.head as usize, id)
                })
                .collect();
            let mut local = baldes::cuts::CutRegistry::default();
            let index = local.register_separated_cut(&[&g7, &g8], &cut).unwrap();
            let expected = control["coefficient"].as_f64().unwrap();
            for enumerated in [false, true] {
                let solution = Solution {
                    graph_id: 7,
                    enumerated,
                    arc_ids: arc_ids.clone(),
                };
                assert_eq!(separator.coefficient(&solution, &cut), expected);
                assert_eq!(
                    f64::from(
                        baldes::cuts::coefficient(&local.cuts[index], 7, &route, enumerated)
                            .unwrap()
                    ),
                    expected
                );
            }
            let mut solver = baldes::pricing::Solver::<2>::new(8, 0).unwrap();
            solver.graph.edit().graph_id = 7;
            for &(tail, head, id) in &route {
                solver.graph.add_arc_with_id(id, tail, head, 0.).unwrap();
            }
            local.cuts[index].dual = if kind == 0 { -1. } else { 1. };
            solver.cuts = local;
            let priced = solver.price_phase(0, 10, false).unwrap();
            assert_eq!(priced.columns.len(), 1);
            let charge = if kind == 0 { expected } else { -expected };
            assert_eq!(priced.columns[0].cost, charge);
            let enumerated = solver.enumerate(100.).unwrap();
            assert_eq!(enumerated.len(), 1);
            assert_eq!(enumerated[0].cost, charge);
        }
    }
    let calls: Vec<_> = match capture["calls"].as_array() {
        Some(calls) => calls.iter().collect(),
        None => vec![&capture["call"]],
    };
    let mut registry = baldes::cuts::CutRegistry::default();
    let mut prior_cuts: Vec<baldes::cuts::separation::SeparatedCut> = Vec::new();
    let branch_parents = capture["branch_parents"].as_object();
    let mut snapshots: Vec<(
        Separation,
        baldes::cuts::IncidenceCutSeparator,
        baldes::cuts::CutRegistry,
        Vec<baldes::cuts::separation::SeparatedCut>,
    )> = vec![];
    for (round, call) in calls.into_iter().enumerate() {
        if capture["registry_reset_round"].as_u64() == Some(round as u64) {
            registry = baldes::cuts::CutRegistry::default();
            prior_cuts.clear();
        }
        if let Some(parent) = branch_parents
            .and_then(|parents| parents.get(&round.to_string()))
            .and_then(serde_json::Value::as_u64)
        {
            let (parent_separator, parent_integrated, parent_registry, parent_cuts) = snapshots
                .get(parent as usize)
                .expect("missing captured parent state");
            if capture["shared_separator_history"] != true {
                separator = parent_separator.clone();
                integrated = parent_integrated.clone();
            }
            registry = parent_registry.clone();
            prior_cuts = parent_cuts.clone();
        }
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
        let active_cuts = call["active_cuts"]
            .as_array()
            .map(|active| {
                active
                    .iter()
                    .map(|entry| {
                        let descriptor = &entry["cut"];
                        let cut = prior_cuts
                            .iter()
                            .find(|cut| cut.id == number(&descriptor["id"]))
                            .expect("captured active cut was generated in a prior round");
                        let expected_sets: Vec<_> = descriptor["set_ids"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(number)
                            .collect();
                        let expected_memory: BTreeMap<i32, BTreeSet<i32>> = entry["memory"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|item| {
                                (
                                    number(&item["graph"]),
                                    item["values"]
                                        .as_array()
                                        .unwrap()
                                        .iter()
                                        .map(number)
                                        .collect(),
                                )
                            })
                            .collect();
                        assert_eq!(cut.set_ids, expected_sets);
                        let mut active = cut.clone();
                        if capture["controlled_active_memory"].as_bool() == Some(true)
                            && number(&call["phase"]) == 5
                            && active.id == 2
                        {
                            assert!(active.memory.get_mut(&7).unwrap().insert(0));
                        }
                        assert_eq!(active.memory, expected_memory);
                        assert_eq!(active.memory_type, number(&entry["memory_type"]));
                        active
                    })
                    .collect()
            })
            .unwrap_or_default();
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
            active_cuts,
        };
        if capture["retire_absent_active"].as_bool() == Some(true) {
            for (index, cut) in prior_cuts.iter().enumerate() {
                if !input.active_cuts.iter().any(|active| active.id == cut.id) {
                    registry.retire(index);
                }
            }
            if let Some(expected) = capture["retired_counts"][round.to_string()].as_u64() {
                assert_eq!(registry.retired.len(), expected as usize);
            }
        }
        let mut sibling = integrated.clone();
        let mut sibling_registry = registry.clone();
        let cuts = separator.separate(&input).unwrap();
        let native = call["cuts"].as_array().unwrap();
        assert_eq!(cuts.len(), native.len());
        let registered = integrated
            .separate_and_register(&mut registry, &input)
            .unwrap();
        assert_eq!(registered.len(), cuts.len());
        let sibling_cuts = sibling
            .separate_and_register(&mut sibling_registry, &input)
            .unwrap();
        assert_eq!(sibling_cuts, registered);
        assert_eq!(sibling_registry.cuts, registry.cuts);
        if let Some((index, _)) = sibling_cuts.first() {
            sibling_registry.cuts[*index].dual += 1.;
            assert_ne!(
                sibling_registry.cuts[*index].dual,
                registry.cuts[*index].dual
            );
        }
        for ((index, actual), expected) in registered.iter().zip(&cuts) {
            assert_eq!(actual, expected);
            assert_eq!(registry.cuts[*index].id, *index as i32);
        }
        if let Some(regenerated) = capture["regenerated_cuts"].as_array() {
            for entry in regenerated {
                let [expected_round, old, new] = entry.as_array().unwrap().as_slice() else {
                    panic!("invalid regenerated cut capture");
                };
                if number(expected_round) != round as i32 {
                    continue;
                }
                let old_id = number(old);
                let new_id = number(new);
                let old_cut = prior_cuts.iter().find(|cut| cut.id == old_id).unwrap();
                let new_cut = cuts.iter().find(|cut| cut.id == new_id).unwrap();
                let mut expected = old_cut.clone();
                expected.id = new_id;
                assert_eq!(*new_cut, expected);
                let old_index = registry
                    .cuts
                    .iter()
                    .position(|cut| cut.id == old_id)
                    .unwrap();
                let new_index = registry
                    .cuts
                    .iter()
                    .position(|cut| cut.id == new_id)
                    .unwrap();
                assert!(registry.retired.contains(&old_index));
                assert!(!registry.retired.contains(&new_index));
                let mut sibling = registry.clone();
                sibling.retire(new_index);
                assert!(sibling.retired.contains(&new_index));
                assert!(!registry.retired.contains(&new_index));
                assert_eq!(registry.cuts[new_index], sibling.cuts[new_index]);
            }
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
            let index = registered
                .iter()
                .find(|(_, descriptor)| descriptor.id == cut.id)
                .unwrap()
                .0;
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
        prior_cuts.extend(cuts);
        if branch_parents.is_some() {
            snapshots.push((
                separator.clone(),
                integrated.clone(),
                registry.clone(),
                prior_cuts.clone(),
            ));
        }
    }
    if let Some(prices) = capture["pricing_control"].as_array() {
        // Give retired rows stale nonzero duals to verify they cannot reenter
        // the pricing input after the captured cleanup sequence.
        for cut in &mut registry.cuts {
            cut.dual = -1.;
        }
        for entry in prices {
            let id = number(&entry["id"]) as usize;
            let price = entry["price"].as_f64().unwrap();
            registry.cuts[id].dual = -price;
        }
        let cached = capture["cached_cut_control"].as_array().unwrap();
        registry.pricing_order = cached
            .iter()
            .map(|entry| number(&entry["id"]) as usize)
            .collect();
        registry.round_duals = true;
        let expected: Vec<_> = cached
            .iter()
            .map(|entry| entry["price"].as_f64().unwrap())
            .collect();
        for graph in [&g7, &g8] {
            let mut pricing_graph =
                baldes::graph::Graph::<2>::new(num_sets as usize + 2, 0).unwrap();
            pricing_graph.edit().graph_id = graph.id;
            for arc in &graph.arcs {
                pricing_graph
                    .add_arc(arc.tail as usize, arc.head as usize, 0.)
                    .unwrap();
            }
            let (limited, kpath, full, full_kpath) =
                baldes::cuts::state::build_inputs(&registry, &pricing_graph);
            assert!(kpath.is_empty() && full_kpath.is_empty());
            for (input, entry) in limited.iter().zip(cached) {
                let id = number(&entry["id"]);
                let captured = capture["calls"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|call| call["cuts"].as_array().unwrap())
                    .find(|cut| number(&cut["descriptor"]["id"]) == id)
                    .unwrap();
                let descriptor = &captured["descriptor"];
                assert_eq!(input.denominator, number(&descriptor["denominator"]));
                let memory = captured["memory"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|memory| number(&memory["graph"]) == graph.id)
                    .unwrap()["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(number)
                    .collect();
                assert_eq!(input.memory_arcs, memory);
            }
            for actual in [
                limited.iter().map(|input| input.dual).collect::<Vec<_>>(),
                full.iter().map(|input| input.dual).collect::<Vec<_>>(),
            ] {
                assert_eq!(actual.len(), expected.len());
                for (actual, price) in actual.iter().zip(&expected) {
                    assert_eq!(actual.to_bits(), price.to_bits());
                }
            }
            if let Some(case) = capture["route_controls"]
                .as_array()
                .and_then(|cases| cases.iter().find(|case| number(&case["graph"]) == graph.id))
            {
                let mut solver = Solver::<2>::new(num_sets as usize + 2, 1).unwrap();
                solver.graph.edit().graph_id = graph.id;
                for vertex in 0..num_sets as usize + 2 {
                    solver.graph.set_bounds(vertex, 0, 0., 2.).unwrap();
                }
                let membership = (0..num_sets as usize + 2)
                    .map(|v| {
                        if (1..=num_sets as usize).contains(&v) {
                            vec![v - 1]
                        } else {
                            vec![]
                        }
                    })
                    .collect();
                solver
                    .graph
                    .configure_ng(
                        num_sets as usize,
                        membership,
                        vec![(0..num_sets as usize).collect(); num_sets as usize + 2],
                    )
                    .unwrap();
                for (arc, cost) in graph
                    .arcs
                    .iter()
                    .zip(case["cached_arc_costs"].as_array().unwrap())
                {
                    let id = solver
                        .graph
                        .add_arc(arc.tail as usize, arc.head as usize, 0.)
                        .unwrap();
                    let edge = solver.graph.arc_mut(id).unwrap();
                    edge.consumption[0] = if arc.head as usize == num_sets as usize + 1 {
                        0.
                    } else {
                        1.
                    };
                    edge.reduced_cost = cost.as_f64().unwrap();
                }
                solver.cuts = registry.clone();
                let result = solver.price_phase(0, 1000, false).unwrap();
                assert!(result.exact);
                if capture["covering_vertex_divergence"] == true {
                    assert_eq!(result.columns.len(), 64);
                    let mut different = 0;
                    for route in &result.columns {
                        let path: Vec<_> = route
                            .arcs
                            .iter()
                            .map(|&id| {
                                let arc = &graph.arcs[id as usize];
                                (arc.tail as usize, arc.head as usize, id)
                            })
                            .collect();
                        let mut expected: f64 = route
                            .arcs
                            .iter()
                            .map(|&id| case["cached_arc_costs"][id as usize].as_f64().unwrap())
                            .sum();
                        let mut native_expected = expected;
                        for entry in cached {
                            let id = number(&entry["id"]) as usize;
                            let charge = entry["price"].as_f64().unwrap()
                                * f64::from(
                                    baldes::cuts::coefficient(
                                        &registry.cuts[id],
                                        graph.id,
                                        &path,
                                        false,
                                    )
                                    .unwrap(),
                                );
                            expected += charge;
                            if id != 3 {
                                native_expected += charge;
                            }
                        }
                        assert!((route.cost - expected).abs() < 1e-8);
                        let native = case["recorded_costs"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|native| native["arcs"] == serde_json::json!(route.arcs))
                            .unwrap();
                        assert!((native["cost"].as_f64().unwrap() - native_expected).abs() < 1e-8);
                        if (expected - native_expected).abs() > 1e-8 {
                            different += 1;
                        }
                    }
                    assert!(different > 0);
                    if graph.id == 7 {
                        let arcs = [45, 53, 64];
                        let actual = result.columns.iter().find(|r| r.arcs == arcs).unwrap();
                        let native = case["recorded_costs"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|r| r["arcs"] == serde_json::json!(arcs))
                            .unwrap();
                        let path: Vec<_> = arcs
                            .iter()
                            .map(|&id| {
                                let arc = &graph.arcs[id as usize];
                                (arc.tail as usize, arc.head as usize, id)
                            })
                            .collect();
                        for id in [0, 3] {
                            assert_eq!(
                                baldes::cuts::coefficient(
                                    &registry.cuts[id],
                                    graph.id,
                                    &path,
                                    false
                                )
                                .unwrap(),
                                case["witness_native_coefficients"][id.to_string()]
                                    .as_f64()
                                    .unwrap() as i32
                            );
                        }
                        let arc_cost: f64 = arcs
                            .iter()
                            .map(|&id| case["cached_arc_costs"][id as usize].as_f64().unwrap())
                            .sum();
                        assert_eq!(arc_cost, 0.0);
                        assert_eq!(actual.cost, 0.0);
                        assert_eq!(native["cost"].as_f64().unwrap(), 0.1);
                        for (phase, key) in [
                            (1, "phase_one_recorded_costs"),
                            (2, "phase_two_recorded_costs"),
                        ] {
                            let priced = solver.price_phase(phase, 1000, false).unwrap();
                            let route = priced.columns.iter().find(|r| r.arcs == arcs).unwrap();
                            let native = case[key]
                                .as_array()
                                .unwrap()
                                .iter()
                                .find(|r| r["arcs"] == serde_json::json!(arcs))
                                .unwrap();
                            assert_eq!(route.cost, 0.0);
                            assert_eq!(native["cost"].as_f64().unwrap(), 0.1);
                        }
                    } else {
                        let arcs = [26, 18, 7];
                        for (phase, key) in [
                            (0, "recorded_costs"),
                            (1, "phase_one_recorded_costs"),
                            (2, "phase_two_recorded_costs"),
                        ] {
                            let priced = solver.price_phase(phase, 1000, false).unwrap();
                            let route = priced.columns.iter().find(|r| r.arcs == arcs).unwrap();
                            let native = case[key]
                                .as_array()
                                .unwrap()
                                .iter()
                                .find(|r| r["arcs"] == serde_json::json!(arcs))
                                .unwrap();
                            assert!((route.cost - 0.03).abs() < 1e-12);
                            assert!((native["cost"].as_f64().unwrap() - 0.13).abs() < 1e-12);
                        }
                    }
                    continue;
                }
                for native in case["recorded_costs"].as_array().unwrap() {
                    let arcs: Vec<_> = native["arcs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(number)
                        .collect();
                    let route = result
                        .columns
                        .iter()
                        .find(|route| route.arcs == arcs)
                        .unwrap();
                    assert_eq!(
                        route.cost.to_bits(),
                        native["cost"].as_f64().unwrap().to_bits(),
                        "graph {} arcs {:?}",
                        graph.id,
                        arcs
                    );
                }
                let mut actual_costs: Vec<_> =
                    result.columns.iter().map(|route| route.cost).collect();
                let mut native_costs: Vec<_> = case["candidate_costs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|cost| cost.as_f64().unwrap())
                    .collect();
                actual_costs.sort_by(f64::total_cmp);
                native_costs.sort_by(f64::total_cmp);
                assert_eq!(actual_costs.len(), native_costs.len());
                for (actual, native) in actual_costs.iter().zip(&native_costs) {
                    assert!((actual - native).abs() < 1e-8);
                }
                for route in &result.columns {
                    let mut expected_cost = route
                        .arcs
                        .iter()
                        .map(|&id| case["cached_arc_costs"][id as usize].as_f64().unwrap())
                        .sum::<f64>();
                    let path: Vec<_> = route
                        .arcs
                        .iter()
                        .map(|&id| {
                            let arc = &graph.arcs[id as usize];
                            (arc.tail as usize, arc.head as usize, id)
                        })
                        .collect();
                    for entry in prices {
                        let id = number(&entry["id"]) as usize;
                        expected_cost += entry["price"].as_f64().unwrap()
                            * f64::from(
                                baldes::cuts::coefficient(
                                    &registry.cuts[id],
                                    graph.id,
                                    &path,
                                    false,
                                )
                                .unwrap(),
                            );
                    }
                    assert!((route.cost - expected_cost).abs() < 1e-8);
                }
                assert!(
                    (result.minimum_reduced_cost - case["bound"].as_f64().unwrap()).abs() < 1e-8,
                    "graph {}",
                    graph.id
                );
                let mut actual: Vec<_> = result
                    .columns
                    .iter()
                    .map(|route| route.arcs.clone())
                    .collect();
                let mut expected: Vec<Vec<i32>> = case["routes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|route| route.as_array().unwrap().iter().map(number).collect())
                    .collect();
                assert_eq!(actual, expected);
                for (route, native) in result
                    .columns
                    .iter()
                    .zip(case["recorded_costs"].as_array().unwrap())
                {
                    assert_eq!(
                        route.cost.to_bits(),
                        native["cost"].as_f64().unwrap().to_bits()
                    );
                }
                actual.sort();
                expected.sort();
                assert_eq!(expected.len(), 64);
                assert_eq!(actual, expected);
                let relaxed = solver.price_phase(1, 20, false).unwrap();
                assert!(!relaxed.exact);
                for native in case["phase_one_recorded_costs"].as_array().unwrap() {
                    let arcs: Vec<_> = native["arcs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(number)
                        .collect();
                    let route = relaxed
                        .columns
                        .iter()
                        .find(|route| route.arcs == arcs)
                        .unwrap();
                    assert_eq!(
                        route.cost.to_bits(),
                        native["cost"].as_f64().unwrap().to_bits()
                    );
                }
                assert!(
                    (relaxed.minimum_reduced_cost - case["phase_one_bound"].as_f64().unwrap())
                        .abs()
                        < 1e-8
                );
                let mut actual: Vec<_> = relaxed
                    .columns
                    .iter()
                    .map(|route| route.arcs.clone())
                    .collect();
                let mut expected: Vec<Vec<i32>> = case["phase_one_routes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|route| route.as_array().unwrap().iter().map(number).collect())
                    .collect();
                assert_eq!(actual, expected);
                actual.sort();
                expected.sort();
                assert_eq!(actual, expected);
                let restricted = solver.price_phase(2, 16, false).unwrap();
                assert!(!restricted.exact);
                for native in case["phase_two_recorded_costs"].as_array().unwrap() {
                    let arcs: Vec<_> = native["arcs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(number)
                        .collect();
                    let route = restricted
                        .columns
                        .iter()
                        .find(|route| route.arcs == arcs)
                        .unwrap();
                    assert_eq!(
                        route.cost.to_bits(),
                        native["cost"].as_f64().unwrap().to_bits()
                    );
                }
                assert!(
                    (restricted.minimum_reduced_cost - case["phase_two_bound"].as_f64().unwrap())
                        .abs()
                        < 1e-8
                );
                let mut actual: Vec<_> = restricted
                    .columns
                    .iter()
                    .map(|route| route.arcs.clone())
                    .collect();
                let mut expected: Vec<Vec<i32>> = case["phase_two_routes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|route| route.as_array().unwrap().iter().map(number).collect())
                    .collect();
                assert_eq!(actual, expected);
                actual.sort();
                expected.sort();
                assert_eq!(actual, expected);
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
                solver.graph.edit().graph_id = graph_id;
                solver.graph.edit().vertices[0].elements.clear();
                solver.graph.edit().vertices[7].elements.clear();
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
            s.graph.edit().vertices[0].elements = vec![0];
            s.graph.edit().vertices[4].elements = vec![4];
            s.graph.edit().initial[0] = Some(3.);
            for v in 0..5 {
                s.graph.set_bounds(v, 0, 3., 7.).unwrap();
            }
            s.price_phase(0, 100, false).unwrap();
            // Head-assigned customer duals are conservative under reversal,
            // despite unequal individual reverse-arc reduced costs.
            let duals = [0., 2., 3., 1.5, 0.];
            for a in &mut s.graph.edit().arcs {
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
        s.graph.edit().elementary = true;
        s.enumeration_preserve_arc_paths = true;
        for v in 0..5 {
            s.graph.set_bounds(v, 0, 2., 8.).unwrap();
            s.graph.set_bounds(v, 1, 1., 10.).unwrap();
        }
        s.graph.edit().initial = [Some(2.), Some(1.)].into();
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
    for arc in &mut s.graph.edit().arcs {
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
    for (i, a) in s.graph.edit().arcs.iter_mut().enumerate() {
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
                for (i, arc) in child.graph.edit().arcs.iter_mut().enumerate() {
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
                    s.graph.edit().vertices[v].elements.clear();
                    s.graph.set_bounds(v, 0, 0., 4.).unwrap();
                }
                if equality {
                    s.graph.edit().resources[0] = ResourceKind::Equality;
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
        symmetric.graph.edit().initial[0] = explicit.then_some(initial);
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
        s.graph.edit().elementary = false;
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
        s.graph.edit().vertices[0].elements.clear();
        s.graph.edit().vertices[3].elements.clear();
        s.graph.edit().elementary = true;
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

#[test]
fn rank_one_incidence_accepts_sets_on_terminal_arcs() {
    use baldes::cuts::separation::{prepare_incidence, ArcData, GraphData, Parameters, VertexData};
    let graph = GraphData {
        id: 7,
        source: 0,
        sink: 2,
        elem_set_count: 2,
        packing_set_count: 2,
        covering_set_count: 2,
        vertices: (0..3)
            .map(|id| VertexData {
                id,
                ..Default::default()
            })
            .collect(),
        arcs: [(0, 0, 1), (1, 1, 2)]
            .into_iter()
            .map(|(id, tail, head)| ArcData {
                id,
                tail,
                head,
                elem_sets: vec![id],
                packing_sets: vec![id],
                covering_sets: vec![id],
            })
            .collect(),
        ..Default::default()
    };
    let incidence = prepare_incidence(&graph).unwrap();
    assert_eq!(incidence.arc_packing_sets, [vec![0], vec![1]]);
    assert_eq!(incidence.arc_covering_sets, [vec![0], vec![1]]);
    let params = Parameters {
        max_num_rows: 3,
        max_num_per_round: 2,
        memory_type: 1,
        arc_memory_type: 2,
        neighbourhood_size: 16,
        violation_tolerance: 0.02,
        ..Default::default()
    };
    assert!(baldes::cuts::IncidenceCutSeparator::new(params, &[&graph]).is_ok());
}

#[test]
fn covering_vertex_separator_matches_native_across_rounds() {
    check_covering_vertex_rounds(
        include_str!("../fixtures/cuts/rank_one_covering_vertex_multiround_native.json"),
        1000,
        false,
    );
}

#[test]
fn covering_vertex_short_search_retains_native_random_state() {
    check_covering_vertex_rounds(
        include_str!("../fixtures/cuts/rank_one_covering_vertex_short_search_native.json"),
        1,
        false,
    );
}

#[test]
fn covering_vertex_two_search_steps_retain_native_random_state() {
    check_covering_vertex_rounds(
        include_str!("../fixtures/cuts/rank_one_covering_vertex_two_search_native.json"),
        2,
        false,
    );
}

#[test]
fn mixed_vertex_default_separator_matches_native_rounds() {
    check_covering_vertex_rounds(
        include_str!("../fixtures/cuts/rank_one_vertex_mixed_short_native.json"),
        1,
        true,
    );
}

fn check_covering_vertex_rounds(fixture: &str, iterations: i32, mixed: bool) {
    use baldes::cuts::{CandidateMode, CutClass, CutRegistry, FractionalColumn, MemoryType};
    use baldes::graph::Graph;
    let capture: serde_json::Value = serde_json::from_str(fixture).unwrap();
    assert_eq!(
        capture["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let number = |value: &serde_json::Value| value.as_i64().unwrap() as i32;
    let mut graph = Graph::<2>::new(11, 0).unwrap();
    graph.edit().graph_id = 7;
    let arcs = capture["arcs"].as_array().unwrap();
    for arc in arcs {
        graph
            .add_arc_with_id(
                number(&arc["id"]),
                number(&arc["tail"]) as usize,
                number(&arc["head"]) as usize,
                0.,
            )
            .unwrap();
    }
    let mut registry = CutRegistry::default();
    registry.parameters.graph_id = 7;
    registry.parameters.covering = true;
    registry.parameters.memory_type = MemoryType::Vertex;
    registry.parameters.max_cuts = 1;
    registry.parameters.max_rank = 5;
    registry.parameters.mode = CandidateMode::LocalSearch;
    registry.parameters.local_search_rounds = iterations;
    registry.parameters.violation_epsilon = 0.02;
    registry.covering_sets_present = mixed;
    let mut options = baldes::core::options::Options::default();
    for (name, value) in [
        ("BALDESrankOneCutsMaxNumRows", 5.),
        ("BALDESrankOneCutsMaxNumPerRound", 1.),
        ("BALDESrankOneCutsLSnumIterations", iterations as f64),
        ("BALDESrankOneCutsMemoryType", 2.),
    ] {
        options.set(name, value).unwrap();
    }
    for call in capture["calls"].as_array().unwrap() {
        registry.parameters.start_rank = number(&call["phase"]);
        registry.separation_phase = number(&call["phase"]);
        let columns: Vec<_> = call["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|column| {
                let ids: Vec<_> = column["arc_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(number)
                    .collect();
                let vertices = ids
                    .iter()
                    .take(ids.len().saturating_sub(1))
                    .map(|&id| number(&arcs[id as usize]["head"]) as usize)
                    .collect();
                FractionalColumn {
                    vertices,
                    arc_ids: Some(ids),
                    value: column["value"].as_f64().unwrap(),
                    graph_id: 7,
                    enumerated: number(&column["enumerated"]) != 0,
                }
            })
            .collect();
        let before = registry.cuts.len();
        let mut sibling = registry.clone();
        for target in [&mut registry, &mut sibling] {
            if mixed {
                target
                    .separate_default(&graph, &options, &(1..=9).collect::<Vec<_>>(), &columns)
                    .unwrap();
            } else {
                target
                    .separate(&graph, &(1..=9).collect::<Vec<_>>(), &columns)
                    .unwrap();
            }
        }
        assert_eq!(registry.cuts, sibling.cuts);
        let actual = &registry.cuts[before..];
        let expected = call["cuts"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (cut, expected) in actual.iter().zip(expected) {
            let descriptor = &expected["descriptor"];
            let support: Vec<_> = descriptor["set_ids"]
                .as_array()
                .unwrap()
                .iter()
                .zip(descriptor["numerators"].as_array().unwrap())
                .map(|(set, numerator)| (number(set) as usize + 1, number(numerator).abs()))
                .collect();
            assert_eq!(
                cut.class,
                if number(&descriptor["kind"]) == 1 {
                    CutClass::Covering
                } else {
                    CutClass::Packing
                }
            );
            assert_eq!(cut.kind, number(&descriptor["class"]));
            assert_eq!(cut.id, number(&descriptor["id"]));
            assert_eq!(cut.ordered_support.as_ref().unwrap(), &support);
            let memory: Vec<_> = cut.graph_memories[&7]
                .iter()
                .enumerate()
                .filter_map(|(vertex, &present)| present.then_some(vertex as i32))
                .collect();
            let expected_memory: Vec<_> = expected["memory"][0]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect();
            assert_eq!(memory, expected_memory);
            assert_eq!(cut.denominator, number(&descriptor["denominator"]).abs());
            assert_eq!(cut.rhs, descriptor["rhs"].as_f64().unwrap() as i32);
        }
    }
}
