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
