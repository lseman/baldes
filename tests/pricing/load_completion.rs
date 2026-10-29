use baldes::labeling::engine::{EngineControls, LabelingEngine};
use baldes::pricing::{phases, Solver};
use baldes::{Controls, Graph, PricingResult};
use std::time::Duration;

fn network(loads: &[f64], capacity: f64, seed: usize) -> Graph<2> {
    // Match theta_cut_graph's guard for mean-demand sums. Without this,
    // decimal sums exactly at capacity can change the feasible path set
    // across forward and backward floating-point resource replay.
    let capacity = capacity * (1. + 1e-9) + 1e-9;
    let sink = loads.len() + 1;
    let mut g = Graph::new(sink + 1, 1).unwrap();
    for v in 0..=sink {
        g.set_bounds(v, 0, 0., capacity).unwrap();
    }
    for v in 1..sink {
        g.add_elementarity_set(&[v]).unwrap();
    }
    for v in 0..=sink {
        g.set_ng(v, &(0..loads.len()).collect::<Vec<_>>()).unwrap();
    }
    for from in 0..sink {
        for to in 1..=sink {
            if from == to || (from == 0 && to == sink) {
                continue;
            }
            let cost = ((from * 19 + to * 7 + seed * 11) % 23) as f64 - 10.;
            let id = g.add_arc(from, to, cost).unwrap();
            g.arc_mut(id).unwrap().consumption[0] = loads.get(to - 1).copied().unwrap_or(0.);
        }
    }
    g
}

fn run(g: &Graph<2>, enabled: bool, mode: i32, threshold: f64) -> PricingResult {
    // Use the production coordinator, including its forward replay fallback
    // when backward subtraction accepts a floating-point resource boundary.
    let mut solver = Solver::<2>::new(g.vertices.len(), 1).unwrap();
    solver.graph = g.clone();
    solver.controls.max_columns = 10_000;
    solver.controls.time_limit = Duration::from_secs(10);
    solver
        .set_option("BALDESuseBidirectionalSearch", mode as f64)
        .unwrap();
    solver
        .set_option(
            "BALDESloadCompletionResource",
            if enabled { 0. } else { -1. },
        )
        .unwrap();
    solver
        .set_option("BALDESloadCompletionThreshold", threshold)
        .unwrap();
    solver.price_phase(0, 10_000, false).unwrap()
}

fn elementary_minimum(g: &Graph<2>) -> f64 {
    fn visit(g: &Graph<2>, v: usize, visited: u64, load: f64, cost: f64, best: &mut f64) {
        if v == g.sink {
            *best = best.min(cost);
            return;
        }
        for arc in g.arcs.iter().filter(|a| a.tail == v) {
            let bit = 1 << arc.head;
            if visited & bit != 0 {
                continue;
            }
            if let Some(next) = baldes::labeling::extend_resources(g, arc, [load], true) {
                visit(
                    g,
                    arc.head,
                    visited | bit,
                    next[0],
                    cost + arc.reduced_cost,
                    best,
                );
            }
        }
    }
    let mut best = f64::INFINITY;
    visit(g, g.source, 1 << g.source, 0., 0., &mut best);
    best
}

#[test]
fn load_completion_preserves_negative_pricing_minima_in_both_directions() {
    let mut pruned = 0;
    for loads in [vec![1., 1., 1., 1.], vec![0.6, 0.7, 1.1, 0.3]] {
        for seed in 0..8 {
            let g = network(&loads, 2.4, seed);
            let oracle = elementary_minimum(&g);
            for mode in [0, 1, 2] {
                for threshold in [0., -3.] {
                    let result = run(&g, true, mode, threshold);
                    assert!(result.exact);
                    assert!(result.minimum_reduced_cost <= oracle + 1e-7);
                    if oracle < threshold - 1e-7 {
                        assert!(
                            (result.minimum_reduced_cost - oracle).abs() < 1e-7,
                            "seed={seed} mode={mode} threshold={threshold}"
                        );
                    }
                    pruned += result.statistics.completion_pruned;
                }
            }
        }
    }
    assert!(pruned > 0);
}

#[test]
fn positive_pricing_minimum_remains_a_conservative_bound_after_pruning() {
    let mut g = network(&[1., 1., 1.], 3., 0);
    for arc in &mut g.arcs {
        arc.reduced_cost = 2.;
    }
    let baseline = run(&g, false, 1, 0.);
    let bounded = run(&g, true, 1, 0.);
    assert!(baseline.minimum_reduced_cost > 0.);
    assert_eq!(bounded.minimum_reduced_cost, 0.);
    assert!(bounded.statistics.completion_pruned > 0);
    assert!(bounded.exact);
}

#[test]
fn load_completion_allows_ng_revisits_and_declines_unbounded_zero_load_walks() {
    for zero in [false, true] {
        let mut g = Graph::<2>::new(4, 1).unwrap();
        for v in 0..4 {
            g.set_bounds(v, 0, 0., 4.).unwrap();
        }
        for v in 1..3 {
            g.add_elementarity_set(&[v]).unwrap();
            g.set_ng(v, &[v - 1]).unwrap();
        }
        for (from, to, cost) in [(0, 1, 0.), (1, 2, -3.), (2, 1, -3.), (1, 3, 0.), (2, 3, 0.)] {
            let id = g.add_arc(from, to, cost).unwrap();
            g.arc_mut(id).unwrap().consumption[0] = if zero || to == 3 { 0. } else { 1. };
        }
        if zero {
            // A finite, fully elementary pricing domain with an unbounded
            // zero-load walk relaxation must simply decline this bound.
            for v in 0..4 {
                g.set_ng(v, &[0, 1]).unwrap();
            }
        }
        let a = run(&g, false, 1, 0.);
        let b = run(&g, true, 1, 0.);
        assert!((a.minimum_reduced_cost - b.minimum_reduced_cost).abs() < 1e-7);
        if zero {
            assert_eq!(b.statistics.completion_pruned, 0);
        } else {
            assert!(a
                .columns
                .iter()
                .any(|r| r.nodes.iter().filter(|&&v| v == 1).count() > 1));
        }
    }
}

#[test]
fn fixing_and_enumeration_rebuild_labels_pruned_by_load_completion() {
    let mut g = network(&[1., 1., 1.], 3., 0);
    for arc in &mut g.arcs {
        arc.cost = 1.;
        arc.reduced_cost = 1.;
    }
    let mut solver = Solver::<2>::new(g.vertices.len(), 1).unwrap();
    solver.graph = g;
    solver.enumeration_preserve_arc_paths = true;
    solver
        .set_option("BALDESloadCompletionResource", 0.)
        .unwrap();
    let result = solver.price_phase(0, 100, false).unwrap();
    assert!(result.statistics.completion_pruned > 0);
    let fixed = solver.run_fixing_enumeration(0, 0., 100.).unwrap();
    assert_eq!(fixed.state, 2);
    assert_eq!(fixed.routes, 15);
}

#[test]
fn packing_cut_penalties_and_asymmetric_arc_prices_keep_valid_completion_bounds() {
    use baldes::cuts::state::LimitedInput;
    use std::collections::{BTreeMap, BTreeSet};
    let mut seen_pruning = 0;
    for seed in 0..6 {
        let mut g = network(&[0.6, 0.8, 1.2, 0.5], 2.4, seed);
        // A positive directed arc charge models a Theta dual contribution;
        // other reduced arc prices already include coverage/branch prices.
        g.arcs
            .iter_mut()
            .find(|a| a.tail == 1 && a.head == 2)
            .unwrap()
            .reduced_cost += 4.7;
        for covering in [false, true] {
            let cut = LimitedInput {
                covering,
                type2: false,
                arc_memory: false,
                denominator: 2,
                dual: if covering { -2. } else { 2. },
                vertex_coefficient: vec![0, 1, 1, 1, 0, 0],
                arc_coefficient: BTreeMap::new(),
                memory_vertices: vec![0, 1, 2, 3, 4, 5],
                memory_arcs: BTreeSet::new(),
            };
            let mut minima = vec![];
            for enabled in [false, true] {
                let mut engine = LabelingEngine::prepare(
                    &g,
                    EngineControls {
                        use_symmetric_execution: false,
                        load_completion_resource: enabled.then_some(0),
                        ..Default::default()
                    },
                    phases::initialize(0, &[]).unwrap()[0],
                )
                .unwrap();
                assert!(engine.load_cuts(&g, vec![cut.clone()], vec![]));
                let result = engine
                    .solve(
                        &g,
                        &Controls {
                            max_columns: 10_000,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                minima.push(result.minimum_reduced_cost);
                if covering {
                    assert_eq!(result.statistics.completion_pruned, 0);
                }
                seen_pruning += result.statistics.completion_pruned;
            }
            assert!(minima[1] <= minima[0] + 1e-7);
            if minima[0] < -1e-7 {
                assert!((minima[1] - minima[0]).abs() < 1e-7);
            }
        }
    }
    assert!(seen_pruning > 0);
}
