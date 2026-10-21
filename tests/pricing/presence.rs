use super::*;
use baldes::cuts::soft_conflicts::{Mode, PresenceRow, PricingInput};
use baldes::labeling::engine::{EngineControls, LabelingEngine};
use baldes::pricing::phases::Phase;
use std::collections::HashSet;

fn fixtures(symmetric: bool) -> (Graph<2>, Vec<Vec<usize>>) {
    let mut g = Graph::<2>::new(5, 1).unwrap();
    let mut events = vec![];
    if symmetric {
        for v in 1..4 {
            g.add_arc(0, v, 1.).unwrap();
            events.push(vec![v - 1]);
            g.add_arc(v, 4, 1.).unwrap();
            events.push(vec![]);
        }
        for a in 1..4 {
            for b in 1..4 {
                if a != b {
                    g.add_arc(a, b, 0.25).unwrap();
                    events.push(vec![b - 1]);
                }
            }
        }
    } else {
        for (a, b, c, e) in [
            (0, 1, 0.25, vec![0]),
            (0, 2, 0., vec![1]),
            (1, 3, 0., vec![1]),
            (2, 3, 0., vec![2]),
            (3, 4, 0., vec![]),
            (1, 2, 0.125, vec![2]),
            (2, 1, 0.125, vec![0]),
            (0, 1, 0.75, vec![0]),
        ] {
            g.add_arc(a, b, c).unwrap();
            events.push(e);
        }
    }
    for v in 0..5 {
        g.set_bounds(v, 0, 0., 6.).unwrap();
    }
    for a in &mut g.arcs {
        a.consumption[0] = 1.;
    }
    (g, events)
}
fn input(g: &Graph<2>, events: &[Vec<usize>], mode: Mode, dual: f64) -> Vec<PricingInput> {
    vec![
        PricingInput {
            row: PresenceRow::new(3, vec![(0, 1), (1, 2), (0, 1)], mode).unwrap(),
            arc_events: events.to_vec(),
            dual,
        },
        PricingInput {
            row: PresenceRow::new(1, vec![(0, 0)], Mode::AnyPair).unwrap(),
            arc_events: g
                .arcs
                .iter()
                .map(|a| if a.head == g.sink { vec![0] } else { vec![] })
                .collect(),
            dual: -0.2,
        },
    ]
}
fn cost(g: &Graph<2>, events: &[Vec<usize>], mode: Mode, dual: f64, route: &[usize]) -> f64 {
    let seen: HashSet<_> = route
        .iter()
        .flat_map(|&a| events[a].iter().copied())
        .collect();
    let count = [(0, 1), (1, 2), (0, 1)]
        .iter()
        .filter(|&&(a, b)| seen.contains(&a) && seen.contains(&b))
        .count();
    let coefficient = if mode == Mode::PairCount {
        count
    } else {
        usize::from(count > 0)
    };
    route.iter().map(|&a| g.arcs[a].reduced_cost).sum::<f64>() - dual * coefficient as f64 + 0.2
}
fn exhaustive(g: &Graph<2>, events: &[Vec<usize>], mode: Mode, dual: f64) -> f64 {
    fn paths(g: &Graph<2>, at: usize, path: &mut Vec<usize>, routes: &mut Vec<Vec<usize>>) {
        if at == g.sink {
            routes.push(path.clone());
            return;
        }
        if path.len() == 6 {
            return;
        }
        for (i, a) in g.arcs.iter().enumerate().filter(|(_, a)| a.tail == at) {
            path.push(i);
            paths(g, a.head, path, routes);
            path.pop();
        }
    }
    let mut routes = vec![];
    paths(g, g.source, &mut vec![], &mut routes);
    routes
        .iter()
        .map(|p| cost(g, events, mode, dual, p))
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn presence_engine_matches_independent_walks_and_replay() {
    for symmetric in [false, true] {
        let (g, events) = fixtures(symmetric);
        for mode in [Mode::PairCount, Mode::AnyPair] {
            for dual in [-0.625, 0., 1e-8, 0.7] {
                let expected = exhaustive(&g, &events, mode, dual);
                for bidirectional in [false, true] {
                    for filtering in [false, true] {
                        let mut engine = LabelingEngine::prepare(
                            &g,
                            EngineControls {
                                bidirectional,
                                filter_other_buckets: filtering,
                                ..Default::default()
                            },
                            Phase {
                                check_ng: true,
                                jump_arcs: true,
                                labels_per_bucket: i64::MAX,
                            },
                        )
                        .unwrap();
                        if symmetric && bidirectional {
                            assert!(engine.consumption_joins);
                        }
                        engine
                            .load_presence(&g, input(&g, &events, mode, dual))
                            .unwrap();
                        let result = engine.solve(&g, &Controls::default()).unwrap();
                        assert!((result.minimum_reduced_cost-expected).abs()<1e-10,
                            "sym={symmetric} mode={mode:?} dual={dual} bidir={bidirectional} filter={filtering}: {} != {expected}",result.minimum_reduced_cost);
                        if bidirectional {
                            assert!(!engine.consumption_joins);
                        }
                        assert!(!result.columns.is_empty());
                        for route in &result.columns {
                            let arcs: Vec<_> = route
                                .arcs
                                .iter()
                                .map(|id| g.arcs.iter().position(|a| a.id == *id).unwrap())
                                .collect();
                            assert!(
                                (route.cost - cost(&g, &events, mode, dual, &arcs)).abs() < 1e-12
                            );
                        }
                        // Both-direction completion and reduced-cost fixing use
                        // the same row preparation, including overlap rewards.
                        engine
                            .prepare_full_labels(&g, &Controls::default())
                            .unwrap();
                        engine.fix_bucket_arcs(&g, expected + 1e-4).unwrap();
                        let fixed = engine.solve(&g, &Controls::default()).unwrap();
                        assert!((fixed.minimum_reduced_cost - expected).abs() < 1e-10);
                    }
                }
            }
        }
    }
}

#[test]
fn presence_engine_loads_are_atomic_and_rank_preparation_retains_rows() {
    let (g, events) = fixtures(false);
    let mut engine = LabelingEngine::prepare(
        &g,
        EngineControls::default(),
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        },
    )
    .unwrap();
    engine
        .load_presence(&g, input(&g, &events, Mode::PairCount, 0.7))
        .unwrap();
    let expected = engine
        .solve(&g, &Controls::default())
        .unwrap()
        .minimum_reduced_cost;
    let mut invalid = input(&g, &events, Mode::PairCount, 0.7);
    invalid[0].arc_events.clear();
    assert!(engine.load_presence(&g, invalid).is_err());
    let mut overflow = input(&g, &events, Mode::PairCount, 0.7);
    overflow[0].dual = f64::MAX;
    assert!(engine.load_presence(&g, overflow).is_err());
    assert!(engine.load_cuts(&g, vec![], vec![]));
    assert_eq!(
        engine
            .solve(&g, &Controls::default())
            .unwrap()
            .minimum_reduced_cost,
        expected
    );
    engine
        .prepare_full_labels(&g, &Controls::default())
        .unwrap();
    let mut unknown = engine.forward.labels[0].clone();
    unknown.presence_state = None;
    assert!(engine.completion_exists(&g, true, &unknown, -1e6, false));
    assert!(engine.completion_exists(&g, true, &unknown, -1e6, true));
    engine.load_presence(&g, vec![]).unwrap();
    assert!(
        engine
            .solve(&g, &Controls::default())
            .unwrap()
            .minimum_reduced_cost
            > expected
    );
}

#[test]
fn presence_engine_cannot_publish_an_uncharged_enumeration_universe() {
    let (g, events) = fixtures(false);
    let mut engine = LabelingEngine::prepare(
        &g,
        EngineControls::default(),
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        },
    )
    .unwrap();
    engine
        .load_presence(&g, input(&g, &events, Mode::AnyPair, 0.7))
        .unwrap();
    let mut solver = baldes::pricing::Solver::<2>::new(5, 1).unwrap();
    solver.graph = g;
    solver.engine = Some(engine);
    assert!(solver
        .enumerate(10.)
        .unwrap_err()
        .message
        .contains("enumeration state integration"));
    assert!(solver
        .inspect_enumeration(10.)
        .unwrap_err()
        .message
        .contains("enumeration state integration"));
    assert!(solver.enumeration.is_none());
}

#[test]
fn presence_costs_compose_with_packed_rank_one_costs() {
    use baldes::cuts::state::LimitedInput;
    let mut g = Graph::<2>::new(4, 1).unwrap();
    for v in 0..4 {
        g.set_bounds(v, 0, 0., 4.).unwrap();
    }
    for v in 0..3 {
        let id = g.add_arc(v, v + 1, 0.).unwrap();
        g.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    let events = vec![vec![0], vec![1], vec![]];
    for bidirectional in [false, true] {
        for mode in [Mode::PairCount, Mode::AnyPair] {
            for rank_cost in [-0.75, 0.75] {
                let mut engine = LabelingEngine::prepare(
                    &g,
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
                engine
                    .load_presence(&g, input(&g, &events, mode, 0.3))
                    .unwrap();
                assert!(engine.load_cuts(
                    &g,
                    vec![LimitedInput {
                        covering: false,
                        type2: false,
                        arc_memory: false,
                        denominator: 2,
                        dual: rank_cost,
                        vertex_coefficient: vec![0, 1, 1, 0],
                        arc_coefficient: Default::default(),
                        memory_vertices: (0..4).collect(),
                        memory_arcs: Default::default(),
                    }],
                    vec![]
                ));
                let expected = rank_cost + cost(&g, &events, mode, 0.3, &[0, 1, 2]);
                let result = engine.solve(&g, &Controls::default()).unwrap();
                assert!((result.minimum_reduced_cost - expected).abs() < 1e-12);
                assert!(result
                    .columns
                    .iter()
                    .all(|r| (r.cost - expected).abs() < 1e-12));
            }
        }
    }
}

fn owned_solver(g: Graph<2>, bidirectional: bool) -> baldes::pricing::Solver<2> {
    let mut solver =
        baldes::pricing::Solver::<2>::new(g.vertices.len(), g.resources.len()).unwrap();
    solver.graph = g;
    solver.phases = vec![Phase {
        check_ng: true,
        jump_arcs: true,
        labels_per_bucket: i64::MAX,
    }];
    solver
        .set_option("BALDESuseBidirectionalSearch", f64::from(bidirectional))
        .unwrap();
    solver
}

#[path = "presence/enumeration.rs"]
mod enumeration;
#[path = "presence/ownership.rs"]
mod ownership;
