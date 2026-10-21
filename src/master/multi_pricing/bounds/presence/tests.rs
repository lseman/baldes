use super::*;
use crate::cuts::soft_conflicts::PresenceRow;
use std::time::Duration;
fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(20)
}
fn graph() -> Graph<7> {
    let mut graph = Graph::<7>::new(4, 1).unwrap();
    for (tail, head) in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    graph
}
fn matches(mask: usize, mode: Mode, duplicate: bool) -> usize {
    let both = usize::from(mask & 3 == 3);
    let count = if duplicate {
        2 * both + usize::from(mask & 1 != 0)
    } else {
        2 * both
    };
    if mode == Mode::AnyPair {
        usize::from(count > 0)
    } else {
        count
    }
}
#[test]
fn joint_bounds_match_independent_dag_paths_across_12288_signed_cases() {
    let graph = graph();
    let paths = [vec![0, 3], vec![0, 2, 4], vec![1, 4]];
    let mut cases = 0;
    for code in 0..1024usize {
        let emissions: Vec<_> = (0..5).map(|a| (code >> (2 * a)) & 3).collect();
        let weights: Vec<_> = (0..5)
            .map(|a| ((code as i32 + 3 * a) % 9 - 4) as f64 / 4.)
            .collect();
        let blocked = (code >> 4) % 32;
        for mode in [Mode::PairCount, Mode::AnyPair] {
            let other = if mode == Mode::AnyPair {
                Mode::PairCount
            } else {
                Mode::AnyPair
            };
            let rows = [
                PricingInput {
                    row: PresenceRow::new(2, vec![(0, 1), (0, 1), (0, 0)], mode).unwrap(),
                    arc_events: emissions
                        .iter()
                        .map(|&mask| (0..2).filter(|e| mask & (1 << e) != 0).collect())
                        .collect(),
                    dual: 0.,
                },
                PricingInput {
                    row: PresenceRow::new(2, vec![(0, 1), (0, 1)], other).unwrap(),
                    arc_events: emissions
                        .iter()
                        .rev()
                        .map(|&mask| (0..2).filter(|e| mask & (1 << e) != 0).collect())
                        .collect(),
                    dual: 0.,
                },
            ];
            for left in [-7, 0, 5] {
                for right in [-3, 4] {
                    let expected = paths
                        .iter()
                        .filter(|path| path.iter().all(|&a| blocked & (1 << a) == 0))
                        .map(|path| {
                            let weight: i32 = path.iter().map(|&a| (weights[a] * 4.) as i32).sum();
                            let a = path.iter().fold(0, |v, &arc| v | emissions[arc]);
                            let b = path.iter().fold(0, |v, &arc| v | emissions[4 - arc]);
                            weight
                                - left * matches(a, mode, true) as i32
                                - right * matches(b, other, false) as i32
                        })
                        .min()
                        .map(|v| v as f64 / 4.);
                    let mut budget = 100_000;
                    let actual = minimum(
                        &graph,
                        &weights,
                        &[(&rows[0], left as f64 / 4.), (&rows[1], right as f64 / 4.)],
                        |a| blocked & (1 << a) != 0,
                        &mut budget,
                        deadline(),
                    );
                    assert_eq!(
                        actual,
                        Some(expected),
                        "code={code}, mode={mode:?}, duals={left},{right}"
                    );
                    assert!(budget <= 100_000);
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 12288);
}
#[test]
fn directed_non_dyadic_costs_stay_below_exact_binary64_path_values() {
    let graph = graph();
    let weights = [0.1, -0.3, 0.7, -0.2, 0.1];
    let input = PricingInput {
        row: PresenceRow::new(2, vec![(0, 1), (0, 1)], Mode::PairCount).unwrap(),
        arc_events: vec![vec![0], vec![], vec![1], vec![1], vec![]],
        dual: 0.,
    };
    let scale = (1_u64 << 55) as f64;
    let units = |v: f64| (v * scale) as i128;
    for dual in [-0.1, 0.2, 0.3] {
        let exact = [
            units(weights[0]) + units(weights[3]) - 2 * units(dual),
            units(weights[0]) + units(weights[2]) + units(weights[4]) - 2 * units(dual),
            units(weights[1]) + units(weights[4]),
        ]
        .into_iter()
        .min()
        .unwrap();
        let result = minimum(
            &graph,
            &weights,
            &[(&input, dual)],
            |_| false,
            &mut 10_000,
            deadline(),
        )
        .unwrap()
        .unwrap();
        assert!(
            units(result) <= exact,
            "{result} exceeds exact scaled value {exact}"
        );
        assert!(
            exact - units(result) <= 32,
            "unnecessarily loose directed bound"
        );
    }
}
#[test]
fn cycles_charge_presence_once_and_use_only_proved_resource_hop_caps() {
    let mut graph = Graph::<7>::new(3, 1).unwrap();
    for (a, b) in [(0, 1), (1, 1), (1, 2)] {
        graph.add_arc(a, b, 0.).unwrap();
    }
    let input = PricingInput {
        row: PresenceRow::new(2, vec![(0, 1), (0, 1)], Mode::PairCount).unwrap(),
        arc_events: vec![vec![0], vec![1, 1], vec![1]],
        dual: 0.,
    };
    let plain = minimum(
        &graph,
        &[0., 0., 0.],
        &[(&input, 3.)],
        |_| false,
        &mut 10_000,
        deadline(),
    );
    assert_eq!(plain, Some(Some(-6.)));
    assert_eq!(
        minimum(
            &graph,
            &[0., -1., 0.],
            &[(&input, 3.)],
            |_| false,
            &mut 10_000,
            deadline()
        ),
        None
    );
    graph.arcs[1].consumption[0] = 1.;
    for vertex in &mut graph.vertices {
        vertex.bounds[0] = crate::graph::Window::new(0., 2.).unwrap();
    }
    let bounded = minimum(
        &graph,
        &[0., -1., 0.],
        &[(&input, 3.)],
        |_| false,
        &mut 10_000,
        deadline(),
    )
    .unwrap()
    .unwrap();
    // Exactly two resource-feasible loop visits give true minimum -2-6.
    assert!(bounded <= -8. && bounded.is_finite());
    // A negative cycle unable to reach the sink cannot destroy the bound.
    let mut dead = Graph::<7>::new(4, 1).unwrap();
    for (a, b) in [(0, 1), (1, 3), (0, 2), (2, 2)] {
        dead.add_arc(a, b, 0.).unwrap();
    }
    let mapping = PricingInput {
        row: input.row.clone(),
        arc_events: vec![vec![0], vec![1], vec![], vec![0, 1]],
        dual: 0.,
    };
    assert_eq!(
        minimum(
            &dead,
            &[1., 2., 0., -1.],
            &[(&mapping, 1.)],
            |_| false,
            &mut 10_000,
            deadline()
        ),
        Some(Some(1.))
    );
}
#[test]
fn sparse_ids_work_limits_and_deadlines_never_publish_partial_minima() {
    let graph = graph();
    let input = PricingInput {
        row: PresenceRow::new(usize::MAX, vec![(0, usize::MAX - 1)], Mode::AnyPair).unwrap(),
        arc_events: vec![
            vec![0],
            vec![],
            vec![usize::MAX - 1],
            vec![usize::MAX - 1],
            vec![],
        ],
        dual: 0.,
    };
    assert_eq!(
        minimum(
            &graph,
            &[0.; 5],
            &[(&input, 1.)],
            |_| false,
            &mut 10_000,
            deadline()
        ),
        Some(Some(-1.))
    );
    for initial in [0, 1, 10, 25, 100] {
        let mut budget = initial;
        let result = minimum(
            &graph,
            &[0.; 5],
            &[(&input, 1.)],
            |_| false,
            &mut budget,
            deadline(),
        );
        assert!(budget <= initial);
        assert!(result.is_none() || result == Some(Some(-1.)));
    }
    assert_eq!(
        minimum(
            &graph,
            &[0.; 5],
            &[(&input, 1.)],
            |_| false,
            &mut 10_000,
            Instant::now() - Duration::from_secs(1)
        ),
        None
    );
}
