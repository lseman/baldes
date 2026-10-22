use super::*;
use crate::cuts::soft_conflicts::Mode;
use std::time::Duration;

fn run(g: &Graph<2>, input: &PricingInput, budget: usize) -> (Option<Vec<usize>>, usize) {
    zero_arcs(
        g,
        input,
        |_| false,
        budget,
        Instant::now() + Duration::from_secs(10),
    )
}

#[test]
fn joint_pairs_remove_an_arc_that_each_pair_alone_would_allow() {
    let mut g = Graph::<2>::new(6, 1).unwrap();
    for (a, b) in [(0, 1), (1, 2), (2, 5), (1, 3), (3, 5), (0, 4), (4, 5)] {
        g.add_arc(a, b, 0.).unwrap();
    }
    let mut input = PricingInput {
        row: PresenceRow::new(4, vec![(0, 1), (2, 3)], Mode::AnyPair).unwrap(),
        arc_events: vec![vec![], vec![0], vec![1], vec![2], vec![3], vec![], vec![]],
        dual: 0.,
    };
    assert_eq!(run(&g, &input, 10_000).0.unwrap(), vec![0, 1, 2, 3, 4]);
    for pair in [(0, 1), (2, 3)] {
        input.row = PresenceRow::new(4, vec![pair], Mode::PairCount).unwrap();
        assert!(!run(&g, &input, 10_000).0.unwrap().contains(&0));
    }
}

#[test]
fn cycles_repeated_events_and_parallel_arcs_keep_distinct_support() {
    let mut g = Graph::<2>::new(4, 1).unwrap();
    for (a, b) in [(0, 1), (1, 2), (2, 1), (1, 3), (0, 1), (1, 2), (2, 1)] {
        g.add_arc(a, b, 0.).unwrap();
    }
    for mode in [Mode::AnyPair, Mode::PairCount] {
        let mut input = PricingInput {
            row: PresenceRow::new(3, vec![(0, 1), (0, 1)], mode).unwrap(),
            arc_events: vec![
                vec![0],
                vec![0],
                vec![0],
                vec![1],
                vec![2],
                vec![2],
                vec![2],
            ],
            dual: 0.,
        };
        assert_eq!(run(&g, &input, 10_000).0.unwrap(), vec![0, 1, 2]);
        let (blocked, _) = zero_arcs(
            &g,
            &input,
            |a| a == 4,
            10_000,
            Instant::now() + Duration::from_secs(10),
        );
        assert_eq!(blocked.unwrap(), (0..7).collect::<Vec<_>>());
        input.row = PresenceRow::new(3, vec![(0, 1), (2, 2)], mode).unwrap();
        assert_eq!(
            run(&g, &input, 10_000).0.unwrap(),
            (0..7).collect::<Vec<_>>()
        );
    }
}

#[test]
fn every_small_dag_event_assignment_matches_independent_complete_paths() {
    let mut g = Graph::<2>::new(4, 1).unwrap();
    for (a, b) in [(0, 1), (1, 3), (0, 2), (2, 3), (1, 2), (0, 1)] {
        g.add_arc(a, b, 0.).unwrap();
    }
    let paths: &[&[usize]] = &[&[0, 1], &[2, 3], &[0, 4, 3], &[5, 1], &[5, 4, 3]];
    for assignment in 0..4096usize {
        let emissions: Vec<Vec<usize>> = (0..6)
            .map(|a| {
                (0..2)
                    .filter(|&event| assignment & (1 << (2 * a + event)) != 0)
                    .collect()
            })
            .collect();
        for (mode, pairs) in [
            (Mode::AnyPair, vec![(0, 1), (0, 1)]),
            (Mode::PairCount, vec![(0, 0), (0, 1)]),
        ] {
            let input = PricingInput {
                row: PresenceRow::new(2, pairs.clone(), mode).unwrap(),
                arc_events: emissions.clone(),
                dual: 0.,
            };
            for blocked in [None, Some(5)] {
                let mut viable = [false; 6];
                for path in paths {
                    if blocked.is_some_and(|a| path.contains(&a)) {
                        continue;
                    }
                    let events: HashSet<_> = path
                        .iter()
                        .flat_map(|&a| emissions[a].iter().copied())
                        .collect();
                    if !pairs
                        .iter()
                        .any(|(a, b)| events.contains(a) && events.contains(b))
                    {
                        for &a in *path {
                            viable[a] = true;
                        }
                    }
                }
                let expected: Vec<_> = viable
                    .iter()
                    .enumerate()
                    .filter_map(|(a, &used)| (!used).then_some(a))
                    .collect();
                let (actual, _) = zero_arcs(
                    &g,
                    &input,
                    |a| blocked == Some(a),
                    100_000,
                    Instant::now() + Duration::from_secs(10),
                );
                assert_eq!(
                    actual.unwrap(),
                    expected,
                    "assignment {assignment}, {mode:?}, blocked {blocked:?}"
                );
            }
        }
    }
}

#[test]
fn budget_exhaustion_never_publishes_partial_walk_support() {
    let mut g = Graph::<2>::new(3, 1).unwrap();
    g.add_arc(0, 1, 0.).unwrap();
    g.add_arc(1, 2, 0.).unwrap();
    let input = PricingInput {
        row: PresenceRow::new(1_000_001, vec![(7, 1_000_000)], Mode::AnyPair).unwrap(),
        arc_events: vec![vec![7], vec![1_000_000]],
        dual: 0.,
    };
    let (complete, used) = run(&g, &input, 10_000);
    assert_eq!(complete.unwrap(), vec![0, 1]);
    assert!(used < 100);
    for budget in 0..used {
        let (incomplete, spent) = run(&g, &input, budget);
        assert!(incomplete.is_none(), "budget {budget}");
        assert!(spent <= budget);
    }
    assert_eq!(run(&g, &input, used).0.unwrap(), vec![0, 1]);
    let expired = zero_arcs(&g, &input, |_| false, 10_000, Instant::now());
    assert_eq!(expired, (None, 0));
}
