use baldes::pricing::Solver;
use std::str::FromStr;

fn take<T: FromStr>(input: &mut std::str::SplitWhitespace<'_>) -> T {
    input.next().unwrap().parse().ok().unwrap()
}

fn replay_pricing(capture: &str, full_memory: bool) {
    let mut input = capture.split_whitespace();
    let arc_count: usize = take(&mut input);
    let upper: f64 = take(&mut input);
    let rounds: usize = take(&mut input);
    let mut solver = Solver::<2>::new(11, 1).unwrap();
    solver.graph.vertices[0].elements.clear();
    solver.graph.vertices[10].elements.clear();
    for vertex in 0..11 {
        solver.graph.set_bounds(vertex, 0, 0., upper).unwrap();
    }
    if full_memory {
        for vertex in 1..10 {
            solver
                .graph
                .set_ng(vertex, &(1..10).collect::<Vec<_>>())
                .unwrap();
        }
    }
    for _ in 0..arc_count {
        let tail = take(&mut input);
        let head = take(&mut input);
        let cost = take(&mut input);
        let consumption = take(&mut input);
        let id = solver.graph.add_arc(tail, head, cost).unwrap();
        solver.graph.arc_mut(id).unwrap().consumption[0] = consumption;
    }
    for round in 0..rounds {
        let native: f64 = take(&mut input);
        for arc in &mut solver.graph.arcs {
            arc.reduced_cost = take(&mut input);
        }
        let actual = solver.price_phase(0, 150, false).unwrap();
        assert!(actual.exact, "round {round}");
        if full_memory {
            assert_eq!(
                actual.minimum_reduced_cost.to_bits(),
                native.to_bits(),
                "round {round}"
            );
        } else {
            assert!(
                (actual.minimum_reduced_cost - native).abs() <= 2e-12,
                "round {round}: native {native}, Rust {}",
                actual.minimum_reduced_cost
            );
        }
    }
    assert!(input.next().is_none());
}

#[test]
fn limited_memory_pricing_matches_captured_reference_bounds() {
    replay_pricing(
        include_str!("../fixtures/enumeration/window-pricing-limited10.txt"),
        false,
    );
}

#[test]
fn large_window_full_memory_pricing_matches_captured_reference_bounds() {
    replay_pricing(
        include_str!("../fixtures/enumeration/window-pricing-full-large.txt"),
        true,
    );
}

#[test]
fn large_window_limited_memory_pricing_matches_final_reference_bound() {
    replay_pricing(
        include_str!("../fixtures/enumeration/window-pricing-large-limited-tail.txt"),
        false,
    );
}
