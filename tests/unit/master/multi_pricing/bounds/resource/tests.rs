use super::*;
use std::time::Duration;

#[test]
fn joint_axes_bound_cycles_that_no_individual_axis_bounds() {
    // Every pair of dyadic scales and budgets has an independently known
    // integer-walk optimum: exhaust each of two independent loop budgets.
    for left in 0..=4 {
        for right in 0..=4 {
            for (left_scale, right_scale) in [(0.125, 2.), (1., 0.25), (8., 0.5)] {
                let mut graph = Graph::<7>::new(4, 7).unwrap();
                for vertex in 0..4 {
                    graph
                        .set_bounds(vertex, 3, 0., left as f64 * left_scale)
                        .unwrap();
                    graph
                        .set_bounds(vertex, 6, 0., right as f64 * right_scale)
                        .unwrap();
                }
                graph.add_arc(0, 1, 0.).unwrap();
                let a = graph.add_arc(1, 1, -2.).unwrap();
                graph.arc_mut(a).unwrap().consumption[3] = left_scale;
                graph.add_arc(1, 2, 0.).unwrap();
                let b = graph.add_arc(2, 2, -3.).unwrap();
                graph.arc_mut(b).unwrap().consumption[6] = right_scale;
                graph.add_arc(2, 3, 0.).unwrap();
                let mut axis = [false; 7];
                axis[3] = true;
                assert!(!zero_arcs_are_acyclic(&graph, &axis));
                axis = [false; 7];
                axis[6] = true;
                assert!(!zero_arcs_are_acyclic(&graph, &axis));
                axis[3] = true;
                assert!(zero_arcs_are_acyclic(&graph, &axis));
                let hops = hop_limit(&graph, 100_000).unwrap();
                let exact_hops = left + right + 3;
                assert!(hops >= exact_hops);
                let weights: Vec<_> = graph.arcs.iter().map(|a| a.cost).collect();
                let bound = minimum(
                    &graph,
                    &weights,
                    &mut 100_000,
                    Instant::now() + Duration::from_secs(1),
                )
                .unwrap()
                .unwrap();
                let exact_cost = -2. * left as f64 - 3. * right as f64;
                assert!(bound.is_finite() && bound <= exact_cost);
                assert!(hop_limit(&graph, 1).is_none());
                // Removing either resource bound leaves an adverse loop
                // with zero consumption on every usable coordinate.
                graph.arc_mut(a).unwrap().consumption[3] = 0.;
                assert!(hop_limit(&graph, 100_000).is_none());
                graph.arc_mut(a).unwrap().consumption[3] = -left_scale;
                assert!(hop_limit(&graph, 100_000).is_none());
            }
        }
    }
}

#[test]
fn bounded_walks_cover_repeated_vertices_and_budget_failures_stay_unknown() {
    let mut g = Graph::<5>::new(4, 5).unwrap();
    for v in 0..4 {
        g.set_bounds(v, 4, 0., 4.).unwrap();
    }
    for (u, v, c, q) in [
        (0, 1, 0., 1.),
        (1, 2, -1., 1.),
        (2, 1, 0., 1.),
        (1, 3, 0., 0.),
    ] {
        let id = g.add_arc(u, v, c).unwrap();
        g.arc_mut(id).unwrap().consumption[4] = q;
    }
    let deadline = Instant::now() + Duration::from_secs(1);
    let weights: Vec<_> = g.arcs.iter().map(|a| a.cost).collect();
    let mut work = 100_000;
    let value = minimum(&g, &weights, &mut work, deadline).unwrap().unwrap();
    // 0->1->2->1->3 is resource-feasible with cost -1. Excluding repeated
    // vertices would produce 0, an invalid bound on this walk domain.
    assert!(value <= -1.);
    assert!(value.is_finite());
    assert!(minimum(&g, &weights, &mut 0, deadline).is_none());
    assert!(minimum(&g, &weights, &mut 100_000, Instant::now()).is_none());
    // No resource bounds this cycle now. A finite hop cutoff is unjustified.
    g.arc_mut(0).unwrap().consumption[4] = 0.;
    g.arc_mut(1).unwrap().consumption[4] = 0.;
    g.arc_mut(2).unwrap().consumption[4] = 0.;
    assert!(minimum(&g, &weights, &mut 100_000, deadline).is_none());
    // A positive source arc alone cannot bound zero-consumption cycles.
    g.arc_mut(0).unwrap().consumption[4] = 1.;
    assert!(minimum(&g, &weights, &mut 100_000, deadline).is_none());
    g.arc_mut(1).unwrap().consumption[4] = -1.;
    assert!(minimum(&g, &weights, &mut 100_000, deadline).is_none());
}

#[test]
fn resource_hop_limits_round_outward_at_integer_consumption_boundaries() {
    for upper in [
        f64::from_bits(0.375f64.to_bits() - 1),
        0.375,
        f64::from_bits(0.375f64.to_bits() + 1),
    ] {
        let mut g = Graph::<2>::new(4, 1).unwrap();
        for v in 0..4 {
            g.set_bounds(v, 0, 0., upper).unwrap();
        }
        for (u, v, c, q) in [
            (0, 1, 0., 0.),
            (1, 2, -1., 0.125),
            (2, 1, 0., 0.),
            (1, 3, 0., 0.),
        ] {
            let id = g.add_arc(u, v, c).unwrap();
            g.arc_mut(id).unwrap().consumption[0] = q;
        }
        let weights: Vec<_> = g.arcs.iter().map(|a| a.cost).collect();
        // All these quantities are exact binary fractions. The first
        // window permits two cycles; the other two permit three.
        let cycles = if upper < 0.375 { 2. } else { 3. };
        let value = minimum(
            &g,
            &weights,
            &mut 100_000,
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap()
        .unwrap();
        assert!(value.is_finite() && value <= -cycles);
    }
}
