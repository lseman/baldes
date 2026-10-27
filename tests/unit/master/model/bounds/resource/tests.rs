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

#[test]
fn dag_infeasibility_envelopes_never_remove_a_feasible_signed_resource_path() {
    use crate::graph::ResourceKind;
    for kind in [ResourceKind::Disposable, ResourceKind::Equality] {
        for a in [-2., -1., 0., 0.5, 1., 2.] {
            for b in [-2., -1., 0., 0.5, 1., 2.] {
                for cap in [0., 1., 2.] {
                    let mut graph = Graph::<2>::new(3, 2).unwrap();
                    graph.resources[1] = kind;
                    graph.initial[1] = Some(0.);
                    graph.set_bounds(0, 1, 0., 0.).unwrap();
                    for v in [1, 2] {
                        graph.set_bounds(v, 1, -1., cap).unwrap();
                    }
                    for (tail, c) in [(0, a), (1, b)] {
                        let arc = graph.add_arc(tail, tail + 1, -1.).unwrap();
                        graph.arc_mut(arc).unwrap().consumption[0] = 1.;
                        graph.arc_mut(arc).unwrap().consumption[1] = c;
                    }
                    let mut value: f64 = 0.;
                    let mut feasible = true;
                    for c in [a, b] {
                        value += c;
                        if kind == ResourceKind::Disposable {
                            value = value.max(-1.);
                        }
                        if value < -1. || value > cap {
                            feasible = false;
                            break;
                        }
                    }
                    let proved =
                        infeasible(&graph, &mut 1_000, Instant::now() + Duration::from_secs(1))
                            .unwrap();
                    assert!(
                        !proved || !feasible,
                        "kind={kind:?}, a={a}, b={b}, cap={cap}"
                    );
                    if kind == ResourceKind::Disposable {
                        assert_eq!(proved, !feasible);
                    }
                }
            }
        }
    }
}

#[test]
fn coordinate_minima_remain_a_relaxation_and_cycles_stay_unknown() {
    let mut graph = Graph::<2>::new(5, 2).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.arcs[0].consumption[0] = 1.;
    graph.add_arc(0, 2, 0.).unwrap();
    graph.arcs[1].consumption[1] = 1.;
    graph.add_arc(1, 3, 0.).unwrap();
    graph.add_arc(2, 3, 0.).unwrap();
    graph.add_arc(3, 4, 0.).unwrap();
    graph.set_bounds(4, 0, 0., 0.5).unwrap();
    graph.set_bounds(4, 1, 0., 0.5).unwrap();
    assert_eq!(
        infeasible(&graph, &mut 1_000, Instant::now() + Duration::from_secs(1)),
        Some(false)
    );
    graph.add_arc(3, 3, 0.).unwrap();
    assert_eq!(
        infeasible(&graph, &mut 1_000, Instant::now() + Duration::from_secs(1)),
        None
    );
}

#[test]
fn dag_bound_keeps_resource_states_and_path_costs_together() {
    let input = crate::master::model::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/multi_network_native_signed_dag_bound.json"
    )))
    .unwrap();
    let (problem, _) = input.compile::<2>().unwrap();
    let graph = &problem.networks[0].graph;
    let weights: Vec<_> = graph.arcs.iter().map(|a| a.cost).collect();
    let deadline = Instant::now() + Duration::from_secs(1);
    // The geometric -8 path reaches an intermediate state above its upper
    // bound. The cheapest resource-feasible path is 0->1->3->4 at -6.5.
    assert_eq!(
        dag_minimum(graph, &weights, &mut 10_000, deadline),
        Some(Some(-6.5))
    );
    assert!(dag_minimum(graph, &weights, &mut 0, deadline).is_none());
    assert!(dag_minimum(graph, &weights, &mut 10_000, Instant::now()).is_none());
    // Even after some paths reach the sink, exhausting the complete traversal
    // must never publish a minimum from a partially explored domain.
    let mut completed = false;
    for budget in 0..200 {
        if let Some(value) = dag_minimum(graph, &weights, &mut { budget }, deadline) {
            assert_eq!(value, Some(-6.5));
            completed = true;
        }
    }
    assert!(completed);
    let mut cyclic = graph.clone();
    cyclic.add_arc(1, 1, -1.).unwrap();
    assert!(dag_minimum(&cyclic, &vec![0.; cyclic.arcs.len()], &mut 10_000, deadline).is_none());
}

#[test]
fn dag_bound_relaxes_equality_waiting_instead_of_rejecting_a_lower_enclosure() {
    let mut graph = Graph::<2>::new(3, 2).unwrap();
    graph.resources[1] = crate::graph::ResourceKind::Equality;
    for v in 0..3 {
        graph.set_bounds(v, 0, 0., 3.).unwrap();
    }
    graph.set_bounds(0, 1, 0., 0.).unwrap();
    graph.set_bounds(1, 1, 0.3, 1.).unwrap();
    graph.set_bounds(2, 1, 0., 0.).unwrap();
    for (u, v, amount) in [(0, 1, 0.1 + 0.2), (1, 2, -(0.1 + 0.2))] {
        let arc = graph.add_arc(u, v, -1.).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = 1.;
        graph.arc_mut(arc).unwrap().consumption[1] = amount;
    }
    let bound = dag_minimum(
        &graph,
        &[-1., -1.],
        &mut 1000,
        Instant::now() + Duration::from_secs(1),
    )
    .unwrap()
    .unwrap();
    assert!(bound <= -2.);
}

#[test]
fn dag_bounds_keep_selection_feasibility_with_each_path_cost() {
    use crate::graph::coupled::CoupledResource;
    for scalar_resources in [0, 1] {
        let mut graph = Graph::<2>::new(4, scalar_resources).unwrap();
        for vertex in 0..4 {
            if scalar_resources > 0 {
                graph.set_bounds(vertex, 0, 0., 4.).unwrap();
            }
        }
        for (tail, head, cost) in [(0, 1, -8.), (1, 2, -7.), (2, 3, -5.), (0, 3, -10.)] {
            let id = graph.add_arc(tail, head, cost).unwrap();
            if scalar_resources > 0 {
                graph.arc_mut(id).unwrap().consumption[0] = 1.;
            }
        }
        graph
            .set_coupled_resources(&[CoupledResource::Selection {
                initial: 3,
                allowed: vec![1, 2, 3, 3],
                vertices: vec![],
                relaxed: true,
            }])
            .unwrap();
        let weights: Vec<_> = graph.arcs.iter().map(|arc| arc.cost).collect();
        let deadline = Instant::now() + Duration::from_secs(1);
        assert_eq!(
            dag_minimum(&graph, &weights, &mut 100_000, deadline),
            Some(Some(-10.))
        );
        assert!(dag_minimum(&graph, &weights, &mut 1, deadline).is_none());
        assert!(dag_minimum(&graph, &weights, &mut 100_000, Instant::now()).is_none());
        graph.arc_mut(3).unwrap().cost = -30.;
        let weights: Vec<_> = graph.arcs.iter().map(|arc| arc.cost).collect();
        assert_eq!(
            dag_minimum(&graph, &weights, &mut 100_000, deadline),
            Some(Some(-30.))
        );
    }
}

#[test]
fn coupled_dag_feasibility_unknown_keeps_the_path_and_objective_is_not_double_counted() {
    use crate::graph::coupled::CoupledResource;
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, -1.).unwrap();
    graph.add_arc(1, 2, -1.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![1e308, 1e308],
            weights: vec![0., 0.],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    assert!(graph.coupled_kernel().unwrap().certify(&[0, 1]).is_none());
    let deadline = Instant::now() + Duration::from_secs(1);
    assert_eq!(
        dag_minimum(&graph, &[-1., -1.], &mut 100_000, deadline),
        Some(Some(-2.))
    );
    graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![1., 1.],
            weights: vec![-10., -10.],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    // This routine bounds physical arc weights only. The caller separately
    // adds a resource-objective lower bound; the actual coupled cost is -30.
    assert_eq!(
        graph.coupled_kernel().unwrap().certify(&[0, 1]),
        Some((true, [-30., -30.]))
    );
    assert_eq!(
        dag_minimum(&graph, &[-1., -1.], &mut 100_000, deadline),
        Some(Some(-2.))
    );
}
