use baldes::extensions::stochastic::Policy;
use baldes::extensions::stochastic_restocking::{
    BellmanState, Demand, Problem, RestockingResource,
};
use baldes::graph::coupled::{CoupledPlugin, CoupledResource, PluginResource};
use baldes::labeling::engine::{EngineControls, LabelingEngine};
use baldes::pricing::phases::Phase;
use baldes::{Controls, Graph};

fn problem() -> Problem {
    Problem::new(
        4,
        vec![Demand::new(vec![(0, 0.5), (4, 0.5)]).unwrap(); 3],
        vec![
            vec![0., 3., 7., 4.],
            vec![3., 0., 4., 1.],
            vec![7., 4., 0., 3.],
            vec![4., 1., 3., 0.],
        ],
    )
    .unwrap()
}

#[test]
fn recovered_walk_bounds_and_backward_completion_match_exhaustive_routes() {
    use std::time::{Duration, Instant};
    for nonmetric in [false, true] {
        let mut costs = problem().costs().to_vec();
        if nonmetric {
            costs[1][2] = 100.;
        }
        let p = Problem::new(4, problem().demands().to_vec(), costs).unwrap();
        let mut graph = Graph::<2>::new(5, 0).unwrap();
        let mut endpoints = vec![];
        for i in 0..=3 {
            for j in 1..=4 {
                if i == j || (i == 0 && j == 4) {
                    continue;
                }
                let model_j = if j == 4 { 0 } else { j };
                let id = graph.add_arc(i, j, p.costs()[i][model_j]).unwrap();
                graph.arc_mut(id).unwrap().reduced_cost -= if model_j == 0 { 0. } else { 15. };
                endpoints.push((i, model_j));
            }
        }
        let resource = RestockingResource::new(p.clone(), endpoints).unwrap();
        let weights: Vec<_> = graph.arcs.iter().map(|a| a.reduced_cost).collect();
        let mut optimum = f64::INFINITY;
        for i in 1..=3 {
            let cost = p
                .evaluate_route(&[i], Policy::OptimalRestocking)
                .unwrap()
                .total_cost
                - 15.;
            optimum = optimum.min(cost);
            for j in 1..=3 {
                if i == j {
                    continue;
                }
                let cost = p
                    .evaluate_route(&[i, j], Policy::OptimalRestocking)
                    .unwrap()
                    .total_cost
                    - 30.;
                optimum = optimum.min(cost);
            }
        }
        let deadline = Instant::now() + Duration::from_secs(1);
        let bound = resource
            .certify_pricing_minimum(&weights, &mut 1_000_000, deadline)
            .unwrap();
        assert!(bound <= optimum + 1e-9, "{bound} > {optimum}");
        assert!(resource
            .certify_pricing_minimum(&weights, &mut 0, deadline)
            .is_none());
        assert!(resource
            .certify_pricing_minimum(&weights, &mut 1_000_000, Instant::now())
            .is_none());
        graph
            .set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(resource))])
            .unwrap();
        for threshold in [-30., -5., 0., 10.] {
            let mut engine = LabelingEngine::prepare(
                &graph,
                EngineControls {
                    backward_only: true,
                    ..Default::default()
                },
                Phase {
                    check_ng: true,
                    jump_arcs: true,
                    labels_per_bucket: i64::MAX,
                },
            )
            .unwrap();
            let answer = engine
                .solve(
                    &graph,
                    &Controls {
                        negative_only: true,
                        zero_threshold: threshold,
                        time_limit: Duration::from_secs(2),
                        max_columns: 1000,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(answer.exact);
            if optimum < threshold - 1e-6 {
                assert!(
                    (answer.minimum_reduced_cost - optimum).abs() < 1e-7,
                    "{} != {optimum}",
                    answer.minimum_reduced_cost
                );
            } else {
                assert!(answer.columns.is_empty());
                assert!(answer.minimum_reduced_cost <= threshold + 1e-7);
            }
            for column in answer.columns {
                let route: Vec<_> = column.nodes[1..column.nodes.len() - 1]
                    .iter()
                    .map(|&v| v as usize)
                    .collect();
                let exact = p
                    .evaluate_route(&route, Policy::OptimalRestocking)
                    .unwrap()
                    .total_cost;
                assert!((column.original_cost - exact).abs() < 1e-7);
            }
        }
        let mut engine = LabelingEngine::prepare(
            &graph,
            EngineControls {
                backward_only: true,
                ..Default::default()
            },
            Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            },
        )
        .unwrap();
        let batch = engine
            .solve_partial(
                &graph,
                &Controls {
                    negative_only: true,
                    max_columns: 1,
                    time_limit: Duration::from_secs(1),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(!batch.exact);
        assert_eq!(batch.columns.len(), 1);
        assert!(batch.columns[0].cost < 0.);
    }
}

#[test]
fn recovered_global_bounds_cover_fractional_zero_and_subunit_probability_mass() {
    use std::time::{Duration, Instant};
    for outcomes in [
        vec![(0, 0.5), (1, 0.5)],
        vec![(0, 1.)],
        vec![(0, 0.7), (4, 0.3)],
    ] {
        let p = Problem::new(
            4,
            vec![Demand::new(outcomes).unwrap(); 3],
            problem().costs().to_vec(),
        )
        .unwrap();
        let mut arcs = vec![];
        let mut weights = vec![];
        for i in 0..=3 {
            for j in 0..=3 {
                if i == j {
                    continue;
                }
                arcs.push((i, j));
                weights.push(p.costs()[i][j] - if j == 0 { 0. } else { 15. });
            }
        }
        let resource = RestockingResource::new(p.clone(), arcs.clone()).unwrap();
        let prepared = resource
            .prepare_pricing(&weights, 0., Instant::now() + Duration::from_secs(1))
            .unwrap();
        let mut optimum = f64::INFINITY;
        for i in 1..=3 {
            for j in 1..=3 {
                for k in 1..=3 {
                    if i == j || i == k || j == k {
                        continue;
                    }
                    for route in [&[i][..], &[i, j][..], &[i, j, k][..]] {
                        if let Ok(value) = p.evaluate_route(route, Policy::OptimalRestocking) {
                            optimum = optimum.min(value.total_cost - 15. * route.len() as f64);
                            if value.total_cost - 15. * (route.len() as f64) < -1e-6 {
                                let path: Vec<_> = std::iter::once(0)
                                    .chain(route.iter().copied())
                                    .chain(std::iter::once(0))
                                    .collect();
                                let mut state = prepared.0.root(false);
                                for pair in path.windows(2).rev() {
                                    let index = arcs
                                        .iter()
                                        .position(|&(i, j)| i == pair[0] && j == pair[1])
                                        .unwrap();
                                    state = prepared
                                        .0
                                        .extend(&state, index, false)
                                        .unwrap()
                                        .expect("completion pruned an improving route");
                                }
                            }
                        }
                    }
                }
            }
        }
        // Zero-load negative cycles may make the relaxation unbounded. Such a
        // failed proof must remain unknown, never expose a partial minimum.
        if let Some(bound) = resource.certify_pricing_minimum(
            &weights,
            &mut 1_000_000,
            Instant::now() + Duration::from_secs(1),
        ) {
            assert!(bound <= optimum + 1e-9, "{bound} > {optimum}");
        }
    }
}

#[test]
fn global_recourse_envelope_covers_unseen_routes_and_nonmetric_savings() {
    for nonmetric in [false, true] {
        let mut costs = problem().costs().to_vec();
        if nonmetric {
            costs[1][2] = 100.;
        }
        let p = Problem::new(4, problem().demands().to_vec(), costs).unwrap();
        let resource = RestockingResource::new(p.clone(), vec![]).unwrap();
        let mut work = 100_000;
        let bounds = resource.objective_bounds(4, &mut work, None).unwrap();
        if !nonmetric {
            assert!(bounds[0] > -1e-10);
        } else {
            assert!(bounds[0] < 0.);
        }
        for i in 1..=3 {
            for j in 1..=3 {
                if i == j {
                    continue;
                }
                let cost = p
                    .evaluate_route(&[i, j], Policy::OptimalRestocking)
                    .unwrap()
                    .expected_recourse_cost;
                assert!(bounds[0] <= cost && cost <= bounds[1]);
            }
        }
        assert!(work < 100_000);
        assert!(resource.objective_bounds(4, &mut 0, None).is_none());
        assert!(resource
            .objective_bounds(4, &mut work, Some(std::time::Instant::now()))
            .is_none());
    }
}

#[test]
fn restocking_global_envelope_reaches_the_kernel_walk_bound() {
    let p = problem();
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (i, j) in [(0, 1), (1, 2), (2, 3)] {
        graph.add_arc(i, j, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(
            RestockingResource::new(p, vec![(0, 1), (1, 2), (2, 0)]).unwrap(),
        ))])
        .unwrap();
    let bound = graph
        .coupled_kernel()
        .unwrap()
        .pricing_objective_walk_lower_bound(
            &graph,
            3,
            &mut 100_000,
            std::time::Instant::now() + std::time::Duration::from_secs(1),
        )
        .unwrap();
    assert!(bound > -1e-10 && bound <= 0.);
}

#[test]
fn directional_bellman_states_match_route_evaluation_and_all_join_splits() {
    let p = problem();
    for route in [vec![1], vec![1, 2], vec![2, 1], vec![1, 3], vec![3, 2]] {
        let mut path = vec![0];
        path.extend(&route);
        path.push(0);
        let arcs: Vec<_> = path.windows(2).map(|w| (w[0], w[1])).collect();
        let resource = RestockingResource::new(p.clone(), arcs.clone()).unwrap();
        let expected = p
            .evaluate_route(&route, Policy::OptimalRestocking)
            .unwrap()
            .expected_recourse_cost;
        for forward in [true, false] {
            let mut state = resource.root(forward);
            let indices: Vec<_> = if forward {
                (0..arcs.len()).collect()
            } else {
                (0..arcs.len()).rev().collect()
            };
            for arc in indices {
                state = resource.extend(&state, arc, forward).unwrap().unwrap();
            }
            assert!((resource.objective(&state, forward).unwrap() - expected).abs() < 1e-10);
            let bellman = state.get::<BellmanState>().unwrap();
            assert_eq!(bellman.customers(), route);
            assert_eq!(bellman.continuation_values().len(), 5);
        }
        for split in 0..=arcs.len() {
            let mut state = resource.root(true);
            for arc in 0..split {
                state = resource.extend(&state, arc, true).unwrap().unwrap();
            }
            assert!(resource.join(&state, &mut (split..arcs.len())).unwrap());
        }
    }
}

#[test]
fn pricing_includes_expected_recourse_and_mean_capacity() {
    let p = problem();
    let mut g = Graph::<2>::new(5, 0).unwrap();
    g.edit().elementary = true;
    let mut arcs = vec![];
    for i in 0..4 {
        for j in 1..5 {
            if i == j {
                continue;
            }
            let customer = if j == 4 { 0 } else { j };
            let cost = p.costs()[i][customer] - if customer == 0 { 0. } else { 15. };
            let arc = g.add_arc(i, j, cost).unwrap();
            g.arc_mut(arc).unwrap().cost = p.costs()[i][customer];
            arcs.push((i, customer));
        }
    }
    g.set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(
        RestockingResource::new(p.clone(), arcs).unwrap(),
    ))])
    .unwrap();
    let mut best = f64::INFINITY;
    for i in 1..=3 {
        for j in 1..=3 {
            if i == j {
                continue;
            }
            best = best.min(
                p.evaluate_route(&[i, j], Policy::OptimalRestocking)
                    .unwrap()
                    .total_cost
                    - 30.,
            );
        }
    }
    for bidirectional in [false, true] {
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
        let result = engine.solve(&g, &Controls::default()).unwrap();
        assert!(
            (result.minimum_reduced_cost - best).abs() < 1e-9,
            "{result:?}"
        );
        for col in result.columns {
            let route: Vec<usize> = col.nodes[1..col.nodes.len() - 1]
                .iter()
                .map(|&node| node as usize)
                .collect();
            let eval = p.evaluate_route(&route, Policy::OptimalRestocking).unwrap();
            assert!((col.cost - (eval.total_cost - 15. * route.len() as f64)).abs() < 1e-9);
            assert!((col.original_cost - eval.total_cost).abs() < 1e-9);
        }
    }
    let mut solver =
        baldes::pricing::Solver::<2>::new(g.vertices.len(), g.resources.len()).unwrap();
    solver.graph = g;
    for vertex in &mut solver.graph.edit().vertices[1..4] {
        vertex.memory = vec![1, 2, 3];
    }
    solver.set_forbidden_arcs(vec![0]).unwrap();
    let priced = solver.price_phase_partial(0, 20, true).unwrap();
    assert!(!priced.exact, "scalar candidates must not publish a proof");
    assert!(!priced.columns.is_empty());
    for col in priced.columns {
        assert!(!col.arcs.contains(&0));
        let route: Vec<_> = col.nodes[1..col.nodes.len() - 1]
            .iter()
            .map(|&v| v as usize)
            .collect();
        let total = p
            .evaluate_route(&route, Policy::OptimalRestocking)
            .unwrap()
            .total_cost;
        assert!((col.original_cost - total).abs() < 1e-9);
        assert!((col.cost - total + 15. * route.len() as f64).abs() < 1e-9);
        assert!(col.cost < 0.);
    }
}

#[test]
fn mean_capacity_elementarity_continuity_and_closed_routes_are_enforced() {
    let r =
        RestockingResource::new(problem(), vec![(0, 1), (1, 2), (2, 3), (2, 1), (2, 0)]).unwrap();
    let a = r.extend(&r.root(true), 0, true).unwrap().unwrap();
    assert!(r.extend(&a, 2, true).unwrap().is_none());
    let b = r.extend(&a, 1, true).unwrap().unwrap();
    assert!(r.extend(&b, 2, true).unwrap().is_none());
    assert!(r.extend(&b, 3, true).unwrap().is_none());
    let closed = r.extend(&b, 4, true).unwrap().unwrap();
    assert!(r.extend(&closed, 0, true).unwrap().is_none());
    assert!(r.validate(3).is_err());
}

#[test]
fn preventive_refill_is_chosen_before_observing_the_next_demand() {
    // At customer 1, an empty truck can refill for a detour of 6.
    // Direct service at customer 2 would fail with probability 1/2 and
    // cost 14. Customer 1 empties the truck with probability 1/2.
    let resource = RestockingResource::new(problem(), vec![(0, 1), (1, 2), (2, 0)]).unwrap();
    let mut state = resource.root(false);
    for arc in [2, 1] {
        state = resource.extend(&state, arc, false).unwrap().unwrap();
    }
    let bellman = state.get::<BellmanState>().unwrap();
    assert_eq!(bellman.continuation_values(), &[6., 6., 6., 6., 0.]);
    assert_eq!(bellman.expected_recourse_cost(), 0.);
    let closed = resource.extend(&state, 0, false).unwrap().unwrap();
    assert_eq!(
        closed
            .get::<BellmanState>()
            .unwrap()
            .expected_recourse_cost(),
        3.
    );
    assert_eq!(
        problem()
            .evaluate_route(&[1, 2], Policy::Classical)
            .unwrap()
            .expected_recourse_cost,
        3.
    );
}

#[test]
fn outward_certificate_encloses_replay_and_respects_budget_deadline_and_elementarity() {
    let r = RestockingResource::new(problem(), vec![(0, 1), (1, 2), (2, 0), (2, 1)]).unwrap();
    let mut work = 100_000;
    let (valid, bounds) = r.certify_objective(&[0, 1, 2], &mut work, None).unwrap();
    assert!(valid);
    assert!(bounds[0] <= 3. && bounds[1] >= 3.);
    assert!(work < 100_000);
    let mut empty = 0;
    assert!(r.certify_objective(&[0, 1, 2], &mut empty, None).is_none());
    assert_eq!(empty, 0);
    assert!(r
        .certify_objective(&[0, 1, 2], &mut work, Some(std::time::Instant::now()))
        .is_none());
    assert_eq!(
        r.certify_objective(&[0, 1, 3], &mut work, None),
        Some((false, [0.; 2]))
    );
    assert_eq!(
        r.certify_objective(&[1, 2], &mut work, None),
        Some((false, [0.; 2]))
    );
}

#[cfg(feature = "clp")]
#[test]
fn tree_accepts_an_outward_certified_stochastic_incumbent() {
    use baldes::master::model::{FixedRoute, NetworkRoute, Problem as Model};
    use baldes::master::tree::ModelTreeOptions;
    let demand = Demand::new(vec![(0, 0.7), (6, 0.3)]).unwrap();
    let p = Problem::new(
        8,
        vec![demand.clone(), demand],
        vec![vec![0., 3., 7.], vec![3., 0., 4.], vec![7., 4., 0.]],
    )
    .unwrap();
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    graph.add_elementarity_set(&[1]).unwrap();
    graph.add_elementarity_set(&[2]).unwrap();
    for (i, j, c) in [(0, 1, 3.), (1, 2, 4.), (2, 3, 7.)] {
        graph.add_arc(i, j, c).unwrap();
    }
    let resource = RestockingResource::new(p.clone(), vec![(0, 1), (1, 2), (2, 0)]).unwrap();
    let (_, bounds) = resource
        .certify_objective(&[0, 1, 2], &mut 100_000, None)
        .unwrap();
    assert!(bounds[0] < bounds[1]);
    graph
        .set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(resource))])
        .unwrap();
    let route = graph.elementary_route(&[0, 1, 2]).unwrap().unwrap();
    let expected = p
        .evaluate_route(&[1, 2], Policy::OptimalRestocking)
        .unwrap()
        .total_cost;
    let model = Model::set_partitioning(graph, 1.).unwrap();
    let mut options = ModelTreeOptions::default();
    options.node_limit = 0;
    options.time_limit = std::time::Duration::from_secs(1);
    options.node.cg.time_limit = options.time_limit;
    options.initial_solution = Some((
        expected,
        vec![FixedRoute {
            route: NetworkRoute { network: 0, route },
            quantity: 1.,
        }],
    ));
    let result = baldes::master::tree::solve(model, options).unwrap();
    assert!(result.incumbent >= expected);
    assert!(result.incumbent <= expected + 1e-9);
    assert_eq!(result.routes.len(), 1);
}

#[cfg(feature = "clp")]
#[test]
fn interrupted_restocking_returns_an_independent_assignment_lower_bound() {
    use baldes::extensions::stochastic::SolveOptions;
    for nonmetric in [false, true] {
        let mut costs = vec![vec![0., 3., 7.], vec![3., 0., 4.], vec![7., 4., 0.]];
        if nonmetric {
            costs[1][2] = 100.;
        }
        let p = Problem::new(
            8,
            vec![Demand::new(vec![(0, 0.7), (6, 0.3)]).unwrap(); 2],
            costs,
        )
        .unwrap();
        for fleet in [1, 2] {
            let mut options = SolveOptions::default();
            options.tree.node_limit = 0;
            options.tree.time_limit = std::time::Duration::from_secs(1);
            let result =
                baldes::extensions::stochastic_restocking::solve(&p, fleet, options).unwrap();
            assert!(result.tree.bound.is_finite() && result.tree.bound >= 0.);
            assert_eq!(result.lower_bound_source, "RestockingAssignmentRelaxation");
            for route in [&[1, 2][..], &[2, 1][..]] {
                let cost = p.evaluate_route(route, Policy::OptimalRestocking).unwrap();
                assert!(result.tree.bound <= cost.total_cost);
            }
            if fleet == 2 {
                let separate: f64 = [1, 2]
                    .iter()
                    .map(|&j| {
                        p.evaluate_route(&[j], Policy::OptimalRestocking)
                            .unwrap()
                            .total_cost
                    })
                    .sum();
                assert!(result.tree.bound <= separate);
            }
            if !nonmetric {
                // The relaxed customer-only two-cycle costs 8; depot copies
                // may remain unused, so this need not equal a routing optimum.
                assert!(result.tree.bound > 7.9);
            }
        }
    }
}

#[test]
fn continuation_dominance_preserves_every_common_prefix_and_visited_set() {
    use baldes::graph::coupled::PluginState;
    let p = Problem::new(
        4,
        vec![Demand::new(vec![(0, 0.75), (4, 0.25)]).unwrap(); 4],
        (0..5)
            .map(|i| (0..5).map(|j| if i == j { 0. } else { 1. }).collect())
            .collect(),
    )
    .unwrap();
    let arcs: Vec<_> = (0..5)
        .flat_map(|i| (0..5).filter(move |&j| i != j).map(move |j| (i, j)))
        .collect();
    let r = RestockingResource::new(p, arcs.clone()).unwrap();
    let suffix = |route: &[usize]| -> PluginState {
        let mut state = r.root(false);
        let mut next = 0;
        for &i in route.iter().rev() {
            let arc = arcs.iter().position(|&a| a == (i, next)).unwrap();
            state = r.extend(&state, arc, false).unwrap().unwrap();
            next = i;
        }
        state
    };
    let a = suffix(&[1]);
    let b = suffix(&[1, 2, 3]);
    assert!(r.supports_cost_dominance());
    assert!(r.dominates(&a, &b, false));
    assert!(!r.dominates(&b, &a, false));
    assert!(!r.dominates(&a, &b, true));
    assert!(!r.dominates(&suffix(&[1, 4]), &b, false));
    for prefix in [&[][..], &[4][..]] {
        let complete = |mut state: PluginState| {
            let mut next = 1;
            for &i in prefix.iter().rev().chain(std::iter::once(&0)) {
                let arc = arcs.iter().position(|&a| a == (i, next)).unwrap();
                state = r.extend(&state, arc, false).unwrap().unwrap();
                next = i;
            }
            r.objective(&state, false).unwrap()
        };
        assert!(complete(a.clone()) <= complete(b.clone()));
    }
}

#[test]
fn backward_kernel_compares_distinct_suffix_histories_with_compensated_dominance() {
    let p = Problem::new(
        4,
        vec![Demand::new(vec![(0, 1.)]).unwrap(); 2],
        vec![vec![0., 1., 1.], vec![1., 0., 1.], vec![1., 1., 0.]],
    )
    .unwrap();
    let resource =
        RestockingResource::new(p, vec![(1, 0), (2, 0), (1, 2), (0, 1), (0, 2)]).unwrap();
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    graph.add_arc(1, 3, 1.).unwrap();
    graph.add_arc(2, 3, 1.).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(0, 2, 1.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(resource))])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let a = kernel
        .extend(&kernel.root(false), 0, false)
        .unwrap()
        .unwrap()
        .0;
    let b = kernel
        .extend(&kernel.root(false), 1, false)
        .unwrap()
        .unwrap()
        .0;
    let b = kernel.extend(&b, 2, false).unwrap().unwrap().0;
    let penalty = kernel
        .dominance_penalty_at_vertex(&a, &b, false, 1)
        .expect("the resource proof must be reached despite distinct physical suffixes");
    assert!((0. ..1e-8).contains(&penalty));
    assert!(kernel
        .dominance_penalty_at_vertex(&b, &a, false, 1)
        .is_none());
}

#[test]
fn compensated_continuation_dominance_bounds_every_small_common_prefix() {
    fn orders(prefix: &mut Vec<usize>, routes: &mut Vec<Vec<usize>>) {
        if !prefix.is_empty() {
            routes.push(prefix.clone());
        }
        for j in 1..=4 {
            if !prefix.contains(&j) {
                prefix.push(j);
                orders(prefix, routes);
                prefix.pop();
            }
        }
    }
    let mut routes = vec![];
    orders(&mut vec![], &mut routes);
    let mut comparisons = 0;
    for nonmetric in [false, true] {
        for fractional in [false, true] {
            let demand = if fractional {
                Demand::new(vec![(0, 0.1), (1, 0.2), (2, 0.4), (3, 0.2), (4, 0.1)]).unwrap()
            } else {
                Demand::new(vec![(0, 0.75), (4, 0.25)]).unwrap()
            };
            let mut costs = vec![vec![3.; 5]; 5];
            for (i, row) in costs.iter_mut().enumerate() {
                row[i] = 0.;
            }
            if nonmetric {
                costs[1][2] = 100.;
            }
            let p = Problem::new(8, vec![demand; 4], costs).unwrap();
            let arcs: Vec<_> = (0..5)
                .flat_map(|i| (0..5).filter(move |&j| i != j).map(move |j| (i, j)))
                .collect();
            let resource = RestockingResource::new(p, arcs.clone()).unwrap();
            let states: Vec<_> = routes
                .iter()
                .map(|route| {
                    let mut state = resource.root(false);
                    let mut next = 0;
                    for &j in route.iter().rev() {
                        let arc = arcs.iter().position(|&a| a == (j, next)).unwrap();
                        state = resource.extend(&state, arc, false).unwrap().unwrap();
                        next = j;
                    }
                    state
                })
                .collect();
            for (i, a) in states.iter().enumerate() {
                for (j, b) in states.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let Some(penalty) = resource.cost_dominance_penalty(a, b, false) else {
                        continue;
                    };
                    comparisons += 1;
                    let before = resource.objective(a, false).unwrap()
                        - resource.objective(b, false).unwrap();
                    for first in
                        std::iter::once(0).chain((1..=4).filter(|v| !routes[j].contains(v)))
                    {
                        let close = |initial: &baldes::graph::coupled::PluginState| {
                            let mut state = initial.clone();
                            let mut next = routes[j][0];
                            for v in if first == 0 { vec![0] } else { vec![first, 0] } {
                                let arc = arcs.iter().position(|&a| a == (v, next)).unwrap();
                                let Some(extended) = resource.extend(&state, arc, false).unwrap()
                                else {
                                    return None;
                                };
                                state = extended;
                                next = v;
                            }
                            Some(resource.objective(&state, false).unwrap())
                        };
                        if let (Some(left), Some(right)) = (close(a), close(b)) {
                            assert!(
                                left - right - before <= penalty + 1e-12,
                                "{:?} vs {:?}: {} > {penalty}",
                                routes[i],
                                routes[j],
                                left - right - before
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(comparisons > 50);
}

#[cfg(feature = "clp")]
#[test]
fn tree_restores_original_asymmetric_customer_and_arc_order_after_reversed_pricing() {
    use baldes::extensions::stochastic::{Policy, SolveOptions};
    let p = Problem::new(
        8,
        vec![Demand::new(vec![(0, 0.5), (4, 0.5)]).unwrap(); 3],
        vec![
            vec![0., 1., 5., 3.],
            vec![2., 0., 4., 6.],
            vec![7., 1., 0., 2.],
            vec![3., 4., 8., 0.],
        ],
    )
    .unwrap();
    let optimum = [
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [3, 1, 2],
        [3, 2, 1],
    ]
    .iter()
    .map(|r| {
        p.evaluate_route(r, Policy::OptimalRestocking)
            .unwrap()
            .total_cost
    })
    .fold(f64::INFINITY, f64::min);
    let mut options = SolveOptions::default();
    options.tree.time_limit = std::time::Duration::from_secs(5);
    options.tree.node.cg.time_limit = options.tree.time_limit;
    let solved = baldes::extensions::stochastic_restocking::solve(&p, 1, options).unwrap();
    assert!((solved.tree.incumbent - optimum).abs() < 1e-6);
    assert_eq!(solved.tree.routes.len(), 1);
    let route = &solved.tree.routes[0].route.route;
    let customers: Vec<_> = route.nodes[1..route.nodes.len() - 1]
        .iter()
        .map(|&v| v as usize)
        .collect();
    assert_eq!(customers.len(), 3);
    assert!(
        (p.evaluate_route(&customers, Policy::OptimalRestocking)
            .unwrap()
            .total_cost
            - optimum)
            .abs()
            < 1e-6
    );
    let original = p.theta_cut_graph(8).unwrap();
    for (pair, &id) in route.nodes.windows(2).zip(&route.arcs) {
        let arc = original.arcs.iter().find(|a| a.id == id).unwrap();
        assert_eq!((arc.tail, arc.head), (pair[0] as usize, pair[1] as usize));
    }
}

#[cfg(feature = "clp")]
#[test]
fn bellman_load_completion_defaults_and_explicit_overrides_preserve_the_optimum() {
    use baldes::extensions::stochastic::SolveOptions;
    let p = problem();
    let mut optimum = f64::INFINITY;
    for singleton in 1..=3 {
        let pair: Vec<_> = (1..=3).filter(|&i| i != singleton).collect();
        let single_cost = p
            .evaluate_route(&[singleton], Policy::OptimalRestocking)
            .unwrap()
            .total_cost;
        for route in [pair.clone(), vec![pair[1], pair[0]]] {
            optimum = optimum.min(
                single_cost
                    + p.evaluate_route(&route, Policy::OptimalRestocking)
                        .unwrap()
                        .total_cost,
            );
        }
    }
    for override_options in [
        vec![],
        vec![("BALDESloadCompletionResource".into(), -1.)],
        vec![("BALDESloadCompletionSubsetSize".into(), 0.)],
        vec![("BALDESloadCompletionSubsetSize".into(), 4.)],
    ] {
        let mut options = SolveOptions::default();
        options.tree.time_limit = std::time::Duration::from_secs(5);
        options.tree.node.cg.time_limit = options.tree.time_limit;
        options.tree.node.pricing_options = override_options;
        let solved = baldes::extensions::stochastic_restocking::solve(&p, 2, options).unwrap();
        assert!((solved.tree.incumbent - optimum).abs() < 1e-6);
        assert!((solved.tree.bound - optimum).abs() < 1e-6);
        let mut coverage = vec![];
        let mut cost = 0.;
        for selected in &solved.tree.routes {
            let route = &selected.route.route;
            let customers: Vec<_> = route.nodes[1..route.nodes.len() - 1]
                .iter()
                .map(|&n| n as usize)
                .collect();
            coverage.extend(customers.iter().copied());
            cost += p
                .evaluate_route(&customers, Policy::OptimalRestocking)
                .unwrap()
                .total_cost;
        }
        coverage.sort_unstable();
        assert_eq!(coverage, vec![1, 2, 3]);
        assert!((cost - optimum).abs() < 1e-6);
    }
}
