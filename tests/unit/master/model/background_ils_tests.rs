use super::*;
use crate::{
    extensions::{ails_ii::run_ails_ii, common::validate_plan},
    inputs::{Customer, Rounding},
};

fn fixture() -> (
    BackgroundIlsOptions,
    Arc<Graph>,
    Vec<Vec<usize>>,
    Vec<NetworkRoute>,
) {
    let instance = Instance {
        name: "background-ails-ii".into(),
        vehicles: 2,
        capacity: 3.,
        rounding: Rounding::None,
        distances: None,
        customers: [0., 10., 1., 11.]
            .into_iter()
            .enumerate()
            .map(|(i, x)| Customer {
                x,
                y: 0.,
                demand: if i == 0 { 0. } else { 1. },
                ready: 0.,
                due: 1000.,
                service: 0.,
            })
            .collect(),
    };
    let graph = Arc::new(instance.capacity_graph(3, instance.rounding).unwrap());
    let mut options =
        BackgroundIlsOptions::new(instance.heuristic_instance(&graph, instance.rounding));
    options.search.time_limit = 0.5;
    options.search.iterations = 4;
    options.search.phase_length = 2;
    let seed = vec![vec![1, 2], vec![3]];
    let route_map = HeuristicRouteMap::new(&graph);
    let routes = seed
        .iter()
        .map(|r| NetworkRoute {
            network: 0,
            route: route_map.convert(&graph, r).unwrap(),
        })
        .collect();
    (options, graph, seed, routes)
}

fn job(graph: Arc<Graph>, routes: &[NetworkRoute]) -> Job<{ crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    Job {
        graph,
        deadline: Instant::now() + Duration::from_secs(5),
        seeds: routes
            .iter()
            .enumerate()
            .map(|(i, route)| (i as f64, Arc::new(route.clone())))
            .collect(),
    }
}

#[test]
fn background_results_match_the_shared_ails_ii_engine_and_transfer_customers() {
    let (options, graph, seed, routes) = fixture();
    let direct = run_ails_ii(&options.instance, &seed, &options.search).unwrap();
    assert!(direct.best_cost < validate_plan(&options.instance, &seed).unwrap());
    let returned = search_job(
        &options,
        job(graph.clone(), &routes),
        &HeuristicRouteMap::new(&graph),
        &AtomicBool::new(false),
        options.search.seed,
    );
    let mut expected: Vec<_> = direct
        .routes
        .into_iter()
        .filter(|r| !r.is_empty())
        .collect();
    let mut actual: Vec<Vec<usize>> = returned
        .iter()
        .map(|r| {
            r.route.nodes[1..r.route.nodes.len() - 1]
                .iter()
                .map(|&id| id as usize)
                .collect()
        })
        .collect();
    expected.sort();
    actual.sort();
    assert_eq!(actual, expected);
    // Combining visits from different seed routes proves this uses complete-plan
    // moves, rather than the removed within-route ILS implementation.
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].len(), 3);
    assert_eq!(
        validate_plan(&options.instance, &actual).unwrap(),
        direct.best_cost
    );
    for route in returned {
        assert!(graph.exact_route_feasible(&route.route).is_some());
    }
}

#[test]
fn overlapping_pricing_routes_produce_a_disjoint_complete_subproblem() {
    let (options, graph, _, _) = fixture();
    let route_map = HeuristicRouteMap::new(&graph);
    let routes: Vec<_> = [vec![1, 2], vec![2, 3], vec![3]]
        .into_iter()
        .map(|r| NetworkRoute {
            network: 0,
            route: route_map.convert(&graph, &r).unwrap(),
        })
        .collect();
    let selected = neighborhood(
        &options,
        &mut job(graph, &routes),
        &AtomicBool::new(false),
        Instant::now() + Duration::from_secs(1),
    );
    assert_eq!(selected, vec![vec![1, 2], vec![3]]);
}

#[test]
fn shared_subproblem_preserves_customer_mapping_and_explicit_distances() {
    let (mut options, _, _, _) = fixture();
    let instance = Arc::make_mut(&mut options.instance);
    instance.distances = Some(
        (0..4)
            .map(|i| (0..4).map(|j| (i * 10 + j) as f64).collect())
            .collect(),
    );
    let routes = [vec![3], vec![1]];
    let (sub, map, seed) = route_subproblem(instance, routes.iter().map(Vec::as_slice));
    assert_eq!(map, vec![0, 3, 1]);
    assert_eq!(seed, vec![vec![1], vec![2]]);
    assert_eq!(
        sub.distances.unwrap(),
        vec![vec![0., 3., 1.], vec![30., 33., 31.], vec![10., 13., 11.]]
    );
    assert_eq!(sub.customers[1].x, instance.customers[3].x);
    assert_eq!(sub.capacity, instance.capacity);
    assert_eq!(sub.rounding, instance.rounding);
}

#[test]
fn background_selection_rejects_time_window_infeasible_seed_routes() {
    let (mut options, graph, _, routes) = fixture();
    Arc::make_mut(&mut options.instance).customers[2].due = 5.;
    // Keep physical arc IDs unchanged so this specifically exercises the
    // window replay, rather than rejecting IDs from a differently pruned graph.
    let mut timed = graph.as_ref().clone();
    for vertex in 0..timed.vertices.len() {
        timed.set_bounds(vertex, 0, 0., 1000.).unwrap();
    }
    timed.set_bounds(2, 0, 0., 5.).unwrap();
    for arc in &mut timed.edit().arcs {
        arc.consumption[0] = arc.cost;
    }
    let timed = Arc::new(timed);
    assert!(graph.exact_route_feasible(&routes[0].route).is_some());
    let selected = neighborhood(
        &options,
        &mut job(timed, &routes),
        &AtomicBool::new(false),
        Instant::now() + Duration::from_secs(1),
    );
    assert_eq!(selected, vec![vec![3]]);
}

#[test]
fn cancelled_or_expired_jobs_do_not_run_the_engine() {
    let (options, graph, _, routes) = fixture();
    let map = HeuristicRouteMap::new(&graph);
    assert!(search_job(
        &options,
        job(graph.clone(), &routes),
        &map,
        &AtomicBool::new(true),
        1
    )
    .is_empty());
    let mut expired = job(graph, &routes);
    expired.deadline = Instant::now();
    assert!(search_job(&options, expired, &map, &AtomicBool::new(false), 1).is_empty());
}

#[test]
fn collection_reprices_and_rejects_routes_after_live_arc_fixing() {
    let (_, graph, _, routes) = fixture();
    let problem = Problem::set_partitioning(graph.as_ref().clone(), 2.).unwrap();
    let mut options = Options::default();
    options.gap_enumeration = false;
    let (_, mut state, _) =
        solve::prepare_with_seed_routes(problem, vec![], options, &[], &[], &[]).unwrap();
    let route = routes[0].clone();
    let mut duals = vec![0.; state.rhs.len()];
    duals[state.shared_offset] = 30.;
    let columns = collect(&mut state, vec![route.clone()], &duals).unwrap();
    assert_eq!(columns.len(), 1);
    assert_eq!(columns[0].cost, 20.);
    assert_eq!(columns[0].reduced_cost, -10.);
    let arc = state.oracles[0].index[&route.route.arcs[0]];
    let arcs = state.oracles[0].network.graph.arcs.len();
    state.oracles[0].proved_zero_arcs.resize(arcs, false);
    state.oracles[0].proved_zero_arcs[arc] = true;
    assert!(collect(&mut state, vec![route], &duals).unwrap().is_empty());
}

#[test]
fn worker_reuses_one_thread_and_recovers_from_empty_results() {
    let (config, graph, _, routes) = fixture();
    let problem = Problem::set_partitioning(graph.as_ref().clone(), 2.).unwrap();
    let mut options = Options::default();
    options.gap_enumeration = false;
    let (_, mut state, options) =
        solve::prepare_with_seed_routes(problem, vec![], options, &[], &[], &routes).unwrap();
    let mut worker = Worker::new(config).unwrap();
    let thread_id = worker.thread.as_ref().unwrap().thread().id();
    worker.submit(&state, &options.cg.known).unwrap();
    assert!(worker.busy);
    worker.submit(&state, &options.cg.known).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let returned = loop {
        let result = worker.poll().unwrap();
        if !result.is_empty() {
            break result;
        }
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    };
    assert!(!worker.busy);
    let duals = vec![0.; state.rhs.len()];
    let columns = collect(&mut state, returned, &duals).unwrap();
    worker.submit(&state, &columns).unwrap();
    while worker.busy {
        assert!(worker.poll().unwrap().is_empty());
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(worker.thread.as_ref().unwrap().thread().id(), thread_id);
}

#[test]
fn idle_worker_shutdown_wakes_and_joins_the_thread() {
    let (options, _, _, _) = fixture();
    let mut worker = Worker::<{ crate::DEFAULT_RESOURCE_DIMENSIONS }>::new(options).unwrap();
    let thread = worker.thread.take().unwrap();
    drop(worker);
    thread.join().unwrap();
}
