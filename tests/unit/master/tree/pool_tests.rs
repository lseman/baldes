use super::*;

#[test]
fn global_integer_pool_does_not_preload_child_masters() {
    let instance = crate::inputs::Instance::parse("one\nVEHICLE\nNUMBER CAPACITY\n1 1\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n").unwrap();
    let problem = RootProblem::set_partitioning(
        instance.graph(1, crate::inputs::Rounding::None).unwrap(),
        1.,
    )
    .unwrap();
    let mut options = TreeOptions::default();
    options.root.capture_state = true;
    options.root.enumerate = false;
    options.root.cg.time_limit = Duration::ZERO;
    let root = root::solve(problem.clone(), options.root.clone()).unwrap();
    let parent = root.snapshot.unwrap();
    let arcs: Vec<_> = problem.graph.arcs.iter().map(|arc| arc.id).collect();
    assert_eq!(arcs.len(), 2);
    let mut pool = Pool {
        routes: vec![],
        customers: vec![],
        seen: HashSet::new(),
    };
    pool.add(&problem, &arcs);
    assert_eq!(pool.routes.len(), 1);
    let child = evaluate(
        Some(&parent),
        None,
        &problem,
        &[],
        &pool,
        &options,
        f64::INFINITY,
        Instant::now(),
        1,
        0,
        true,
    )
    .unwrap();
    assert!(child.routes.is_empty());
    options.root.initial_routes = pool.routes.clone();
    let seeded = evaluate(
        Some(&parent),
        None,
        &problem,
        &[],
        &pool,
        &options,
        f64::INFINITY,
        Instant::now(),
        1,
        0,
        true,
    )
    .unwrap();
    assert!(seeded.routes.is_empty());
    let fresh = evaluate(
        None,
        None,
        &problem,
        &[],
        &pool,
        &options,
        f64::INFINITY,
        Instant::now(),
        1,
        0,
        true,
    )
    .unwrap();
    assert_eq!(fresh.routes.len(), 1);
    assert_eq!(fresh.routes[0].arcs, arcs);
    assert!(root.routes.is_empty());
}
