use super::*;

#[test]
fn round_completion_passes_node_depth_to_bucket_refinement() {
    for depth in [0, 1] {
        let mut graph = Graph::<2>::new(3, 1).unwrap();
        for vertex in 0..3 {
            graph.set_bounds(vertex, 0, 0., 10.).unwrap();
        }
        graph.add_arc(0, 1, 1.).unwrap();
        graph.add_arc(1, 2, 1.).unwrap();
        let mut options = RootOptions::default();
        options.capture_state = true;
        options.enumerate = false;
        let root = super::super::solve(
            RoutingContext::set_partitioning(graph, 1.).unwrap(),
            options,
        )
        .unwrap();
        let snapshot = root.snapshot.unwrap();
        let mut solver = snapshot.pricing.fork_for_node();
        solver.restore_arcs();
        solver.bucket_steps[0] = 2.;
        solver.set_option("BALDESdynamicBucketSteps", 1.).unwrap();
        solver
            .set_option("BALDESdynBuckStepAdjustRatioThreshold", 0.)
            .unwrap();
        solver
            .set_option("BALDESdynBuckStepAdjustNumBuckArcsThreshold", 1_000_000.)
            .unwrap();
        solver.price_phase(0, 10, false).unwrap();
        solver.round.times.clear();
        solver.round.dominance_ratios.clear();
        solver.round.collect(1., 1, 1, true);
        let mut cg_options = cg::Options::default();
        cg_options.node_depth = depth;
        let state = Rc::new(RefCell::new(State {
            solver,
            routes: snapshot.routes.clone(),
            route_tags: Default::default(),
            incumbent: snapshot.incumbent,
            enumerate: false,
            rounded: true,
            pricing: adapter::RootPricing::new(&cg_options),
            previous_fixing_gap: f64::INFINITY,
            deadline: None,
            pricing_seconds: 0.,
            cut: snapshot.cut.clone(),
            fixing_calls: 0,
            enumerated_routes: 0,
        }));
        let mut controller = RootController {
            problem: Rc::new(snapshot.problem.clone()),
            state: state.clone(),
            checkpoint: None,
            enumeration_checkpoint: None,
            certificate_search: Default::default(),
        };
        let mut master = snapshot.master.try_clone().unwrap();
        controller.separate_round(&mut master, &[]).unwrap();
        assert_eq!(
            state.borrow().solver.bucket_steps[0],
            if depth == 0 { 1. } else { 2. }
        );
        assert!(state.borrow().solver.round.times.is_empty());
    }
}

fn column(tag: i32) -> Column {
    Column {
        tag,
        cost: 10.,
        reduced_cost: -1.,
        acceptable: true,
        identity: format!("route-{tag}"),
        entries: vec![(0, 1.), (1, 9.), (2, 1e-30)],
    }
}

#[test]
fn loaded_cut_incidence_preserves_original_base_rows_and_refreshes_unloaded_routes() {
    let mut current = column(0);
    current.entries = vec![(0, 1.), (1, 0.), (3, 4.)];
    let loaded = vec![Some(current.clone()), None];
    let mut stale_identity = column(0);
    stale_identity.identity = "different route".into();
    let mut stale_cost = column(0);
    stale_cost.cost = 11.;
    let mut pool = vec![column(0), column(1), stale_identity, stale_cost];
    let mut refreshed = vec![];
    synchronize_pool(&mut pool, &loaded, 2, &[true, false, true], |c| {
        refreshed.push(c.identity.clone());
        c.entries = vec![(7, 2.)];
        Ok(())
    })
    .unwrap();
    // Row 2 is a branch row added after cuts; keep its original tiny
    // coefficient even if the LP dropped it. Row 1's zero is omitted.
    assert_eq!(pool[0].entries, vec![(0, 1.), (2, 1e-30), (3, 4.)]);
    assert_eq!(pool[0].reduced_cost, -1.);
    assert_eq!(refreshed, vec!["route-1", "different route", "route-0"]);
    assert!(pool[1..].iter().all(|c| c.entries == vec![(7, 2.)]));
    pool[0].entries[0].1 = 8.;
    assert_eq!(loaded[0].as_ref().unwrap().entries, current.entries);
}

#[test]
fn invalid_route_tags_do_not_bypass_refresh_validation() {
    for tag in [-1, 10] {
        let mut pool = vec![column(tag)];
        let loaded = vec![Some(column(tag))];
        let result = synchronize_pool(&mut pool, &loaded, 2, &[], |_| {
            Err(invalid("cached route tag"))
        });
        assert!(result.is_err());
    }
}

#[test]
fn inactive_pool_declines_finite_domains_full_memory_markers_and_changed_geometry() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    let first = graph.add_arc(0, 1, 1.).unwrap();
    graph.arc_mut(first).unwrap().consumption[0] = 6.;
    graph.add_arc(1, 2, 1.).unwrap();
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.enumerate = false;
    let result = super::super::solve(
        RoutingContext::set_partitioning(graph, 1.).unwrap(),
        options,
    )
    .unwrap();
    let snapshot = result.snapshot.unwrap();
    let problem = Rc::new(snapshot.problem.clone());
    let mut solver = snapshot.pricing.fork_for_node();
    solver.restore_arcs();
    solver.graph = problem.model.networks[0].graph.clone();
    let mut cut = snapshot.cut.clone();
    cut.enumerated.clear();
    let state = Rc::new(RefCell::new(State {
        solver,
        routes: snapshot.routes.clone(),
        route_tags: Default::default(),
        incumbent: snapshot.incumbent,
        enumerate: false,
        rounded: true,
        pricing: adapter::RootPricing::default(),
        previous_fixing_gap: f64::INFINITY,
        deadline: None,
        pricing_seconds: 0.,
        cut,
        fixing_calls: 0,
        enumerated_routes: 0,
    }));
    let controller = RootController {
        problem: problem.clone(),
        state: state.clone(),
        checkpoint: None,
        enumeration_checkpoint: None,
        certificate_search: Default::default(),
    };
    // Base/branch identity follows the physical map, including logical rows
    // located after inherited dynamic cut rows.
    let rows = controller.curvature_rows(4);
    assert_eq!(rows[0].class, cg::CurvatureRowClass::Excluded);
    assert_eq!(rows[1].class, cg::CurvatureRowClass::Ordinary);
    assert_eq!(rows[2].class, cg::CurvatureRowClass::Dynamic);
    let saved_map = problem.row_map.borrow().clone();
    *problem.row_map.borrow_mut() = vec![0, 3];
    let remapped = controller.curvature_rows(4);
    assert_eq!(remapped[1].class, cg::CurvatureRowClass::Dynamic);
    assert_eq!(remapped[3].class, cg::CurvatureRowClass::Ordinary);
    assert!(remapped.iter().all(|r| r.active));
    *problem.row_map.borrow_mut() = saved_map;
    assert!(controller.uses_inactive_column_pool());
    // A complete empty universe excludes even the original feasible route.
    state.borrow_mut().solver.enumeration = Some(crate::enumeration::Enumeration {
        labels: crate::enumeration::Storage::new(1).unwrap(),
        paths: crate::enumeration::PathStorage::new(1).unwrap(),
        completed: true,
        generated: 0,
        pruning: Default::default(),
    });
    assert!(!controller.uses_inactive_column_pool());
    state.borrow_mut().solver.enumeration = None;
    state.borrow_mut().cut.enumerated.insert(vec![first]);
    assert!(!controller.uses_inactive_column_pool());
    state.borrow_mut().cut.enumerated.clear();
    assert!(controller.uses_inactive_column_pool());
    state
        .borrow_mut()
        .solver
        .graph
        .set_bounds(1, 0, 0., 5.)
        .unwrap();
    assert!(!controller.uses_inactive_column_pool());
    assert_eq!(
        problem.model.networks[0].graph.vertices[1].bounds[0].upper,
        10.
    );
}

#[test]
fn owned_checkpoint_restores_warm_pool_rows_and_routes_after_pricing_timeout() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for v in 0..3 {
        graph.set_bounds(v, 0, 0., 10.).unwrap();
    }
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let problem = RoutingContext::set_partitioning(graph, 1.).unwrap();
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.enumerate = false;
    let result = super::super::solve(problem, options).unwrap();
    let snapshot = result.snapshot.as_ref().unwrap();
    let state = Rc::new(RefCell::new(State {
        solver: snapshot.pricing.fork_for_node(),
        routes: snapshot.routes.clone(),
        route_tags: snapshot
            .routes
            .iter()
            .enumerate()
            .map(|(i, route)| (route.identity(), i as i32))
            .collect(),
        incumbent: snapshot.incumbent,
        enumerate: false,
        rounded: true,
        pricing: adapter::RootPricing::default(),
        previous_fixing_gap: 42.,
        deadline: None,
        pricing_seconds: 0.,
        cut: snapshot.cut.clone(),
        fixing_calls: 0,
        enumerated_routes: 7,
    }));
    let problem = Rc::new(snapshot.problem.clone());
    let mut controller = RootController {
        problem: problem.clone(),
        state: state.clone(),
        checkpoint: None,
        enumeration_checkpoint: None,
        certificate_search: Default::default(),
    };
    let expected_warm = format!("{:?}", result.master.warm_start);
    struct Legacy {
        refreshed: bool,
        saved: bool,
    }
    impl Controller<ClpMaster> for Legacy {
        fn refresh_generated_columns(&mut self, _: &mut [Column]) -> Result<()> {
            self.refreshed = true;
            Ok(())
        }
        fn save_pricing_checkpoint(&mut self, _: &ClpMaster, _: &Output) -> Result<()> {
            self.saved = true;
            Ok(())
        }
    }
    let mut legacy = Legacy {
        refreshed: false,
        saved: false,
    };
    legacy.synchronize_generated_columns(&mut [], &[]).unwrap();
    legacy
        .store_pricing_checkpoint(&snapshot.master, result.master.clone())
        .unwrap();
    assert!(legacy.refreshed && legacy.saved);
    controller
        .store_pricing_checkpoint(&snapshot.master, result.master)
        .unwrap();
    assert!(state.borrow().solver.rollback_point_saved);
    let previous_calls = controller.checkpoint.as_ref().unwrap().output.pricing_calls;
    let mut terminal = controller.checkpoint.as_ref().unwrap().output.clone();
    terminal.pricing_calls += 1;
    state.borrow_mut().incumbent = snapshot.master.objective();
    controller
        .store_pricing_checkpoint(&snapshot.master, terminal)
        .unwrap();
    assert_eq!(
        controller.checkpoint.as_ref().unwrap().output.pricing_calls,
        previous_calls
    );
    {
        let mut changed = state.borrow_mut();
        let mut abandoned = changed.routes[0].clone();
        for profile in abandoned.resource_profile.iter_mut().skip(1) {
            for value in profile {
                *value += 0.25;
            }
        }
        for value in &mut abandoned.resources {
            *value += 0.25;
        }
        register(&mut changed, &abandoned).unwrap();
        changed.cut.rows.push(Some(999));
        changed.solver.requires_rollback = true;
        changed.previous_fixing_gap = 99.;
        changed.enumerated_routes = 0;
    }
    problem.row_map.borrow_mut().reverse();
    let (master, output) = controller
        .recover_pricing_timeout(crate::incomplete("pricing time limit"))
        .unwrap();
    assert_eq!(master.costs, snapshot.master.costs);
    assert_eq!(master.rhs, snapshot.master.rhs);
    assert_eq!(master.values, snapshot.master.values);
    assert_eq!(format!("{:?}", output.warm_start), expected_warm);
    assert_eq!(
        *problem.row_map.borrow(),
        *snapshot.problem.row_map.borrow()
    );
    let restored = state.borrow();
    assert_eq!(
        format!("{:?}", restored.routes),
        format!("{:?}", snapshot.routes)
    );
    assert_eq!(restored.cut.rows, snapshot.cut.rows);
    assert_eq!(restored.previous_fixing_gap, 42.);
    assert_eq!(restored.enumerated_routes, 7);
    assert!(!restored.solver.requires_rollback);
    drop(restored);
    // The restored registry must still distinguish schedules before
    // assigning their short column identities.
    let mut restored = state.borrow_mut();
    let original = restored.routes[0].clone();
    let original_tag = register(&mut restored, &original).unwrap();
    let mut waited = original.clone();
    for profile in waited.resource_profile.iter_mut().skip(1) {
        for value in profile {
            *value += 0.25;
        }
    }
    for value in &mut waited.resources {
        *value += 0.25;
    }
    let waited_tag = register(&mut restored, &waited).unwrap();
    assert_ne!(original_tag, waited_tag);
    assert_eq!(waited_tag as usize + 1, restored.routes.len());
    assert_eq!(register(&mut restored, &waited).unwrap(), waited_tag);
    let first = restored
        .column(&problem, &original, &master.duals, original_tag)
        .unwrap();
    let second = restored
        .column(&problem, &waited, &master.duals, waited_tag)
        .unwrap();
    assert_ne!(first.identity, second.identity);
    assert_eq!(first.identity, format!("route:{original_tag}"));
    assert_eq!(first.entries, second.entries);
    let mut signed_zero = original.clone();
    signed_zero.resource_profile[0][0] = -0.;
    assert_eq!(register(&mut restored, &signed_zero).unwrap(), original_tag);
}

#[test]
fn enumeration_rollback_restores_pricing_registry_windows_and_pending_metadata() {
    let mut graph = Graph::<5>::new(3, 5).unwrap();
    for vertex in 0..3 {
        graph.set_bounds(vertex, 4, 0., 10.).unwrap();
    }
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.enumerate = false;
    let result = super::super::solve(
        RoutingContext::set_partitioning(graph, 1.).unwrap(),
        options,
    )
    .unwrap();
    let snapshot = result.snapshot.unwrap();
    let state = Rc::new(RefCell::new(State {
        solver: snapshot.pricing.fork_for_node(),
        route_tags: snapshot
            .routes
            .iter()
            .enumerate()
            .map(|(i, r)| (r.identity(), i as i32))
            .collect(),
        routes: snapshot.routes.clone(),
        incumbent: 10.,
        enumerate: true,
        rounded: true,
        pricing: adapter::RootPricing::default(),
        previous_fixing_gap: 42.,
        deadline: None,
        pricing_seconds: 0.,
        cut: snapshot.cut.clone(),
        fixing_calls: 0,
        enumerated_routes: 7,
    }));
    let problem = Rc::new(snapshot.problem.clone());
    let mut controller = RootController {
        problem: problem.clone(),
        state: state.clone(),
        checkpoint: None,
        enumeration_checkpoint: None,
        certificate_search: Default::default(),
    };
    let original_tags = state.borrow().route_tags.clone();
    let original_windows = state.borrow().solver.graph.vertices[1].bounds.clone();
    let original_forbidden = state.borrow().solver.forbidden_arcs.clone();
    let original_rows = problem.row_map.borrow().clone();
    assert!(controller.begin_enumeration_transaction(2.).unwrap());
    {
        let mut tentative = state.borrow_mut();
        let mut abandoned = tentative.routes[0].clone();
        abandoned.resource_profile[1][4] = 1.;
        register(&mut tentative, &abandoned).unwrap();
        tentative.solver.graph.set_bounds(1, 4, 5., 6.).unwrap();
        tentative.solver.forbidden_arcs[0] = true;
        tentative.solver.requires_rollback = true;
        tentative.cut.enumerated.insert(vec![999]);
        tentative.cut.removed_columns.push(999);
        tentative.previous_fixing_gap = 0.;
        tentative.enumerated_routes = 999;
    }
    problem.row_map.borrow_mut().reverse();
    controller.finish_enumeration_transaction(false);
    {
        let restored = state.borrow();
        assert_eq!(restored.route_tags, original_tags);
        assert_eq!(restored.routes.len(), snapshot.routes.len());
        assert_eq!(restored.solver.graph.vertices[1].bounds, original_windows);
        assert_eq!(restored.solver.forbidden_arcs, original_forbidden);
        assert!(!restored.solver.requires_rollback);
        assert_eq!(restored.cut.enumerated, snapshot.cut.enumerated);
        assert_eq!(restored.cut.removed_columns, snapshot.cut.removed_columns);
        assert_eq!(restored.previous_fixing_gap, 42.);
        assert_eq!(restored.enumerated_routes, 7);
        assert_eq!(*problem.row_map.borrow(), original_rows);
    }
    assert!(controller.enumeration_checkpoint.is_none());
    // Partial, valid fixing with no universe commits its scheduling state;
    // dropping it would repeat the same expensive attempt at the same gap.
    assert!(controller.begin_enumeration_transaction(2.).unwrap());
    state.borrow_mut().previous_fixing_gap = 8.;
    controller.finish_enumeration_transaction(true);
    assert_eq!(state.borrow().previous_fixing_gap, 8.);
    assert!(!controller.begin_enumeration_transaction(2.).unwrap());
    assert!(controller.enumeration_checkpoint.is_none());
}
