use super::*;

fn controller(snapshot: &NodeSnapshot<2>) -> RootController<2> {
    RootController {
        problem: Rc::new(snapshot.problem.clone()),
        state: Rc::new(RefCell::new(State {
            solver: snapshot.pricing.fork_for_node(),
            routes: snapshot.routes.clone(),
            route_tags: snapshot
                .routes
                .iter()
                .enumerate()
                .map(|(i, r)| (r.identity(), i as i32))
                .collect(),
            incumbent: snapshot.incumbent,
            enumerate: false,
            rounded: false,
            previous_fixing_gap: 1e12,
            deadline: None,
            pricing_seconds: 0.,
            cut: snapshot.cut.clone(),
            fixing_calls: 0,
            enumerated_routes: 0,
        })),
        checkpoint: None,
        enumeration_checkpoint: None,
        certificate_search: Default::default(),
    }
}

fn parent(family: usize) -> RootOutput<2> {
    let instance = crate::inputs::Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.enumerate = false;
    options.cg.stabilize = false;
    options = match family {
        1 => options.with_rank_one_cuts(),
        2 => options.with_capacity_cuts(CapacityCutConfig::new(vec![0., 1., 1., 1.], 2.)),
        3 => options.with_kpath_cuts(KPathCutConfig::new(vec![0., 1., 1., 1.], 2.)),
        4 => options.with_knapsack_cuts(KnapsackCutConfig::new(vec![
            crate::cuts::knapsack::KnapsackRow::new(0, 1, 1., 3),
        ])),
        _ => options,
    };
    super::super::super::solve(
        RootProblem::set_partitioning(
            instance.graph(3, crate::inputs::Rounding::None).unwrap(),
            3.,
        )
        .unwrap(),
        options,
    )
    .unwrap()
}

fn remove_real_columns(snapshot: &NodeSnapshot<2>) -> (ClpMaster, WarmStart) {
    let mut master = snapshot.master.try_clone().unwrap();
    let mut warm = snapshot.warm.clone();
    let removed: Vec<_> = warm
        .columns
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.is_some().then_some(i))
        .collect();
    let (_, map) = master.remove(&[], &removed).unwrap();
    warm.columns = warm
        .columns
        .into_iter()
        .zip(&map)
        .filter_map(|(c, i)| i.map(|_| c))
        .collect();
    warm.artificials = warm
        .artificials
        .into_iter()
        .filter_map(|c| map[c])
        .collect();
    warm.cut_artificials = warm
        .cut_artificials
        .into_iter()
        .filter_map(|(r, c)| map[c].map(|c| (r, c)))
        .collect();
    (master, warm)
}

#[test]
fn zero_budgets_do_not_mutate_master_routes_or_objective() {
    let parent = parent(0);
    let snapshot = parent.snapshot.as_ref().unwrap();
    let mut controller = controller(snapshot);
    for (time, iterations, status) in [
        (Duration::ZERO, 100, FeasibilityStatus::TimeLimit),
        (Duration::from_secs(1), 0, FeasibilityStatus::IterationLimit),
    ] {
        let before = controller.state.borrow().routes.clone();
        let costs = snapshot.master.costs.clone();
        let recovery = controller
            .recover_phase_one(
                &snapshot.master,
                &snapshot.warm,
                time,
                iterations,
                1e-6,
                BoundMode::Certified,
            )
            .unwrap()
            .unwrap();
        assert_eq!(recovery.report.status, status);
        assert_eq!(recovery.report.iterations, 0);
        assert!(recovery.columns.is_empty());
        assert_eq!(controller.state.borrow().routes, before);
        assert_eq!(snapshot.master.costs, costs);
    }
}

#[test]
fn zero_quantity_caps_are_preserved_and_do_not_create_a_false_certificate() {
    let parent = parent(0);
    let snapshot = parent.snapshot.as_ref().unwrap();
    let mut controller = controller(snapshot);
    let mut master = snapshot.master.try_clone().unwrap();
    for (column, entry) in snapshot.warm.columns.iter().enumerate() {
        if entry.is_some() {
            master.set_upper(column, 0.).unwrap();
        }
    }
    let upper = master.upper.clone();
    let costs = master.costs.clone();
    let recovery = controller
        .recover_phase_one(
            &master,
            &snapshot.warm,
            Duration::from_secs(5),
            10,
            1e-6,
            BoundMode::Certified,
        )
        .unwrap()
        .unwrap();
    assert_ne!(recovery.report.status, FeasibilityStatus::Feasible);
    assert_ne!(recovery.report.status, FeasibilityStatus::Infeasible);
    assert!(!recovery.report.certified_infeasible);
    assert_eq!(master.upper, upper);
    assert_eq!(master.costs, costs);
}

#[test]
fn pricing_timeout_is_a_resumable_limit_with_original_state_preserved() {
    let parent = parent(0);
    let snapshot = parent.snapshot.as_ref().unwrap();
    let mut controller = controller(snapshot);
    controller
        .state
        .borrow_mut()
        .solver
        .set_option("BALDEShardTimeThresholdInPricing", 0.)
        .unwrap();
    let (master, warm) = remove_real_columns(snapshot);
    let before_costs = master.costs.clone();
    let before_routes = controller.state.borrow().routes.clone();
    let recovery = controller
        .recover_phase_one(
            &master,
            &warm,
            Duration::from_secs(5),
            100,
            1e-6,
            BoundMode::Certified,
        )
        .unwrap()
        .unwrap();
    assert_eq!(recovery.report.status, FeasibilityStatus::TimeLimit);
    assert_eq!(recovery.report.pricing_calls, 1);
    assert!(recovery.columns.is_empty());
    assert_eq!(master.costs, before_costs);
    assert_eq!(controller.state.borrow().routes, before_routes);
    controller
        .state
        .borrow_mut()
        .solver
        .set_option("BALDEShardTimeThresholdInPricing", 1000.)
        .unwrap();
    let retried = controller
        .recover_phase_one(
            &master,
            &warm,
            Duration::from_secs(5),
            100,
            1e-6,
            BoundMode::Certified,
        )
        .unwrap()
        .unwrap();
    assert_eq!(retried.report.status, FeasibilityStatus::Feasible);
}

#[test]
fn active_cut_rows_fixed_offsets_and_recovered_coefficients_survive_private_phase_one() {
    for family in 1..=4 {
        let parent = parent(family);
        let snapshot = parent.snapshot.as_ref().unwrap();
        assert!(snapshot.active_cuts() > 0, "family {family}");
        let mut options = RootOptions::default();
        options.capture_state = true;
        options.enumerate = false;
        options.cg.stabilize = false;
        let fixed_route = parent
            .route_values()
            .into_iter()
            .max_by_key(|(r, _)| r.arcs.len())
            .unwrap()
            .0
            .arcs
            .clone();
        let fixed = super::super::super::inherit_fixed_routes(
            snapshot,
            &[FixedRoute {
                arcs: fixed_route,
                quantity: 1,
            }],
            options.clone(),
        )
        .unwrap();
        let snapshot = fixed.snapshot.as_ref().unwrap();
        let residual_arc = fixed
            .master
            .column_tags
            .iter()
            .zip(&fixed.values)
            .find_map(|(&tag, &value)| {
                (tag >= 0 && value > 1e-6).then(|| fixed.routes[tag as usize].arcs[0])
            })
            .unwrap();
        let arc = snapshot.problem.arc_position(residual_arc).unwrap();
        let branched = super::super::super::inherit_branches(
            snapshot,
            &[BranchConstraint::ArcFlow {
                arc,
                sense: RowSense::GreaterEqual,
                rhs: 1.,
            }],
            options,
        )
        .unwrap();
        assert_eq!(branched.master.status, cg::Status::Optimal);
        let snapshot = branched.snapshot.as_ref().unwrap();
        let mut controller = controller(snapshot);
        let (master, warm) = remove_real_columns(snapshot);
        let costs = master.costs.clone();
        let rhs = master.rhs.clone();
        let offset = master.objective_offset;
        assert!(offset > 0.);
        let recovery = controller
            .recover_phase_one(
                &master,
                &warm,
                Duration::from_secs(5),
                100,
                1e-6,
                BoundMode::Certified,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            recovery.report.status,
            FeasibilityStatus::Feasible,
            "family {family}: {:?}",
            recovery.report
        );
        assert!(!recovery.columns.is_empty());
        for column in recovery.columns {
            let state = controller.state.borrow();
            let route = &state.routes[column.tag as usize];
            let expected = state
                .column(&controller.problem, route, &vec![0.; rhs.len()], column.tag)
                .unwrap();
            assert_eq!(column.cost, expected.cost);
            assert_eq!(column.entries, expected.entries);
        }
        assert_eq!(master.costs, costs);
        assert_eq!(master.rhs, rhs);
        assert_eq!(master.objective_offset, offset);
        assert_eq!(snapshot.master.objective_offset, fixed.fixed_cost);
    }
}
